"""
Scenario Analysis Engine

Core module for conducting comprehensive scenario analysis on financial time series data.
Supports historical stress testing, Monte Carlo simulations, and risk metric calculations.

This engine analyzes how portfolios and individual securities perform under various
market conditions, including historical crises and simulated stress scenarios.

Key Features:
- Historical stress testing (COVID, 2008 crisis, tech crashes)
- Monte Carlo simulation with multiple volatility regimes  
- Advanced risk metrics (VaR, Expected Shortfall, Maximum Drawdown)
- Portfolio optimization under stress conditions
- Correlation analysis during crisis periods

Author: OHLCV Data Pipeline - Scenario Analysis Module
"""

import pandas as pd
import numpy as np
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

class ScenarioAnalysisEngine:
    """
    Main engine for scenario analysis of financial time series data.
    
    This class provides comprehensive tools for analyzing how securities and portfolios
    perform under various market conditions, both historical and simulated.
    """
    
    def __init__(self, 
                 db_path: str,
                 risk_free_rate: float = 0.02,
                 confidence_levels: List[float] = [0.01, 0.05, 0.10],
                 output_dir: str = "output"):
        """
        Initialize the scenario analysis engine.
        
        Args:
            db_path: Path to SQLite database containing OHLCV data
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculations
            confidence_levels: VaR confidence levels (e.g., [0.01, 0.05, 0.10] for 1%, 5%, 10% VaR)
            output_dir: Directory for saving analysis outputs
        """
        self.db_path = db_path
        self.risk_free_rate = risk_free_rate
        self.confidence_levels = confidence_levels
        self.output_dir = Path(output_dir)
        
        # Initialize logging
        self.logger = self._setup_logging()
        
        # Create output directories
        self._create_output_directories()
        
        # Load and cache data for performance
        self.data_cache = {}
        self.returns_cache = {}
        
        self.logger.info(f"ScenarioAnalysisEngine initialized with database: {db_path}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration for scenario analysis."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _create_output_directories(self):
        """Create organized output directory structure."""
        directories = ['reports', 'plots', 'scenarios', 'portfolio']
        
        for dir_name in directories:
            dir_path = self.output_dir / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Output directories created in: {self.output_dir}")
    
    def load_data(self, 
                  symbols: Optional[List[str]] = None,
                  start_date: Optional[str] = None,
                  end_date: Optional[str] = None,
                  force_reload: bool = False) -> pd.DataFrame:
        """
        Load OHLCV data from database with caching for performance.
        
        Args:
            symbols: List of symbols to load (None for all symbols)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format  
            force_reload: Force reload data even if cached
            
        Returns:
            DataFrame with OHLCV data
        """
        # Create cache key
        cache_key = f"{symbols}_{start_date}_{end_date}"
        
        # Return cached data if available and not forced to reload
        if cache_key in self.data_cache and not force_reload:
            self.logger.info(f"Using cached data for key: {cache_key}")
            return self.data_cache[cache_key]
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Build query based on parameters
            base_query = "SELECT * FROM ohlcv_data"
            conditions = []
            params = []
            
            if symbols:
                placeholders = ','.join(['?' for _ in symbols])
                conditions.append(f"symbol IN ({placeholders})")
                params.extend(symbols)
            
            if start_date:
                conditions.append("date >= ?")
                params.append(start_date)
                
            if end_date:
                conditions.append("date <= ?")
                params.append(end_date)
            
            if conditions:
                query = f"{base_query} WHERE {' AND '.join(conditions)}"
            else:
                query = base_query
                
            query += " ORDER BY symbol, date"
            
            # Execute query
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            if df.empty:
                self.logger.warning(f"No data found for query parameters")
                return df
            
            # Data preprocessing
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values(['symbol', 'date'])
            
            # Calculate daily returns for each symbol
            df['daily_return'] = df.groupby('symbol')['close'].pct_change()
            
            # Cache the data
            self.data_cache[cache_key] = df
            
            self.logger.info(f"Loaded {len(df)} records for {df['symbol'].nunique()} symbols")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            return pd.DataFrame()
    
    def get_returns_matrix(self, 
                          symbols: List[str],
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get returns matrix with symbols as columns and dates as index.
        Essential for portfolio analysis and correlation calculations.
        
        Args:
            symbols: List of symbols to include
            start_date: Start date for returns calculation
            end_date: End date for returns calculation
            
        Returns:
            DataFrame with returns matrix (dates x symbols)
        """
        cache_key = f"returns_{symbols}_{start_date}_{end_date}"
        
        if cache_key in self.returns_cache:
            return self.returns_cache[cache_key]
        
        # Load raw data
        df = self.load_data(symbols, start_date, end_date)
        
        if df.empty:
            return pd.DataFrame()
        
        # Pivot to create returns matrix
        returns_matrix = df.pivot(index='date', columns='symbol', values='daily_return')
        
        # Remove any rows with all NaN values
        returns_matrix = returns_matrix.dropna(how='all')
        
        # Cache the results
        self.returns_cache[cache_key] = returns_matrix
        
        self.logger.info(f"Created returns matrix: {returns_matrix.shape[0]} dates x {returns_matrix.shape[1]} symbols")
        return returns_matrix
    
    def calculate_portfolio_returns(self, 
                                  returns_matrix: pd.DataFrame,
                                  weights: Optional[Dict[str, float]] = None) -> pd.Series:
        """
        Calculate portfolio returns given individual security returns and weights.
        
        Args:
            returns_matrix: DataFrame with returns (dates x symbols)
            weights: Dictionary of symbol weights (None for equal weights)
            
        Returns:
            Series of portfolio daily returns
        """
        if returns_matrix.empty:
            return pd.Series()
        
        # Default to equal weights if not provided
        if weights is None:
            num_assets = len(returns_matrix.columns)
            weights = {symbol: 1.0/num_assets for symbol in returns_matrix.columns}
        
        # Convert weights to Series aligned with returns matrix columns
        weight_series = pd.Series(weights, index=returns_matrix.columns).fillna(0)
        
        # Normalize weights to sum to 1
        weight_series = weight_series / weight_series.sum()
        
        # Calculate portfolio returns
        portfolio_returns = (returns_matrix * weight_series).sum(axis=1)
        
        self.logger.info(f"Calculated portfolio returns: {len(portfolio_returns)} observations")
        return portfolio_returns
    
    def get_available_symbols(self) -> List[str]:
        """
        Get list of all available symbols in the database.
        
        Returns:
            List of symbol strings
        """
        try:
            conn = sqlite3.connect(self.db_path)
            query = "SELECT DISTINCT symbol FROM ohlcv_data ORDER BY symbol"
            result = pd.read_sql_query(query, conn)
            conn.close()
            
            symbols = result['symbol'].tolist()
            self.logger.info(f"Found {len(symbols)} available symbols: {symbols}")
            return symbols
            
        except Exception as e:
            self.logger.error(f"Failed to get available symbols: {e}")
            return []
    
    def get_data_date_range(self, symbol: Optional[str] = None) -> Tuple[str, str]:
        """
        Get the date range of available data for a symbol or all data.
        
        Args:
            symbol: Specific symbol (None for all symbols)
            
        Returns:
            Tuple of (start_date, end_date) as strings
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            if symbol:
                query = "SELECT MIN(date) as start_date, MAX(date) as end_date FROM ohlcv_data WHERE symbol = ?"
                result = pd.read_sql_query(query, conn, params=[symbol])
            else:
                query = "SELECT MIN(date) as start_date, MAX(date) as end_date FROM ohlcv_data"
                result = pd.read_sql_query(query, conn)
                
            conn.close()
            
            if not result.empty:
                start_date = result.iloc[0]['start_date']
                end_date = result.iloc[0]['end_date']
                self.logger.info(f"Data range for {symbol or 'all symbols'}: {start_date} to {end_date}")
                return start_date, end_date
            else:
                return "", ""
                
        except Exception as e:
            self.logger.error(f"Failed to get date range: {e}")
            return "", ""
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of available data for analysis planning.
        
        Returns:
            Dictionary with data summary statistics
        """
        try:
            symbols = self.get_available_symbols()
            start_date, end_date = self.get_data_date_range()
            
            # Get record counts per symbol
            conn = sqlite3.connect(self.db_path)
            query = """
                SELECT symbol, COUNT(*) as record_count,
                       MIN(date) as first_date, MAX(date) as last_date
                FROM ohlcv_data 
                GROUP BY symbol 
                ORDER BY symbol
            """
            symbol_stats = pd.read_sql_query(query, conn)
            conn.close()
            
            # Calculate total trading days
            if start_date and end_date:
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                total_days = (end_dt - start_dt).days
                years = total_days / 365.25
            else:
                total_days = 0
                years = 0
            
            summary = {
                'total_symbols': len(symbols),
                'symbols': symbols,
                'date_range': {'start': start_date, 'end': end_date},
                'total_days': total_days,
                'years_of_data': round(years, 1),
                'total_records': int(symbol_stats['record_count'].sum()),
                'avg_records_per_symbol': int(symbol_stats['record_count'].mean()),
                'symbol_details': symbol_stats.to_dict('records')
            }
            
            self.logger.info(f"Data summary: {summary['total_symbols']} symbols, "
                           f"{summary['years_of_data']} years, {summary['total_records']} records")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to generate data summary: {e}")
            return {}
    
    def clear_cache(self):
        """Clear all cached data to free memory."""
        self.data_cache.clear()
        self.returns_cache.clear()
        self.logger.info("Data cache cleared")