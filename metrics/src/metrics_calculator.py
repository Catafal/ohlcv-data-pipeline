"""
Financial Metrics Calculator

This module calculates basic financial metrics from OHLCV data:
- Daily returns
- Volatility (annualized)
- Sharpe ratio
- Basic statistics

Author: Financial Data Pipeline
"""

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple

class FinancialMetricsCalculator:
    """Calculate financial metrics from OHLCV data stored in SQLite databases."""
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize the metrics calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation (default: 2%)
        """
        self.risk_free_rate = risk_free_rate
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def load_data_from_db(self, db_path: str, symbol: Optional[str] = None) -> pd.DataFrame:
        """
        Load OHLCV data from SQLite database.
        
        Args:
            db_path: Path to SQLite database
            symbol: Specific symbol to load (if None, loads all symbols)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            conn = sqlite3.connect(db_path)
            
            if symbol:
                query = "SELECT * FROM ohlcv_data WHERE symbol = ? ORDER BY date"
                df = pd.read_sql_query(query, conn, params=(symbol,))
            else:
                query = "SELECT * FROM ohlcv_data ORDER BY symbol, date"
                df = pd.read_sql_query(query, conn)
            
            conn.close()
            
            if df.empty:
                self.logger.warning(f"No data found in {db_path}")
                return df
                
            # Convert date column to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            self.logger.info(f"Loaded {len(df)} records from {db_path}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data from {db_path}: {e}")
            return pd.DataFrame()
    
    def calculate_returns(self, df: pd.DataFrame, price_column: str = 'close') -> pd.DataFrame:
        """
        Calculate daily returns for each symbol.
        
        Args:
            df: DataFrame with OHLCV data
            price_column: Column to use for return calculation
            
        Returns:
            DataFrame with returns added
        """
        result_df = df.copy()
        
        if 'symbol' in df.columns:
            # Calculate returns for each symbol separately
            result_df['daily_return'] = df.groupby('symbol')[price_column].pct_change()
        else:
            # Single symbol data
            result_df['daily_return'] = df[price_column].pct_change()
        
        return result_df
    
    def calculate_volatility(self, returns: pd.Series, annualize: bool = True) -> float:
        """
        Calculate volatility from returns.
        
        Args:
            returns: Series of daily returns
            annualize: Whether to annualize the volatility (multiply by sqrt(252))
            
        Returns:
            Volatility value
        """
        volatility = returns.std()
        
        if annualize:
            # Annualize using 252 trading days
            volatility = volatility * np.sqrt(252)
            
        return volatility
    
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: Optional[float] = None) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: Series of daily returns
            risk_free_rate: Annual risk-free rate (uses instance default if None)
            
        Returns:
            Sharpe ratio
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
            
        # Convert annual risk-free rate to daily
        daily_risk_free = risk_free_rate / 252
        
        # Calculate excess returns
        excess_returns = returns - daily_risk_free
        
        # Calculate Sharpe ratio
        if excess_returns.std() == 0:
            return 0.0
            
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        return sharpe_ratio
    
    def calculate_metrics_for_symbol(self, df: pd.DataFrame, symbol: str) -> Dict:
        """
        Calculate all metrics for a specific symbol.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Symbol to calculate metrics for
            
        Returns:
            Dictionary with calculated metrics
        """
        # Filter data for the symbol
        symbol_data = df[df['symbol'] == symbol].copy()
        
        if symbol_data.empty:
            self.logger.warning(f"No data found for symbol {symbol}")
            return {}
        
        # Calculate returns
        symbol_data = self.calculate_returns(symbol_data)
        returns = symbol_data['daily_return'].dropna()
        
        if len(returns) < 2:
            self.logger.warning(f"Insufficient data for {symbol}")
            return {}
        
        # Calculate metrics
        metrics = {
            'symbol': symbol,
            'data_points': len(symbol_data),
            'date_range': f"{symbol_data['date'].min().date()} to {symbol_data['date'].max().date()}",
            'total_return': ((symbol_data['close'].iloc[-1] / symbol_data['close'].iloc[0]) - 1) * 100,
            'annualized_return': (returns.mean() * 252) * 100,
            'volatility': self.calculate_volatility(returns) * 100,
            'sharpe_ratio': self.calculate_sharpe_ratio(returns),
            'max_daily_return': returns.max() * 100,
            'min_daily_return': returns.min() * 100,
            'avg_daily_return': returns.mean() * 100,
            'avg_volume': symbol_data['volume'].mean(),
            'price_range': f"${symbol_data['low'].min():.2f} - ${symbol_data['high'].max():.2f}"
        }
        
        return metrics
    
    def calculate_all_metrics(self, db_paths: List[str]) -> pd.DataFrame:
        """
        Calculate metrics for all symbols across multiple databases.
        
        Args:
            db_paths: List of database paths to analyze
            
        Returns:
            DataFrame with metrics for all symbols
        """
        all_metrics = []
        
        for db_path in db_paths:
            self.logger.info(f"Processing database: {db_path}")
            
            # Load data from database
            df = self.load_data_from_db(db_path)
            
            if df.empty:
                continue
                
            # Get unique symbols
            symbols = df['symbol'].unique()
            
            # Calculate metrics for each symbol
            for symbol in symbols:
                metrics = self.calculate_metrics_for_symbol(df, symbol)
                if metrics:
                    metrics['data_source'] = db_path.split('/')[-1].replace('.db', '')
                    all_metrics.append(metrics)
        
        # Convert to DataFrame
        if all_metrics:
            metrics_df = pd.DataFrame(all_metrics)
            return metrics_df
        else:
            self.logger.warning("No metrics calculated")
            return pd.DataFrame()
    
    def display_metrics(self, metrics_df: pd.DataFrame) -> None:
        """
        Display calculated metrics in a formatted table.
        
        Args:
            metrics_df: DataFrame with calculated metrics
        """
        if metrics_df.empty:
            print("No metrics to display")
            return
            
        print("\n" + "="*80)
        print("FINANCIAL METRICS SUMMARY")
        print("="*80)
        
        for _, row in metrics_df.iterrows():
            print(f"\n📊 {row['symbol']} ({row['data_source']})")
            print(f"   Data Range: {row['date_range']} ({row['data_points']} days)")
            print(f"   Total Return: {row['total_return']:.2f}%")
            print(f"   Annualized Return: {row['annualized_return']:.2f}%")
            print(f"   Volatility: {row['volatility']:.2f}%")
            print(f"   Sharpe Ratio: {row['sharpe_ratio']:.3f}")
            print(f"   Daily Return Range: {row['min_daily_return']:.2f}% to {row['max_daily_return']:.2f}%")
            print(f"   Price Range: {row['price_range']}")
            print(f"   Avg Daily Volume: {row['avg_volume']:,.0f}")
    
    def save_metrics_to_csv(self, metrics_df: pd.DataFrame, output_path: str = "financial_metrics.csv") -> None:
        """
        Save calculated metrics to CSV file.
        
        Args:
            metrics_df: DataFrame with calculated metrics
            output_path: Path to save CSV file
        """
        try:
            metrics_df.to_csv(output_path, index=False)
            self.logger.info(f"Metrics saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving metrics to {output_path}: {e}")