#!/usr/bin/env python3
"""
Alpha Vantage Data Pipeline Module
Handles data fetching, validation, and storage for Alpha Vantage market data
"""

import os
import json
import sqlite3
import pandas as pd
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

try:
    from alpha_vantage.timeseries import TimeSeries
except ImportError as e:
    raise ImportError("❌ alpha_vantage library not found. Run: pip install alpha_vantage") from e

class AlphaVantageDataPipeline:
    """Main pipeline class for Alpha Vantage data operations"""
    
    def __init__(self, config_path):
        """Initialize pipeline with configuration"""
        self.config = self._load_config(config_path)
        self.ts = self._init_api()
        self._setup_logging()
        self._create_database()
    
    def _load_config(self, config_path):
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise Exception(f"Failed to load config from {config_path}: {e}")
    
    def _init_api(self):
        """Initialize Alpha Vantage API connection"""
        try:
            return TimeSeries(
                key=self.config['ALPHA_VANTAGE_API_KEY'],
                output_format='pandas'
            )
        except Exception as e:
            raise Exception(f"Failed to initialize Alpha Vantage API: {e}")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/alpha_vantage_pipeline.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _create_database(self):
        """Create SQLite database and tables"""
        try:
            db_path = self.config['database_path']
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ohlcv_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date DATE NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    adj_close REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, date)
                )
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_symbol_date 
                ON ohlcv_data(symbol, date)
            ''')
            
            conn.commit()
            conn.close()
            self.logger.info("Database initialized successfully")
            
        except Exception as e:
            raise Exception(f"Database creation failed: {e}")
    
    def fetch_historical_data(self, symbol, start_date, end_date, outputsize='full'):
        """Fetch historical data for a symbol"""
        try:
            self.logger.info(f"Fetching {symbol} data from {start_date} to {end_date}")
            
            # Alpha Vantage daily data (free tier gets full historical data)
            data, meta_data = self.ts.get_daily(symbol=symbol, outputsize=outputsize)
            
            if data.empty:
                self.logger.warning(f"No data returned for {symbol}")
                return None
            
            # Reset index to get date as column
            data = data.reset_index()
            data['symbol'] = symbol
            
            # Rename columns to match our schema
            data = data.rename(columns={
                'date': 'date',
                '1. open': 'open',
                '2. high': 'high', 
                '3. low': 'low',
                '4. close': 'close',
                '5. volume': 'volume'
            })
            
            # Add adj_close (same as close for Alpha Vantage daily data)
            data['adj_close'] = data['close']
            
            # Filter by date range
            data['date'] = pd.to_datetime(data['date'])
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            data = data[(data['date'] >= start_dt) & (data['date'] <= end_dt)]
            
            # Sort by date ascending
            data = data.sort_values('date')
            
            self.logger.info(f"Fetched {len(data)} records for {symbol}")
            
            # Rate limiting - Alpha Vantage free tier: 5 calls per minute
            rate_limit_delay = self.config.get('rate_limit_delay', 12)  # 12 seconds between calls
            time.sleep(rate_limit_delay)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to fetch data for {symbol}: {e}")
            return None
    
    def validate_ohlcv_data(self, df):
        """Validate OHLCV data against configured rules"""
        errors = []
        
        if df is None or df.empty:
            errors.append("No data to validate")
            return df, errors
        
        rules = self.config.get('validation_rules', {})
        
        # Calculate daily returns
        df['daily_return'] = df['close'].pct_change()
        
        # Check for extreme returns
        max_return = rules.get('max_daily_return', 0.4)
        extreme_returns = df[abs(df['daily_return']) > max_return]
        if not extreme_returns.empty:
            errors.append(f"{len(extreme_returns)} extreme return days")
        
        # Check volume constraints
        min_vol = rules.get('min_volume', 0)
        max_vol = rules.get('max_volume', 1000000000000)
        
        invalid_volume = df[(df['volume'] < min_vol) | (df['volume'] > max_vol)]
        if not invalid_volume.empty:
            errors.append(f"{len(invalid_volume)} invalid volume records")
        
        # Check for missing data
        null_data = df.isnull().sum()
        if null_data.sum() > 0:
            errors.append(f"Missing data: {null_data.to_dict()}")
        
        self.logger.info(f"Validation complete. Errors: {len(errors)}")
        return df, errors
    
    def store_data(self, df, errors):
        """Store validated data in database"""
        if df is None or df.empty:
            self.logger.warning("No data to store")
            return False
        
        try:
            conn = sqlite3.connect(self.config['database_path'])
            
            # Select columns for storage
            columns_to_store = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'adj_close']
            df_store = df[columns_to_store].copy()
            
            # Convert date to string format
            df_store['date'] = pd.to_datetime(df_store['date']).dt.date
            
            # Insert data, handling duplicates by inserting one by one
            success_count = 0
            for _, row in df_store.iterrows():
                try:
                    row.to_frame().T.to_sql('ohlcv_data', conn, if_exists='append', index=False)
                    success_count += 1
                except sqlite3.IntegrityError:
                    # Skip duplicate records (symbol, date unique constraint)
                    continue
            
            conn.close()
            self.logger.info(f"Stored {success_count}/{len(df_store)} records (skipped duplicates)")
            
            # Log any validation errors
            if errors:
                self.logger.warning(f"Validation errors: {', '.join(errors)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store data: {e}")
            return False
    
    def get_adjusted_prices(self, symbol):
        """Retrieve adjusted prices for a symbol"""
        try:
            conn = sqlite3.connect(self.config['database_path'])
            
            query = """
                SELECT * FROM ohlcv_data 
                WHERE symbol = ? 
                ORDER BY date DESC
            """
            
            df = pd.read_sql_query(query, conn, params=[symbol])
            conn.close()
            
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                # Add raw close column for compatibility
                df['close_raw'] = df['close']
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve data for {symbol}: {e}")
            return pd.DataFrame()
    
    def run_pipeline(self, symbols, start_date, end_date):
        """Run complete pipeline for multiple symbols"""
        results = {}
        
        for symbol in symbols:
            try:
                # Fetch data
                df = self.fetch_historical_data(symbol, start_date, end_date)
                
                if df is not None:
                    # Validate data
                    validated_df, errors = self.validate_ohlcv_data(df)
                    
                    # Store data
                    success = self.store_data(validated_df, errors)
                    
                    results[symbol] = {
                        'success': success,
                        'records': len(df) if success else 0,
                        'errors': errors
                    }
                else:
                    results[symbol] = {
                        'success': False,
                        'records': 0,
                        'errors': ['No data returned']
                    }
                    
            except Exception as e:
                results[symbol] = {
                    'success': False,
                    'records': 0,
                    'errors': [str(e)]
                }
                self.logger.error(f"Pipeline failed for {symbol}: {e}")
        
        return results

if __name__ == "__main__":
    # Simple test
    try:
        pipeline = AlphaVantageDataPipeline("config/alpha_vantage_config.json")
        print("✅ AlphaVantageDataPipeline initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")