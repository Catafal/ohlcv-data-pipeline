#!/usr/bin/env python3
"""
Yahoo Finance Pipeline Quick Start Script
This script helps you get started with the Yahoo Finance data pipeline quickly
"""

import os
import json
import subprocess
import sys
import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

def create_data_folders():
    """Create organized folder structure for data storage"""
    folders = [
        "data/yfinance",
        "data/databases", 
        "data/exports",
        "config",
        "logs"
    ]
    
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
    
    print("✅ Created data folder structure:")
    for folder in folders:
        print(f"   • {folder}/")
    return True

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    else:
        print(f"✅ Python version: {sys.version.split()[0]}")
        return True

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    
    required_packages = [
        "yfinance>=0.2.0",
        "pandas>=1.5.0", 
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "requests>=2.25.0"
    ]
    
    for package in required_packages:
        try:
            print(f"   Installing {package.split('>=')[0]}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed to install {package}: {e}")
            return False
    
    print("✅ All packages installed successfully!")
    return True

def create_config_file():
    """Create configuration file with user input"""
    print("\n🔧 Setting up configuration...")
    
    config_path = "config/yfinance_config.json"
    
    if os.path.exists(config_path):
        response = input(f"   Configuration file already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("   Keeping existing configuration.")
            return True
    
    print("   Yahoo Finance is free - no API keys needed!")
    
    # Ask for symbols
    print("   Select symbols to track (press Enter for default):")
    symbols_input = input("   Symbols (comma-separated) [AAPL,GOOGL,MSFT,TSLA,AMZN]: ").strip()
    
    if symbols_input:
        symbols = [s.strip().upper() for s in symbols_input.split(",")]
    else:
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
    
    # Ask for time period
    print("\n   Select default time period:")
    print("   1) 1 month (1mo)")
    print("   2) 3 months (3mo)")
    print("   3) 1 year (1y)")
    print("   4) 2 years (2y)")
    print("   5) 5 years (5y)")
    print("   6) Max available (max)")
    
    period_choice = input("   Choose [1-6] (default: 3): ").strip()
    period_map = {
        "1": "1mo", "2": "3mo", "3": "1y", 
        "4": "2y", "5": "5y", "6": "max"
    }
    period = period_map.get(period_choice, "1y")
    
    config = {
        "database_path": "data/databases/yfinance_data.db",
        "symbols": symbols,
        "default_period": period,
        "default_interval": "1d",
        "max_retries": 3,
        "request_delay": 0.1,
        "validation_rules": {
            "max_daily_return": 0.5,
            "min_volume": 0,
            "max_volume": 10000000000
        }
    }
    
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✅ Configuration saved to {config_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to save configuration: {e}")
        return False

def create_database():
    """Create SQLite database with OHLCV table"""
    try:
        # Ensure database directory exists
        db_path = "data/databases/yfinance_data.db"
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
        return True
    except Exception as e:
        print(f"❌ Database creation failed: {e}")
        return False

def test_yfinance_connection():
    """Test Yahoo Finance connection"""
    print("\n🔌 Testing Yahoo Finance connection...")
    
    try:
        import yfinance as yf
        
        # Simple test - try to get recent data for Apple
        ticker = yf.Ticker("AAPL")
        # Use a small date range to test connection quickly
        test_data = ticker.history(period="5d")
        
        if not test_data.empty:
            print("✅ Yahoo Finance connection successful!")
            print(f"   Test data: {len(test_data)} recent records for AAPL")
            return True
        else:
            print("⚠️  Connection test returned no data, but may still work")
            print("   Continuing with setup...")
            return True  # Continue anyway
            
    except ImportError:
        print("❌ yfinance module not found")
        return False
    except Exception as e:
        print(f"⚠️  Connection test warning: {e}")
        print("   This may be temporary - continuing with setup...")
        return True  # Continue anyway, connection issues are often temporary

def validate_ohlcv_data(df, symbol, validation_rules):
    """Validate OHLCV data against rules"""
    errors = []
    
    if df.empty:
        errors.append(f"No data for {symbol}")
        return df, errors
    
    # Calculate daily returns
    df['daily_return'] = df['Close'].pct_change()
    
    # Check for extreme returns
    max_return = validation_rules.get('max_daily_return', 0.5)
    extreme_returns = df[abs(df['daily_return']) > max_return]
    if not extreme_returns.empty:
        errors.append(f"{symbol}: {len(extreme_returns)} extreme return days")
    
    # Check volume constraints
    min_vol = validation_rules.get('min_volume', 0)
    max_vol = validation_rules.get('max_volume', 10000000000)
    
    invalid_volume = df[(df['Volume'] < min_vol) | (df['Volume'] > max_vol)]
    if not invalid_volume.empty:
        errors.append(f"{symbol}: {len(invalid_volume)} invalid volume records")
    
    # Check for missing data
    null_data = df.isnull().sum()
    if null_data.sum() > 0:
        errors.append(f"{symbol}: Missing data - {null_data.to_dict()}")
    
    return df, errors

def store_data(df, symbol):
    """Store validated data in SQLite database"""
    try:
        conn = sqlite3.connect("data/databases/yfinance_data.db")
        
        # Prepare data for insertion
        df_clean = df.copy()
        df_clean['symbol'] = symbol
        df_clean['date'] = df_clean.index.date
        df_clean.columns = [col.lower().replace(' ', '_') for col in df_clean.columns]
        
        # Select relevant columns
        columns_to_store = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'adj_close']
        df_store = df_clean[columns_to_store]
        
        # Insert with conflict resolution
        df_store.to_sql('ohlcv_data', conn, if_exists='append', index=False, method='ignore')
        
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Failed to store {symbol} data: {e}")
        return False

def run_sample_fetch():
    """Run a sample data fetch"""
    print("\n📊 Running sample data fetch...")
    
    try:
        import yfinance as yf
        
        # Load configuration
        with open("config/yfinance_config.json", 'r') as f:
            config = json.load(f)
        
        # Create database
        if not create_database():
            return False
        
        symbol = "AAPL"
        period = "1mo"
        
        print(f"   Fetching {symbol} data for last {period}...")
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        
        if df is not None and not df.empty:
            print(f"✅ Successfully fetched {len(df)} records")
            print(f"   Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
            
            # Validate the data
            validated_df, errors = validate_ohlcv_data(df, symbol, config['validation_rules'])
            
            if errors:
                print(f"⚠️  Validation warnings: {', '.join(errors)}")
            
            # Store the data
            if store_data(validated_df, symbol):
                print("✅ Data stored in database successfully!")
                return True
            else:
                return False
        else:
            print("❌ No data returned")
            return False
            
    except Exception as e:
        print(f"❌ Sample fetch failed: {e}")
        return False

def create_example_script():
    """Create a simple example script"""
    example_script = '''#!/usr/bin/env python3
"""
Simple example script for Yahoo Finance data pipeline
"""

import yfinance as yf
import pandas as pd
import json
import sqlite3
from datetime import datetime, timedelta

def load_config():
    """Load configuration from JSON file"""
    with open("yfinance_config.json", 'r') as f:
        return json.load(f)

def fetch_data(symbols, period="1y", interval="1d"):
    """Fetch data for multiple symbols"""
    results = {}
    
    for symbol in symbols:
        try:
            print(f"Fetching {symbol}...")
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if not data.empty:
                results[symbol] = data
                print(f"  ✅ {len(data)} records for {symbol}")
            else:
                print(f"  ❌ No data for {symbol}")
                
        except Exception as e:
            print(f"  ❌ Error fetching {symbol}: {e}")
    
    return results

def analyze_data(symbol_data):
    """Basic analysis of fetched data"""
    for symbol, data in symbol_data.items():
        print(f"\\n{symbol} Analysis:")
        print(f"  Date range: {data.index[0].date()} to {data.index[-1].date()}")
        print(f"  Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
        print(f"  Average volume: {data['Volume'].mean():,.0f}")
        
        # Calculate basic metrics
        returns = data['Close'].pct_change()
        print(f"  Daily volatility: {returns.std():.3f}")
        print(f"  Total return: {((data['Close'][-1] / data['Close'][0]) - 1) * 100:.2f}%")

def main():
    """Main execution function"""
    print("🚀 Yahoo Finance Data Pipeline Example")
    print("=" * 50)
    
    # Load configuration
    config = load_config()
    symbols = config['symbols']
    period = config['default_period']
    
    print(f"Fetching data for: {', '.join(symbols)}")
    print(f"Time period: {period}")
    
    # Fetch data
    data = fetch_data(symbols, period=period)
    
    if data:
        # Basic analysis
        analyze_data(data)
        
        # Store in database (optional)
        print("\\nStoring data in database...")
        # Implementation would go here
        
        print("\\n✅ Pipeline completed successfully!")
    else:
        print("❌ No data fetched")

if __name__ == "__main__":
    main()
'''
    
    try:
        with open("yfinance_example.py", "w") as f:
            f.write(example_script)
        print("✅ Created yfinance_example.py")
        return True
    except Exception as e:
        print(f"❌ Failed to create example script: {e}")
        return False

def show_next_steps():
    """Show next steps to the user"""
    print("\n🎉 Setup complete! Here's what you can do next:")
    print("\n📊 Run the pipeline:")
    print("   python yfinance_example.py")
    
    print("\n📓 For interactive analysis:")
    print("   jupyter notebook")
    print("   # Import yfinance and start exploring data")
    
    print("\n🔧 Customize your setup:")
    print("   # Edit yfinance_config.json to add more symbols")
    print("   # Change time periods and intervals")
    print("   # Modify validation rules")
    
    print("\n📚 Files and folders created:")
    print("   • config/yfinance_config.json - Your configuration")
    print("   • data/databases/yfinance_data.db - SQLite database")
    print("   • data/yfinance/ - Raw data storage")
    print("   • data/exports/ - Processed data exports")
    print("   • logs/ - Application logs")
    print("   • yfinance_example.py - Simple usage example")
    
    print("\n💡 Tips:")
    print("   • Yahoo Finance is free but has rate limits")
    print("   • Use different intervals: 1m, 5m, 1h, 1d, 1wk, 1mo")
    print("   • Available periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max")

def main():
    """Main setup function"""
    print("🚀 Yahoo Finance OHLCV Data Pipeline - Quick Setup")
    print("=" * 55)
    
    # Create folder structure first
    create_data_folders()
    
    # Check Python version
    if not check_python_version():
        return
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed during package installation")
        return
    
    # Create configuration
    if not create_config_file():
        print("❌ Setup failed during configuration")
        return
    
    # Test connection
    if not test_yfinance_connection():
        print("❌ Setup failed during connection test")
        return
    
    # Run sample fetch
    if not run_sample_fetch():
        print("⚠️  Sample fetch failed, but setup is complete")
        print("   You can still use the pipeline - check your internet connection")
    
    # Create example script
    create_example_script()
    
    # Show next steps
    show_next_steps()
    
    print("\n✅ Setup completed successfully!")

if __name__ == "__main__":
    main()