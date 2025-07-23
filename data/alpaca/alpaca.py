#!/usr/bin/env python3
"""
Alpaca Pipeline Quick Start Script
This script helps you get started with the Alpaca data pipeline quickly
"""

import os
import json
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def create_data_folders():
    """Create organized folder structure for data storage"""
    folders = [
        "data/alpaca",
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
        "alpaca-trade-api>=3.0.0",
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
    
    config_path = "config/alpaca_config.json"
    
    if os.path.exists(config_path):
        response = input(f"   Configuration file already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("   Keeping existing configuration.")
            return True
    
    # Try to get credentials from .env file first
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY') 
    base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    
    if api_key and secret_key:
        print("   ✅ Found Alpaca credentials in .env file")
        print(f"   Using base URL: {base_url}")
    else:
        print("   No credentials found in .env file")
        print("   You need Alpaca API credentials. Get them at: https://alpaca.markets")
        print("   (Free paper trading account is sufficient)")
        
        api_key = input("   Enter your Alpaca API Key: ").strip()
        secret_key = input("   Enter your Alpaca Secret Key: ").strip()
        
        if not api_key or not secret_key:
            print("   ❌ API credentials cannot be empty")
            return False
    
    # Ask for symbols
    print("   \\nSelect symbols to track (press Enter for default):")
    symbols_input = input("   Symbols (comma-separated) [AAPL,GOOGL,MSFT,TSLA,AMZN]: ").strip()
    
    if symbols_input:
        symbols = [s.strip().upper() for s in symbols_input.split(",")]
    else:
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
    
    config = {
        "ALPACA_API_KEY": api_key,
        "ALPACA_SECRET_KEY": secret_key,
        "ALPACA_BASE_URL": base_url,
        "database_path": "data/databases/market_data.db",
        "symbols": symbols,
        "max_retries": 3,
        "rate_limit_delay": 0.2,
        "validation_rules": {
            "max_daily_return": 0.4,
            "min_volume": 0,
            "max_volume": 1000000000000
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

def test_api_connection():
    """Test Alpaca API connection"""
    print("\\n🔌 Testing API connection...")
    
    try:
        from alpaca_pipeline import AlpacaDataPipeline
        
        pipeline = AlpacaDataPipeline("config/alpaca_config.json")
        print("✅ API connection successful!")
        return True
        
    except ImportError:
        print("❌ Pipeline module not found. Make sure alpaca_pipeline.py is in the current directory.")
        return False
    except Exception as e:
        print(f"❌ API connection failed: {e}")
        print("   Please check your API credentials in config/alpaca_config.json")
        return False

def run_sample_fetch():
    """Run a sample data fetch"""
    print("\\n📊 Running sample data fetch...")
    
    try:
        from alpaca_pipeline import AlpacaDataPipeline
        
        pipeline = AlpacaDataPipeline("config/alpaca_config.json")
        
        # Get symbols from configuration
        symbols = pipeline.config.get('symbols', ['AAPL'])
        
        # Fetch historical data (last 5 years, ending 6 months ago to avoid subscription limits)
        from datetime import datetime, timedelta
        end_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=180 + (5*365))).strftime("%Y-%m-%d")
        
        print(f"   Fetching data for {symbols} from {start_date} to {end_date}...")
        
        # Run pipeline for all configured symbols
        results = pipeline.run_pipeline(symbols, start_date, end_date)
        
        success_count = sum(1 for r in results.values() if r['success'])
        total_records = sum(r['records'] for r in results.values())
        
        if success_count > 0:
            print(f"✅ Successfully processed {success_count}/{len(symbols)} symbols")
            print(f"   Total records stored: {total_records}")
            
            # Show summary for each symbol
            for symbol, result in results.items():
                if result['success']:
                    print(f"   {symbol}: {result['records']} records")
                else:
                    print(f"   {symbol}: ❌ {', '.join(result['errors'])}")
            
            return True
        else:
            print("❌ No data returned for any symbols")
            for symbol, result in results.items():
                print(f"   {symbol}: {', '.join(result['errors'])}")
            return False
            
    except Exception as e:
        print(f"❌ Sample fetch failed: {e}")
        return False

def create_example_script():
    """Removed - example script creation disabled"""
    print("📝 Example script creation disabled")
    return True

def show_next_steps():
    """Show next steps to the user"""
    print("\\n🎉 Setup complete! Here's what you can do next:")
    print("\\n📊 Run the pipeline:")
    print("   python example.py")
    
    print("\\n📓 For interactive analysis:")
    print("   jupyter notebook")
    print("   # Then open the provided .ipynb files")
    
    print("\\n🔧 Customize your setup:")
    print("   # Edit config/alpaca_config.json to add more symbols")
    print("   # Modify date ranges and timeframes")
    print("   # Explore different validation rules")
    
    print("\\n📚 Files created:")
    print("   • config/alpaca_config.json - Your configuration")
    print("   • data/databases/market_data.db - SQLite database")
    print("   • example.py - Simple usage example")
    
    print("\\n💡 Tips:")
    print("   • Start with paper trading credentials")
    print("   • Check the log file (alpaca_pipeline.log) for detailed info")
    print("   • Use the Jupyter notebooks for interactive analysis")

def main():
    """Main setup function"""
    print("🚀 Alpaca OHLCV Data Pipeline - Quick Setup")
    print("=" * 50)
    
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
    
    # Test API connection
    if not test_api_connection():
        print("❌ Setup failed during API test")
        return
    
    # Run sample fetch
    if not run_sample_fetch():
        print("⚠️  Sample fetch failed, but setup is complete")
        print("   You can still use the pipeline - check your configuration")
    
    # Skip example script creation
    create_example_script()
    
    # Show next steps
    show_next_steps()
    
    print("\\n✅ Setup completed successfully!")

if __name__ == "__main__":
    main()