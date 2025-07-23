#!/usr/bin/env python3
"""
Alpha Vantage Pipeline Quick Start Script
This script helps you get started with the Alpha Vantage data pipeline quickly
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
        "data/alpha_vantage",
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
        "alpha_vantage>=2.3.1",
        "pandas>=1.5.0", 
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "requests>=2.25.0",
        "python-dotenv"
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
    
    config_path = "config/alpha_vantage_config.json"
    
    if os.path.exists(config_path):
        response = input(f"   Configuration file already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("   Keeping existing configuration.")
            return True
    
    # Try to get API key from .env file first
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    
    if api_key:
        print("   ✅ Found Alpha Vantage API key in .env file")
    else:
        print("   No API key found in .env file")
        print("   You need Alpha Vantage API key. Get it free at: https://www.alphavantage.co/support/#api-key")
        print("   (Free tier: 500 requests/day, 5 requests/minute)")
        
        api_key = input("   Enter your Alpha Vantage API Key: ").strip()
        
        if not api_key:
            print("   ❌ API key cannot be empty")
            return False
    
    # Ask for symbols
    print("   \nSelect symbols to track (press Enter for default):")
    symbols_input = input("   Symbols (comma-separated) [AAPL,GOOGL,MSFT,TSLA,AMZN]: ").strip()
    
    if symbols_input:
        symbols = [s.strip().upper() for s in symbols_input.split(",")]
    else:
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
    
    config = {
        "ALPHA_VANTAGE_API_KEY": api_key,
        "database_path": "data/databases/alpha_vantage_data.db",
        "symbols": symbols,
        "max_retries": 3,
        "rate_limit_delay": 12,  # 12 seconds between calls for free tier (5 calls/minute)
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
    """Test Alpha Vantage API connection"""
    print("\n🔌 Testing API connection...")
    
    try:
        from alpha_vantage_pipeline import AlphaVantageDataPipeline
        
        pipeline = AlphaVantageDataPipeline("config/alpha_vantage_config.json")
        print("✅ API connection successful!")
        return True
        
    except ImportError:
        print("❌ Pipeline module not found. Make sure alpha_vantage_pipeline.py is in the current directory.")
        return False
    except Exception as e:
        print(f"❌ API connection failed: {e}")
        print("   Please check your API key in config/alpha_vantage_config.json")
        return False

def run_sample_fetch():
    """Run a sample data fetch"""
    print("\n📊 Running sample data fetch...")
    
    try:
        from alpha_vantage_pipeline import AlphaVantageDataPipeline
        
        pipeline = AlphaVantageDataPipeline("config/alpha_vantage_config.json")
        
        # Get symbols from configuration
        symbols = pipeline.config.get('symbols', ['AAPL'])
        
        # Fetch historical data (last 2 years for free tier efficiency)
        from datetime import datetime, timedelta
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=2*365)).strftime("%Y-%m-%d")
        
        print(f"   Fetching data for {symbols[:2]} from {start_date} to {end_date}...")  # Limit to 2 symbols for demo
        print(f"   Note: Free tier has 5 requests/minute limit - this may take a few minutes")
        
        # Run pipeline for first 2 symbols to avoid rate limiting during demo
        demo_symbols = symbols[:2]
        results = pipeline.run_pipeline(demo_symbols, start_date, end_date)
        
        success_count = sum(1 for r in results.values() if r['success'])
        total_records = sum(r['records'] for r in results.values())
        
        if success_count > 0:
            print(f"✅ Successfully processed {success_count}/{len(demo_symbols)} symbols")
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
    print("\n🎉 Setup complete! Here's what you can do next:")
    print("\n📊 Run the pipeline:")
    print("   python alpha_vantage.py")
    
    print("\n📓 For interactive analysis:")
    print("   jupyter notebook")
    print("   # Then create notebooks to analyze your Alpha Vantage data")
    
    print("\n🔧 Customize your setup:")
    print("   # Edit config/alpha_vantage_config.json to add more symbols")
    print("   # Note: Free tier allows 500 requests/day, 5 requests/minute")
    print("   # Consider upgrading for higher limits if needed")
    
    print("\n📚 Files created:")
    print("   • config/alpha_vantage_config.json - Your configuration")
    print("   • data/databases/alpha_vantage_data.db - SQLite database")
    
    print("\n💡 Tips:")
    print("   • Free API key gives 500 requests/day")
    print("   • Rate limited to 5 requests/minute")
    print("   • Check the log file (logs/alpha_vantage_pipeline.log) for detailed info")
    print("   • Use smaller symbol lists or date ranges to manage rate limits")

def main():
    """Main setup function"""
    print("🚀 Alpha Vantage OHLCV Data Pipeline - Quick Setup")
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
    
    print("\n✅ Alpha Vantage setup completed successfully!")

if __name__ == "__main__":
    main()