#!/usr/bin/env python3
"""
Alpha Vantage Data Collection Runner
Collects 10 years of historical data for Magnificent 7 companies for scenario analysis.

Usage:
    python run_alpha_vantage_data.py

This script fetches historical data from 2015-01-01 to present day for:
- AAPL (Apple)
- MSFT (Microsoft) 
- GOOGL (Google/Alphabet)
- AMZN (Amazon)
- TSLA (Tesla)
- META (Meta/Facebook)
- NVDA (Nvidia)

Author: OHLCV Data Pipeline - Scenario Analysis Module
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from alpha_vantage_pipeline import AlphaVantageDataPipeline

def main():
    """Main execution function for data collection."""
    print("🚀 Alpha Vantage Data Collection for Scenario Analysis")
    print("=" * 60)
    
    # Configuration
    config_path = "config/alpha_vantage_config.json"
    
    # Date range: 10 years of data for comprehensive scenario analysis
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=10*365)).strftime('%Y-%m-%d')  # ~10 years ago
    
    print(f"📅 Collecting data from {start_date} to {end_date}")
    print(f"🎯 Target: Magnificent 7 companies for scenario analysis")
    
    try:
        # Initialize pipeline
        print(f"🔧 Initializing Alpha Vantage pipeline...")
        pipeline = AlphaVantageDataPipeline(config_path)
        
        # Get symbols from config
        symbols = pipeline.config['symbols']
        print(f"📊 Symbols to collect: {', '.join(symbols)} ({len(symbols)} total)")
        
        # Estimate time (12 seconds per symbol due to rate limiting)
        estimated_time = len(symbols) * 12 / 60  # minutes
        print(f"⏱️  Estimated collection time: ~{estimated_time:.1f} minutes")
        print(f"⚠️  Alpha Vantage rate limit: 5 calls/minute (12s delays between symbols)")
        
        # Confirm before starting
        user_input = input(f"\n🤔 Proceed with data collection? (y/N): ").strip().lower()
        if user_input != 'y':
            print("❌ Data collection cancelled by user")
            return
        
        print(f"\\n🏃 Starting data collection...")
        print("-" * 60)
        
        # Run pipeline for all symbols
        results = pipeline.run_pipeline(symbols, start_date, end_date)
        
        # Display results
        print("\\n" + "=" * 60)
        print("📋 DATA COLLECTION RESULTS")
        print("=" * 60)
        
        total_records = 0
        successful_symbols = 0
        
        for symbol, result in results.items():
            status = "✅" if result['success'] else "❌"
            records = result['records']
            errors = result.get('errors', [])
            
            print(f"{status} {symbol}: {records:,} records")
            
            if result['success']:
                successful_symbols += 1
                total_records += records
            
            # Display any errors
            if errors:
                for error in errors[:3]:  # Show first 3 errors
                    print(f"    ⚠️  {error}")
                if len(errors) > 3:
                    print(f"    ... and {len(errors) - 3} more errors")
        
        # Summary statistics
        print("\\n" + "-" * 40)
        print(f"📈 SUMMARY")
        print("-" * 40)
        print(f"✅ Successful symbols: {successful_symbols}/{len(symbols)}")
        print(f"📊 Total records collected: {total_records:,}")
        print(f"🗄️  Database: {pipeline.config['database_path']}")
        
        if successful_symbols == len(symbols):
            print(f"\\n🎉 SUCCESS: All symbols collected successfully!")
            print(f"🔍 Ready for scenario analysis with {total_records:,} data points")
        elif successful_symbols > 0:
            print(f"\\n⚠️  PARTIAL SUCCESS: {successful_symbols}/{len(symbols)} symbols collected")
            print(f"💡 Consider re-running for failed symbols")
        else:
            print(f"\\n❌ FAILURE: No symbols collected successfully")
            print(f"🔧 Check API key and configuration")
        
        print(f"\\n📝 Next step: Run scenario analysis with collected data")
        print(f"   Command: cd ../../metrics && python run_scenario_analysis.py")
        
    except Exception as e:
        print(f"❌ Fatal error during data collection: {e}")
        print(f"🔧 Check configuration and API key in {config_path}")
        sys.exit(1)

def check_prerequisites():
    """Check if all prerequisites are met before data collection."""
    issues = []
    
    # Check if config file exists
    config_path = "config/alpha_vantage_config.json"
    if not os.path.exists(config_path):
        issues.append(f"Configuration file not found: {config_path}")
    
    # Check if alpha_vantage module is available
    try:
        from alpha_vantage.timeseries import TimeSeries
    except ImportError:
        issues.append("alpha_vantage library not installed. Run: pip install alpha_vantage")
    
    # Check if data directory will be created
    data_dir = Path("data/databases")
    if not data_dir.parent.exists():
        issues.append(f"Data directory structure needs setup: {data_dir.parent}")
    
    if issues:
        print("❌ Prerequisites check failed:")
        for issue in issues:
            print(f"   • {issue}")
        return False
    
    return True

if __name__ == "__main__":
    print("🔍 Checking prerequisites...")
    
    if not check_prerequisites():
        print("\\n🛠️  Please resolve the issues above and try again")
        sys.exit(1)
    
    print("✅ Prerequisites check passed")
    main()