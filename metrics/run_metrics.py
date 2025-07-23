#!/usr/bin/env python3
"""
Financial Metrics Runner

Main script to calculate and display financial metrics from OHLCV data.
This script processes data from all available databases and generates comprehensive metrics.

Usage:
    python run_metrics.py

Author: Jordi Catfal - OHLCV Data Pipeline - Financial Metrics Module
"""

import os
import sys
from src.metrics_calculator import FinancialMetricsCalculator

def find_database_files():
    """
    Find all OHLCV database files in the project.
    
    Returns:
        List of database file paths
    """
    db_files = []
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    # Known database locations (relative to project root)
    db_locations = [
        '../data/alpaca/data/databases/market_data.db',
        '../data/alpha_vantage_pipeline/data/databases/alpha_vantage_data.db', 
        '../data/yahoo-finance/data/databases/yfinance_data.db'
    ]
    
    for db_path in db_locations:
        full_path = os.path.join(base_path, db_path)
        if os.path.exists(full_path):
            db_files.append(full_path)
            print(f"✅ Found database: {db_path}")
        else:
            print(f"⚠️  Database not found: {db_path}")
    
    return db_files

def main():
    """Main execution function."""
    print("🚀 Financial Metrics Calculator")
    print("=" * 50)
    
    # Find available databases
    db_files = find_database_files()
    
    if not db_files:
        print("❌ No database files found. Please run the data pipelines first.")
        sys.exit(1)
    
    print(f"\n📊 Processing {len(db_files)} database(s)...")
    
    # Initialize calculator
    calculator = FinancialMetricsCalculator(risk_free_rate=0.02)  # 2% risk-free rate
    
    try:
        # Calculate metrics for all symbols
        metrics_df = calculator.calculate_all_metrics(db_files)
        
        if metrics_df.empty:
            print("❌ No metrics could be calculated. Check if databases contain data.")
            sys.exit(1)
        
        # Display results
        calculator.display_metrics(metrics_df)
        
        # Save to CSV in output folder
        output_file = "output/financial_metrics_report.csv"
        calculator.save_metrics_to_csv(metrics_df, output_file)
        
        print(f"\n💾 Detailed metrics saved to: {output_file}")
        print(f"📈 Analyzed {len(metrics_df)} symbols total")
        
        # Performance summary
        print("\n🏆 PERFORMANCE SUMMARY")
        print("-" * 30)
        best_return = metrics_df.loc[metrics_df['total_return'].idxmax()]
        best_sharpe = metrics_df.loc[metrics_df['sharpe_ratio'].idxmax()]
        
        print(f"Best Total Return: {best_return['symbol']} ({best_return['total_return']:.2f}%)")
        print(f"Best Sharpe Ratio: {best_sharpe['symbol']} ({best_sharpe['sharpe_ratio']:.3f})")
        
    except Exception as e:
        print(f"❌ Error calculating metrics: {e}")
        sys.exit(1)
    
    print("\n✨ Metrics calculation completed successfully!")

if __name__ == "__main__":
    main()