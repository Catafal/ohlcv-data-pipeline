#!/usr/bin/env python3
"""
Scenario Analysis Runner

Main execution script for comprehensive scenario analysis of the Magnificent 7 companies.
This script orchestrates the complete analysis workflow including:

1. Data loading and validation
2. Historical scenario analysis  
3. Monte Carlo simulations
4. Risk metrics calculation
5. Visualization and reporting

The analysis covers 10 years of historical data and generates detailed reports
with interactive visualizations for investment decision making.

Usage:
    python run_scenario_analysis.py [--symbols SYMBOL1,SYMBOL2] [--scenarios all|specific]

Author: Jordi Catfal - OHLCV Data Pipeline - Scenario Analysis Module
"""

import sys
import os
import argparse
from pathlib import Path
import logging
from datetime import datetime, date
import json
import numpy as np
from typing import Dict, List, Optional, Any

# Add source directory to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

# Import our modules
from scenario_analysis_engine import ScenarioAnalysisEngine
from scenario_definitions import ScenarioLibrary, ScenarioType
from monte_carlo_simulator import MonteCarloSimulator, SimulationParameters, DistributionType, VolatilityRegime
from risk_metrics import RiskMetricsCalculator
from scenario_plots import ScenarioPlotter

class ScenarioAnalysisRunner:
    """
    Main runner class for comprehensive scenario analysis.
    
    Orchestrates the complete analysis workflow from data loading
    through visualization and reporting.
    """
    
    def __init__(self, 
                 db_path: str = "../data/alpha_vantage_pipeline/data/databases/alpha_vantage_data.db",
                 output_dir: str = "output"):
        """
        Initialize the scenario analysis runner.
        
        Args:
            db_path: Path to the SQLite database with OHLCV data
            output_dir: Directory for saving analysis outputs
        """
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        
        # Initialize components
        self.engine = ScenarioAnalysisEngine(db_path, output_dir=str(self.output_dir))
        self.scenario_library = ScenarioLibrary()
        self.risk_calculator = RiskMetricsCalculator()
        self.plotter = ScenarioPlotter(output_dir=str(self.output_dir / "plots"), plot_backend="both")
        
        # Set up logging
        self._setup_logging()
        
        self.logger.info(f"ScenarioAnalysisRunner initialized with database: {db_path}")
    
    def _setup_logging(self):
        """Setup comprehensive logging for the analysis."""
        log_file = self.output_dir / "scenario_analysis.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def run_complete_analysis(self, 
                            symbols: Optional[List[str]] = None,
                            scenarios: Optional[List[str]] = None,
                            monte_carlo_sims: int = 10000) -> Dict[str, Any]:
        """
        Run the complete scenario analysis workflow.
        
        Args:
            symbols: List of symbols to analyze (None for all available)
            scenarios: List of scenario IDs to analyze (None for all)
            monte_carlo_sims: Number of Monte Carlo simulations
            
        Returns:
            Dictionary with complete analysis results
        """
        self.logger.info("🚀 Starting comprehensive scenario analysis")
        self.logger.info("=" * 60)
        
        # Step 1: Data validation and loading
        self.logger.info("📊 Step 1: Data Loading and Validation")
        available_symbols = self.engine.get_available_symbols()
        
        if symbols is None:
            symbols = available_symbols
        else:
            # Validate requested symbols
            missing_symbols = [s for s in symbols if s not in available_symbols]
            if missing_symbols:
                self.logger.warning(f"Missing symbols: {missing_symbols}")
                symbols = [s for s in symbols if s in available_symbols]
        
        self.logger.info(f"Analyzing symbols: {symbols}")
        
        # Get data summary
        data_summary = self.engine.get_data_summary()
        self.logger.info(f"Data period: {data_summary['date_range']['start']} to {data_summary['date_range']['end']}")
        self.logger.info(f"Total records: {data_summary['total_records']:,}")
        
        # Step 2: Historical scenario analysis
        self.logger.info("\\n📈 Step 2: Historical Scenario Analysis")
        scenario_results = self._run_historical_scenarios(symbols, scenarios)
        
        # Step 3: Monte Carlo simulation
        self.logger.info("\\n🎲 Step 3: Monte Carlo Simulation")
        monte_carlo_results = self._run_monte_carlo_analysis(symbols, monte_carlo_sims)
        
        # Step 4: Risk metrics calculation
        self.logger.info("\\n⚠️  Step 4: Risk Metrics Calculation")
        risk_metrics = self._calculate_comprehensive_risk_metrics(symbols)
        
        # Step 5: Generate visualizations
        self.logger.info("\\n📊 Step 5: Visualization Generation")
        visualizations = self._generate_visualizations(
            symbols, scenario_results, monte_carlo_results, risk_metrics
        )
        
        # Step 6: Generate reports
        self.logger.info("\\n📋 Step 6: Report Generation")
        reports = self._generate_reports(
            symbols, scenario_results, monte_carlo_results, risk_metrics, data_summary
        )
        
        # Package complete results
        complete_results = {
            'metadata': {
                'analysis_date': datetime.now().isoformat(),
                'symbols_analyzed': symbols,
                'data_period': data_summary['date_range'],
                'total_observations': data_summary['total_records'],
                'monte_carlo_simulations': monte_carlo_sims
            },
            'data_summary': data_summary,
            'historical_scenarios': scenario_results,
            'monte_carlo_results': monte_carlo_results,
            'risk_metrics': {k: self._risk_metrics_to_dict(v) for k, v in risk_metrics.items()},
            'visualizations': visualizations,
            'reports': reports
        }
        
        # Save complete results
        self._save_complete_results(complete_results)
        
        self.logger.info("\\n✅ Scenario analysis completed successfully!")
        self.logger.info(f"📁 Results saved to: {self.output_dir}")
        
        return complete_results
    
    def _run_historical_scenarios(self, 
                                symbols: List[str], 
                                scenario_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run analysis for historical market scenarios."""
        
        # Get scenarios to analyze
        all_scenarios = self.scenario_library.get_all_scenarios()
        
        if scenario_ids is None:
            scenarios_to_analyze = list(all_scenarios.values())
        else:
            scenarios_to_analyze = [all_scenarios[sid] for sid in scenario_ids if sid in all_scenarios]
        
        self.logger.info(f"Analyzing {len(scenarios_to_analyze)} historical scenarios")
        
        scenario_results = {}
        
        for scenario in scenarios_to_analyze:
            self.logger.info(f"  📊 Analyzing: {scenario.name}")
            
            try:
                # Load data for scenario period
                scenario_data = self.engine.load_data(
                    symbols=symbols,
                    start_date=scenario.start_date.strftime('%Y-%m-%d'),
                    end_date=scenario.end_date.strftime('%Y-%m-%d')
                )
                
                if scenario_data.empty:
                    self.logger.warning(f"    No data available for {scenario.name}")
                    continue
                
                # Get returns matrix
                returns_matrix = self.engine.get_returns_matrix(
                    symbols=symbols,
                    start_date=scenario.start_date.strftime('%Y-%m-%d'),
                    end_date=scenario.end_date.strftime('%Y-%m-%d')
                )
                
                if returns_matrix.empty:
                    continue
                
                # Calculate scenario performance metrics
                scenario_performance = {}
                
                for symbol in symbols:
                    if symbol in returns_matrix.columns:
                        symbol_returns = returns_matrix[symbol].dropna()
                        
                        if len(symbol_returns) > 0:
                            total_return = (1 + symbol_returns).prod() - 1
                            volatility = symbol_returns.std() * np.sqrt(252)
                            max_drawdown = self._calculate_max_drawdown(symbol_returns)
                            
                            scenario_performance[symbol] = {
                                'total_return': float(total_return),
                                'annualized_volatility': float(volatility),
                                'max_drawdown': float(max_drawdown),
                                'num_observations': len(symbol_returns)
                            }
                
                # Portfolio performance (equal weighted)
                portfolio_returns = self.engine.calculate_portfolio_returns(returns_matrix)
                if len(portfolio_returns) > 0:
                    portfolio_total_return = (1 + portfolio_returns).prod() - 1
                    portfolio_volatility = portfolio_returns.std() * np.sqrt(252)
                    portfolio_max_drawdown = self._calculate_max_drawdown(portfolio_returns)
                    
                    scenario_performance['portfolio'] = {
                        'total_return': float(portfolio_total_return),
                        'annualized_volatility': float(portfolio_volatility),
                        'max_drawdown': float(portfolio_max_drawdown),
                        'num_observations': len(portfolio_returns)
                    }
                
                scenario_results[scenario.scenario_id] = {
                    'scenario_info': {
                        'name': scenario.name,
                        'type': scenario.scenario_type.value,
                        'start_date': scenario.start_date.strftime('%Y-%m-%d'),
                        'end_date': scenario.end_date.strftime('%Y-%m-%d'),
                        'duration_days': scenario.duration_days,
                        'severity_level': scenario.severity_level
                    },
                    'performance': scenario_performance
                }
                
                self.logger.info(f"    ✓ Completed analysis for {scenario.name}")
                
            except Exception as e:
                self.logger.error(f"    ❌ Error analyzing {scenario.name}: {e}")
                continue
        
        return scenario_results
    
    def _run_monte_carlo_analysis(self, symbols: List[str], num_simulations: int) -> Dict[str, Any]:
        """Run Monte Carlo simulation analysis."""
        
        # Load historical data for simulation calibration
        returns_data = self.engine.get_returns_matrix(symbols)
        
        if returns_data.empty:
            self.logger.error("No returns data available for Monte Carlo simulation")
            return {}
        
        self.logger.info(f"Running Monte Carlo simulation with {num_simulations:,} iterations")
        
        # Configure simulation parameters
        sim_params = SimulationParameters(
            num_simulations=num_simulations,
            time_horizon_days=252,  # 1 year
            distribution_type=DistributionType.STUDENT_T,
            volatility_regime=VolatilityRegime.NORMAL,
            random_seed=42
        )
        
        # Initialize simulator
        simulator = MonteCarloSimulator(returns_data, sim_params)
        
        # Run simulations
        results = {}
        
        # Normal market conditions
        self.logger.info("  🎯 Simulating normal market conditions")
        normal_results = simulator.generate_scenarios(sim_params)
        results['normal_market'] = normal_results
        
        # Stressed scenarios
        self.logger.info("  ⚡ Simulating stressed market conditions")
        stressed_results = simulator.generate_stressed_scenarios(
            stress_factor=2.0,
            correlation_stress=1.5
        )
        results['stressed_market'] = stressed_results
        
        # High volatility regime
        self.logger.info("  📈 Simulating high volatility regime")
        high_vol_params = SimulationParameters(
            num_simulations=num_simulations,
            time_horizon_days=252,
            distribution_type=DistributionType.STUDENT_T,
            volatility_regime=VolatilityRegime.HIGH,
            random_seed=42
        )
        high_vol_results = simulator.generate_scenarios(high_vol_params)
        results['high_volatility'] = high_vol_results
        
        return results
    
    def _calculate_comprehensive_risk_metrics(self, symbols: List[str]) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics for all symbols."""
        
        risk_metrics = {}
        
        # Load full historical data
        returns_data = self.engine.get_returns_matrix(symbols)
        
        if returns_data.empty:
            self.logger.error("No returns data available for risk metrics calculation")
            return {}
        
        # Calculate risk metrics for each symbol
        for symbol in symbols:
            if symbol in returns_data.columns:
                symbol_returns = returns_data[symbol].dropna()
                
                if len(symbol_returns) > 30:  # Minimum data requirement
                    try:
                        metrics = self.risk_calculator.calculate_comprehensive_metrics(
                            symbol_returns, symbol=symbol
                        )
                        risk_metrics[symbol] = metrics
                        
                        self.logger.info(f"  ✓ Risk metrics calculated for {symbol}")
                        
                    except Exception as e:
                        self.logger.error(f"  ❌ Error calculating risk metrics for {symbol}: {e}")
        
        # Calculate portfolio risk metrics (equal weighted)
        portfolio_returns = self.engine.calculate_portfolio_returns(returns_data)
        
        if len(portfolio_returns) > 30:
            try:
                portfolio_metrics = self.risk_calculator.calculate_comprehensive_metrics(
                    portfolio_returns, symbol="Portfolio"
                )
                risk_metrics['Portfolio'] = portfolio_metrics
                
                self.logger.info("  ✓ Portfolio risk metrics calculated")
                
            except Exception as e:
                self.logger.error(f"  ❌ Error calculating portfolio risk metrics: {e}")
        
        return risk_metrics
    
    def _generate_visualizations(self, 
                               symbols: List[str],
                               scenario_results: Dict[str, Any],
                               monte_carlo_results: Dict[str, Any],
                               risk_metrics: Dict[str, Any]) -> Dict[str, str]:
        """Generate all visualizations and return file paths."""
        
        visualizations = {}
        
        try:
            # Load price data for overlays
            price_data = self.engine.load_data(symbols)
            
            if not price_data.empty:
                # Convert to price levels (pivot on close prices)
                price_matrix = price_data.pivot(index='date', columns='symbol', values='close')
                
                # Historical scenario overlay
                scenarios = list(self.scenario_library.get_all_scenarios().values())
                fig1 = self.plotter.plot_historical_scenario_overlay(
                    price_matrix, scenarios[:5], symbols  # Limit to first 5 scenarios
                )
                visualizations['historical_overlay'] = "Historical scenario overlay created"
                
            # Monte Carlo distributions (both formats)
            if 'normal_market' in monte_carlo_results:
                fig2 = self.plotter.plot_monte_carlo_distributions(
                    monte_carlo_results['normal_market'], symbols
                )
                fig2_mpl = self.plotter.plot_monte_carlo_distributions_matplotlib(
                    monte_carlo_results['normal_market'], symbols
                )
                visualizations['monte_carlo'] = "Monte Carlo distributions created (HTML + PNG)"
            
            # Risk metrics comparison (both Plotly and Matplotlib)
            if risk_metrics:
                fig3 = self.plotter.plot_risk_metrics_comparison(risk_metrics)
                fig3_mpl = self.plotter.plot_risk_metrics_matplotlib(risk_metrics)
                visualizations['risk_comparison'] = "Risk metrics comparison created (HTML + PNG)"
            
            # Drawdown analysis (both formats)
            returns_data = self.engine.get_returns_matrix(symbols)
            if not returns_data.empty:
                fig4 = self.plotter.plot_drawdown_analysis(returns_data, symbols)
                fig4_mpl = self.plotter.plot_drawdown_matplotlib(returns_data, symbols)
                visualizations['drawdown'] = "Drawdown analysis created (HTML + PNG)"
                
                # Correlation heatmap (both formats)
                fig5 = self.plotter.plot_correlation_heatmap(returns_data)
                fig5_mpl = self.plotter.plot_correlation_heatmap_matplotlib(returns_data)
                visualizations['correlation'] = "Correlation heatmap created (HTML + PNG)"
                
                # Efficient frontier
                fig6 = self.plotter.plot_efficient_frontier(returns_data)
                visualizations['efficient_frontier'] = "Efficient frontier created"
            
            self.logger.info(f"  ✓ Generated {len(visualizations)} visualizations")
            
        except Exception as e:
            self.logger.error(f"  ❌ Error generating visualizations: {e}")
        
        return visualizations
    
    def _generate_reports(self, 
                        symbols: List[str],
                        scenario_results: Dict[str, Any],
                        monte_carlo_results: Dict[str, Any],
                        risk_metrics: Dict[str, Any],
                        data_summary: Dict[str, Any]) -> Dict[str, str]:
        """Generate comprehensive text and CSV reports."""
        
        reports = {}
        
        try:
            # Risk metrics CSV report
            risk_df = self._create_risk_metrics_dataframe(risk_metrics)
            risk_csv_path = self.output_dir / "reports" / "risk_metrics_detailed.csv"
            risk_df.to_csv(risk_csv_path, index=False)
            reports['risk_metrics_csv'] = str(risk_csv_path)
            
            # Scenario performance CSV
            scenario_df = self._create_scenario_performance_dataframe(scenario_results)
            scenario_csv_path = self.output_dir / "reports" / "scenario_performance.csv"
            scenario_df.to_csv(scenario_csv_path, index=False)
            reports['scenario_performance_csv'] = str(scenario_csv_path)
            
            # Summary text report
            summary_path = self._generate_summary_report(
                symbols, scenario_results, monte_carlo_results, risk_metrics, data_summary
            )
            reports['summary_report'] = str(summary_path)
            
            self.logger.info(f"  ✓ Generated {len(reports)} reports")
            
        except Exception as e:
            self.logger.error(f"  ❌ Error generating reports: {e}")
        
        return reports
    
    def _calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown from returns series."""
        import numpy as np
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _risk_metrics_to_dict(self, metrics):
        """Convert RiskMetrics object to dictionary."""
        return {
            'symbol': metrics.symbol,
            'volatility': metrics.volatility,
            'sharpe_ratio': metrics.sharpe_ratio,
            'maximum_drawdown': metrics.maximum_drawdown,
            'var_95': metrics.var_95,
            'var_99': metrics.var_99,
            'expected_shortfall_95': metrics.expected_shortfall_95,
            'expected_shortfall_99': metrics.expected_shortfall_99
        }
    
    def _create_risk_metrics_dataframe(self, risk_metrics):
        """Create DataFrame from risk metrics for CSV export."""
        import pandas as pd
        
        data = []
        for symbol, metrics in risk_metrics.items():
            if hasattr(metrics, 'symbol'):  # Ensure it's a RiskMetrics object
                data.append({
                    'Symbol': metrics.symbol,
                    'Volatility (%)': metrics.volatility * 100,
                    'Sharpe Ratio': metrics.sharpe_ratio,
                    'Max Drawdown (%)': abs(metrics.maximum_drawdown) * 100,
                    '95% VaR (%)': abs(metrics.var_95) * 100,
                    '99% VaR (%)': abs(metrics.var_99) * 100,
                    'Expected Shortfall 95% (%)': abs(metrics.expected_shortfall_95) * 100,
                    'Expected Shortfall 99% (%)': abs(metrics.expected_shortfall_99) * 100,
                    'Sortino Ratio': metrics.sortino_ratio,
                    'Calmar Ratio': metrics.calmar_ratio
                })
        
        return pd.DataFrame(data)
    
    def _create_scenario_performance_dataframe(self, scenario_results):
        """Create DataFrame from scenario results for CSV export."""
        import pandas as pd
        
        data = []
        for scenario_id, results in scenario_results.items():
            scenario_info = results['scenario_info']
            performance = results['performance']
            
            for symbol, perf in performance.items():
                data.append({
                    'Scenario': scenario_info['name'],
                    'Symbol': symbol,
                    'Total Return (%)': perf['total_return'] * 100,
                    'Annualized Volatility (%)': perf['annualized_volatility'] * 100,
                    'Max Drawdown (%)': abs(perf['max_drawdown']) * 100,
                    'Duration (days)': scenario_info['duration_days'],
                    'Severity Level': scenario_info['severity_level']
                })
        
        return pd.DataFrame(data)
    
    def _generate_summary_report(self, symbols, scenario_results, monte_carlo_results, risk_metrics, data_summary):
        """Generate comprehensive summary text report."""
        
        report_path = self.output_dir / "reports" / "analysis_summary.txt"
        
        with open(report_path, 'w') as f:
            f.write(f"""
COMPREHENSIVE SCENARIO ANALYSIS REPORT
{'='*50}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATA SUMMARY
{'─'*20}
Symbols Analyzed: {', '.join(symbols)}
Data Period: {data_summary['date_range']['start']} to {data_summary['date_range']['end']}
Years of Data: {data_summary['years_of_data']}
Total Records: {data_summary['total_records']:,}

RISK METRICS SUMMARY
{'─'*20}
""")
            
            # Risk metrics summary
            if risk_metrics:
                for symbol, metrics in risk_metrics.items():
                    if hasattr(metrics, 'symbol'):
                        f.write(f"""
{symbol}:
  Volatility: {metrics.volatility:.2%}
  Sharpe Ratio: {metrics.sharpe_ratio:.3f}
  Max Drawdown: {abs(metrics.maximum_drawdown):.2%}
  95% VaR: {abs(metrics.var_95):.2%}
""")
            
            f.write(f"""
HISTORICAL SCENARIOS ANALYZED
{'─'*30}
""")
            
            # Scenario summary
            for scenario_id, results in scenario_results.items():
                scenario_info = results['scenario_info']
                f.write(f"""
{scenario_info['name']}:
  Period: {scenario_info['start_date']} to {scenario_info['end_date']}
  Duration: {scenario_info['duration_days']} days
  Severity: {scenario_info['severity_level']}/10
""")
        
        return report_path
    
    def _save_complete_results(self, results):
        """Save complete results to JSON file."""
        results_path = self.output_dir / "complete_analysis_results.json"
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Complete results saved to: {results_path}")

def main():
    """Main execution function with command line interface."""
    parser = argparse.ArgumentParser(description="Comprehensive Scenario Analysis for Magnificent 7")
    
    parser.add_argument('--symbols', type=str, 
                       help="Comma-separated list of symbols to analyze (default: all available)")
    parser.add_argument('--scenarios', type=str,
                       help="Comma-separated list of scenario IDs (default: all)")
    parser.add_argument('--simulations', type=int, default=5000,
                       help="Number of Monte Carlo simulations (default: 5000)")
    parser.add_argument('--output', type=str, default="output",
                       help="Output directory (default: output)")
    
    args = parser.parse_args()
    
    # Parse symbols
    symbols = None
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]
    
    # Parse scenarios
    scenarios = None
    if args.scenarios:
        scenarios = [s.strip() for s in args.scenarios.split(',')]
    
    # Initialize and run analysis
    try:
        runner = ScenarioAnalysisRunner(output_dir=args.output)
        
        print("🚀 Starting Comprehensive Scenario Analysis")
        print(f"📊 Magnificent 7 Companies Analysis")
        print(f"🎯 Monte Carlo Simulations: {args.simulations:,}")
        print(f"📁 Output Directory: {args.output}")
        print("=" * 60)
        
        results = runner.run_complete_analysis(
            symbols=symbols,
            scenarios=scenarios,
            monte_carlo_sims=args.simulations
        )
        
        print("\\n🎉 Analysis completed successfully!")
        print(f"📈 Analyzed {len(results['metadata']['symbols_analyzed'])} symbols")
        print(f"📊 Generated {len(results['visualizations'])} visualizations")
        print(f"📋 Created {len(results['reports'])} reports")
        print(f"\\n📁 All results saved to: {args.output}/")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()