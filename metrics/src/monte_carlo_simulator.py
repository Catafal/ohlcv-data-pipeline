"""
Monte Carlo Simulator

Advanced Monte Carlo simulation engine for generating realistic market scenarios.
Supports multiple distribution models, volatility regimes, and correlation structures.

This module provides sophisticated simulation capabilities including:
- Multiple distribution models (Normal, Student-t, Skewed distributions)
- Volatility regime modeling (Low, Normal, High volatility periods)
- Dynamic correlation modeling during stress periods
- Bootstrap resampling from historical data
- Custom scenario generation with user-defined parameters

The simulator is designed to generate thousands of realistic market scenarios
for comprehensive risk analysis and stress testing.

Author: OHLCV Data Pipeline - Scenario Analysis Module
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta

class DistributionType(Enum):
    """Types of probability distributions for return modeling."""
    NORMAL = "normal"                    # Standard normal distribution
    STUDENT_T = "student_t"             # Student-t distribution (fat tails)
    SKEWED_NORMAL = "skewed_normal"     # Skewed normal distribution
    EMPIRICAL = "empirical"             # Empirical distribution from data
    MIXTURE = "mixture"                 # Mixture of distributions

class VolatilityRegime(Enum):
    """Different market volatility regimes."""
    LOW = "low_volatility"              # Calm market periods (VIX < 15)
    NORMAL = "normal_volatility"        # Average market conditions (VIX 15-25)
    HIGH = "high_volatility"            # Stressed market periods (VIX 25-40)
    CRISIS = "crisis_volatility"        # Crisis periods (VIX > 40)

@dataclass
class SimulationParameters:
    """
    Parameters for Monte Carlo simulation configuration.
    
    Contains all settings needed to generate realistic market scenarios
    including distribution assumptions, correlation models, and regime parameters.
    """
    num_simulations: int = 10000           # Number of simulation paths
    time_horizon_days: int = 252           # Simulation period (1 year default)
    distribution_type: DistributionType = DistributionType.STUDENT_T
    volatility_regime: VolatilityRegime = VolatilityRegime.NORMAL
    correlation_model: str = "constant"    # "constant", "dynamic", "regime_dependent"
    random_seed: Optional[int] = 42        # For reproducible results
    confidence_levels: List[float] = None  # VaR confidence levels
    
    def __post_init__(self):
        """Set default confidence levels if not provided."""
        if self.confidence_levels is None:
            self.confidence_levels = [0.01, 0.05, 0.10]

class MonteCarloSimulator:
    """
    Advanced Monte Carlo simulation engine for financial scenario generation.
    
    This class provides comprehensive Monte Carlo simulation capabilities
    for generating realistic market scenarios based on historical data patterns.
    """
    
    def __init__(self, 
                 returns_data: pd.DataFrame,
                 simulation_params: Optional[SimulationParameters] = None):
        """
        Initialize Monte Carlo simulator with historical returns data.
        
        Args:
            returns_data: DataFrame with historical returns (dates x symbols)
            simulation_params: Simulation configuration parameters
        """
        self.returns_data = returns_data.dropna()  # Remove any NaN values
        self.symbols = list(returns_data.columns)
        self.params = simulation_params or SimulationParameters()
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Calculate historical statistics
        self._calculate_historical_statistics()
        
        # Fit distribution parameters
        self._fit_distributions()
        
        # Set random seed for reproducibility
        if self.params.random_seed is not None:
            np.random.seed(self.params.random_seed)
        
        self.logger.info(f"MonteCarloSimulator initialized with {len(self.symbols)} symbols, "
                        f"{len(self.returns_data)} historical observations")
    
    def _calculate_historical_statistics(self):
        """Calculate comprehensive historical statistics for simulation calibration."""
        
        # Basic statistics
        self.mean_returns = self.returns_data.mean()
        self.volatilities = self.returns_data.std()
        self.correlation_matrix = self.returns_data.corr()
        
        # Higher moments for distribution fitting
        self.skewness = self.returns_data.skew()
        self.kurtosis = self.returns_data.kurtosis()
        
        # Volatility regime analysis
        self._analyze_volatility_regimes()
        
        # Rolling correlations for dynamic modeling
        self._calculate_rolling_correlations()
        
        self.logger.info("Historical statistics calculated")
    
    def _analyze_volatility_regimes(self):
        """Analyze different volatility regimes in historical data."""
        
        # Calculate rolling volatility (21-day window)
        rolling_vol = self.returns_data.rolling(window=21).std() * np.sqrt(252)
        portfolio_vol = rolling_vol.mean(axis=1)  # Average across securities
        
        # Define regime thresholds (approximate VIX equivalents)
        low_threshold = portfolio_vol.quantile(0.25)     # Low volatility
        normal_threshold = portfolio_vol.quantile(0.75)  # Normal volatility
        high_threshold = portfolio_vol.quantile(0.90)    # High volatility
        
        # Categorize periods
        regime_conditions = [
            (portfolio_vol <= low_threshold, VolatilityRegime.LOW),
            ((portfolio_vol > low_threshold) & (portfolio_vol <= normal_threshold), VolatilityRegime.NORMAL),
            ((portfolio_vol > normal_threshold) & (portfolio_vol <= high_threshold), VolatilityRegime.HIGH),
            (portfolio_vol > high_threshold, VolatilityRegime.CRISIS)
        ]
        
        # Calculate regime-specific statistics
        self.regime_stats = {}
        
        for condition, regime in regime_conditions:
            if condition.sum() > 10:  # Need sufficient observations
                regime_data = self.returns_data[condition]
                self.regime_stats[regime] = {
                    'mean_returns': regime_data.mean(),
                    'volatilities': regime_data.std(),
                    'correlations': regime_data.corr(),
                    'frequency': condition.sum() / len(portfolio_vol),
                    'avg_volatility': portfolio_vol[condition].mean()
                }
        
        self.logger.info(f"Identified {len(self.regime_stats)} volatility regimes")
    
    def _calculate_rolling_correlations(self, window: int = 63):
        """Calculate rolling correlations for dynamic correlation modeling."""
        
        # Calculate rolling correlations between all pairs
        self.rolling_correlations = {}
        
        for i, symbol1 in enumerate(self.symbols):
            for j, symbol2 in enumerate(self.symbols[i+1:], i+1):
                pair = f"{symbol1}_{symbol2}"
                rolling_corr = self.returns_data[symbol1].rolling(window=window).corr(
                    self.returns_data[symbol2]
                )
                self.rolling_correlations[pair] = rolling_corr.dropna()
        
        self.logger.info(f"Calculated rolling correlations with {window}-day window")
    
    def _fit_distributions(self):
        """Fit various probability distributions to historical returns."""
        
        self.distribution_params = {}
        
        for symbol in self.symbols:
            returns = self.returns_data[symbol].dropna()
            
            # Fit different distributions
            distributions = {}
            
            # Normal distribution
            distributions['normal'] = {
                'params': (returns.mean(), returns.std()),
                'distribution': stats.norm
            }
            
            # Student-t distribution  
            try:
                t_params = stats.t.fit(returns)
                distributions['student_t'] = {
                    'params': t_params,
                    'distribution': stats.t
                }
            except Exception as e:
                self.logger.warning(f"Failed to fit t-distribution for {symbol}: {e}")
            
            # Skewed normal distribution
            try:
                skewnorm_params = stats.skewnorm.fit(returns)
                distributions['skewed_normal'] = {
                    'params': skewnorm_params,
                    'distribution': stats.skewnorm
                }
            except Exception as e:
                self.logger.warning(f"Failed to fit skewed normal for {symbol}: {e}")
            
            self.distribution_params[symbol] = distributions
        
        self.logger.info("Distribution parameters fitted for all symbols")
    
    def generate_scenarios(self, 
                          custom_params: Optional[SimulationParameters] = None) -> Dict[str, Any]:
        """
        Generate Monte Carlo scenarios based on configured parameters.
        
        Args:
            custom_params: Override default simulation parameters
            
        Returns:
            Dictionary containing simulation results and statistics
        """
        params = custom_params or self.params
        
        self.logger.info(f"Generating {params.num_simulations} scenarios over "
                        f"{params.time_horizon_days} days")
        
        # Generate scenarios based on distribution type
        if params.distribution_type == DistributionType.NORMAL:
            scenarios = self._generate_normal_scenarios(params)
        elif params.distribution_type == DistributionType.STUDENT_T:
            scenarios = self._generate_student_t_scenarios(params)
        elif params.distribution_type == DistributionType.EMPIRICAL:
            scenarios = self._generate_bootstrap_scenarios(params)
        else:
            # Default to Student-t
            scenarios = self._generate_student_t_scenarios(params)
        
        # Calculate scenario statistics
        scenario_stats = self._calculate_scenario_statistics(scenarios, params)
        
        # Package results
        results = {
            'scenarios': scenarios,
            'statistics': scenario_stats,
            'parameters': params,
            'symbols': self.symbols,
            'simulation_date': datetime.now().isoformat()
        }
        
        self.logger.info("Monte Carlo simulation completed successfully")
        return results
    
    def _generate_normal_scenarios(self, params: SimulationParameters) -> np.ndarray:
        """Generate scenarios using multivariate normal distribution."""
        
        # Get regime-specific parameters if using regime modeling
        if params.volatility_regime in self.regime_stats:
            mean_returns = self.regime_stats[params.volatility_regime]['mean_returns']
            correlation_matrix = self.regime_stats[params.volatility_regime]['correlations']
            volatilities = self.regime_stats[params.volatility_regime]['volatilities']
        else:
            mean_returns = self.mean_returns
            correlation_matrix = self.correlation_matrix
            volatilities = self.volatilities
        
        # Convert to covariance matrix
        cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix
        
        # Generate random scenarios
        scenarios = np.random.multivariate_normal(
            mean=mean_returns.values,
            cov=cov_matrix.values,
            size=(params.num_simulations, params.time_horizon_days)
        )
        
        return scenarios
    
    def _generate_student_t_scenarios(self, params: SimulationParameters) -> np.ndarray:
        """Generate scenarios using multivariate Student-t distribution."""
        
        # Get parameters
        if params.volatility_regime in self.regime_stats:
            mean_returns = self.regime_stats[params.volatility_regime]['mean_returns']
            correlation_matrix = self.regime_stats[params.volatility_regime]['correlations']
            volatilities = self.regime_stats[params.volatility_regime]['volatilities']
        else:
            mean_returns = self.mean_returns
            correlation_matrix = self.correlation_matrix
            volatilities = self.volatilities
        
        # Estimate degrees of freedom from kurtosis
        # Kurtosis of t-distribution = 6/(df-4) for df > 4
        avg_kurtosis = self.kurtosis.mean()
        if avg_kurtosis > 0:
            df = 6 / avg_kurtosis + 4
            df = max(df, 5)  # Minimum 5 degrees of freedom
        else:
            df = 10  # Default
        
        # Generate scenarios using Cholesky decomposition
        L = np.linalg.cholesky(correlation_matrix)
        
        scenarios = np.zeros((params.num_simulations, params.time_horizon_days, len(self.symbols)))
        
        for sim in range(params.num_simulations):
            for day in range(params.time_horizon_days):
                # Generate independent Student-t random variables
                t_random = stats.t.rvs(df=df, size=len(self.symbols))
                
                # Apply correlation structure
                correlated_random = L @ t_random
                
                # Scale by volatility and add mean
                scenarios[sim, day, :] = (mean_returns.values + 
                                        volatilities.values * correlated_random)
        
        return scenarios
    
    def _generate_bootstrap_scenarios(self, params: SimulationParameters) -> np.ndarray:
        """Generate scenarios using bootstrap resampling from historical data."""
        
        # Determine data to sample from based on volatility regime
        if params.volatility_regime in self.regime_stats:
            # Sample from specific regime data
            regime_condition = self._get_regime_condition(params.volatility_regime)
            sample_data = self.returns_data[regime_condition]
        else:
            sample_data = self.returns_data
        
        scenarios = np.zeros((params.num_simulations, params.time_horizon_days, len(self.symbols)))
        
        for sim in range(params.num_simulations):
            # Bootstrap sample with replacement
            sampled_indices = np.random.choice(
                len(sample_data), 
                size=params.time_horizon_days, 
                replace=True
            )
            scenarios[sim, :, :] = sample_data.iloc[sampled_indices].values
        
        return scenarios
    
    def _get_regime_condition(self, regime: VolatilityRegime) -> pd.Series:
        """Get boolean condition for specific volatility regime."""
        
        # Calculate portfolio volatility
        rolling_vol = self.returns_data.rolling(window=21).std() * np.sqrt(252)
        portfolio_vol = rolling_vol.mean(axis=1)
        
        # Define thresholds
        low_threshold = portfolio_vol.quantile(0.25)
        normal_threshold = portfolio_vol.quantile(0.75)
        high_threshold = portfolio_vol.quantile(0.90)
        
        if regime == VolatilityRegime.LOW:
            return portfolio_vol <= low_threshold
        elif regime == VolatilityRegime.NORMAL:
            return (portfolio_vol > low_threshold) & (portfolio_vol <= normal_threshold)
        elif regime == VolatilityRegime.HIGH:
            return (portfolio_vol > normal_threshold) & (portfolio_vol <= high_threshold)
        elif regime == VolatilityRegime.CRISIS:
            return portfolio_vol > high_threshold
        else:
            return pd.Series([True] * len(portfolio_vol), index=portfolio_vol.index)
    
    def _calculate_scenario_statistics(self, 
                                     scenarios: np.ndarray, 
                                     params: SimulationParameters) -> Dict[str, Any]:
        """Calculate comprehensive statistics from generated scenarios."""
        
        # Convert to returns format: (simulations, days, symbols)
        num_sims, num_days, num_symbols = scenarios.shape
        
        # Calculate cumulative returns for each simulation
        cumulative_returns = np.cumprod(1 + scenarios, axis=1) - 1
        
        # Final period returns
        final_returns = cumulative_returns[:, -1, :]  # Shape: (simulations, symbols)
        
        # Portfolio returns (equal weighted)
        portfolio_weights = np.ones(num_symbols) / num_symbols
        portfolio_returns = final_returns @ portfolio_weights
        
        # Calculate statistics
        stats_dict = {}
        
        # Individual security statistics
        for i, symbol in enumerate(self.symbols):
            symbol_returns = final_returns[:, i]
            
            stats_dict[symbol] = {
                'mean_return': float(np.mean(symbol_returns)),
                'volatility': float(np.std(symbol_returns)),
                'percentiles': {
                    '1%': float(np.percentile(symbol_returns, 1)),
                    '5%': float(np.percentile(symbol_returns, 5)),
                    '10%': float(np.percentile(symbol_returns, 10)),
                    '25%': float(np.percentile(symbol_returns, 25)),
                    '50%': float(np.percentile(symbol_returns, 50)),
                    '75%': float(np.percentile(symbol_returns, 75)),
                    '90%': float(np.percentile(symbol_returns, 90)),
                    '95%': float(np.percentile(symbol_returns, 95)),
                    '99%': float(np.percentile(symbol_returns, 99))
                },
                'var_estimates': {
                    f'{int(level*100)}%': float(np.percentile(symbol_returns, level*100))
                    for level in params.confidence_levels
                }
            }
        
        # Portfolio statistics
        stats_dict['portfolio'] = {
            'mean_return': float(np.mean(portfolio_returns)),
            'volatility': float(np.std(portfolio_returns)),
            'percentiles': {
                '1%': float(np.percentile(portfolio_returns, 1)),
                '5%': float(np.percentile(portfolio_returns, 5)),
                '10%': float(np.percentile(portfolio_returns, 10)),
                '25%': float(np.percentile(portfolio_returns, 25)),
                '50%': float(np.percentile(portfolio_returns, 50)),
                '75%': float(np.percentile(portfolio_returns, 75)),
                '90%': float(np.percentile(portfolio_returns, 90)),
                '95%': float(np.percentile(portfolio_returns, 95)),
                '99%': float(np.percentile(portfolio_returns, 99))
            },
            'var_estimates': {
                f'{int(level*100)}%': float(np.percentile(portfolio_returns, level*100))
                for level in params.confidence_levels
            }
        }
        
        # Correlation statistics
        final_corr_matrix = np.corrcoef(final_returns.T)
        stats_dict['correlations'] = {
            'mean_correlation': float(np.mean(final_corr_matrix[np.triu_indices_from(final_corr_matrix, k=1)])),
            'min_correlation': float(np.min(final_corr_matrix[np.triu_indices_from(final_corr_matrix, k=1)])),
            'max_correlation': float(np.max(final_corr_matrix[np.triu_indices_from(final_corr_matrix, k=1)])),
            'correlation_matrix': final_corr_matrix.tolist()
        }
        
        return stats_dict
    
    def generate_stressed_scenarios(self, 
                                  stress_factor: float = 2.0,
                                  correlation_stress: float = 1.5) -> Dict[str, Any]:
        """
        Generate stressed scenarios with amplified volatilities and correlations.
        
        Args:
            stress_factor: Factor to multiply volatilities by
            correlation_stress: Factor to increase correlations toward 1
            
        Returns:
            Dictionary with stressed scenario results
        """
        self.logger.info(f"Generating stressed scenarios with {stress_factor}x volatility")
        
        # Create stressed parameters
        stressed_params = SimulationParameters(
            num_simulations=self.params.num_simulations,
            time_horizon_days=self.params.time_horizon_days,
            distribution_type=DistributionType.STUDENT_T,  # Use fat-tailed distribution
            volatility_regime=VolatilityRegime.CRISIS,
            random_seed=self.params.random_seed
        )
        
        # Temporarily modify statistics for stress testing
        original_volatilities = self.volatilities.copy()
        original_correlation = self.correlation_matrix.copy()
        
        try:
            # Apply stress to volatilities
            self.volatilities = self.volatilities * stress_factor
            
            # Apply stress to correlations (increase toward 1)
            stressed_corr = self.correlation_matrix.copy()
            for i in range(len(stressed_corr)):
                for j in range(len(stressed_corr)):
                    if i != j:
                        stressed_corr.iloc[i, j] = min(
                            1.0, 
                            stressed_corr.iloc[i, j] * correlation_stress
                        )
            self.correlation_matrix = stressed_corr
            
            # Generate stressed scenarios
            results = self.generate_scenarios(stressed_params)
            results['stress_parameters'] = {
                'volatility_stress_factor': stress_factor,
                'correlation_stress_factor': correlation_stress
            }
            
        finally:
            # Restore original parameters
            self.volatilities = original_volatilities
            self.correlation_matrix = original_correlation
        
        return results