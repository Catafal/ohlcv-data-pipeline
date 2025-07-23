"""
Risk Metrics Calculator

Comprehensive suite of financial risk metrics and measures for scenario analysis.
Includes traditional risk measures, tail risk metrics, and advanced portfolio risk analytics.

This module provides a complete toolkit for quantifying financial risk including:
- Value at Risk (VaR) with multiple methodologies
- Expected Shortfall (Conditional VaR)
- Maximum Drawdown and recovery analysis
- Beta and correlation metrics
- Risk-adjusted performance measures
- Tail risk and extreme event analysis
- Portfolio risk decomposition

All metrics are calculated with both parametric and non-parametric methods
where applicable, providing robust risk assessment capabilities.

Author: OHLCV Data Pipeline - Scenario Analysis Module
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
from datetime import datetime
import warnings

@dataclass
class RiskMetrics:
    """
    Container for comprehensive risk metrics results.
    
    Stores all calculated risk measures with metadata for easy access
    and reporting. Includes both single-asset and portfolio-level metrics.
    """
    symbol: str                          # Security or portfolio identifier
    calculation_date: str                # When metrics were calculated
    data_period: str                     # Data period used for calculation
    observations: int                    # Number of data points used
    
    # Basic risk measures
    volatility: float                    # Annualized volatility
    downside_volatility: float          # Downside deviation
    tracking_error: Optional[float]      # Tracking error vs benchmark
    
    # Value at Risk measures
    var_95: float                       # 95% Value at Risk
    var_99: float                       # 99% Value at Risk
    var_parametric_95: float            # Parametric VaR (95%)
    var_parametric_99: float            # Parametric VaR (99%)
    
    # Expected Shortfall (Conditional VaR)
    expected_shortfall_95: float        # Expected loss beyond 95% VaR
    expected_shortfall_99: float        # Expected loss beyond 99% VaR
    
    # Drawdown analysis
    maximum_drawdown: float             # Maximum peak-to-trough decline
    average_drawdown: float             # Average of all drawdowns
    drawdown_duration: int              # Days in current/longest drawdown
    recovery_time: Optional[int]        # Average recovery time in days
    
    # Performance risk measures
    sharpe_ratio: float                 # Risk-adjusted return measure
    sortino_ratio: float                # Downside risk-adjusted return
    calmar_ratio: float                 # Return / Maximum Drawdown
    
    # Tail risk measures
    skewness: float                     # Return distribution skewness
    kurtosis: float                     # Return distribution kurtosis
    tail_ratio: float                   # 95th percentile / 5th percentile
    
    # Market risk measures
    beta: Optional[float]               # Market beta (if benchmark provided)
    correlation_to_market: Optional[float]  # Correlation with benchmark
    
    # Additional metrics
    var_ratio: float                    # VaR(99%) / VaR(95%) ratio
    hit_rate_95: float                  # Actual % of returns below VaR 95%
    hit_rate_99: float                  # Actual % of returns below VaR 99%

class RiskMetricsCalculator:
    """
    Comprehensive calculator for financial risk metrics and measures.
    
    This class provides a complete suite of risk calculation capabilities
    including traditional measures, tail risk analytics, and advanced
    portfolio risk decomposition methods.
    """
    
    def __init__(self, 
                 risk_free_rate: float = 0.02,
                 confidence_levels: List[float] = None,
                 annualization_factor: int = 252):
        """
        Initialize the risk metrics calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculations
            confidence_levels: VaR confidence levels (e.g., [0.95, 0.99])
            annualization_factor: Days per year for annualization (252 for daily data)
        """
        self.risk_free_rate = risk_free_rate
        self.confidence_levels = confidence_levels or [0.95, 0.99]
        self.annualization_factor = annualization_factor
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Daily risk-free rate
        self.daily_risk_free = risk_free_rate / annualization_factor
        
        self.logger.info(f"RiskMetricsCalculator initialized with {risk_free_rate:.2%} risk-free rate")
    
    def calculate_comprehensive_metrics(self, 
                                      returns: pd.Series,
                                      benchmark_returns: Optional[pd.Series] = None,
                                      symbol: str = "Portfolio") -> RiskMetrics:
        """
        Calculate comprehensive risk metrics for a return series.
        
        Args:
            returns: Series of daily returns
            benchmark_returns: Optional benchmark returns for beta calculation
            symbol: Identifier for the security/portfolio
            
        Returns:
            RiskMetrics object with all calculated measures
        """
        returns_clean = returns.dropna()
        
        if len(returns_clean) < 30:
            raise ValueError(f"Insufficient data: need at least 30 observations, got {len(returns_clean)}")
        
        self.logger.info(f"Calculating comprehensive risk metrics for {symbol}")
        
        # Basic statistics
        volatility = self._calculate_volatility(returns_clean)
        downside_vol = self._calculate_downside_volatility(returns_clean)
        
        # VaR calculations
        var_95, var_99 = self._calculate_var_historical(returns_clean)
        var_param_95, var_param_99 = self._calculate_var_parametric(returns_clean)
        
        # Expected Shortfall
        es_95 = self._calculate_expected_shortfall(returns_clean, 0.95)
        es_99 = self._calculate_expected_shortfall(returns_clean, 0.99)
        
        # Drawdown analysis
        dd_metrics = self._calculate_drawdown_metrics(returns_clean)
        
        # Performance metrics
        sharpe = self._calculate_sharpe_ratio(returns_clean)
        sortino = self._calculate_sortino_ratio(returns_clean)
        calmar = self._calculate_calmar_ratio(returns_clean, dd_metrics['max_drawdown'])
        
        # Distribution properties
        skew = float(returns_clean.skew())
        kurt = float(returns_clean.kurtosis())
        tail_ratio = self._calculate_tail_ratio(returns_clean)
        
        # Market risk metrics
        beta = None
        correlation = None
        if benchmark_returns is not None:
            beta, correlation = self._calculate_market_metrics(returns_clean, benchmark_returns)
        
        # Additional metrics
        var_ratio = abs(var_99) / abs(var_95) if var_95 != 0 else np.nan
        hit_rate_95 = self._calculate_var_hit_rate(returns_clean, var_95)
        hit_rate_99 = self._calculate_var_hit_rate(returns_clean, var_99)
        
        # Package results
        metrics = RiskMetrics(
            symbol=symbol,
            calculation_date=datetime.now().strftime('%Y-%m-%d'),
            data_period=f"{returns_clean.index[0].strftime('%Y-%m-%d')} to {returns_clean.index[-1].strftime('%Y-%m-%d')}",
            observations=len(returns_clean),
            
            # Basic risk measures
            volatility=volatility,
            downside_volatility=downside_vol,
            tracking_error=None,  # TODO: Implement if benchmark provided
            
            # VaR measures
            var_95=var_95,
            var_99=var_99,
            var_parametric_95=var_param_95,
            var_parametric_99=var_param_99,
            
            # Expected Shortfall
            expected_shortfall_95=es_95,
            expected_shortfall_99=es_99,
            
            # Drawdown metrics
            maximum_drawdown=dd_metrics['max_drawdown'],
            average_drawdown=dd_metrics['avg_drawdown'],
            drawdown_duration=dd_metrics['max_duration'],
            recovery_time=dd_metrics['avg_recovery_time'],
            
            # Performance metrics
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            
            # Tail risk
            skewness=skew,
            kurtosis=kurt,
            tail_ratio=tail_ratio,
            
            # Market risk
            beta=beta,
            correlation_to_market=correlation,
            
            # Additional
            var_ratio=var_ratio,
            hit_rate_95=hit_rate_95,
            hit_rate_99=hit_rate_99
        )
        
        self.logger.info(f"Risk metrics calculated for {symbol}: "
                        f"{volatility:.2%} volatility, {var_95:.2%} 95% VaR, "
                        f"{dd_metrics['max_drawdown']:.2%} max drawdown")
        
        return metrics
    
    def _calculate_volatility(self, returns: pd.Series) -> float:
        """Calculate annualized volatility."""
        return float(returns.std() * np.sqrt(self.annualization_factor))
    
    def _calculate_downside_volatility(self, returns: pd.Series, target_return: float = 0.0) -> float:
        """Calculate downside volatility (semi-deviation)."""
        downside_returns = returns[returns < target_return] - target_return
        if len(downside_returns) == 0:
            return 0.0
        return float(downside_returns.std() * np.sqrt(self.annualization_factor))
    
    def _calculate_var_historical(self, returns: pd.Series) -> Tuple[float, float]:
        """Calculate Value at Risk using historical method."""
        var_95 = float(np.percentile(returns, 5))   # 5th percentile for 95% VaR
        var_99 = float(np.percentile(returns, 1))   # 1st percentile for 99% VaR
        return var_95, var_99
    
    def _calculate_var_parametric(self, returns: pd.Series) -> Tuple[float, float]:
        """Calculate Value at Risk using parametric (normal distribution) method."""
        mean = returns.mean()
        std = returns.std()
        
        # Z-scores for confidence levels
        z_95 = stats.norm.ppf(0.05)  # -1.645
        z_99 = stats.norm.ppf(0.01)  # -2.326
        
        var_95 = float(mean + z_95 * std)
        var_99 = float(mean + z_99 * std)
        
        return var_95, var_99
    
    def _calculate_expected_shortfall(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Expected Shortfall (Conditional VaR)."""
        percentile = (1 - confidence_level) * 100
        var_threshold = np.percentile(returns, percentile)
        
        # Average of returns beyond VaR threshold
        tail_returns = returns[returns <= var_threshold]
        
        if len(tail_returns) == 0:
            return var_threshold
        
        return float(tail_returns.mean())
    
    def _calculate_drawdown_metrics(self, returns: pd.Series) -> Dict[str, Any]:
        """Calculate comprehensive drawdown metrics."""
        
        # Calculate cumulative returns
        cumulative_returns = (1 + returns).cumprod()
        
        # Calculate running maximum (peak)
        running_max = cumulative_returns.expanding().max()
        
        # Calculate drawdown
        drawdown = (cumulative_returns - running_max) / running_max
        
        # Maximum drawdown
        max_drawdown = float(drawdown.min())
        
        # Average drawdown (only negative values)
        negative_drawdowns = drawdown[drawdown < 0]
        avg_drawdown = float(negative_drawdowns.mean()) if len(negative_drawdowns) > 0 else 0.0
        
        # Drawdown duration analysis
        drawdown_periods = self._identify_drawdown_periods(drawdown)
        
        max_duration = 0
        recovery_times = []
        
        for period in drawdown_periods:
            duration = len(period)
            max_duration = max(max_duration, duration)
            
            # Recovery time is the length of the drawdown period
            recovery_times.append(duration)
        
        avg_recovery_time = int(np.mean(recovery_times)) if recovery_times else None
        
        return {
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'max_duration': max_duration,
            'avg_recovery_time': avg_recovery_time,
            'drawdown_series': drawdown
        }
    
    def _identify_drawdown_periods(self, drawdown: pd.Series) -> List[pd.Series]:
        """Identify individual drawdown periods."""
        periods = []
        in_drawdown = False
        start_idx = None
        
        for i, dd in enumerate(drawdown):
            if dd < -0.001 and not in_drawdown:  # Start of drawdown (0.1% threshold)
                in_drawdown = True
                start_idx = i
            elif dd >= -0.001 and in_drawdown:  # End of drawdown
                in_drawdown = False
                if start_idx is not None:
                    periods.append(drawdown.iloc[start_idx:i])
        
        # Handle case where we end in a drawdown
        if in_drawdown and start_idx is not None:
            periods.append(drawdown.iloc[start_idx:])
        
        return periods
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        excess_returns = returns - self.daily_risk_free
        
        if excess_returns.std() == 0:
            return 0.0
        
        return float(excess_returns.mean() / excess_returns.std() * np.sqrt(self.annualization_factor))
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio (using downside deviation)."""
        excess_returns = returns - self.daily_risk_free
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return float('inf') if excess_returns.mean() > 0 else 0.0
        
        return float(excess_returns.mean() / downside_returns.std() * np.sqrt(self.annualization_factor))
    
    def _calculate_calmar_ratio(self, returns: pd.Series, max_drawdown: float) -> float:
        """Calculate Calmar ratio (return / max drawdown)."""
        annual_return = returns.mean() * self.annualization_factor
        
        if max_drawdown == 0:
            return float('inf') if annual_return > 0 else 0.0
        
        return float(annual_return / abs(max_drawdown))
    
    def _calculate_tail_ratio(self, returns: pd.Series) -> float:
        """Calculate tail ratio (95th percentile / 5th percentile)."""
        p95 = np.percentile(returns, 95)
        p5 = np.percentile(returns, 5)
        
        if p5 == 0:
            return float('inf')
        
        return float(abs(p95 / p5))
    
    def _calculate_market_metrics(self, 
                                returns: pd.Series, 
                                benchmark_returns: pd.Series) -> Tuple[float, float]:
        """Calculate beta and correlation with benchmark."""
        
        # Align the series
        aligned_data = pd.DataFrame({
            'returns': returns,
            'benchmark': benchmark_returns
        }).dropna()
        
        if len(aligned_data) < 30:
            return None, None
        
        # Calculate correlation
        correlation = float(aligned_data['returns'].corr(aligned_data['benchmark']))
        
        # Calculate beta
        covariance = aligned_data['returns'].cov(aligned_data['benchmark'])
        benchmark_variance = aligned_data['benchmark'].var()
        
        if benchmark_variance == 0:
            beta = None
        else:
            beta = float(covariance / benchmark_variance)
        
        return beta, correlation
    
    def _calculate_var_hit_rate(self, returns: pd.Series, var_threshold: float) -> float:
        """Calculate the hit rate for VaR (percentage of returns below threshold)."""
        violations = (returns <= var_threshold).sum()
        return float(violations / len(returns))
    
    def calculate_portfolio_risk_decomposition(self, 
                                             returns_matrix: pd.DataFrame,
                                             weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Calculate portfolio risk decomposition and component contributions.
        
        Args:
            returns_matrix: DataFrame with asset returns (dates x symbols)
            weights: Portfolio weights (equal weight if None)
            
        Returns:
            Dictionary with risk decomposition analysis
        """
        if returns_matrix.empty:
            return {}
        
        # Default to equal weights
        if weights is None:
            n_assets = len(returns_matrix.columns)
            weights = {symbol: 1.0/n_assets for symbol in returns_matrix.columns}
        
        # Convert to aligned series
        weight_series = pd.Series(weights, index=returns_matrix.columns).fillna(0)
        weight_series = weight_series / weight_series.sum()  # Normalize
        
        # Calculate portfolio returns
        portfolio_returns = (returns_matrix * weight_series).sum(axis=1)
        
        # Portfolio risk metrics
        portfolio_vol = self._calculate_volatility(portfolio_returns)
        
        # Calculate covariance matrix
        cov_matrix = returns_matrix.cov() * self.annualization_factor
        
        # Calculate individual asset volatilities
        asset_vols = returns_matrix.std() * np.sqrt(self.annualization_factor)
        
        # Risk contribution calculation
        risk_contributions = {}
        
        for asset in returns_matrix.columns:
            # Marginal contribution to risk
            marginal_contrib = (cov_matrix @ weight_series)[asset] / portfolio_vol
            
            # Component contribution to risk
            component_contrib = weight_series[asset] * marginal_contrib
            
            # Percentage contribution
            pct_contrib = component_contrib / portfolio_vol if portfolio_vol != 0 else 0
            
            risk_contributions[asset] = {
                'weight': float(weight_series[asset]),
                'individual_volatility': float(asset_vols[asset]),
                'marginal_contribution': float(marginal_contrib),
                'component_contribution': float(component_contrib),
                'percent_contribution': float(pct_contrib)
            }
        
        # Portfolio-level metrics
        decomposition = {
            'portfolio_volatility': portfolio_vol,
            'diversification_ratio': float(sum(weight_series * asset_vols) / portfolio_vol) if portfolio_vol != 0 else 1.0,
            'risk_contributions': risk_contributions,
            'correlation_matrix': returns_matrix.corr().to_dict(),
            'weights': weights
        }
        
        self.logger.info(f"Portfolio risk decomposition calculated: {portfolio_vol:.2%} volatility")
        
        return decomposition
    
    def calculate_scenario_risk_metrics(self, 
                                      scenario_returns: Dict[str, pd.Series]) -> Dict[str, RiskMetrics]:
        """
        Calculate risk metrics for multiple scenarios.
        
        Args:
            scenario_returns: Dictionary mapping scenario names to return series
            
        Returns:
            Dictionary mapping scenario names to RiskMetrics objects
        """
        scenario_metrics = {}
        
        for scenario_name, returns in scenario_returns.items():
            if len(returns) > 10:  # Minimum data requirement
                metrics = self.calculate_comprehensive_metrics(returns, symbol=scenario_name)
                scenario_metrics[scenario_name] = metrics
        
        return scenario_metrics
    
    def generate_risk_report(self, metrics: RiskMetrics) -> str:
        """
        Generate a formatted text report of risk metrics.
        
        Args:
            metrics: RiskMetrics object to report on
            
        Returns:
            Formatted string report
        """
        report = f"""
RISK METRICS REPORT - {metrics.symbol}
{'='*50}
Calculation Date: {metrics.calculation_date}
Data Period: {metrics.data_period}
Observations: {metrics.observations:,}

VOLATILITY MEASURES
{'─'*20}
Annualized Volatility: {metrics.volatility:.2%}
Downside Volatility: {metrics.downside_volatility:.2%}

VALUE AT RISK
{'─'*15}
95% VaR (Historical): {metrics.var_95:.2%}
99% VaR (Historical): {metrics.var_99:.2%}
95% VaR (Parametric): {metrics.var_parametric_95:.2%}
99% VaR (Parametric): {metrics.var_parametric_99:.2%}

EXPECTED SHORTFALL
{'─'*20}
95% Expected Shortfall: {metrics.expected_shortfall_95:.2%}
99% Expected Shortfall: {metrics.expected_shortfall_99:.2%}

DRAWDOWN ANALYSIS
{'─'*20}
Maximum Drawdown: {metrics.maximum_drawdown:.2%}
Average Drawdown: {metrics.average_drawdown:.2%}
Max Drawdown Duration: {metrics.drawdown_duration} days
Average Recovery Time: {metrics.recovery_time or 'N/A'} days

RISK-ADJUSTED PERFORMANCE
{'─'*30}
Sharpe Ratio: {metrics.sharpe_ratio:.3f}
Sortino Ratio: {metrics.sortino_ratio:.3f}
Calmar Ratio: {metrics.calmar_ratio:.3f}

TAIL RISK MEASURES
{'─'*20}
Skewness: {metrics.skewness:.3f}
Kurtosis: {metrics.kurtosis:.3f}
Tail Ratio: {metrics.tail_ratio:.2f}

VAR BACKTESTING
{'─'*20}
95% VaR Hit Rate: {metrics.hit_rate_95:.2%} (Expected: 5.00%)
99% VaR Hit Rate: {metrics.hit_rate_99:.2%} (Expected: 1.00%)
VaR Ratio (99%/95%): {metrics.var_ratio:.2f}
"""
        
        if metrics.beta is not None:
            report += f"""
MARKET RISK MEASURES
{'─'*20}
Beta: {metrics.beta:.3f}
Correlation to Market: {metrics.correlation_to_market:.3f}
"""
        
        return report