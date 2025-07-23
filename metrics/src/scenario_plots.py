"""
Scenario Analysis Plotting Module

Comprehensive visualization tools for scenario analysis results.
Creates publication-quality charts and interactive visualizations for risk analysis.

This module provides a complete suite of plotting capabilities including:
- Historical scenario overlay charts
- Monte Carlo simulation distribution plots  
- Drawdown analysis and underwater curves
- Risk metric comparison charts
- Correlation heatmaps and network plots
- Interactive dashboards with plotly
- Portfolio performance visualization

All plots are optimized for both static export and interactive analysis,
with consistent styling and professional formatting.

Author: OHLCV Data Pipeline - Scenario Analysis Module
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import logging
from datetime import datetime, timedelta

# Import our modules
from scenario_definitions import MarketScenario
from risk_metrics import RiskMetrics

class ScenarioPlotter:
    """
    Comprehensive plotting class for scenario analysis visualization.
    
    Provides both static (matplotlib/seaborn) and interactive (plotly) 
    visualization capabilities for all aspects of scenario analysis.
    """
    
    def __init__(self, 
                 output_dir: str = "output/plots",
                 style: str = "darkgrid",
                 figsize: Tuple[int, int] = (12, 8),
                 dpi: int = 300,
                 plot_backend: str = "both"):
        """
        Initialize the scenario plotter.
        
        Args:
            output_dir: Directory to save plot files
            style: Seaborn style theme
            figsize: Default figure size for matplotlib plots
            dpi: DPI for saved figures
            plot_backend: "matplotlib", "plotly", or "both"
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.figsize = figsize
        self.dpi = dpi
        self.plot_backend = plot_backend
        
        # Set up plotting styles
        plt.style.use('default')
        sns.set_style(style)
        sns.set_palette("husl")
        
        # Color palettes for different chart types
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72', 
            'accent': '#F18F01',
            'success': '#C73E1D',
            'warning': '#F4A261',
            'danger': '#E63946',
            'info': '#264653',
            'light': '#F8F9FA',
            'dark': '#212529'
        }
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"ScenarioPlotter initialized, saving plots to: {self.output_dir}")
        self.logger.info(f"Plot backend: {self.plot_backend}")
    
    def plot_risk_metrics_matplotlib(self, 
                                   risk_metrics: Dict[str, RiskMetrics],
                                   save_plot: bool = True) -> plt.Figure:
        """
        Create matplotlib version of risk metrics comparison.
        
        Args:
            risk_metrics: Dictionary mapping names to RiskMetrics objects
            save_plot: Whether to save the plot
            
        Returns:
            Matplotlib figure object
        """
        if not risk_metrics:
            raise ValueError("No risk metrics provided")
        
        self.logger.info(f"Creating matplotlib risk metrics comparison for {len(risk_metrics)} items")
        
        # Extract data for plotting
        symbols = list(risk_metrics.keys())
        
        # Create data arrays
        volatilities = [risk_metrics[s].volatility * 100 for s in symbols]
        max_drawdowns = [abs(risk_metrics[s].maximum_drawdown) * 100 for s in symbols]
        sharpe_ratios = [risk_metrics[s].sharpe_ratio for s in symbols]
        var_95 = [abs(risk_metrics[s].var_95) * 100 for s in symbols]
        var_99 = [abs(risk_metrics[s].var_99) * 100 for s in symbols]
        
        # Create subplot figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Risk Metrics Comparison Across Assets', fontsize=16, fontweight='bold')
        
        # 1. Volatility
        axes[0, 0].bar(symbols, volatilities, color=self.colors['primary'], alpha=0.7)
        axes[0, 0].set_title('Annualized Volatility (%)')
        axes[0, 0].set_ylabel('Volatility (%)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Maximum Drawdown
        axes[0, 1].bar(symbols, max_drawdowns, color=self.colors['danger'], alpha=0.7)
        axes[0, 1].set_title('Maximum Drawdown (%)')
        axes[0, 1].set_ylabel('Max Drawdown (%)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Sharpe Ratio
        axes[0, 2].bar(symbols, sharpe_ratios, color=self.colors['success'], alpha=0.7)
        axes[0, 2].set_title('Sharpe Ratio')
        axes[0, 2].set_ylabel('Sharpe Ratio')
        axes[0, 2].tick_params(axis='x', rotation=45)
        axes[0, 2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # 4. 95% VaR
        axes[1, 0].bar(symbols, var_95, color=self.colors['warning'], alpha=0.7)
        axes[1, 0].set_title('95% Value at Risk (%)')
        axes[1, 0].set_ylabel('95% VaR (%)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. 99% VaR
        axes[1, 1].bar(symbols, var_99, color=self.colors['accent'], alpha=0.7)
        axes[1, 1].set_title('99% Value at Risk (%)')
        axes[1, 1].set_ylabel('99% VaR (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 6. Risk vs Return scatter
        returns = [risk_metrics[s].sharpe_ratio * risk_metrics[s].volatility * 100 for s in symbols]
        axes[1, 2].scatter(volatilities, returns, s=100, alpha=0.7, color=self.colors['secondary'])
        for i, symbol in enumerate(symbols):
            axes[1, 2].annotate(symbol, (volatilities[i], returns[i]), 
                              xytext=(5, 5), textcoords='offset points')
        axes[1, 2].set_xlabel('Volatility (%)')
        axes[1, 2].set_ylabel('Risk-Adjusted Return (%)')
        axes[1, 2].set_title('Risk vs Return')
        
        plt.tight_layout()
        
        if save_plot:
            filename = self.output_dir / "risk_metrics_comparison.png"
            fig.savefig(filename, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Matplotlib risk metrics comparison saved to: {filename}")
        
        return fig
    
    def plot_drawdown_matplotlib(self, 
                               returns_data: pd.DataFrame,
                               symbols: Optional[List[str]] = None,
                               save_plot: bool = True) -> plt.Figure:
        """
        Create matplotlib version of drawdown analysis.
        
        Args:
            returns_data: DataFrame with return data (dates x symbols)
            symbols: Symbols to analyze (None for all)
            save_plot: Whether to save the plot
            
        Returns:
            Matplotlib figure object
        """
        if symbols is None:
            symbols = list(returns_data.columns)
        
        self.logger.info(f"Creating matplotlib drawdown analysis for {len(symbols)} symbols")
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle('Drawdown Analysis - Performance and Risk Over Time', fontsize=16, fontweight='bold')
        
        # Color palette
        colors = plt.cm.Set1(np.linspace(0, 1, len(symbols)))
        
        for i, symbol in enumerate(symbols):
            if symbol in returns_data.columns:
                returns = returns_data[symbol].dropna()
                
                # Calculate cumulative returns
                cumulative = (1 + returns).cumprod()
                
                # Calculate drawdown
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                
                # Plot cumulative performance
                ax1.plot(cumulative.index, cumulative, 
                        label=f"{symbol}", color=colors[i], linewidth=2)
                
                # Plot drawdown (underwater chart)
                ax2.fill_between(drawdown.index, drawdown * 100, 0, 
                               alpha=0.6, color=colors[i], label=f"{symbol}")
        
        # Format top plot
        ax1.set_ylabel('Cumulative Return')
        ax1.set_title('Cumulative Performance Over Time')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        
        # Format bottom plot
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_title('Drawdown (Underwater) Chart')
        ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.8)
        
        plt.tight_layout()
        
        if save_plot:
            filename = self.output_dir / "drawdown_analysis.png"
            fig.savefig(filename, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Matplotlib drawdown analysis saved to: {filename}")
        
        return fig
    
    def plot_correlation_heatmap_matplotlib(self, 
                                          returns_data: pd.DataFrame,
                                          title: str = "Asset Correlation Matrix",
                                          save_plot: bool = True) -> plt.Figure:
        """
        Create matplotlib version of correlation heatmap.
        
        Args:
            returns_data: DataFrame with return data (dates x symbols)
            title: Chart title
            save_plot: Whether to save the plot
            
        Returns:
            Matplotlib figure object
        """
        correlation_matrix = returns_data.corr()
        
        self.logger.info(f"Creating matplotlib correlation heatmap for {len(correlation_matrix)} assets")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(correlation_matrix, 
                   annot=True, 
                   cmap='RdBu_r', 
                   center=0, 
                   square=True,
                   fmt='.3f',
                   cbar_kws={'label': 'Correlation'},
                   ax=ax)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_plot:
            filename = self.output_dir / "correlation_heatmap.png"
            fig.savefig(filename, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Matplotlib correlation heatmap saved to: {filename}")
        
        return fig
    
    def plot_monte_carlo_distributions_matplotlib(self, 
                                                simulation_results: Dict[str, Any],
                                                symbols: Optional[List[str]] = None,
                                                save_plot: bool = True) -> plt.Figure:
        """
        Create matplotlib version of Monte Carlo distributions.
        
        Args:
            simulation_results: Results from Monte Carlo simulation
            symbols: Symbols to plot (None for all)
            save_plot: Whether to save the plot
            
        Returns:
            Matplotlib figure object
        """
        if 'statistics' not in simulation_results:
            raise ValueError("Simulation results must contain 'statistics' key")
        
        stats = simulation_results['statistics']
        
        if symbols is None:
            symbols = [s for s in stats.keys() if s != 'portfolio' and s != 'correlations']
        
        # Add portfolio if not in symbols
        symbols_to_plot = symbols + ['portfolio'] if 'portfolio' in stats else symbols
        
        self.logger.info(f"Creating matplotlib Monte Carlo distributions for {len(symbols_to_plot)} assets")
        
        # Create subplots
        n_plots = len(symbols_to_plot)
        cols = min(3, n_plots)
        rows = (n_plots + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        fig.suptitle('Monte Carlo Simulation - Return Distributions', fontsize=16, fontweight='bold')
        
        # Handle single plot case
        if n_plots == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if n_plots > 1 else [axes]
        else:
            axes = axes.flatten()
        
        for i, symbol in enumerate(symbols_to_plot):
            ax = axes[i] if n_plots > 1 else axes[0]
            
            if symbol in stats:
                symbol_stats = stats[symbol]
                percentiles = symbol_stats['percentiles']
                
                # Create histogram data (approximate from percentiles)
                x_values = list(percentiles.values())
                
                # Plot histogram
                ax.hist(x_values, bins=20, alpha=0.7, color=self.colors['primary'], 
                       edgecolor='black', linewidth=0.5)
                
                # Add VaR lines
                var_5 = symbol_stats['var_estimates'].get('5%', x_values[0])
                var_1 = symbol_stats['var_estimates'].get('1%', x_values[0])
                
                ax.axvline(var_5, color='red', linestyle='--', linewidth=2, label='5% VaR')
                ax.axvline(var_1, color='darkred', linestyle='--', linewidth=2, label='1% VaR')
                
                ax.set_title(f"{symbol} Return Distribution")
                ax.set_xlabel('Return')
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        
        if save_plot:
            filename = self.output_dir / "monte_carlo_distributions.png"
            fig.savefig(filename, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Matplotlib Monte Carlo distributions saved to: {filename}")
        
        return fig
    
    def plot_historical_scenario_overlay(self, 
                                       price_data: pd.DataFrame,
                                       scenarios: List[MarketScenario],
                                       symbols: Optional[List[str]] = None,
                                       normalize: bool = True,
                                       save_plot: bool = True) -> go.Figure:
        """
        Create overlay plot showing historical price performance during specific scenarios.
        
        Args:
            price_data: DataFrame with price data (dates x symbols)
            scenarios: List of MarketScenario objects to highlight
            symbols: Symbols to plot (None for all)
            normalize: Whether to normalize prices to 100 at scenario start
            save_plot: Whether to save the plot
            
        Returns:
            Plotly figure object
        """
        if symbols is None:
            symbols = list(price_data.columns)
        
        self.logger.info(f"Creating historical scenario overlay for {len(scenarios)} scenarios")
        
        # Create subplot with secondary y-axis for annotations
        fig = make_subplots(
            rows=1, cols=1,
            subplot_titles=["Historical Performance During Market Scenarios"],
            specs=[[{"secondary_y": True}]]
        )
        
        # Plot price data for each symbol
        for i, symbol in enumerate(symbols):
            if symbol in price_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=price_data.index,
                        y=price_data[symbol],
                        name=symbol,
                        line=dict(width=2),
                        opacity=0.8
                    ),
                    secondary_y=False
                )
        
        # Add scenario period highlighting
        colors_cycle = ['rgba(255,0,0,0.2)', 'rgba(0,255,0,0.2)', 'rgba(0,0,255,0.2)', 
                       'rgba(255,255,0,0.2)', 'rgba(255,0,255,0.2)', 'rgba(0,255,255,0.2)']
        
        for i, scenario in enumerate(scenarios):
            color = colors_cycle[i % len(colors_cycle)]
            
            # Add shaded region for scenario period
            fig.add_vrect(
                x0=scenario.start_date,
                x1=scenario.end_date,
                fillcolor=color,
                opacity=0.3,
                layer="below",
                line_width=0,
            )
            
            # Add scenario label
            mid_date = scenario.start_date + (scenario.end_date - scenario.start_date) / 2
            
            fig.add_annotation(
                x=mid_date,
                y=0.95,
                xref="x",
                yref="paper",
                text=f"{scenario.name}<br>({scenario.start_date} to {scenario.end_date})",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=color.replace('0.2', '1.0'),
                font=dict(size=10),
                bgcolor="white",
                bordercolor=color.replace('0.2', '1.0'),
                borderwidth=1
            )
        
        # Update layout
        fig.update_layout(
            title="Historical Performance During Major Market Scenarios",
            xaxis_title="Date",
            yaxis_title="Price Level",
            hovermode='x unified',
            width=1200,
            height=700,
            showlegend=True,
            legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
            template="plotly_white"
        )
        
        if save_plot:
            filename = self.output_dir / "historical_scenario_overlay.html"
            fig.write_html(str(filename))
            self.logger.info(f"Historical scenario overlay saved to: {filename}")
        
        return fig
    
    def plot_monte_carlo_distributions(self, 
                                     simulation_results: Dict[str, Any],
                                     symbols: Optional[List[str]] = None,
                                     save_plot: bool = True) -> go.Figure:
        """
        Create distribution plots for Monte Carlo simulation results.
        
        Args:
            simulation_results: Results from Monte Carlo simulation
            symbols: Symbols to plot (None for all)
            save_plot: Whether to save the plot
            
        Returns:
            Plotly figure object
        """
        if 'statistics' not in simulation_results:
            raise ValueError("Simulation results must contain 'statistics' key")
        
        stats = simulation_results['statistics']
        
        if symbols is None:
            symbols = [s for s in stats.keys() if s != 'portfolio' and s != 'correlations']
        
        # Add portfolio if not in symbols
        symbols_to_plot = symbols + ['portfolio'] if 'portfolio' in stats else symbols
        
        self.logger.info(f"Creating Monte Carlo distribution plots for {len(symbols_to_plot)} assets")
        
        # Create subplots
        n_plots = len(symbols_to_plot)
        cols = min(3, n_plots)
        rows = (n_plots + cols - 1) // cols
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[f"{symbol} Return Distribution" for symbol in symbols_to_plot],
            vertical_spacing=0.12,
            horizontal_spacing=0.08
        )
        
        for i, symbol in enumerate(symbols_to_plot):
            row = i // cols + 1
            col = i % cols + 1
            
            if symbol in stats:
                symbol_stats = stats[symbol]
                percentiles = symbol_stats['percentiles']
                
                # Create histogram data (approximate from percentiles)
                x_values = list(percentiles.values())
                
                # Add histogram trace
                fig.add_trace(
                    go.Histogram(
                        x=x_values,
                        name=f"{symbol} Returns",
                        nbinsx=30,
                        opacity=0.7,
                        showlegend=False
                    ),
                    row=row, col=col
                )
                
                # Add VaR lines
                var_5 = symbol_stats['var_estimates'].get('5%', x_values[0])
                var_1 = symbol_stats['var_estimates'].get('1%', x_values[0])
                
                # Add vertical lines for VaR
                fig.add_vline(
                    x=var_5, line_dash="dash", line_color="red",
                    annotation_text="5% VaR", annotation_position="top",
                    row=row, col=col
                )
                
                fig.add_vline(
                    x=var_1, line_dash="dash", line_color="darkred", 
                    annotation_text="1% VaR", annotation_position="top",
                    row=row, col=col
                )
        
        # Update layout
        fig.update_layout(
            title="Monte Carlo Simulation - Return Distributions",
            width=1400,
            height=300 * rows,
            template="plotly_white",
            showlegend=False
        )
        
        if save_plot:
            filename = self.output_dir / "monte_carlo_distributions.html"
            fig.write_html(str(filename))
            self.logger.info(f"Monte Carlo distributions saved to: {filename}")
        
        return fig
    
    def plot_drawdown_analysis(self, 
                             returns_data: pd.DataFrame,
                             symbols: Optional[List[str]] = None,
                             save_plot: bool = True) -> go.Figure:
        """
        Create comprehensive drawdown analysis charts.
        
        Args:
            returns_data: DataFrame with return data (dates x symbols)
            symbols: Symbols to analyze (None for all)
            save_plot: Whether to save the plot
            
        Returns:
            Plotly figure object
        """
        if symbols is None:
            symbols = list(returns_data.columns)
        
        self.logger.info(f"Creating drawdown analysis for {len(symbols)} symbols")
        
        # Create subplots: equity curves and underwater plots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=["Cumulative Performance", "Drawdown (Underwater) Chart"],
            vertical_spacing=0.1,
            row_heights=[0.6, 0.4]
        )
        
        for symbol in symbols:
            if symbol in returns_data.columns:
                returns = returns_data[symbol].dropna()
                
                # Calculate cumulative returns
                cumulative = (1 + returns).cumprod()
                
                # Calculate drawdown
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                
                # Add cumulative performance line
                fig.add_trace(
                    go.Scatter(
                        x=cumulative.index,
                        y=cumulative,
                        name=f"{symbol} Cumulative",
                        line=dict(width=2),
                        opacity=0.8
                    ),
                    row=1, col=1
                )
                
                # Add drawdown (underwater) chart
                fig.add_trace(
                    go.Scatter(
                        x=drawdown.index,
                        y=drawdown * 100,  # Convert to percentage
                        name=f"{symbol} Drawdown",
                        fill='tonexty' if symbol == symbols[0] else 'tozeroy',
                        line=dict(width=1),
                        opacity=0.6
                    ),
                    row=2, col=1
                )
        
        # Update layout
        fig.update_layout(
            title="Drawdown Analysis - Performance and Risk Over Time",
            width=1200,
            height=800,
            hovermode='x unified',
            template="plotly_white",
            legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)")
        )
        
        # Update y-axes labels
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        
        if save_plot:
            filename = self.output_dir / "drawdown_analysis.html"
            fig.write_html(str(filename))
            self.logger.info(f"Drawdown analysis saved to: {filename}")
        
        return fig
    
    def plot_correlation_heatmap(self, 
                               returns_data: pd.DataFrame,
                               title: str = "Asset Correlation Matrix",
                               save_plot: bool = True) -> go.Figure:
        """
        Create correlation heatmap with interactive features.
        
        Args:
            returns_data: DataFrame with return data (dates x symbols)
            title: Chart title
            save_plot: Whether to save the plot
            
        Returns:
            Plotly figure object
        """
        correlation_matrix = returns_data.corr()
        
        self.logger.info(f"Creating correlation heatmap for {len(correlation_matrix)} assets")
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.round(3).values,
            texttemplate="%{text}",
            textfont={"size": 12},
            hoverongaps=False,
            colorbar=dict(title="Correlation")
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            width=600,
            height=600,
            template="plotly_white"
        )
        
        if save_plot:
            filename = self.output_dir / "correlation_heatmap.html"
            fig.write_html(str(filename))
            self.logger.info(f"Correlation heatmap saved to: {filename}")
        
        return fig
    
    def plot_risk_metrics_comparison(self, 
                                   risk_metrics: Dict[str, RiskMetrics],
                                   save_plot: bool = True) -> go.Figure:
        """
        Create comparison chart of risk metrics across assets/scenarios.
        
        Args:
            risk_metrics: Dictionary mapping names to RiskMetrics objects
            save_plot: Whether to save the plot
            
        Returns:
            Plotly figure object
        """
        if not risk_metrics:
            raise ValueError("No risk metrics provided")
        
        self.logger.info(f"Creating risk metrics comparison for {len(risk_metrics)} items")
        
        # Extract data for plotting
        symbols = list(risk_metrics.keys())
        
        metrics_data = {
            'Volatility (%)': [risk_metrics[s].volatility * 100 for s in symbols],
            'Max Drawdown (%)': [abs(risk_metrics[s].maximum_drawdown) * 100 for s in symbols],
            'Sharpe Ratio': [risk_metrics[s].sharpe_ratio for s in symbols],
            '95% VaR (%)': [abs(risk_metrics[s].var_95) * 100 for s in symbols],
            '99% VaR (%)': [abs(risk_metrics[s].var_99) * 100 for s in symbols]
        }
        
        # Create subplots for different metrics
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=list(metrics_data.keys()),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        positions = [(1,1), (1,2), (1,3), (2,1), (2,2)]
        
        for i, (metric_name, values) in enumerate(metrics_data.items()):
            if i < len(positions):
                row, col = positions[i]
                
                fig.add_trace(
                    go.Bar(
                        x=symbols,
                        y=values,
                        name=metric_name,
                        showlegend=False,
                        marker_color=px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)]
                    ),
                    row=row, col=col
                )
        
        # Update layout
        fig.update_layout(
            title="Risk Metrics Comparison Across Assets",
            width=1400,
            height=800,
            template="plotly_white"
        )
        
        if save_plot:
            filename = self.output_dir / "risk_metrics_comparison.html"
            fig.write_html(str(filename))
            self.logger.info(f"Risk metrics comparison saved to: {filename}")
        
        return fig
    
    def plot_efficient_frontier(self, 
                              returns_data: pd.DataFrame,
                              num_portfolios: int = 10000,
                              save_plot: bool = True) -> go.Figure:
        """
        Create efficient frontier plot for portfolio optimization.
        
        Args:
            returns_data: DataFrame with return data (dates x symbols)
            num_portfolios: Number of random portfolios to generate
            save_plot: Whether to save the plot
            
        Returns:
            Plotly figure object
        """
        self.logger.info(f"Creating efficient frontier with {num_portfolios} random portfolios")
        
        # Calculate expected returns and covariance matrix
        expected_returns = returns_data.mean() * 252  # Annualized
        cov_matrix = returns_data.cov() * 252  # Annualized covariance
        
        # Generate random portfolios
        n_assets = len(returns_data.columns)
        results = np.zeros((3, num_portfolios))
        
        np.random.seed(42)  # For reproducibility
        
        for i in range(num_portfolios):
            # Generate random weights
            weights = np.random.random(n_assets)
            weights /= np.sum(weights)  # Normalize to sum to 1
            
            # Calculate portfolio return and volatility
            portfolio_return = np.sum(weights * expected_returns)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            # Calculate Sharpe ratio
            sharpe_ratio = (portfolio_return - 0.02) / portfolio_volatility  # Assuming 2% risk-free rate
            
            results[0, i] = portfolio_return
            results[1, i] = portfolio_volatility
            results[2, i] = sharpe_ratio
        
        # Create scatter plot
        fig = go.Figure()
        
        # Add random portfolios
        fig.add_trace(
            go.Scatter(
                x=results[1] * 100,  # Volatility in %
                y=results[0] * 100,  # Return in %
                mode='markers',
                marker=dict(
                    size=3,
                    color=results[2],  # Color by Sharpe ratio
                    colorscale='Viridis',
                    colorbar=dict(title="Sharpe Ratio"),
                    opacity=0.6
                ),
                name='Random Portfolios',
                text=[f'Sharpe: {sr:.3f}' for sr in results[2]],
                hovertemplate='<b>Portfolio</b><br>' +
                             'Return: %{y:.2f}%<br>' +
                             'Volatility: %{x:.2f}%<br>' +
                             '%{text}<extra></extra>'
            )
        )
        
        # Add individual assets
        individual_returns = expected_returns * 100
        individual_vols = np.sqrt(np.diag(cov_matrix)) * 100
        
        fig.add_trace(
            go.Scatter(
                x=individual_vols,
                y=individual_returns,
                mode='markers+text',
                marker=dict(size=10, color='red', symbol='star'),
                text=returns_data.columns,
                textposition='top center',
                name='Individual Assets',
                hovertemplate='<b>%{text}</b><br>' +
                             'Return: %{y:.2f}%<br>' +
                             'Volatility: %{x:.2f}%<extra></extra>'
            )
        )
        
        # Update layout
        fig.update_layout(
            title="Efficient Frontier - Risk vs Return",
            xaxis_title="Volatility (%)",
            yaxis_title="Expected Return (%)",
            width=1000,
            height=700,
            template="plotly_white",
            hovermode='closest'
        )
        
        if save_plot:
            filename = self.output_dir / "efficient_frontier.html"
            fig.write_html(str(filename))
            self.logger.info(f"Efficient frontier saved to: {filename}")
        
        return fig
    
    def create_scenario_dashboard(self, 
                                scenario_results: Dict[str, Any],
                                monte_carlo_results: Dict[str, Any],
                                risk_metrics: Dict[str, RiskMetrics],
                                save_plot: bool = True) -> go.Figure:
        """
        Create comprehensive dashboard combining multiple analysis views.
        
        Args:
            scenario_results: Historical scenario analysis results
            monte_carlo_results: Monte Carlo simulation results
            risk_metrics: Risk metrics for all assets
            save_plot: Whether to save the dashboard
            
        Returns:
            Plotly figure object
        """
        self.logger.info("Creating comprehensive scenario analysis dashboard")
        
        # Create dashboard with multiple subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                "Risk Metrics Overview", "Monte Carlo VaR Distribution",
                "Drawdown Comparison", "Correlation Matrix",
                "Scenario Performance", "Portfolio Composition"
            ],
            specs=[[{"type": "bar"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "heatmap"}], 
                   [{"type": "scatter"}, {"type": "pie"}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # 1. Risk Metrics Overview (Bar chart)
        symbols = list(risk_metrics.keys())
        sharpe_ratios = [risk_metrics[s].sharpe_ratio for s in symbols]
        
        fig.add_trace(
            go.Bar(x=symbols, y=sharpe_ratios, name="Sharpe Ratios", showlegend=False),
            row=1, col=1
        )
        
        # 2. Monte Carlo VaR Distribution (Histogram)
        if 'statistics' in monte_carlo_results and 'portfolio' in monte_carlo_results['statistics']:
            portfolio_stats = monte_carlo_results['statistics']['portfolio']
            var_values = list(portfolio_stats['percentiles'].values())
            
            fig.add_trace(
                go.Histogram(x=var_values, nbinsx=30, name="Return Distribution", showlegend=False),
                row=1, col=2
            )
        
        # 3. Drawdown Comparison (Scatter)
        max_drawdowns = [abs(risk_metrics[s].maximum_drawdown) * 100 for s in symbols]
        volatilities = [risk_metrics[s].volatility * 100 for s in symbols]
        
        fig.add_trace(
            go.Scatter(
                x=volatilities, y=max_drawdowns, 
                mode='markers+text', text=symbols,
                name="Risk vs Drawdown", showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. Simple correlation display (if we have returns data)
        # This is a placeholder - in real implementation you'd pass correlation matrix
        
        # 5. Equal-weighted portfolio pie chart
        n_assets = len(symbols)
        equal_weights = [100/n_assets] * n_assets
        
        fig.add_trace(
            go.Pie(labels=symbols, values=equal_weights, name="Portfolio Weights"),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Comprehensive Scenario Analysis Dashboard",
            width=1600,
            height=1200,
            template="plotly_white",
            showlegend=True
        )
        
        if save_plot:
            filename = self.output_dir / "scenario_dashboard.html"
            fig.write_html(str(filename))
            self.logger.info(f"Scenario dashboard saved to: {filename}")
        
        return fig
    
    def save_all_plots_summary(self, 
                              figures: Dict[str, go.Figure],
                              title: str = "Scenario Analysis Report") -> str:
        """
        Create an HTML summary page with all plots.
        
        Args:
            figures: Dictionary mapping plot names to Plotly figures
            title: Main title for the report
            
        Returns:
            Path to saved HTML file
        """
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #2E86AB; text-align: center; }}
        h2 {{ color: #A23B72; border-bottom: 2px solid #A23B72; padding-bottom: 5px; }}
        .plot-container {{ margin: 20px 0; }}
        .summary {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .timestamp {{ text-align: right; color: #6c757d; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="timestamp">Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    
    <div class="summary">
        <h3>Analysis Summary</h3>
        <p>This report contains comprehensive scenario analysis results including:</p>
        <ul>
            <li>Historical scenario performance analysis</li>
            <li>Monte Carlo simulation results</li>
            <li>Risk metrics comparison</li>
            <li>Drawdown and correlation analysis</li>
            <li>Portfolio optimization insights</li>
        </ul>
    </div>
"""
        
        # Add each figure to the HTML
        for plot_name, fig in figures.items():
            html_content += f"""
    <div class="plot-container">
        <h2>{plot_name.replace('_', ' ').title()}</h2>
        {fig.to_html(include_plotlyjs='inline', div_id=f"{plot_name}_div")}
    </div>
"""
        
        html_content += """
</body>
</html>
"""
        
        # Save HTML file
        filename = self.output_dir / f"scenario_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"Complete analysis report saved to: {filename}")
        return str(filename)