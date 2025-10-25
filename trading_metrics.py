#!/usr/bin/env python3
"""
Trading-Aligned Evaluation Metrics
==================================

Comprehensive evaluation metrics for trading strategies:
- Information Ratio, Sharpe Ratio, Sortino Ratio
- Maximum Drawdown, Calmar Ratio
- Hit Rate, Profit Factor, Average Win/Loss
- Turnover, Transaction Costs
- Risk-adjusted returns
- Portfolio-level metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from scipy import stats
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class TradingMetrics:
    """Comprehensive trading performance metrics"""
    
    def __init__(self, 
                 risk_free_rate: float = 0.02,
                 trading_days: int = 252,
                 transaction_cost: float = 0.001):
        """
        Initialize trading metrics calculator
        
        Args:
            risk_free_rate: Annual risk-free rate
            trading_days: Number of trading days per year
            transaction_cost: Transaction cost as fraction of trade value
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days
        self.transaction_cost = transaction_cost
    
    def calculate_all_metrics(self, 
                            returns: np.ndarray,
                            benchmark_returns: Optional[np.ndarray] = None,
                            positions: Optional[np.ndarray] = None,
                            prices: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate all trading metrics"""
        
        metrics = {}
        
        # Basic return metrics
        metrics.update(self._calculate_return_metrics(returns))
        
        # Risk metrics
        metrics.update(self._calculate_risk_metrics(returns))
        
        # Risk-adjusted metrics
        metrics.update(self._calculate_risk_adjusted_metrics(returns, benchmark_returns))
        
        # Drawdown metrics
        metrics.update(self._calculate_drawdown_metrics(returns))
        
        # Trading metrics
        if positions is not None:
            metrics.update(self._calculate_trading_metrics(returns, positions, prices))
        
        # Benchmark comparison
        if benchmark_returns is not None:
            metrics.update(self._calculate_benchmark_metrics(returns, benchmark_returns))
        
        return metrics
    
    def _calculate_return_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate basic return metrics"""
        
        total_return = np.prod(1 + returns) - 1
        annualized_return = (1 + total_return) ** (self.trading_days / len(returns)) - 1
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'mean_return': np.mean(returns),
            'median_return': np.median(returns)
        }
    
    def _calculate_risk_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate risk metrics"""
        
        volatility = np.std(returns, ddof=1)
        annualized_volatility = volatility * np.sqrt(self.trading_days)
        
        # Downside deviation
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns, ddof=1) if len(downside_returns) > 0 else 0
        annualized_downside_deviation = downside_deviation * np.sqrt(self.trading_days)
        
        # Value at Risk (VaR)
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # Conditional Value at Risk (CVaR)
        cvar_95 = np.mean(returns[returns <= var_95]) if len(returns[returns <= var_95]) > 0 else 0
        cvar_99 = np.mean(returns[returns <= var_99]) if len(returns[returns <= var_99]) > 0 else 0
        
        return {
            'volatility': volatility,
            'annualized_volatility': annualized_volatility,
            'downside_deviation': downside_deviation,
            'annualized_downside_deviation': annualized_downside_deviation,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99
        }
    
    def _calculate_risk_adjusted_metrics(self, 
                                       returns: np.ndarray, 
                                       benchmark_returns: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate risk-adjusted metrics"""
        
        # Sharpe Ratio
        excess_returns = returns - self.risk_free_rate / self.trading_days
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns, ddof=1) if np.std(excess_returns, ddof=1) > 0 else 0
        annualized_sharpe = sharpe_ratio * np.sqrt(self.trading_days)
        
        # Sortino Ratio
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns, ddof=1) if len(downside_returns) > 0 else 0
        sortino_ratio = np.mean(excess_returns) / downside_deviation if downside_deviation > 0 else 0
        annualized_sortino = sortino_ratio * np.sqrt(self.trading_days)
        
        # Information Ratio
        if benchmark_returns is not None:
            active_returns = returns - benchmark_returns
            tracking_error = np.std(active_returns, ddof=1)
            information_ratio = np.mean(active_returns) / tracking_error if tracking_error > 0 else 0
        else:
            information_ratio = 0
        
        # Calmar Ratio
        total_return = np.prod(1 + returns) - 1
        annualized_return = (1 + total_return) ** (self.trading_days / len(returns)) - 1
        max_drawdown = self._calculate_max_drawdown(returns)
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Omega Ratio
        omega_ratio = self._calculate_omega_ratio(returns)
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'annualized_sharpe': annualized_sharpe,
            'sortino_ratio': sortino_ratio,
            'annualized_sortino': annualized_sortino,
            'information_ratio': information_ratio,
            'calmar_ratio': calmar_ratio,
            'omega_ratio': omega_ratio
        }
    
    def _calculate_drawdown_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate drawdown metrics"""
        
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        
        max_drawdown = np.min(drawdowns)
        max_drawdown_duration = self._calculate_max_drawdown_duration(drawdowns)
        average_drawdown = np.mean(drawdowns[drawdowns < 0]) if len(drawdowns[drawdowns < 0]) > 0 else 0
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': max_drawdown_duration,
            'average_drawdown': average_drawdown
        }
    
    def _calculate_trading_metrics(self, 
                                 returns: np.ndarray, 
                                 positions: np.ndarray,
                                 prices: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate trading-specific metrics"""
        
        # Position changes (trades)
        position_changes = np.diff(positions, prepend=positions[0])
        trades = np.abs(position_changes)
        
        # Hit rate
        winning_trades = returns[returns > 0]
        losing_trades = returns[returns < 0]
        hit_rate = len(winning_trades) / len(returns) if len(returns) > 0 else 0
        
        # Profit factor
        total_profit = np.sum(winning_trades) if len(winning_trades) > 0 else 0
        total_loss = abs(np.sum(losing_trades)) if len(losing_trades) > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Average win/loss
        average_win = np.mean(winning_trades) if len(winning_trades) > 0 else 0
        average_loss = np.mean(losing_trades) if len(losing_trades) > 0 else 0
        
        # Turnover
        turnover = np.mean(trades) if len(trades) > 0 else 0
        
        # Transaction costs
        total_transaction_costs = np.sum(trades) * self.transaction_cost
        
        return {
            'hit_rate': hit_rate,
            'profit_factor': profit_factor,
            'average_win': average_win,
            'average_loss': average_loss,
            'turnover': turnover,
            'total_transaction_costs': total_transaction_costs,
            'net_return_after_costs': np.sum(returns) - total_transaction_costs
        }
    
    def _calculate_benchmark_metrics(self, 
                                   returns: np.ndarray, 
                                   benchmark_returns: np.ndarray) -> Dict[str, float]:
        """Calculate benchmark comparison metrics"""
        
        # Alpha and Beta
        alpha, beta = self._calculate_alpha_beta(returns, benchmark_returns)
        
        # Tracking error
        active_returns = returns - benchmark_returns
        tracking_error = np.std(active_returns, ddof=1)
        
        # Information ratio
        information_ratio = np.mean(active_returns) / tracking_error if tracking_error > 0 else 0
        
        # Up/down capture ratios
        up_capture, down_capture = self._calculate_capture_ratios(returns, benchmark_returns)
        
        return {
            'alpha': alpha,
            'beta': beta,
            'tracking_error': tracking_error,
            'up_capture_ratio': up_capture,
            'down_capture_ratio': down_capture
        }
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        return np.min(drawdowns)
    
    def _calculate_max_drawdown_duration(self, drawdowns: np.ndarray) -> int:
        """Calculate maximum drawdown duration"""
        in_drawdown = drawdowns < 0
        if not np.any(in_drawdown):
            return 0
        
        # Find consecutive drawdown periods
        drawdown_periods = []
        current_period = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                    current_period = 0
        
        # Add final period if still in drawdown
        if current_period > 0:
            drawdown_periods.append(current_period)
        
        return max(drawdown_periods) if drawdown_periods else 0
    
    def _calculate_omega_ratio(self, returns: np.ndarray, threshold: float = 0.0) -> float:
        """Calculate Omega ratio"""
        excess_returns = returns - threshold
        gains = excess_returns[excess_returns > 0].sum()
        losses = abs(excess_returns[excess_returns < 0].sum())
        return gains / losses if losses > 0 else float('inf')
    
    def _calculate_alpha_beta(self, returns: np.ndarray, benchmark_returns: np.ndarray) -> Tuple[float, float]:
        """Calculate Alpha and Beta"""
        if len(returns) != len(benchmark_returns):
            return 0.0, 1.0
        
        # Calculate beta using covariance
        covariance = np.cov(returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns, ddof=1)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0
        
        # Calculate alpha
        alpha = np.mean(returns) - beta * np.mean(benchmark_returns)
        
        return alpha, beta
    
    def _calculate_capture_ratios(self, returns: np.ndarray, benchmark_returns: np.ndarray) -> Tuple[float, float]:
        """Calculate up and down capture ratios"""
        if len(returns) != len(benchmark_returns):
            return 1.0, 1.0
        
        # Up periods
        up_mask = benchmark_returns > 0
        if np.any(up_mask):
            up_capture = np.mean(returns[up_mask]) / np.mean(benchmark_returns[up_mask])
        else:
            up_capture = 1.0
        
        # Down periods
        down_mask = benchmark_returns < 0
        if np.any(down_mask):
            down_capture = np.mean(returns[down_mask]) / np.mean(benchmark_returns[down_mask])
        else:
            down_capture = 1.0
        
        return up_capture, down_capture


class PortfolioMetrics:
    """Portfolio-level evaluation metrics"""
    
    def __init__(self, 
                 risk_free_rate: float = 0.02,
                 trading_days: int = 252):
        """
        Initialize portfolio metrics calculator
        
        Args:
            risk_free_rate: Annual risk-free rate
            trading_days: Number of trading days per year
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days
    
    def calculate_portfolio_metrics(self, 
                                  portfolio_returns: np.ndarray,
                                  asset_returns: pd.DataFrame,
                                  weights: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Calculate portfolio-level metrics"""
        
        metrics = {}
        
        # Portfolio performance
        portfolio_calculator = TradingMetrics(self.risk_free_rate, self.trading_days)
        metrics.update(portfolio_calculator.calculate_all_metrics(portfolio_returns))
        
        # Diversification metrics
        metrics.update(self._calculate_diversification_metrics(asset_returns, weights))
        
        # Risk decomposition
        metrics.update(self._calculate_risk_decomposition(portfolio_returns, asset_returns, weights))
        
        return metrics
    
    def _calculate_diversification_metrics(self, 
                                         asset_returns: pd.DataFrame,
                                         weights: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """Calculate diversification metrics"""
        
        # Correlation matrix
        correlation_matrix = asset_returns.corr()
        avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
        
        # Effective number of assets
        if weights is not None:
            # Herfindahl index
            herfindahl_index = np.sum(weights.values ** 2)
            effective_number = 1 / herfindahl_index if herfindahl_index > 0 else len(weights.columns)
        else:
            effective_number = len(asset_returns.columns)
        
        # Concentration risk
        if weights is not None:
            max_weight = weights.max().max()
            concentration_risk = max_weight
        else:
            concentration_risk = 1.0 / len(asset_returns.columns)
        
        return {
            'avg_correlation': avg_correlation,
            'effective_number_of_assets': effective_number,
            'concentration_risk': concentration_risk
        }
    
    def _calculate_risk_decomposition(self, 
                                    portfolio_returns: np.ndarray,
                                    asset_returns: pd.DataFrame,
                                    weights: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Calculate risk decomposition"""
        
        if weights is None:
            # Equal weights
            weights = pd.DataFrame(
                np.ones((len(asset_returns), len(asset_returns.columns))) / len(asset_returns.columns),
                index=asset_returns.index,
                columns=asset_returns.columns
            )
        
        # Portfolio variance
        portfolio_variance = np.var(portfolio_returns, ddof=1)
        
        # Asset contributions to portfolio variance
        asset_contributions = {}
        for asset in asset_returns.columns:
            asset_weight = weights[asset].mean()
            asset_variance = np.var(asset_returns[asset], ddof=1)
            asset_contribution = (asset_weight ** 2) * asset_variance
            asset_contributions[asset] = asset_contribution
        
        # Risk contribution ratio
        total_asset_risk = sum(asset_contributions.values())
        risk_contribution_ratio = total_asset_risk / portfolio_variance if portfolio_variance > 0 else 1.0
        
        return {
            'portfolio_variance': portfolio_variance,
            'asset_risk_contributions': asset_contributions,
            'risk_contribution_ratio': risk_contribution_ratio
        }


class PerformanceAttribution:
    """Performance attribution analysis"""
    
    def __init__(self):
        self.attribution_results = {}
    
    def calculate_attribution(self, 
                            portfolio_returns: np.ndarray,
                            benchmark_returns: np.ndarray,
                            asset_returns: pd.DataFrame,
                            weights: pd.DataFrame,
                            benchmark_weights: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Calculate performance attribution"""
        
        # Total active return
        active_return = np.mean(portfolio_returns - benchmark_returns)
        
        # Asset allocation effect
        allocation_effect = self._calculate_allocation_effect(
            asset_returns, weights, benchmark_weights
        )
        
        # Security selection effect
        selection_effect = self._calculate_selection_effect(
            asset_returns, weights, benchmark_weights
        )
        
        # Interaction effect
        interaction_effect = active_return - allocation_effect - selection_effect
        
        self.attribution_results = {
            'total_active_return': active_return,
            'allocation_effect': allocation_effect,
            'selection_effect': selection_effect,
            'interaction_effect': interaction_effect
        }
        
        return self.attribution_results
    
    def _calculate_allocation_effect(self, 
                                   asset_returns: pd.DataFrame,
                                   weights: pd.DataFrame,
                                   benchmark_weights: Optional[pd.DataFrame] = None) -> float:
        """Calculate asset allocation effect"""
        
        if benchmark_weights is None:
            # Equal weight benchmark
            benchmark_weights = pd.DataFrame(
                np.ones_like(weights.values) / len(weights.columns),
                index=weights.index,
                columns=weights.columns
            )
        
        # Calculate allocation effect for each asset
        allocation_effects = []
        for asset in asset_returns.columns:
            weight_diff = weights[asset] - benchmark_weights[asset]
            asset_return = asset_returns[asset]
            allocation_effect = np.mean(weight_diff * asset_return)
            allocation_effects.append(allocation_effect)
        
        return np.sum(allocation_effects)
    
    def _calculate_selection_effect(self, 
                                  asset_returns: pd.DataFrame,
                                  weights: pd.DataFrame,
                                  benchmark_weights: Optional[pd.DataFrame] = None) -> float:
        """Calculate security selection effect"""
        
        if benchmark_weights is None:
            # Equal weight benchmark
            benchmark_weights = pd.DataFrame(
                np.ones_like(weights.values) / len(weights.columns),
                index=weights.index,
                columns=weights.columns
            )
        
        # Calculate selection effect for each asset
        selection_effects = []
        for asset in asset_returns.columns:
            benchmark_weight = benchmark_weights[asset]
            asset_return = asset_returns[asset]
            selection_effect = np.mean(benchmark_weight * asset_return)
            selection_effects.append(selection_effect)
        
        return np.sum(selection_effects)


# Convenience functions
def calculate_trading_metrics(returns: np.ndarray, 
                            benchmark_returns: Optional[np.ndarray] = None,
                            **kwargs) -> Dict[str, float]:
    """Convenience function to calculate trading metrics"""
    calculator = TradingMetrics(**kwargs)
    return calculator.calculate_all_metrics(returns, benchmark_returns)


def calculate_portfolio_metrics(portfolio_returns: np.ndarray,
                              asset_returns: pd.DataFrame,
                              **kwargs) -> Dict[str, Any]:
    """Convenience function to calculate portfolio metrics"""
    calculator = PortfolioMetrics(**kwargs)
    return calculator.calculate_portfolio_metrics(portfolio_returns, asset_returns)


def calculate_performance_attribution(portfolio_returns: np.ndarray,
                                    benchmark_returns: np.ndarray,
                                    asset_returns: pd.DataFrame,
                                    weights: pd.DataFrame,
                                    **kwargs) -> Dict[str, Any]:
    """Convenience function to calculate performance attribution"""
    calculator = PerformanceAttribution()
    return calculator.calculate_attribution(
        portfolio_returns, benchmark_returns, asset_returns, weights, **kwargs
    )


if __name__ == "__main__":
    # Demo
    print("Trading Metrics Demo")
    print("=" * 50)
    
    # Generate sample data
    np.random.seed(42)
    n_days = 252
    n_assets = 5
    
    # Generate asset returns
    asset_returns = pd.DataFrame(
        np.random.normal(0.0005, 0.02, (n_days, n_assets)),
        columns=[f'Asset_{i}' for i in range(n_assets)]
    )
    
    # Generate portfolio weights
    weights = pd.DataFrame(
        np.random.dirichlet(np.ones(n_assets), n_days),
        columns=[f'Asset_{i}' for i in range(n_assets)]
    )
    
    # Calculate portfolio returns
    portfolio_returns = (asset_returns * weights).sum(axis=1).values
    
    # Generate benchmark returns
    benchmark_returns = np.random.normal(0.0003, 0.015, n_days)
    
    # Calculate trading metrics
    print("\nCalculating Trading Metrics...")
    trading_metrics = calculate_trading_metrics(portfolio_returns, benchmark_returns)
    
    print(f"Total Return: {trading_metrics['total_return']:.4f}")
    print(f"Annualized Return: {trading_metrics['annualized_return']:.4f}")
    print(f"Sharpe Ratio: {trading_metrics['sharpe_ratio']:.4f}")
    print(f"Sortino Ratio: {trading_metrics['sortino_ratio']:.4f}")
    print(f"Max Drawdown: {trading_metrics['max_drawdown']:.4f}")
    print(f"Information Ratio: {trading_metrics['information_ratio']:.4f}")
    
    # Calculate portfolio metrics
    print("\nCalculating Portfolio Metrics...")
    portfolio_metrics = calculate_portfolio_metrics(portfolio_returns, asset_returns, weights)
    
    print(f"Average Correlation: {portfolio_metrics['avg_correlation']:.4f}")
    print(f"Effective Number of Assets: {portfolio_metrics['effective_number_of_assets']:.2f}")
    print(f"Concentration Risk: {portfolio_metrics['concentration_risk']:.4f}")
    
    # Calculate performance attribution
    print("\nCalculating Performance Attribution...")
    attribution = calculate_performance_attribution(
        portfolio_returns, benchmark_returns, asset_returns, weights
    )
    
    print(f"Total Active Return: {attribution['total_active_return']:.4f}")
    print(f"Allocation Effect: {attribution['allocation_effect']:.4f}")
    print(f"Selection Effect: {attribution['selection_effect']:.4f}")
    print(f"Interaction Effect: {attribution['interaction_effect']:.4f}")
    
    print("\nDemo completed!")
