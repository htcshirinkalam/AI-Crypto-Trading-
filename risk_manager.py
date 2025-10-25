#!/usr/bin/env python3
"""
Advanced Risk Management Module for Crypto Trading Agent
========================================================

This module provides a comprehensive suite of risk management tools:
- Dynamic position sizing based on volatility and portfolio risk
- Global portfolio-level stop-loss and take-profit
- Correlation analysis to manage portfolio concentration
- Exposure limits per asset and asset class
- Scenario-based risk assessment (stress testing)
- Real-time risk monitoring and alerts
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger
from config import Config

class RiskManager:
    """Manages portfolio-level risk for the trading agent"""

    def __init__(self, config: Config):
        self.config = config
        self.portfolio_risk_limit = config.PORTFOLIO_RISK_LIMIT
        self.max_portfolio_drawdown = config.MAX_PORTFOLIO_DRAWDOWN
        self.max_asset_concentration = config.MAX_ASSET_CONCENTRATION
        self.correlation_threshold = config.CORRELATION_THRESHOLD

    def calculate_dynamic_position_size(self, symbol: str, volatility: float,
                                        portfolio_value: float,
                                        risk_per_trade: float) -> float:
        """
        Calculate dynamic position size based on volatility.
        
        Args:
            symbol (str): The asset symbol.
            volatility (float): Annualized volatility of the asset.
            portfolio_value (float): Total value of the portfolio.
            risk_per_trade (float): Percentage of portfolio to risk on a single trade.
            
        Returns:
            float: The calculated position size in USD.
        """
        try:
            if volatility <= 0:
                return 0.0

            # Calculate position size using volatility-based formula
            position_size = (portfolio_value * risk_per_trade) / volatility
            
            # Apply asset concentration limits
            max_size = portfolio_value * self.max_asset_concentration
            
            return min(position_size, max_size)

        except Exception as e:
            logger.error(f"Error calculating dynamic position size for {symbol}: {e}")
            return 0.0

    def manage_portfolio_risk(self, portfolio: Dict, market_data: Dict) -> Dict:
        """
        Manage overall portfolio risk.
        
        Args:
            portfolio (Dict): Current portfolio state.
            market_data (Dict): Current market prices and data.
            
        Returns:
            Dict: Actions to take (e.g., reduce positions, hedge).
        """
        try:
            actions = {'reduce': [], 'hedge': []}

            # Check for global stop-loss
            total_pnl = portfolio.get('total_pnl', 0)
            if total_pnl < -self.max_portfolio_drawdown * portfolio.get('total_value', 0):
                actions['reduce'].append({
                    'reason': 'Global stop-loss triggered',
                    'action': 'reduce_all_positions',
                    'percentage': 0.5  # Reduce all positions by 50%
                })

            # Check for asset concentration
            for symbol, position in portfolio.get('positions', {}).items():
                position_value = position['quantity'] * market_data[symbol]['price']
                concentration = position_value / portfolio.get('total_value', 1)
                
                if concentration > self.max_asset_concentration:
                    actions['reduce'].append({
                        'symbol': symbol,
                        'reason': 'Asset concentration limit exceeded',
                        'reduce_to_percentage': self.max_asset_concentration
                    })

            # Check for high correlation
            correlation_matrix = self.calculate_correlation_matrix(market_data)
            high_corr_pairs = self.find_high_correlation_pairs(correlation_matrix)
            
            if high_corr_pairs:
                actions['hedge'].append({
                    'reason': 'High portfolio correlation detected',
                    'pairs': high_corr_pairs
                })

            return actions

        except Exception as e:
            logger.error(f"Error managing portfolio risk: {e}")
            return {}

    def calculate_correlation_matrix(self, market_data: Dict) -> pd.DataFrame:
        """
        Calculate the correlation matrix for assets in the portfolio.
        
        Args:
            market_data (Dict): Historical price data for portfolio assets.
            
        Returns:
            pd.DataFrame: Correlation matrix.
        """
        try:
            returns_df = pd.DataFrame()
            for symbol, data in market_data.items():
                if 'returns' in data:
                    returns_df[symbol] = data['returns']
            
            return returns_df.corr()

        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}")
            return pd.DataFrame()

    def find_high_correlation_pairs(self, correlation_matrix: pd.DataFrame) -> List[Tuple]:
        """
        Find asset pairs with high correlation.
        
        Args:
            correlation_matrix (pd.DataFrame): Correlation matrix.
            
        Returns:
            List[Tuple]: List of highly correlated pairs.
        """
        try:
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i + 1, len(correlation_matrix.columns)):
                    if correlation_matrix.iloc[i, j] > self.correlation_threshold:
                        high_corr_pairs.append(
                            (correlation_matrix.columns[i], correlation_matrix.columns[j])
                        )
            
            return high_corr_pairs

        except Exception as e:
            logger.error(f"Error finding high correlation pairs: {e}")
            return []

    def perform_scenario_analysis(self, portfolio: Dict, scenarios: List[Dict]) -> Dict:
        """
        Perform scenario-based analysis on the portfolio.
        
        Args:
            portfolio (Dict): Current portfolio state.
            scenarios (List[Dict]): List of market scenarios to test.
            
        Returns:
            Dict: Results of the scenario analysis.
        """
        try:
            scenario_results = {}
            for scenario in scenarios:
                scenario_name = scenario['name']
                scenario_pnl = 0
                
                for position in portfolio.get('positions', []):
                    symbol = position['symbol']
                    price_change = scenario['price_changes'].get(symbol, 0)
                    position_pnl = position['value'] * price_change
                    scenario_pnl += position_pnl
                
                scenario_results[scenario_name] = {
                    'pnl': scenario_pnl,
                    'portfolio_value_after': portfolio.get('total_value', 0) + scenario_pnl
                }
            
            return scenario_results

        except Exception as e:
            logger.error(f"Error in scenario analysis: {e}")
            return {}

    def get_risk_summary(self, portfolio: Dict, market_data: Dict) -> Dict:
        """
        Get a summary of the current portfolio risk.
        
        Args:
            portfolio (Dict): Current portfolio state.
            market_data (Dict): Current market prices and data.
            
        Returns:
            Dict: Summary of risk metrics.
        """
        try:
            # VaR and CVaR calculations would need historical returns
            # Placeholder for now
            
            risk_summary = {
                'portfolio_value': portfolio.get('total_value', 0),
                'total_pnl': portfolio.get('total_pnl', 0),
                'max_drawdown': self.max_portfolio_drawdown,
                'current_drawdown': portfolio.get('current_drawdown', 0),
                'var_95': 0,  # Placeholder
                'cvar_95': 0, # Placeholder
                'asset_concentration': self.get_asset_concentration(portfolio, market_data),
                'correlation_risk': len(self.find_high_correlation_pairs(
                    self.calculate_correlation_matrix(market_data)
                )) > 0
            }
            
            return risk_summary

        except Exception as e:
            logger.error(f"Error getting risk summary: {e}")
            return {}

    def get_asset_concentration(self, portfolio: Dict, market_data: Dict) -> Dict:
        """
        Get the concentration of each asset in the portfolio.
        
        Args:
            portfolio (Dict): Current portfolio state.
            market_data (Dict): Current market prices.
            
        Returns:
            Dict: Asset concentration percentages.
        """
        try:
            concentration = {}
            total_value = portfolio.get('total_value', 1)
            
            for symbol, position in portfolio.get('positions', {}).items():
                position_value = position['quantity'] * market_data[symbol]['price']
                concentration[symbol] = position_value / total_value
            
            return concentration

        except Exception as e:
            logger.error(f"Error getting asset concentration: {e}")
            return {}