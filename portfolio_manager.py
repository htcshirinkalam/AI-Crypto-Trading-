#!/usr/bin/env python3
"""
Portfolio Manager for Crypto Trading Platform
============================================

Handles:
- Initial cash management
- Trade validation (sufficient funds/assets)
- Trade execution and storage
- Portfolio calculation from trade history
- Profit/loss tracking
- Real-time asset holdings
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from loguru import logger
from database.database_manager import DatabaseManager


class PortfolioManager:
    """Manages portfolio state, trades, and calculations"""
    
    def __init__(self, db_manager: DatabaseManager, initial_cash: float = 10000.0):
        self.db = db_manager
        self.initial_cash = initial_cash
        self._current_holdings = {}  # Cache for current holdings
        self._cash_balance = initial_cash
        self._last_update = None
        
    def set_initial_cash(self, amount: float) -> bool:
        """Set the initial cash for the portfolio"""
        try:
            if amount < 0:
                logger.error("Initial cash cannot be negative")
                return False
            
            # Store as a transaction
            transaction = {
                'transaction_type': 'deposit',
                'asset': 'USD',
                'amount': amount,
                'value_usd': amount,
                'description': 'Initial cash deposit',
                'timestamp': datetime.now(),
                'status': 'completed'
            }
            
            self.db.store_data('transaction', transaction)
            self.initial_cash = amount
            self._cash_balance = amount
            logger.info(f"Set initial cash to ${amount:,.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set initial cash: {e}")
            return False
    
    def get_current_holdings(self, force_recalculate: bool = False) -> Dict[str, Dict]:
        """
        Calculate current holdings from trade history
        
        Returns:
            Dict with symbol as key and holdings info as value:
            {
                'BTC': {
                    'quantity': 0.5,
                    'avg_price': 45000.0,
                    'total_cost': 22500.0
                }
            }
        """
        try:
            # Use cache if available and not forced
            if not force_recalculate and self._current_holdings and self._last_update:
                if (datetime.now() - self._last_update).seconds < 60:  # Cache for 1 minute
                    return self._current_holdings
            
            # Get all trades from database
            trades_df = self.db.get_data('trade_history', limit=10000)
            
            if trades_df.empty:
                self._current_holdings = {}
                self._cash_balance = self.initial_cash
                self._last_update = datetime.now()
                return {}
            
            # Calculate holdings from trades
            holdings = {}
            cash_balance = self.initial_cash
            
            # Sort trades by timestamp
            trades_df = trades_df.sort_values('timestamp')
            
            for _, trade in trades_df.iterrows():
                symbol = trade['symbol']
                side = trade['side'].upper()
                quantity = float(trade['quantity'])
                price = float(trade['price'])
                total_value = float(trade['total_value'])
                fee = float(trade.get('fee', 0))
                
                if symbol not in holdings:
                    holdings[symbol] = {
                        'quantity': 0.0,
                        'total_cost': 0.0,
                        'avg_price': 0.0
                    }
                
                if side == 'BUY':
                    # Add to holdings
                    new_quantity = holdings[symbol]['quantity'] + quantity
                    new_cost = holdings[symbol]['total_cost'] + total_value + fee
                    holdings[symbol]['quantity'] = new_quantity
                    holdings[symbol]['total_cost'] = new_cost
                    holdings[symbol]['avg_price'] = new_cost / new_quantity if new_quantity > 0 else 0
                    
                    # Deduct from cash
                    cash_balance -= (total_value + fee)
                    
                elif side == 'SELL':
                    # Remove from holdings
                    if holdings[symbol]['quantity'] >= quantity:
                        # Calculate cost basis of sold portion
                        cost_basis = (quantity / holdings[symbol]['quantity']) * holdings[symbol]['total_cost'] if holdings[symbol]['quantity'] > 0 else 0
                        
                        holdings[symbol]['quantity'] -= quantity
                        holdings[symbol]['total_cost'] -= cost_basis
                        
                        if holdings[symbol]['quantity'] > 0:
                            holdings[symbol]['avg_price'] = holdings[symbol]['total_cost'] / holdings[symbol]['quantity']
                        else:
                            holdings[symbol]['avg_price'] = 0
                        
                        # Add to cash
                        cash_balance += (total_value - fee)
                    else:
                        logger.warning(f"Attempted to sell more {symbol} than available")
            
            # Remove zero holdings
            holdings = {k: v for k, v in holdings.items() if v['quantity'] > 0.000001}
            
            self._current_holdings = holdings
            self._cash_balance = cash_balance
            self._last_update = datetime.now()
            
            return holdings
            
        except Exception as e:
            logger.error(f"Failed to calculate current holdings: {e}")
            return {}
    
    def get_cash_balance(self) -> float:
        """Get current cash balance"""
        self.get_current_holdings()  # Update cache
        return self._cash_balance
    
    def validate_trade(self, symbol: str, side: str, quantity: float, price: float) -> Tuple[bool, str]:
        """
        Validate if a trade can be executed
        
        Returns:
            (is_valid, error_message)
        """
        try:
            side = side.upper()
            
            if quantity <= 0:
                return False, "Quantity must be positive"
            
            if price <= 0:
                return False, "Price must be positive"
            
            total_value = quantity * price
            
            if side == 'BUY':
                # Check if enough cash
                cash_balance = self.get_cash_balance()
                if total_value > cash_balance:
                    return False, f"Insufficient funds. Need ${total_value:,.2f}, have ${cash_balance:,.2f}"
                return True, ""
                
            elif side == 'SELL':
                # Check if enough holdings
                holdings = self.get_current_holdings()
                if symbol not in holdings:
                    return False, f"No {symbol} holdings to sell"
                
                if quantity > holdings[symbol]['quantity']:
                    return False, f"Insufficient {symbol}. Trying to sell {quantity:.6f}, have {holdings[symbol]['quantity']:.6f}"
                return True, ""
            
            else:
                return False, "Side must be 'BUY' or 'SELL'"
                
        except Exception as e:
            logger.error(f"Trade validation error: {e}")
            return False, str(e)
    
    def execute_trade(self, symbol: str, side: str, quantity: float, price: float, 
                     notes: str = '', strategy: str = None) -> Tuple[bool, str]:
        """
        Execute and store a trade
        
        Returns:
            (success, message)
        """
        try:
            # Validate trade
            is_valid, error_msg = self.validate_trade(symbol, side, quantity, price)
            if not is_valid:
                return False, error_msg
            
            # Calculate total value
            total_value = quantity * price
            fee = total_value * 0.001  # 0.1% fee
            
            # Create trade record
            trade = {
                'symbol': symbol.upper(),
                'side': side.upper(),
                'quantity': quantity,
                'price': price,
                'total_value': total_value,
                'fee': fee,
                'fee_currency': 'USD',
                'exchange': 'manual',
                'order_type': 'market',
                'status': 'completed',
                'strategy': strategy,
                'timestamp': datetime.now(),
                'executed_at': datetime.now(),
                'notes': notes
            }
            
            # Store in database
            success = self.db.store_data('trade', trade)
            
            if success:
                # Invalidate cache
                self._last_update = None
                action = "Bought" if side.upper() == 'BUY' else "Sold"
                msg = f"{action} {quantity:.6f} {symbol} @ ${price:,.2f} (Total: ${total_value:,.2f})"
                logger.info(msg)
                return True, msg
            else:
                return False, "Failed to store trade in database"
                
        except Exception as e:
            logger.error(f"Failed to execute trade: {e}")
            return False, str(e)
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> Dict:
        """
        Calculate total portfolio value with current market prices
        
        Args:
            current_prices: Dict mapping symbol to current price
            
        Returns:
            Dict with portfolio metrics
        """
        try:
            holdings = self.get_current_holdings()
            cash_balance = self.get_cash_balance()
            
            # Calculate crypto holdings value
            crypto_value = 0.0
            holdings_detail = []
            
            for symbol, holding in holdings.items():
                current_price = current_prices.get(symbol, 0.0)
                if current_price == 0:
                    logger.warning(f"No price available for {symbol}")
                
                value = holding['quantity'] * current_price
                pnl = value - holding['total_cost']
                pnl_pct = (pnl / holding['total_cost'] * 100) if holding['total_cost'] > 0 else 0
                
                holdings_detail.append({
                    'symbol': symbol,
                    'quantity': holding['quantity'],
                    'avg_price': holding['avg_price'],
                    'current_price': current_price,
                    'cost_basis': holding['total_cost'],
                    'current_value': value,
                    'unrealized_pnl': pnl,
                    'unrealized_pnl_pct': pnl_pct
                })
                
                crypto_value += value
            
            # Total portfolio value
            total_value = cash_balance + crypto_value
            
            # Calculate total return
            total_return_value = total_value - self.initial_cash
            total_return_pct = (total_return_value / self.initial_cash * 100) if self.initial_cash > 0 else 0
            
            return {
                'initial_cash': self.initial_cash,
                'cash_balance': cash_balance,
                'crypto_value': crypto_value,
                'total_value': total_value,
                'total_return': total_return_pct,
                'total_return_value': total_return_value,
                'holdings': holdings_detail,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate portfolio value: {e}")
            return {
                'initial_cash': self.initial_cash,
                'cash_balance': self.initial_cash,
                'crypto_value': 0.0,
                'total_value': self.initial_cash,
                'total_return': 0.0,
                'total_return_value': 0.0,
                'holdings': [],
                'timestamp': datetime.now()
            }
    
    def get_trade_history(self, symbol: str = None, days: int = 30) -> pd.DataFrame:
        """Get recent trade history"""
        try:
            start_date = datetime.now() - timedelta(days=days)
            return self.db.get_data('trade_history', symbol=symbol, start_date=start_date, limit=1000)
        except Exception as e:
            logger.error(f"Failed to get trade history: {e}")
            return pd.DataFrame()
    
    def get_performance_summary(self) -> Dict:
        """Get overall trading performance summary"""
        try:
            trades_df = self.get_trade_history(days=365)
            
            if trades_df.empty:
                return {
                    'total_trades': 0,
                    'buy_trades': 0,
                    'sell_trades': 0,
                    'total_volume': 0.0,
                    'total_fees': 0.0,
                    'avg_trade_size': 0.0
                }
            
            buy_trades = trades_df[trades_df['side'] == 'BUY']
            sell_trades = trades_df[trades_df['side'] == 'SELL']
            
            return {
                'total_trades': len(trades_df),
                'buy_trades': len(buy_trades),
                'sell_trades': len(sell_trades),
                'total_volume': trades_df['total_value'].sum(),
                'total_fees': trades_df['fee'].sum(),
                'avg_trade_size': trades_df['total_value'].mean()
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            return {}


# Example usage
if __name__ == "__main__":
    from config import Config
    
    # Initialize
    config = Config()
    db_manager = DatabaseManager(config.__dict__)
    portfolio = PortfolioManager(db_manager, initial_cash=10000.0)
    
    # Test trades
    print("Initial cash:", portfolio.get_cash_balance())
    
    # Buy BTC
    success, msg = portfolio.execute_trade('BTC', 'BUY', 0.1, 45000.0, notes='Test buy')
    print(f"Buy BTC: {msg}")
    
    # Check holdings
    holdings = portfolio.get_current_holdings()
    print("Holdings:", holdings)
    
    # Check cash
    print("Cash after buy:", portfolio.get_cash_balance())
    
    # Try to buy more than we have
    success, msg = portfolio.execute_trade('ETH', 'BUY', 10, 3000.0)
    print(f"Over-buy attempt: {msg}")
    
    # Sell some BTC
    success, msg = portfolio.execute_trade('BTC', 'SELL', 0.05, 46000.0, notes='Test sell')
    print(f"Sell BTC: {msg}")
    
    # Portfolio value
    current_prices = {'BTC': 47000.0}
    portfolio_value = portfolio.get_portfolio_value(current_prices)
    print(f"\nPortfolio Value: ${portfolio_value['total_value']:,.2f}")
    print(f"Total Return: {portfolio_value['total_return']:+.2f}%")
    print(f"P&L: ${portfolio_value['total_return_value']:+,.2f}")
