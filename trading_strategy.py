import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import json
from loguru import logger
from config import Config

class CryptoTradingStrategy:
    def __init__(self):
        self.config = Config()
        self.portfolio = {}
        self.trade_history = []
        self.current_positions = {}
        self.risk_metrics = {}
        
    def generate_trading_signals(self, features_df: pd.DataFrame, 
                                sentiment_data: Dict = None) -> pd.DataFrame:
        """Generate trading signals based on features and sentiment"""
        try:
            if features_df.empty:
                logger.error("Features dataframe is empty")
                return pd.DataFrame()
            
            df_signals = features_df.copy()
            
            # Initialize signal columns
            df_signals['buy_signal'] = 0
            df_signals['sell_signal'] = 0
            df_signals['hold_signal'] = 0
            df_signals['signal_strength'] = 0.0
            df_signals['confidence'] = 0.0
            
            # Technical analysis signals
            df_signals = self._generate_technical_signals(df_signals)
            
            # Sentiment-based signals
            if sentiment_data:
                df_signals = self._generate_sentiment_signals(df_signals, sentiment_data)
            
            # Combined signal generation
            df_signals = self._combine_signals(df_signals)
            
            # Risk assessment
            df_signals = self._assess_risk(df_signals)
            
            logger.info(f"Generated trading signals for {len(df_signals)} data points")
            return df_signals
            
        except Exception as e:
            logger.error(f"Error generating trading signals: {e}")
            return pd.DataFrame()
    
    def _generate_technical_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on technical indicators"""
        try:
            # RSI signals - use rsi_14 which is created by feature engineer
            if 'rsi_14' in df.columns:
                df['rsi_buy'] = np.where(df['rsi_14'] < 30, 1, 0)
                df['rsi_sell'] = np.where(df['rsi_14'] > 70, 1, 0)
            else:
                # Fallback if rsi_14 not available
                df['rsi_buy'] = 0
                df['rsi_sell'] = 0
            
            # MACD signals - use columns created by feature engineer
            if 'macd' in df.columns and 'macd_signal' in df.columns and 'macd_histogram' in df.columns:
                df['macd_buy'] = np.where(
                    (df['macd'] > df['macd_signal']) &
                    (df['macd_histogram'] > 0), 1, 0
                )
                df['macd_sell'] = np.where(
                    (df['macd'] < df['macd_signal']) &
                    (df['macd_histogram'] < 0), 1, 0
                )
            else:
                # Fallback if MACD columns not available
                df['macd_buy'] = 0
                df['macd_sell'] = 0
            
            # Moving average signals - use SMA columns created by feature engineer
            if 'sma_5' in df.columns and 'sma_10' in df.columns:
                df['ma_buy'] = np.where(
                    (df['close'] > df['sma_5']) &
                    (df['sma_5'] > df['sma_10']), 1, 0
                )
                df['ma_sell'] = np.where(
                    (df['close'] < df['sma_5']) &
                    (df['sma_5'] < df['sma_10']), 1, 0
                )
            else:
                # Fallback if MA columns not available
                df['ma_buy'] = 0
                df['ma_sell'] = 0
            
            # Bollinger Bands signals - use BB columns created by feature engineer
            if 'bb_lower_20' in df.columns and 'bb_upper_20' in df.columns and 'bb_position_20' in df.columns:
                df['bb_buy'] = np.where(
                    (df['close'] < df['bb_lower_20']) &
                    (df['bb_position_20'] < 0.2), 1, 0
                )
                df['bb_sell'] = np.where(
                    (df['close'] > df['bb_upper_20']) &
                    (df['bb_position_20'] > 0.8), 1, 0
                )
            else:
                # Fallback if BB columns not available
                df['bb_buy'] = 0
                df['bb_sell'] = 0
            
            # Volume signals - use volume_ratio created by feature engineer
            if 'volume_ratio_5' in df.columns:
                df['volume_buy'] = np.where(df['volume_ratio_5'] > 1.5, 1, 0)
                df['volume_sell'] = np.where(df['volume_ratio_5'] < 0.5, 1, 0)
            else:
                df['volume_buy'] = 0
                df['volume_sell'] = 0

            # Momentum signals - use price change as momentum proxy
            if 'price_change_1' in df.columns:
                df['momentum_buy'] = np.where(df['price_change_1'] > 0, 1, 0)
                df['momentum_sell'] = np.where(df['price_change_1'] < 0, 1, 0)
            else:
                df['momentum_buy'] = 0
                df['momentum_sell'] = 0
            
            # Stochastic signals - use columns created by feature engineer
            if 'stoch_k' in df.columns and 'stoch_d' in df.columns:
                df['stoch_buy'] = np.where(
                    (df['stoch_k'] < 20) & (df['stoch_d'] < 20), 1, 0
                )
                df['stoch_sell'] = np.where(
                    (df['stoch_k'] > 80) & (df['stoch_d'] > 80), 1, 0
                )
            else:
                # Fallback if stochastic columns not available
                df['stoch_buy'] = 0
                df['stoch_sell'] = 0
            
            return df
            
        except Exception as e:
            logger.error(f"Error generating technical signals: {e}")
            return df
    
    def _generate_sentiment_signals(self, df: pd.DataFrame, sentiment_data: Dict) -> pd.DataFrame:
        """Generate signals based on sentiment analysis"""
        try:
            # Always initialize sentiment columns to avoid KeyError
            df['sentiment_buy'] = 0
            df['sentiment_sell'] = 0
            df['sentiment_momentum_buy'] = 0
            df['sentiment_momentum_sell'] = 0
            df['divergence_buy'] = 0
            df['divergence_sell'] = 0

            if sentiment_data is None or 'sentiment_score' not in df.columns:
                return df

            # Sentiment-based signals
            df['sentiment_buy'] = np.where(
                (df['sentiment_score'] > 0.3) &
                (df.get('sentiment_confidence', 0) > 0.7), 1, 0
            )
            df['sentiment_sell'] = np.where(
                (df['sentiment_score'] < -0.3) &
                (df.get('sentiment_confidence', 0) > 0.7), 1, 0
            )

            # Sentiment momentum signals
            if 'sentiment_momentum' in df.columns and 'sentiment_momentum_ma' in df.columns:
                df['sentiment_momentum_buy'] = np.where(
                    (df['sentiment_momentum'] > 0.1) &
                    (df['sentiment_momentum_ma'] > 0), 1, 0
                )
                df['sentiment_momentum_sell'] = np.where(
                    (df['sentiment_momentum'] < -0.1) &
                    (df['sentiment_momentum_ma'] < 0), 1, 0
                )

            # Price-sentiment divergence signals
            if 'price_sentiment_divergence' in df.columns:
                df['divergence_buy'] = np.where(
                    (df['price_sentiment_divergence'] > 0.1) &
                    (df['sentiment_score'] > 0), 1, 0
                )
                df['divergence_sell'] = np.where(
                    (df['price_sentiment_divergence'] < -0.1) &
                    (df['sentiment_score'] < 0), 1, 0
                )

            return df

        except Exception as e:
            logger.error(f"Error generating sentiment signals: {e}")
            return df
    
    def _combine_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Combine all signals into final trading decisions"""
        try:
            # Ensure all required columns exist with default values
            required_cols = [
                'rsi_buy', 'rsi_sell', 'macd_buy', 'macd_sell', 'ma_buy', 'ma_sell',
                'bb_buy', 'bb_sell', 'volume_buy', 'volume_sell', 'momentum_buy', 'momentum_sell',
                'stoch_buy', 'stoch_sell', 'sentiment_buy', 'sentiment_sell',
                'sentiment_momentum_buy', 'sentiment_momentum_sell', 'divergence_buy', 'divergence_sell'
            ]

            for col in required_cols:
                if col not in df.columns:
                    df[col] = 0

            # Calculate signal strength for each category
            technical_buy = (df['rsi_buy'] + df['macd_buy'] + df['ma_buy'] +
                           df['bb_buy'] + df['volume_buy'] + df['momentum_buy'] +
                           df['stoch_buy'])

            technical_sell = (df['rsi_sell'] + df['macd_sell'] + df['ma_sell'] +
                            df['bb_sell'] + df['volume_sell'] + df['momentum_sell'] +
                            df['stoch_sell'])

            # Calculate sentiment signals
            sentiment_buy = (df['sentiment_buy'] + df['sentiment_momentum_buy'] + df['divergence_buy'])
            sentiment_sell = (df['sentiment_sell'] + df['sentiment_momentum_sell'] + df['divergence_sell'])
            
            # Weighted combination
            total_buy_signals = (0.7 * technical_buy + 0.3 * sentiment_buy)
            total_sell_signals = (0.7 * technical_sell + 0.3 * sentiment_sell)
            
            # Generate final signals
            df['buy_signal'] = np.where(
                (total_buy_signals >= 3) & (total_sell_signals < 2), 1, 0
            )
            
            df['sell_signal'] = np.where(
                (total_sell_signals >= 3) & (total_buy_signals < 2), 1, 0
            )
            
            df['hold_signal'] = np.where(
                (df['buy_signal'] == 0) & (df['sell_signal'] == 0), 1, 0
            )
            
            # Calculate signal strength (0-1 scale)
            df['signal_strength'] = np.where(
                df['buy_signal'] == 1, total_buy_signals / 7,
                np.where(df['sell_signal'] == 1, total_sell_signals / 7, 0)
            )
            
            # Calculate confidence based on signal agreement
            df['confidence'] = np.where(
                df['buy_signal'] == 1, 
                np.minimum(total_buy_signals / 7, 1.0),
                np.where(df['sell_signal'] == 1, 
                        np.minimum(total_sell_signals / 7, 1.0), 0.5)
            )
            
            return df
            
        except Exception as e:
            logger.error(f"Error combining signals: {e}")
            return df
    
    def _assess_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assess risk for each trading signal"""
        try:
            # Volatility risk
            if 'atr' in df.columns:
                df['volatility_risk'] = np.where(
                    df['atr'] > df['atr'].rolling(20).mean() * 1.5, 'HIGH',
                    np.where(df['atr'] > df['atr'].rolling(20).mean(), 'MEDIUM', 'LOW')
                )
            else:
                df['volatility_risk'] = 'MEDIUM'

            # Market regime risk
            if 'bull_market' in df.columns and 'high_volatility' in df.columns and 'bear_market' in df.columns:
                df['market_regime_risk'] = np.where(
                    (df['bull_market'] == 1) & (df['high_volatility'] == 1), 'HIGH',
                    np.where(df['bear_market'] == 1, 'HIGH', 'MEDIUM')
                )
            else:
                df['market_regime_risk'] = 'MEDIUM'

            # Liquidity risk (based on volume)
            if 'volume_ratio' in df.columns:
                df['liquidity_risk'] = np.where(
                    df['volume_ratio'] < 0.5, 'HIGH',
                    np.where(df['volume_ratio'] < 1.0, 'MEDIUM', 'LOW')
                )
            else:
                df['liquidity_risk'] = 'MEDIUM'

            # Overall risk score
            risk_mapping = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3}
            df['risk_score'] = (
                df['volatility_risk'].map(risk_mapping) +
                df['market_regime_risk'].map(risk_mapping) +
                df['liquidity_risk'].map(risk_mapping)
            ) / 3
            
            # Adjust signals based on risk
            df['buy_signal'] = np.where(
                (df['buy_signal'] == 1) & (df['risk_score'] <= 2.5), 1, 0
            )
            
            df['sell_signal'] = np.where(
                (df['sell_signal'] == 1) & (df['risk_score'] <= 2.5), 1, 0
            )
            
            return df
            
        except Exception as e:
            logger.error(f"Error assessing risk: {e}")
            return df
    
    def execute_trades(self, signals_df: pd.DataFrame, portfolio_value: float = 10000.0,
                      max_position_size: float = 0.1) -> List[Dict]:
        """Execute trades based on signals"""
        try:
            trades = []
            current_cash = portfolio_value
            positions = {}
            
            for idx, row in signals_df.iterrows():
                symbol = row.get('symbol', 'UNKNOWN')
                timestamp = row.get('timestamp')
                
                if pd.isna(timestamp):
                    continue
                
                # Get current price
                current_price = row.get('close', 0)
                if current_price <= 0:
                    continue
                
                # Execute buy signal
                if row['buy_signal'] == 1 and row['confidence'] >= self.config.TRADING_PARAMS['min_confidence']:
                    # Calculate position size
                    signal_strength = row['signal_strength']
                    confidence = row['confidence']
                    risk_score = row['risk_score']
                    
                    # Adjust position size based on confidence and risk
                    base_size = max_position_size * signal_strength * confidence
                    risk_adjustment = max(0.5, 1 - (risk_score - 1) * 0.2)
                    position_size = base_size * risk_adjustment
                    
                    # Calculate investment amount
                    investment = portfolio_value * position_size
                    
                    if investment <= current_cash:
                        # Execute buy
                        shares = investment / current_price
                        positions[symbol] = {
                            'shares': shares,
                            'entry_price': current_price,
                            'entry_time': timestamp,
                            'investment': investment
                        }
                        
                        current_cash -= investment
                        
                        trades.append({
                            'timestamp': timestamp,
                            'symbol': symbol,
                            'action': 'BUY',
                            'shares': shares,
                            'price': current_price,
                            'investment': investment,
                            'confidence': confidence,
                            'signal_strength': signal_strength,
                            'risk_score': risk_score
                        })
                        
                        logger.info(f"BUY: {symbol} - {shares:.4f} shares at ${current_price:.2f}")
                
                # Execute sell signal
                elif row['sell_signal'] == 1 and symbol in positions:
                    position = positions[symbol]
                    shares = position['shares']
                    exit_price = current_price
                    exit_value = shares * exit_price
                    
                    # Calculate P&L
                    entry_value = position['investment']
                    pnl = exit_value - entry_value
                    pnl_percentage = (pnl / entry_value) * 100
                    
                    # Execute sell
                    current_cash += exit_value
                    del positions[symbol]
                    
                    trades.append({
                        'timestamp': timestamp,
                            'symbol': symbol,
                            'action': 'SELL',
                            'shares': shares,
                            'price': exit_price,
                            'value': exit_value,
                            'pnl': pnl,
                            'pnl_percentage': pnl_percentage,
                            'confidence': row['confidence'],
                            'signal_strength': row['signal_strength'],
                            'risk_score': row['risk_score']
                    })
                    
                    logger.info(f"SELL: {symbol} - {shares:.4f} shares at ${exit_price:.2f}, P&L: ${pnl:.2f} ({pnl_percentage:.2f}%)")
            
            # Update portfolio state
            self.portfolio = positions
            self.trade_history.extend(trades)
            
            # Calculate portfolio metrics
            self._calculate_portfolio_metrics(portfolio_value, current_cash)
            
            return trades
            
        except Exception as e:
            logger.error(f"Error executing trades: {e}")
            return []
    
    def _calculate_portfolio_metrics(self, initial_value: float, current_cash: float):
        """Calculate portfolio performance metrics"""
        try:
            # Calculate current portfolio value
            current_portfolio_value = current_cash
            for symbol, position in self.portfolio.items():
                # This would need current market prices in a real implementation
                current_portfolio_value += position['investment']
            
            # Calculate returns
            total_return = ((current_portfolio_value - initial_value) / initial_value) * 100
            
            # Calculate trade statistics
            if self.trade_history:
                buy_trades = [t for t in self.trade_history if t['action'] == 'BUY']
                sell_trades = [t for t in self.trade_history if t['action'] == 'SELL']
                
                total_trades = len(buy_trades) + len(sell_trades)
                winning_trades = len([t for t in sell_trades if t.get('pnl', 0) > 0])
                losing_trades = len([t for t in sell_trades if t.get('pnl', 0) < 0])
                
                win_rate = (winning_trades / len(sell_trades) * 100) if sell_trades else 0
                
                # Calculate average P&L
                if sell_trades:
                    avg_win = np.mean([t.get('pnl', 0) for t in sell_trades if t.get('pnl', 0) > 0])
                    avg_loss = np.mean([t.get('pnl', 0) for t in sell_trades if t.get('pnl', 0) < 0])
                    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
                else:
                    avg_win = avg_loss = profit_factor = 0
                
                self.risk_metrics = {
                    'initial_value': initial_value,
                    'current_value': current_portfolio_value,
                    'total_return': total_return,
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades,
                    'win_rate': win_rate,
                    'avg_win': avg_win,
                    'avg_loss': avg_loss,
                    'profit_factor': profit_factor,
                    'current_positions': len(self.portfolio),
                    'cash_ratio': current_cash / current_portfolio_value if current_portfolio_value > 0 else 0
                }
                
                logger.info(f"Portfolio metrics calculated - Total Return: {total_return:.2f}%, Win Rate: {win_rate:.2f}%")
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
    
    def implement_risk_management(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        """Implement risk management rules"""
        try:
            df_risk = signals_df.copy()
            
            # Stop loss implementation
            df_risk['stop_loss_price'] = np.where(
                df_risk['buy_signal'] == 1,
                df_risk['close'] * (1 - self.config.TRADING_PARAMS['stop_loss_percentage']),
                0
            )
            
            # Take profit implementation
            df_risk['take_profit_price'] = np.where(
                df_risk['buy_signal'] == 1,
                df_risk['close'] * (1 + self.config.TRADING_PARAMS['take_profit_percentage']),
                0
            )
            
            # Position sizing based on volatility
            df_risk['position_size'] = np.where(
                df_risk['buy_signal'] == 1,
                self.config.TRADING_PARAMS['max_position_size'] * 
                (1 - df_risk['risk_score'] / 3),  # Reduce size for higher risk
                0
            )
            
            # Maximum drawdown protection
            df_risk['max_drawdown'] = np.where(
                df_risk['close'] < df_risk['close'].rolling(20).max() * 0.9,
                1, 0  # Signal to reduce positions
            )
            
            # Correlation-based position limits
            # This would require correlation matrix calculation in a real implementation
            
            return df_risk
            
        except Exception as e:
            logger.error(f"Error implementing risk management: {e}")
            return signals_df
    
    def generate_trading_recommendations(self, signals_df: pd.DataFrame, 
                                       top_n: int = 5) -> List[Dict]:
        """Generate trading recommendations for top opportunities"""
        try:
            # Filter for buy signals with high confidence
            buy_signals = signals_df[
                (signals_df['buy_signal'] == 1) & 
                (signals_df['confidence'] >= self.config.TRADING_PARAMS['min_confidence'])
            ].copy()
            
            if buy_signals.empty:
                return []
            
            # Calculate opportunity score
            buy_signals['opportunity_score'] = (
                buy_signals['signal_strength'] * 0.4 +
                buy_signals['confidence'] * 0.3 +
                (1 - buy_signals['risk_score'] / 3) * 0.3
            )
            
            # Sort by opportunity score and select top N
            top_opportunities = buy_signals.nlargest(top_n, 'opportunity_score')
            
            recommendations = []
            for _, row in top_opportunities.iterrows():
                recommendation = {
                    'symbol': row['symbol'],
                    'action': 'BUY',
                    'current_price': row['close'],
                    'opportunity_score': row['opportunity_score'],
                    'confidence': row['confidence'],
                    'signal_strength': row['signal_strength'],
                    'risk_level': row['volatility_risk'],
                    'stop_loss': row.get('stop_loss_price', 0),
                    'take_profit': row.get('take_profit_price', 0),
                    'position_size': row.get('position_size', 0),
                    'timestamp': row['timestamp'],
                    'reasoning': self._generate_reasoning(row)
                }
                recommendations.append(recommendation)
            
            logger.info(f"Generated {len(recommendations)} trading recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating trading recommendations: {e}")
            return []
    
    def _generate_reasoning(self, row: pd.Series) -> str:
        """Generate reasoning for trading recommendation"""
        try:
            reasons = []
            
            # Technical reasons
            if row.get('rsi_buy', 0) == 1:
                reasons.append("RSI oversold")
            if row.get('macd_buy', 0) == 1:
                reasons.append("MACD bullish crossover")
            if row.get('ma_buy', 0) == 1:
                reasons.append("Price above moving averages")
            if row.get('bb_buy', 0) == 1:
                reasons.append("Price near Bollinger Band support")
            
            # Sentiment reasons
            if row.get('sentiment_buy', 0) == 1:
                reasons.append("Positive sentiment")
            if row.get('sentiment_momentum_buy', 0) == 1:
                reasons.append("Improving sentiment momentum")
            
            # Risk considerations
            if row.get('volatility_risk') == 'LOW':
                reasons.append("Low volatility environment")
            elif row.get('volatility_risk') == 'HIGH':
                reasons.append("High volatility - use smaller position")
            
            if not reasons:
                reasons.append("Multiple technical indicators aligned")
            
            return "; ".join(reasons)
            
        except Exception as e:
            logger.error(f"Error generating reasoning: {e}")
            return "Multiple factors indicate buying opportunity"
    
    def save_trading_results(self, filepath: str):
        """Save trading results and portfolio state"""
        try:
            import os
            os.makedirs(filepath, exist_ok=True)
            
            # Save trade history
            if self.trade_history:
                trades_df = pd.DataFrame(self.trade_history)
                trades_file = os.path.join(filepath, "trade_history.csv")
                trades_df.to_csv(trades_file, index=False)
                logger.info(f"Saved trade history to {trades_file}")
            
            # Save portfolio state
            if self.portfolio:
                portfolio_file = os.path.join(filepath, "portfolio_state.json")
                with open(portfolio_file, 'w') as f:
                    json.dump(self.portfolio, f, indent=2, default=str)
                logger.info(f"Saved portfolio state to {portfolio_file}")
            
            # Save risk metrics
            if self.risk_metrics:
                metrics_file = os.path.join(filepath, "risk_metrics.json")
                with open(metrics_file, 'w') as f:
                    json.dump(self.risk_metrics, f, indent=2, default=str)
                logger.info(f"Saved risk metrics to {metrics_file}")
            
        except Exception as e:
            logger.error(f"Error saving trading results: {e}")

# Example usage
def main():
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'symbol': ['BTC'] * 100,
        'open': np.random.uniform(40000, 50000, 100),
        'high': np.random.uniform(40000, 50000, 100),
        'low': np.random.uniform(40000, 50000, 100),
        'close': np.random.uniform(40000, 50000, 100),
        'volume': np.random.uniform(1000, 5000, 100),
        'rsi': np.random.uniform(20, 80, 100),
        'macd': np.random.uniform(-100, 100, 100),
        'macd_signal': np.random.uniform(-100, 100, 100),
        'ma_short': np.random.uniform(40000, 50000, 100),
        'ma_long': np.random.uniform(40000, 50000, 100),
        'atr': np.random.uniform(1000, 3000, 100),
        'sentiment_score': np.random.uniform(-1, 1, 100),
        'sentiment_confidence': np.random.uniform(0.5, 1.0, 100)
    })
    
    # Initialize trading strategy
    strategy = CryptoTradingStrategy()
    
    # Generate trading signals
    signals = strategy.generate_trading_signals(sample_data)
    
    print(f"Generated signals for {len(signals)} data points")
    print(f"Buy signals: {signals['buy_signal'].sum()}")
    print(f"Sell signals: {signals['sell_signal'].sum()}")
    
    # Execute trades
    trades = strategy.execute_trades(signals, portfolio_value=10000.0)
    print(f"Executed {len(trades)} trades")
    
    # Generate recommendations
    recommendations = strategy.generate_trading_recommendations(signals, top_n=3)
    print(f"Generated {len(recommendations)} recommendations")
    
    # Save results
    strategy.save_trading_results("trading_results")

if __name__ == "__main__":
    main()
