#!/usr/bin/env python3
"""
Comprehensive Technical Analysis Engine
======================================

Integrates:
- Technical Indicators (26+ indicators)
- On-Chain Metrics
- Sentiment Analysis
- Risk Management
- Multi-Timeframe Analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger


class TechnicalAnalysisEngine:
    """Advanced technical analysis with comprehensive indicators"""
    
    def __init__(self):
        self.indicators = {}
        
    def calculate_all_indicators(self, df: pd.DataFrame, timeframe: str = '1D') -> pd.DataFrame:
        """
        Calculate all technical indicators for given OHLCV data
        
        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
            timeframe: Trading timeframe
            
        Returns:
            DataFrame with all indicators calculated
        """
        try:
            result = df.copy()
            
            # Price Action & Trend
            result = self._calculate_price_action(result)
            result = self._calculate_support_resistance(result)
            
            # Moving Averages
            result = self._calculate_sma(result, periods=[20, 50, 100, 200])
            result = self._calculate_ema(result, periods=[12, 26, 50, 200])
            result = self._calculate_vwap(result)
            
            # Momentum Indicators
            result = self._calculate_rsi(result, period=14)
            result = self._calculate_stochastic(result)
            result = self._calculate_macd(result)
            
            # Volatility Indicators
            result = self._calculate_bollinger_bands(result)
            result = self._calculate_atr(result)
            
            # Volume Indicators
            result = self._calculate_obv(result)
            result = self._calculate_cmf(result)
            
            # Chart Patterns
            result = self._detect_patterns(result)
            
            # Fibonacci Levels
            result = self._calculate_fibonacci(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return df
    
    def _calculate_price_action(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate price action metrics"""
        df['price_change'] = df['close'].pct_change()
        df['price_change_abs'] = df['close'].diff()
        df['high_low_range'] = df['high'] - df['low']
        df['close_open_range'] = df['close'] - df['open']
        
        # Candle body percentage
        df['body_pct'] = (df['close'] - df['open']).abs() / (df['high'] - df['low'])
        
        # Candle type
        df['is_bullish'] = (df['close'] > df['open']).astype(int)
        
        return df
    
    def _calculate_support_resistance(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Identify support and resistance levels"""
        # Rolling highs and lows
        df['resistance'] = df['high'].rolling(window=window).max()
        df['support'] = df['low'].rolling(window=window).min()
        
        # Distance from support/resistance
        df['dist_from_resistance'] = ((df['close'] - df['resistance']) / df['resistance'] * 100)
        df['dist_from_support'] = ((df['close'] - df['support']) / df['support'] * 100)
        
        return df

    def _calculate_sma(self, df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """Calculate Simple Moving Averages"""
        for period in periods:
            df[f'SMA_{period}'] = df['close'].rolling(window=period).mean()
        return df
    
    def _calculate_ema(self, df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """Calculate Exponential Moving Averages"""
        for period in periods:
            df[f'EMA_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        return df
    
    def _calculate_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Volume Weighted Average Price"""
        df['VWAP'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        return df
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Relative Strength Index"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        return df
    
    def _calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """Calculate Stochastic Oscillator"""
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        df['Stoch_K'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
        df['Stoch_D'] = df['Stoch_K'].rolling(window=d_period).mean()
        return df
    
    def _calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Calculate MACD"""
        ema_fast = df['close'].ewm(span=fast).mean()
        ema_slow = df['close'].ewm(span=slow).mean()
        df['MACD'] = ema_fast - ema_slow
        df['MACD_Signal'] = df['MACD'].ewm(span=signal).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        return df
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        df[f'BB_Middle_{period}'] = df['close'].rolling(window=period).mean()
        bb_std = df['close'].rolling(window=period).std()
        df[f'BB_Upper_{period}'] = df[f'BB_Middle_{period}'] + (bb_std * std_dev)
        df[f'BB_Lower_{period}'] = df[f'BB_Middle_{period}'] - (bb_std * std_dev)
        df[f'BB_Width_{period}'] = df[f'BB_Upper_{period}'] - df[f'BB_Lower_{period}']
        df[f'BB_Position_{period}'] = (df['close'] - df[f'BB_Lower_{period}']) / df[f'BB_Width_{period}']
        return df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        df[f'ATR_{period}'] = true_range.rolling(window=period).mean()
        return df
    
    def _calculate_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate On-Balance Volume"""
        df['OBV'] = (df['volume'] * np.sign(df['close'].diff())).cumsum()
        return df
    
    def _calculate_cmf(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Calculate Chaikin Money Flow"""
        mfv = df['volume'] * ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        df[f'CMF_{period}'] = mfv.rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
        return df
    
    def _detect_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect basic chart patterns"""
        # Doji pattern
        df['is_doji'] = (abs(df['close'] - df['open']) <= (df['high'] - df['low']) * 0.1).astype(int)
        
        # Hammer pattern
        body = abs(df['close'] - df['open'])
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        df['is_hammer'] = ((lower_shadow > 2 * body) & (upper_shadow < body)).astype(int)
        
        # Engulfing patterns
        df['bullish_engulfing'] = ((df['close'] > df['open']) & 
                                  (df['close'].shift(1) < df['open'].shift(1)) &
                                  (df['open'] < df['close'].shift(1)) &
                                  (df['close'] > df['open'].shift(1))).astype(int)
        
        df['bearish_engulfing'] = ((df['close'] < df['open']) & 
                                  (df['close'].shift(1) > df['open'].shift(1)) &
                                  (df['open'] > df['close'].shift(1)) &
                                  (df['close'] < df['open'].shift(1))).astype(int)
        
        return df
    
    def _calculate_fibonacci(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Calculate Fibonacci retracement levels"""
        high = df['high'].rolling(window=period).max()
        low = df['low'].rolling(window=period).min()
        diff = high - low
        
        df['Fib_23.6'] = high - (diff * 0.236)
        df['Fib_38.2'] = high - (diff * 0.382)
        df['Fib_50.0'] = high - (diff * 0.5)
        df['Fib_61.8'] = high - (diff * 0.618)
        df['Fib_78.6'] = high - (diff * 0.786)
        
        return df
    
    def generate_signal_score(self, df: pd.DataFrame, sentiment_score: float = 0.5, risk_score: float = 0.5) -> Dict[str, float]:
        """
        Generate a comprehensive trading signal score based on technical indicators
        
        Args:
            df: DataFrame with technical indicators calculated
            sentiment_score: Sentiment score (0.0 to 1.0, where 0.5 is neutral)
            risk_score: Risk score (0.0 to 1.0, where 0.5 is neutral)
            
        Returns:
            Dictionary with signal scores and overall recommendation
        """
        try:
            if df.empty:
                return {
                    'overall_score': 0.0,
                    'trend_score': 0.0,
                    'momentum_score': 0.0,
                    'volume_score': 0.0,
                    'volatility_score': 0.0,
                    'sentiment_score': 0.0,
                    'risk_score': 0.0,
                    'signal': 'NEUTRAL',
                    'confidence': 0.0
                }
            
            # Get the latest values (last row)
            latest = df.iloc[-1]
            
            # Trend Score (0-100)
            trend_score = 50.0  # Neutral starting point
            
            # Moving Average signals
            if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
                if latest['SMA_20'] > latest['SMA_50']:
                    trend_score += 20
                else:
                    trend_score -= 20
            
            if 'EMA_12' in df.columns and 'EMA_26' in df.columns:
                if latest['EMA_12'] > latest['EMA_26']:
                    trend_score += 15
                else:
                    trend_score -= 15
            
            # Price vs MA signals
            if 'SMA_20' in df.columns:
                if latest['close'] > latest['SMA_20']:
                    trend_score += 10
                else:
                    trend_score -= 10
            
            # Momentum Score (0-100)
            momentum_score = 50.0
            
            # RSI signals
            if 'RSI_14' in df.columns:
                rsi = latest['RSI_14']
                if rsi > 70:
                    momentum_score -= 20  # Overbought
                elif rsi < 30:
                    momentum_score += 20  # Oversold
                elif 40 < rsi < 60:
                    momentum_score += 5   # Neutral zone
            
            # MACD signals
            if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
                if latest['MACD'] > latest['MACD_Signal']:
                    momentum_score += 15
                else:
                    momentum_score -= 15
            
            # Stochastic signals
            if 'Stoch_K' in df.columns:
                stoch = latest['Stoch_K']
                if stoch > 80:
                    momentum_score -= 15  # Overbought
                elif stoch < 20:
                    momentum_score += 15  # Oversold
            
            # Volume Score (0-100)
            volume_score = 50.0
            
            # Volume trend
            if 'volume' in df.columns and len(df) > 5:
                recent_volume = df['volume'].tail(5).mean()
                avg_volume = df['volume'].mean()
                if recent_volume > avg_volume * 1.2:
                    volume_score += 20  # High volume
                elif recent_volume < avg_volume * 0.8:
                    volume_score -= 10  # Low volume
            
            # OBV trend
            if 'OBV' in df.columns and len(df) > 5:
                obv_trend = df['OBV'].tail(5).iloc[-1] - df['OBV'].tail(5).iloc[0]
                if obv_trend > 0:
                    volume_score += 10
                else:
                    volume_score -= 10
            
            # Volatility Score (0-100)
            volatility_score = 50.0
            
            # Bollinger Bands position
            if 'BB_Position_20' in df.columns:
                bb_pos = latest['BB_Position_20']
                if bb_pos > 0.8:
                    volatility_score -= 15  # Near upper band (overbought)
                elif bb_pos < 0.2:
                    volatility_score += 15  # Near lower band (oversold)
            
            # ATR for volatility assessment
            if 'ATR_14' in df.columns and len(df) > 20:
                current_atr = latest['ATR_14']
                avg_atr = df['ATR_14'].tail(20).mean()
                if current_atr > avg_atr * 1.5:
                    volatility_score -= 10  # High volatility
                elif current_atr < avg_atr * 0.5:
                    volatility_score += 5   # Low volatility
            
            # Sentiment Score (0-100)
            sentiment_score_100 = sentiment_score * 100  # Convert 0-1 to 0-100
            
            # Risk Score (0-100) - higher risk reduces confidence
            risk_score_100 = (1.0 - risk_score) * 100  # Invert risk (lower risk = higher score)
            
            # Calculate overall score (weighted average)
            overall_score = (
                trend_score * 0.25 +
                momentum_score * 0.25 +
                volume_score * 0.15 +
                volatility_score * 0.15 +
                sentiment_score_100 * 0.1 +
                risk_score_100 * 0.1
            )
            
            # Determine signal
            if overall_score >= 70:
                signal = 'STRONG_BUY'
                confidence = min(95, overall_score)
            elif overall_score >= 60:
                signal = 'BUY'
                confidence = overall_score
            elif overall_score >= 40:
                signal = 'NEUTRAL'
                confidence = 100 - abs(overall_score - 50) * 2
            elif overall_score >= 30:
                signal = 'SELL'
                confidence = 100 - overall_score
            else:
                signal = 'STRONG_SELL'
                confidence = min(95, 100 - overall_score)
            
            return {
                'overall_score': round(overall_score, 2),
                'trend_score': round(trend_score, 2),
                'momentum_score': round(momentum_score, 2),
                'volume_score': round(volume_score, 2),
                'volatility_score': round(volatility_score, 2),
                'sentiment_score': round(sentiment_score_100, 2),
                'risk_score': round(risk_score_100, 2),
                'signal': signal,
                'confidence': round(confidence, 2)
            }
            
        except Exception as e:
            logger.error(f"Error generating signal score: {e}")
            return {
                'overall_score': 50.0,
                'trend_score': 50.0,
                'momentum_score': 50.0,
                'volume_score': 50.0,
                'volatility_score': 50.0,
                'sentiment_score': 50.0,
                'risk_score': 50.0,
                'signal': 'NEUTRAL',
                'confidence': 0.0,
                'error': str(e)
            }

def add_regime_features(df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
    """Add regime features: volatility state, trend state, liquidity proxy."""
    if df.empty or price_col not in df.columns:
        return df

    out = df.copy()

    # Volatility state: rolling realized volatility
    out['ret_1'] = out[price_col].pct_change()
    out['vol_20'] = out['ret_1'].rolling(20).std() * np.sqrt(365)
    out['vol_state'] = pd.qcut(out['vol_20'].fillna(method='ffill'), q=3, labels=[0, 1, 2]).astype('Int64')

    # Trend state: normalized slope over 20 bars
    window = 20
    roll = out[price_col].rolling(window)
    out['trend_slope'] = roll.apply(lambda s: np.polyfit(np.arange(len(s)), s, 1)[0] if len(s.dropna()) == window else np.nan, raw=False)
    out['trend_strength'] = out['trend_slope'] / out[price_col].rolling(window).std()
    out['trend_state'] = np.select([out['trend_strength'] > 0.5, out['trend_strength'] < -0.5], [1, -1], default=0)

    # Liquidity proxy: volume relative to rolling median
    if 'volume' in out.columns:
        out['volume_rel'] = out['volume'] / out['volume'].rolling(20).median()
        out['liquidity_state'] = pd.qcut(out['volume_rel'].replace([np.inf, -np.inf], np.nan).fillna(1.0), q=3, labels=[0, 1, 2]).astype('Int64')

    # Clean
    out = out.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')
    return out
    
