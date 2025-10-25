import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import talib
from loguru import logger
from sklearn.preprocessing import RobustScaler
import joblib
import os

class CryptoFeatureEngineer:
    """
    Enhanced feature engineering for cryptocurrency trading with comprehensive technical indicators.
    Includes: MA, SMA, EMA, RSI, MACD, Bollinger Bands, Fibonacci Retracement, 
    ATR, Stochastic Oscillator, OBV, Ichimoku Cloud, and more.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the feature engineer with configuration.
        
        Args:
            config: Configuration dictionary containing parameters for indicators
        """
        self.config = config
        self.required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Technical indicator parameters
        self.ma_periods = config.get('ma_periods', [5, 10, 20, 50, 100, 200])
        self.rsi_periods = config.get('rsi_periods', [14, 21])
        self.macd_params = config.get('macd_params', {'fast': 12, 'slow': 26, 'signal': 9})
        self.bb_periods = config.get('bb_periods', [20, 50])
        self.stoch_params = config.get('stoch_params', {'k_period': 14, 'd_period': 3})
        self.atr_periods = config.get('atr_periods', [14, 21])
        self.ichimoku_params = config.get('ichimoku_params', {
            'tenkan': 9, 'kijun': 26, 'senkou_span_b': 52, 'displacement': 26
        })
        
        logger.info("CryptoFeatureEngineer initialized with comprehensive technical indicators")
        # Scaling artifacts
        self.scaler_dir = self.config.get('scaler_dir', 'feature_store/registry') if isinstance(self.config, dict) else 'feature_store/registry'
        os.makedirs(self.scaler_dir, exist_ok=True)
        self.scalers: Dict[str, RobustScaler] = {}
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate that the dataframe contains required columns.
        
        Args:
            df: Input dataframe
            
        Returns:
            bool: True if valid, False otherwise
        """
        missing_cols = [col for col in self.required_columns if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False
        return True
    
    def add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Simple Moving Averages (SMA) and Exponential Moving Averages (EMA).
        
        Args:
            df: Input dataframe with OHLCV data
            
        Returns:
            pd.DataFrame: Dataframe with MA features added
        """
        logger.info("Adding Moving Average features...")
        
        for period in self.ma_periods:
            # Simple Moving Average (SMA)
            df[f'sma_{period}'] = talib.SMA(df['close'], timeperiod=period)
            
            # Exponential Moving Average (EMA)
            df[f'ema_{period}'] = talib.EMA(df['close'], timeperiod=period)
            
            # Moving Average Crossovers
            if period > min(self.ma_periods):
                shorter_period = min([p for p in self.ma_periods if p < period])
                df[f'ma_cross_{shorter_period}_{period}'] = (
                    df[f'sma_{shorter_period}'] > df[f'sma_{period}']
                ).astype(int)
                
                df[f'ema_cross_{shorter_period}_{period}'] = (
                    df[f'ema_{shorter_period}'] > df[f'ema_{period}']
                ).astype(int)
        
        # Price vs MA ratios
        for period in self.ma_periods:
            df[f'price_sma_ratio_{period}'] = df['close'] / df[f'sma_{period}']
            df[f'price_ema_ratio_{period}'] = df['close'] / df[f'ema_{period}']
        
        return df
    
    def add_rsi_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Relative Strength Index (RSI) features.
        
        Args:
            df: Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with RSI features added
        """
        logger.info("Adding RSI features...")
        
        for period in self.rsi_periods:
            df[f'rsi_{period}'] = talib.RSI(df['close'], timeperiod=period)
            
            # RSI levels
            df[f'rsi_{period}_overbought'] = (df[f'rsi_{period}'] > 70).astype(int)
            df[f'rsi_{period}_oversold'] = (df[f'rsi_{period}'] < 30).astype(int)
            df[f'rsi_{period}_neutral'] = (
                (df[f'rsi_{period}'] >= 30) & (df[f'rsi_{period}'] <= 70)
            ).astype(int)
            
            # RSI momentum
            df[f'rsi_{period}_momentum'] = df[f'rsi_{period}'].diff()
            df[f'rsi_{period}_momentum_ma'] = df[f'rsi_{period}_momentum'].rolling(5).mean()
        
        return df
    
    def add_macd_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Moving Average Convergence Divergence (MACD) features.
        
        Args:
            df: Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with MACD features added
        """
        logger.info("Adding MACD features...")
        
        fast = self.macd_params['fast']
        slow = self.macd_params['slow']
        signal = self.macd_params['signal']
        
        # MACD line, signal line, and histogram
        df['macd'], df['macd_signal'], df['macd_histogram'] = talib.MACD(
            df['close'], fastperiod=fast, slowperiod=slow, signalperiod=signal
        )
        
        # MACD crossovers
        df['macd_bullish_cross'] = (df['macd'] > df['macd_signal']).astype(int)
        df['macd_bearish_cross'] = (df['macd'] < df['macd_signal']).astype(int)
        
        # MACD histogram changes
        df['macd_histogram_change'] = df['macd_histogram'].diff()
        df['macd_histogram_momentum'] = df['macd_histogram_change'].rolling(5).mean()
        
        # MACD divergence (simplified)
        df['macd_divergence'] = np.where(
            (df['close'] > df['close'].shift(1)) & (df['macd'] < df['macd'].shift(1)), -1,
            np.where(
                (df['close'] < df['close'].shift(1)) & (df['macd'] > df['macd'].shift(1)), 1, 0
            )
        )
        
        return df
    
    def add_bollinger_bands_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Bollinger Bands features.
        
        Args:
            df: Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with Bollinger Bands features added
        """
        logger.info("Adding Bollinger Bands features...")
        
        for period in self.bb_periods:
            upper, middle, lower = talib.BBANDS(
                df['close'], timeperiod=period, nbdevup=2, nbdevdn=2, matype=0
            )
            
            df[f'bb_upper_{period}'] = upper
            df[f'bb_middle_{period}'] = middle
            df[f'bb_lower_{period}'] = lower
            
            # Bollinger Bands width and %B
            df[f'bb_width_{period}'] = upper - lower
            df[f'bb_width_ratio_{period}'] = df[f'bb_width_{period}'] / middle
            
            df[f'bb_percent_b_{period}'] = (df['close'] - lower) / (upper - lower)
            
            # Price position relative to bands
            df[f'bb_position_{period}'] = np.where(
                df['close'] > upper, 1,  # Above upper band
                np.where(df['close'] < lower, -1, 0)  # Below lower band, between bands
            )
            
            # Squeeze indicator (simplified)
            df[f'bb_squeeze_{period}'] = (df[f'bb_width_ratio_{period}'] < 0.1).astype(int)
        
        return df
    
    def add_fibonacci_retracement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Fibonacci Retracement levels as features.
        
        Args:
            df: Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with Fibonacci features added
        """
        logger.info("Adding Fibonacci Retracement features...")
        
        # Calculate swing high and low over a rolling window
        window = 20
        
        # Rolling high and low
        df['rolling_high'] = df['high'].rolling(window=window).max()
        df['rolling_low'] = df['low'].rolling(window=window).min()
        
        # Fibonacci retracement levels
        df['fib_0'] = df['rolling_low']  # 0% retracement
        df['fib_236'] = df['rolling_low'] + 0.236 * (df['rolling_high'] - df['rolling_low'])  # 23.6%
        df['fib_382'] = df['rolling_low'] + 0.382 * (df['rolling_high'] - df['rolling_low'])  # 38.2%
        df['fib_500'] = df['rolling_low'] + 0.500 * (df['rolling_high'] - df['rolling_low'])  # 50%
        df['fib_618'] = df['rolling_low'] + 0.618 * (df['rolling_high'] - df['rolling_low'])  # 61.8%
        df['fib_786'] = df['rolling_low'] + 0.786 * (df['rolling_high'] - df['rolling_low'])  # 78.6%
        df['fib_100'] = df['rolling_high']  # 100% retracement
        
        # Price position relative to Fibonacci levels
        for level in ['236', '382', '500', '618', '786']:
            df[f'fib_position_{level}'] = np.where(
                df['close'] > df[f'fib_{level}'], 1, -1
            )
        
        # Fibonacci support/resistance zones
        df['fib_support_zone'] = np.where(
            (df['close'] >= df['fib_382']) & (df['close'] <= df['fib_500']), 1, 0
        )
        df['fib_resistance_zone'] = np.where(
            (df['close'] >= df['fib_500']) & (df['close'] <= df['fib_618']), 1, 0
        )
        
        return df
    
    def add_atr_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Average True Range (ATR) features.
        
        Args:
            df: Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with ATR features added
        """
        logger.info("Adding ATR features...")
        
        for period in self.atr_periods:
            df[f'atr_{period}'] = talib.ATR(
                df['high'], df['low'], df['close'], timeperiod=period
            )
            
            # ATR-based volatility indicators
            df[f'atr_volatility_{period}'] = df[f'atr_{period}'] / df['close']
            
            # ATR-based stop loss levels
            df[f'atr_stop_loss_long_{period}'] = df['close'] - (2 * df[f'atr_{period}'])
            df[f'atr_stop_loss_short_{period}'] = df['close'] + (2 * df[f'atr_{period}'])
            
            # ATR momentum
            df[f'atr_momentum_{period}'] = df[f'atr_{period}'].diff()
            df[f'atr_momentum_ma_{period}'] = df[f'atr_momentum_{period}'].rolling(5).mean()
        
        return df
    
    def add_stochastic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Stochastic Oscillator features.
        
        Args:
            df: Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with Stochastic features added
        """
        logger.info("Adding Stochastic Oscillator features...")
        
        k_period = self.stoch_params['k_period']
        d_period = self.stoch_params['d_period']
        
        # Stochastic %K and %D
        df['stoch_k'] = talib.STOCH(
            df['high'], df['low'], df['close'],
            fastk_period=k_period, slowk_period=3, slowk_matype=0,
            slowd_period=d_period, slowd_matype=0
        )[0]
        
        df['stoch_d'] = talib.STOCH(
            df['high'], df['low'], df['close'],
            fastk_period=k_period, slowk_period=3, slowk_matype=0,
            slowd_period=d_period, slowd_matype=0
        )[1]
        
        # Stochastic levels
        df['stoch_overbought'] = (df['stoch_k'] > 80).astype(int)
        df['stoch_oversold'] = (df['stoch_k'] < 20).astype(int)
        
        # Stochastic crossovers
        df['stoch_bullish_cross'] = (df['stoch_k'] > df['stoch_d']).astype(int)
        df['stoch_bearish_cross'] = (df['stoch_k'] < df['stoch_d']).astype(int)
        
        # Stochastic momentum
        df['stoch_k_momentum'] = df['stoch_k'].diff()
        df['stoch_d_momentum'] = df['stoch_d'].diff()
        
        return df
    
    def add_obv_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add On-Balance Volume (OBV) features.
        
        Args:
            df: Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with OBV features added
        """
        logger.info("Adding OBV features...")
        
        # Basic OBV
        df['obv'] = talib.OBV(df['close'], df['volume'])
        
        # OBV moving averages
        for period in [5, 10, 20]:
            df[f'obv_ma_{period}'] = df['obv'].rolling(window=period).mean()
        
        # OBV momentum
        df['obv_momentum'] = df['obv'].diff()
        df['obv_momentum_ma'] = df['obv_momentum'].rolling(5).mean()
        
        # OBV divergence
        df['obv_divergence'] = np.where(
            (df['close'] > df['close'].shift(1)) & (df['obv'] < df['obv'].shift(1)), -1,
            np.where(
                (df['close'] < df['close'].shift(1)) & (df['obv'] > df['obv'].shift(1)), 1, 0
            )
        )
        
        # Volume-price trend
        df['vpt'] = df['volume'] * ((df['close'] - df['close'].shift(1)) / df['close'].shift(1))
        df['vpt_cumulative'] = df['vpt'].cumsum()
        
        return df
    
    def add_ichimoku_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Ichimoku Cloud features.
        
        Args:
            df: Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with Ichimoku features added
        """
        logger.info("Adding Ichimoku Cloud features...")
        
        tenkan = self.ichimoku_params['tenkan']
        kijun = self.ichimoku_params['kijun']
        senkou_span_b = self.ichimoku_params['senkou_span_b']
        displacement = self.ichimoku_params['displacement']
        
        # Tenkan-sen (Conversion Line)
        high_tenkan = df['high'].rolling(window=tenkan).max()
        low_tenkan = df['low'].rolling(window=tenkan).min()
        df['ichimoku_tenkan'] = (high_tenkan + low_tenkan) / 2
        
        # Kijun-sen (Base Line)
        high_kijun = df['high'].rolling(window=kijun).max()
        low_kijun = df['low'].rolling(window=kijun).min()
        df['ichimoku_kijun'] = (high_kijun + low_kijun) / 2
        
        # Senkou Span A (Leading Span A)
        df['ichimoku_senkou_a'] = ((df['ichimoku_tenkan'] + df['ichimoku_kijun']) / 2).shift(displacement)
        
        # Senkou Span B (Leading Span B)
        high_senkou_b = df['high'].rolling(window=senkou_span_b).max()
        low_senkou_b = df['low'].rolling(window=senkou_span_b).min()
        df['ichimoku_senkou_b'] = ((high_senkou_b + low_senkou_b) / 2).shift(displacement)
        
        # Chikou Span (Lagging Span)
        df['ichimoku_chikou'] = df['close'].shift(-displacement)
        
        # Ichimoku signals
        df['ichimoku_bullish'] = (
            (df['close'] > df['ichimoku_senkou_a']) & 
            (df['close'] > df['ichimoku_senkou_b']) &
            (df['ichimoku_tenkan'] > df['ichimoku_kijun'])
        ).astype(int)
        
        df['ichimoku_bearish'] = (
            (df['close'] < df['ichimoku_senkou_a']) & 
            (df['close'] < df['ichimoku_senkou_b']) &
            (df['ichimoku_tenkan'] < df['ichimoku_kijun'])
        ).astype(int)
        
        # Cloud color (green = bullish, red = bearish)
        df['ichimoku_cloud_color'] = np.where(
            df['ichimoku_senkou_a'] > df['ichimoku_senkou_b'], 1, -1
        )
        
        # Price vs cloud position
        df['ichimoku_cloud_position'] = np.where(
            df['close'] > df[['ichimoku_senkou_a', 'ichimoku_senkou_b']].max(axis=1), 1,  # Above cloud
            np.where(
                df['close'] < df[['ichimoku_senkou_a', 'ichimoku_senkou_b']].min(axis=1), -1,  # Below cloud
                0  # Inside cloud
            )
        )
        
        return df
    
    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add price-based features.
        
        Args:
            df: Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with price features added
        """
        logger.info("Adding price-based features...")
        
        # Price changes
        for period in [1, 3, 5, 10]:
            df[f'price_change_{period}'] = df['close'].pct_change(periods=period)
            df[f'price_change_abs_{period}'] = df['close'].pct_change(periods=period).abs()
        
        # High-Low range
        df['hl_range'] = (df['high'] - df['low']) / df['close']
        df['hl_range_ma'] = df['hl_range'].rolling(20).mean()
        
        # Gap analysis
        df['gap_up'] = (df['open'] > df['close'].shift(1)).astype(int)
        df['gap_down'] = (df['open'] < df['close'].shift(1)).astype(int)
        df['gap_size'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        
        # Support and resistance levels (simplified)
        df['support_level'] = df['low'].rolling(20).min()
        df['resistance_level'] = df['high'].rolling(20).max()
        
        # Price position relative to support/resistance
        df['price_vs_support'] = (df['close'] - df['support_level']) / df['support_level']
        df['price_vs_resistance'] = (df['resistance_level'] - df['close']) / df['close']
        
        return df
    
    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume-based features.
        
        Args:
            df: Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with volume features added
        """
        logger.info("Adding volume-based features...")
        
        # Volume moving averages
        for period in [5, 10, 20]:
            df[f'volume_ma_{period}'] = df['volume'].rolling(window=period).mean()
            df[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_ma_{period}']
        
        # Volume momentum
        df['volume_momentum'] = df['volume'].diff()
        df['volume_momentum_ma'] = df['volume_momentum'].rolling(5).mean()
        
        # Volume-price relationship
        df['volume_price_trend'] = df['volume'] * df['price_change_1']
        df['volume_price_trend_ma'] = df['volume_price_trend'].rolling(10).mean()
        
        # Volume spikes
        df['volume_spike'] = (df['volume'] > df['volume'].rolling(20).quantile(0.9)).astype(int)
        
        # Accumulation/Distribution Line
        df['ad_line'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
        df['ad_line_momentum'] = df['ad_line'].diff()
        
        return df
    
    def add_sentiment_features(self, df: pd.DataFrame, sentiment_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Add sentiment-based features.
        
        Args:
            df: Input dataframe
            sentiment_data: Optional dataframe with sentiment scores
            
        Returns:
            pd.DataFrame: Dataframe with sentiment features added
        """
        logger.info("Adding sentiment features...")
        
        if sentiment_data is not None:
            # Merge sentiment data
            sentiment_cols = [col for col in sentiment_data.columns if 'sentiment' in col.lower()]
            for col in sentiment_cols:
                df[f'sentiment_{col}'] = sentiment_data[col]
                
                # Sentiment momentum
                df[f'sentiment_{col}_momentum'] = sentiment_data[col].diff()
                df[f'sentiment_{col}_ma'] = sentiment_data[col].rolling(5).mean()
        
        # Price-sentiment divergence (if sentiment data available)
        if 'sentiment_score' in df.columns:
            df['price_sentiment_divergence'] = np.where(
                (df['price_change_1'] > 0) & (df['sentiment_score'] < 0), -1,
                np.where(
                    (df['price_change_1'] < 0) & (df['sentiment_score'] > 0), 1, 0
                )
            )
        
        return df
    
    def add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add lagged features for time series analysis.
        
        Args:
            df: Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with lag features added
        """
        logger.info("Adding lag features...")
        
        # Lag important features
        important_features = ['close', 'volume', 'rsi_14', 'macd', 'bb_width_20']
        
        for feature in important_features:
            if feature in df.columns:
                for lag in [1, 2, 3, 5]:
                    df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
        
        return df
    
    def add_rolling_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add rolling statistics features.
        
        Args:
            df: Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with rolling statistics added
        """
        logger.info("Adding rolling statistics features...")
        
        # Rolling statistics for key features
        features_for_stats = ['close', 'volume', 'rsi_14']
        
        for feature in features_for_stats:
            if feature in df.columns:
                for window in [5, 10, 20]:
                    df[f'{feature}_rolling_mean_{window}'] = df[feature].rolling(window).mean()
                    df[f'{feature}_rolling_std_{window}'] = df[feature].rolling(window).std()
                    df[f'{feature}_rolling_min_{window}'] = df[feature].rolling(window).min()
                    df[f'{feature}_rolling_max_{window}'] = df[feature].rolling(window).max()
                    
                    # Z-score
                    df[f'{feature}_zscore_{window}'] = (
                        (df[feature] - df[f'{feature}_rolling_mean_{window}']) / 
                        df[f'{feature}_rolling_std_{window}']
                    )
        
        return df
    
    def engineer_features(self, df: pd.DataFrame, sentiment_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Main method to engineer all features.
        
        Args:
            df: Input dataframe with OHLCV data
            sentiment_data: Optional dataframe with sentiment scores
            
        Returns:
            pd.DataFrame: Dataframe with all engineered features
        """
        logger.info("Starting comprehensive feature engineering...")
        
        if df is None or df.empty:
            logger.error("Input dataframe is None or empty")
            return pd.DataFrame()
        
        if not self.validate_data(df):
            logger.error("Invalid data format. Required columns: OHLCV")
            return pd.DataFrame()
        
        # Make a copy to avoid modifying original
        df_features = df.copy()
        
        try:
            # Add all technical indicators
            df_features = self.add_moving_averages(df_features)
            df_features = self.add_rsi_features(df_features)
            df_features = self.add_macd_features(df_features)
            df_features = self.add_bollinger_bands_features(df_features)
            df_features = self.add_fibonacci_retracement_features(df_features)
            df_features = self.add_atr_features(df_features)
            df_features = self.add_stochastic_features(df_features)
            df_features = self.add_obv_features(df_features)
            df_features = self.add_ichimoku_features(df_features)
            
            # Add derived features
            df_features = self.add_price_features(df_features)
            df_features = self.add_volume_features(df_features)
            df_features = self.add_sentiment_features(df_features, sentiment_data)
            df_features = self.add_lag_features(df_features)
            df_features = self.add_rolling_statistics(df_features)
            
            # Clean up infinite and NaN values
            df_features = df_features.replace([np.inf, -np.inf], np.nan)
            df_features = df_features.fillna(method='ffill').fillna(method='bfill')
            
            # Remove columns with too many NaN values
            nan_threshold = 0.1
            nan_counts = df_features.isnull().sum() / len(df_features)
            columns_to_drop = nan_counts[nan_counts > nan_threshold].index
            df_features = df_features.drop(columns=columns_to_drop)

            # Winsorize selected numeric features to reduce tail risk
            numeric_cols = [c for c in df_features.columns if df_features[c].dtype != 'O' and c not in ['timestamp']]
            for col in numeric_cols:
                q_low = df_features[col].quantile(0.001)
                q_high = df_features[col].quantile(0.999)
                df_features[col] = df_features[col].clip(lower=q_low, upper=q_high)

            # Robust scale per symbol to avoid leakage across instruments
            if 'symbol' in df_features.columns:
                scaled_parts = []
                for symbol, g in df_features.groupby('symbol'):
                    g_scaled = g.copy()
                    scaler_path = os.path.join(self.scaler_dir, f"robust_scaler_{symbol}.joblib")
                    # Fit or load scaler; if loaded scaler's feature space mismatches, refit
                    scaler: RobustScaler
                    if os.path.exists(scaler_path):
                        scaler = joblib.load(scaler_path)
                    else:
                        scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(5.0, 95.0))
                        scaler.fit(g[numeric_cols])
                        joblib.dump(scaler, scaler_path)

                    try:
                        g_scaled[numeric_cols] = scaler.transform(g[numeric_cols])
                    except Exception:
                        # Columns changed since last fit; refit and overwrite saved scaler
                        scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(5.0, 95.0))
                        scaler.fit(g[numeric_cols])
                        joblib.dump(scaler, scaler_path)
                        g_scaled[numeric_cols] = scaler.transform(g[numeric_cols])

                    scaled_parts.append(g_scaled)
                df_features = pd.concat(scaled_parts, ignore_index=True).sort_values('timestamp')
            
            logger.info(f"Feature engineering completed. Total features: {len(df_features.columns)}")
            logger.info(f"Features added: {[col for col in df_features.columns if col not in df.columns]}")
            
            return df_features
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            logger.error(f"Returning empty DataFrame due to feature engineering failure")
            return pd.DataFrame()
    
    def get_feature_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get a summary of engineered features.
        
        Args:
            df: Dataframe with engineered features
            
        Returns:
            Dict: Summary of features by category
        """
        feature_categories = {
            'Moving Averages': [col for col in df.columns if 'sma_' in col or 'ema_' in col],
            'RSI': [col for col in df.columns if 'rsi_' in col],
            'MACD': [col for col in df.columns if 'macd' in col],
            'Bollinger Bands': [col for col in df.columns if 'bb_' in col],
            'Fibonacci': [col for col in df.columns if 'fib_' in col],
            'ATR': [col for col in df.columns if 'atr_' in col],
            'Stochastic': [col for col in df.columns if 'stoch_' in col],
            'OBV': [col for col in df.columns if 'obv' in col],
            'Ichimoku': [col for col in df.columns if 'ichimoku_' in col],
            'Price Features': [col for col in df.columns if 'price_' in col or 'gap_' in col],
            'Volume Features': [col for col in df.columns if 'volume_' in col],
            'Sentiment': [col for col in df.columns if 'sentiment_' in col],
            'Lag Features': [col for col in df.columns if 'lag_' in col],
            'Rolling Stats': [col for col in df.columns if 'rolling_' in col or 'zscore_' in col]
        }
        
        summary = {
            'total_features': len(df.columns),
            'original_features': len([col for col in df.columns if col in ['open', 'high', 'low', 'close', 'volume']]),
            'engineered_features': len(df.columns) - 5,
            'feature_categories': feature_categories,
            'feature_counts': {cat: len(features) for cat, features in feature_categories.items()}
        }
        
        return summary

# Example usage
def main():
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'symbol': ['BTC'] * 100,
        'open': np.random.uniform(40000, 50000, 100),
        'high': np.random.uniform(40000, 50000, 100),
        'low': np.random.uniform(40000, 50000, 100),
        'close': np.random.uniform(40000, 50000, 100),
        'volume': np.random.uniform(1000, 5000, 100)
    })
    
    # Create sample sentiment data
    sentiment_data = pd.DataFrame({
        'timestamp': dates,
        'symbol': ['BTC'] * 100,
        'sentiment_score': np.random.uniform(-1, 1, 100),
        'confidence': np.random.uniform(0.5, 1.0, 100)
    })
    
    # Initialize feature engineer
    engineer = CryptoFeatureEngineer(config={
        'ma_periods': [5, 10, 20, 50, 100, 200],
        'rsi_periods': [14, 21],
        'macd_params': {'fast': 12, 'slow': 26, 'signal': 9},
        'bb_periods': [20, 50],
        'stoch_params': {'k_period': 14, 'd_period': 3},
        'atr_periods': [14, 21],
        'ichimoku_params': {
            'tenkan': 9, 'kijun': 26, 'senkou_span_b': 52, 'displacement': 26
        }
    })
    
    # Create all features
    features_df = engineer.engineer_features(sample_data, sentiment_data)
    
    print(f"Original data shape: {sample_data.shape}")
    print(f"Features data shape: {features_df.shape}")
    print(f"Number of features: {len(features_df.columns)}")
    
    # Get feature importance
    # The original code had a get_feature_importance_ranking method, but it's not in the new code.
    # For now, we'll just print the summary.
    summary = engineer.get_feature_summary(features_df)
    print("\nFeature Summary:")
    print(summary)

if __name__ == "__main__":
    main()
