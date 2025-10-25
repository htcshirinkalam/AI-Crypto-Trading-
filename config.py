import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Secure configuration for public deployment - Read-only access"""

    # APILayer Configuration (Primary API access method)
    APILAYER_API_KEY = os.getenv('APILAYER_API_KEY', '')
    USE_APILAYER = bool(APILAYER_API_KEY and APILAYER_API_KEY.strip())

    # Fallback API Keys (used when APILayer is not available)
    COINGECKO_API_KEY = os.getenv('COINGECKO_API_KEY', '')
    CRYPTOCOMPARE_API_KEY = os.getenv('CRYPTOCOMPARE_API_KEY', '')
    # Note: Trading API keys removed for security
    
    # Data Collection Settings
    UPDATE_INTERVAL = 300  # 5 minutes
    MAX_RETRIES = 5
    REQUEST_TIMEOUT = 60
    
    # Core Cryptocurrencies (always included)
    CORE_CRYPTO_SYMBOLS = ['BTC', 'ETH', 'BNB', 'SOL']

    # Dynamic selection will add top 5 most volatile cryptos
    CRYPTO_SYMBOLS = CORE_CRYPTO_SYMBOLS.copy()  # Will be updated dynamically
    
    # Time Frames for Analysis
    TIME_FRAMES = ['1m', '5m', '15m', '1h', '4h', '1d', '1w']
    
    # Sentiment Analysis Settings
    SENTIMENT_WEIGHTS = {
        'news': 0.4,
        'social_media': 0.3,
        'technical_indicators': 0.3
    }

    # Composite Scoring Weights (used by recommendations)
    COMPOSITE_WEIGHTS = {
        'opportunity_signal_strength': 0.4,
        'opportunity_confidence': 0.3,
        'opportunity_risk_inverse': 0.3,  # (1 - normalized_risk)
        'sell_signal_strength': 0.4,
        'sell_risk': 0.2,
        'sell_momentum': 0.4
    }

    # Signal Thresholds
    SIGNAL_THRESHOLDS = {
        'min_confidence': 0.7,
        'min_signal_strength': 0.2,
        'max_risk_score': 2.5  # lower is safer (1..3 scale)
    }
    
    # Technical Indicators
    TECHNICAL_INDICATORS = {
        # Moving Averages
        'ma_periods': [5, 10, 20, 50, 100, 200],
        
        # RSI
        'rsi_periods': [14, 21],
        
        # MACD
        'macd_params': {
            'fast': 12,
            'slow': 26,
            'signal': 9
        },
        
        # Bollinger Bands
        'bb_periods': [20, 50],
        'bb_std_dev': 2,
        
        # Stochastic Oscillator
        'stoch_params': {
            'k_period': 14,
            'd_period': 3
        },
        
        # Average True Range (ATR)
        'atr_periods': [14, 21],
        
        # Ichimoku Cloud
        'ichimoku_params': {
            'tenkan': 9,
            'kijun': 26,
            'senkou_span_b': 52,
            'displacement': 26
        },
        
        # Fibonacci Retracement
        'fibonacci_levels': [0, 0.236, 0.382, 0.500, 0.618, 0.786, 1.0],
        'fibonacci_window': 20,
        
        # Volume Analysis
        'volume_ma_periods': [5, 10, 20],
        'volume_spike_threshold': 0.9,  # 90th percentile
        
        # Price Analysis
        'price_change_periods': [1, 3, 5, 10],
        'support_resistance_window': 20,
        
        # Lag Features
        'lag_periods': [1, 2, 3, 5],
        
        # Rolling Statistics
        'rolling_windows': [5, 10, 20],
        'zscore_threshold': 2.0
    }
    
    # Trading Strategy Parameters
    TRADING_PARAMS = {
        'min_confidence': 0.7,
        'stop_loss_percentage': 0.05,
        'take_profit_percentage': 0.15,
        'max_position_size': 0.1,  # 10% of portfolio
        'risk_reward_ratio': 2.0
    }
    
    # Model Parameters
    MODEL_PARAMS = {
        'lstm_units': 128,
        'lstm_dropout': 0.2,
        'cnn_filters': 64,
        'cnn_kernel_size': 3,
        'random_forest_n_estimators': 100,
        'decision_tree_max_depth': 10
    }
    
    # Database Settings
    DATABASE_URL = 'sqlite:///crypto_trading.db'

    # Database configuration for DatabaseManager
    DATABASE = {
        'type': 'sqlite',
        'sqlite_path': 'data/crypto_trading.db'
    }
    
    # Logging
    LOG_LEVEL = 'INFO'
    LOG_FILE = 'crypto_trading.log'
    
    # News Sources (prefer accessible sources)
    NEWS_SOURCES = [
        'cryptonews.com',
        'coincodex.com',
        'coinmarketcap.com'
    ]

    
    # Social Media Sources
    SOCIAL_SOURCES = [
        'twitter.com',
        'reddit.com/r/cryptocurrency',
        'reddit.com/r/bitcoin'
    ]

    # Risk Management (Conservative settings for public access)
    PORTFOLIO_RISK_LIMIT = 0.03  # 3% max portfolio risk (reduced for safety)
    MAX_PORTFOLIO_DRAWDOWN = 0.15  # 15% max drawdown (reduced for safety)
    MAX_ASSET_CONCENTRATION = 0.25  # 25% max concentration (reduced for safety)
    CORRELATION_THRESHOLD = 0.8  # Correlation threshold for hedging

    # API Server Configuration
    API_HOST = "0.0.0.0"
    API_PORT = 8000

    def __getitem__(self, key):
        """Allow dict-like access to attributes"""
        return getattr(self, key)

    def get(self, key, default=None):
        """Allow dict-like get method"""
        return getattr(self, key, default)
