import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import re
import nltk
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import torch
from loguru import logger
from config import Config
import warnings
warnings.filterwarnings('ignore')
import time
import os

# Ensure TensorFlow uses the legacy Keras implementation when available to avoid
# incompatibilities with standalone Keras 3.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")


def _looks_like_keras_compat_error(exc: Exception) -> bool:
    """Detect the common Transformers runtime error caused by Keras 3."""
    msg = str(exc)
    return any(token in msg for token in (
        "Your currently installed version of Keras is Keras 3",
        "backwards-compatible tf-keras",
        "Please install the backwards-compatible tf-keras",
    ))


def _force_transformers_to_use_pytorch() -> bool:
    """Force Transformers to skip TensorFlow imports when PyTorch is available."""
    current = os.environ.get("USE_TF", "AUTO").strip().lower()
    if current in {"0", "false", "no"}:
        return False
    os.environ["USE_TF"] = "0"
    # Align with Transformers' documented env var for selecting torch backend
    os.environ.setdefault("TRANSFORMERS_BACKEND", "torch")
    # Silence the warning that Transformers emits when TensorFlow support is disabled
    os.environ.setdefault("TRANSFORMERS_NO_TF_WARNING", "1")
    return True

# Simple in-memory cache for sentiment results to avoid repeated heavy inference
_SENTIMENT_CACHE = {
    # key: (timestamp, result_dict)
}
_SENTIMENT_TTL = 300  # seconds

# We perform lazy import of transformers inside setup_bert().
# If transformers is missing, setup_bert will log a pip command using the
# active Python executable so users can install into the correct venv.


# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

class CryptoSentimentAnalyzer:
    def __init__(self):
        self.config = Config()
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.bert_analyzer = None
        self.finbert_analyzer = None  # Financial BERT (BloombergGPT alternative)
        # instance-level flag for transformers availability
        self.transformers_available = False
        self.setup_bert()
        self.setup_finbert()  # Setup FinBERT for financial sentiment
        self.setup_custom_lexicon()
        
    def setup_bert(self):
        """Initialize BERT model for sentiment analysis using lazy imports.

        This will try to import the `transformers` library at runtime. If the
        import fails, we leave `self.bert_analyzer` as None and provide a clear
        log message explaining how to install the dependency.
        """

        # Try lazy import so the module can be used without transformers installed
        try:
            from transformers import pipeline as hf_pipeline, AutoTokenizer, AutoModelForSequenceClassification
        except ImportError:
            # Use sys.executable so instructions point to the active Python interpreter
            import sys
            pip_cmd = f'"{sys.executable}" -m pip install "transformers[torch]" torch'
            logger.warning(
                "Transformers library not available, skipping BERT setup. "
                f"Install into the active Python environment with: {pip_cmd} or see requirements.txt"
            )
            self.bert_analyzer = None
            self.transformers_available = False
            return
        except Exception as e:
            if _looks_like_keras_compat_error(e):
                forced = _force_transformers_to_use_pytorch()
                if forced:
                    logger.warning(
                        "Detected Keras 3 compatibility issue. Disabled TensorFlow backend for transformers and "
                        "will retry using PyTorch-only pipeline."
                    )
                    try:
                        from transformers import pipeline as hf_pipeline, AutoTokenizer, AutoModelForSequenceClassification
                    except Exception as retry_exc:
                        logger.warning(f"Retrying transformers import after disabling TensorFlow failed: {retry_exc}")
                        self.bert_analyzer = None
                        self.transformers_available = False
                        return
                else:
                    logger.warning(
                        "Transformers import failed due to Keras compatibility issue, but TensorFlow backend was already disabled. "
                        f"Error: {e}"
                    )
                    self.bert_analyzer = None
                    self.transformers_available = False
                    return
            else:
                logger.warning(f"Unexpected error importing transformers: {e}")
                self.bert_analyzer = None
                self.transformers_available = False
                return

        self.transformers_available = True
        logger.info("Transformers library available; attempting to load BERT model")

        # Try to load the model with retries and fallbacks to smaller models if network/download fails
        import os
        # Allow user to point to a local safetensors/bin file via SENTIMENT_MODEL_PATH
        local_model_path = os.environ.get('SENTIMENT_MODEL_PATH', '').strip()
        if local_model_path:
            try:
                lm_path = os.path.expanduser(local_model_path)
                if os.path.exists(lm_path):
                    logger.info(f"Local model path provided: {lm_path}. Attempting to load local weights.")
                    # Determine loader based on file extension
                    ext = os.path.splitext(lm_path)[1].lower()
                    # Decide base architecture to instantiate (allow override via SENTIMENT_MODEL)
                    base_model = os.environ.get('SENTIMENT_MODEL', 'distilbert-base-uncased')
                    try:
                        # Import here to avoid top-level heavy deps when not used
                        from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
                        # load tokenizer/config (from hub or cache)
                        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
                        config = AutoConfig.from_pretrained(base_model)
                        # instantiate model from config (random init)
                        model = AutoModelForSequenceClassification.from_config(config)
                        # load weights from safetensors or torch .bin
                        try:
                            if ext == '.safetensors':
                                try:
                                    from safetensors.torch import load_file as safeload
                                    state_dict = safeload(lm_path)
                                except Exception:
                                    # fallback to huggingface safe loader
                                    import safetensors
                                    from safetensors.torch import load_file as safeload
                                    state_dict = safeload(lm_path)
                            else:
                                # assume a PyTorch .bin checkpoint
                                import torch as _torch
                                state_dict = _torch.load(lm_path, map_location='cpu')
                        except Exception as e:
                            logger.warning(f"Failed to load state dict from local file {lm_path}: {e}")
                            state_dict = None

                        if state_dict is not None:
                            try:
                                # Try to load state dict into model (allow partial loads)
                                model.load_state_dict(state_dict, strict=False)
                                self.bert_tokenizer = tokenizer
                                self.bert_model = model
                                # create pipeline
                                from transformers import pipeline as hf_pipeline
                                self.bert_analyzer = hf_pipeline('sentiment-analysis', model=self.bert_model, tokenizer=self.bert_tokenizer)
                                logger.info('Loaded local safetensors/binary weights into model successfully')
                                loaded = True
                            except Exception as e:
                                logger.warning(f"Failed to assign local weights to model: {e}")
                        else:
                            logger.warning('No state_dict available from local model file')
                    except Exception as e:
                        logger.warning(f"Local model load path encountered error: {e}")
                else:
                    logger.warning(f"SENTIMENT_MODEL_PATH set but file does not exist: {lm_path}")
            except Exception as e:
                logger.warning(f"Error processing SENTIMENT_MODEL_PATH: {e}")
        model_env = os.environ.get('SENTIMENT_MODEL', '').strip()
        candidates = [model_env] if model_env else []
        # Preferred lightweight models (order matters)
        candidates += [
            'distilbert-base-uncased-finetuned-sst-2-english',
            'cardiffnlp/twitter-roberta-base-sentiment',
            'nlptown/bert-base-multilingual-uncased-sentiment'
        ]

        loaded = False
        for model_name in candidates:
            if not model_name:
                continue
            # Try several attempts with exponential backoff
            for attempt in range(1, 4):
                try:
                    logger.info(f"Attempting to load sentiment model '{model_name}' (attempt {attempt})")
                    self.bert_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
                    self.bert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
                    self.bert_analyzer = hf_pipeline("sentiment-analysis", model=self.bert_model, tokenizer=self.bert_tokenizer)
                    logger.info(f"BERT model '{model_name}' loaded successfully")
                    loaded = True
                    break
                except Exception as e:
                    logger.warning(f"Loading model '{model_name}' failed on attempt {attempt}: {e}")
                    # Short backoff - increase on each attempt
                    try:
                        time.sleep(min(10, 2 ** attempt))
                    except Exception:
                        pass
            if loaded:
                break

            # If network downloads failed, try to load from local cache only before moving to next candidate
            try:
                logger.info(f"Attempting local-only load for '{model_name}'")
                self.bert_tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True, use_fast=True)
                self.bert_model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True)
                self.bert_analyzer = hf_pipeline("sentiment-analysis", model=self.bert_model, tokenizer=self.bert_tokenizer)
                logger.info(f"BERT model '{model_name}' loaded from local files")
                loaded = True
                break
            except Exception as e:
                logger.warning(f"Local-only load for '{model_name}' also failed: {e}")
                # continue to next candidate

        if not loaded:
            logger.warning(
                "All attempts to load a BERT sentiment model failed. "
                "This can happen due to network errors, proxy/tunnel issues, or missing cached files. "
                "You can:\n"
                " - Ensure you have internet access and retry, or\n"
                " - Pre-download a model to the local Hugging Face cache, or\n"
                " - Set the environment variable SENTIMENT_MODEL to a smaller model you prefer."
            )
            self.bert_analyzer = None
            # keep transformers_available True since import succeeded, but model is not ready
    
    def setup_finbert(self):
        """Initialize FinBERT model for financial sentiment analysis.
        
        FinBERT is a BERT model fine-tuned on financial texts, providing superior
        sentiment analysis for financial news and crypto market commentary.
        This serves as the open-source alternative to proprietary BloombergGPT.
        """
        if not self.transformers_available:
            logger.info("Transformers not available, skipping FinBERT setup")
            return
        
        try:
            from transformers import pipeline as hf_pipeline, AutoTokenizer, AutoModelForSequenceClassification
            
            # FinBERT model candidates (in order of preference)
            finbert_candidates = [
                'ProsusAI/finbert',  # Primary FinBERT model
                'yiyanghkust/finbert-tone',  # Alternative FinBERT for tone analysis
                'ahmedrachid/FinancialBERT-Sentiment-Analysis'  # Backup financial BERT
            ]
            
            loaded = False
            for model_name in finbert_candidates:
                try:
                    logger.info(f"Attempting to load FinBERT model: {model_name}")
                    
                    # Try to load with retries
                    for attempt in range(1, 3):
                        try:
                            finbert_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
                            finbert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
                            self.finbert_analyzer = hf_pipeline(
                                "sentiment-analysis",
                                model=finbert_model,
                                tokenizer=finbert_tokenizer
                            )
                            logger.info(f"FinBERT model '{model_name}' loaded successfully")
                            loaded = True
                            break
                        except Exception as e:
                            if attempt < 2:
                                logger.warning(f"Attempt {attempt} failed for {model_name}: {e}")
                                time.sleep(2 ** attempt)
                            else:
                                raise
                    
                    if loaded:
                        break
                        
                except Exception as e:
                    logger.warning(f"Failed to load FinBERT model '{model_name}': {e}")
                    # Try local-only load as fallback
                    try:
                        finbert_tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True, use_fast=True)
                        finbert_model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True)
                        self.finbert_analyzer = hf_pipeline(
                            "sentiment-analysis",
                            model=finbert_model,
                            tokenizer=finbert_tokenizer
                        )
                        logger.info(f"FinBERT model '{model_name}' loaded from local cache")
                        loaded = True
                        break
                    except Exception as local_e:
                        logger.warning(f"Local load also failed for '{model_name}': {local_e}")
                        continue
            
            if not loaded:
                logger.warning(
                    "Could not load FinBERT model. Financial sentiment analysis will fall back to "
                    "general BERT model. For better financial text analysis, ensure network access "
                    "or pre-download FinBERT models to local cache."
                )
                self.finbert_analyzer = None
        
        except Exception as e:
            logger.error(f"Error setting up FinBERT: {e}")
            self.finbert_analyzer = None
    
    def setup_custom_lexicon(self):
        """Setup custom cryptocurrency lexicon for VADER"""
        try:
            # Crypto-specific positive words
            crypto_positive = {
                'bullish': 2.0, 'moon': 2.0, 'pump': 1.5, 'rally': 1.5,
                'breakout': 1.5, 'surge': 1.5, 'adoption': 1.0, 'partnership': 1.0,
                'upgrade': 1.0, 'innovation': 1.0, 'decentralized': 0.5,
                'blockchain': 0.5, 'defi': 0.5, 'nft': 0.5
            }
            
            # Crypto-specific negative words
            crypto_negative = {
                'bearish': -2.0, 'dump': -1.5, 'crash': -2.0, 'selloff': -1.5,
                'fud': -1.5, 'scam': -2.0, 'hack': -2.0, 'regulation': -0.5,
                'ban': -1.5, 'restriction': -1.0, 'volatility': -0.5
            }
            
            # Update VADER lexicon
            self.vader_analyzer.lexicon.update(crypto_positive)
            self.vader_analyzer.lexicon.update(crypto_negative)
            
            logger.info("Custom crypto lexicon added to VADER")
            
        except Exception as e:
            logger.warning(f"Failed to setup custom lexicon: {e}")
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for sentiment analysis"""
        if not isinstance(text, str):
            return ""
        
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Remove URLs
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
            
            # Remove special characters but keep crypto symbols
            text = re.sub(r'[^\w\s#@$%&*+-]', ' ', text)
            
            # Remove extra whitespace
            text = ' '.join(text.split())
            
            return text
            
        except Exception as e:
            logger.error(f"Error preprocessing text: {e}")
            return ""
    
    def analyze_vader_sentiment(self, text: str) -> Dict:
        """Analyze sentiment using VADER"""
        try:
            processed_text = self.preprocess_text(text)
            if not processed_text:
                return {'compound': 0, 'pos': 0, 'neg': 0, 'neu': 0}
            
            scores = self.vader_analyzer.polarity_scores(processed_text)
            return scores
            
        except Exception as e:
            logger.error(f"Error in VADER analysis: {e}")
            return {'compound': 0, 'pos': 0, 'neg': 0, 'neu': 0}
    
    def analyze_textblob_sentiment(self, text: str) -> Dict:
        """Analyze sentiment using TextBlob"""
        try:
            processed_text = self.preprocess_text(text)
            if not processed_text:
                return {'polarity': 0, 'subjectivity': 0}
            
            blob = TextBlob(processed_text)
            return {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
            
        except Exception as e:
            logger.error(f"Error in TextBlob analysis: {e}")
            return {'polarity': 0, 'subjectivity': 0}
    
    def analyze_bert_sentiment(self, text: str) -> Dict:
        """Analyze sentiment using BERT"""
        try:
            if not self.transformers_available or not self.bert_analyzer:
                return {'label': 'NEUTRAL', 'score': 0.5}

            processed_text = self.preprocess_text(text)
            if not processed_text:
                return {'label': 'NEUTRAL', 'score': 0.5}

            # Truncate text if too long for BERT
            if len(processed_text) > 500:
                processed_text = processed_text[:500]

            result = self.bert_analyzer(processed_text)[0]

            # Map BERT labels to sentiment scores
            label_mapping = {
                'LABEL_0': 'NEGATIVE',
                'LABEL_1': 'NEUTRAL',
                'LABEL_2': 'POSITIVE'
            }

            return {
                'label': label_mapping.get(result['label'], 'NEUTRAL'),
                'score': result['score']
            }

        except Exception as e:
            logger.error(f"Error in BERT analysis: {e}")
            return {'label': 'NEUTRAL', 'score': 0.5}
    
    def analyze_finbert_sentiment(self, text: str) -> Dict:
        """Analyze sentiment using FinBERT (Financial BERT).
        
        FinBERT provides superior financial sentiment analysis compared to general
        BERT models, making it ideal for crypto and financial news analysis.
        This is the open-source alternative to proprietary BloombergGPT.
        """
        try:
            if not self.finbert_analyzer:
                return {'label': 'NEUTRAL', 'score': 0.5, 'available': False}
            
            processed_text = self.preprocess_text(text)
            if not processed_text:
                return {'label': 'NEUTRAL', 'score': 0.5, 'available': True}
            
            # Truncate text if too long for FinBERT (512 tokens max)
            if len(processed_text) > 500:
                processed_text = processed_text[:500]
            
            result = self.finbert_analyzer(processed_text)[0]
            
            # FinBERT returns 'positive', 'negative', 'neutral' (lowercase)
            # Standardize to uppercase for consistency
            label = result['label'].upper()
            
            return {
                'label': label,
                'score': result['score'],
                'available': True
            }
        
        except Exception as e:
            logger.error(f"Error in FinBERT analysis: {e}")
            return {'label': 'NEUTRAL', 'score': 0.5, 'available': False}
    
    def preprocess_financial_text(self, text: str) -> str:
        """Enhanced preprocessing for financial texts.
        
        Preserves financial-specific terms and patterns that are important
        for accurate sentiment analysis of crypto and financial news.
        """
        if not isinstance(text, str):
            return ""
        
        try:
            # Preserve case for financial acronyms and ticker symbols
            # e.g., BTC, ETH, USD, NASDAQ
            
            # Remove URLs
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
            
            # Preserve percentage signs and currency symbols
            # Remove most special characters but keep: $ % + - . (for numbers)
            text = re.sub(r'[^\w\s$%+\-.]', ' ', text)
            
            # Normalize whitespace
            text = ' '.join(text.split())
            
            return text
            
        except Exception as e:
            logger.error(f"Error preprocessing financial text: {e}")
            return text
    
    def get_combined_sentiment(self, text: str) -> Dict:
        """Get combined sentiment analysis from all methods, with enhanced financial AI.
        
        Now uses FinBERT (BloombergGPT alternative) for superior financial sentiment analysis.
        """
        try:
            # Check cache first
            key = (text or '').strip()
            now = time.time()
            cached = _SENTIMENT_CACHE.get(key)
            if cached:
                ts, val = cached
                if now - ts < _SENTIMENT_TTL:
                    return val

            # Analyze with all available methods
            vader_scores = self.analyze_vader_sentiment(text)
            textblob_scores = self.analyze_textblob_sentiment(text)
            
            # FinBERT analysis (if available) - prioritized for financial texts
            finbert_scores = self.analyze_finbert_sentiment(text)
            finbert_available = finbert_scores.get('available', False)

            # BERT analysis - used as fallback or when signals are ambiguous
            bert_scores = {'label': 'NEUTRAL', 'score': 0.5}
            vader_compound = vader_scores.get('compound', 0)
            textblob_polarity = textblob_scores.get('polarity', 0)
            ambiguous = abs(vader_compound) < 0.15 and abs(textblob_polarity) < 0.15
            if ambiguous and self.transformers_available and not finbert_available:
                bert_scores = self.analyze_bert_sentiment(text)

            # Normalize scores to [-1, 1] range
            vader_normalized = vader_scores['compound']
            textblob_normalized = textblob_scores['polarity']
            
            # Normalize BERT scores
            bert_mapping = {'POSITIVE': 1, 'NEUTRAL': 0, 'NEGATIVE': -1}
            bert_normalized = bert_mapping.get(bert_scores['label'], 0) * bert_scores['score']
            
            # Normalize FinBERT scores
            finbert_normalized = 0
            if finbert_available:
                finbert_normalized = bert_mapping.get(finbert_scores['label'], 0) * finbert_scores['score']

            # Weighted ensemble with FinBERT prioritization
            if finbert_available:
                # FinBERT available: prioritize financial AI model
                weights = [0.25, 0.15, 0.60]  # VADER, TextBlob, FinBERT
                combined_score = (
                    weights[0] * vader_normalized +
                    weights[1] * textblob_normalized +
                    weights[2] * finbert_normalized
                )
                model_used = 'FinBERT (Financial AI)'
            elif self.transformers_available and self.bert_analyzer:
                # Standard BERT available
                weights = [0.4, 0.3, 0.3]  # VADER, TextBlob, BERT
                combined_score = (
                    weights[0] * vader_normalized +
                    weights[1] * textblob_normalized +
                    weights[2] * bert_normalized
                )
                model_used = 'BERT'
            else:
                # Fallback to rule-based only
                weights = [0.5, 0.5]  # VADER, TextBlob only
                combined_score = (
                    weights[0] * vader_normalized +
                    weights[1] * textblob_normalized
                )
                model_used = 'Rule-based'

            # Determine sentiment category with financial-aware thresholds
            if combined_score > 0.1:
                sentiment_category = 'POSITIVE'
            elif combined_score < -0.1:
                sentiment_category = 'NEGATIVE'
            else:
                sentiment_category = 'NEUTRAL'

            # Calculate confidence (higher when FinBERT agrees with other methods)
            base_confidence = max(vader_scores['pos'], vader_scores['neg'], vader_scores['neu'])
            if finbert_available:
                # Boost confidence when FinBERT is used
                base_confidence = min(1.0, base_confidence * 1.2)

            # Build comprehensive result dict
            res = {
                'combined_score': combined_score,
                'sentiment_category': sentiment_category,
                'confidence': base_confidence,
                'model_used': model_used,
                'vader': vader_scores,
                'textblob': textblob_scores,
                'bert': bert_scores,
                'finbert': finbert_scores,
                'subjectivity': textblob_scores.get('subjectivity', 0)
            }

            # Cache result
            try:
                _SENTIMENT_CACHE[key] = (time.time(), res)
            except Exception:
                pass

            return res

        except Exception as e:
            logger.error(f"Error in combined sentiment analysis: {e}")
            return {
                'combined_score': 0,
                'sentiment_category': 'NEUTRAL',
                'confidence': 0,
                'model_used': 'Error',
                'vader': {'compound': 0, 'pos': 0, 'neg': 0, 'neu': 0},
                'textblob': {'polarity': 0, 'subjectivity': 0},
                'bert': {'label': 'NEUTRAL', 'score': 0.5},
                'finbert': {'label': 'NEUTRAL', 'score': 0.5, 'available': False},
                'subjectivity': 0
            }
    
    def analyze_news_sentiment(self, news_data: List[Dict]) -> pd.DataFrame:
        """Analyze sentiment for news articles"""
        try:
            if not news_data:
                return pd.DataFrame()

            # Prepare texts for batch processing
            texts = [news.get('title', '') for news in news_data if news.get('title')]
            
            if not texts:
                return pd.DataFrame()

            # Get combined sentiment scores for all texts (uses caching and VADER-first)
            sentiment_results = []
            for text in texts:
                # Use title text as cache key in get_combined_sentiment
                sentiment_results.append(self.get_combined_sentiment(text))

            # Create a DataFrame from the results
            analyzed_news = []
            for i, news_item in enumerate(news_data):
                if news_item.get('title'):
                    sentiment_result = sentiment_results.pop(0)
                    analyzed_item = {
                        'title': news_item.get('title', ''),
                        'url': news_item.get('url', ''),
                        'source': news_item.get('source', ''),
                        'symbol': news_item.get('symbol', ''),
                        'timestamp': news_item.get('timestamp', ''),
                        'sentiment_score': sentiment_result['combined_score'],
                        'sentiment_category': sentiment_result['sentiment_category'],
                        'confidence': sentiment_result['confidence'],
                        'subjectivity': sentiment_result['subjectivity'],
                        'vader_compound': sentiment_result['vader']['compound'],
                        'textblob_polarity': sentiment_result['textblob']['polarity'],
                        'bert_label': sentiment_result['bert']['label']
                    }
                    analyzed_news.append(analyzed_item)

            return pd.DataFrame(analyzed_news)

        except Exception as e:
            logger.error(f"Error analyzing news sentiment: {e}")
            return pd.DataFrame()
    
    def calculate_sentiment_metrics(self, sentiment_df: pd.DataFrame, symbol: str) -> Dict:
        """Calculate aggregated sentiment metrics for a symbol"""
        try:
            if sentiment_df.empty:
                return {
                    'symbol': symbol,
                    'total_articles': 0,
                    'positive_count': 0,
                    'negative_count': 0,
                    'neutral_count': 0,
                    'avg_sentiment_score': 0,
                    'sentiment_bias': 'NEUTRAL',
                    'confidence': 0
                }
            
            # Filter by symbol
            symbol_df = sentiment_df[sentiment_df['symbol'] == symbol]
            
            if symbol_df.empty:
                return {
                    'symbol': symbol,
                    'total_articles': 0,
                    'positive_count': 0,
                    'negative_count': 0,
                    'neutral_count': 0,
                    'avg_sentiment_score': 0,
                    'sentiment_bias': 'NEUTRAL',
                    'confidence': 0
                }
            
            total_articles = len(symbol_df)
            positive_count = len(symbol_df[symbol_df['sentiment_category'] == 'POSITIVE'])
            negative_count = len(symbol_df[symbol_df['sentiment_category'] == 'NEGATIVE'])
            neutral_count = len(symbol_df[symbol_df['sentiment_category'] == 'NEUTRAL'])
            
            avg_sentiment_score = symbol_df['sentiment_score'].mean()
            avg_confidence = symbol_df['confidence'].mean()
            
            # Determine sentiment bias
            if positive_count > negative_count:
                sentiment_bias = 'POSITIVE'
            elif negative_count > positive_count:
                sentiment_bias = 'NEGATIVE'
            else:
                sentiment_bias = 'NEUTRAL'
            
            return {
                'symbol': symbol,
                'total_articles': total_articles,
                'positive_count': positive_count,
                'negative_count': negative_count,
                'neutral_count': neutral_count,
                'positive_ratio': positive_count / total_articles if total_articles > 0 else 0,
                'negative_ratio': negative_count / total_articles if total_articles > 0 else 0,
                'neutral_ratio': neutral_count / total_articles if total_articles > 0 else 0,
                'avg_sentiment_score': avg_sentiment_score,
                'sentiment_bias': sentiment_bias,
                'confidence': avg_confidence,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating sentiment metrics for {symbol}: {e}")
            return {
                'symbol': symbol,
                'total_articles': 0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'avg_sentiment_score': 0,
                'sentiment_bias': 'NEUTRAL',
                'confidence': 0
            }
    
    def analyze_social_sentiment(self, social_data: Dict) -> Dict:
        """Analyze sentiment for social media data"""
        try:
            # This is a placeholder for social media sentiment analysis
            # In production, you'd implement actual social media API calls
            
            analyzed_social = {
                'twitter': {'positive': 0, 'negative': 0, 'neutral': 0, 'total': 0},
                'reddit': {'positive': 0, 'negative': 0, 'neutral': 0, 'total': 0}
            }
            
            # Mock analysis for demonstration
            # In production, you'd analyze actual social media posts
            
            return analyzed_social
            
        except Exception as e:
            logger.error(f"Error analyzing social sentiment: {e}")
            return {}
    
    def get_sentiment_summary(self, news_sentiment_df: pd.DataFrame, social_sentiment: Dict, symbols: List[str]) -> Dict:
        """Get comprehensive sentiment summary for all symbols"""
        try:
            sentiment_summary = {}
            
            for symbol in symbols:
                # News sentiment metrics
                news_metrics = self.calculate_sentiment_metrics(news_sentiment_df, symbol)
                
                # Social sentiment metrics (placeholder)
                social_metrics = social_sentiment.get(symbol, {})
                
                # Combine metrics
                sentiment_summary[symbol] = {
                    'news_sentiment': news_metrics,
                    'social_sentiment': social_metrics,
                    'overall_sentiment': self.calculate_overall_sentiment(news_metrics, social_metrics)
                }
            
            return sentiment_summary
            
        except Exception as e:
            logger.error(f"Error getting sentiment summary: {e}")
            return {}
    
    def calculate_overall_sentiment(self, news_metrics: Dict, social_metrics: Dict) -> Dict:
        """Calculate overall sentiment combining news and social metrics"""
        try:
            # Weight news sentiment more heavily than social
            news_weight = 0.7
            social_weight = 0.3
            
            # News sentiment score
            news_score = news_metrics.get('avg_sentiment_score', 0)
            news_confidence = news_metrics.get('confidence', 0)
            
            # Social sentiment score (placeholder)
            social_score = 0  # In production, calculate from actual social data
            social_confidence = 0.5
            
            # Weighted average
            overall_score = (news_weight * news_score + social_weight * social_score)
            overall_confidence = (news_weight * news_confidence + social_weight * social_confidence)
            
            # Determine overall sentiment
            if overall_score > 0.1:
                overall_sentiment = 'POSITIVE'
            elif overall_score < -0.1:
                overall_sentiment = 'NEGATIVE'
            else:
                overall_sentiment = 'NEUTRAL'
            
            return {
                'score': overall_score,
                'sentiment': overall_sentiment,
                'confidence': overall_confidence,
                'news_weight': news_weight,
                'social_weight': social_weight
            }
            
        except Exception as e:
            logger.error(f"Error calculating overall sentiment: {e}")
            return {
                'score': 0,
                'sentiment': 'NEUTRAL',
                'confidence': 0,
                'news_weight': 0.7,
                'social_weight': 0.3
            }

# Example usage
def main():
    analyzer = CryptoSentimentAnalyzer()
    
    # Test sentiment analysis
    test_texts = [
        "Bitcoin is going to the moon! 🚀",
        "Crypto market crash causes panic selling",
        "New blockchain partnership announced",
        "Regulatory concerns impact crypto prices"
    ]
    
    print("Testing sentiment analysis:")
    for text in test_texts:
        sentiment = analyzer.get_combined_sentiment(text)
        print(f"\nText: {text}")
        print(f"Sentiment: {sentiment['sentiment_category']}")
        print(f"Score: {sentiment['combined_score']:.3f}")
        print(f"Confidence: {sentiment['confidence']:.3f}")

if __name__ == "__main__":
    main()
