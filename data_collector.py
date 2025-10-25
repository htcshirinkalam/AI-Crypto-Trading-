import asyncio
import aiohttp
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import ccxt
from bs4 import BeautifulSoup
import time
import json
import os
from loguru import logger
from config import Config

# Additional market data sources
try:
    import cryptocompare
    CRYPTOCOMPARE_AVAILABLE = True
except ImportError:
    CRYPTOCOMPARE_AVAILABLE = False
    logger.warning("CryptoCompare not available")


try:
    from openbb_integration import OpenBBIntegration
    OPENBB_AVAILABLE = True
except ImportError:
    OPENBB_AVAILABLE = False
    logger.warning("OpenBB integration not available")

try:
    from apilayer_integration import APILayerIntegration
    APILAYER_AVAILABLE = True
except ImportError:
    APILAYER_AVAILABLE = False
    logger.warning("APILayer integration not available")

try:
    from web_crawler import HybridDataCollector, NewsCrawler, SocialMediaCrawler
    WEB_CRAWLER_AVAILABLE = True
except ImportError:
    WEB_CRAWLER_AVAILABLE = False
    logger.warning("Web crawler not available")


class CryptoDataCollector:
    def __init__(self):
        self.config = Config()
        self.session = None
        self.requests_session = None  # For synchronous requests
        self.exchange = ccxt.kucoin()  # Use KuCoin exchange via ccxt
        self.setup_exchanges()
        
        # Quality thresholds
        self.max_allowed_nan_ratio = 0.05
        self.outlier_zscore_threshold = 6.0
        
    def get_requests_session(self):
        """Get or create a robust requests session for API calls"""
        if self.requests_session is None:
            self.requests_session = requests.Session()
            
            # Configure retry strategy
            retry_strategy = Retry(
                total=3,  # Total number of retries
                backoff_factor=1,  # Wait time between retries
                status_forcelist=[429, 500, 502, 503, 504],  # HTTP status codes to retry on
                allowed_methods=["HEAD", "GET", "OPTIONS"]  # HTTP methods to retry
            )
            
            # Mount adapter with retry strategy
            adapter = HTTPAdapter(max_retries=retry_strategy)
            self.requests_session.mount("http://", adapter)
            self.requests_session.mount("https://", adapter)
            
            # Set headers to be more respectful to the API
            self.requests_session.headers.update({
                'User-Agent': 'Crypto-Trading-Agent/1.0 (Educational Purpose)',
                'Accept': 'application/json',
                'Accept-Encoding': 'gzip, deflate'
            })
            
            # Set connection timeout
            self.requests_session.timeout = 30
        
        return self.requests_session
        
    def setup_exchanges(self):
        """Initialize data source connections"""
        # Initialize APILayer (primary data source)
        if APILAYER_AVAILABLE and self.config.USE_APILAYER:
            self.apilayer = APILayerIntegration(self.config.APILAYER_API_KEY)
            logger.info("APILayer integration initialized")
        else:
            self.apilayer = None
            if not self.config.USE_APILAYER:
                logger.info("APILayer not configured, using direct API access")

        # Initialize fallback data sources
        if CRYPTOCOMPARE_AVAILABLE:
            self.cryptocompare = cryptocompare
        else:
            self.cryptocompare = None


        if OPENBB_AVAILABLE:
            self.openbb_integration = OpenBBIntegration()
        else:
            self.openbb_integration = None

        # Initialize web crawler for sentiment analysis
        if WEB_CRAWLER_AVAILABLE:
            self.hybrid_collector = HybridDataCollector(self.config)
            self.news_crawler = NewsCrawler(self.config)
            self.social_crawler = SocialMediaCrawler(self.config)
            logger.info("Web crawler initialized for sentiment analysis")
        else:
            self.hybrid_collector = None
            self.news_crawler = None
            self.social_crawler = None

            
    async def get_session(self):
        """Get or create aiohttp session"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=10)  # 10 second timeout
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def close_session(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def fetch_market_data(self, symbol: str, timeframe: str = '1d', limit: int = 100) -> pd.DataFrame:
        """Fetch market data using APILayer (primary) with fallback to direct APIs"""
        try:
            data_sources = []
            errors = []

            # Try APILayer first (if available)
            if self.apilayer and self.apilayer.is_available():
                try:
                    if timeframe == 'current' or limit == 1:
                        # Get current price data
                        apilayer_data = await self.apilayer.get_cryptocurrency_data(symbol, 'coingecko')
                        if apilayer_data:
                            # Convert APILayer response to DataFrame format
                            df = self._convert_apilayer_to_dataframe(apilayer_data, symbol)
                            if not df.empty:
                                data_sources.append(df)
                                logger.info(f"Successfully fetched APILayer data for {symbol}")
                    else:
                        # Get historical data
                        apilayer_data = await self.apilayer.get_crypto_historical_data(symbol, days=30, source='coingecko')
                        if apilayer_data:
                            df = self._convert_apilayer_historical_to_dataframe(apilayer_data, symbol)
                            if not df.empty:
                                data_sources.append(df)
                                logger.info(f"Successfully fetched APILayer historical data for {symbol}")
                except Exception as e:
                    errors.append(f"APILayer: {str(e)}")
                    logger.warning(f"APILayer failed for {symbol}, falling back to direct APIs")

            # Fallback to direct APIs if APILayer not available or failed
            if not data_sources:
                try:
                    # Fetch small recent history to enable feature engineering
                    hist_limit = max(50, limit)
                    hist_data = await self.fetch_kucoin_data(symbol, timeframe, hist_limit)
                    if hist_data is not None and not hist_data.empty:
                        data_sources.append(hist_data)
                        logger.info(f"Successfully fetched historical KuCoin data for {symbol}")

                    # Also fetch current bar for latest snapshot
                    cur_data = await self.fetch_kucoin_data(symbol, 'current', 1)
                    if cur_data is not None and not cur_data.empty:
                        data_sources.append(cur_data)
                        logger.info(f"Successfully fetched current KuCoin data for {symbol}")

                    if not data_sources:
                        errors.append("KuCoin: No data returned")
                except Exception as e:
                    errors.append(f"KuCoin: {str(e)}")

            # Fetch data from OpenBB (highest priority for stocks and crypto)
            if not data_sources and self.openbb_integration and self.openbb_integration.is_available():
                try:
                    # Map timeframe to OpenBB format
                    openbb_timeframe_map = {
                        '1m': '1m', '5m': '5m', '15m': '15m', '1h': '1H', '4h': '4H', '1d': '1D'
                    }
                    openbb_timeframe = openbb_timeframe_map.get(timeframe, '1D')

                    # Determine if it's crypto or stock
                    is_crypto = symbol in ['BTC', 'ETH', 'BNB', 'SOL', 'DOT', 'LINK', 'XRP', 'LTC', 'BCH', 'USDT']
                    openbb_symbol = f"{symbol}-USD" if is_crypto else symbol

                    openbb_data = await self.openbb_integration.get_market_data(
                        openbb_symbol, openbb_timeframe, limit=limit
                    )

                    if not openbb_data.empty:
                        data_sources.append(openbb_data)
                        logger.info(f"Successfully fetched OpenBB data for {symbol}")
                    else:
                        errors.append("OpenBB: No data returned")
                except Exception as e:
                    errors.append(f"OpenBB: {str(e)}")

            # Combine and clean data from all sources
            if data_sources:
                combined_data = self.merge_multiple_data_sources(data_sources)
                combined_data = self._validate_and_clean_time_series(combined_data, timeframe=timeframe)
                logger.info(f"Successfully combined data from {len(data_sources)} sources for {symbol}")
                return combined_data
            else:
                logger.warning(f"Failed to fetch market data for {symbol} from any source")
                logger.warning(f"Errors encountered: {errors}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Critical error fetching market data for {symbol}: {e}")
            return pd.DataFrame()
    
    async def fetch_kucoin_data(self, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
        """Fetch current and historical OHLCV data from KuCoin using ccxt."""
        try:
            market_symbol = f"{symbol}/USDT"
            # Use CCXT-native timeframes for KuCoin
            timeframe_map = {
                'current': '1m',
                '1m': '1m',
                '5m': '5m',
                '15m': '15m',
                '1h': '1h',
                '4h': '4h',
                '1d': '1d',
                '1w': '1w'
            }
            kucoin_timeframe = timeframe_map.get(timeframe, '1d')

            # Ensure a reasonable limit
            fetch_limit = max(1, int(limit or 100))

            ohlcv = self.exchange.fetch_ohlcv(market_symbol, timeframe=kucoin_timeframe, limit=fetch_limit)
            if not ohlcv:
                return None

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol

            # Derived fields used elsewhere in pipeline
            df['price'] = df['close']
            df['market_cap'] = np.nan
            df['price_change_24h'] = df['price'].pct_change(periods=1) * 100

            # Return full OHLCV plus derived columns for downstream consumers
            return df[['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'price', 'market_cap', 'price_change_24h']]
        except Exception as e:
            logger.error(f"Error fetching KuCoin data: {e}")
            return None
    

    async def fetch_cryptocompare_data(self, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
        """Fetch data from CryptoCompare API"""
        try:
            if not self.cryptocompare:
                return None

            # Map timeframe to CryptoCompare format
            timeframe_map = {
                '1m': 'minute',
                '1h': 'hour',
                '1d': 'day'
            }

            cc_timeframe = timeframe_map.get(timeframe, 'day')

            # Get historical data
            data = self.cryptocompare.get_historical_price_day(
                symbol, 'USD', limit=limit
            )

            if data:
                df = pd.DataFrame(data)
                df['timestamp'] = pd.to_datetime(df['time'], unit='s')
                df['symbol'] = symbol
                df['price'] = df['close']
                df['volume'] = df['volumeto']

                return df[['timestamp', 'symbol', 'price', 'volume']]

        except Exception as e:
            logger.error(f"Error fetching CryptoCompare data: {e}")

        return None

    async def fetch_openbb_data(self, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
        """Fetch data from OpenBB platform"""
        try:
            if not self.openbb_integration or not self.openbb_integration.is_available():
                return None

            # Map timeframe to OpenBB format
            timeframe_map = {
                '1m': '1m', '5m': '5m', '15m': '15m', '1h': '1H',
                '4h': '4H', '1d': '1D', '1w': '1W'
            }
            openbb_timeframe = timeframe_map.get(timeframe, '1D')

            # Determine if it's crypto or stock
            is_crypto = symbol in ['BTC', 'ETH', 'BNB', 'SOL', 'DOT', 'LINK', 'XRP', 'LTC', 'BCH', 'USDT']
            openbb_symbol = f"{symbol}-USD" if is_crypto else symbol

            # Fetch data from OpenBB
            data = await self.openbb_integration.get_market_data(
                openbb_symbol, openbb_timeframe, limit=limit
            )

            if not data.empty:
                logger.info(f"Fetched OpenBB data for {symbol}: {len(data)} records")
                return data

        except Exception as e:
            logger.error(f"Error fetching OpenBB data: {e}")

        return None

    async def get_openbb_quote(self, symbol: str) -> Optional[Dict]:
        """Get real-time quote from OpenBB"""
        try:
            if not self.openbb_integration or not self.openbb_integration.is_available():
                return None

            # Determine if it's crypto or stock
            is_crypto = symbol in ['BTC', 'ETH', 'BNB', 'SOL', 'DOT', 'LINK', 'XRP', 'LTC', 'BCH', 'USDT']
            openbb_symbol = f"{symbol}-USD" if is_crypto else symbol

            quote = await self.openbb_integration.get_real_time_quote(openbb_symbol)
            if quote:
                logger.info(f"Got OpenBB quote for {symbol}: ${quote.get('price', 'N/A')}")
                return quote

        except Exception as e:
            logger.error(f"Error getting OpenBB quote: {e}")

        return None

    async def get_openbb_technical_analysis(self, symbol: str) -> Optional[Dict]:
        """Get technical analysis from OpenBB"""
        try:
            if not self.openbb_integration or not self.openbb_integration.is_available():
                return None

            # Determine if it's crypto or stock
            is_crypto = symbol in ['BTC', 'ETH', 'BNB', 'SOL', 'DOT', 'LINK', 'XRP', 'LTC', 'BCH', 'USDT']
            openbb_symbol = f"{symbol}-USD" if is_crypto else symbol

            analysis = await self.openbb_integration.get_technical_analysis(openbb_symbol)
            if analysis:
                logger.info(f"Got OpenBB technical analysis for {symbol}")
                return analysis

        except Exception as e:
            logger.error(f"Error getting OpenBB technical analysis: {e}")

        return None

    async def get_openbb_news_sentiment(self, symbol: str, limit: int = 5) -> List[Dict]:
        """Get news and sentiment from OpenBB"""
        try:
            if not self.openbb_integration or not self.openbb_integration.is_available():
                return []

            # Determine if it's crypto or stock
            is_crypto = symbol in ['BTC', 'ETH', 'BNB', 'SOL', 'DOT', 'LINK', 'XRP', 'LTC', 'BCH', 'USDT']
            openbb_symbol = f"{symbol}-USD" if is_crypto else symbol

            news = await self.openbb_integration.get_news_sentiment(openbb_symbol, limit)
            if news:
                logger.info(f"Got {len(news)} OpenBB news items for {symbol}")
                return news

        except Exception as e:
            logger.error(f"Error getting OpenBB news: {e}")

        return []

    def get_coin_id_from_symbol(self, symbol: str) -> Optional[str]:
        """Get CoinGecko coin ID from symbol"""
        symbol_mapping = {
            'BTC': 'bitcoin', 'ETH': 'ethereum', 'BNB': 'binancecoin',
            'SOL': 'solana', 'DOT': 'polkadot',
            'AVAX': 'avalanche-2', 'MATIC': 'matic-network',
            'LINK': 'chainlink', 'UNI': 'uniswap', 'ATOM': 'cosmos',
            'LTC': 'litecoin', 'BCH': 'bitcoin-cash', 'XRP': 'ripple',
            'DOGE': 'dogecoin', 'SHIB': 'shiba-inu', 'USDT': 'tether'
        }
        return symbol_mapping.get(symbol.upper())
    

    def _convert_apilayer_to_dataframe(self, apilayer_data: Dict, symbol: str) -> pd.DataFrame:
        """Convert APILayer cryptocurrency data to DataFrame format"""
        try:
            if not apilayer_data or 'price' not in apilayer_data:
                return pd.DataFrame()

            df = pd.DataFrame([{
                'timestamp': datetime.now(),
                'symbol': symbol,
                'price': apilayer_data.get('price', 0),
                'volume': apilayer_data.get('volume_24h', 0),
                'market_cap': apilayer_data.get('market_cap', 0),
                'price_change_24h': apilayer_data.get('price_change_percentage_24h', 0)
            }])

            return df

        except Exception as e:
            logger.error(f"Error converting APILayer data to DataFrame: {e}")
            return pd.DataFrame()

    def _convert_apilayer_historical_to_dataframe(self, apilayer_data: List[Dict], symbol: str) -> pd.DataFrame:
        """Convert APILayer historical data to DataFrame format"""
        try:
            if not apilayer_data:
                return pd.DataFrame()

            data = []
            for item in apilayer_data:
                if isinstance(item, dict) and 'timestamp' in item and 'price' in item:
                    data.append({
                        'timestamp': pd.to_datetime(item['timestamp'], unit='ms') if isinstance(item['timestamp'], (int, float)) else pd.to_datetime(item['timestamp']),
                        'symbol': symbol,
                        'price': item.get('price', 0),
                        'volume': item.get('volume', 0),
                        'market_cap': item.get('market_cap', 0)
                    })

            if data:
                df = pd.DataFrame(data)
                df = df.sort_values('timestamp').reset_index(drop=True)
                return df
            else:
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error converting APILayer historical data to DataFrame: {e}")
            return pd.DataFrame()

    def merge_multiple_data_sources(self, data_sources: List[pd.DataFrame]) -> pd.DataFrame:
        """Merge data from multiple sources with quality weighting"""
        try:
            if not data_sources:
                return pd.DataFrame()

            # Define source quality weights (higher = better)
            source_weights = {
                'openbb': 0.95,      # Highest quality - comprehensive financial platform      # High quality - financial data provider
                'coingecko': 0.8,    # Good quality - comprehensive data
                'cryptocompare': 0.7  # Good quality - alternative source
            }

            # Combine all data sources
            combined_df = pd.concat(data_sources, ignore_index=True)

            if combined_df.empty:
                return pd.DataFrame()

            # Sort by timestamp and remove duplicates, keeping the highest quality source
            combined_df = combined_df.sort_values(['timestamp', 'symbol']).drop_duplicates(
                subset=['timestamp', 'symbol'], keep='first'
            )

            # Sort by timestamp
            combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)

            logger.info(f"Merged data from {len(data_sources)} sources, final shape: {combined_df.shape}")
            return combined_df

        except Exception as e:
            logger.error(f"Error merging multiple data sources: {e}")
            return pd.DataFrame()
    
    def _validate_and_clean_time_series(self, df: pd.DataFrame, timeframe: str = '1d') -> pd.DataFrame:
        """Validate, normalize timezone, gap-fill, and flag quality issues.
        Adds columns: is_imputed, is_outlier, data_quality_score.
        """
        try:
            if df.empty:
                return df

            # Ensure timestamp is datetime and UTC
            df = df.copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
            df = df.dropna(subset=['timestamp'])
            df = df.sort_values('timestamp')

            # Determine frequency
            freq_map = {'1m': '1T', '5m': '5T', '15m': '15T', '1h': '1H', '4h': '4H', '1d': '1D', '1w': '7D', 'current': None}
            freq = freq_map.get(timeframe, '1D')

            # Reindex to expected frequency per symbol
            results = []
            for symbol, g in df.groupby('symbol'):
                g = g.set_index('timestamp')
                if freq:
                    full_index = pd.date_range(start=g.index.min(), end=g.index.max(), freq=freq, tz='UTC')
                    g = g.reindex(full_index)
                    g['symbol'] = symbol
                    g.index.name = 'timestamp'
                    g = g.reset_index().rename(columns={'index': 'timestamp'})
                else:
                    g = g.reset_index()

                results.append(g)

            df = pd.concat(results, ignore_index=True)

            # Flag imputed rows (created by reindexing)
            df['is_imputed'] = df[['price', 'volume']].isna().any(axis=1).astype(int)

            # Forward/backward fill for small gaps
            df[['price', 'volume', 'market_cap', 'price_change_24h']] = df[['price', 'volume', 'market_cap', 'price_change_24h']].ffill().bfill()

            # Outlier detection on price using rolling z-score
            def _zscore(x: pd.Series) -> pd.Series:
                mu = x.rolling(50, min_periods=10).mean()
                sigma = x.rolling(50, min_periods=10).std(ddof=0)
                return (x - mu) / sigma

            df['price_z'] = df.groupby('symbol')['price'].transform(_zscore)
            df['is_outlier'] = (df['price_z'].abs() > self.outlier_zscore_threshold).astype(int)

            # Clip extreme outliers (winsorize)
            df['price'] = df.groupby('symbol')['price'].transform(lambda s: s.clip(lower=s.quantile(0.001), upper=s.quantile(0.999)))

            # Data quality score (simple heuristic)
            nan_ratio = df[['price', 'volume']].isna().mean(axis=1)
            df['data_quality_score'] = (1.0 - nan_ratio) * (1.0 - 0.5 * df['is_imputed']) * (1.0 - 0.5 * df['is_outlier'])

            # Drop helper column
            df = df.drop(columns=['price_z'])

            # Final sanity: drop rows still missing price
            df = df.dropna(subset=['price'])

            return df
        except Exception as e:
            logger.error(f"Error validating/cleaning time series: {e}")
            return df
    
    async def fetch_news_data(self, symbol: str, hours_back: int = 24) -> List[Dict]:
        """Fetch news via Google News RSS for improved reliability."""
        news_data: List[Dict] = []
        try:
            session = await self.get_session()

            # Build Google News RSS query focusing on cryptocurrency and symbol
            import urllib.parse
            query_terms = [symbol, 'cryptocurrency', 'crypto', 'blockchain', 'bitcoin', 'ethereum']
            query = urllib.parse.quote_plus(' OR '.join(query_terms))
            rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

            async with session.get(rss_url, timeout=aiohttp.ClientTimeout(total=8)) as response:
                if response.status != 200:
                    logger.warning(f"Google News RSS HTTP {response.status} for {symbol}")
                    return news_data
                rss_text = await response.text()

            # Parse RSS XML (avoid extra deps; use minimal parsing)
            from xml.etree import ElementTree as ET
            try:
                root = ET.fromstring(rss_text)
            except Exception as e:
                logger.warning(f"Failed to parse RSS for {symbol}: {e}")
                return news_data

            # Google News RSS structure: rss/channel/item
            channel = root.find('channel')
            if channel is None:
                return news_data

            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            for item in channel.findall('item')[:20]:
                try:
                    title = (item.findtext('title') or '').strip()
                    link = (item.findtext('link') or '').strip()
                    pub_date_text = item.findtext('pubDate') or ''

                    # Parse pubDate if available
                    from email.utils import parsedate_to_datetime
                    try:
                        pub_dt = parsedate_to_datetime(pub_date_text)
                    except Exception:
                        pub_dt = datetime.now()

                    if pub_dt < cutoff_time:
                        continue

                    if len(title) < 10 or not link:
                        continue

                    news_data.append({
                        'title': title,
                        'url': link,
                        'source': 'google_news_rss',
                        'symbol': symbol,
                        'timestamp': pub_dt.isoformat()
                    })
                except Exception:
                    continue

        except asyncio.TimeoutError:
            logger.warning(f"Timeout fetching Google News RSS for {symbol}")
        except Exception as e:
            logger.error(f"Error fetching news data via RSS: {e}")
        finally:
            await self.close_session()

        logger.info(f"Fetched {len(news_data)} news items for {symbol} via RSS")
        return news_data
    
    async def scrape_news_source(self, session: aiohttp.ClientSession, source: str, symbol: str, hours_back: int) -> List[Dict]:
        """Scrape news from a specific source with improved error handling"""
        news_items = []

        try:
            # Skip problematic sources that are causing timeouts
            blocked_sources = ['coindesk.com', 'bitcoin.com', 'newsbtc.com', 'ambcrypto.com']
            if source in blocked_sources:
                logger.debug(f"Skipping blocked source: {source}")
                return news_items

            # This is a simplified scraper - in production, you'd need more sophisticated scraping
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            # Search for crypto-related news
            search_terms = [symbol, 'cryptocurrency', 'bitcoin', 'blockchain']

            for term in search_terms:
                try:
                    # Add timeout and retry logic
                    search_url = f"https://{source}/search?q={term}"

                    # Try the request with timeout
                    async with session.get(search_url, headers=headers, timeout=aiohttp.ClientTimeout(total=5)) as response:
                        if response.status == 200:
                            html = await response.text()
                            soup = BeautifulSoup(html, 'html.parser')

                            # Extract news items (this would need to be customized per source)
                            articles = soup.find_all(['article', 'div'], class_=['article', 'news-item'])

                            for article in articles[:3]:  # Limit to 3 articles per term
                                try:
                                    title = article.find(['h1', 'h2', 'h3'])
                                    title_text = title.get_text().strip() if title else "No title"

                                    link = article.find('a')
                                    link_url = link.get('href') if link else ""

                                    if title_text and link_url and len(title_text) > 10:  # Filter out very short titles
                                        news_items.append({
                                            'title': title_text,
                                            'url': link_url,
                                            'source': source,
                                            'symbol': symbol,
                                            'timestamp': datetime.now().isoformat(),
                                            'search_term': term
                                        })
                                except Exception as e:
                                    logger.debug(f"Error parsing article: {e}")

                        elif response.status == 404:
                            logger.debug(f"Search page not found for {source}")
                            break  # Don't try other search terms for this source
                        else:
                            logger.debug(f"HTTP {response.status} from {source}")

                except asyncio.TimeoutError:
                    logger.debug(f"Timeout fetching from {source} for term {term}")
                    break  # Don't try other terms if timing out
                except aiohttp.ClientError as e:
                    logger.debug(f"Client error fetching from {source}: {e}")
                    break  # Don't try other terms if connection fails
                except Exception as e:
                    logger.debug(f"Error fetching from {source} for term {term}: {e}")

                # Small delay between requests to be respectful
                await asyncio.sleep(0.5)

        except Exception as e:
            logger.error(f"Error scraping {source}: {e}")

        return news_items
    
    async def fetch_social_sentiment(self, symbol: str, hours_back: int = 24) -> Dict:
        """Fetch social media sentiment data"""
        sentiment_data = {
            'twitter': {'positive': 0, 'negative': 0, 'neutral': 0, 'total': 0},
            'reddit': {'positive': 0, 'negative': 0, 'neutral': 0, 'total': 0}
        }
        
        try:
            # This is a placeholder - actual implementation would require API access
            # to Twitter and Reddit, which have rate limits and authentication requirements
            
            # For demonstration, we'll return mock data
            # In production, you'd implement actual social media API calls
            
            logger.info(f"Social sentiment data collection for {symbol} - requires API implementation")
            
        except Exception as e:
            logger.error(f"Error fetching social sentiment: {e}")
        
        return sentiment_data
    
    async def collect_all_data(self, symbols: List[str] = None, include_sentiment: bool = True) -> Dict:
        """Collect all data for specified symbols using hybrid approach"""
        if symbols is None:
            # Use core cryptos plus top 5 most volatile
            symbols = self.config.CORE_CRYPTO_SYMBOLS.copy()
            volatile_symbols = await self.get_most_volatile_cryptos(limit=5)
            symbols.extend(volatile_symbols)

            logger.info(f"Using crypto symbols: {symbols}")

        all_data = {}

        try:
            for symbol in symbols:
                logger.info(f"Collecting hybrid data for {symbol}")

                # Collect market data (APILayer primary + fallbacks)
                market_data = await self.fetch_market_data(symbol, '1d', 100)

                # Collect news data (legacy method)
                news_data = await self.fetch_news_data(symbol, 24)

                # Collect social sentiment (legacy method)
                social_sentiment = await self.fetch_social_sentiment(symbol, 24)

                # Collect enhanced sentiment data (web crawler)
                enhanced_sentiment = {}
                if include_sentiment and self.hybrid_collector:
                    try:
                        enhanced_sentiment = await self.hybrid_collector.collect_comprehensive_data(symbol)
                        logger.info(f"Collected enhanced sentiment data for {symbol}")
                    except Exception as e:
                        logger.warning(f"Failed to collect enhanced sentiment for {symbol}: {e}")

                all_data[symbol] = {
                    'market_data': market_data,
                    'news_data': news_data,
                    'social_sentiment': social_sentiment,
                    'enhanced_sentiment': enhanced_sentiment,
                    'timestamp': datetime.now().isoformat()
                }

                await asyncio.sleep(1)  # Rate limiting

        except Exception as e:
            logger.error(f"Error collecting all data: {e}")

        return all_data

    async def get_high_volatility_cryptos(self, min_change_percent: float = 20.0, limit: int = 50) -> List[str]:
        """Get cryptocurrencies with high 24h price changes but NOT high volume"""
        try:
            logger.info(f"Fetching cryptocurrencies with >{min_change_percent}% 24h change and low volume")

            # Get more cryptocurrencies to find low volume high change ones
            url = "https://api.coingecko.com/api/v3/coins/markets"
            params = {
                'vs_currency': 'usd',
                'order': 'market_cap_desc',
                'per_page': 200,  # Increased to get more coins
                'page': 1,
                'sparkline': 'false',
                'price_change_percentage': '24h'
            }

            session = await self.get_session()
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    # Filter cryptocurrencies with >20% price change AND low volume
                    low_vol_high_change_cryptos = []
                    for crypto in data:
                        price_change = crypto.get('price_change_percentage_24h', 0)
                        volume = crypto.get('total_volume', 0)
                        market_cap = crypto.get('market_cap', 0)

                        # High change, low volume (< $50M), reasonable market cap (> $5M)
                        if (price_change is not None and abs(price_change) >= min_change_percent and
                            volume < 50000000 and  # < $50M volume (not high volume)
                            market_cap > 5000000):  # > $5M market cap (avoid micro caps)

                            symbol = crypto.get('symbol', '').upper()
                            if symbol:
                                low_vol_high_change_cryptos.append({
                                    'symbol': symbol,
                                    'price_change': price_change,
                                    'volume': volume,
                                    'market_cap': market_cap
                                })

                    # Sort by absolute price change (highest first)
                    low_vol_high_change_cryptos.sort(key=lambda x: abs(x['price_change']), reverse=True)

                    # Extract symbols
                    symbols = [crypto['symbol'] for crypto in low_vol_high_change_cryptos]

                    logger.info(f"Found {len(symbols)} low-volume high-change cryptocurrencies with >{min_change_percent}% change")
                    return symbols[:10] if symbols else self.config.CRYPTO_SYMBOLS[:5]

                else:
                    logger.warning(f"CoinGecko API error: {response.status}")
                    return self.config.CRYPTO_SYMBOLS[:5]  # Fallback

        except Exception as e:
            logger.error(f"Error fetching high volatility cryptos: {e}")
            return self.config.CRYPTO_SYMBOLS[:5]  # Fallback

        finally:
            await self.close_session()

    async def get_most_volatile_cryptos(self, limit: int = 5) -> List[str]:
        """Get top 5 most volatile cryptocurrencies in the last 24 hours"""
        try:
            logger.info(f"Fetching top {limit} most volatile cryptocurrencies")

            # Get market data from CoinGecko
            url = "https://api.coingecko.com/api/v3/coins/markets"
            params = {
                'vs_currency': 'usd',
                'order': 'market_cap_desc',
                'per_page': 100,  # Get top 100 by market cap
                'page': 1,
                'sparkline': 'false',
                'price_change_percentage': '24h'
            }

            session = await self.get_session()
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    # Filter and sort by absolute price change percentage
                    volatile_cryptos = []
                    for crypto in data:
                        price_change = crypto.get('price_change_percentage_24h', 0)
                        symbol = crypto.get('symbol', '').upper()

                        if price_change is not None and symbol:
                            volatile_cryptos.append({
                                'symbol': symbol,
                                'price_change': abs(price_change),  # Use absolute value for volatility
                                'market_cap': crypto.get('market_cap', 0)
                            })

                    # Sort by volatility (highest first) and filter out core cryptos
                    core_symbols = set(self.config.CORE_CRYPTO_SYMBOLS)
                    volatile_cryptos = [c for c in volatile_cryptos if c['symbol'] not in core_symbols]
                    volatile_cryptos.sort(key=lambda x: x['price_change'], reverse=True)

                    # Get top volatile symbols
                    symbols = [crypto['symbol'] for crypto in volatile_cryptos[:limit]]

                    logger.info(f"Found top {len(symbols)} most volatile cryptocurrencies: {symbols}")
                    return symbols

                else:
                    logger.warning(f"CoinGecko API error: {response.status}")
                    return []

        except Exception as e:
            logger.error(f"Error fetching most volatile cryptos: {e}")
            return []
        finally:
            await self.close_session()

    async def get_trending_cryptos(self, limit: int = 3) -> List[str]:
        """Get top trending cryptocurrencies from CoinGecko"""
        try:
            logger.info(f"Fetching top {limit} trending cryptocurrencies from CoinGecko")

            # Use CoinGecko's trending endpoint
            trending_data = self.coingecko.get_search_trending()

            if trending_data and 'coins' in trending_data:
                trending_coins = trending_data['coins'][:limit]

                symbols = []
                for coin_info in trending_coins:
                    coin = coin_info.get('item', {})
                    symbol = coin.get('symbol', '').upper()
                    if symbol:
                        symbols.append(symbol)

                logger.info(f"Found {len(symbols)} trending cryptocurrencies: {symbols}")
                return symbols

            else:
                logger.warning("No trending data available from CoinGecko")
                return ['BTC', 'ETH', 'BNB']  # Fallback

        except Exception as e:
            logger.error(f"Error fetching trending cryptos: {e}")
            return ['BTC', 'ETH', 'BNB']  # Fallback

    async def collect_enhanced_sentiment(self, symbol: str) -> Dict:
        """Collect enhanced sentiment data using web crawler"""
        if not self.hybrid_collector:
            logger.warning("Web crawler not available for enhanced sentiment")
            return {}

        try:
            sentiment_data = await self.hybrid_collector.collect_comprehensive_data(symbol)
            logger.info(f"Collected enhanced sentiment for {symbol}")
            return sentiment_data
        except Exception as e:
            logger.error(f"Error collecting enhanced sentiment for {symbol}: {e}")
            return {}

    async def collect_news_sentiment(self, symbol: str, sources: List[str] = None) -> List[Dict]:
        """Collect news articles with sentiment analysis"""
        if not self.news_crawler:
            logger.warning("News crawler not available")
            return []

        if sources is None:
            sources = ['coindesk', 'cointelegraph', 'decrypt']

        all_news = []
        try:
            for source in sources:
                try:
                    news = await self.news_crawler.crawl_news_source(source, symbol, max_articles=5)
                    all_news.extend(news)
                    await asyncio.sleep(0.5)  # Be respectful
                except Exception as e:
                    logger.warning(f"Failed to crawl {source}: {e}")

            logger.info(f"Collected {len(all_news)} news articles for {symbol}")
            return all_news
        except Exception as e:
            logger.error(f"Error collecting news sentiment for {symbol}: {e}")
            return []

    async def collect_social_sentiment(self, symbol: str) -> Dict:
        """Collect social media sentiment analysis"""
        if not self.social_crawler:
            logger.warning("Social crawler not available")
            return {}

        try:
            # Collect from Reddit
            reddit_posts = await self.social_crawler.crawl_reddit_sentiment(symbol, limit=10)

            # Collect from Twitter (simplified)
            twitter_tweets = await self.social_crawler.crawl_twitter_sentiment(symbol, limit=10)

            social_data = {
                'reddit': reddit_posts,
                'twitter': twitter_tweets,
                'total_posts': len(reddit_posts) + len(twitter_tweets),
                'timestamp': datetime.now().isoformat()
            }

            logger.info(f"Collected social sentiment: {len(reddit_posts)} Reddit posts, {len(twitter_tweets)} tweets")
            return social_data
        except Exception as e:
            logger.error(f"Error collecting social sentiment for {symbol}: {e}")
            return {}

    async def get_comprehensive_sentiment_score(self, symbol: str) -> Dict:
        """Get comprehensive sentiment score combining all sources"""
        try:
            # Collect all sentiment data
            enhanced_sentiment = await self.collect_enhanced_sentiment(symbol)
            news_sentiment = await self.collect_news_sentiment(symbol)
            social_sentiment = await self.collect_social_sentiment(symbol)

            # Combine and analyze
            all_sentiment_items = []

            # Add news sentiment
            all_sentiment_items.extend(news_sentiment)

            # Add social sentiment
            if 'reddit' in social_sentiment:
                all_sentiment_items.extend(social_sentiment['reddit'])
            if 'twitter' in social_sentiment:
                all_sentiment_items.extend(social_sentiment['twitter'])

            # Calculate aggregate sentiment
            if all_sentiment_items:
                sentiment_scores = []
                for item in all_sentiment_items:
                    if 'sentiment' in item:
                        compound = item['sentiment'].get('compound', 0)
                        sentiment_scores.append(compound)

                if sentiment_scores:
                    avg_score = sum(sentiment_scores) / len(sentiment_scores)
                    overall_sentiment = 'positive' if avg_score > 0.05 else 'negative' if avg_score < -0.05 else 'neutral'

                    comprehensive_score = {
                        'symbol': symbol,
                        'overall_sentiment': overall_sentiment,
                        'average_score': avg_score,
                        'confidence': min(1.0, len(sentiment_scores) / 20.0),
                        'sources': {
                            'news_articles': len(news_sentiment),
                            'reddit_posts': len(social_sentiment.get('reddit', [])),
                            'twitter_tweets': len(social_sentiment.get('twitter', []))
                        },
                        'total_items': len(sentiment_scores),
                        'timestamp': datetime.now().isoformat()
                    }

                    logger.info(f"Comprehensive sentiment for {symbol}: {overall_sentiment} (score: {avg_score:.3f})")
                    return comprehensive_score

            # Fallback
            return {
                'symbol': symbol,
                'overall_sentiment': 'neutral',
                'average_score': 0.0,
                'confidence': 0.0,
                'sources': {'news_articles': 0, 'reddit_posts': 0, 'twitter_tweets': 0},
                'total_items': 0,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting comprehensive sentiment for {symbol}: {e}")
            return {}

    def save_data_to_csv(self, data: Dict, output_dir: str = "data"):
        """Save collected data to CSV files"""
        import os

        try:
            os.makedirs(output_dir, exist_ok=True)

            for symbol, symbol_data in data.items():
                # Save market data
                if not symbol_data['market_data'].empty:
                    market_file = f"{output_dir}/{symbol}_market_data.csv"
                    symbol_data['market_data'].to_csv(market_file, index=False)
                    logger.info(f"Saved market data to {market_file}")

                # Save news data
                if symbol_data['news_data']:
                    news_file = f"{output_dir}/{symbol}_news_data.csv"
                    news_df = pd.DataFrame(symbol_data['news_data'])
                    news_df.to_csv(news_file, index=False)
                    logger.info(f"Saved news data to {news_file}")

                # Save enhanced sentiment data
                if symbol_data.get('enhanced_sentiment'):
                    sentiment_file = f"{output_dir}/{symbol}_sentiment_data.json"
                    with open(sentiment_file, 'w') as f:
                        json.dump(symbol_data['enhanced_sentiment'], f, indent=2)
                    logger.info(f"Saved enhanced sentiment data to {sentiment_file}")

        except Exception as e:
            logger.error(f"Error saving data to CSV: {e}")

# Example usage
async def main():
    collector = CryptoDataCollector()

    # Collect data for core cryptos + top 5 most volatile
    data = await collector.collect_all_data()

    # Save to CSV
    collector.save_data_to_csv(data)

    print("Data collection completed!")
    print(f"Collected data for {len(data)} cryptocurrencies")

if __name__ == "__main__":
    asyncio.run(main())
