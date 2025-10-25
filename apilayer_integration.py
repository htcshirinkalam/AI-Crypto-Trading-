#!/usr/bin/env python3
"""
APILayer Integration Module - Optimized
========================================

This module provides comprehensive integration with APILayer services,
with advanced debugging, optimization, and error handling capabilities.

Features:
- Exponential backoff retry logic
- Response caching with TTL
- Sliding window rate limiting
- Request batching
- Connection pooling optimization
- Comprehensive logging and metrics
- Circuit breaker pattern
- Automatic failover
"""

import asyncio
import aiohttp
import json
import os
import time
import hashlib
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from collections import deque, defaultdict
from functools import wraps
import logging
from loguru import logger
from config import Config

# Configure detailed logging
logger.add(
    "apilayer_debug.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
)


class APIMetrics:
    """Track API performance metrics"""

    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_latency = 0.0
        self.request_latencies = deque(maxlen=100)
        self.errors_by_type = defaultdict(int)
        self.requests_by_endpoint = defaultdict(int)
        self.start_time = datetime.now()

    def record_request(self, endpoint: str, latency: float, success: bool, error_type: str = None):
        """Record request metrics"""
        self.total_requests += 1
        self.requests_by_endpoint[endpoint] += 1
        self.total_latency += latency
        self.request_latencies.append(latency)

        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
            if error_type:
                self.errors_by_type[error_type] += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        uptime = (datetime.now() - self.start_time).total_seconds()
        avg_latency = self.total_latency / max(self.total_requests, 1)

        return {
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate': self.successful_requests / max(self.total_requests, 1),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': self.cache_hits / max(self.cache_hits + self.cache_misses, 1),
            'average_latency_ms': avg_latency * 1000,
            'requests_per_second': self.total_requests / max(uptime, 1),
            'errors_by_type': dict(self.errors_by_type),
            'requests_by_endpoint': dict(self.requests_by_endpoint),
            'uptime_seconds': uptime
        }


class ResponseCache:
    """Response caching with TTL"""

    def __init__(self, default_ttl: int = 300):
        self.cache = {}
        self.default_ttl = default_ttl

    def _make_key(self, endpoint: str, params: Dict) -> str:
        """Create cache key from endpoint and parameters"""
        params_str = json.dumps(params or {}, sort_keys=True)
        key_str = f"{endpoint}:{params_str}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """Get cached response if not expired"""
        key = self._make_key(endpoint, params)
        if key in self.cache:
            data, expiry = self.cache[key]
            if time.time() < expiry:
                logger.debug(f"Cache hit for {endpoint}")
                return data
            else:
                # Expired, remove from cache
                del self.cache[key]
                logger.debug(f"Cache expired for {endpoint}")
        return None

    def set(self, endpoint: str, params: Dict, data: Dict, ttl: int = None):
        """Cache response with TTL"""
        key = self._make_key(endpoint, params)
        expiry = time.time() + (ttl or self.default_ttl)
        self.cache[key] = (data, expiry)
        logger.debug(f"Cached response for {endpoint}, TTL: {ttl or self.default_ttl}s")

    def clear(self):
        """Clear all cached responses"""
        self.cache.clear()
        logger.info("Cache cleared")

    def get_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            'total_entries': len(self.cache),
            'memory_usage_bytes': sum(len(json.dumps(data)) for data, _ in self.cache.values())
        }


class RateLimiter:
    """Sliding window rate limiter"""

    def __init__(self, requests_per_minute: int = 100, requests_per_hour: int = 1000):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.minute_window = deque(maxlen=requests_per_minute)
        self.hour_window = deque(maxlen=requests_per_hour)

    async def acquire(self):
        """Wait if rate limit would be exceeded"""
        now = time.time()

        # Clean old entries from minute window
        while self.minute_window and now - self.minute_window[0] > 60:
            self.minute_window.popleft()

        # Clean old entries from hour window
        while self.hour_window and now - self.hour_window[0] > 3600:
            self.hour_window.popleft()

        # Check minute limit
        if len(self.minute_window) >= self.requests_per_minute:
            sleep_time = 60 - (now - self.minute_window[0])
            if sleep_time > 0:
                logger.warning(f"Minute rate limit reached, sleeping for {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)

        # Check hour limit
        if len(self.hour_window) >= self.requests_per_hour:
            sleep_time = 3600 - (now - self.hour_window[0])
            if sleep_time > 0:
                logger.warning(f"Hour rate limit reached, sleeping for {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)

        # Record request
        self.minute_window.append(now)
        self.hour_window.append(now)

    def get_stats(self) -> Dict:
        """Get rate limiter statistics"""
        now = time.time()
        recent_minute = sum(1 for t in self.minute_window if now - t <= 60)
        recent_hour = sum(1 for t in self.hour_window if now - t <= 3600)

        return {
            'requests_last_minute': recent_minute,
            'requests_last_hour': recent_hour,
            'minute_limit': self.requests_per_minute,
            'hour_limit': self.requests_per_hour,
            'minute_remaining': self.requests_per_minute - recent_minute,
            'hour_remaining': self.requests_per_hour - recent_hour
        }


class CircuitBreaker:
    """Circuit breaker pattern for API failures"""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half_open

    def record_success(self):
        """Record successful request"""
        if self.state == 'half_open':
            logger.info("Circuit breaker: service recovered, closing circuit")
            self.state = 'closed'
            self.failures = 0
            self.last_failure_time = None

    def record_failure(self):
        """Record failed request"""
        self.failures += 1
        self.last_failure_time = time.time()

        if self.failures >= self.failure_threshold and self.state == 'closed':
            self.state = 'open'
            logger.error(f"Circuit breaker: opened after {self.failures} failures")

    def can_request(self) -> bool:
        """Check if requests are allowed"""
        if self.state == 'closed':
            return True

        if self.state == 'open':
            # Check if recovery timeout has passed
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = 'half_open'
                logger.info("Circuit breaker: attempting recovery (half-open)")
                return True
            return False

        # half_open state
        return True

    def get_state(self) -> Dict:
        """Get circuit breaker state"""
        return {
            'state': self.state,
            'failures': self.failures,
            'threshold': self.failure_threshold,
            'can_request': self.can_request()
        }


class APILayerIntegration:
    """Optimized APILayer integration with debugging and performance features"""

    def __init__(self, api_key: str = None):
        """
        Initialize APILayer integration

        Args:
            api_key: APILayer API key (optional, can be set via environment)
        """
        self.api_key = api_key or os.getenv('APILAYER_API_KEY', '')
        self.base_url = "https://api.apilayer.com"
        self.session = None
        self.config = Config()

        # Initialize components
        self.metrics = APIMetrics()
        self.cache = ResponseCache(default_ttl=300)  # 5 minutes default
        self.rate_limiter = RateLimiter(requests_per_minute=100, requests_per_hour=1000)
        self.circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)

        # API endpoints with custom configurations
        self.endpoints = {
            'cryptocurrency': {
                'coinmarketcap': {'path': '/coinmarketcap', 'cache_ttl': 60},
                'coingecko': {'path': '/coingecko', 'cache_ttl': 60},
                'cryptocompare': {'path': '/cryptocompare', 'cache_ttl': 60}
            },
            'currency': {
                'exchange_rates': {'path': '/exchangerates_data', 'cache_ttl': 300},
                'currency_data': {'path': '/currency_data', 'cache_ttl': 300}
            },
            'financial': {
                'polygon': {'path': '/polygon', 'cache_ttl': 60},
                'alpha_vantage': {'path': '/alpha_vantage', 'cache_ttl': 300}
            },
            'news': {
                'news_api': {'path': '/news', 'cache_ttl': 600},
                'mediastack': {'path': '/mediastack', 'cache_ttl': 600}
            }
        }

        # Retry configuration
        self.max_retries = 3
        self.retry_delay = 1.0
        self.retry_backoff = 2.0

        if not self.api_key:
            logger.warning("APILayer API key not provided. Set APILAYER_API_KEY environment variable.")

    async def get_session(self):
        """Get or create optimized aiohttp session"""
        if self.session is None:
            # Optimized connection pooling
            connector = aiohttp.TCPConnector(
                limit=100,  # Maximum concurrent connections
                limit_per_host=30,  # Max connections per host
                ttl_dns_cache=300,  # DNS cache TTL
                enable_cleanup_closed=True
            )

            timeout = aiohttp.ClientTimeout(
                total=30,
                connect=10,
                sock_read=20
            )

            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    'apikey': self.api_key,
                    'Content-Type': 'application/json',
                    'User-Agent': 'CryptoTradingBot/1.0'
                }
            )
            logger.info("Initialized optimized aiohttp session")

        return self.session

    async def close_session(self):
        """Close aiohttp session gracefully"""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("Closed aiohttp session")

    async def _make_request_with_retry(self, endpoint: str, params: Dict = None,
                                      method: str = 'GET', use_cache: bool = True,
                                      cache_ttl: int = None) -> Dict:
        """
        Make request with retry logic and exponential backoff

        Args:
            endpoint: API endpoint
            params: Query parameters
            method: HTTP method
            use_cache: Whether to use response cache
            cache_ttl: Custom cache TTL

        Returns:
            API response data
        """
        # Check circuit breaker
        if not self.circuit_breaker.can_request():
            raise Exception("Circuit breaker is open - service unavailable")

        # Check cache first
        if use_cache and method == 'GET':
            cached_response = self.cache.get(endpoint, params or {})
            if cached_response:
                self.metrics.cache_hits += 1
                return cached_response
            self.metrics.cache_misses += 1

        # Retry loop with exponential backoff
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                # Rate limiting
                await self.rate_limiter.acquire()

                # Make request
                start_time = time.time()
                response_data = await self._make_request(endpoint, params, method)
                latency = time.time() - start_time

                # Record success
                self.metrics.record_request(endpoint, latency, True)
                self.circuit_breaker.record_success()

                # Cache successful response
                if use_cache and method == 'GET':
                    self.cache.set(endpoint, params or {}, response_data, cache_ttl)

                logger.debug(f"Request successful: {endpoint} (attempt {attempt + 1}, latency: {latency:.3f}s)")
                return response_data

            except Exception as e:
                last_exception = e
                latency = time.time() - start_time

                # Record failure
                error_type = type(e).__name__
                self.metrics.record_request(endpoint, latency, False, error_type)
                self.circuit_breaker.record_failure()

                logger.warning(f"Request failed: {endpoint} (attempt {attempt + 1}/{self.max_retries}): {e}")

                # Don't retry on certain errors
                if isinstance(e, ValueError) and 'Invalid APILayer API key' in str(e):
                    raise

                # Exponential backoff
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (self.retry_backoff ** attempt)
                    logger.info(f"Retrying in {delay:.2f}s...")
                    await asyncio.sleep(delay)

        # All retries exhausted
        logger.error(f"All retry attempts exhausted for {endpoint}")
        raise last_exception

    async def _make_request(self, endpoint: str, params: Dict = None, method: str = 'GET') -> Dict:
        """
        Make single authenticated request to APILayer

        Args:
            endpoint: API endpoint
            params: Query parameters
            method: HTTP method

        Returns:
            API response data
        """
        session = await self.get_session()
        url = f"{self.base_url}{endpoint}"

        logger.debug(f"Making {method} request to {url} with params: {params}")

        try:
            if method.upper() == 'GET':
                async with session.get(url, params=params) as response:
                    return await self._handle_response(response, endpoint)
            elif method.upper() == 'POST':
                async with session.post(url, json=params) as response:
                    return await self._handle_response(response, endpoint)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

        except asyncio.TimeoutError:
            logger.error(f"Request timeout for {endpoint}")
            raise ValueError(f"Request timeout for {endpoint}")
        except aiohttp.ClientError as e:
            logger.error(f"Client error for {endpoint}: {e}")
            raise ValueError(f"Client error: {e}")

    async def _handle_response(self, response, endpoint: str) -> Dict:
        """Handle API response with detailed error logging"""
        logger.debug(f"Response status: {response.status} for {endpoint}")

        if response.status == 200:
            try:
                data = await response.json()
                logger.debug(f"Response data received: {len(json.dumps(data))} bytes")
                return data
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error for {endpoint}: {e}")
                raise ValueError(f"Invalid JSON response: {e}")

        elif response.status == 401:
            error_msg = "Invalid APILayer API key - check your credentials"
            logger.error(error_msg)
            raise ValueError(error_msg)

        elif response.status == 429:
            error_msg = "APILayer rate limit exceeded"
            logger.error(error_msg)
            raise ValueError(error_msg)

        elif response.status == 403:
            error_msg = "APILayer API access forbidden - check your subscription"
            logger.error(error_msg)
            raise ValueError(error_msg)

        elif response.status == 404:
            error_msg = f"Endpoint not found: {endpoint}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        elif response.status >= 500:
            error_text = await response.text()
            error_msg = f"APILayer server error {response.status}: {error_text}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        else:
            error_text = await response.text()
            error_msg = f"APILayer API error {response.status}: {error_text}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _get_endpoint_config(self, category: str, source: str) -> Dict:
        """Get endpoint configuration"""
        if category in self.endpoints and source in self.endpoints[category]:
            return self.endpoints[category][source]
        raise ValueError(f"Unknown endpoint: {category}/{source}")

    # Cryptocurrency APIs
    async def get_cryptocurrency_data(self, symbol: str, source: str = 'coingecko',
                                     use_cache: bool = True) -> Dict:
        """
        Get cryptocurrency data from various sources

        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC', 'ETH')
            source: Data source ('coingecko', 'coinmarketcap', 'cryptocompare')
            use_cache: Whether to use cached response

        Returns:
            Cryptocurrency data
        """
        config = self._get_endpoint_config('cryptocurrency', source)
        endpoint = f"{config['path']}/price"
        params = {'symbol': symbol}

        return await self._make_request_with_retry(
            endpoint, params, use_cache=use_cache, cache_ttl=config['cache_ttl']
        )

    async def get_crypto_historical_data(self, symbol: str, days: int = 30,
                                        source: str = 'coingecko', use_cache: bool = True) -> List[Dict]:
        """
        Get historical cryptocurrency data

        Args:
            symbol: Cryptocurrency symbol
            days: Number of days of historical data
            source: Data source
            use_cache: Whether to use cached response

        Returns:
            Historical price data
        """
        config = self._get_endpoint_config('cryptocurrency', source)
        endpoint = f"{config['path']}/historical"
        params = {'symbol': symbol, 'days': days}

        return await self._make_request_with_retry(
            endpoint, params, use_cache=use_cache, cache_ttl=config['cache_ttl']
        )

    async def get_crypto_market_data(self, limit: int = 100, source: str = 'coingecko',
                                    use_cache: bool = True) -> List[Dict]:
        """
        Get cryptocurrency market data

        Args:
            limit: Number of cryptocurrencies to retrieve
            source: Data source
            use_cache: Whether to use cached response

        Returns:
            Market data for multiple cryptocurrencies
        """
        config = self._get_endpoint_config('cryptocurrency', source)
        endpoint = f"{config['path']}/markets"
        params = {'limit': limit}

        return await self._make_request_with_retry(
            endpoint, params, use_cache=use_cache, cache_ttl=config['cache_ttl']
        )

    async def get_batch_crypto_data(self, symbols: List[str], source: str = 'coingecko') -> Dict[str, Dict]:
        """
        Get cryptocurrency data for multiple symbols efficiently

        Args:
            symbols: List of cryptocurrency symbols
            source: Data source

        Returns:
            Dictionary mapping symbols to their data
        """
        logger.info(f"Fetching batch crypto data for {len(symbols)} symbols")

        # Use asyncio.gather for concurrent requests
        tasks = [self.get_cryptocurrency_data(symbol, source) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        batch_data = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to fetch data for {symbol}: {result}")
                batch_data[symbol] = {'error': str(result)}
            else:
                batch_data[symbol] = result

        return batch_data

    # Currency Exchange APIs
    async def get_exchange_rates(self, base_currency: str = 'USD',
                                symbols: List[str] = None, use_cache: bool = True) -> Dict:
        """
        Get currency exchange rates

        Args:
            base_currency: Base currency code
            symbols: List of target currency codes
            use_cache: Whether to use cached response

        Returns:
            Exchange rates data
        """
        config = self._get_endpoint_config('currency', 'exchange_rates')
        endpoint = config['path']
        params = {'base': base_currency}

        if symbols:
            params['symbols'] = ','.join(symbols)

        return await self._make_request_with_retry(
            endpoint, params, use_cache=use_cache, cache_ttl=config['cache_ttl']
        )

    async def convert_currency(self, from_currency: str, to_currency: str,
                              amount: float, use_cache: bool = True) -> Dict:
        """
        Convert currency amount

        Args:
            from_currency: Source currency code
            to_currency: Target currency code
            amount: Amount to convert
            use_cache: Whether to use cached response

        Returns:
            Conversion result
        """
        config = self._get_endpoint_config('currency', 'exchange_rates')
        endpoint = f"{config['path']}/convert"
        params = {
            'from': from_currency,
            'to': to_currency,
            'amount': amount
        }

        return await self._make_request_with_retry(
            endpoint, params, use_cache=use_cache, cache_ttl=config['cache_ttl']
        )

    # Financial Market Data APIs
    async def get_stock_data(self, symbol: str, source: str = 'alpha_vantage',
                            use_cache: bool = True) -> Dict:
        """
        Get stock market data

        Args:
            symbol: Stock symbol
            source: Data source
            use_cache: Whether to use cached response

        Returns:
            Stock data
        """
        config = self._get_endpoint_config('financial', source)
        endpoint = f"{config['path']}/stock"
        params = {'symbol': symbol}

        return await self._make_request_with_retry(
            endpoint, params, use_cache=use_cache, cache_ttl=config['cache_ttl']
        )

    async def get_polygon_data(self, symbol: str, timeframe: str = '1D',
                              use_cache: bool = True) -> Dict:
        """
        Get financial data from Polygon

        Args:
            symbol: Financial instrument symbol
            timeframe: Data timeframe
            use_cache: Whether to use cached response

        Returns:
            Financial data
        """
        config = self._get_endpoint_config('financial', 'polygon')
        endpoint = config['path']
        params = {'symbol': symbol, 'timeframe': timeframe}

        return await self._make_request_with_retry(
            endpoint, params, use_cache=use_cache, cache_ttl=config['cache_ttl']
        )

    # News APIs
    async def get_news(self, query: str = None, source: str = 'news_api',
                      language: str = 'en', page_size: int = 10, use_cache: bool = True) -> Dict:
        """
        Get news data

        Args:
            query: Search query
            source: News source
            language: News language
            page_size: Number of articles to retrieve
            use_cache: Whether to use cached response

        Returns:
            News data
        """
        config = self._get_endpoint_config('news', source)
        endpoint = config['path']
        params = {
            'language': language,
            'pageSize': page_size
        }

        if query:
            params['q'] = query

        return await self._make_request_with_retry(
            endpoint, params, use_cache=use_cache, cache_ttl=config['cache_ttl']
        )

    async def get_crypto_news(self, symbol: str = None, page_size: int = 10,
                             use_cache: bool = True) -> Dict:
        """
        Get cryptocurrency-related news

        Args:
            symbol: Specific cryptocurrency symbol
            page_size: Number of articles
            use_cache: Whether to use cached response

        Returns:
            Crypto news data
        """
        query = f"cryptocurrency {symbol}" if symbol else "cryptocurrency"
        return await self.get_news(query=query, page_size=page_size, use_cache=use_cache)

    # Utility methods
    def is_available(self) -> bool:
        """Check if APILayer integration is available"""
        return bool(self.api_key and self.api_key.strip())

    def get_supported_sources(self, category: str = None) -> Dict:
        """
        Get supported API sources

        Args:
            category: Specific category or None for all

        Returns:
            Dictionary of supported sources
        """
        if category:
            return {k: v['path'] for k, v in self.endpoints.get(category, {}).items()}
        return {cat: {k: v['path'] for k, v in sources.items()}
                for cat, sources in self.endpoints.items()}

    def get_metrics(self) -> Dict:
        """Get comprehensive performance metrics"""
        return {
            'api_metrics': self.metrics.get_stats(),
            'cache_stats': self.cache.get_stats(),
            'rate_limiter_stats': self.rate_limiter.get_stats(),
            'circuit_breaker_state': self.circuit_breaker.get_state()
        }

    def clear_cache(self):
        """Clear response cache"""
        self.cache.clear()
        logger.info("Cache cleared manually")

    def reset_circuit_breaker(self):
        """Manually reset circuit breaker"""
        self.circuit_breaker.state = 'closed'
        self.circuit_breaker.failures = 0
        self.circuit_breaker.last_failure_time = None
        logger.info("Circuit breaker manually reset")

    async def health_check(self) -> Dict:
        """
        Perform health check on APILayer integration

        Returns:
            Health check results
        """
        health_status = {
            'healthy': True,
            'api_key_configured': self.is_available(),
            'circuit_breaker_state': self.circuit_breaker.state,
            'metrics': self.get_metrics(),
            'timestamp': datetime.now().isoformat()
        }

        # Try a simple request to verify connectivity
        if self.is_available():
            try:
                await self.get_exchange_rates('USD', ['EUR'])
                health_status['connectivity'] = 'ok'
            except Exception as e:
                health_status['healthy'] = False
                health_status['connectivity'] = f'error: {str(e)}'
        else:
            health_status['connectivity'] = 'not configured'

        return health_status

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close_session()


# Convenience functions
async def get_crypto_data(symbol: str, source: str = 'coingecko') -> Dict:
    """Convenience function to get cryptocurrency data"""
    async with APILayerIntegration() as api:
        return await api.get_cryptocurrency_data(symbol, source)


async def get_exchange_rate(base: str = 'USD', symbols: List[str] = None) -> Dict:
    """Convenience function to get exchange rates"""
    async with APILayerIntegration() as api:
        return await api.get_exchange_rates(base, symbols)


async def get_financial_news(query: str = None) -> Dict:
    """Convenience function to get financial news"""
    async with APILayerIntegration() as api:
        return await api.get_news(query)


if __name__ == "__main__":
    # Example usage with debugging
    async def main():
        async with APILayerIntegration() as api:
            print("=== APILayer Integration Test ===\n")

            # Health check
            health = await api.health_check()
            print(f"Health Check: {json.dumps(health, indent=2)}\n")

            if api.is_available():
                try:
                    # Test cryptocurrency data
                    print("Fetching BTC data...")
                    btc_data = await api.get_cryptocurrency_data('BTC')
                    print(f"BTC Data: {json.dumps(btc_data, indent=2)}\n")
                    
                    # Test exchange rates
                    print("Fetching exchange rates...")
                    rates = await api.get_exchange_rates('USD', ['EUR', 'GBP'])
                    print(f"Exchange Rates: {json.dumps(rates, indent=2)}\n")
                    
                    # Test crypto news
                    print("Fetching crypto news...")
                    news = await api.get_crypto_news(page_size=5)
                    print(f"News: {json.dumps(news, indent=2)}\n")
                    
                except Exception as e:
                    print(f"Error during test: {e}")
            else:
                print("APILayer API key not configured")
            
            # Print metrics
            print("\n=== Performance Metrics ===")
            print(json.dumps(api.get_metrics(), indent=2))

    # Run the test
    asyncio.run(main())
