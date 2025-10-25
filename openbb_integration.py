#!/usr/bin/env python3
"""
OpenBB Platform Integration Module
==================================

This module provides comprehensive integration with the OpenBB platform,
enabling access to extensive financial data sources and analytical tools.

Features:
- Multi-source market data collection
- Advanced technical analysis
- Economic indicators and news
- Portfolio optimization tools
- Risk management analytics
- Real-time data streaming
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
from loguru import logger

try:
    from openbb import obb
    OPENBB_AVAILABLE = True
except ImportError:
    OPENBB_AVAILABLE = False
    logger.warning("OpenBB not available. Install with: pip install openbb")

class OpenBBIntegration:
    """OpenBB platform integration for enhanced financial data and analytics"""

    def __init__(self, api_key: str = None):
        """
        Initialize OpenBB integration

        Args:
            api_key: OpenBB API key (optional, can be set via environment)
        """
        self.api_key = api_key
        self.is_initialized = False
        self.available_providers = []

        if OPENBB_AVAILABLE:
            self._initialize_openbb()
        else:
            logger.warning("OpenBB integration disabled - package not installed")

    def _initialize_openbb(self):
        """Initialize OpenBB connection and check available providers"""
        try:
            # Set API key if provided
            if self.api_key:
                import os
                os.environ['OPENBB_API_KEY'] = self.api_key

            # Initialize OpenBB
            obb.account.login(pat=self.api_key) if self.api_key else None

            # Get available providers
            self.available_providers = self._get_available_providers()

            self.is_initialized = True
            logger.info(f"OpenBB integration initialized with {len(self.available_providers)} providers")

        except Exception as e:
            logger.error(f"Failed to initialize OpenBB: {e}")
            self.is_initialized = False

    def _get_available_providers(self) -> List[str]:
        """Get list of available data providers"""
        try:
            # This would typically query OpenBB for available providers
            # For now, return common providers
            return [
                'alpha_vantage', 'polygon', 'yfinance', 'fmp', 'intrinio',
                'fred', 'tiingo', 'tradier', 'oanda', 'binance', 'coinbase',
                'kraken', 'bitfinex', 'bitstamp', 'gemini'
            ]
        except Exception as e:
            logger.warning(f"Could not get providers: {e}")
            return []

    async def get_market_data(self, symbol: str, timeframe: str = '1D',
                            start_date: str = None, end_date: str = None,
                            provider: str = 'yfinance') -> pd.DataFrame:
        """
        Get market data from OpenBB

        Args:
            symbol: Trading symbol (e.g., 'AAPL', 'BTC-USD')
            timeframe: Data timeframe ('1D', '1H', '5m', etc.)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            provider: Data provider to use

        Returns:
            DataFrame with OHLCV data
        """
        if not self.is_initialized:
            logger.warning("OpenBB not initialized")
            return pd.DataFrame()

        try:
            # Set default dates if not provided
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

            # Fetch data using OpenBB
            params = {
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date,
                'interval': timeframe.lower(),
                'provider': provider
            }

            # Use OpenBB equity price endpoint
            if '-' in symbol or symbol in ['BTC', 'ETH', 'USDT', 'BNB', 'SOL']:
                # Crypto data
                result = obb.crypto.price.historical(**params)
            else:
                # Stock data
                result = obb.equity.price.historical(**params)

            if result and hasattr(result, 'results'):
                df = result.to_dataframe()
                if not df.empty:
                    # Standardize column names
                    column_mapping = {
                        'open': 'open',
                        'high': 'high',
                        'low': 'low',
                        'close': 'close',
                        'volume': 'volume',
                        'date': 'timestamp'
                    }
                    df = df.rename(columns=column_mapping)

                    # Ensure timestamp is datetime
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])

                    logger.info(f"Retrieved {len(df)} records for {symbol}")
                    return df

        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")

        return pd.DataFrame()

    async def get_real_time_quote(self, symbol: str, provider: str = 'yfinance') -> Dict:
        """
        Get real-time quote for a symbol

        Args:
            symbol: Trading symbol
            provider: Data provider

        Returns:
            Dictionary with quote data
        """
        if not self.is_initialized:
            return {}

        try:
            if '-' in symbol or symbol in ['BTC', 'ETH', 'USDT', 'BNB', 'SOL']:
                result = obb.crypto.price.quote(symbol=symbol, provider=provider)
            else:
                result = obb.equity.price.quote(symbol=symbol, provider=provider)

            if result and hasattr(result, 'results'):
                data = result.results
                if data:
                    return {
                        'symbol': symbol,
                        'price': data.get('last_price', data.get('price')),
                        'change': data.get('change'),
                        'change_percent': data.get('change_percent'),
                        'volume': data.get('volume'),
                        'timestamp': datetime.now().isoformat()
                    }

        except Exception as e:
            logger.error(f"Error getting real-time quote for {symbol}: {e}")

        return {}

    async def get_news_sentiment(self, symbol: str, limit: int = 10) -> List[Dict]:
        """
        Get news and sentiment analysis for a symbol

        Args:
            symbol: Trading symbol
            limit: Maximum number of news items

        Returns:
            List of news items with sentiment scores
        """
        if not self.is_initialized:
            return []

        try:
            # Get news data
            result = obb.news.company(symbol=symbol, limit=limit, provider='benzinga')

            if result and hasattr(result, 'results'):
                news_items = []
                for item in result.results[:limit]:
                    news_items.append({
                        'title': item.get('title', ''),
                        'content': item.get('content', ''),
                        'url': item.get('url', ''),
                        'published_at': item.get('published_at', ''),
                        'source': item.get('source', ''),
                        'sentiment_score': self._analyze_sentiment(item.get('title', '') + ' ' + item.get('content', ''))
                    })

                return news_items

        except Exception as e:
            logger.error(f"Error getting news for {symbol}: {e}")

        return []

    def _analyze_sentiment(self, text: str) -> float:
        """Simple sentiment analysis (placeholder for more advanced analysis)"""
        try:
            # This would use OpenBB's sentiment analysis if available
            # For now, return a neutral score
            return 0.0
        except Exception:
            return 0.0

    async def get_economic_indicators(self, indicators: List[str] = None) -> pd.DataFrame:
        """
        Get economic indicators data

        Args:
            indicators: List of economic indicators to fetch

        Returns:
            DataFrame with economic data
        """
        if not self.is_initialized:
            return pd.DataFrame()

        if indicators is None:
            indicators = ['GDP', 'CPI', 'UNRATE', 'FEDFUNDS']

        try:
            all_data = []

            for indicator in indicators:
                result = obb.economy.fred_series(symbol=indicator, provider='fred')

                if result and hasattr(result, 'results'):
                    df = result.to_dataframe()
                    if not df.empty:
                        df['indicator'] = indicator
                        all_data.append(df)

            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                return combined_df

        except Exception as e:
            logger.error(f"Error getting economic indicators: {e}")

        return pd.DataFrame()

    async def get_technical_analysis(self, symbol: str, indicators: List[str] = None) -> Dict:
        """
        Get technical analysis for a symbol

        Args:
            symbol: Trading symbol
            indicators: List of technical indicators to calculate

        Returns:
            Dictionary with technical analysis results
        """
        if not self.is_initialized:
            return {}

        if indicators is None:
            indicators = ['sma', 'ema', 'rsi', 'macd', 'bollinger']

        try:
            # Get price data first
            price_data = await self.get_market_data(symbol, '1D')

            if price_data.empty:
                return {}

            analysis_results = {}

            # Calculate technical indicators
            for indicator in indicators:
                if indicator == 'sma':
                    analysis_results['sma_20'] = price_data['close'].rolling(20).mean().iloc[-1]
                    analysis_results['sma_50'] = price_data['close'].rolling(50).mean().iloc[-1]
                elif indicator == 'ema':
                    analysis_results['ema_12'] = price_data['close'].ewm(span=12).mean().iloc[-1]
                    analysis_results['ema_26'] = price_data['close'].ewm(span=26).mean().iloc[-1]
                elif indicator == 'rsi':
                    delta = price_data['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                    rs = gain / loss
                    analysis_results['rsi'] = 100 - (100 / (1 + rs)).iloc[-1]
                elif indicator == 'macd':
                    ema12 = price_data['close'].ewm(span=12).mean()
                    ema26 = price_data['close'].ewm(span=26).mean()
                    analysis_results['macd'] = (ema12 - ema26).iloc[-1]
                    analysis_results['macd_signal'] = analysis_results['macd'].ewm(span=9).mean().iloc[-1]
                elif indicator == 'bollinger':
                    sma = price_data['close'].rolling(20).mean()
                    std = price_data['close'].rolling(20).std()
                    analysis_results['bb_upper'] = (sma + 2 * std).iloc[-1]
                    analysis_results['bb_lower'] = (sma - 2 * std).iloc[-1]
                    analysis_results['bb_middle'] = sma.iloc[-1]

            return analysis_results

        except Exception as e:
            logger.error(f"Error in technical analysis for {symbol}: {e}")
            return {}

    async def get_portfolio_analytics(self, portfolio: Dict[str, float],
                                    benchmark: str = 'SPY') -> Dict:
        """
        Get portfolio analytics and comparison to benchmark

        Args:
            portfolio: Dictionary of symbol -> weight pairs
            benchmark: Benchmark symbol

        Returns:
            Dictionary with portfolio analytics
        """
        if not self.is_initialized:
            return {}

        try:
            # Get historical data for portfolio components
            portfolio_data = []
            for symbol, weight in portfolio.items():
                data = await self.get_market_data(symbol, '1D')
                if not data.empty:
                    data['symbol'] = symbol
                    data['weight'] = weight
                    portfolio_data.append(data)

            if not portfolio_data:
                return {}

            # Calculate portfolio returns
            combined_data = pd.concat(portfolio_data)
            pivot_data = combined_data.pivot(columns='symbol', values='close')

            # Calculate weighted portfolio returns
            weights = np.array([portfolio[symbol] for symbol in pivot_data.columns])
            portfolio_returns = (pivot_data.pct_change() * weights).sum(axis=1)

            # Get benchmark data
            benchmark_data = await self.get_market_data(benchmark, '1D')
            if not benchmark_data.empty:
                benchmark_returns = benchmark_data['close'].pct_change()

                # Calculate performance metrics
                analytics = {
                    'portfolio_return': (portfolio_returns + 1).prod() - 1,
                    'benchmark_return': (benchmark_returns + 1).prod() - 1,
                    'alpha': portfolio_returns.mean() - benchmark_returns.mean(),
                    'beta': portfolio_returns.cov(benchmark_returns) / benchmark_returns.var(),
                    'sharpe_ratio': portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252),
                    'max_drawdown': self._calculate_max_drawdown(portfolio_returns)
                }

                return analytics

        except Exception as e:
            logger.error(f"Error in portfolio analytics: {e}")

        return {}

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        try:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return drawdown.min()
        except Exception:
            return 0.0

    async def get_market_overview(self) -> Dict:
        """
        Get comprehensive market overview

        Returns:
            Dictionary with market overview data
        """
        if not self.is_initialized:
            return {}

        try:
            # Get major indices
            indices = ['SPY', 'QQQ', 'IWM', 'VWO']
            index_data = {}

            for index in indices:
                quote = await self.get_real_time_quote(index)
                if quote:
                    index_data[index] = quote

            # Get sector performance
            sectors = ['XLE', 'XLF', 'XLK', 'XLV', 'XLY', 'XLI', 'XLB', 'XLRE', 'XLC', 'XLU']
            sector_data = {}

            for sector in sectors:
                quote = await self.get_real_time_quote(sector)
                if quote:
                    sector_data[sector] = quote

            # Get crypto overview
            crypto_symbols = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD']
            crypto_data = {}

            for symbol in crypto_symbols:
                quote = await self.get_real_time_quote(symbol, 'yfinance')
                if quote:
                    crypto_data[symbol] = quote

            return {
                'indices': index_data,
                'sectors': sector_data,
                'crypto': crypto_data,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting market overview: {e}")
            return {}

    def get_available_providers(self) -> List[str]:
        """Get list of available data providers"""
        return self.available_providers

    def is_available(self) -> bool:
        """Check if OpenBB integration is available and working"""
        return self.is_initialized and OPENBB_AVAILABLE


# Convenience functions for easy integration
async def get_openbb_data(symbol: str, **kwargs) -> pd.DataFrame:
    """Convenience function to get OpenBB data"""
    integration = OpenBBIntegration()
    return await integration.get_market_data(symbol, **kwargs)

async def get_openbb_quote(symbol: str, **kwargs) -> Dict:
    """Convenience function to get OpenBB quote"""
    integration = OpenBBIntegration()
    return await integration.get_real_time_quote(symbol, **kwargs)

async def get_openbb_technical_analysis(symbol: str, **kwargs) -> Dict:
    """Convenience function to get technical analysis"""
    integration = OpenBBIntegration()
    return await integration.get_technical_analysis(symbol, **kwargs)


if __name__ == "__main__":
    # Example usage
    async def main():
        integration = OpenBBIntegration()

        if integration.is_available():
            # Get market data
            data = await integration.get_market_data('AAPL', '1D')
            print(f"Retrieved {len(data)} records for AAPL")

            # Get real-time quote
            quote = await integration.get_real_time_quote('AAPL')
            print(f"AAPL Quote: ${quote.get('price', 'N/A')}")

            # Get technical analysis
            analysis = await integration.get_technical_analysis('AAPL')
            print(f"Technical Analysis: {analysis}")

        else:
            print("OpenBB integration not available")

    asyncio.run(main())