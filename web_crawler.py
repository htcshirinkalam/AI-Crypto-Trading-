#!/usr/bin/env python3
"""
Enhanced Web Crawler for Sentiment Analysis and News Aggregation
=================================================================

This module provides web crawling capabilities to complement APILayer APIs,
focusing on sentiment data collection from social media and additional news sources.

Features:
- Social media sentiment analysis (Twitter, Reddit)
- News aggregation from additional sources
- Rate limiting and anti-bot measures
- Data quality validation
- Integration with existing data collection pipeline
"""

import asyncio
import aiohttp
import requests
from bs4 import BeautifulSoup
import json
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from urllib.parse import urljoin, urlparse
import logging
from loguru import logger
from config import Config
import pandas as pd
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download NLTK data if not present
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

class SentimentAnalyzer:
    """Advanced sentiment analysis using multiple methods"""

    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def analyze_text(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using multiple methods"""
        if not text or not text.strip():
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}

        # NLTK VADER analysis
        vader_scores = self.sia.polarity_scores(text)

        # TextBlob analysis
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity

        # Combine results
        combined_score = {
            'compound': vader_scores['compound'],
            'positive': vader_scores['pos'],
            'negative': vader_scores['neg'],
            'neutral': vader_scores['neu'],
            'textblob_polarity': textblob_polarity,
            'textblob_subjectivity': textblob_subjectivity,
            'overall_sentiment': 'positive' if vader_scores['compound'] > 0.05 else 'negative' if vader_scores['compound'] < -0.05 else 'neutral'
        }

        return combined_score

class WebCrawler:
    """Enhanced web crawler for sentiment data and news aggregation"""

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.session = None
        self.user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'

        # Rate limiting
        self.request_delay = 1.0  # seconds between requests
        self.last_request_time = 0

        # Crawling limits
        self.max_pages_per_source = 5
        self.max_retries = 3
        self.timeout = 10

        # Data quality filters
        self.min_content_length = 50
        self.max_content_length = 10000

    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session with proper headers"""
        if self.session is None:
            headers = {
                'User-Agent': self.user_agent,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self.session = aiohttp.ClientSession(headers=headers, timeout=timeout)
        return self.session

    async def close_session(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None

    async def _rate_limit_wait(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.request_delay:
            await asyncio.sleep(self.request_delay - time_since_last)
        self.last_request_time = time.time()

    async def fetch_page(self, url: str, retries: int = None) -> Optional[str]:
        """Fetch a web page with retry logic and rate limiting"""
        if retries is None:
            retries = self.max_retries

        for attempt in range(retries):
            try:
                await self._rate_limit_wait()
                session = await self.get_session()

                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.text()
                        logger.debug(f"Successfully fetched: {url}")
                        return content
                    elif response.status == 429:  # Rate limited
                        wait_time = 2 ** attempt  # Exponential backoff
                        logger.warning(f"Rate limited, waiting {wait_time}s before retry")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.warning(f"HTTP {response.status} for {url}")
                        return None

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(1)
                continue

        logger.error(f"Failed to fetch {url} after {retries} attempts")
        return None

    def extract_text_content(self, html: str, selector: str = None) -> str:
        """Extract clean text content from HTML"""
        try:
            soup = BeautifulSoup(html, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Try to find main content
            if selector:
                content = soup.select_one(selector)
                if content:
                    return content.get_text(separator=' ', strip=True)

            # Fallback: find largest text block
            paragraphs = soup.find_all('p')
            if paragraphs:
                text_blocks = [p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20]
                return ' '.join(text_blocks)

            # Last resort: get all text
            return soup.get_text(separator=' ', strip=True)

        except Exception as e:
            logger.error(f"Error extracting text content: {e}")
            return ""

    def validate_content(self, content: str) -> bool:
        """Validate content quality"""
        if not content or len(content) < self.min_content_length:
            return False
        if len(content) > self.max_content_length:
            return False
        if len(content.split()) < 10:  # Minimum word count
            return False
        return True

class NewsCrawler(WebCrawler):
    """Specialized crawler for news aggregation"""

    def __init__(self, config: Config = None):
        super().__init__(config)
        self.news_sources = {
            'coinmarketcap': {
                'url': 'https://www.coinmarketcap.com',
                'article_selector': 'article',
                'title_selector': 'h1, h2, h3',
                'content_selector': '.article-content, .post-content'
            },
            'cryptonews': {
                'url': 'https://cryptonews.com',
                'article_selector': 'article',
                'title_selector': 'h1, h2',
                'content_selector': '.article, .post-content, .content'
            },
            'coincodex': {
                'url': 'https://coincodex.com/news/',
                'article_selector': 'article',
                'title_selector': 'h1, h2',
                'content_selector': '.article, .post-content, .content'
            }
        }

    async def crawl_news_source(self, source_name: str, symbol: str = None, max_articles: int = 5) -> List[Dict]:
        """Crawl news from a specific source"""
        if source_name not in self.news_sources:
            logger.error(f"Unknown news source: {source_name}")
            return []

        source_config = self.news_sources[source_name]
        articles = []

        try:
            # Fetch main page
            html = await self.fetch_page(source_config['url'])
            if not html:
                return articles

            soup = BeautifulSoup(html, 'html.parser')

            # Find article links
            article_links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                if 'article' in href.lower() or 'news' in href.lower() or 'post' in href.lower():
                    full_url = urljoin(source_config['url'], href)
                    if full_url not in [a['url'] for a in articles]:  # Avoid duplicates
                        article_links.append(full_url)

            # Limit to max_articles
            article_links = article_links[:max_articles]

            # Crawl individual articles
            for url in article_links:
                article_data = await self.crawl_article(url, source_config, symbol)
                if article_data:
                    articles.append(article_data)
                    await asyncio.sleep(0.5)  # Be respectful

        except Exception as e:
            logger.error(f"Error crawling {source_name}: {e}")

        return articles

    async def crawl_article(self, url: str, source_config: Dict, symbol: str = None) -> Optional[Dict]:
        """Crawl individual article"""
        try:
            html = await self.fetch_page(url)
            if not html:
                return None

            soup = BeautifulSoup(html, 'html.parser')

            # Extract title
            title_elem = soup.select_one(source_config.get('title_selector', 'h1, h2, h3'))
            title = title_elem.get_text(strip=True) if title_elem else "No title"

            # Extract content
            content = self.extract_text_content(html, source_config.get('content_selector'))

            if not self.validate_content(content):
                return None

            # Analyze sentiment
            sentiment = self.sentiment_analyzer.analyze_text(content)

            # Check if article is relevant to symbol
            relevant = True
            if symbol:
                symbol_lower = symbol.lower()
                content_lower = content.lower()
                title_lower = title.lower()
                relevant = (symbol_lower in content_lower or
                           symbol_lower in title_lower or
                           symbol_lower.replace('btc', 'bitcoin') in content_lower or
                           symbol_lower.replace('eth', 'ethereum') in content_lower)

            if not relevant:
                return None

            article_data = {
                'title': title,
                'content': content[:1000] + '...' if len(content) > 1000 else content,
                'url': url,
                'source': list(source_config.keys())[0] if source_config else 'unknown',
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol or 'general',
                'sentiment': sentiment,
                'word_count': len(content.split()),
                'relevance_score': 1.0 if relevant else 0.0
            }

            return article_data

        except Exception as e:
            logger.error(f"Error crawling article {url}: {e}")
            return None

class SocialMediaCrawler(WebCrawler):
    """Crawler for social media sentiment analysis"""

    def __init__(self, config: Config = None):
        super().__init__(config)
        self.reddit_base_url = 'https://www.reddit.com'
        self.twitter_search_url = 'https://twitter.com/search'

    async def crawl_reddit_sentiment(self, symbol: str, subreddit: str = 'cryptocurrency', limit: int = 10) -> List[Dict]:
        """Crawl Reddit for sentiment analysis"""
        posts = []

        try:
            # Reddit API endpoint for subreddit posts
            url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit={limit * 2}"

            session = await self.get_session()
            await self._rate_limit_wait()

            async with session.get(url, headers={'User-Agent': self.user_agent}) as response:
                if response.status == 200:
                    data = await response.json()
                    posts_data = data.get('data', {}).get('children', [])

                    for post in posts_data[:limit]:
                        post_data = post.get('data', {})
                        title = post_data.get('title', '')
                        selftext = post_data.get('selftext', '')

                        # Combine title and content
                        full_text = f"{title} {selftext}".strip()

                        if self.validate_content(full_text):
                            sentiment = self.sentiment_analyzer.analyze_text(full_text)

                            post_info = {
                                'title': title,
                                'content': selftext,
                                'url': f"https://reddit.com{post_data.get('permalink', '')}",
                                'source': 'reddit',
                                'subreddit': subreddit,
                                'timestamp': datetime.fromtimestamp(post_data.get('created_utc', 0)).isoformat(),
                                'symbol': symbol,
                                'sentiment': sentiment,
                                'score': post_data.get('score', 0),
                                'num_comments': post_data.get('num_comments', 0),
                                'upvote_ratio': post_data.get('upvote_ratio', 0)
                            }
                            posts.append(post_info)

        except Exception as e:
            logger.error(f"Error crawling Reddit for {symbol}: {e}")

        return posts

    async def crawl_twitter_sentiment(self, symbol: str, limit: int = 10) -> List[Dict]:
        """Crawl Twitter for sentiment analysis (using search)"""
        tweets = []

        try:
            # Note: This is a simplified approach. In production, you'd use Twitter API v2
            # For now, we'll use a basic search approach (may be limited by Twitter)

            search_query = f"{symbol} cryptocurrency OR crypto"
            url = f"https://twitter.com/search?q={search_query}&src=typed_query&f=live"

            html = await self.fetch_page(url)
            if html:
                soup = BeautifulSoup(html, 'html.parser')

                # Extract tweet text (this selector may change)
                tweet_elements = soup.find_all('div', {'data-testid': 'tweetText'})

                for tweet_elem in tweet_elements[:limit]:
                    tweet_text = tweet_elem.get_text(strip=True)

                    if self.validate_content(tweet_text):
                        sentiment = self.sentiment_analyzer.analyze_text(tweet_text)

                        tweet_data = {
                            'content': tweet_text,
                            'source': 'twitter',
                            'timestamp': datetime.now().isoformat(),
                            'symbol': symbol,
                            'sentiment': sentiment,
                            'url': 'https://twitter.com/search'  # Would need actual tweet URL
                        }
                        tweets.append(tweet_data)

        except Exception as e:
            logger.error(f"Error crawling Twitter for {symbol}: {e}")

        return tweets

class HybridDataCollector:
    """Main class that combines API and crawling approaches"""

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.news_crawler = NewsCrawler(self.config)
        self.social_crawler = SocialMediaCrawler(self.config)

    async def collect_comprehensive_data(self, symbol: str) -> Dict[str, Any]:
        """Collect data using both APIs and crawling"""
        comprehensive_data = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'api_data': {},
            'crawled_data': {
                'news': [],
                'social_sentiment': [],
                'aggregated_sentiment': {}
            }
        }

        try:
            # Collect news from multiple sources
            news_sources = ['coinmarketcap', 'cryptonews', 'coincodex']
            all_news = []

            for source in news_sources:
                try:
                    news = await self.news_crawler.crawl_news_source(source, symbol, max_articles=3)
                    all_news.extend(news)
                    await asyncio.sleep(0.5)  # Be respectful
                except Exception as e:
                    logger.warning(f"Failed to crawl {source}: {e}")

            comprehensive_data['crawled_data']['news'] = all_news

            # Collect social media sentiment
            reddit_posts = await self.social_crawler.crawl_reddit_sentiment(symbol, limit=5)
            twitter_tweets = await self.social_crawler.crawl_twitter_sentiment(symbol, limit=5)

            comprehensive_data['crawled_data']['social_sentiment'] = reddit_posts + twitter_tweets

            # Aggregate sentiment analysis
            comprehensive_data['crawled_data']['aggregated_sentiment'] = self._aggregate_sentiment(
                all_news + reddit_posts + twitter_tweets
            )

        except Exception as e:
            logger.error(f"Error in comprehensive data collection: {e}")

        finally:
            await self.news_crawler.close_session()
            await self.social_crawler.close_session()

        return comprehensive_data

    def _aggregate_sentiment(self, data_items: List[Dict]) -> Dict[str, Any]:
        """Aggregate sentiment from multiple sources"""
        if not data_items:
            return {'overall_sentiment': 'neutral', 'confidence': 0.0, 'sources_count': 0}

        sentiment_scores = []
        source_counts = {'news': 0, 'reddit': 0, 'twitter': 0}

        for item in data_items:
            if 'sentiment' in item:
                sentiment = item['sentiment']
                compound_score = sentiment.get('compound', 0)
                sentiment_scores.append(compound_score)

                source = item.get('source', 'unknown')
                if source in source_counts:
                    source_counts[source] += 1

        if not sentiment_scores:
            return {'overall_sentiment': 'neutral', 'confidence': 0.0, 'sources_count': sum(source_counts.values())}

        # Calculate aggregate metrics
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        sentiment_std = pd.Series(sentiment_scores).std() if len(sentiment_scores) > 1 else 0

        # Determine overall sentiment
        if avg_sentiment > 0.1:
            overall = 'positive'
        elif avg_sentiment < -0.1:
            overall = 'negative'
        else:
            overall = 'neutral'

        # Calculate confidence based on consistency and sample size
        confidence = min(1.0, len(sentiment_scores) / 20.0) * (1.0 - min(sentiment_std, 0.5))

        return {
            'overall_sentiment': overall,
            'average_score': avg_sentiment,
            'standard_deviation': sentiment_std,
            'confidence': confidence,
            'sources_count': sum(source_counts.values()),
            'source_breakdown': source_counts,
            'sample_size': len(sentiment_scores)
        }

# Convenience functions
async def collect_sentiment_data(symbol: str) -> Dict[str, Any]:
    """Convenience function to collect sentiment data"""
    collector = HybridDataCollector()
    return await collector.collect_comprehensive_data(symbol)

async def get_news_sentiment(symbol: str) -> Dict[str, Any]:
    """Get news-based sentiment analysis"""
    collector = HybridDataCollector()
    data = await collector.collect_comprehensive_data(symbol)
    return data['crawled_data']['aggregated_sentiment']

if __name__ == "__main__":
    # Example usage
    async def main():
        symbol = "BTC"
        collector = HybridDataCollector()

        print(f"Collecting comprehensive data for {symbol}...")
        data = await collector.collect_comprehensive_data(symbol)

        print(f"Found {len(data['crawled_data']['news'])} news articles")
        print(f"Found {len(data['crawled_data']['social_sentiment'])} social media posts")
        print(f"Overall sentiment: {data['crawled_data']['aggregated_sentiment']['overall_sentiment']}")
        print(f"Confidence: {data['crawled_data']['aggregated_sentiment']['confidence']:.2f}")

    asyncio.run(main())