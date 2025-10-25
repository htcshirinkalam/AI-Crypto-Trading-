#!/usr/bin/env python3
"""
Secure Pipeline Wrapper
======================

This module provides a secure way to run the full trading pipeline
without exposing the core trading agent and optimization algorithms.
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
from loguru import logger

class SecureTradingPipeline:
    """
    Secure wrapper for the full trading pipeline.
    Provides pipeline functionality without exposing core algorithms.
    """
    
    def __init__(self):
        self.config = None
        self.data_collector = None
        self.sentiment_analyzer = None
        self.feature_engineer = None
        self.trading_strategy = None
        self.portfolio_manager = None
        self.risk_manager = None
        self.evaluation_framework = None
        
    async def initialize_components(self):
        """Initialize all available components"""
        try:
            from config import Config
            from data_collector import CryptoDataCollector
            from sentiment_analyzer import CryptoSentimentAnalyzer
            from feature_engineer import CryptoFeatureEngineer
            from trading_strategy import CryptoTradingStrategy
            from portfolio_manager import PortfolioManager
            from risk_manager import RiskManager
            from evaluation_metrics import AdvancedMetricsCalculator
            
            self.config = Config()
            self.data_collector = CryptoDataCollector()
            self.sentiment_analyzer = CryptoSentimentAnalyzer()
            self.feature_engineer = CryptoFeatureEngineer()
            self.trading_strategy = CryptoTradingStrategy()
            self.risk_manager = RiskManager()
            self.evaluation_framework = AdvancedMetricsCalculator()
            
            # Initialize portfolio manager with database
            from database.database_manager import DatabaseManager
            db_manager = DatabaseManager(self.config)
            self.portfolio_manager = PortfolioManager(db_manager)
            
            logger.info("✅ Secure pipeline components initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize components: {e}")
            return False
    
    async def run_full_pipeline(self, 
                              symbols: List[str] = ['BTC', 'ETH'],
                              retrain_models: bool = False,
                              timeframe: str = '1D',
                              model_variant: str = 'secure') -> Dict[str, Any]:
        """
        Run the complete trading pipeline securely
        
        Args:
            symbols: List of cryptocurrency symbols to analyze
            retrain_models: Whether to retrain models (not available in secure mode)
            timeframe: Prediction timeframe
            model_variant: Model variant to use
            
        Returns:
            Dictionary with pipeline results
        """
        try:
            logger.info(f"🚀 Starting secure pipeline for {symbols}")
            
            # Initialize components if not already done
            if not self.config:
                await self.initialize_components()
            
            results = {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'symbols': symbols,
                'timeframe': timeframe,
                'model_variant': model_variant,
                'steps_completed': [],
                'data': {},
                'signals': {},
                'portfolio': {},
                'risk_metrics': {},
                'performance': {}
            }
            
            # Step 1: Data Collection
            logger.info("📊 Step 1: Collecting market data...")
            market_data = await self._collect_market_data(symbols)
            results['data']['market_data'] = market_data
            results['steps_completed'].append('data_collection')
            
            # Step 2: Sentiment Analysis
            logger.info("💭 Step 2: Analyzing sentiment...")
            sentiment_data = await self._analyze_sentiment(symbols)
            results['data']['sentiment'] = sentiment_data
            results['steps_completed'].append('sentiment_analysis')
            
            # Step 3: Feature Engineering
            logger.info("🔧 Step 3: Engineering features...")
            features = await self._engineer_features(market_data)
            results['data']['features'] = features
            results['steps_completed'].append('feature_engineering')
            
            # Step 4: Signal Generation
            logger.info("🎯 Step 4: Generating trading signals...")
            signals = await self._generate_signals(features, sentiment_data)
            results['signals'] = signals
            results['steps_completed'].append('signal_generation')
            
            # Step 5: Risk Assessment
            logger.info("⚠️ Step 5: Assessing risk...")
            risk_metrics = await self._assess_risk(signals, market_data)
            results['risk_metrics'] = risk_metrics
            results['steps_completed'].append('risk_assessment')
            
            # Step 6: Portfolio Analysis
            logger.info("💼 Step 6: Analyzing portfolio...")
            portfolio_analysis = await self._analyze_portfolio(signals, risk_metrics)
            results['portfolio'] = portfolio_analysis
            results['steps_completed'].append('portfolio_analysis')
            
            # Step 7: Performance Evaluation
            logger.info("📈 Step 7: Evaluating performance...")
            performance = await self._evaluate_performance(results)
            results['performance'] = performance
            results['steps_completed'].append('performance_evaluation')
            
            logger.info("✅ Secure pipeline completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _collect_market_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Collect market data for symbols"""
        try:
            market_data = {}
            for symbol in symbols:
                # Get current price data
                price_data = await self.data_collector.get_current_price(symbol)
                # Get historical data
                historical_data = await self.data_collector.get_historical_data(symbol, days=30)
                
                market_data[symbol] = {
                    'current_price': price_data,
                    'historical_data': historical_data,
                    'timestamp': datetime.now().isoformat()
                }
            
            return market_data
        except Exception as e:
            logger.error(f"Error collecting market data: {e}")
            return {}
    
    async def _analyze_sentiment(self, symbols: List[str]) -> Dict[str, Any]:
        """Analyze sentiment for symbols"""
        try:
            sentiment_data = {}
            for symbol in symbols:
                # Get news sentiment
                news_sentiment = await self.sentiment_analyzer.analyze_news_sentiment(symbol)
                # Get social sentiment
                social_sentiment = await self.sentiment_analyzer.analyze_social_sentiment(symbol)
                
                sentiment_data[symbol] = {
                    'news_sentiment': news_sentiment,
                    'social_sentiment': social_sentiment,
                    'combined_sentiment': self._combine_sentiment(news_sentiment, social_sentiment),
                    'timestamp': datetime.now().isoformat()
                }
            
            return sentiment_data
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {}
    
    async def _engineer_features(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Engineer features from market data"""
        try:
            features = {}
            for symbol, data in market_data.items():
                if 'historical_data' in data and not data['historical_data'].empty:
                    # Calculate technical indicators
                    df = data['historical_data']
                    engineered_df = self.feature_engineer.engineer_features(df)
                    
                    features[symbol] = {
                        'technical_indicators': engineered_df,
                        'feature_count': len(engineered_df.columns),
                        'timestamp': datetime.now().isoformat()
                    }
            
            return features
        except Exception as e:
            logger.error(f"Error engineering features: {e}")
            return {}
    
    async def _generate_signals(self, features: Dict[str, Any], sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals"""
        try:
            signals = {}
            for symbol in features.keys():
                if symbol in features and symbol in sentiment_data:
                    # Generate signals using trading strategy
                    signal_df = self.trading_strategy.generate_trading_signals(
                        features[symbol]['technical_indicators'],
                        sentiment_data[symbol]
                    )
                    
                    # Get latest signal
                    latest_signal = signal_df.iloc[-1] if not signal_df.empty else {}
                    
                    signals[symbol] = {
                        'signals': signal_df.to_dict('records') if not signal_df.empty else [],
                        'latest_signal': latest_signal,
                        'recommendation': self._get_recommendation(latest_signal),
                        'confidence': latest_signal.get('confidence', 0.0),
                        'timestamp': datetime.now().isoformat()
                    }
            
            return signals
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return {}
    
    async def _assess_risk(self, signals: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess portfolio risk"""
        try:
            risk_metrics = {}
            for symbol in signals.keys():
                if symbol in market_data:
                    # Calculate risk metrics
                    risk_score = self.risk_manager.calculate_risk_score(
                        market_data[symbol]['current_price'],
                        signals[symbol]['latest_signal']
                    )
                    
                    risk_metrics[symbol] = {
                        'risk_score': risk_score,
                        'volatility': self._calculate_volatility(market_data[symbol]['historical_data']),
                        'recommendation': self._get_risk_recommendation(risk_score),
                        'timestamp': datetime.now().isoformat()
                    }
            
            return risk_metrics
        except Exception as e:
            logger.error(f"Error assessing risk: {e}")
            return {}
    
    async def _analyze_portfolio(self, signals: Dict[str, Any], risk_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze portfolio performance"""
        try:
            portfolio_analysis = {
                'total_value': 0.0,
                'positions': {},
                'allocation': {},
                'performance': {},
                'recommendations': []
            }
            
            # Get current portfolio
            if self.portfolio_manager:
                current_portfolio = self.portfolio_manager.get_portfolio_summary()
                portfolio_analysis.update(current_portfolio)
            
            # Generate recommendations based on signals and risk
            for symbol in signals.keys():
                if symbol in risk_metrics:
                    recommendation = self._generate_portfolio_recommendation(
                        signals[symbol], risk_metrics[symbol]
                    )
                    portfolio_analysis['recommendations'].append(recommendation)
            
            return portfolio_analysis
        except Exception as e:
            logger.error(f"Error analyzing portfolio: {e}")
            return {}
    
    async def _evaluate_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate overall performance"""
        try:
            performance = {
                'overall_score': 0.0,
                'signal_quality': 0.0,
                'risk_management': 0.0,
                'portfolio_health': 0.0,
                'recommendations': []
            }
            
            # Calculate overall performance score
            if results['signals']:
                avg_confidence = np.mean([
                    signal.get('confidence', 0.0) 
                    for signal in results['signals'].values() 
                    if 'confidence' in signal
                ])
                performance['signal_quality'] = avg_confidence
            
            if results['risk_metrics']:
                avg_risk = np.mean([
                    risk.get('risk_score', 0.0) 
                    for risk in results['risk_metrics'].values() 
                    if 'risk_score' in risk
                ])
                performance['risk_management'] = 1.0 - avg_risk  # Lower risk is better
            
            # Calculate overall score
            performance['overall_score'] = (
                performance['signal_quality'] * 0.4 +
                performance['risk_management'] * 0.3 +
                performance['portfolio_health'] * 0.3
            )
            
            return performance
        except Exception as e:
            logger.error(f"Error evaluating performance: {e}")
            return {}
    
    def _combine_sentiment(self, news_sentiment: Dict, social_sentiment: Dict) -> Dict:
        """Combine news and social sentiment"""
        try:
            news_score = news_sentiment.get('sentiment_score', 0.0)
            social_score = social_sentiment.get('sentiment_score', 0.0)
            
            combined_score = (news_score * 0.6 + social_score * 0.4)
            
            return {
                'sentiment_score': combined_score,
                'sentiment_label': 'positive' if combined_score > 0.1 else 'negative' if combined_score < -0.1 else 'neutral'
            }
        except:
            return {'sentiment_score': 0.0, 'sentiment_label': 'neutral'}
    
    def _get_recommendation(self, signal: Dict) -> str:
        """Get trading recommendation from signal"""
        try:
            buy_signal = signal.get('buy_signal', 0)
            sell_signal = signal.get('sell_signal', 0)
            confidence = signal.get('confidence', 0.0)
            
            if buy_signal > sell_signal and confidence > 0.7:
                return 'BUY'
            elif sell_signal > buy_signal and confidence > 0.7:
                return 'SELL'
            else:
                return 'HOLD'
        except:
            return 'HOLD'
    
    def _calculate_volatility(self, historical_data: pd.DataFrame) -> float:
        """Calculate volatility from historical data"""
        try:
            if 'close' in historical_data.columns:
                returns = historical_data['close'].pct_change().dropna()
                return returns.std() * np.sqrt(252)  # Annualized volatility
            return 0.0
        except:
            return 0.0
    
    def _get_risk_recommendation(self, risk_score: float) -> str:
        """Get risk recommendation"""
        if risk_score < 0.3:
            return 'LOW_RISK'
        elif risk_score < 0.7:
            return 'MEDIUM_RISK'
        else:
            return 'HIGH_RISK'
    
    def _generate_portfolio_recommendation(self, signal: Dict, risk: Dict) -> Dict:
        """Generate portfolio recommendation"""
        return {
            'symbol': signal.get('symbol', ''),
            'action': self._get_recommendation(signal),
            'confidence': signal.get('confidence', 0.0),
            'risk_level': risk.get('recommendation', 'MEDIUM_RISK'),
            'timestamp': datetime.now().isoformat()
        }

# Global instance for easy access
secure_pipeline = SecureTradingPipeline()

async def run_secure_pipeline(symbols: List[str] = ['BTC', 'ETH'],
                            retrain_models: bool = False,
                            timeframe: str = '1D',
                            model_variant: str = 'secure') -> Dict[str, Any]:
    """
    Run the secure trading pipeline
    
    Args:
        symbols: List of cryptocurrency symbols to analyze
        retrain_models: Whether to retrain models (not available in secure mode)
        timeframe: Prediction timeframe
        model_variant: Model variant to use
        
    Returns:
        Dictionary with pipeline results
    """
    return await secure_pipeline.run_full_pipeline(
        symbols=symbols,
        retrain_models=retrain_models,
        timeframe=timeframe,
        model_variant=model_variant
    )
