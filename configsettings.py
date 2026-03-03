"""
Configuration management for the trading system.
Centralizes all settings with type hints and validation.
"""
import os
from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime
import logging

@dataclass
class DataConfig:
    """Data collection and preprocessing configuration"""
    # Exchange settings
    EXCHANGE: str = "binance"
    SYMBOL: str = "BTC/USDT"
    TIMEFRAME: str = "1h"
    LOOKBACK_DAYS: int = 365
    
    # Feature engineering
    TECHNICAL_INDICATORS: List[str] = field(default_factory=lambda: [
        "RSI", "MACD", "BBANDS", "ATR", "OBV"
    ])
    NORMALIZATION_METHOD: str = "minmax"
    TRAIN_TEST_SPLIT: float = 0.8
    
    # Data validation
    MIN_DATA_POINTS: int = 1000
    MAX_MISSING_PERCENT: float = 0.05

@dataclass
class GANConfig:
    """GAN model configuration"""
    # Architecture
    LATENT_DIM: int = 100
    GENERATOR_HIDDEN: List[int] = field(default_factory=lambda: [256, 512, 256])
    DISCRIMINATOR_HIDDEN: List[int] = field(default_factory=lambda: [512, 256, 128])
    DROPOUT_RATE: float = 0.2
    
    # Training
    BATCH_SIZE: int = 64
    EPOCHS: int = 1000
    LEARNING_RATE: float = 0.0002
    BETA1: float = 0.5
    SYNTHETIC_DATA_RATIO: float = 0.3
    
    # Validation
    CRITIC_ITERATIONS: int = 5
    GRADIENT_PENALTY_WEIGHT: float = 10.0

@dataclass
class RLConfig:
    """Reinforcement learning configuration"""
    # Environment
    INITIAL_BALANCE: float = 10000.0
    COMMISSION_PERCENT: float = 0.001
    SLIPPAGE_PERCENT: float = 0.0005
    MAX_POSITION_SIZE: float = 0.1  # 10% of portfolio
    
    # PPO Agent
    POLICY: str = "MlpPolicy"
    LEARNING_RATE: float = 0.0003
    N_STEPS: int = 2048
    BATCH_SIZE: int = 64
    N_EPOCHS: int = 10
    GAMMA: float = 0.99
    GAE_LAMBDA: float = 0.95
    CLIP_RANGE: float = 0.2
    
    # Training
    TOTAL_TIMESTEPS: int = 100000
    EVAL_FREQUENCY: int = 10000
    SAVE_FREQUENCY: int = 50000

@dataclass
class FirebaseConfig:
    """Firebase configuration"""
    PROJECT_ID: Optional[str] = os.getenv("FIREBASE_PROJECT_ID")
    CREDENTIALS_PATH: Optional[str] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    COLLECTION_NAME: str = "trading_system"
    
    # Collections
    COLLECTIONS = {
        "trades": "trades",
        "models": "model_versions",
        "performance": "performance_metrics",
        "errors": "system_errors"
    }

@dataclass
class TradingConfig:
    """Live trading configuration"""
    PAPER_TRADING: bool = True
    MAX_DAILY_LOSS: float = 0.05  # 5% max daily loss
    MAX_POSITIONS: int = 5
    RISK_PER_TRADE: float = 0.02  # 2% risk per trade
    
    # Emergency stop
    ENABLE_CIRCUIT_BREAKER: bool = True
    CIRCUIT_BREAKER_THRESHOLD: float = -0.1  # -10% drawdown
    
    # Telegram alerts
    ENABLE_TELEGRAM_ALERTS: bool = True
    TELEGRAM_BOT_TOKEN: Optional[str] = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID: Optional[str] = os.getenv("TELEGRAM_CHAT_ID")

class Config:
    """Main configuration class"""
    def __init__(self):
        self.data = DataConfig()
        self.gan = GANConfig()
        self.rl = RLConfig()
        self.firebase = FirebaseConfig()
        self.trading = TradingConfig()
        
        # Validate critical configurations
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate all configuration parameters"""
        # Check Firebase credentials
        if not self.firebase.PROJECT_ID:
            logging.warning("FIREBASE_PROJECT_ID not set. Firebase features will be disabled.")
        
        # Check Telegram configuration
        if self.trading.ENABLE_TELEGRAM_ALERTS and not all([
            self.trading.TELEGRAM_BOT_TOKEN,
            self.trading.TELEGRAM_CHAT_ID
        ]):
            logging.warning("Telegram bot token or chat ID not set. Alerts will be disabled.")
            self.trading.ENABLE_TELEGRAM_ALERTS = False
        
        # Validate risk parameters
        if self.trading.MAX_DAILY_LOSS <= 0 or self.trading.MAX_DAILY_LOSS > 1:
            raise