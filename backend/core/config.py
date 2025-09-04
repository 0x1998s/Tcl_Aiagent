"""
应用配置管理
"""

from pydantic_settings import BaseSettings
from typing import Optional, List
import os


class Settings(BaseSettings):
    """应用配置类"""
    
    # 基本配置
    APP_NAME: str = "TCL AI Agent"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # API配置
    API_PREFIX: str = "/api"
    ALLOWED_HOSTS: List[str] = ["*"]
    
    # 数据库配置
    # 生产环境 替换这个 下面使用SQLite为了测试环境调试 
    # DATABASE_URL: str = "postgresql://postgres:password@localhost:5432/tcl_agent"
    DATABASE_URL: str = "sqlite+aiosqlite:///./tcl_agent.db"
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20
    
    # Redis配置
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_POOL_SIZE: int = 10
    
    # ChromaDB配置
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8001
    CHROMA_COLLECTION_NAME: str = "tcl_knowledge"
    
    # LLM配置
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    DEFAULT_MODEL: str = "TCL-AI-Agent"
    MAX_TOKENS: int = 4000
    TEMPERATURE: float = 0.1
    
    # NewAPI配置
    NEWAPI_BASE_URL: str = "http://localhost:3000"
    NEWAPI_API_KEY: Optional[str] = None
    
    # Ollama配置
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama2:7b"
    
    # 向量化配置
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    
    # 安全配置
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    ALGORITHM: str = "HS256"
    
    # 文件上传配置
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    UPLOAD_DIR: str = "./uploads"
    
    # 日志配置
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/app.log"
    LOG_ROTATION: str = "1 day"
    LOG_RETENTION: str = "30 days"
    
    # 缓存配置
    CACHE_TTL: int = 3600  # 1小时
    QUERY_CACHE_TTL: int = 300  # 5分钟
    
    # Agent配置
    MAX_AGENT_ITERATIONS: int = 10
    AGENT_TIMEOUT: int = 300  # 5分钟
    
    # 数据处理配置
    MAX_ROWS_PER_QUERY: int = 10000
    QUERY_TIMEOUT: int = 60  # 1分钟
    
    # 可视化配置
    CHART_WIDTH: int = 800
    CHART_HEIGHT: int = 600
    CHART_THEME: str = "light"
    
    # 预警配置
    ALERT_CHECK_INTERVAL: int = 300  # 5分钟
    ANOMALY_THRESHOLD: float = 3.0  # 3σ
    TREND_THRESHOLD: float = 0.05   # 5%变化
    
    # A/B测试配置
    MIN_SAMPLE_SIZE: int = 1000
    CONFIDENCE_LEVEL: float = 0.95
    STATISTICAL_POWER: float = 0.8
    
    # 报告配置
    REPORT_TEMPLATE_DIR: str = "./templates"
    REPORT_OUTPUT_DIR: str = "./reports"
    
    # 通知配置
    FEISHU_WEBHOOK_URL: Optional[str] = None
    DINGTALK_WEBHOOK_URL: Optional[str] = None
    EMAIL_SMTP_HOST: Optional[str] = None
    EMAIL_SMTP_PORT: int = 587
    EMAIL_USERNAME: Optional[str] = None
    EMAIL_PASSWORD: Optional[str] = None
    
    # Dify集成配置
    ENABLE_DIFY: bool = False
    DIFY_BASE_URL: str = "http://localhost:5001"
    DIFY_API_KEY: Optional[str] = None
    DIFY_TIMEOUT: int = 30
    
    # 插件系统配置
    PLUGIN_DIR: str = "./plugins"
    ENABLE_PLUGINS: bool = True
    PLUGIN_AUTO_RELOAD: bool = False
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# 全局设置实例
settings = Settings()
