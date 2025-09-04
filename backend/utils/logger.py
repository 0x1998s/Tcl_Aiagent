"""
日志工具模块
提供统一的日志配置和管理
"""

import sys
from pathlib import Path
from loguru import logger
from typing import Optional

from core.config import Settings

# 全局设置实例
settings = Settings()

def setup_logger(name: Optional[str] = None) -> "logger":
    """设置日志配置"""
    
    # 移除默认处理器
    logger.remove()
    
    # 确保日志目录存在
    log_dir = Path(settings.LOG_FILE).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 控制台输出
    logger.add(
        sys.stdout,
        level=settings.LOG_LEVEL,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True
    )
    
    # 文件输出
    logger.add(
        settings.LOG_FILE,
        level=settings.LOG_LEVEL,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation=settings.LOG_ROTATION,
        retention=settings.LOG_RETENTION,
        compression="zip",
        encoding="utf-8"
    )
    
    # 错误日志单独文件
    error_log_file = str(Path(settings.LOG_FILE).with_suffix('.error.log'))
    logger.add(
        error_log_file,
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation=settings.LOG_ROTATION,
        retention=settings.LOG_RETENTION,
        compression="zip",
        encoding="utf-8"
    )
    
    return logger


def get_logger(name: str) -> "logger":
    """获取指定名称的logger"""
    return logger.bind(name=name)


# 默认logger
default_logger = setup_logger()
