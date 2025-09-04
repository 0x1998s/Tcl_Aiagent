"""
基础Agent类
所有专业Agent的基类
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import asyncio
import logging
from datetime import datetime

from services.llm_service import LLMService
from services.data_service import DataService
from core.config import Settings
from utils.logger import get_logger


class BaseAgent(ABC):
    """基础Agent抽象类"""
    
    def __init__(
        self, 
        llm_service: LLMService,
        data_service: DataService,
        settings: Settings,
        agent_name: str = "BaseAgent"
    ):
        self.llm_service = llm_service
        self.data_service = data_service
        self.settings = settings
        self.agent_name = agent_name
        self.logger = get_logger(f"agents.{agent_name.lower()}")
        
        # Agent状态
        self.is_initialized = False
        self.is_busy = False
        self.last_activity = None
        
        # 性能指标
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.average_response_time = 0.0
    
    async def initialize(self):
        """初始化Agent"""
        self.logger.info(f"初始化 {self.agent_name}...")
        
        try:
            await self._initialize_agent()
            self.is_initialized = True
            self.logger.info(f"{self.agent_name} 初始化成功")
            
        except Exception as e:
            self.logger.error(f"{self.agent_name} 初始化失败: {str(e)}")
            raise
    
    @abstractmethod
    async def _initialize_agent(self):
        """子类实现的初始化逻辑"""
        pass
    
    async def process(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """处理请求的主入口"""
        if not self.is_initialized:
            raise RuntimeError(f"{self.agent_name} 尚未初始化")
        
        if self.is_busy:
            self.logger.warning(f"{self.agent_name} 正忙，请稍后重试")
            return {"error": "Agent正忙，请稍后重试"}
        
        start_time = datetime.now()
        self.is_busy = True
        self.total_requests += 1
        
        try:
            self.logger.info(f"{self.agent_name} 开始处理请求: {query[:100]}...")
            
            # 执行具体的处理逻辑
            result = await self._process_request(query, context or {})
            
            # 更新统计信息
            self.successful_requests += 1
            self.last_activity = datetime.now()
            
            # 计算响应时间
            response_time = (datetime.now() - start_time).total_seconds()
            self._update_response_time(response_time)
            
            self.logger.info(f"{self.agent_name} 处理完成，耗时: {response_time:.2f}秒")
            
            return result
            
        except Exception as e:
            self.failed_requests += 1
            self.logger.error(f"{self.agent_name} 处理失败: {str(e)}")
            
            return {
                "error": str(e),
                "agent": self.agent_name,
                "timestamp": datetime.now().isoformat()
            }
            
        finally:
            self.is_busy = False
    
    @abstractmethod
    async def _process_request(
        self, 
        query: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """子类实现的具体处理逻辑"""
        pass
    
    def _update_response_time(self, response_time: float):
        """更新平均响应时间"""
        if self.successful_requests == 1:
            self.average_response_time = response_time
        else:
            # 使用指数移动平均
            alpha = 0.1
            self.average_response_time = (
                alpha * response_time + 
                (1 - alpha) * self.average_response_time
            )
    
    async def get_status(self) -> Dict[str, Any]:
        """获取Agent状态"""
        return {
            "agent_name": self.agent_name,
            "is_initialized": self.is_initialized,
            "is_busy": self.is_busy,
            "last_activity": self.last_activity.isoformat() if self.last_activity else None,
            "metrics": {
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "success_rate": (
                    self.successful_requests / self.total_requests 
                    if self.total_requests > 0 else 0.0
                ),
                "average_response_time": self.average_response_time
            }
        }
    
    async def cleanup(self):
        """清理Agent资源"""
        self.logger.info(f"清理 {self.agent_name} 资源...")
        
        try:
            await self._cleanup_agent()
            self.is_initialized = False
            self.logger.info(f"{self.agent_name} 清理完成")
            
        except Exception as e:
            self.logger.error(f"{self.agent_name} 清理失败: {str(e)}")
    
    async def _cleanup_agent(self):
        """子类实现的清理逻辑"""
        pass
    
    async def generate_llm_response(
        self, 
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """生成LLM响应的便捷方法"""
        try:
            response = await self.llm_service.generate(
                prompt=prompt,
                max_tokens=max_tokens or self.settings.MAX_TOKENS,
                temperature=temperature or self.settings.TEMPERATURE
            )
            return response
            
        except Exception as e:
            self.logger.error(f"LLM生成失败: {str(e)}")
            raise
    
    async def execute_sql(
        self, 
        sql: str,
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """执行SQL查询的便捷方法"""
        try:
            result = await self.data_service.execute_query(sql, params)
            return result
            
        except Exception as e:
            self.logger.error(f"SQL执行失败: {str(e)}")
            raise
    
    def create_error_response(
        self, 
        error_message: str, 
        error_code: Optional[str] = None
    ) -> Dict[str, Any]:
        """创建错误响应"""
        return {
            "error": error_message,
            "error_code": error_code,
            "agent": self.agent_name,
            "timestamp": datetime.now().isoformat()
        }
