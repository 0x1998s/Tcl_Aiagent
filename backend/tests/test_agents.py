"""
Agent测试用例
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from agents.query_agent import QueryAgent
from agents.analysis_agent import AnalysisAgent
from services.llm_service import LLMService
from services.data_service import DataService
from core.config import Settings


@pytest.fixture
def mock_settings():
    """模拟配置"""
    settings = Mock(spec=Settings)
    settings.DATABASE_URL = "sqlite+aiosqlite:///:memory:"
    settings.DEBUG = True
    return settings


@pytest.fixture
def mock_llm_service():
    """模拟LLM服务"""
    service = Mock(spec=LLMService)
    service.generate = AsyncMock(return_value="模拟响应")
    return service


@pytest.fixture  
def mock_data_service():
    """模拟数据服务"""
    service = Mock(spec=DataService)
    service.execute_query = AsyncMock(return_value=[])
    return service


class TestQueryAgent:
    """查询Agent测试"""
    
    @pytest.mark.asyncio
    async def test_query_agent_initialization(self, mock_llm_service, mock_data_service, mock_settings):
        """测试查询Agent初始化"""
        agent = QueryAgent(mock_llm_service, mock_data_service, mock_settings)
        await agent.initialize()
        assert agent.agent_name == "QueryAgent"
    
    @pytest.mark.asyncio
    async def test_query_processing(self, mock_llm_service, mock_data_service, mock_settings):
        """测试查询处理"""
        agent = QueryAgent(mock_llm_service, mock_data_service, mock_settings)
        await agent.initialize()
        
        result = await agent.process("查询用户数量", {})
        assert "response" in result


class TestAnalysisAgent:
    """分析Agent测试"""
    
    @pytest.mark.asyncio
    async def test_analysis_agent_initialization(self, mock_llm_service, mock_data_service, mock_settings):
        """测试分析Agent初始化"""
        agent = AnalysisAgent(mock_llm_service, mock_data_service, mock_settings)
        await agent.initialize()
        assert agent.agent_name == "AnalysisAgent"
