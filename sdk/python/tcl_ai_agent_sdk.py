"""
TCL AI Agent Python SDK
提供便捷的Python客户端接口
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Union, AsyncIterator
from datetime import datetime
import httpx
import websockets
from dataclasses import dataclass, asdict


@dataclass
class AgentResponse:
    """Agent响应数据类"""
    response: str
    data: Optional[Dict[str, Any]] = None
    charts: Optional[List[Dict[str, Any]]] = None
    session_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisRequest:
    """分析请求数据类"""
    analysis_type: str
    metrics: List[str]
    dimensions: Optional[List[str]] = None
    filters: Optional[Dict[str, Any]] = None
    time_range: Optional[Dict[str, str]] = None
    user_id: str = "default"


@dataclass
class ExperimentRequest:
    """A/B实验请求数据类"""
    experiment_name: str
    control_group: Dict[str, Any]
    test_group: Dict[str, Any]
    metric: str
    hypothesis: str
    confidence_level: float = 0.95


class TCLAIAgentSDK:
    """TCL AI Agent SDK主类"""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: int = 30
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.session_id = None
        
        # HTTP客户端
        self._headers = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        
        self._client = httpx.AsyncClient(
            headers=self._headers,
            timeout=timeout
        )
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def close(self):
        """关闭SDK连接"""
        if self._client:
            await self._client.aclose()
    
    # ==================== 聊天接口 ====================
    
    async def chat(
        self,
        message: str,
        user_id: str = "default",
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """发送聊天消息"""
        
        payload = {
            "message": message,
            "user_id": user_id,
            "session_id": self.session_id,
            "context": context
        }
        
        response = await self._client.post(
            f"{self.base_url}/chat",
            json=payload
        )
        
        response.raise_for_status()
        data = response.json()
        
        # 更新session_id
        self.session_id = data.get("session_id")
        
        return AgentResponse(
            response=data.get("response", ""),
            data=data.get("data"),
            charts=data.get("charts"),
            session_id=data.get("session_id"),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat()))
        )
    
    async def stream_chat(
        self,
        message: str,
        user_id: str = "default",
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[str]:
        """流式聊天"""
        
        # WebSocket连接用于流式响应
        ws_url = self.base_url.replace("http", "ws") + "/ws/chat"
        
        payload = {
            "message": message,
            "user_id": user_id,
            "session_id": self.session_id,
            "context": context
        }
        
        try:
            async with websockets.connect(ws_url) as websocket:
                await websocket.send(json.dumps(payload))
                
                async for message in websocket:
                    data = json.loads(message)
                    if data.get("type") == "chunk":
                        yield data.get("content", "")
                    elif data.get("type") == "complete":
                        self.session_id = data.get("session_id")
                        break
                        
        except Exception as e:
            # 如果WebSocket不可用，降级到普通请求
            response = await self.chat(message, user_id, context)
            yield response.response
    
    # ==================== 分析接口 ====================
    
    async def analyze(self, request: AnalysisRequest) -> Dict[str, Any]:
        """执行数据分析"""
        
        response = await self._client.post(
            f"{self.base_url}/api/analysis/analyze",
            json=asdict(request)
        )
        
        response.raise_for_status()
        return response.json()
    
    async def trend_analysis(
        self,
        metrics: List[str],
        time_range: Dict[str, str],
        dimensions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """趋势分析"""
        
        request = AnalysisRequest(
            analysis_type="trend",
            metrics=metrics,
            dimensions=dimensions,
            time_range=time_range
        )
        
        return await self.analyze(request)
    
    async def correlation_analysis(
        self,
        metrics: List[str],
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """相关性分析"""
        
        request = AnalysisRequest(
            analysis_type="correlation",
            metrics=metrics,
            filters=filters
        )
        
        return await self.analyze(request)
    
    # ==================== 查询接口 ====================
    
    async def query(
        self,
        sql: str,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """执行SQL查询"""
        
        payload = {
            "query": sql,
            "query_type": "sql",
            "use_cache": use_cache
        }
        
        response = await self._client.post(
            f"{self.base_url}/api/query/execute",
            json=payload
        )
        
        response.raise_for_status()
        return response.json()
    
    async def natural_language_query(
        self,
        question: str,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """自然语言查询"""
        
        payload = {
            "query": question,
            "query_type": "natural_language",
            "use_cache": use_cache
        }
        
        response = await self._client.post(
            f"{self.base_url}/api/query/execute",
            json=payload
        )
        
        response.raise_for_status()
        return response.json()
    
    # ==================== A/B实验接口 ====================
    
    async def create_experiment(self, request: ExperimentRequest) -> Dict[str, Any]:
        """创建A/B实验"""
        
        response = await self._client.post(
            f"{self.base_url}/api/experiments/create",
            json=asdict(request)
        )
        
        response.raise_for_status()
        return response.json()
    
    async def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """获取实验结果"""
        
        response = await self._client.get(
            f"{self.base_url}/api/experiments/{experiment_id}/results"
        )
        
        response.raise_for_status()
        return response.json()
    
    async def list_experiments(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出实验"""
        
        params = {}
        if status:
            params["status"] = status
        
        response = await self._client.get(
            f"{self.base_url}/api/experiments/",
            params=params
        )
        
        response.raise_for_status()
        return response.json()
    
    # ==================== 预警接口 ====================
    
    async def create_alert(
        self,
        alert_name: str,
        metric: str,
        condition: str,
        threshold: float,
        notification_channels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """创建预警规则"""
        
        payload = {
            "alert_name": alert_name,
            "metric": metric,
            "condition": condition,
            "threshold": threshold,
            "notification_channels": notification_channels or []
        }
        
        response = await self._client.post(
            f"{self.base_url}/api/alerts/create",
            json=payload
        )
        
        response.raise_for_status()
        return response.json()
    
    async def get_alerts(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取预警列表"""
        
        params = {}
        if status:
            params["status"] = status
        
        response = await self._client.get(
            f"{self.base_url}/api/alerts/",
            params=params
        )
        
        response.raise_for_status()
        return response.json()
    
    # ==================== 报告接口 ====================
    
    async def generate_report(
        self,
        report_type: str = "business_summary",
        time_range: Optional[Dict[str, str]] = None,
        metrics: Optional[List[str]] = None,
        format: str = "ppt"
    ) -> Dict[str, Any]:
        """生成报告"""
        
        payload = {
            "report_type": report_type,
            "time_range": time_range,
            "metrics": metrics,
            "format": format
        }
        
        response = await self._client.post(
            f"{self.base_url}/api/reports/generate",
            json=payload
        )
        
        response.raise_for_status()
        return response.json()
    
    async def get_reports(self) -> List[Dict[str, Any]]:
        """获取报告列表"""
        
        response = await self._client.get(f"{self.base_url}/api/reports/")
        response.raise_for_status()
        return response.json()
    
    # ==================== 系统接口 ====================
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        
        response = await self._client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计"""
        
        response = await self._client.get(f"{self.base_url}/api/system/stats")
        response.raise_for_status()
        return response.json()
    
    # ==================== 便捷方法 ====================
    
    async def quick_insight(self, question: str) -> str:
        """快速洞察 - 返回简单的文本响应"""
        
        response = await self.chat(question)
        return response.response
    
    async def get_kpi_dashboard(
        self,
        kpis: List[str],
        time_range: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """获取KPI仪表板数据"""
        
        request = AnalysisRequest(
            analysis_type="kpi_dashboard",
            metrics=kpis,
            time_range=time_range
        )
        
        return await self.analyze(request)
    
    async def compare_segments(
        self,
        metric: str,
        segment_dimension: str,
        segments: List[str],
        time_range: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """细分对比分析"""
        
        request = AnalysisRequest(
            analysis_type="segment_comparison",
            metrics=[metric],
            dimensions=[segment_dimension],
            filters={"segments": segments},
            time_range=time_range
        )
        
        return await self.analyze(request)


# ==================== 同步包装器 ====================

class TCLAIAgentClient:
    """同步版本的SDK客户端"""
    
    def __init__(self, *args, **kwargs):
        self._async_client = TCLAIAgentSDK(*args, **kwargs)
        self._loop = None
    
    def _run_async(self, coro):
        """运行异步协程"""
        if self._loop is None:
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        
        return self._loop.run_until_complete(coro)
    
    def chat(self, message: str, user_id: str = "default", context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """同步聊天"""
        return self._run_async(self._async_client.chat(message, user_id, context))
    
    def analyze(self, request: AnalysisRequest) -> Dict[str, Any]:
        """同步分析"""
        return self._run_async(self._async_client.analyze(request))
    
    def query(self, sql: str, use_cache: bool = True) -> Dict[str, Any]:
        """同步查询"""
        return self._run_async(self._async_client.query(sql, use_cache))
    
    def natural_language_query(self, question: str, use_cache: bool = True) -> Dict[str, Any]:
        """同步自然语言查询"""
        return self._run_async(self._async_client.natural_language_query(question, use_cache))
    
    def quick_insight(self, question: str) -> str:
        """同步快速洞察"""
        return self._run_async(self._async_client.quick_insight(question))
    
    def health_check(self) -> Dict[str, Any]:
        """同步健康检查"""
        return self._run_async(self._async_client.health_check())
    
    def close(self):
        """关闭客户端"""
        self._run_async(self._async_client.close())
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ==================== 工具函数 ====================

def create_client(
    base_url: str = "http://localhost:8000",
    api_key: Optional[str] = None,
    async_mode: bool = True
) -> Union[TCLAIAgentSDK, TCLAIAgentClient]:
    """创建客户端实例"""
    
    if async_mode:
        return TCLAIAgentSDK(base_url, api_key)
    else:
        return TCLAIAgentClient(base_url, api_key)


# ==================== 示例用法 ====================

async def example_usage():
    """SDK使用示例"""
    
    async with TCLAIAgentSDK() as client:
        # 健康检查
        health = await client.health_check()
        print("系统状态:", health)
        
        # 聊天对话
        response = await client.chat("分析一下用户增长趋势")
        print("AI响应:", response.response)
        
        # 自然语言查询
        result = await client.natural_language_query("过去30天的订单总数")
        print("查询结果:", result)
        
        # 趋势分析
        trend = await client.trend_analysis(
            metrics=["revenue", "orders"],
            time_range={"start": "2024-01-01", "end": "2024-01-31"}
        )
        print("趋势分析:", trend)
        
        # 创建A/B实验
        experiment = await client.create_experiment(
            ExperimentRequest(
                experiment_name="新功能测试",
                control_group={"feature_enabled": False},
                test_group={"feature_enabled": True},
                metric="conversion_rate",
                hypothesis="新功能能提高转化率"
            )
        )
        print("实验创建:", experiment)


if __name__ == "__main__":
    # 异步示例
    asyncio.run(example_usage())
    
    # 同步示例
    with TCLAIAgentClient() as client:
        health = client.health_check()
        print("同步健康检查:", health)
        
        insight = client.quick_insight("今天的核心数据如何？")
        print("快速洞察:", insight)
