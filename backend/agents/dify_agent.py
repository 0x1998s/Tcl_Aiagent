"""
Dify集成Agent - 负责与Dify平台的交互
实现业务用户自助工具功能
"""

import httpx
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

from agents.base_agent import BaseAgent
from services.llm_service import LLMService
from services.data_service import DataService
from core.config import Settings


class DifyAgent(BaseAgent):
    """Dify集成Agent - 处理Dify工作流调用"""
    
    def __init__(
        self, 
        llm_service: LLMService,
        data_service: DataService,
        settings: Settings
    ):
        super().__init__(llm_service, data_service, settings, "DifyAgent")
        
        # Dify配置
        self.dify_base_url = getattr(settings, 'DIFY_BASE_URL', 'http://localhost:5001')
        self.dify_api_key = getattr(settings, 'DIFY_API_KEY', None)
        self.enabled = getattr(settings, 'ENABLE_DIFY', False)
        
        # HTTP客户端
        self.client = httpx.AsyncClient(timeout=30.0)
        
        # 工作流映射
        self.workflow_mapping = {
            "simple_query": "simple-data-query-workflow",
            "report_generation": "auto-report-workflow", 
            "data_visualization": "chart-generation-workflow",
            "business_analysis": "business-insight-workflow"
        }
    
    async def _initialize_agent(self):
        """初始化Dify Agent"""
        if not self.enabled:
            self.logger.info("Dify集成已禁用，跳过初始化")
            return
        
        if not self.dify_api_key:
            self.logger.warning("Dify API Key未配置，Dify功能将不可用")
            return
        
        # 测试Dify连接
        await self._test_dify_connection()
        
        self.logger.info("Dify Agent初始化完成")
    
    async def _process_request(
        self, 
        query: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """处理Dify请求"""
        
        if not self.enabled:
            return {"error": "Dify集成未启用"}
        
        # 1. 识别适合的Dify工作流
        workflow_type = self._identify_workflow_type(query, context)
        
        # 2. 调用Dify工作流
        if workflow_type:
            result = await self._execute_dify_workflow(workflow_type, query, context)
        else:
            # 回退到默认处理
            result = await self._fallback_processing(query, context)
        
        return result
    
    def _identify_workflow_type(self, query: str, context: Dict[str, Any]) -> Optional[str]:
        """识别适合的Dify工作流类型"""
        
        query_lower = query.lower()
        
        # 简单查询
        if any(keyword in query_lower for keyword in ["查询", "数据", "多少", "统计"]):
            return "simple_query"
        
        # 报告生成
        elif any(keyword in query_lower for keyword in ["报告", "汇总", "总结"]):
            return "report_generation"
        
        # 数据可视化
        elif any(keyword in query_lower for keyword in ["图表", "可视化", "趋势图"]):
            return "data_visualization"
        
        # 业务分析
        elif any(keyword in query_lower for keyword in ["分析", "洞察", "原因"]):
            return "business_analysis"
        
        return None
    
    async def _execute_dify_workflow(
        self, 
        workflow_type: str, 
        query: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行Dify工作流"""
        
        workflow_id = self.workflow_mapping.get(workflow_type)
        if not workflow_id:
            return {"error": f"未找到工作流类型: {workflow_type}"}
        
        try:
            # 构建Dify API请求
            payload = {
                "inputs": {
                    "query": query,
                    "context": json.dumps(context),
                    "user_id": context.get("user_id", "default")
                },
                "response_mode": "blocking",
                "user": context.get("user_id", "default")
            }
            
            headers = {
                "Authorization": f"Bearer {self.dify_api_key}",
                "Content-Type": "application/json"
            }
            
            # 调用Dify API
            response = await self.client.post(
                f"{self.dify_base_url}/v1/workflows/run",
                json=payload,
                headers=headers
            )
            
            response.raise_for_status()
            dify_result = response.json()
            
            # 转换Dify结果为标准格式
            return self._convert_dify_result(dify_result, workflow_type)
            
        except Exception as e:
            self.logger.error(f"Dify工作流执行失败: {str(e)}")
            return await self._fallback_processing(query, context)
    
    def _convert_dify_result(self, dify_result: Dict[str, Any], workflow_type: str) -> Dict[str, Any]:
        """将Dify结果转换为标准格式"""
        
        try:
            # 提取Dify结果
            outputs = dify_result.get("data", {}).get("outputs", {})
            
            # 根据工作流类型转换结果
            if workflow_type == "simple_query":
                return {
                    "response": outputs.get("answer", "查询完成"),
                    "data": outputs.get("data", []),
                    "source": "dify"
                }
            
            elif workflow_type == "report_generation":
                return {
                    "response": outputs.get("report_summary", "报告生成完成"),
                    "report_content": outputs.get("report_content", ""),
                    "charts": outputs.get("charts", []),
                    "source": "dify"
                }
            
            elif workflow_type == "data_visualization":
                return {
                    "response": outputs.get("description", "图表生成完成"),
                    "charts": outputs.get("charts", []),
                    "source": "dify"
                }
            
            elif workflow_type == "business_analysis":
                return {
                    "response": outputs.get("insights", "分析完成"),
                    "analysis": outputs.get("detailed_analysis", {}),
                    "recommendations": outputs.get("recommendations", []),
                    "source": "dify"
                }
            
            else:
                return {
                    "response": outputs.get("result", "处理完成"),
                    "data": outputs,
                    "source": "dify"
                }
                
        except Exception as e:
            self.logger.error(f"Dify结果转换失败: {str(e)}")
            return {
                "response": "Dify处理完成，但结果解析失败",
                "raw_result": dify_result,
                "source": "dify"
            }
    
    async def _fallback_processing(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Dify失败时的回退处理"""
        
        self.logger.info("Dify处理失败，使用本地Agent处理")
        
        # 调用相应的本地Agent
        if "报告" in query:
            return await self._call_local_agent("report", query, context)
        elif "分析" in query:
            return await self._call_local_agent("analysis", query, context)
        else:
            return await self._call_local_agent("query", query, context)
    
    async def _call_local_agent(self, agent_type: str, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """调用本地Agent处理"""
        
        # 这里可以直接调用其他Agent
        # 或者返回一个标准的错误响应
        return {
            "response": f"Dify服务不可用，已切换到本地{agent_type}处理",
            "fallback": True,
            "original_query": query
        }
    
    async def _test_dify_connection(self):
        """测试Dify连接"""
        
        try:
            headers = {"Authorization": f"Bearer {self.dify_api_key}"}
            response = await self.client.get(
                f"{self.dify_base_url}/v1/apps",
                headers=headers
            )
            response.raise_for_status()
            self.logger.info("Dify连接测试成功")
            
        except Exception as e:
            self.logger.error(f"Dify连接测试失败: {str(e)}")
            raise
    
    async def get_available_workflows(self) -> List[Dict[str, Any]]:
        """获取可用的Dify工作流"""
        
        if not self.enabled:
            return []
        
        try:
            headers = {"Authorization": f"Bearer {self.dify_api_key}"}
            response = await self.client.get(
                f"{self.dify_base_url}/v1/workflows",
                headers=headers
            )
            response.raise_for_status()
            
            workflows = response.json().get("data", [])
            return [
                {
                    "id": wf["id"],
                    "name": wf["name"],
                    "description": wf.get("description", ""),
                    "status": wf.get("status", "unknown")
                }
                for wf in workflows
            ]
            
        except Exception as e:
            self.logger.error(f"获取Dify工作流失败: {str(e)}")
            return []
    
    async def _cleanup_agent(self):
        """清理Dify Agent资源"""
        if self.client:
            await self.client.aclose()
