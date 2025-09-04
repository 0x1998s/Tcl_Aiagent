"""
多Agent协同编排器
实现TCL AI Agent的核心协调逻辑
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import logging
from uuid import uuid4

from langgraph.graph import StateGraph, END
from langchain.schema import BaseMessage, HumanMessage

from agents.query_agent import QueryAgent
from agents.analysis_agent import AnalysisAgent
from agents.alert_agent import AlertAgent
from agents.report_agent import ReportAgent
from agents.ab_test_agent import ABTestAgent
from agents.dify_agent import DifyAgent
from services.llm_service import LLMService
from services.data_service import DataService
from core.config import Settings
from core.plugin_system import plugin_manager
from utils.logger import get_logger

logger = get_logger(__name__)


class AgentState:
    """Agent状态管理"""
    
    def __init__(self):
        self.query: str = ""
        self.user_id: str = ""
        self.session_id: str = ""
        self.context: Dict[str, Any] = {}
        self.current_step: str = ""
        self.results: Dict[str, Any] = {}
        self.messages: List[BaseMessage] = []
        self.metadata: Dict[str, Any] = {}
        self.timestamp: datetime = datetime.now()


class AgentOrchestrator:
    """多Agent协同编排器"""
    
    def __init__(
        self, 
        llm_service: LLMService,
        data_service: DataService,
        settings: Settings
    ):
        self.llm_service = llm_service
        self.data_service = data_service
        self.settings = settings
        
        # 初始化各个Agent
        self.query_agent = QueryAgent(llm_service, data_service, settings)
        self.analysis_agent = AnalysisAgent(llm_service, data_service, settings)
        self.alert_agent = AlertAgent(llm_service, data_service, settings)
        self.report_agent = ReportAgent(llm_service, data_service, settings)
        self.ab_test_agent = ABTestAgent(llm_service, data_service, settings)
        
        # 可选的Dify Agent
        self.dify_agent = DifyAgent(llm_service, data_service, settings) if settings.ENABLE_DIFY else None
        
        # 插件Agent字典
        self.plugin_agents = {}
        
        # 构建状态图
        self.workflow = None
        self.sessions: Dict[str, AgentState] = {}
        
    async def initialize(self):
        """初始化编排器"""
        logger.info("初始化Agent编排器...")
        
        # 初始化各个Agent
        await self.query_agent.initialize()
        await self.analysis_agent.initialize()
        await self.alert_agent.initialize()
        await self.report_agent.initialize()
        await self.ab_test_agent.initialize()
        
        # 初始化Dify Agent（如果启用）
        if self.dify_agent:
            await self.dify_agent.initialize()
        
        # 初始化插件系统
        if self.settings.ENABLE_PLUGINS:
            await plugin_manager.initialize()
            await self._load_plugin_agents()
        
        # 构建工作流
        self._build_workflow()
        
        logger.info("Agent编排器初始化完成")
    
    async def _load_plugin_agents(self):
        """加载插件Agent"""
        plugins = plugin_manager.list_plugins()
        
        for plugin_info in plugins:
            try:
                agent = await plugin_manager.create_agent_instance(
                    plugin_info.name,
                    self.llm_service,
                    self.data_service,
                    self.settings
                )
                
                if agent:
                    self.plugin_agents[plugin_info.name] = agent
                    logger.info(f"插件Agent加载成功: {plugin_info.name}")
                    
            except Exception as e:
                logger.error(f"加载插件Agent失败 {plugin_info.name}: {str(e)}")
    
    def _build_workflow(self):
        """构建Agent协同工作流"""
        
        # 创建状态图
        workflow = StateGraph(AgentState)
        
        # 添加节点（各个Agent）
        workflow.add_node("intent_classification", self._classify_intent)
        workflow.add_node("query_processing", self._process_query)
        workflow.add_node("data_analysis", self._analyze_data)
        workflow.add_node("anomaly_detection", self._detect_anomaly)
        workflow.add_node("report_generation", self._generate_report)
        workflow.add_node("ab_test_execution", self._execute_ab_test)
        workflow.add_node("response_synthesis", self._synthesize_response)
        
        # 设置入口点
        workflow.set_entry_point("intent_classification")
        
        # 添加条件边
        # 动态构建路由映射
        route_mapping = {
            "query": "query_processing",
            "analysis": "data_analysis", 
            "alert": "anomaly_detection",
            "report": "report_generation",
            "experiment": "ab_test_execution",
            "end": END
        }
        
        # 如果启用Dify，添加Dify路由
        if self.dify_agent:
            workflow.add_node("dify_processing", self._process_dify)
            route_mapping["dify"] = "dify_processing"
            workflow.add_edge("dify_processing", "response_synthesis")
        
        workflow.add_conditional_edges(
            "intent_classification",
            self._route_by_intent,
            route_mapping
        )
        
        # 添加边
        workflow.add_edge("query_processing", "response_synthesis")
        workflow.add_edge("data_analysis", "response_synthesis")
        workflow.add_edge("anomaly_detection", "response_synthesis")
        workflow.add_edge("report_generation", "response_synthesis")
        workflow.add_edge("ab_test_execution", "response_synthesis")
        workflow.add_edge("response_synthesis", END)
        
        # 编译工作流
        self.workflow = workflow.compile()
    
    async def process_query(
        self, 
        query: str,
        user_id: str = "default",
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """处理用户查询的主入口"""
        
        # 生成会话ID
        if not session_id:
            session_id = str(uuid4())
        
        # 创建或获取会话状态
        if session_id not in self.sessions:
            self.sessions[session_id] = AgentState()
        
        state = self.sessions[session_id]
        state.query = query
        state.user_id = user_id
        state.session_id = session_id
        state.context = context or {}
        state.timestamp = datetime.now()
        
        logger.info(f"处理用户查询: {query[:100]}...")
        
        try:
            # 执行工作流
            result = await self.workflow.ainvoke(state)
            
            # 返回结果
            return {
                "response": result.results.get("response", ""),
                "data": result.results.get("data"),
                "charts": result.results.get("charts"),
                "session_id": session_id,
                "metadata": result.metadata
            }
            
        except Exception as e:
            logger.error(f"处理查询时出错: {str(e)}")
            return {
                "response": f"抱歉，处理您的查询时遇到了问题: {str(e)}",
                "session_id": session_id,
                "error": str(e)
            }
    
    async def _classify_intent(self, state: AgentState) -> AgentState:
        """意图分类"""
        logger.info("执行意图分类...")
        
        try:
            # 使用LLM进行意图分类
            prompt = f"""
            请分析用户查询的意图，从以下类别中选择最合适的一个：
            
            1. query - 数据查询（如：查看销售额、用户数量等）
            2. analysis - 数据分析（如：分析趋势、对比分析等）
            3. alert - 异常检测（如：检查指标异常、监控预警等）
            4. report - 报告生成（如：生成月报、周报等）
            5. experiment - A/B测试（如：设计实验、分析实验结果等）
            6. dify - 使用Dify工作流（如：业务用户自助分析）
            
            用户查询：{state.query}
            
            请只返回类别名称（query/analysis/alert/report/experiment/dify）：
            """
            
            response = await self.llm_service.generate(prompt)
            intent = response.strip().lower()
            
            state.current_step = "intent_classification"
            state.metadata["intent"] = intent
            
            logger.info(f"识别意图: {intent}")
            
        except Exception as e:
            logger.error(f"意图分类失败: {str(e)}")
            state.metadata["intent"] = "query"  # 默认为查询
        
        return state
    
    def _route_by_intent(self, state: AgentState) -> str:
        """根据意图路由到相应的Agent"""
        intent = state.metadata.get("intent", "query")
        
        intent_mapping = {
            "query": "query",
            "analysis": "analysis", 
            "alert": "alert",
            "report": "report",
            "experiment": "experiment"
        }
        
        return intent_mapping.get(intent, "query")
    
    async def _process_query(self, state: AgentState) -> AgentState:
        """处理数据查询"""
        logger.info("执行数据查询...")
        
        try:
            result = await self.query_agent.process(state.query, state.context)
            state.results.update(result)
            state.current_step = "query_processing"
            
        except Exception as e:
            logger.error(f"查询处理失败: {str(e)}")
            state.results["error"] = str(e)
        
        return state
    
    async def _analyze_data(self, state: AgentState) -> AgentState:
        """执行数据分析"""
        logger.info("执行数据分析...")
        
        try:
            result = await self.analysis_agent.process(state.query, state.context)
            state.results.update(result)
            state.current_step = "data_analysis"
            
        except Exception as e:
            logger.error(f"数据分析失败: {str(e)}")
            state.results["error"] = str(e)
        
        return state
    
    async def _detect_anomaly(self, state: AgentState) -> AgentState:
        """异常检测"""
        logger.info("执行异常检测...")
        
        try:
            result = await self.alert_agent.process(state.query, state.context)
            state.results.update(result)
            state.current_step = "anomaly_detection"
            
        except Exception as e:
            logger.error(f"异常检测失败: {str(e)}")
            state.results["error"] = str(e)
        
        return state
    
    async def _generate_report(self, state: AgentState) -> AgentState:
        """生成报告"""
        logger.info("生成报告...")
        
        try:
            result = await self.report_agent.process(state.query, state.context)
            state.results.update(result)
            state.current_step = "report_generation"
            
        except Exception as e:
            logger.error(f"报告生成失败: {str(e)}")
            state.results["error"] = str(e)
        
        return state
    
    async def _execute_ab_test(self, state: AgentState) -> AgentState:
        """执行A/B测试"""
        logger.info("执行A/B测试...")
        
        try:
            result = await self.ab_test_agent.process(state.query, state.context)
            state.results.update(result)
            state.current_step = "ab_test_execution"
            
        except Exception as e:
            logger.error(f"A/B测试失败: {str(e)}")
            state.results["error"] = str(e)
        
        return state
    
    async def _process_dify(self, state: AgentState) -> AgentState:
        """处理Dify请求"""
        logger.info("执行Dify处理...")
        
        try:
            if self.dify_agent:
                result = await self.dify_agent.process(state.query, state.context)
                state.results.update(result)
                state.current_step = "dify_processing"
            else:
                # 如果Dify不可用，回退到查询处理
                result = await self.query_agent.process(state.query, state.context)
                state.results.update(result)
                state.current_step = "query_processing"
                
        except Exception as e:
            logger.error(f"Dify处理失败: {str(e)}")
            state.results["error"] = str(e)
        
        return state
    
    async def _synthesize_response(self, state: AgentState) -> AgentState:
        """合成最终响应"""
        logger.info("合成最终响应...")
        
        try:
            # 如果有错误，返回错误信息
            if "error" in state.results:
                state.results["response"] = f"处理过程中遇到错误: {state.results['error']}"
                return state
            
            # 根据不同的步骤生成不同的响应
            if state.current_step == "query_processing":
                response = self._format_query_response(state.results)
            elif state.current_step == "data_analysis":
                response = self._format_analysis_response(state.results)
            elif state.current_step == "anomaly_detection":
                response = self._format_alert_response(state.results)
            elif state.current_step == "report_generation":
                response = self._format_report_response(state.results)
            elif state.current_step == "ab_test_execution":
                response = self._format_experiment_response(state.results)
            else:
                response = "已完成处理，但无法生成响应。"
            
            state.results["response"] = response
            
        except Exception as e:
            logger.error(f"响应合成失败: {str(e)}")
            state.results["response"] = f"响应生成失败: {str(e)}"
        
        return state
    
    def _format_query_response(self, results: Dict[str, Any]) -> str:
        """格式化查询响应"""
        if "sql_result" in results:
            return f"查询完成！找到 {len(results['sql_result'])} 条记录。"
        return "查询已完成。"
    
    def _format_analysis_response(self, results: Dict[str, Any]) -> str:
        """格式化分析响应"""
        return "数据分析已完成，请查看详细结果。"
    
    def _format_alert_response(self, results: Dict[str, Any]) -> str:
        """格式化预警响应"""
        return "异常检测已完成，请查看检测结果。"
    
    def _format_report_response(self, results: Dict[str, Any]) -> str:
        """格式化报告响应"""
        return "报告已生成完成，请查看详细内容。"
    
    def _format_experiment_response(self, results: Dict[str, Any]) -> str:
        """格式化实验响应"""
        return "A/B测试分析已完成，请查看实验结果。"
    
    async def analyze(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """高级分析接口"""
        # TODO: 实现高级分析逻辑
        return {"message": "高级分析功能正在开发中"}
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """获取系统指标"""
        return {
            "active_sessions": len(self.sessions),
            "total_queries": sum(1 for s in self.sessions.values() if s.query),
            "uptime": datetime.now().isoformat(),
            "agents_status": {
                "query_agent": "active",
                "analysis_agent": "active", 
                "alert_agent": "active",
                "report_agent": "active",
                "ab_test_agent": "active"
            }
        }
    
    async def cleanup(self):
        """清理资源"""
        logger.info("清理Agent编排器资源...")
        
        # 清理各个Agent
        await self.query_agent.cleanup()
        await self.analysis_agent.cleanup()
        await self.alert_agent.cleanup()
        await self.report_agent.cleanup()
        await self.ab_test_agent.cleanup()
        
        # 清理会话
        self.sessions.clear()
        
        logger.info("Agent编排器资源清理完成")
