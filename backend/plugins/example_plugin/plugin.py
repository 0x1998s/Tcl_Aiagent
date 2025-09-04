"""
示例插件 - 演示如何创建自定义Agent
"""

from typing import Dict, Any
from core.plugin_system import PluginAgent


class ExampleAgent(PluginAgent):
    """示例Agent - 展示插件开发模式"""
    
    def __init__(self, llm_service, data_service, settings):
        super().__init__(llm_service, data_service, settings, "ExampleAgent")
        
    async def _initialize_agent(self):
        """初始化示例Agent"""
        self.logger.info("示例Agent初始化完成")
        
        # 加载插件配置
        self.load_plugin_config("./plugins/example_plugin/plugin.json")
        
    async def _process_request(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """处理示例请求"""
        
        # 示例处理逻辑
        if "示例" in query or "example" in query.lower():
            response = f"这是示例Agent的响应。您的查询是：{query}"
            
            return {
                "response": response,
                "agent": self.agent_name,
                "plugin_type": "example",
                "processed_query": query,
                "context_keys": list(context.keys())
            }
        
        # 如果不匹配，返回默认响应
        return {
            "response": "示例Agent无法处理此查询",
            "agent": self.agent_name,
            "error": "查询不匹配示例模式"
        }
