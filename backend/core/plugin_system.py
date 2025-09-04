"""
插件系统 - 支持动态加载和管理外部Agent
实现可扩展的插件化架构
"""

import importlib
import inspect
import os
from typing import Dict, Any, List, Optional, Type
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from agents.base_agent import BaseAgent
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PluginInfo:
    """插件信息"""
    name: str
    version: str
    description: str
    author: str
    agent_class: str
    dependencies: List[str] = None
    config_schema: Dict[str, Any] = None


class PluginManager:
    """插件管理器"""
    
    def __init__(self, plugin_dir: str = "./plugins"):
        self.plugin_dir = Path(plugin_dir)
        self.plugins: Dict[str, PluginInfo] = {}
        self.loaded_agents: Dict[str, BaseAgent] = {}
        
    async def initialize(self):
        """初始化插件管理器"""
        
        # 创建插件目录
        self.plugin_dir.mkdir(exist_ok=True)
        
        # 扫描并加载插件
        await self._scan_plugins()
        
        logger.info(f"插件管理器初始化完成，发现 {len(self.plugins)} 个插件")
    
    async def _scan_plugins(self):
        """扫描插件目录"""
        
        if not self.plugin_dir.exists():
            return
        
        for plugin_path in self.plugin_dir.iterdir():
            if plugin_path.is_dir() and (plugin_path / "plugin.py").exists():
                try:
                    await self._load_plugin(plugin_path)
                except Exception as e:
                    logger.error(f"加载插件失败 {plugin_path.name}: {str(e)}")
    
    async def _load_plugin(self, plugin_path: Path):
        """加载单个插件"""
        
        plugin_name = plugin_path.name
        
        # 读取插件配置
        config_file = plugin_path / "plugin.json"
        if config_file.exists():
            import json
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                plugin_info = PluginInfo(**config)
        else:
            # 默认配置
            plugin_info = PluginInfo(
                name=plugin_name,
                version="1.0.0",
                description="",
                author="Unknown",
                agent_class="PluginAgent"
            )
        
        # 动态导入插件模块
        spec = importlib.util.spec_from_file_location(
            f"plugins.{plugin_name}",
            plugin_path / "plugin.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # 获取Agent类
        agent_class = getattr(module, plugin_info.agent_class, None)
        if not agent_class or not issubclass(agent_class, BaseAgent):
            raise ValueError(f"插件 {plugin_name} 未提供有效的Agent类")
        
        # 验证插件
        if not self._validate_plugin(agent_class):
            raise ValueError(f"插件 {plugin_name} 验证失败")
        
        self.plugins[plugin_name] = plugin_info
        logger.info(f"插件加载成功: {plugin_name} v{plugin_info.version}")
    
    def _validate_plugin(self, agent_class: Type[BaseAgent]) -> bool:
        """验证插件Agent类"""
        
        # 检查必要方法
        required_methods = ['_initialize_agent', '_process_request']
        for method in required_methods:
            if not hasattr(agent_class, method):
                logger.error(f"插件缺少必要方法: {method}")
                return False
        
        # 检查方法签名
        try:
            init_method = getattr(agent_class, '_initialize_agent')
            process_method = getattr(agent_class, '_process_request')
            
            # 简单的签名检查
            if not inspect.iscoroutinefunction(init_method):
                logger.error("_initialize_agent必须是异步方法")
                return False
            
            if not inspect.iscoroutinefunction(process_method):
                logger.error("_process_request必须是异步方法")
                return False
                
        except Exception as e:
            logger.error(f"方法签名验证失败: {str(e)}")
            return False
        
        return True
    
    async def create_agent_instance(
        self, 
        plugin_name: str,
        llm_service,
        data_service,
        settings
    ) -> Optional[BaseAgent]:
        """创建插件Agent实例"""
        
        if plugin_name not in self.plugins:
            logger.error(f"插件不存在: {plugin_name}")
            return None
        
        try:
            plugin_info = self.plugins[plugin_name]
            plugin_path = self.plugin_dir / plugin_name
            
            # 重新导入模块
            spec = importlib.util.spec_from_file_location(
                f"plugins.{plugin_name}",
                plugin_path / "plugin.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # 创建实例
            agent_class = getattr(module, plugin_info.agent_class)
            agent_instance = agent_class(llm_service, data_service, settings)
            
            # 初始化
            await agent_instance.initialize()
            
            self.loaded_agents[plugin_name] = agent_instance
            logger.info(f"插件Agent实例创建成功: {plugin_name}")
            
            return agent_instance
            
        except Exception as e:
            logger.error(f"创建插件Agent实例失败 {plugin_name}: {str(e)}")
            return None
    
    def get_plugin_info(self, plugin_name: str) -> Optional[PluginInfo]:
        """获取插件信息"""
        return self.plugins.get(plugin_name)
    
    def list_plugins(self) -> List[PluginInfo]:
        """列出所有插件"""
        return list(self.plugins.values())
    
    def get_loaded_agent(self, plugin_name: str) -> Optional[BaseAgent]:
        """获取已加载的Agent实例"""
        return self.loaded_agents.get(plugin_name)
    
    async def reload_plugin(self, plugin_name: str, llm_service, data_service, settings):
        """重新加载插件"""
        
        if plugin_name in self.loaded_agents:
            # 清理旧实例
            old_agent = self.loaded_agents[plugin_name]
            await old_agent.cleanup()
            del self.loaded_agents[plugin_name]
        
        # 重新扫描和加载
        plugin_path = self.plugin_dir / plugin_name
        if plugin_path.exists():
            await self._load_plugin(plugin_path)
            return await self.create_agent_instance(
                plugin_name, llm_service, data_service, settings
            )
        
        return None
    
    async def unload_plugin(self, plugin_name: str):
        """卸载插件"""
        
        if plugin_name in self.loaded_agents:
            agent = self.loaded_agents[plugin_name]
            await agent.cleanup()
            del self.loaded_agents[plugin_name]
        
        if plugin_name in self.plugins:
            del self.plugins[plugin_name]
        
        logger.info(f"插件已卸载: {plugin_name}")
    
    async def cleanup(self):
        """清理插件管理器"""
        
        for agent in self.loaded_agents.values():
            try:
                await agent.cleanup()
            except Exception as e:
                logger.error(f"清理插件Agent失败: {str(e)}")
        
        self.loaded_agents.clear()
        self.plugins.clear()


# 全局插件管理器实例
plugin_manager = PluginManager()


class PluginAgent(BaseAgent):
    """插件Agent基类 - 为插件开发者提供的基础类"""
    
    def __init__(self, llm_service, data_service, settings, plugin_name: str = "PluginAgent"):
        super().__init__(llm_service, data_service, settings, plugin_name)
        self.plugin_config = {}
    
    def load_plugin_config(self, config_path: str):
        """加载插件配置"""
        try:
            import json
            with open(config_path, 'r', encoding='utf-8') as f:
                self.plugin_config = json.load(f)
        except Exception as e:
            self.logger.error(f"加载插件配置失败: {str(e)}")
    
    @abstractmethod
    async def _initialize_agent(self):
        """插件初始化逻辑"""
        pass
    
    @abstractmethod
    async def _process_request(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """插件处理逻辑"""
        pass
