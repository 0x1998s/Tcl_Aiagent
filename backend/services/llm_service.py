"""
大模型服务 - 统一LLM接口管理
支持OpenAI、NewAPI、Ollama等多种LLM提供商
"""

import asyncio
import json
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import httpx
from openai import AsyncOpenAI

from core.config import Settings
from utils.logger import get_logger


class LLMService:
    """大模型服务类"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = get_logger(__name__)
        
        # 客户端实例
        self.openai_client = None
        self.newapi_client = None
        self.ollama_client = None
        
        # 服务状态
        self.available_providers = []
        self.current_provider = "openai"
        
        # 请求统计
        self.request_count = 0
        self.error_count = 0
        self.total_tokens = 0
    
    async def initialize(self):
        """初始化LLM服务"""
        self.logger.info("初始化LLM服务...")
        
        # 初始化OpenAI客户端
        if self.settings.OPENAI_API_KEY:
            try:
                self.openai_client = AsyncOpenAI(
                    api_key=self.settings.OPENAI_API_KEY,
                    base_url=self.settings.OPENAI_BASE_URL
                )
                self.available_providers.append("openai")
                self.logger.info("OpenAI客户端初始化成功")
                
            except Exception as e:
                self.logger.error(f"OpenAI客户端初始化失败: {str(e)}")
        
        # 初始化NewAPI客户端
        if self.settings.NEWAPI_BASE_URL:
            try:
                await self._test_newapi_connection()
                self.available_providers.append("newapi")
                self.logger.info("NewAPI客户端初始化成功")
                
            except Exception as e:
                self.logger.error(f"NewAPI客户端初始化失败: {str(e)}")
        
        # 初始化Ollama客户端
        try:
            await self._test_ollama_connection()
            self.available_providers.append("ollama")
            self.logger.info("Ollama客户端初始化成功")
            
        except Exception as e:
            self.logger.error(f"Ollama客户端初始化失败: {str(e)}")
        
        # 设置默认提供商
        if self.available_providers:
            self.current_provider = self.available_providers[0]
            self.logger.info(f"默认LLM提供商: {self.current_provider}")
        else:
            raise RuntimeError("没有可用的LLM提供商")
    
    async def _test_newapi_connection(self):
        """测试NewAPI连接"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.settings.NEWAPI_BASE_URL}/v1/models",
                headers={"Authorization": f"Bearer {self.settings.NEWAPI_API_KEY}"}
                if self.settings.NEWAPI_API_KEY else {}
            )
            response.raise_for_status()
    
    async def _test_ollama_connection(self):
        """测试Ollama连接"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.settings.OLLAMA_BASE_URL}/api/tags")
            response.raise_for_status()
    
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        provider: Optional[str] = None
    ) -> str:
        """生成文本响应"""
        
        # 参数处理
        provider = provider or self.current_provider
        model = model or self.settings.DEFAULT_MODEL
        max_tokens = max_tokens or self.settings.MAX_TOKENS
        temperature = temperature or self.settings.TEMPERATURE
        
        self.request_count += 1
        
        try:
            self.logger.debug(f"使用 {provider} 生成响应，prompt长度: {len(prompt)}")
            
            if provider == "openai":
                response = await self._generate_openai(prompt, model, max_tokens, temperature)
            elif provider == "newapi":
                response = await self._generate_newapi(prompt, model, max_tokens, temperature)
            elif provider == "ollama":
                response = await self._generate_ollama(prompt, model, max_tokens, temperature)
            else:
                raise ValueError(f"不支持的LLM提供商: {provider}")
            
            self.logger.debug(f"响应长度: {len(response)}")
            return response
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"LLM生成失败: {str(e)}")
            
            # 尝试fallback到其他提供商
            if provider != self.current_provider:
                return await self.generate(prompt, model, max_tokens, temperature)
            
            raise
    
    async def _generate_openai(
        self, 
        prompt: str, 
        model: str, 
        max_tokens: int, 
        temperature: float
    ) -> str:
        """使用OpenAI生成响应"""
        
        if not self.openai_client:
            raise RuntimeError("OpenAI客户端未初始化")
        
        try:
            response = await self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是一个专业的数据分析助手。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # 更新token统计
            if hasattr(response, 'usage') and response.usage:
                self.total_tokens += response.usage.total_tokens
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"OpenAI API调用失败: {str(e)}")
            raise
    
    async def _generate_newapi(
        self, 
        prompt: str, 
        model: str, 
        max_tokens: int, 
        temperature: float
    ) -> str:
        """使用NewAPI生成响应"""
        
        headers = {"Content-Type": "application/json"}
        if self.settings.NEWAPI_API_KEY:
            headers["Authorization"] = f"Bearer {self.settings.NEWAPI_API_KEY}"
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "你是一个专业的数据分析助手。"},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.settings.NEWAPI_BASE_URL}/v1/chat/completions",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                
                result = response.json()
                
                # 更新token统计
                if "usage" in result:
                    self.total_tokens += result["usage"].get("total_tokens", 0)
                
                return result["choices"][0]["message"]["content"]
                
        except Exception as e:
            self.logger.error(f"NewAPI调用失败: {str(e)}")
            raise
    
    async def _generate_ollama(
        self, 
        prompt: str, 
        model: str, 
        max_tokens: int, 
        temperature: float
    ) -> str:
        """使用Ollama生成响应"""
        
        payload = {
            "model": self.settings.OLLAMA_MODEL,
            "prompt": f"你是一个专业的数据分析助手。\n\n{prompt}",
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature
            }
        }
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.settings.OLLAMA_BASE_URL}/api/generate",
                    json=payload
                )
                response.raise_for_status()
                
                result = response.json()
                return result.get("response", "")
                
        except Exception as e:
            self.logger.error(f"Ollama调用失败: {str(e)}")
            raise
    
    async def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        provider: Optional[str] = None
    ) -> Dict[str, Any]:
        """生成结构化响应"""
        
        structured_prompt = f"""
        {prompt}
        
        请严格按照以下JSON schema格式返回结果：
        {json.dumps(schema, ensure_ascii=False, indent=2)}
        
        只返回JSON，不要包含其他文字：
        """
        
        try:
            response = await self.generate(structured_prompt, provider=provider)
            
            # 尝试解析JSON
            result = json.loads(response)
            
            # 简单验证schema
            if not self._validate_schema(result, schema):
                raise ValueError("响应不符合指定schema")
            
            return result
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON解析失败: {str(e)}")
            raise ValueError("LLM返回的不是有效的JSON格式")
    
    def _validate_schema(self, data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """简单的schema验证"""
        # 这里可以实现更复杂的schema验证逻辑
        # 目前只做基本检查
        
        if "required" in schema:
            for field in schema["required"]:
                if field not in data:
                    return False
        
        return True
    
    async def embed_text(self, text: str, model: str = "text-embedding-ada-002") -> List[float]:
        """文本向量化"""
        
        if not self.openai_client:
            raise RuntimeError("需要OpenAI客户端进行文本向量化")
        
        try:
            response = await self.openai_client.embeddings.create(
                model=model,
                input=text
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            self.logger.error(f"文本向量化失败: {str(e)}")
            raise
    
    async def batch_generate(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[str]:
        """批量生成响应"""
        
        tasks = []
        for prompt in prompts:
            task = self.generate(prompt, **kwargs)
            tasks.append(task)
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理异常结果
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"批量生成第{i}个请求失败: {str(result)}")
                    processed_results.append(f"生成失败: {str(result)}")
                else:
                    processed_results.append(result)
            
            return processed_results
            
        except Exception as e:
            self.logger.error(f"批量生成失败: {str(e)}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """获取服务统计信息"""
        return {
            "available_providers": self.available_providers,
            "current_provider": self.current_provider,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "success_rate": (
                (self.request_count - self.error_count) / self.request_count
                if self.request_count > 0 else 0.0
            ),
            "total_tokens": self.total_tokens
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        health_status = {}
        
        for provider in self.available_providers:
            try:
                test_prompt = "Hello"
                await self.generate(test_prompt, provider=provider, max_tokens=10)
                health_status[provider] = "healthy"
                
            except Exception as e:
                health_status[provider] = f"unhealthy: {str(e)}"
        
        return {
            "providers": health_status,
            "overall_status": "healthy" if any(
                status == "healthy" for status in health_status.values()
            ) else "unhealthy"
        }
    
    def switch_provider(self, provider: str):
        """切换LLM提供商"""
        if provider not in self.available_providers:
            raise ValueError(f"提供商 {provider} 不可用")
        
        self.current_provider = provider
        self.logger.info(f"切换到LLM提供商: {provider}")
