"""
TCL AI Agent 主应用入口
实现数据分析智能助手的核心功能
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime
import logging

from core.config import Settings
from core.database import get_db
from core.agent_orchestrator import AgentOrchestrator
from api.routes import query, analysis, alerts, reports, experiments
from services.llm_service import LLMService
from services.data_service import DataService
from utils.logger import setup_logger

# 初始化设置和日志
settings = Settings()
logger = setup_logger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="TCL AI Agent - 数据分析智能助手",
    description="基于多Agent协同的企业级数据分析平台",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境需要限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局Agent编排器
orchestrator: Optional[AgentOrchestrator] = None


class ChatRequest(BaseModel):
    """聊天请求模型"""
    message: str
    user_id: str = "default"
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    """聊天响应模型"""
    response: str
    data: Optional[Dict[str, Any]] = None
    charts: Optional[List[Dict[str, Any]]] = None
    session_id: str
    timestamp: datetime


@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    global orchestrator
    
    logger.info("启动TCL AI Agent应用...")
    
    # 初始化服务
    try:
        # 初始化LLM服务
        llm_service = LLMService(settings)
        await llm_service.initialize()
        
        # 初始化数据服务
        data_service = DataService(settings)
        await data_service.initialize()
        
        # 初始化Agent编排器
        orchestrator = AgentOrchestrator(
            llm_service=llm_service,
            data_service=data_service,
            settings=settings
        )
        await orchestrator.initialize()
        
        logger.info("TCL AI Agent应用启动成功！")
        
    except Exception as e:
        logger.error(f"应用启动失败: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件"""
    global orchestrator
    
    logger.info("关闭TCL AI Agent应用...")
    
    if orchestrator:
        await orchestrator.cleanup()
    
    logger.info("TCL AI Agent应用已关闭")


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "TCL AI Agent - 数据分析智能助手",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "services": {
            "orchestrator": orchestrator is not None,
            "database": True,  # TODO: 实际检查数据库连接
            "cache": True,     # TODO: 实际检查Redis连接
        }
    }


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    background_tasks: BackgroundTasks
):
    """
    主聊天接口 - 处理用户的自然语言查询
    实现"对话即洞察"功能
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="服务未就绪")
    
    try:
        logger.info(f"收到用户查询: {request.message}")
        
        # 通过Agent编排器处理查询
        result = await orchestrator.process_query(
            query=request.message,
            user_id=request.user_id,
            session_id=request.session_id,
            context=request.context
        )
        
        # 构建响应
        response = ChatResponse(
            response=result.get("response", ""),
            data=result.get("data"),
            charts=result.get("charts"),
            session_id=result.get("session_id", request.session_id or "default"),
            timestamp=datetime.now()
        )
        
        # 异步记录用户行为（用于优化）
        background_tasks.add_task(
            log_user_interaction,
            request.user_id,
            request.message,
            result
        )
        
        return response
        
    except Exception as e:
        logger.error(f"处理查询时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"查询处理失败: {str(e)}")


@app.post("/analyze")
async def analyze_endpoint(
    request: Dict[str, Any]
):
    """
    高级分析接口 - 处理复杂分析任务
    实现"逻辑推演"功能
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="服务未就绪")
    
    try:
        result = await orchestrator.analyze(request)
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"分析处理出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")


@app.get("/metrics")
async def get_metrics():
    """获取系统指标"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="服务未就绪")
    
    try:
        metrics = await orchestrator.get_system_metrics()
        return JSONResponse(content=metrics)
        
    except Exception as e:
        logger.error(f"获取指标出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取指标失败: {str(e)}")


async def log_user_interaction(
    user_id: str, 
    query: str, 
    result: Dict[str, Any]
):
    """记录用户交互日志（异步任务）"""
    try:
        # TODO: 实现用户行为记录逻辑
        logger.info(f"记录用户 {user_id} 的交互")
    except Exception as e:
        logger.error(f"记录用户交互失败: {str(e)}")


# 包含其他路由
app.include_router(query.router, prefix="/api/query", tags=["查询"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["分析"])
app.include_router(alerts.router, prefix="/api/alerts", tags=["预警"])
app.include_router(reports.router, prefix="/api/reports", tags=["报告"])
app.include_router(experiments.router, prefix="/api/experiments", tags=["实验"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
