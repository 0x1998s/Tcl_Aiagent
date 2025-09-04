"""
A/B实验相关API路由
处理实验设计、执行和分析功能
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from enum import Enum

from utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


class ExperimentStatus(str, Enum):
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ExperimentType(str, Enum):
    AB_TEST = "ab_test"
    MULTIVARIATE = "multivariate"
    FEATURE_FLAG = "feature_flag"


class Experiment(BaseModel):
    """实验模型"""
    experiment_id: Optional[str] = None
    name: str
    description: str
    type: ExperimentType
    status: ExperimentStatus = ExperimentStatus.DRAFT
    hypothesis: str
    success_metric: str
    sample_size: int
    confidence_level: float = 0.95
    statistical_power: float = 0.8
    variants: List[Dict[str, Any]]
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    created_by: str = "default"


class ExperimentResult(BaseModel):
    """实验结果模型"""
    experiment_id: str
    variant_results: List[Dict[str, Any]]
    winner: Optional[str] = None
    confidence: float
    significance: bool
    recommendation: str
    detailed_analysis: Dict[str, Any]


@router.post("/create", response_model=Dict[str, Any])
async def create_experiment(experiment: Experiment):
    """创建A/B实验"""
    try:
        logger.info(f"创建实验: {experiment.name}")
        
        # 生成实验ID
        import uuid
        experiment_id = str(uuid.uuid4())
        
        # 计算推荐样本量
        recommended_sample_size = _calculate_sample_size(
            experiment.confidence_level,
            experiment.statistical_power
        )
        
        return {
            "success": True,
            "experiment_id": experiment_id,
            "message": f"实验 '{experiment.name}' 创建成功",
            "recommended_sample_size": recommended_sample_size,
            "estimated_duration_days": _estimate_duration(recommended_sample_size)
        }
        
    except Exception as e:
        logger.error(f"创建实验失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"创建实验失败: {str(e)}")


@router.get("/list")
async def list_experiments(
    status: Optional[ExperimentStatus] = None,
    limit: int = 20
):
    """获取实验列表"""
    try:
        # 模拟实验数据
        mock_experiments = [
            {
                "experiment_id": "exp_1",
                "name": "首页按钮颜色测试",
                "type": "ab_test",
                "status": "running",
                "success_metric": "conversion_rate",
                "sample_size": 2000,
                "start_date": "2024-01-10T00:00:00",
                "progress": 0.65,
                "variants": [
                    {"name": "控制组", "traffic": 0.5},
                    {"name": "红色按钮", "traffic": 0.5}
                ]
            },
            {
                "experiment_id": "exp_2",
                "name": "推荐算法优化",
                "type": "ab_test", 
                "status": "completed",
                "success_metric": "click_through_rate",
                "sample_size": 5000,
                "start_date": "2024-01-01T00:00:00",
                "end_date": "2024-01-08T00:00:00",
                "winner": "算法V2",
                "variants": [
                    {"name": "算法V1", "traffic": 0.5},
                    {"name": "算法V2", "traffic": 0.5}
                ]
            }
        ]
        
        # 应用状态过滤
        if status:
            mock_experiments = [exp for exp in mock_experiments if exp["status"] == status.value]
        
        return {
            "success": True,
            "experiments": mock_experiments[:limit],
            "total_count": len(mock_experiments)
        }
        
    except Exception as e:
        logger.error(f"获取实验列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取实验列表失败: {str(e)}")


@router.get("/{experiment_id}")
async def get_experiment(experiment_id: str):
    """获取实验详情"""
    try:
        # 模拟实验详情
        experiment_detail = {
            "experiment_id": experiment_id,
            "name": "首页按钮颜色测试",
            "description": "测试不同按钮颜色对转化率的影响",
            "type": "ab_test",
            "status": "running",
            "hypothesis": "红色按钮将提高转化率",
            "success_metric": "conversion_rate",
            "sample_size": 2000,
            "confidence_level": 0.95,
            "statistical_power": 0.8,
            "start_date": "2024-01-10T00:00:00",
            "estimated_end_date": "2024-01-20T00:00:00",
            "progress": 0.65,
            "variants": [
                {
                    "name": "控制组",
                    "description": "原始蓝色按钮",
                    "traffic": 0.5,
                    "current_sample_size": 650,
                    "conversion_rate": 0.042
                },
                {
                    "name": "红色按钮",
                    "description": "新的红色按钮",
                    "traffic": 0.5,
                    "current_sample_size": 650,
                    "conversion_rate": 0.048
                }
            ],
            "current_results": {
                "significance": False,
                "confidence": 0.78,
                "leading_variant": "红色按钮",
                "lift": 0.14
            }
        }
        
        return {
            "success": True,
            "experiment": experiment_detail
        }
        
    except Exception as e:
        logger.error(f"获取实验详情失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取实验详情失败: {str(e)}")


@router.post("/{experiment_id}/start")
async def start_experiment(experiment_id: str):
    """启动实验"""
    try:
        logger.info(f"启动实验: {experiment_id}")
        
        return {
            "success": True,
            "experiment_id": experiment_id,
            "message": "实验已启动",
            "started_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"启动实验失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"启动实验失败: {str(e)}")


@router.post("/{experiment_id}/stop")
async def stop_experiment(experiment_id: str, reason: str = ""):
    """停止实验"""
    try:
        logger.info(f"停止实验: {experiment_id}, 原因: {reason}")
        
        return {
            "success": True,
            "experiment_id": experiment_id,
            "message": "实验已停止",
            "stopped_at": datetime.now().isoformat(),
            "reason": reason
        }
        
    except Exception as e:
        logger.error(f"停止实验失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"停止实验失败: {str(e)}")


@router.get("/{experiment_id}/results", response_model=ExperimentResult)
async def get_experiment_results(experiment_id: str):
    """获取实验结果"""
    try:
        logger.info(f"获取实验结果: {experiment_id}")
        
        # 模拟实验结果
        results = ExperimentResult(
            experiment_id=experiment_id,
            variant_results=[
                {
                    "variant_name": "控制组",
                    "sample_size": 1000,
                    "conversion_rate": 0.042,
                    "confidence_interval": [0.038, 0.046]
                },
                {
                    "variant_name": "红色按钮",
                    "sample_size": 1000, 
                    "conversion_rate": 0.048,
                    "confidence_interval": [0.044, 0.052]
                }
            ],
            winner="红色按钮",
            confidence=0.95,
            significance=True,
            recommendation="建议采用红色按钮方案，预计可提升转化率14.3%",
            detailed_analysis={
                "statistical_test": "two-sample t-test",
                "p_value": 0.032,
                "effect_size": 0.143,
                "practical_significance": True,
                "risk_assessment": "低风险，建议全量上线"
            }
        )
        
        return results
        
    except Exception as e:
        logger.error(f"获取实验结果失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取实验结果失败: {str(e)}")


@router.post("/{experiment_id}/decide")
async def make_experiment_decision(
    experiment_id: str,
    decision: str,  # "winner", "no_winner", "continue"
    winner_variant: Optional[str] = None
):
    """做出实验决策"""
    try:
        logger.info(f"实验决策: {experiment_id}, 决策: {decision}")
        
        decision_result = {
            "experiment_id": experiment_id,
            "decision": decision,
            "winner_variant": winner_variant,
            "decided_at": datetime.now().isoformat(),
            "next_steps": []
        }
        
        if decision == "winner":
            decision_result["next_steps"] = [
                "准备全量上线计划",
                "监控上线后的关键指标",
                "记录实验学习和最佳实践"
            ]
        elif decision == "no_winner":
            decision_result["next_steps"] = [
                "分析无显著差异的原因",
                "考虑其他优化方向",
                "保持现状或设计新实验"
            ]
        else:  # continue
            decision_result["next_steps"] = [
                "继续收集数据",
                "监控实验进展",
                "预计X天后可得出结论"
            ]
        
        return {
            "success": True,
            "decision_result": decision_result
        }
        
    except Exception as e:
        logger.error(f"实验决策失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"实验决策失败: {str(e)}")


@router.get("/calculate/sample-size")
async def calculate_sample_size(
    baseline_rate: float,
    minimum_effect: float,
    confidence_level: float = 0.95,
    statistical_power: float = 0.8
):
    """计算所需样本量"""
    try:
        # 简化的样本量计算
        import math
        
        # Z值查找（简化）
        z_alpha = 1.96 if confidence_level == 0.95 else 2.58
        z_beta = 0.84 if statistical_power == 0.8 else 1.28
        
        p1 = baseline_rate
        p2 = baseline_rate * (1 + minimum_effect)
        p_pooled = (p1 + p2) / 2
        
        sample_size_per_group = (
            (z_alpha * math.sqrt(2 * p_pooled * (1 - p_pooled)) + 
             z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
        ) / (p2 - p1) ** 2
        
        total_sample_size = int(sample_size_per_group * 2)
        
        return {
            "success": True,
            "sample_size_per_group": int(sample_size_per_group),
            "total_sample_size": total_sample_size,
            "estimated_duration_days": _estimate_duration(total_sample_size),
            "parameters": {
                "baseline_rate": baseline_rate,
                "minimum_effect": minimum_effect,
                "confidence_level": confidence_level,
                "statistical_power": statistical_power
            }
        }
        
    except Exception as e:
        logger.error(f"计算样本量失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"计算样本量失败: {str(e)}")


def _calculate_sample_size(confidence_level: float, statistical_power: float) -> int:
    """计算推荐样本量（简化版）"""
    base_size = 1000
    
    if confidence_level >= 0.99:
        base_size *= 1.5
    if statistical_power >= 0.9:
        base_size *= 1.2
    
    return int(base_size)


def _estimate_duration(sample_size: int) -> int:
    """估算实验持续时间（天）"""
    # 假设每天能获得100个样本
    daily_samples = 100
    return max(7, int(sample_size / daily_samples))
