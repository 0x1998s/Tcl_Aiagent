"""
预警相关API路由
处理异常检测和预警系统
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from enum import Enum

from utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


class AlertSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    ACTIVE = "active"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class AlertRule(BaseModel):
    """预警规则模型"""
    rule_id: Optional[str] = None
    name: str
    metric: str
    condition: str  # "greater_than", "less_than", "change_rate", etc.
    threshold: float
    time_window: int  # minutes
    severity: AlertSeverity
    enabled: bool = True
    notification_channels: List[str] = []


class Alert(BaseModel):
    """预警模型"""
    alert_id: str
    rule_id: str
    metric: str
    current_value: float
    threshold: float
    severity: AlertSeverity
    status: AlertStatus
    message: str
    created_at: datetime
    resolved_at: Optional[datetime] = None


class AlertResponse(BaseModel):
    """预警响应模型"""
    success: bool
    alerts: List[Alert]
    total_count: int


@router.post("/rules", response_model=Dict[str, Any])
async def create_alert_rule(rule: AlertRule):
    """创建预警规则"""
    try:
        logger.info(f"创建预警规则: {rule.name}")
        
        # 生成规则ID
        import uuid
        rule_id = str(uuid.uuid4())
        
        # 这里应该保存到数据库
        # 目前返回模拟响应
        
        return {
            "success": True,
            "rule_id": rule_id,
            "message": f"预警规则 '{rule.name}' 创建成功"
        }
        
    except Exception as e:
        logger.error(f"创建预警规则失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"创建预警规则失败: {str(e)}")


@router.get("/rules", response_model=List[AlertRule])
async def get_alert_rules():
    """获取所有预警规则"""
    try:
        # 模拟预警规则数据
        mock_rules = [
            AlertRule(
                rule_id="rule_1",
                name="销售额下跌预警",
                metric="daily_sales",
                condition="change_rate",
                threshold=-0.05,  # 下跌5%
                time_window=60,
                severity=AlertSeverity.MEDIUM,
                enabled=True,
                notification_channels=["feishu", "email"]
            ),
            AlertRule(
                rule_id="rule_2", 
                name="用户数异常预警",
                metric="active_users",
                condition="less_than",
                threshold=1000,
                time_window=30,
                severity=AlertSeverity.HIGH,
                enabled=True,
                notification_channels=["dingtalk"]
            )
        ]
        
        return mock_rules
        
    except Exception as e:
        logger.error(f"获取预警规则失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取预警规则失败: {str(e)}")


@router.get("/active", response_model=AlertResponse)
async def get_active_alerts():
    """获取当前活跃的预警"""
    try:
        # 模拟活跃预警数据
        mock_alerts = [
            Alert(
                alert_id="alert_1",
                rule_id="rule_1", 
                metric="daily_sales",
                current_value=8500.0,
                threshold=9000.0,
                severity=AlertSeverity.MEDIUM,
                status=AlertStatus.ACTIVE,
                message="今日销售额较昨日下跌5.6%，低于预警阈值",
                created_at=datetime.now() - timedelta(minutes=30)
            ),
            Alert(
                alert_id="alert_2",
                rule_id="rule_2",
                metric="active_users", 
                current_value=850.0,
                threshold=1000.0,
                severity=AlertSeverity.HIGH,
                status=AlertStatus.ACTIVE,
                message="当前活跃用户数850，低于预警阈值1000",
                created_at=datetime.now() - timedelta(minutes=15)
            )
        ]
        
        return AlertResponse(
            success=True,
            alerts=mock_alerts,
            total_count=len(mock_alerts)
        )
        
    except Exception as e:
        logger.error(f"获取活跃预警失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取活跃预警失败: {str(e)}")


@router.post("/check")
async def check_metrics():
    """手动触发指标检查"""
    try:
        logger.info("执行手动指标检查")
        
        # 模拟检查结果
        check_results = {
            "checked_metrics": ["daily_sales", "active_users", "conversion_rate"],
            "new_alerts": 1,
            "resolved_alerts": 0,
            "total_active_alerts": 2,
            "check_time": datetime.now().isoformat()
        }
        
        return {
            "success": True,
            "message": "指标检查完成",
            "results": check_results
        }
        
    except Exception as e:
        logger.error(f"指标检查失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"指标检查失败: {str(e)}")


@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str):
    """解决预警"""
    try:
        logger.info(f"解决预警: {alert_id}")
        
        # 这里应该更新数据库中的预警状态
        
        return {
            "success": True,
            "alert_id": alert_id,
            "message": "预警已解决",
            "resolved_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"解决预警失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"解决预警失败: {str(e)}")


@router.post("/alerts/{alert_id}/suppress")
async def suppress_alert(alert_id: str, duration_minutes: int = 60):
    """抑制预警"""
    try:
        logger.info(f"抑制预警: {alert_id}，持续时间: {duration_minutes}分钟")
        
        suppress_until = datetime.now() + timedelta(minutes=duration_minutes)
        
        return {
            "success": True,
            "alert_id": alert_id,
            "message": f"预警已抑制{duration_minutes}分钟",
            "suppress_until": suppress_until.isoformat()
        }
        
    except Exception as e:
        logger.error(f"抑制预警失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"抑制预警失败: {str(e)}")


@router.get("/history")
async def get_alert_history(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    severity: Optional[AlertSeverity] = None,
    limit: int = 100
):
    """获取预警历史"""
    try:
        # 模拟历史预警数据
        mock_history = [
            {
                "alert_id": "alert_hist_1",
                "rule_id": "rule_1",
                "metric": "daily_sales",
                "severity": "medium",
                "status": "resolved",
                "message": "销售额下跌预警",
                "created_at": (datetime.now() - timedelta(days=1)).isoformat(),
                "resolved_at": (datetime.now() - timedelta(hours=2)).isoformat()
            },
            {
                "alert_id": "alert_hist_2", 
                "rule_id": "rule_2",
                "metric": "conversion_rate",
                "severity": "low",
                "status": "resolved",
                "message": "转化率轻微下降",
                "created_at": (datetime.now() - timedelta(days=2)).isoformat(),
                "resolved_at": (datetime.now() - timedelta(days=1)).isoformat()
            }
        ]
        
        # 应用过滤条件
        filtered_history = mock_history
        if severity:
            filtered_history = [h for h in filtered_history if h["severity"] == severity.value]
        
        return {
            "success": True,
            "alerts": filtered_history[:limit],
            "total_count": len(filtered_history),
            "filters": {
                "start_date": start_date,
                "end_date": end_date,
                "severity": severity.value if severity else None,
                "limit": limit
            }
        }
        
    except Exception as e:
        logger.error(f"获取预警历史失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取预警历史失败: {str(e)}")


@router.get("/stats")
async def get_alert_stats():
    """获取预警统计信息"""
    try:
        # 模拟统计数据
        stats = {
            "total_rules": 5,
            "active_rules": 4,
            "total_alerts_today": 3,
            "resolved_alerts_today": 1,
            "active_alerts": 2,
            "alert_rate_24h": 0.12,  # 每小时预警数
            "most_frequent_metric": "daily_sales",
            "severity_distribution": {
                "low": 1,
                "medium": 4,
                "high": 2,
                "critical": 0
            }
        }
        
        return {
            "success": True,
            "stats": stats,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"获取预警统计失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取预警统计失败: {str(e)}")
