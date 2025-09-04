"""
预警模型
"""

from sqlalchemy import Column, String, Text, Float
from .base import BaseModel


class Alert(BaseModel):
    """预警记录表"""
    __tablename__ = "alerts"
    
    alert_type = Column(String(50), nullable=False)  # anomaly, threshold, trend
    metric_name = Column(String(100), nullable=False)
    current_value = Column(Float)
    threshold_value = Column(Float)
    severity = Column(String(20), default='medium')  # low, medium, high, critical
    message = Column(Text)
    status = Column(String(20), default='active')  # active, resolved, ignored
    assigned_to = Column(String(50))
