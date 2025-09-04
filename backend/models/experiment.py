"""
A/B实验模型
"""

from sqlalchemy import Column, String, Text, Float, Integer
from .base import BaseModel


class Experiment(BaseModel):
    """A/B实验表"""
    __tablename__ = "experiments"
    
    name = Column(String(100), nullable=False)
    description = Column(Text)
    status = Column(String(20), default='draft')  # draft, running, completed, paused
    control_group_size = Column(Integer)
    test_group_size = Column(Integer)
    confidence_level = Column(Float, default=0.95)
    statistical_power = Column(Float, default=0.8)
    results = Column(Text)  # JSON格式的实验结果
    created_by = Column(String(50))
