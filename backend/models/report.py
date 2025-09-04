"""
报告模型
"""

from sqlalchemy import Column, String, Text
from .base import BaseModel


class Report(BaseModel):
    """报告表"""
    __tablename__ = "reports"
    
    title = Column(String(200), nullable=False)
    report_type = Column(String(50), nullable=False)  # daily, weekly, monthly, custom
    content = Column(Text)  # HTML或Markdown格式的报告内容
    charts = Column(Text)   # JSON格式的图表配置
    status = Column(String(20), default='draft')  # draft, published, archived
    created_by = Column(String(50))
    tags = Column(String(200))  # 逗号分隔的标签
