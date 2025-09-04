"""
会话模型
"""

from sqlalchemy import Column, String, Text, ForeignKey
from sqlalchemy.orm import relationship
from .base import BaseModel


class Session(BaseModel):
    """用户会话表"""
    __tablename__ = "sessions"
    
    session_id = Column(String(100), unique=True, nullable=False)
    user_id = Column(String(50), ForeignKey('users.username'), nullable=False)
    context = Column(Text)  # JSON格式的会话上下文
    status = Column(String(20), default='active')  # active, inactive, expired
