"""
用户模型
"""

from sqlalchemy import Column, String, Boolean, Text
from .base import BaseModel


class User(BaseModel):
    """用户表"""
    __tablename__ = "users"
    
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    full_name = Column(String(100))
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    preferences = Column(Text)  # JSON格式的用户偏好设置
