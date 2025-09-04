"""
查询模型
"""

from sqlalchemy import Column, String, Text, ForeignKey
from .base import BaseModel


class Query(BaseModel):
    """用户查询记录表"""
    __tablename__ = "queries"
    
    query_text = Column(Text, nullable=False)
    user_id = Column(String(50), ForeignKey('users.username'), nullable=False)
    session_id = Column(String(100), ForeignKey('sessions.session_id'))
    intent = Column(String(50))  # 查询意图
    sql_query = Column(Text)     # 生成的SQL
    result_data = Column(Text)   # JSON格式的结果
    execution_time = Column(String(20))  # 执行时间
    status = Column(String(20), default='completed')  # completed, failed, pending
