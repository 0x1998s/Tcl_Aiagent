"""
查询相关API路由
处理数据查询和Text-to-SQL功能
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from datetime import datetime

from core.database import get_db
from services.data_service import DataService
from utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


class QueryRequest(BaseModel):
    """查询请求模型"""
    query: str
    user_id: str = "default"
    context: Optional[Dict[str, Any]] = None
    use_cache: bool = True


class QueryResponse(BaseModel):
    """查询响应模型"""
    success: bool
    data: List[Dict[str, Any]]
    sql: Optional[str] = None
    execution_time: float
    row_count: int
    charts: Optional[List[Dict[str, Any]]] = None
    message: str = ""


class SchemaResponse(BaseModel):
    """Schema响应模型"""
    tables: Dict[str, Any]
    total_tables: int


@router.post("/execute", response_model=QueryResponse)
async def execute_query(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    db=Depends(get_db)
):
    """
    执行数据查询
    支持自然语言查询转SQL
    """
    try:
        logger.info(f"执行查询: {request.query[:100]}...")
        
        start_time = datetime.now()
        
        # 这里应该调用QueryAgent处理查询
        # 目前先直接返回模拟数据
        
        mock_data = [
            {
                "date": "2024-01-01",
                "sales": 10000,
                "orders": 50
            },
            {
                "date": "2024-01-02", 
                "sales": 12000,
                "orders": 60
            }
        ]
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # 异步记录查询日志
        background_tasks.add_task(
            log_query_execution,
            request.user_id,
            request.query,
            execution_time
        )
        
        return QueryResponse(
            success=True,
            data=mock_data,
            sql="SELECT * FROM orders WHERE order_date >= '2024-01-01';",
            execution_time=execution_time,
            row_count=len(mock_data),
            charts=[
                {
                    "type": "line",
                    "title": "销售趋势",
                    "x_axis": "date",
                    "y_axis": "sales",
                    "data": mock_data
                }
            ],
            message="查询执行成功"
        )
        
    except Exception as e:
        logger.error(f"查询执行失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"查询执行失败: {str(e)}")


@router.get("/schema", response_model=SchemaResponse)
async def get_schema(db=Depends(get_db)):
    """获取数据库schema信息"""
    try:
        # 模拟schema数据
        schema_data = {
            "orders": {
                "columns": [
                    {"name": "order_id", "type": "integer", "nullable": False},
                    {"name": "user_id", "type": "integer", "nullable": False},
                    {"name": "amount", "type": "decimal", "nullable": False},
                    {"name": "order_date", "type": "timestamp", "nullable": False}
                ],
                "description": "订单表"
            },
            "users": {
                "columns": [
                    {"name": "user_id", "type": "integer", "nullable": False},
                    {"name": "username", "type": "varchar", "nullable": False},
                    {"name": "email", "type": "varchar", "nullable": False},
                    {"name": "created_at", "type": "timestamp", "nullable": False}
                ],
                "description": "用户表"
            }
        }
        
        return SchemaResponse(
            tables=schema_data,
            total_tables=len(schema_data)
        )
        
    except Exception as e:
        logger.error(f"获取schema失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取schema失败: {str(e)}")


@router.get("/tables/{table_name}/sample")
async def get_sample_data(
    table_name: str,
    limit: int = 10,
    db=Depends(get_db)
):
    """获取表的示例数据"""
    try:
        # 模拟示例数据
        if table_name == "orders":
            sample_data = [
                {
                    "order_id": 1,
                    "user_id": 1,
                    "amount": 199.99,
                    "order_date": "2024-01-01T10:00:00"
                },
                {
                    "order_id": 2,
                    "user_id": 2,
                    "amount": 299.99,
                    "order_date": "2024-01-02T11:00:00"
                }
            ]
        else:
            sample_data = []
        
        return {
            "table_name": table_name,
            "data": sample_data[:limit],
            "total_rows": len(sample_data)
        }
        
    except Exception as e:
        logger.error(f"获取示例数据失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取示例数据失败: {str(e)}")


@router.post("/validate-sql")
async def validate_sql(
    sql: str,
    db=Depends(get_db)
):
    """验证SQL语句"""
    try:
        # 简单的SQL验证
        sql_upper = sql.upper().strip()
        
        # 安全检查
        dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE']
        errors = []
        warnings = []
        
        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                errors.append(f"不允许使用 {keyword} 语句")
        
        if not sql_upper.startswith('SELECT'):
            warnings.append("查询应该以SELECT开头")
        
        is_valid = len(errors) == 0
        
        return {
            "is_valid": is_valid,
            "errors": errors,
            "warnings": warnings,
            "sql": sql
        }
        
    except Exception as e:
        logger.error(f"SQL验证失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"SQL验证失败: {str(e)}")


async def log_query_execution(
    user_id: str,
    query: str,
    execution_time: float
):
    """记录查询执行日志（异步任务）"""
    try:
        logger.info(f"用户 {user_id} 执行查询，耗时 {execution_time:.2f}秒")
        # 这里可以记录到数据库或其他日志系统
    except Exception as e:
        logger.error(f"记录查询日志失败: {str(e)}")
