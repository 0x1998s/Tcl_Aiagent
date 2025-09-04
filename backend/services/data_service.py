"""
数据服务 - 统一数据访问层
支持PostgreSQL、DuckDB等多种数据源
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
import duckdb
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from core.config import Settings
from utils.logger import get_logger


class DataService:
    """数据服务类"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = get_logger(__name__)
        
        # 数据库引擎
        self.async_engine = None
        self.sync_engine = None
        self.async_session_maker = None
        
        # DuckDB连接（用于OLAP查询）
        self.duckdb_conn = None
        
        # 缓存
        self.query_cache = {}
        self.schema_cache = {}
        
        # 统计信息
        self.query_count = 0
        self.cache_hits = 0
        
    async def initialize(self):
        """初始化数据服务"""
        self.logger.info("初始化数据服务...")
        
        # 初始化PostgreSQL连接
        await self._init_postgresql()
        
        # 初始化DuckDB连接
        await self._init_duckdb()
        
        # 加载schema信息
        await self._load_schema_info()
        
        # 创建示例数据（如果不存在）
        await self._create_sample_data()
        
        self.logger.info("数据服务初始化完成")
    
    async def _init_postgresql(self):
        """初始化PostgreSQL连接"""
        try:
            # 异步引擎
            self.async_engine = create_async_engine(
                self.settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://"),
                pool_size=self.settings.DATABASE_POOL_SIZE,
                max_overflow=self.settings.DATABASE_MAX_OVERFLOW,
                echo=self.settings.DEBUG
            )
            
            # 同步引擎（用于某些操作）
            self.sync_engine = create_engine(
                self.settings.DATABASE_URL,
                pool_size=self.settings.DATABASE_POOL_SIZE,
                max_overflow=self.settings.DATABASE_MAX_OVERFLOW
            )
            
            # 会话工厂
            self.async_session_maker = sessionmaker(
                self.async_engine, 
                class_=AsyncSession, 
                expire_on_commit=False
            )
            
            # 测试连接
            async with self.async_engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            
            self.logger.info("PostgreSQL连接初始化成功")
            
        except Exception as e:
            self.logger.error(f"PostgreSQL连接初始化失败: {str(e)}")
            raise
    
    async def _init_duckdb(self):
        """初始化DuckDB连接"""
        try:
            # 创建内存数据库用于快速分析
            self.duckdb_conn = duckdb.connect(":memory:")
            
            # 安装并加载扩展
            self.duckdb_conn.execute("INSTALL httpfs")
            self.duckdb_conn.execute("LOAD httpfs")
            
            self.logger.info("DuckDB连接初始化成功")
            
        except Exception as e:
            self.logger.error(f"DuckDB连接初始化失败: {str(e)}")
            # DuckDB失败不阻塞启动
            self.duckdb_conn = None
    
    async def _load_schema_info(self):
        """加载数据库schema信息"""
        try:
            schema_query = """
            SELECT 
                t.table_name,
                c.column_name,
                c.data_type,
                c.is_nullable,
                c.column_default,
                tc.constraint_type
            FROM information_schema.tables t
            LEFT JOIN information_schema.columns c ON t.table_name = c.table_name
            LEFT JOIN information_schema.constraint_column_usage ccu ON c.column_name = ccu.column_name AND c.table_name = ccu.table_name
            LEFT JOIN information_schema.table_constraints tc ON ccu.constraint_name = tc.constraint_name
            WHERE t.table_schema = 'public'
            ORDER BY t.table_name, c.ordinal_position;
            """
            
            results = await self.execute_query(schema_query)
            
            # 组织schema信息
            self.schema_cache = {}
            for row in results:
                table_name = row['table_name']
                if table_name not in self.schema_cache:
                    self.schema_cache[table_name] = {
                        'columns': [],
                        'constraints': [],
                        'description': ''
                    }
                
                if row['column_name']:
                    column_info = {
                        'name': row['column_name'],
                        'type': row['data_type'],
                        'nullable': row['is_nullable'] == 'YES',
                        'default': row['column_default']
                    }
                    
                    # 避免重复添加
                    if column_info not in self.schema_cache[table_name]['columns']:
                        self.schema_cache[table_name]['columns'].append(column_info)
                
                if row['constraint_type']:
                    constraint_info = {
                        'column': row['column_name'],
                        'type': row['constraint_type']
                    }
                    if constraint_info not in self.schema_cache[table_name]['constraints']:
                        self.schema_cache[table_name]['constraints'].append(constraint_info)
            
            self.logger.info(f"加载了 {len(self.schema_cache)} 个表的schema信息")
            
        except Exception as e:
            self.logger.error(f"加载schema信息失败: {str(e)}")
            # 使用默认schema
            self._setup_default_schema()
    
    def _setup_default_schema(self):
        """设置默认schema（用于演示）"""
        self.schema_cache = {
            "orders": {
                "columns": [
                    {"name": "order_id", "type": "integer", "nullable": False, "default": None},
                    {"name": "user_id", "type": "integer", "nullable": False, "default": None},
                    {"name": "product_id", "type": "integer", "nullable": False, "default": None},
                    {"name": "quantity", "type": "integer", "nullable": False, "default": None},
                    {"name": "amount", "type": "decimal", "nullable": False, "default": None},
                    {"name": "order_date", "type": "timestamp", "nullable": False, "default": None},
                    {"name": "status", "type": "varchar", "nullable": False, "default": "'pending'"}
                ],
                "constraints": [
                    {"column": "order_id", "type": "PRIMARY KEY"}
                ],
                "description": "订单表"
            },
            "users": {
                "columns": [
                    {"name": "user_id", "type": "integer", "nullable": False, "default": None},
                    {"name": "username", "type": "varchar", "nullable": False, "default": None},
                    {"name": "email", "type": "varchar", "nullable": False, "default": None},
                    {"name": "age", "type": "integer", "nullable": True, "default": None},
                    {"name": "city", "type": "varchar", "nullable": True, "default": None},
                    {"name": "created_at", "type": "timestamp", "nullable": False, "default": None},
                    {"name": "last_login", "type": "timestamp", "nullable": True, "default": None}
                ],
                "constraints": [
                    {"column": "user_id", "type": "PRIMARY KEY"}
                ],
                "description": "用户表"
            },
            "products": {
                "columns": [
                    {"name": "product_id", "type": "integer", "nullable": False, "default": None},
                    {"name": "product_name", "type": "varchar", "nullable": False, "default": None},
                    {"name": "category", "type": "varchar", "nullable": False, "default": None},
                    {"name": "price", "type": "decimal", "nullable": False, "default": None},
                    {"name": "stock", "type": "integer", "nullable": False, "default": "0"}
                ],
                "constraints": [
                    {"column": "product_id", "type": "PRIMARY KEY"}
                ],
                "description": "产品表"
            }
        }
    
    async def _create_sample_data(self):
        """创建示例数据"""
        try:
            # 检查是否已有数据
            count_result = await self.execute_query("SELECT COUNT(*) as count FROM information_schema.tables WHERE table_schema = 'public' AND table_name IN ('orders', 'users', 'products')")
            
            if count_result[0]['count'] > 0:
                self.logger.info("数据表已存在，跳过示例数据创建")
                return
            
            # 创建表结构
            await self._create_tables()
            
            # 插入示例数据
            await self._insert_sample_data()
            
            self.logger.info("示例数据创建完成")
            
        except Exception as e:
            self.logger.error(f"创建示例数据失败: {str(e)}")
    
    async def _create_tables(self):
        """创建数据表"""
        
        create_users_table = """
        CREATE TABLE IF NOT EXISTS users (
            user_id SERIAL PRIMARY KEY,
            username VARCHAR(100) NOT NULL,
            email VARCHAR(255) NOT NULL UNIQUE,
            age INTEGER,
            city VARCHAR(100),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        );
        """
        
        create_products_table = """
        CREATE TABLE IF NOT EXISTS products (
            product_id SERIAL PRIMARY KEY,
            product_name VARCHAR(255) NOT NULL,
            category VARCHAR(100) NOT NULL,
            price DECIMAL(10,2) NOT NULL,
            stock INTEGER DEFAULT 0
        );
        """
        
        create_orders_table = """
        CREATE TABLE IF NOT EXISTS orders (
            order_id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(user_id),
            product_id INTEGER REFERENCES products(product_id),
            quantity INTEGER NOT NULL,
            amount DECIMAL(10,2) NOT NULL,
            order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status VARCHAR(50) DEFAULT 'pending'
        );
        """
        
        await self.execute_query(create_users_table)
        await self.execute_query(create_products_table)
        await self.execute_query(create_orders_table)
    
    async def _insert_sample_data(self):
        """插入示例数据"""
        
        # 插入用户数据
        users_data = """
        INSERT INTO users (username, email, age, city, created_at, last_login) VALUES
        ('张三', 'zhangsan@example.com', 25, '北京', '2024-01-01', '2024-01-15'),
        ('李四', 'lisi@example.com', 30, '上海', '2024-01-02', '2024-01-14'),
        ('王五', 'wangwu@example.com', 28, '广州', '2024-01-03', '2024-01-13'),
        ('赵六', 'zhaoliu@example.com', 35, '深圳', '2024-01-04', '2024-01-12'),
        ('钱七', 'qianqi@example.com', 22, '杭州', '2024-01-05', '2024-01-11');
        """
        
        # 插入产品数据
        products_data = """
        INSERT INTO products (product_name, category, price, stock) VALUES
        ('iPhone 15', '手机', 7999.00, 100),
        ('MacBook Pro', '电脑', 15999.00, 50),
        ('iPad Air', '平板', 4599.00, 80),
        ('AirPods Pro', '耳机', 1999.00, 200),
        ('Apple Watch', '手表', 2999.00, 150);
        """
        
        # 插入订单数据
        orders_data = """
        INSERT INTO orders (user_id, product_id, quantity, amount, order_date, status) VALUES
        (1, 1, 1, 7999.00, '2024-01-10', 'completed'),
        (2, 2, 1, 15999.00, '2024-01-11', 'completed'),
        (3, 3, 2, 9198.00, '2024-01-12', 'pending'),
        (4, 4, 1, 1999.00, '2024-01-13', 'completed'),
        (5, 5, 1, 2999.00, '2024-01-14', 'shipped'),
        (1, 4, 2, 3998.00, '2024-01-15', 'completed'),
        (2, 1, 1, 7999.00, '2024-01-16', 'pending');
        """
        
        await self.execute_query(users_data)
        await self.execute_query(products_data)
        await self.execute_query(orders_data)
    
    async def execute_query(
        self, 
        query: str, 
        params: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """执行SQL查询"""
        
        self.query_count += 1
        
        # 检查缓存
        cache_key = self._generate_cache_key(query, params)
        if use_cache and cache_key in self.query_cache:
            self.cache_hits += 1
            self.logger.debug("使用查询缓存")
            return self.query_cache[cache_key]
        
        try:
            async with self.async_engine.begin() as conn:
                if params:
                    result = await conn.execute(text(query), params)
                else:
                    result = await conn.execute(text(query))
                
                # 转换为字典列表
                columns = result.keys()
                rows = result.fetchall()
                
                data = []
                for row in rows:
                    row_dict = {}
                    for i, column in enumerate(columns):
                        value = row[i]
                        # 处理日期时间类型
                        if isinstance(value, datetime):
                            row_dict[column] = value.isoformat()
                        else:
                            row_dict[column] = value
                    data.append(row_dict)
                
                # 缓存结果
                if use_cache:
                    self.query_cache[cache_key] = data
                
                return data
                
        except Exception as e:
            self.logger.error(f"SQL查询失败: {str(e)}")
            self.logger.error(f"查询语句: {query}")
            if params:
                self.logger.error(f"查询参数: {params}")
            raise
    
    async def execute_duckdb_query(
        self, 
        query: str,
        data_source: Optional[Union[str, pd.DataFrame]] = None
    ) -> List[Dict[str, Any]]:
        """使用DuckDB执行OLAP查询"""
        
        if not self.duckdb_conn:
            raise RuntimeError("DuckDB连接未初始化")
        
        try:
            # 如果提供了数据源，先注册
            if isinstance(data_source, pd.DataFrame):
                self.duckdb_conn.register("temp_table", data_source)
            elif isinstance(data_source, str):
                # 假设是CSV文件路径或其他数据源
                self.duckdb_conn.execute(f"CREATE OR REPLACE TABLE temp_table AS SELECT * FROM '{data_source}'")
            
            # 执行查询
            result = self.duckdb_conn.execute(query).fetchdf()
            
            # 转换为字典列表
            return result.to_dict('records')
            
        except Exception as e:
            self.logger.error(f"DuckDB查询失败: {str(e)}")
            raise
    
    async def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """获取表信息"""
        
        if table_name in self.schema_cache:
            return self.schema_cache[table_name]
        
        # 动态查询表信息
        try:
            columns_query = """
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns 
            WHERE table_name = :table_name AND table_schema = 'public'
            ORDER BY ordinal_position;
            """
            
            columns_result = await self.execute_query(
                columns_query, 
                {"table_name": table_name},
                use_cache=False
            )
            
            table_info = {
                "columns": [
                    {
                        "name": row["column_name"],
                        "type": row["data_type"],
                        "nullable": row["is_nullable"] == "YES",
                        "default": row["column_default"]
                    }
                    for row in columns_result
                ],
                "constraints": [],
                "description": f"{table_name}表"
            }
            
            # 缓存表信息
            self.schema_cache[table_name] = table_info
            
            return table_info
            
        except Exception as e:
            self.logger.error(f"获取表信息失败: {str(e)}")
            raise
    
    async def get_sample_data(
        self, 
        table_name: str, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """获取表的示例数据"""
        
        query = f"SELECT * FROM {table_name} LIMIT {limit};"
        
        try:
            return await self.execute_query(query)
        except Exception as e:
            self.logger.error(f"获取示例数据失败: {str(e)}")
            return []
    
    def get_all_tables(self) -> List[str]:
        """获取所有表名"""
        return list(self.schema_cache.keys())
    
    def get_schema_summary(self) -> Dict[str, Any]:
        """获取schema摘要"""
        return {
            "total_tables": len(self.schema_cache),
            "tables": {
                name: {
                    "column_count": len(info["columns"]),
                    "description": info["description"]
                }
                for name, info in self.schema_cache.items()
            }
        }
    
    def _generate_cache_key(
        self, 
        query: str, 
        params: Optional[Dict[str, Any]]
    ) -> str:
        """生成查询缓存键"""
        import hashlib
        
        cache_data = {
            "query": query.strip().lower(),
            "params": params or {}
        }
        
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取数据服务统计信息"""
        return {
            "query_count": self.query_count,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": (
                self.cache_hits / self.query_count 
                if self.query_count > 0 else 0.0
            ),
            "cached_queries": len(self.query_cache),
            "schema_tables": len(self.schema_cache)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        health_status = {}
        
        # 检查PostgreSQL连接
        try:
            await self.execute_query("SELECT 1", use_cache=False)
            health_status["postgresql"] = "healthy"
        except Exception as e:
            health_status["postgresql"] = f"unhealthy: {str(e)}"
        
        # 检查DuckDB连接
        if self.duckdb_conn:
            try:
                self.duckdb_conn.execute("SELECT 1")
                health_status["duckdb"] = "healthy"
            except Exception as e:
                health_status["duckdb"] = f"unhealthy: {str(e)}"
        else:
            health_status["duckdb"] = "not_initialized"
        
        return {
            "databases": health_status,
            "overall_status": "healthy" if "healthy" in health_status.values() else "unhealthy"
        }
    
    async def cleanup(self):
        """清理数据服务资源"""
        self.logger.info("清理数据服务资源...")
        
        try:
            # 关闭数据库连接
            if self.async_engine:
                await self.async_engine.dispose()
            
            if self.sync_engine:
                self.sync_engine.dispose()
            
            if self.duckdb_conn:
                self.duckdb_conn.close()
            
            # 清理缓存
            self.query_cache.clear()
            self.schema_cache.clear()
            
            self.logger.info("数据服务资源清理完成")
            
        except Exception as e:
            self.logger.error(f"数据服务资源清理失败: {str(e)}")
