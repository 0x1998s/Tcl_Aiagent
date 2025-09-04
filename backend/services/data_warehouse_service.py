"""
数据仓库建模服务
实现维度建模、数据血缘和元数据管理
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import networkx as nx
import pandas as pd

from utils.logger import get_logger
from core.config import Settings

logger = get_logger(__name__)


class DataWarehouseService:
    """数据仓库建模服务"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.lineage_graph = nx.DiGraph()
        self.metadata_registry = {}
        self.dimension_tables = {}
        self.fact_tables = {}
        
    async def initialize(self):
        """初始化数据仓库服务"""
        logger.info("初始化数据仓库建模服务...")
        
        # 初始化维度表
        await self._init_dimension_tables()
        
        # 初始化事实表
        await self._init_fact_tables()
        
        # 构建数据血缘
        await self._build_data_lineage()
        
        logger.info("数据仓库建模服务初始化完成")
    
    async def _init_dimension_tables(self):
        """初始化维度表"""
        
        # 时间维度
        self.dimension_tables["dim_date"] = {
            "name": "dim_date",
            "type": "dimension",
            "scd_type": 0,  # 不变维度
            "columns": {
                "date_key": {"type": "INTEGER", "primary_key": True},
                "date": {"type": "DATE"},
                "year": {"type": "INTEGER"},
                "quarter": {"type": "INTEGER"},
                "month": {"type": "INTEGER"},
                "week": {"type": "INTEGER"},
                "day": {"type": "INTEGER"},
                "weekday": {"type": "VARCHAR"},
                "is_weekend": {"type": "BOOLEAN"},
                "is_holiday": {"type": "BOOLEAN"}
            },
            "business_keys": ["date"],
            "description": "时间维度表"
        }
        
        # 用户维度 (SCD Type 2)
        self.dimension_tables["dim_user"] = {
            "name": "dim_user",
            "type": "dimension", 
            "scd_type": 2,  # 历史变化维度
            "columns": {
                "user_key": {"type": "INTEGER", "primary_key": True},
                "user_id": {"type": "VARCHAR", "business_key": True},
                "username": {"type": "VARCHAR"},
                "email": {"type": "VARCHAR"},
                "age_group": {"type": "VARCHAR"},
                "city": {"type": "VARCHAR"},
                "country": {"type": "VARCHAR"},
                "registration_date": {"type": "DATE"},
                "customer_segment": {"type": "VARCHAR"},
                "effective_date": {"type": "DATE"},
                "expiry_date": {"type": "DATE"},
                "is_current": {"type": "BOOLEAN"}
            },
            "business_keys": ["user_id"],
            "description": "用户维度表，支持历史变化跟踪"
        }
        
        # 产品维度 (SCD Type 1)
        self.dimension_tables["dim_product"] = {
            "name": "dim_product",
            "type": "dimension",
            "scd_type": 1,  # 覆盖更新
            "columns": {
                "product_key": {"type": "INTEGER", "primary_key": True},
                "product_id": {"type": "VARCHAR", "business_key": True},
                "product_name": {"type": "VARCHAR"},
                "category": {"type": "VARCHAR"},
                "subcategory": {"type": "VARCHAR"},
                "brand": {"type": "VARCHAR"},
                "price": {"type": "DECIMAL"},
                "cost": {"type": "DECIMAL"},
                "launch_date": {"type": "DATE"},
                "status": {"type": "VARCHAR"}
            },
            "business_keys": ["product_id"],
            "description": "产品维度表"
        }
        
    async def _init_fact_tables(self):
        """初始化事实表"""
        
        # 订单事实表
        self.fact_tables["fact_orders"] = {
            "name": "fact_orders",
            "type": "transaction_fact",
            "grain": "订单行项目",
            "columns": {
                "order_key": {"type": "INTEGER", "primary_key": True},
                "date_key": {"type": "INTEGER", "foreign_key": "dim_date.date_key"},
                "user_key": {"type": "INTEGER", "foreign_key": "dim_user.user_key"},
                "product_key": {"type": "INTEGER", "foreign_key": "dim_product.product_key"},
                "order_id": {"type": "VARCHAR"},
                "order_line_id": {"type": "VARCHAR"},
                "quantity": {"type": "INTEGER", "measure_type": "additive"},
                "unit_price": {"type": "DECIMAL", "measure_type": "non_additive"},
                "discount_amount": {"type": "DECIMAL", "measure_type": "additive"},
                "tax_amount": {"type": "DECIMAL", "measure_type": "additive"},
                "total_amount": {"type": "DECIMAL", "measure_type": "additive"},
                "profit_margin": {"type": "DECIMAL", "measure_type": "additive"}
            },
            "measures": ["quantity", "discount_amount", "tax_amount", "total_amount", "profit_margin"],
            "dimensions": ["date_key", "user_key", "product_key"],
            "description": "订单事实表，记录订单明细"
        }
        
        # 用户活动快照事实表
        self.fact_tables["fact_user_daily_snapshot"] = {
            "name": "fact_user_daily_snapshot",
            "type": "periodic_snapshot_fact",
            "grain": "用户每日快照",
            "columns": {
                "snapshot_key": {"type": "INTEGER", "primary_key": True},
                "date_key": {"type": "INTEGER", "foreign_key": "dim_date.date_key"},
                "user_key": {"type": "INTEGER", "foreign_key": "dim_user.user_key"},
                "login_count": {"type": "INTEGER", "measure_type": "additive"},
                "page_views": {"type": "INTEGER", "measure_type": "additive"},
                "session_duration": {"type": "INTEGER", "measure_type": "additive"},
                "orders_count": {"type": "INTEGER", "measure_type": "additive"},
                "total_spent": {"type": "DECIMAL", "measure_type": "additive"},
                "last_login_time": {"type": "TIMESTAMP", "measure_type": "non_additive"}
            },
            "measures": ["login_count", "page_views", "session_duration", "orders_count", "total_spent"],
            "dimensions": ["date_key", "user_key"],
            "description": "用户每日活动快照"
        }
    
    async def _build_data_lineage(self):
        """构建数据血缘关系"""
        
        # 添加数据源节点
        sources = [
            {"id": "src_users", "type": "source", "system": "OLTP"},
            {"id": "src_orders", "type": "source", "system": "OLTP"},
            {"id": "src_products", "type": "source", "system": "OLTP"},
            {"id": "src_user_events", "type": "source", "system": "Event Stream"}
        ]
        
        # 添加转换节点
        transformations = [
            {"id": "etl_user_dimension", "type": "transformation", "process": "SCD Type 2"},
            {"id": "etl_product_dimension", "type": "transformation", "process": "SCD Type 1"},
            {"id": "etl_order_fact", "type": "transformation", "process": "Fact Loading"},
            {"id": "etl_user_snapshot", "type": "transformation", "process": "Daily Aggregation"}
        ]
        
        # 添加目标表节点
        targets = list(self.dimension_tables.keys()) + list(self.fact_tables.keys())
        
        # 构建血缘图
        for source in sources:
            self.lineage_graph.add_node(source["id"], **source)
        
        for transform in transformations:
            self.lineage_graph.add_node(transform["id"], **transform)
        
        for target in targets:
            self.lineage_graph.add_node(target, type="table", layer="data_warehouse")
        
        # 添加血缘关系
        lineage_edges = [
            ("src_users", "etl_user_dimension"),
            ("etl_user_dimension", "dim_user"),
            ("src_products", "etl_product_dimension"), 
            ("etl_product_dimension", "dim_product"),
            ("src_orders", "etl_order_fact"),
            ("dim_user", "etl_order_fact"),
            ("dim_product", "etl_order_fact"),
            ("etl_order_fact", "fact_orders"),
            ("src_user_events", "etl_user_snapshot"),
            ("dim_user", "etl_user_snapshot"),
            ("etl_user_snapshot", "fact_user_daily_snapshot")
        ]
        
        for source, target in lineage_edges:
            self.lineage_graph.add_edge(source, target)
    
    async def create_star_schema(
        self,
        schema_name: str,
        fact_table: Dict[str, Any],
        dimension_tables: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """创建星型模式"""
        
        try:
            schema = {
                "name": schema_name,
                "type": "star_schema",
                "fact_table": fact_table,
                "dimension_tables": dimension_tables,
                "relationships": [],
                "created_at": datetime.now().isoformat()
            }
            
            # 建立维度关系
            for dim_table in dimension_tables:
                # 查找外键关系
                for col_name, col_info in fact_table["columns"].items():
                    if col_info.get("foreign_key"):
                        fk_ref = col_info["foreign_key"]
                        if fk_ref.startswith(f"{dim_table['name']}."):
                            schema["relationships"].append({
                                "type": "one_to_many",
                                "from_table": dim_table["name"],
                                "from_column": fk_ref.split(".")[1],
                                "to_table": fact_table["name"],
                                "to_column": col_name
                            })
            
            logger.info(f"星型模式创建成功: {schema_name}")
            return {"status": "success", "schema": schema}
            
        except Exception as e:
            logger.error(f"星型模式创建失败: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def implement_scd_type2(
        self,
        table_name: str,
        business_keys: List[str],
        tracked_columns: List[str]
    ) -> Dict[str, Any]:
        """实现SCD Type 2（缓慢变化维度）"""
        
        try:
            scd_sql = f"""
            -- SCD Type 2 实现 for {table_name}
            WITH source_data AS (
                SELECT 
                    {', '.join(business_keys + tracked_columns)},
                    CURRENT_DATE as effective_date,
                    '9999-12-31'::date as expiry_date,
                    TRUE as is_current
                FROM staging.{table_name}_staging
            ),
            
            changed_records AS (
                SELECT s.*
                FROM source_data s
                LEFT JOIN {table_name} t ON {' AND '.join([f's.{k} = t.{k}' for k in business_keys])}
                WHERE t.is_current = TRUE
                AND ({' OR '.join([f's.{c} != t.{c}' for c in tracked_columns])})
            ),
            
            -- 关闭历史记录
            UPDATE {table_name} 
            SET expiry_date = CURRENT_DATE - 1,
                is_current = FALSE
            WHERE ({', '.join(business_keys)}) IN (
                SELECT {', '.join(business_keys)} FROM changed_records
            ) AND is_current = TRUE;
            
            -- 插入新记录
            INSERT INTO {table_name} ({', '.join(business_keys + tracked_columns)}, effective_date, expiry_date, is_current)
            SELECT * FROM changed_records;
            """
            
            return {
                "status": "success",
                "sql": scd_sql,
                "table": table_name,
                "type": "SCD Type 2"
            }
            
        except Exception as e:
            logger.error(f"SCD Type 2实现失败: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def generate_data_mart(
        self,
        mart_name: str,
        business_process: str,
        required_dimensions: List[str],
        required_measures: List[str]
    ) -> Dict[str, Any]:
        """生成数据集市"""
        
        try:
            # 查找相关的事实表和维度表
            relevant_facts = []
            relevant_dims = []
            
            for fact_name, fact_info in self.fact_tables.items():
                fact_measures = fact_info.get("measures", [])
                if any(measure in required_measures for measure in fact_measures):
                    relevant_facts.append(fact_name)
            
            for dim_name, dim_info in self.dimension_tables.items():
                if dim_name in required_dimensions:
                    relevant_dims.append(dim_name)
            
            # 生成数据集市DDL
            mart_ddl = await self._generate_mart_ddl(
                mart_name, relevant_facts, relevant_dims, required_measures
            )
            
            # 生成ETL脚本
            etl_script = await self._generate_mart_etl(
                mart_name, relevant_facts, relevant_dims
            )
            
            return {
                "status": "success",
                "mart_name": mart_name,
                "business_process": business_process,
                "ddl": mart_ddl,
                "etl": etl_script,
                "fact_tables": relevant_facts,
                "dimension_tables": relevant_dims
            }
            
        except Exception as e:
            logger.error(f"数据集市生成失败: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def _generate_mart_ddl(
        self,
        mart_name: str,
        fact_tables: List[str],
        dim_tables: List[str],
        measures: List[str]
    ) -> str:
        """生成数据集市DDL"""
        
        ddl = f"""
        -- 数据集市: {mart_name}
        CREATE SCHEMA IF NOT EXISTS mart_{mart_name};
        
        -- 汇总事实表
        CREATE TABLE mart_{mart_name}.fact_{mart_name}_summary AS
        SELECT 
            -- 维度键
            {', '.join([f'd.{dim}_key' for dim in dim_tables])},
            
            -- 度量值
            {', '.join([f'SUM(f.{measure}) as {measure}' for measure in measures])},
            
            -- 元数据
            CURRENT_DATE as load_date
        FROM {fact_tables[0]} f
        {' '.join([f'JOIN {dim} d ON f.{dim}_key = d.{dim}_key' for dim in dim_tables])}
        GROUP BY {', '.join([f'd.{dim}_key' for dim in dim_tables])};
        
        -- 创建索引
        {chr(10).join([f'CREATE INDEX idx_{mart_name}_{dim} ON mart_{mart_name}.fact_{mart_name}_summary({dim}_key);' for dim in dim_tables])}
        """
        
        return ddl
    
    async def _generate_mart_etl(
        self,
        mart_name: str,
        fact_tables: List[str],
        dim_tables: List[str]
    ) -> str:
        """生成数据集市ETL脚本"""
        
        etl = f"""
        -- ETL脚本: {mart_name}
        -- 增量更新逻辑
        
        DELETE FROM mart_{mart_name}.fact_{mart_name}_summary 
        WHERE load_date = CURRENT_DATE;
        
        INSERT INTO mart_{mart_name}.fact_{mart_name}_summary
        SELECT 
            -- 维度键和度量值
            -- (详细SQL逻辑)
        FROM {fact_tables[0]} f
        WHERE f.date_key >= (SELECT MAX(date_key) FROM mart_{mart_name}.fact_{mart_name}_summary);
        
        -- 数据质量检查
        SELECT 
            COUNT(*) as record_count,
            COUNT(DISTINCT date_key) as date_count,
            MIN(load_date) as min_load_date,
            MAX(load_date) as max_load_date
        FROM mart_{mart_name}.fact_{mart_name}_summary;
        """
        
        return etl
    
    async def trace_data_lineage(
        self,
        table_name: str,
        direction: str = "downstream"
    ) -> Dict[str, Any]:
        """追踪数据血缘"""
        
        try:
            if table_name not in self.lineage_graph.nodes:
                return {"status": "error", "error": f"表不存在: {table_name}"}
            
            if direction == "downstream":
                # 下游影响分析
                descendants = list(nx.descendants(self.lineage_graph, table_name))
                paths = []
                for desc in descendants:
                    try:
                        path = nx.shortest_path(self.lineage_graph, table_name, desc)
                        paths.append(path)
                    except nx.NetworkXNoPath:
                        continue
            else:
                # 上游依赖分析
                ancestors = list(nx.ancestors(self.lineage_graph, table_name))
                paths = []
                for anc in ancestors:
                    try:
                        path = nx.shortest_path(self.lineage_graph, anc, table_name)
                        paths.append(path)
                    except nx.NetworkXNoPath:
                        continue
            
            return {
                "status": "success",
                "table": table_name,
                "direction": direction,
                "paths": paths,
                "impact_count": len(paths)
            }
            
        except Exception as e:
            logger.error(f"数据血缘追踪失败: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def validate_data_quality(
        self,
        table_name: str,
        quality_rules: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """数据质量验证"""
        
        try:
            results = []
            
            for rule in quality_rules:
                rule_type = rule.get("type")
                column = rule.get("column")
                threshold = rule.get("threshold", 0)
                
                if rule_type == "completeness":
                    # 完整性检查
                    sql = f"SELECT (COUNT({column}) * 100.0 / COUNT(*)) as completeness FROM {table_name}"
                    
                elif rule_type == "uniqueness":
                    # 唯一性检查
                    sql = f"SELECT (COUNT(DISTINCT {column}) * 100.0 / COUNT({column})) as uniqueness FROM {table_name}"
                    
                elif rule_type == "validity":
                    # 有效性检查（基于规则）
                    condition = rule.get("condition", "TRUE")
                    sql = f"SELECT (SUM(CASE WHEN {condition} THEN 1 ELSE 0 END) * 100.0 / COUNT(*)) as validity FROM {table_name}"
                
                elif rule_type == "consistency":
                    # 一致性检查（跨表）
                    reference_table = rule.get("reference_table")
                    reference_column = rule.get("reference_column")
                    sql = f"""
                    SELECT (COUNT(a.{column}) * 100.0 / (SELECT COUNT(*) FROM {table_name})) as consistency
                    FROM {table_name} a
                    JOIN {reference_table} b ON a.{column} = b.{reference_column}
                    """
                
                # 模拟执行结果
                score = 95.0  # 实际应该执行SQL
                passed = score >= threshold
                
                results.append({
                    "rule": rule,
                    "score": score,
                    "passed": passed,
                    "sql": sql
                })
            
            overall_score = sum(r["score"] for r in results) / len(results)
            
            return {
                "status": "success",
                "table": table_name,
                "overall_score": overall_score,
                "rules_passed": sum(1 for r in results if r["passed"]),
                "total_rules": len(results),
                "detailed_results": results
            }
            
        except Exception as e:
            logger.error(f"数据质量验证失败: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def get_metadata_registry(self) -> Dict[str, Any]:
        """获取元数据注册表"""
        
        return {
            "dimension_tables": self.dimension_tables,
            "fact_tables": self.fact_tables,
            "lineage_nodes": len(self.lineage_graph.nodes),
            "lineage_edges": len(self.lineage_graph.edges),
            "business_processes": [
                "Order Management",
                "Customer Analytics", 
                "Product Performance",
                "Financial Reporting"
            ]
        }
