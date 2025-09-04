"""
数据工程服务
支持Flink、Spark和dbt的集成
"""

import asyncio
import json
import subprocess
import tempfile
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
import yaml
import pandas as pd

try:
    from pyflink.datastream import StreamExecutionEnvironment
    from pyflink.table import StreamTableEnvironment
    FLINK_AVAILABLE = True
except ImportError:
    FLINK_AVAILABLE = False

try:
    from pyspark.sql import SparkSession
    from pyspark.streaming import StreamingContext
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False

from utils.logger import get_logger
from core.config import Settings

logger = get_logger(__name__)


class DataEngineeringService:
    """数据工程服务"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.flink_env = None
        self.spark_session = None
        self.dbt_project_path = Path("./dbt_project")
        
        # 作业管理
        self.running_jobs = {}
        self.job_history = []
        
    async def initialize(self):
        """初始化数据工程服务"""
        logger.info("初始化数据工程服务...")
        
        # 初始化Flink
        if FLINK_AVAILABLE:
            await self._init_flink()
        else:
            logger.warning("PyFlink不可用，Flink功能将受限")
        
        # 初始化Spark
        if SPARK_AVAILABLE:
            await self._init_spark()
        else:
            logger.warning("PySpark不可用，Spark功能将受限")
        
        # 初始化dbt项目
        await self._init_dbt_project()
        
        logger.info("数据工程服务初始化完成")
    
    async def _init_flink(self):
        """初始化Flink环境"""
        try:
            self.flink_env = StreamExecutionEnvironment.get_execution_environment()
            self.flink_env.set_parallelism(2)
            
            logger.info("Flink环境初始化成功")
            
        except Exception as e:
            logger.error(f"Flink环境初始化失败: {str(e)}")
            self.flink_env = None
    
    async def _init_spark(self):
        """初始化Spark会话"""
        try:
            self.spark_session = SparkSession.builder \
                .appName("TCL-AI-Agent") \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                .getOrCreate()
            
            logger.info("Spark会话初始化成功")
            
        except Exception as e:
            logger.error(f"Spark会话初始化失败: {str(e)}")
            self.spark_session = None
    
    async def _init_dbt_project(self):
        """初始化dbt项目"""
        try:
            # 创建dbt项目目录结构
            await self._create_dbt_project_structure()
            
            # 创建默认配置文件
            await self._create_dbt_profiles()
            
            logger.info("dbt项目初始化成功")
            
        except Exception as e:
            logger.error(f"dbt项目初始化失败: {str(e)}")
    
    async def _create_dbt_project_structure(self):
        """创建dbt项目结构"""
        
        # 项目目录
        directories = [
            self.dbt_project_path,
            self.dbt_project_path / "models",
            self.dbt_project_path / "models" / "staging",
            self.dbt_project_path / "models" / "marts",
            self.dbt_project_path / "macros",
            self.dbt_project_path / "tests",
            self.dbt_project_path / "seeds",
            self.dbt_project_path / "snapshots"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        # dbt_project.yml
        project_config = {
            "name": "tcl_ai_agent",
            "version": "1.0.0",
            "config-version": 2,
            "profile": "tcl_ai_agent",
            "model-paths": ["models"],
            "analysis-paths": ["analyses"],
            "test-paths": ["tests"],
            "seed-paths": ["seeds"],
            "macro-paths": ["macros"],
            "snapshot-paths": ["snapshots"],
            "target-path": "target",
            "clean-targets": ["target", "dbt_packages"],
            "models": {
                "tcl_ai_agent": {
                    "staging": {
                        "materialized": "view"
                    },
                    "marts": {
                        "materialized": "table"
                    }
                }
            }
        }
        
        with open(self.dbt_project_path / "dbt_project.yml", "w") as f:
            yaml.dump(project_config, f, default_flow_style=False)
        
        # 创建示例模型
        await self._create_sample_dbt_models()
    
    async def _create_dbt_profiles(self):
        """创建dbt profiles.yml"""
        
        profiles_dir = Path.home() / ".dbt"
        profiles_dir.mkdir(exist_ok=True)
        
        profiles_config = {
            "tcl_ai_agent": {
                "target": "dev",
                "outputs": {
                    "dev": {
                        "type": "postgres",
                        "host": "localhost",
                        "user": "postgres",
                        "password": "password",
                        "port": 5432,
                        "dbname": "tcl_agent",
                        "schema": "dbt_dev",
                        "threads": 4,
                        "keepalives_idle": 0
                    },
                    "prod": {
                        "type": "postgres", 
                        "host": "localhost",
                        "user": "postgres",
                        "password": "password",
                        "port": 5432,
                        "dbname": "tcl_agent",
                        "schema": "dbt_prod",
                        "threads": 8,
                        "keepalives_idle": 0
                    }
                }
            }
        }
        
        profiles_path = profiles_dir / "profiles.yml"
        if not profiles_path.exists():
            with open(profiles_path, "w") as f:
                yaml.dump(profiles_config, f, default_flow_style=False)
    
    async def _create_sample_dbt_models(self):
        """创建示例dbt模型"""
        
        # staging层模型
        staging_users_sql = """
{{ config(materialized='view') }}

select
    id as user_id,
    username,
    email,
    created_at,
    updated_at
from {{ source('raw', 'users') }}
        """
        
        staging_orders_sql = """
{{ config(materialized='view') }}

select
    id as order_id,
    user_id,
    product_id,
    amount,
    order_date,
    status
from {{ source('raw', 'orders') }}
        """
        
        # marts层模型
        user_metrics_sql = """
{{ config(materialized='table') }}

with user_orders as (
    select
        u.user_id,
        u.username,
        u.email,
        count(o.order_id) as total_orders,
        sum(o.amount) as total_spent,
        avg(o.amount) as avg_order_value,
        min(o.order_date) as first_order_date,
        max(o.order_date) as last_order_date
    from {{ ref('stg_users') }} u
    left join {{ ref('stg_orders') }} o on u.user_id = o.user_id
    group by u.user_id, u.username, u.email
)

select
    *,
    case 
        when total_orders >= 10 then 'High Value'
        when total_orders >= 5 then 'Medium Value'
        else 'Low Value'
    end as customer_segment,
    {{ datediff('first_order_date', 'last_order_date', 'day') }} as customer_lifetime_days
from user_orders
        """
        
        # 写入文件
        models = {
            "staging/stg_users.sql": staging_users_sql,
            "staging/stg_orders.sql": staging_orders_sql,
            "marts/user_metrics.sql": user_metrics_sql
        }
        
        for path, content in models.items():
            model_path = self.dbt_project_path / "models" / path
            model_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(model_path, "w") as f:
                f.write(content.strip())
        
        # schema.yml
        schema_yml = """
version: 2

sources:
  - name: raw
    tables:
      - name: users
        columns:
          - name: id
            tests:
              - unique
              - not_null
      - name: orders
        columns:
          - name: id
            tests:
              - unique
              - not_null
          - name: user_id
            tests:
              - not_null

models:
  - name: stg_users
    description: "Staged user data"
    columns:
      - name: user_id
        tests:
          - unique
          - not_null
  
  - name: stg_orders
    description: "Staged order data"
    columns:
      - name: order_id
        tests:
          - unique
          - not_null
  
  - name: user_metrics
    description: "User-level metrics and segmentation"
    columns:
      - name: user_id
        tests:
          - unique
          - not_null
      - name: customer_segment
        tests:
          - accepted_values:
              values: ['High Value', 'Medium Value', 'Low Value']
        """
        
        with open(self.dbt_project_path / "models" / "schema.yml", "w") as f:
            f.write(schema_yml.strip())
    
    async def run_flink_job(
        self,
        job_name: str,
        source_config: Dict[str, Any],
        sink_config: Dict[str, Any],
        transformation: Optional[str] = None
    ) -> Dict[str, Any]:
        """运行Flink流处理作业"""
        
        if not FLINK_AVAILABLE or not self.flink_env:
            return {
                "status": "error",
                "error": "Flink不可用"
            }
        
        try:
            logger.info(f"启动Flink作业: {job_name}")
            
            # 这里是简化的Flink作业示例
            # 实际生产中需要根据具体需求实现
            
            job_id = f"flink_{job_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # 模拟作业执行
            self.running_jobs[job_id] = {
                "job_name": job_name,
                "status": "running",
                "start_time": datetime.now(),
                "source_config": source_config,
                "sink_config": sink_config
            }
            
            # 记录作业历史
            self.job_history.append({
                "job_id": job_id,
                "job_name": job_name,
                "type": "flink",
                "status": "started",
                "timestamp": datetime.now()
            })
            
            return {
                "status": "success",
                "job_id": job_id,
                "message": f"Flink作业 {job_name} 启动成功"
            }
            
        except Exception as e:
            logger.error(f"Flink作业执行失败: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def run_spark_job(
        self,
        job_name: str,
        sql_query: str,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """运行Spark批处理作业"""
        
        if not SPARK_AVAILABLE or not self.spark_session:
            return {
                "status": "error",
                "error": "Spark不可用"
            }
        
        try:
            logger.info(f"启动Spark作业: {job_name}")
            
            job_id = f"spark_{job_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # 执行SQL查询
            df = self.spark_session.sql(sql_query)
            
            # 如果指定输出路径，保存结果
            if output_path:
                df.write.mode("overwrite").parquet(output_path)
            
            # 获取结果统计
            row_count = df.count()
            
            # 记录作业信息
            self.job_history.append({
                "job_id": job_id,
                "job_name": job_name,
                "type": "spark",
                "status": "completed",
                "row_count": row_count,
                "timestamp": datetime.now()
            })
            
            return {
                "status": "success",
                "job_id": job_id,
                "row_count": row_count,
                "output_path": output_path,
                "message": f"Spark作业 {job_name} 完成"
            }
            
        except Exception as e:
            logger.error(f"Spark作业执行失败: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def run_dbt_command(
        self,
        command: str,
        models: Optional[List[str]] = None,
        vars: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """运行dbt命令"""
        
        try:
            # 构建dbt命令
            cmd = ["dbt", command, "--project-dir", str(self.dbt_project_path)]
            
            if models:
                cmd.extend(["--models", " ".join(models)])
            
            if vars:
                vars_str = json.dumps(vars)
                cmd.extend(["--vars", vars_str])
            
            logger.info(f"执行dbt命令: {' '.join(cmd)}")
            
            # 执行命令
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.dbt_project_path
            )
            
            # 记录作业历史
            job_id = f"dbt_{command}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.job_history.append({
                "job_id": job_id,
                "job_name": f"dbt {command}",
                "type": "dbt",
                "status": "completed" if result.returncode == 0 else "failed",
                "timestamp": datetime.now()
            })
            
            return {
                "status": "success" if result.returncode == 0 else "error",
                "job_id": job_id,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": " ".join(cmd)
            }
            
        except Exception as e:
            logger.error(f"dbt命令执行失败: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def create_data_pipeline(
        self,
        pipeline_name: str,
        pipeline_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """创建数据管道"""
        
        try:
            # 解析管道配置
            steps = pipeline_config.get("steps", [])
            
            pipeline_id = f"pipeline_{pipeline_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            results = []
            
            for i, step in enumerate(steps):
                step_type = step.get("type")
                step_name = step.get("name", f"step_{i}")
                
                logger.info(f"执行管道步骤: {step_name} ({step_type})")
                
                if step_type == "flink":
                    result = await self.run_flink_job(
                        step_name,
                        step.get("source", {}),
                        step.get("sink", {}),
                        step.get("transformation")
                    )
                elif step_type == "spark":
                    result = await self.run_spark_job(
                        step_name,
                        step.get("sql", "SELECT 1"),
                        step.get("output_path")
                    )
                elif step_type == "dbt":
                    result = await self.run_dbt_command(
                        step.get("command", "run"),
                        step.get("models"),
                        step.get("vars")
                    )
                else:
                    result = {
                        "status": "error",
                        "error": f"不支持的步骤类型: {step_type}"
                    }
                
                results.append({
                    "step_name": step_name,
                    "step_type": step_type,
                    "result": result
                })
                
                # 如果步骤失败，停止管道
                if result.get("status") != "success":
                    logger.error(f"管道步骤失败: {step_name}")
                    break
            
            # 记录管道执行
            self.job_history.append({
                "job_id": pipeline_id,
                "job_name": pipeline_name,
                "type": "pipeline",
                "status": "completed",
                "steps": len(results),
                "timestamp": datetime.now()
            })
            
            return {
                "status": "success",
                "pipeline_id": pipeline_id,
                "pipeline_name": pipeline_name,
                "steps_executed": len(results),
                "results": results
            }
            
        except Exception as e:
            logger.error(f"数据管道执行失败: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def schedule_job(
        self,
        job_name: str,
        job_config: Dict[str, Any],
        schedule: str
    ) -> Dict[str, Any]:
        """调度作业"""
        
        # 简化的调度实现
        # 实际生产中应该使用Airflow或其他调度系统
        
        try:
            job_id = f"scheduled_{job_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # 这里应该实现实际的调度逻辑
            # 比如创建cron job或使用调度框架
            
            logger.info(f"作业调度成功: {job_name}, 调度规则: {schedule}")
            
            return {
                "status": "success",
                "job_id": job_id,
                "job_name": job_name,
                "schedule": schedule,
                "message": "作业调度成功"
            }
            
        except Exception as e:
            logger.error(f"作业调度失败: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """获取作业状态"""
        
        if job_id in self.running_jobs:
            return self.running_jobs[job_id]
        
        # 从历史记录中查找
        for job in self.job_history:
            if job["job_id"] == job_id:
                return job
        
        return {
            "status": "not_found",
            "error": f"作业不存在: {job_id}"
        }
    
    async def list_jobs(
        self,
        job_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """列出作业"""
        
        jobs = self.job_history.copy()
        
        # 添加运行中的作业
        for job_id, job_info in self.running_jobs.items():
            jobs.append({
                "job_id": job_id,
                "job_name": job_info["job_name"],
                "type": "running",
                "status": job_info["status"],
                "timestamp": job_info["start_time"]
            })
        
        # 过滤
        if job_type:
            jobs = [job for job in jobs if job.get("type") == job_type]
        
        if status:
            jobs = [job for job in jobs if job.get("status") == status]
        
        # 排序并限制数量
        jobs.sort(key=lambda x: x.get("timestamp", datetime.min), reverse=True)
        
        return jobs[:limit]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取服务统计信息"""
        
        return {
            "flink_available": FLINK_AVAILABLE and self.flink_env is not None,
            "spark_available": SPARK_AVAILABLE and self.spark_session is not None,
            "dbt_project_path": str(self.dbt_project_path),
            "running_jobs": len(self.running_jobs),
            "total_jobs": len(self.job_history),
            "job_types": {
                job_type: len([j for j in self.job_history if j.get("type") == job_type])
                for job_type in ["flink", "spark", "dbt", "pipeline"]
            }
        }
