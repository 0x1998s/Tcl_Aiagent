"""
查询Agent - 负责Text-to-SQL转换和数据查询
实现"对话即洞察"核心功能
"""

import re
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

from agents.base_agent import BaseAgent
from services.llm_service import LLMService
from services.data_service import DataService
from core.config import Settings


class QueryAgent(BaseAgent):
    """查询Agent - 处理自然语言到SQL的转换"""
    
    def __init__(
        self, 
        llm_service: LLMService,
        data_service: DataService,
        settings: Settings
    ):
        super().__init__(llm_service, data_service, settings, "QueryAgent")
        
        # 缓存查询结果
        self.query_cache = {}
        self.schema_cache = {}
        
        # SQL模板库
        self.sql_templates = {
            "sales_query": """
            SELECT 
                DATE(order_date) as date,
                SUM(amount) as total_sales,
                COUNT(*) as order_count
            FROM orders 
            WHERE order_date >= '{start_date}' 
            AND order_date <= '{end_date}'
            GROUP BY DATE(order_date)
            ORDER BY date;
            """,
            
            "user_metrics": """
            SELECT 
                COUNT(DISTINCT user_id) as total_users,
                COUNT(DISTINCT CASE WHEN last_login >= '{recent_date}' THEN user_id END) as active_users,
                AVG(age) as avg_age
            FROM users
            WHERE created_at <= '{end_date}';
            """,
            
            "product_analysis": """
            SELECT 
                p.category,
                p.product_name,
                SUM(oi.quantity) as total_sold,
                SUM(oi.amount) as total_revenue
            FROM products p
            JOIN order_items oi ON p.product_id = oi.product_id
            JOIN orders o ON oi.order_id = o.order_id
            WHERE o.order_date >= '{start_date}'
            AND o.order_date <= '{end_date}'
            GROUP BY p.category, p.product_name
            ORDER BY total_revenue DESC;
            """
        }
    
    async def _initialize_agent(self):
        """初始化查询Agent"""
        # 加载数据库schema
        await self._load_database_schema()
        
        # 预热常用查询
        await self._warmup_common_queries()
    
    async def _load_database_schema(self):
        """加载数据库schema信息"""
        try:
            # 获取所有表信息
            tables_sql = """
            SELECT table_name, column_name, data_type, is_nullable
            FROM information_schema.columns 
            WHERE table_schema = 'public'
            ORDER BY table_name, ordinal_position;
            """
            
            schema_info = await self.execute_sql(tables_sql)
            
            # 组织schema信息
            self.schema_cache = {}
            for row in schema_info:
                table_name = row['table_name']
                if table_name not in self.schema_cache:
                    self.schema_cache[table_name] = {
                        'columns': [],
                        'description': ''
                    }
                
                self.schema_cache[table_name]['columns'].append({
                    'name': row['column_name'],
                    'type': row['data_type'],
                    'nullable': row['is_nullable'] == 'YES'
                })
            
            self.logger.info(f"加载了 {len(self.schema_cache)} 个表的schema信息")
            
        except Exception as e:
            self.logger.error(f"加载数据库schema失败: {str(e)}")
            # 使用默认schema
            self._setup_default_schema()
    
    def _setup_default_schema(self):
        """设置默认schema（用于演示）"""
        self.schema_cache = {
            "orders": {
                "columns": [
                    {"name": "order_id", "type": "integer", "nullable": False},
                    {"name": "user_id", "type": "integer", "nullable": False},
                    {"name": "order_date", "type": "timestamp", "nullable": False},
                    {"name": "amount", "type": "decimal", "nullable": False},
                    {"name": "status", "type": "varchar", "nullable": False}
                ],
                "description": "订单表，包含所有订单信息"
            },
            "users": {
                "columns": [
                    {"name": "user_id", "type": "integer", "nullable": False},
                    {"name": "username", "type": "varchar", "nullable": False},
                    {"name": "email", "type": "varchar", "nullable": False},
                    {"name": "age", "type": "integer", "nullable": True},
                    {"name": "created_at", "type": "timestamp", "nullable": False},
                    {"name": "last_login", "type": "timestamp", "nullable": True}
                ],
                "description": "用户表，包含用户基本信息"
            },
            "products": {
                "columns": [
                    {"name": "product_id", "type": "integer", "nullable": False},
                    {"name": "product_name", "type": "varchar", "nullable": False},
                    {"name": "category", "type": "varchar", "nullable": False},
                    {"name": "price", "type": "decimal", "nullable": False}
                ],
                "description": "产品表，包含产品信息"
            }
        }
    
    async def _warmup_common_queries(self):
        """预热常用查询"""
        # 预执行一些常用查询以建立缓存
        common_queries = [
            "今天的订单数量",
            "本月销售额",
            "活跃用户数"
        ]
        
        for query in common_queries:
            try:
                await self._process_request(query, {})
            except Exception as e:
                self.logger.debug(f"预热查询失败: {query} - {str(e)}")
    
    async def _process_request(
        self, 
        query: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """处理查询请求"""
        
        # 1. 检查缓存
        cache_key = self._generate_cache_key(query, context)
        if cache_key in self.query_cache:
            self.logger.info("使用缓存结果")
            return self.query_cache[cache_key]
        
        # 2. 解析查询意图
        intent = await self._parse_query_intent(query)
        
        # 3. 生成SQL
        sql_info = await self._generate_sql(query, intent, context)
        
        # 4. 执行SQL
        result = await self._execute_and_format_query(sql_info)
        
        # 5. 生成自然语言响应
        response = await self._generate_natural_response(query, result)
        
        # 6. 缓存结果
        final_result = {
            "response": response,
            "sql": sql_info.get("sql", ""),
            "data": result.get("data", []),
            "charts": result.get("charts", []),
            "metadata": {
                "intent": intent,
                "execution_time": result.get("execution_time", 0),
                "row_count": len(result.get("data", []))
            }
        }
        
        self.query_cache[cache_key] = final_result
        
        return final_result
    
    async def _parse_query_intent(self, query: str) -> Dict[str, Any]:
        """解析查询意图"""
        
        prompt = f"""
        请分析以下用户查询的意图，提取关键信息：

        用户查询：{query}

        请识别以下信息并以JSON格式返回：
        1. 查询类型（metrics/trend/comparison/filter）
        2. 时间范围（如：今天、本月、上季度等）
        3. 指标名称（如：销售额、用户数、订单数等）
        4. 维度（如：按地区、按产品、按时间等）
        5. 筛选条件

        返回格式：
        {{
            "query_type": "metrics",
            "time_range": "today",
            "metrics": ["sales", "orders"],
            "dimensions": ["date"],
            "filters": []
        }}
        """
        
        try:
            response = await self.generate_llm_response(prompt)
            
            # 尝试解析JSON
            intent = json.loads(response)
            
            # 标准化时间范围
            intent["time_range"] = self._normalize_time_range(
                intent.get("time_range", "")
            )
            
            return intent
            
        except Exception as e:
            self.logger.error(f"解析查询意图失败: {str(e)}")
            
            # 返回默认意图
            return {
                "query_type": "metrics",
                "time_range": "today",
                "metrics": ["count"],
                "dimensions": ["date"],
                "filters": []
            }
    
    def _normalize_time_range(self, time_range: str) -> Tuple[str, str]:
        """标准化时间范围"""
        now = datetime.now()
        
        time_mapping = {
            "today": (now.strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d")),
            "yesterday": (
                (now - timedelta(days=1)).strftime("%Y-%m-%d"),
                (now - timedelta(days=1)).strftime("%Y-%m-%d")
            ),
            "this_week": (
                (now - timedelta(days=now.weekday())).strftime("%Y-%m-%d"),
                now.strftime("%Y-%m-%d")
            ),
            "this_month": (
                now.replace(day=1).strftime("%Y-%m-%d"),
                now.strftime("%Y-%m-%d")
            ),
            "last_month": (
                (now.replace(day=1) - timedelta(days=1)).replace(day=1).strftime("%Y-%m-%d"),
                (now.replace(day=1) - timedelta(days=1)).strftime("%Y-%m-%d")
            ),
            "this_quarter": self._get_quarter_range(now),
            "last_quarter": self._get_quarter_range(now, -1)
        }
        
        return time_mapping.get(time_range.lower().replace(" ", "_"), 
                              (now.strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d")))
    
    def _get_quarter_range(self, date: datetime, offset: int = 0) -> Tuple[str, str]:
        """获取季度范围"""
        quarter = (date.month - 1) // 3 + 1 + offset
        year = date.year
        
        if quarter <= 0:
            quarter += 4
            year -= 1
        elif quarter > 4:
            quarter -= 4
            year += 1
        
        start_month = (quarter - 1) * 3 + 1
        end_month = quarter * 3
        
        start_date = datetime(year, start_month, 1)
        
        if end_month == 12:
            end_date = datetime(year, 12, 31)
        else:
            end_date = datetime(year, end_month + 1, 1) - timedelta(days=1)
        
        return (start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    
    async def _generate_sql(
        self, 
        query: str, 
        intent: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """生成SQL查询"""
        
        # 构建schema信息用于prompt
        schema_info = self._build_schema_prompt()
        
        # 解析时间范围
        start_date, end_date = intent["time_range"]
        
        prompt = f"""
        你是一个专业的SQL生成专家。根据用户的自然语言查询和数据库schema，生成准确的SQL查询。

        数据库Schema：
        {schema_info}

        用户查询：{query}
        
        查询意图分析：{json.dumps(intent, ensure_ascii=False)}
        
        时间范围：{start_date} 到 {end_date}

        请生成SQL查询，要求：
        1. 使用标准SQL语法
        2. 包含适当的聚合函数
        3. 正确处理时间过滤
        4. 添加必要的ORDER BY和LIMIT
        5. 确保查询效率

        只返回SQL语句，不要包含其他文字：
        """
        
        try:
            sql = await self.generate_llm_response(prompt)
            
            # 清理SQL
            sql = self._clean_sql(sql)
            
            # 验证SQL
            validation_result = self._validate_sql(sql)
            
            return {
                "sql": sql,
                "intent": intent,
                "validation": validation_result,
                "parameters": {
                    "start_date": start_date,
                    "end_date": end_date
                }
            }
            
        except Exception as e:
            self.logger.error(f"SQL生成失败: {str(e)}")
            
            # 使用模板SQL作为fallback
            return self._get_template_sql(intent)
    
    def _build_schema_prompt(self) -> str:
        """构建schema信息的prompt"""
        schema_text = ""
        
        for table_name, table_info in self.schema_cache.items():
            schema_text += f"\n表名: {table_name}\n"
            schema_text += f"描述: {table_info.get('description', '')}\n"
            schema_text += "字段:\n"
            
            for column in table_info['columns']:
                nullable = "NULL" if column['nullable'] else "NOT NULL"
                schema_text += f"  - {column['name']}: {column['type']} {nullable}\n"
        
        return schema_text
    
    def _clean_sql(self, sql: str) -> str:
        """清理和格式化SQL"""
        # 移除markdown代码块标记
        sql = re.sub(r'```sql\n?', '', sql)
        sql = re.sub(r'```\n?', '', sql)
        
        # 移除多余的空白
        sql = re.sub(r'\s+', ' ', sql.strip())
        
        # 确保以分号结尾
        if not sql.endswith(';'):
            sql += ';'
        
        return sql
    
    def _validate_sql(self, sql: str) -> Dict[str, Any]:
        """验证SQL语法（简单检查）"""
        validation = {
            "is_valid": True,
            "warnings": [],
            "errors": []
        }
        
        # 基本安全检查
        dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE']
        sql_upper = sql.upper()
        
        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                validation["is_valid"] = False
                validation["errors"].append(f"不允许使用 {keyword} 语句")
        
        # 检查是否包含SELECT
        if not sql_upper.startswith('SELECT'):
            validation["warnings"].append("查询应该以SELECT开头")
        
        return validation
    
    def _get_template_sql(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        """获取模板SQL（fallback）"""
        query_type = intent.get("query_type", "metrics")
        start_date, end_date = intent["time_range"]
        
        if query_type == "metrics":
            sql = self.sql_templates["sales_query"].format(
                start_date=start_date,
                end_date=end_date
            )
        else:
            sql = "SELECT 1 as dummy_result;"
        
        return {
            "sql": sql,
            "intent": intent,
            "validation": {"is_valid": True, "warnings": [], "errors": []},
            "parameters": {"start_date": start_date, "end_date": end_date},
            "is_template": True
        }
    
    async def _execute_and_format_query(self, sql_info: Dict[str, Any]) -> Dict[str, Any]:
        """执行SQL并格式化结果"""
        
        if not sql_info["validation"]["is_valid"]:
            return {
                "error": "SQL验证失败",
                "validation_errors": sql_info["validation"]["errors"]
            }
        
        try:
            start_time = datetime.now()
            
            # 执行SQL
            raw_data = await self.execute_sql(sql_info["sql"])
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # 格式化数据
            formatted_data = self._format_query_result(raw_data, sql_info["intent"])
            
            return {
                "data": formatted_data["data"],
                "charts": formatted_data.get("charts", []),
                "summary": formatted_data.get("summary", {}),
                "execution_time": execution_time,
                "row_count": len(raw_data)
            }
            
        except Exception as e:
            self.logger.error(f"SQL执行失败: {str(e)}")
            return {
                "error": f"查询执行失败: {str(e)}",
                "sql": sql_info["sql"]
            }
    
    def _format_query_result(
        self, 
        raw_data: List[Dict[str, Any]], 
        intent: Dict[str, Any]
    ) -> Dict[str, Any]:
        """格式化查询结果"""
        
        if not raw_data:
            return {
                "data": [],
                "summary": {"message": "未找到匹配的数据"}
            }
        
        # 基本数据格式化
        formatted_data = []
        for row in raw_data:
            formatted_row = {}
            for key, value in row.items():
                if isinstance(value, datetime):
                    formatted_row[key] = value.isoformat()
                else:
                    formatted_row[key] = value
            formatted_data.append(formatted_row)
        
        # 生成图表建议
        charts = self._suggest_charts(formatted_data, intent)
        
        # 生成摘要
        summary = self._generate_summary(formatted_data, intent)
        
        return {
            "data": formatted_data,
            "charts": charts,
            "summary": summary
        }
    
    def _suggest_charts(
        self, 
        data: List[Dict[str, Any]], 
        intent: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """建议合适的图表类型"""
        
        if not data:
            return []
        
        charts = []
        
        # 分析数据结构
        first_row = data[0]
        columns = list(first_row.keys())
        
        # 时间序列图表
        date_columns = [col for col in columns if 'date' in col.lower() or 'time' in col.lower()]
        numeric_columns = [col for col in columns if isinstance(first_row[col], (int, float))]
        
        if date_columns and numeric_columns:
            charts.append({
                "type": "line",
                "title": "趋势分析",
                "x_axis": date_columns[0],
                "y_axis": numeric_columns[0],
                "data": data
            })
        
        # 柱状图
        if len(numeric_columns) >= 1:
            charts.append({
                "type": "bar",
                "title": "数值对比",
                "x_axis": columns[0],
                "y_axis": numeric_columns[0],
                "data": data[:10]  # 限制显示前10条
            })
        
        return charts
    
    def _generate_summary(
        self, 
        data: List[Dict[str, Any]], 
        intent: Dict[str, Any]
    ) -> Dict[str, Any]:
        """生成数据摘要"""
        
        if not data:
            return {"message": "无数据"}
        
        summary = {
            "total_records": len(data),
            "query_type": intent.get("query_type", "unknown")
        }
        
        # 计算数值列的统计信息
        first_row = data[0]
        numeric_columns = [col for col in first_row.keys() if isinstance(first_row[col], (int, float))]
        
        for col in numeric_columns:
            values = [row[col] for row in data if row[col] is not None]
            if values:
                summary[f"{col}_sum"] = sum(values)
                summary[f"{col}_avg"] = sum(values) / len(values)
                summary[f"{col}_max"] = max(values)
                summary[f"{col}_min"] = min(values)
        
        return summary
    
    async def _generate_natural_response(
        self, 
        original_query: str, 
        result: Dict[str, Any]
    ) -> str:
        """生成自然语言响应"""
        
        if "error" in result:
            return f"抱歉，查询过程中遇到了问题：{result['error']}"
        
        data = result.get("data", [])
        summary = result.get("summary", {})
        
        if not data:
            return "根据您的查询条件，没有找到相关数据。"
        
        # 构建响应prompt
        prompt = f"""
        用户查询：{original_query}
        
        查询结果摘要：{json.dumps(summary, ensure_ascii=False)}
        数据条数：{len(data)}
        
        请生成一个简洁、专业的自然语言响应，包括：
        1. 直接回答用户的问题
        2. 提及关键数据指标
        3. 如果有趋势，简单说明
        
        响应应该像数据分析师在回答业务问题，简洁明了：
        """
        
        try:
            response = await self.generate_llm_response(prompt, max_tokens=200)
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"生成自然语言响应失败: {str(e)}")
            return f"查询完成！找到 {len(data)} 条记录。"
    
    def _generate_cache_key(self, query: str, context: Dict[str, Any]) -> str:
        """生成缓存键"""
        import hashlib
        
        cache_data = {
            "query": query.lower().strip(),
            "context": context
        }
        
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    async def _cleanup_agent(self):
        """清理查询Agent资源"""
        # 清理缓存
        self.query_cache.clear()
        self.schema_cache.clear()
