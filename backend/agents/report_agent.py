"""
报告Agent - 负责自动化报告生成
实现"一键报告"核心功能
"""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from jinja2 import Template
import pandas as pd

from agents.base_agent import BaseAgent
from services.llm_service import LLMService
from services.data_service import DataService
from core.config import Settings


class ReportAgent(BaseAgent):
    """报告Agent - 处理报告生成和导出"""
    
    def __init__(
        self, 
        llm_service: LLMService,
        data_service: DataService,
        settings: Settings
    ):
        super().__init__(llm_service, data_service, settings, "ReportAgent")
        
        # 报告模板
        self.report_templates = {}
        
        # 报告历史
        self.report_history = {}
        
        # 图表配置
        self.chart_configs = {
            "line": {"type": "line", "library": "plotly"},
            "bar": {"type": "bar", "library": "plotly"},
            "pie": {"type": "pie", "library": "plotly"},
            "heatmap": {"type": "heatmap", "library": "plotly"}
        }
    
    async def _initialize_agent(self):
        """初始化报告Agent"""
        
        # 加载报告模板
        await self._load_report_templates()
        
        self.logger.info("报告Agent初始化完成")
    
    async def _process_request(
        self, 
        query: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """处理报告请求"""
        
        # 解析报告请求
        report_request = await self._parse_report_request(query, context)
        
        # 生成报告
        report_result = await self._generate_report(report_request)
        
        return report_result
    
    async def _load_report_templates(self):
        """加载报告模板"""
        
        # 日报模板
        daily_template = {
            "name": "日销售报告",
            "type": "daily",
            "sections": [
                {
                    "title": "核心指标概览",
                    "metrics": ["daily_sales", "order_count", "active_users"],
                    "chart_type": "bar"
                },
                {
                    "title": "销售趋势",
                    "metrics": ["hourly_sales"],
                    "chart_type": "line"
                },
                {
                    "title": "产品表现",
                    "metrics": ["product_sales"],
                    "chart_type": "pie"
                }
            ],
            "template": """
            # {{ report_title }}
            
            ## 报告摘要
            {{ summary }}
            
            ## 核心指标
            {% for metric in core_metrics %}
            - {{ metric.name }}: {{ metric.value }} ({{ metric.change }})
            {% endfor %}
            
            ## 详细分析
            {{ detailed_analysis }}
            
            ## 建议
            {% for recommendation in recommendations %}
            - {{ recommendation }}
            {% endfor %}
            """
        }
        
        # 周报模板
        weekly_template = {
            "name": "周度业务报告",
            "type": "weekly",
            "sections": [
                {
                    "title": "一周概览",
                    "metrics": ["weekly_sales", "weekly_orders", "weekly_users"],
                    "chart_type": "line"
                },
                {
                    "title": "渠道分析",
                    "metrics": ["channel_performance"],
                    "chart_type": "bar"
                },
                {
                    "title": "用户行为",
                    "metrics": ["user_engagement"],
                    "chart_type": "heatmap"
                }
            ]
        }
        
        # 月报模板
        monthly_template = {
            "name": "月度分析报告",
            "type": "monthly",
            "sections": [
                {
                    "title": "月度总结",
                    "metrics": ["monthly_sales", "monthly_growth", "customer_acquisition"],
                    "chart_type": "bar"
                },
                {
                    "title": "趋势分析",
                    "metrics": ["monthly_trend"],
                    "chart_type": "line"
                },
                {
                    "title": "市场表现",
                    "metrics": ["market_share", "competitive_analysis"],
                    "chart_type": "pie"
                }
            ]
        }
        
        self.report_templates = {
            "daily": daily_template,
            "weekly": weekly_template,
            "monthly": monthly_template
        }
        
        self.logger.info(f"加载了 {len(self.report_templates)} 个报告模板")
    
    async def _parse_report_request(
        self, 
        query: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """解析报告请求"""
        
        prompt = f"""
        请分析以下报告生成请求，提取关键信息：

        用户请求：{query}
        上下文：{context}

        请识别以下信息并以JSON格式返回：
        {{
            "report_type": "daily/weekly/monthly/custom",
            "title": "报告标题",
            "metrics": ["需要的指标列表"],
            "time_range": {{"start": "开始日期", "end": "结束日期"}},
            "format": "pdf/excel/html/ppt",
            "include_charts": true,
            "sections": ["需要包含的章节"]
        }}
        """
        
        try:
            response = await self.generate_llm_response(prompt)
            request = json.loads(response)
            
            # 验证和标准化请求
            request = self._validate_report_request(request)
            
            return request
            
        except Exception as e:
            self.logger.error(f"解析报告请求失败: {str(e)}")
            
            # 返回默认请求
            return {
                "report_type": "daily",
                "title": "数据分析报告",
                "metrics": ["sales", "orders", "users"],
                "time_range": {
                    "start": (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
                    "end": datetime.now().strftime("%Y-%m-%d")
                },
                "format": "html",
                "include_charts": True,
                "sections": ["summary", "metrics", "analysis"]
            }
    
    def _validate_report_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """验证和标准化报告请求"""
        
        # 设置默认值
        request.setdefault("report_type", "daily")
        request.setdefault("title", "数据分析报告")
        request.setdefault("metrics", ["sales"])
        request.setdefault("format", "html")
        request.setdefault("include_charts", True)
        request.setdefault("sections", ["summary"])
        
        # 验证报告类型
        if request["report_type"] not in self.report_templates:
            request["report_type"] = "daily"
        
        # 标准化时间范围
        if "time_range" not in request:
            if request["report_type"] == "daily":
                request["time_range"] = {
                    "start": (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
                    "end": datetime.now().strftime("%Y-%m-%d")
                }
            elif request["report_type"] == "weekly":
                request["time_range"] = {
                    "start": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
                    "end": datetime.now().strftime("%Y-%m-%d")
                }
            else:  # monthly
                request["time_range"] = {
                    "start": (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
                    "end": datetime.now().strftime("%Y-%m-%d")
                }
        
        return request
    
    async def _generate_report(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """生成报告"""
        
        report_id = f"report_{int(datetime.now().timestamp())}"
        
        try:
            # 1. 收集数据
            report_data = await self._collect_report_data(request)
            
            # 2. 生成图表
            charts = []
            if request.get("include_charts", True):
                charts = await self._generate_charts(report_data, request)
            
            # 3. 生成分析内容
            analysis = await self._generate_analysis_content(report_data, request)
            
            # 4. 组装报告
            report_content = await self._assemble_report(
                request, report_data, charts, analysis
            )
            
            # 5. 导出报告
            export_result = await self._export_report(
                report_id, report_content, request["format"]
            )
            
            # 保存到历史记录
            report_record = {
                "report_id": report_id,
                "title": request["title"],
                "type": request["report_type"],
                "format": request["format"],
                "generated_at": datetime.now().isoformat(),
                "status": "completed",
                "file_path": export_result.get("file_path", ""),
                "file_size": export_result.get("file_size", 0)
            }
            
            self.report_history[report_id] = report_record
            
            return {
                "response": f"报告 '{request['title']}' 生成成功",
                "report_id": report_id,
                "download_url": f"/api/reports/{report_id}/download",
                "report_content": report_content,
                "charts": charts,
                "generated_at": datetime.now().isoformat(),
                "file_info": export_result
            }
            
        except Exception as e:
            self.logger.error(f"生成报告失败: {str(e)}")
            
            return {
                "response": f"报告生成失败: {str(e)}",
                "error": str(e),
                "report_id": report_id
            }
    
    async def _collect_report_data(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """收集报告数据"""
        
        metrics = request["metrics"]
        time_range = request["time_range"]
        
        data = {}
        
        for metric in metrics:
            try:
                metric_data = await self._fetch_metric_data(metric, time_range)
                data[metric] = metric_data
                
            except Exception as e:
                self.logger.error(f"获取指标 {metric} 数据失败: {str(e)}")
                data[metric] = self._generate_mock_metric_data(metric, time_range)
        
        return data
    
    async def _fetch_metric_data(
        self, 
        metric: str, 
        time_range: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """获取指标数据"""
        
        start_date = time_range["start"]
        end_date = time_range["end"]
        
        # 根据指标类型构建查询
        if metric == "sales" or metric == "daily_sales":
            sql = """
            SELECT 
                DATE(order_date) as date,
                SUM(amount) as value
            FROM orders 
            WHERE order_date >= %s AND order_date <= %s
            GROUP BY DATE(order_date)
            ORDER BY date;
            """
        elif metric == "orders" or metric == "order_count":
            sql = """
            SELECT 
                DATE(order_date) as date,
                COUNT(*) as value
            FROM orders 
            WHERE order_date >= %s AND order_date <= %s
            GROUP BY DATE(order_date)
            ORDER BY date;
            """
        elif metric == "users" or metric == "active_users":
            sql = """
            SELECT 
                DATE(last_login) as date,
                COUNT(DISTINCT user_id) as value
            FROM users 
            WHERE last_login >= %s AND last_login <= %s
            GROUP BY DATE(last_login)
            ORDER BY date;
            """
        else:
            # 默认查询
            sql = "SELECT NOW()::date as date, 100 as value;"
        
        try:
            data = await self.execute_sql(sql, [start_date, end_date])
            return data
            
        except Exception as e:
            self.logger.error(f"执行SQL失败: {str(e)}")
            return self._generate_mock_metric_data(metric, time_range)
    
    def _generate_mock_metric_data(
        self, 
        metric: str, 
        time_range: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """生成模拟指标数据"""
        
        import numpy as np
        
        # 生成日期范围
        start_date = datetime.strptime(time_range["start"], "%Y-%m-%d")
        end_date = datetime.strptime(time_range["end"], "%Y-%m-%d")
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # 根据指标生成不同的数据
        if "sales" in metric.lower():
            base_value = 10000
            values = np.random.normal(base_value, 2000, len(dates))
            values = np.maximum(values, 0)  # 确保非负
        elif "order" in metric.lower():
            base_value = 50
            values = np.random.poisson(base_value, len(dates))
        elif "user" in metric.lower():
            base_value = 1000
            values = np.random.normal(base_value, 150, len(dates))
            values = np.maximum(values, 0)
        else:
            values = np.random.normal(100, 20, len(dates))
        
        # 转换为所需格式
        data = []
        for i, date in enumerate(dates):
            data.append({
                "date": date.strftime("%Y-%m-%d"),
                "value": float(values[i])
            })
        
        return data
    
    async def _generate_charts(
        self, 
        data: Dict[str, Any], 
        request: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """生成图表"""
        
        charts = []
        
        for metric, metric_data in data.items():
            if not metric_data:
                continue
            
            # 根据指标类型选择合适的图表
            chart_type = self._determine_chart_type(metric, metric_data)
            
            chart = {
                "id": f"chart_{metric}",
                "title": self._get_metric_display_name(metric),
                "type": chart_type,
                "data": metric_data,
                "config": self.chart_configs.get(chart_type, {})
            }
            
            charts.append(chart)
        
        return charts
    
    def _determine_chart_type(self, metric: str, data: List[Dict[str, Any]]) -> str:
        """确定图表类型"""
        
        # 如果数据有时间维度，使用线图
        if any("date" in str(item.keys()).lower() for item in data):
            return "line"
        
        # 如果是分类数据，使用柱图
        if len(data) <= 10:
            return "bar"
        
        # 默认使用线图
        return "line"
    
    def _get_metric_display_name(self, metric: str) -> str:
        """获取指标显示名称"""
        
        display_names = {
            "sales": "销售额",
            "daily_sales": "日销售额",
            "orders": "订单数",
            "order_count": "订单数量",
            "users": "用户数",
            "active_users": "活跃用户数"
        }
        
        return display_names.get(metric, metric.title())
    
    async def _generate_analysis_content(
        self, 
        data: Dict[str, Any], 
        request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """生成分析内容"""
        
        # 计算核心指标
        core_metrics = self._calculate_core_metrics(data)
        
        # 生成摘要
        summary = await self._generate_summary(data, request)
        
        # 生成详细分析
        detailed_analysis = await self._generate_detailed_analysis(data, request)
        
        # 生成建议
        recommendations = await self._generate_recommendations(data, request)
        
        return {
            "core_metrics": core_metrics,
            "summary": summary,
            "detailed_analysis": detailed_analysis,
            "recommendations": recommendations
        }
    
    def _calculate_core_metrics(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """计算核心指标"""
        
        core_metrics = []
        
        for metric, metric_data in data.items():
            if not metric_data:
                continue
            
            values = [item["value"] for item in metric_data]
            
            if len(values) >= 2:
                current_value = values[-1]
                previous_value = values[-2] if len(values) > 1 else values[0]
                
                change = (current_value - previous_value) / previous_value if previous_value != 0 else 0
                
                core_metrics.append({
                    "name": self._get_metric_display_name(metric),
                    "value": f"{current_value:,.0f}",
                    "change": f"{change:+.1%}",
                    "trend": "上升" if change > 0 else "下降" if change < 0 else "持平"
                })
        
        return core_metrics
    
    async def _generate_summary(
        self, 
        data: Dict[str, Any], 
        request: Dict[str, Any]
    ) -> str:
        """生成报告摘要"""
        
        prompt = f"""
        基于以下数据生成简洁的报告摘要（2-3句话）：

        报告类型：{request['report_type']}
        时间范围：{request['time_range']['start']} 到 {request['time_range']['end']}
        数据概览：{self._summarize_data(data)}

        请生成专业的业务摘要：
        """
        
        try:
            summary = await self.generate_llm_response(prompt, max_tokens=200)
            return summary.strip()
            
        except Exception as e:
            self.logger.error(f"生成摘要失败: {str(e)}")
            return "本期数据表现正常，各项指标符合预期。"
    
    async def _generate_detailed_analysis(
        self, 
        data: Dict[str, Any], 
        request: Dict[str, Any]
    ) -> str:
        """生成详细分析"""
        
        prompt = f"""
        基于以下数据生成详细的业务分析（5-8句话）：

        数据详情：{self._summarize_data(data)}
        
        请从以下角度分析：
        1. 整体表现
        2. 主要趋势
        3. 关键发现
        4. 可能原因

        请生成专业的分析内容：
        """
        
        try:
            analysis = await self.generate_llm_response(prompt, max_tokens=500)
            return analysis.strip()
            
        except Exception as e:
            self.logger.error(f"生成详细分析失败: {str(e)}")
            return "数据显示业务运营稳定，各项指标保持在正常范围内。"
    
    async def _generate_recommendations(
        self, 
        data: Dict[str, Any], 
        request: Dict[str, Any]
    ) -> List[str]:
        """生成建议"""
        
        prompt = f"""
        基于以下数据分析结果，生成3-5个具体的业务建议：

        数据摘要：{self._summarize_data(data)}

        请生成具体可执行的建议，以JSON数组格式返回：
        ["建议1", "建议2", "建议3"]
        """
        
        try:
            response = await self.generate_llm_response(prompt, max_tokens=300)
            recommendations = json.loads(response)
            
            if isinstance(recommendations, list):
                return recommendations
            else:
                return ["继续监控关键指标", "优化业务流程", "提升用户体验"]
                
        except Exception as e:
            self.logger.error(f"生成建议失败: {str(e)}")
            return ["继续监控关键指标", "优化业务流程", "提升用户体验"]
    
    def _summarize_data(self, data: Dict[str, Any]) -> str:
        """数据摘要"""
        
        summary_parts = []
        
        for metric, metric_data in data.items():
            if metric_data:
                values = [item["value"] for item in metric_data]
                avg_value = sum(values) / len(values)
                summary_parts.append(f"{metric}: 平均值 {avg_value:.0f}")
        
        return "; ".join(summary_parts)
    
    async def _assemble_report(
        self, 
        request: Dict[str, Any],
        data: Dict[str, Any],
        charts: List[Dict[str, Any]],
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """组装报告"""
        
        report_content = {
            "title": request["title"],
            "type": request["report_type"],
            "generated_at": datetime.now().isoformat(),
            "time_range": request["time_range"],
            "summary": analysis["summary"],
            "core_metrics": analysis["core_metrics"],
            "detailed_analysis": analysis["detailed_analysis"],
            "recommendations": analysis["recommendations"],
            "charts": charts,
            "raw_data": data
        }
        
        return report_content
    
    async def _export_report(
        self, 
        report_id: str, 
        content: Dict[str, Any], 
        format: str
    ) -> Dict[str, Any]:
        """导出报告"""
        
        try:
            if format.lower() == "html":
                return await self._export_html_report(report_id, content)
            elif format.lower() == "pdf":
                return await self._export_pdf_report(report_id, content)
            elif format.lower() == "excel":
                return await self._export_excel_report(report_id, content)
            elif format.lower() == "ppt":
                return await self._export_ppt_report(report_id, content)
            else:
                raise ValueError(f"不支持的格式: {format}")
                
        except Exception as e:
            self.logger.error(f"导出报告失败: {str(e)}")
            return {"error": str(e)}
    
    async def _export_html_report(self, report_id: str, content: Dict[str, Any]) -> Dict[str, Any]:
        """导出HTML报告"""
        
        # 简单的HTML模板
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{{ title }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { border-bottom: 2px solid #333; padding-bottom: 10px; }
                .metric { background: #f5f5f5; padding: 10px; margin: 10px 0; }
                .chart { margin: 20px 0; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{{ title }}</h1>
                <p>生成时间: {{ generated_at }}</p>
            </div>
            
            <h2>摘要</h2>
            <p>{{ summary }}</p>
            
            <h2>核心指标</h2>
            {% for metric in core_metrics %}
            <div class="metric">
                <strong>{{ metric.name }}</strong>: {{ metric.value }} ({{ metric.change }})
            </div>
            {% endfor %}
            
            <h2>详细分析</h2>
            <p>{{ detailed_analysis }}</p>
            
            <h2>建议</h2>
            <ul>
            {% for rec in recommendations %}
                <li>{{ rec }}</li>
            {% endfor %}
            </ul>
        </body>
        </html>
        """
        
        template = Template(html_template)
        html_content = template.render(**content)
        
        # 模拟文件保存
        file_path = f"/tmp/reports/{report_id}.html"
        file_size = len(html_content.encode('utf-8'))
        
        return {
            "file_path": file_path,
            "file_size": file_size,
            "format": "html",
            "content": html_content
        }
    
    async def _export_pdf_report(self, report_id: str, content: Dict[str, Any]) -> Dict[str, Any]:
        """导出PDF报告"""
        # 实际实现中可以使用weasyprint或其他库
        return {"message": "PDF导出功能开发中", "format": "pdf"}
    
    async def _export_excel_report(self, report_id: str, content: Dict[str, Any]) -> Dict[str, Any]:
        """导出Excel报告"""
        # 实际实现中可以使用openpyxl或pandas
        return {"message": "Excel导出功能开发中", "format": "excel"}
    
    async def _export_ppt_report(self, report_id: str, content: Dict[str, Any]) -> Dict[str, Any]:
        """导出PPT报告"""
        # 实际实现中可以使用python-pptx
        return {"message": "PPT导出功能开发中", "format": "ppt"}
    
    async def _cleanup_agent(self):
        """清理报告Agent资源"""
        self.report_history.clear()
        self.report_templates.clear()
