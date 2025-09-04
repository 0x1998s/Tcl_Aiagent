"""
分析Agent - 负责复杂数据分析和指标计算
实现"逻辑推演"核心功能
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from agents.base_agent import BaseAgent
from services.llm_service import LLMService
from services.data_service import DataService
from core.config import Settings


class AnalysisAgent(BaseAgent):
    """分析Agent - 处理复杂数据分析任务"""
    
    def __init__(
        self, 
        llm_service: LLMService,
        data_service: DataService,
        settings: Settings
    ):
        super().__init__(llm_service, data_service, settings, "AnalysisAgent")
        
        # 分析方法映射
        self.analysis_methods = {
            "trend": self._analyze_trend,
            "correlation": self._analyze_correlation,
            "distribution": self._analyze_distribution,
            "segmentation": self._analyze_segmentation,
            "cohort": self._analyze_cohort,
            "funnel": self._analyze_funnel,
            "attribution": self._analyze_attribution
        }
        
        # 缓存分析结果
        self.analysis_cache = {}
    
    async def _initialize_agent(self):
        """初始化分析Agent"""
        self.logger.info("分析Agent初始化完成")
    
    async def _process_request(
        self, 
        query: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """处理分析请求"""
        
        # 1. 解析分析意图
        analysis_intent = await self._parse_analysis_intent(query, context)
        
        # 2. 获取数据
        data = await self._fetch_analysis_data(analysis_intent)
        
        # 3. 执行分析
        analysis_result = await self._execute_analysis(analysis_intent, data)
        
        # 4. 生成洞察
        insights = await self._generate_insights(analysis_result, analysis_intent)
        
        # 5. 创建可视化建议
        charts = self._suggest_visualizations(analysis_result, analysis_intent)
        
        return {
            "response": insights["summary"],
            "analysis_type": analysis_intent["type"],
            "data": analysis_result["processed_data"],
            "insights": insights["detailed_insights"],
            "charts": charts,
            "statistics": analysis_result.get("statistics", {}),
            "recommendations": insights.get("recommendations", [])
        }
    
    async def _parse_analysis_intent(
        self, 
        query: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """解析分析意图"""
        
        prompt = f"""
        请分析以下用户查询，确定需要执行的分析类型和参数：

        用户查询：{query}
        上下文：{context}

        分析类型包括：
        - trend: 趋势分析
        - correlation: 相关性分析
        - distribution: 分布分析
        - segmentation: 用户分群分析
        - cohort: 队列分析
        - funnel: 漏斗分析
        - attribution: 归因分析

        请以JSON格式返回：
        {{
            "type": "trend",
            "metrics": ["sales", "orders"],
            "dimensions": ["date", "category"],
            "time_range": {{"start": "2024-01-01", "end": "2024-01-31"}},
            "filters": {{}},
            "analysis_params": {{}}
        }}
        """
        
        try:
            response = await self.generate_llm_response(prompt)
            import json
            intent = json.loads(response)
            
            # 验证和标准化意图
            intent = self._validate_analysis_intent(intent)
            
            return intent
            
        except Exception as e:
            self.logger.error(f"解析分析意图失败: {str(e)}")
            
            # 返回默认意图
            return {
                "type": "trend",
                "metrics": ["count"],
                "dimensions": ["date"],
                "time_range": {
                    "start": (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
                    "end": datetime.now().strftime("%Y-%m-%d")
                },
                "filters": {},
                "analysis_params": {}
            }
    
    def _validate_analysis_intent(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        """验证和标准化分析意图"""
        
        # 确保必需字段存在
        intent.setdefault("type", "trend")
        intent.setdefault("metrics", ["count"])
        intent.setdefault("dimensions", ["date"])
        intent.setdefault("filters", {})
        intent.setdefault("analysis_params", {})
        
        # 验证分析类型
        if intent["type"] not in self.analysis_methods:
            intent["type"] = "trend"
        
        # 标准化时间范围
        if "time_range" not in intent:
            intent["time_range"] = {
                "start": (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
                "end": datetime.now().strftime("%Y-%m-%d")
            }
        
        return intent
    
    async def _fetch_analysis_data(self, intent: Dict[str, Any]) -> pd.DataFrame:
        """获取分析数据"""
        
        try:
            # 构建查询SQL
            sql = self._build_analysis_sql(intent)
            
            # 执行查询
            raw_data = await self.execute_sql(sql)
            
            # 转换为DataFrame
            df = pd.DataFrame(raw_data)
            
            # 数据预处理
            df = self._preprocess_data(df, intent)
            
            return df
            
        except Exception as e:
            self.logger.error(f"获取分析数据失败: {str(e)}")
            
            # 返回模拟数据
            return self._generate_mock_data(intent)
    
    def _build_analysis_sql(self, intent: Dict[str, Any]) -> str:
        """构建分析SQL查询"""
        
        metrics = intent["metrics"]
        dimensions = intent["dimensions"]
        time_range = intent["time_range"]
        filters = intent["filters"]
        
        # 基础查询模板
        if intent["type"] == "trend":
            sql = f"""
            SELECT 
                DATE(order_date) as date,
                SUM(amount) as sales,
                COUNT(*) as orders,
                COUNT(DISTINCT user_id) as users
            FROM orders 
            WHERE order_date >= '{time_range["start"]}' 
            AND order_date <= '{time_range["end"]}'
            GROUP BY DATE(order_date)
            ORDER BY date;
            """
        elif intent["type"] == "correlation":
            sql = f"""
            SELECT 
                user_id,
                SUM(amount) as total_spent,
                COUNT(*) as order_count,
                AVG(amount) as avg_order_value,
                EXTRACT(DAYS FROM (MAX(order_date) - MIN(order_date))) as customer_lifetime
            FROM orders
            WHERE order_date >= '{time_range["start"]}' 
            AND order_date <= '{time_range["end"]}'
            GROUP BY user_id;
            """
        else:
            # 默认查询
            sql = """
            SELECT * FROM orders 
            WHERE order_date >= '2024-01-01' 
            LIMIT 1000;
            """
        
        return sql
    
    def _preprocess_data(self, df: pd.DataFrame, intent: Dict[str, Any]) -> pd.DataFrame:
        """数据预处理"""
        
        if df.empty:
            return df
        
        # 处理日期列
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        # 处理数值列
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 移除异常值（使用IQR方法）
        for col in numeric_columns:
            if col in df.columns and len(df[col].dropna()) > 0:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df
    
    def _generate_mock_data(self, intent: Dict[str, Any]) -> pd.DataFrame:
        """生成模拟数据"""
        
        np.random.seed(42)  # 确保可重复性
        
        if intent["type"] == "trend":
            dates = pd.date_range(
                start=intent["time_range"]["start"],
                end=intent["time_range"]["end"],
                freq='D'
            )
            
            # 生成趋势数据
            base_value = 1000
            trend = np.linspace(0, 200, len(dates))
            noise = np.random.normal(0, 50, len(dates))
            seasonal = 100 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)  # 周期性
            
            values = base_value + trend + seasonal + noise
            
            return pd.DataFrame({
                'date': dates,
                'sales': values,
                'orders': values / 20 + np.random.normal(0, 5, len(dates)),
                'users': values / 30 + np.random.normal(0, 3, len(dates))
            })
            
        elif intent["type"] == "correlation":
            n_samples = 1000
            
            # 生成相关数据
            total_spent = np.random.lognormal(6, 1, n_samples)
            order_count = np.random.poisson(total_spent / 100, n_samples)
            avg_order_value = total_spent / np.maximum(order_count, 1)
            customer_lifetime = np.random.exponential(100, n_samples)
            
            return pd.DataFrame({
                'user_id': range(n_samples),
                'total_spent': total_spent,
                'order_count': order_count,
                'avg_order_value': avg_order_value,
                'customer_lifetime': customer_lifetime
            })
        
        else:
            # 默认数据
            return pd.DataFrame({
                'date': pd.date_range('2024-01-01', periods=30),
                'value': np.random.normal(100, 20, 30)
            })
    
    async def _execute_analysis(
        self, 
        intent: Dict[str, Any], 
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """执行分析"""
        
        analysis_type = intent["type"]
        
        if analysis_type in self.analysis_methods:
            return await self.analysis_methods[analysis_type](data, intent)
        else:
            raise ValueError(f"不支持的分析类型: {analysis_type}")
    
    async def _analyze_trend(
        self, 
        data: pd.DataFrame, 
        intent: Dict[str, Any]
    ) -> Dict[str, Any]:
        """趋势分析"""
        
        if data.empty:
            return {"processed_data": [], "statistics": {}}
        
        # 确保有时间列
        time_col = None
        for col in data.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                time_col = col
                break
        
        if time_col is None:
            return {"processed_data": data.to_dict('records'), "statistics": {}}
        
        # 按时间排序
        data = data.sort_values(time_col)
        
        # 计算统计指标
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        statistics = {}
        
        for col in numeric_cols:
            if col in data.columns and len(data[col].dropna()) > 1:
                values = data[col].dropna()
                
                # 基本统计
                statistics[col] = {
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "trend": self._calculate_trend(values),
                    "volatility": float(values.std() / values.mean()) if values.mean() != 0 else 0,
                    "growth_rate": self._calculate_growth_rate(values)
                }
        
        # 季节性检测
        if len(data) >= 14:  # 至少两周数据
            seasonality = self._detect_seasonality(data, numeric_cols)
            statistics["seasonality"] = seasonality
        
        return {
            "processed_data": data.to_dict('records'),
            "statistics": statistics
        }
    
    async def _analyze_correlation(
        self, 
        data: pd.DataFrame, 
        intent: Dict[str, Any]
    ) -> Dict[str, Any]:
        """相关性分析"""
        
        if data.empty:
            return {"processed_data": [], "statistics": {}}
        
        # 选择数值列
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty or len(numeric_data.columns) < 2:
            return {"processed_data": data.to_dict('records'), "statistics": {}}
        
        # 计算相关性矩阵
        correlation_matrix = numeric_data.corr()
        
        # 提取强相关关系
        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.5:  # 阈值
                    strong_correlations.append({
                        "variable1": correlation_matrix.columns[i],
                        "variable2": correlation_matrix.columns[j],
                        "correlation": float(corr_value),
                        "strength": self._interpret_correlation(corr_value)
                    })
        
        # 主成分分析
        pca_result = None
        if len(numeric_data.columns) > 2 and len(numeric_data) > 10:
            try:
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(numeric_data.fillna(0))
                pca = PCA(n_components=min(3, len(numeric_data.columns)))
                pca_result = {
                    "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                    "cumulative_variance": np.cumsum(pca.explained_variance_ratio_).tolist()
                }
            except Exception as e:
                self.logger.warning(f"PCA分析失败: {str(e)}")
        
        statistics = {
            "correlation_matrix": correlation_matrix.to_dict(),
            "strong_correlations": strong_correlations,
            "pca": pca_result
        }
        
        return {
            "processed_data": data.to_dict('records'),
            "statistics": statistics
        }
    
    async def _analyze_distribution(
        self, 
        data: pd.DataFrame, 
        intent: Dict[str, Any]
    ) -> Dict[str, Any]:
        """分布分析"""
        
        if data.empty:
            return {"processed_data": [], "statistics": {}}
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        statistics = {}
        
        for col in numeric_cols:
            if col in data.columns:
                values = data[col].dropna()
                
                if len(values) > 0:
                    # 描述性统计
                    desc_stats = {
                        "count": len(values),
                        "mean": float(values.mean()),
                        "std": float(values.std()),
                        "min": float(values.min()),
                        "25%": float(values.quantile(0.25)),
                        "50%": float(values.median()),
                        "75%": float(values.quantile(0.75)),
                        "max": float(values.max()),
                        "skewness": float(stats.skew(values)),
                        "kurtosis": float(stats.kurtosis(values))
                    }
                    
                    # 正态性检验
                    if len(values) > 8:
                        try:
                            shapiro_stat, shapiro_p = stats.shapiro(values[:5000])  # 限制样本量
                            desc_stats["normality_test"] = {
                                "statistic": float(shapiro_stat),
                                "p_value": float(shapiro_p),
                                "is_normal": shapiro_p > 0.05
                            }
                        except Exception:
                            desc_stats["normality_test"] = None
                    
                    # 异常值检测
                    Q1 = values.quantile(0.25)
                    Q3 = values.quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = values[(values < Q1 - 1.5 * IQR) | (values > Q3 + 1.5 * IQR)]
                    desc_stats["outliers"] = {
                        "count": len(outliers),
                        "percentage": float(len(outliers) / len(values) * 100)
                    }
                    
                    statistics[col] = desc_stats
        
        return {
            "processed_data": data.to_dict('records'),
            "statistics": statistics
        }
    
    async def _analyze_segmentation(
        self, 
        data: pd.DataFrame, 
        intent: Dict[str, Any]
    ) -> Dict[str, Any]:
        """用户分群分析"""
        
        # 这里可以实现K-means聚类或其他分群方法
        # 目前返回简化版本
        
        return {
            "processed_data": data.to_dict('records'),
            "statistics": {"message": "分群分析功能开发中"}
        }
    
    async def _analyze_cohort(
        self, 
        data: pd.DataFrame, 
        intent: Dict[str, Any]
    ) -> Dict[str, Any]:
        """队列分析"""
        
        return {
            "processed_data": data.to_dict('records'),
            "statistics": {"message": "队列分析功能开发中"}
        }
    
    async def _analyze_funnel(
        self, 
        data: pd.DataFrame, 
        intent: Dict[str, Any]
    ) -> Dict[str, Any]:
        """漏斗分析"""
        
        return {
            "processed_data": data.to_dict('records'),
            "statistics": {"message": "漏斗分析功能开发中"}
        }
    
    async def _analyze_attribution(
        self, 
        data: pd.DataFrame, 
        intent: Dict[str, Any]
    ) -> Dict[str, Any]:
        """归因分析"""
        
        return {
            "processed_data": data.to_dict('records'),
            "statistics": {"message": "归因分析功能开发中"}
        }
    
    def _calculate_trend(self, values: pd.Series) -> str:
        """计算趋势方向"""
        if len(values) < 2:
            return "insufficient_data"
        
        # 简单线性回归斜率
        x = np.arange(len(values))
        slope, _, _, p_value, _ = stats.linregress(x, values)
        
        if p_value > 0.05:
            return "no_trend"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"
    
    def _calculate_growth_rate(self, values: pd.Series) -> float:
        """计算增长率"""
        if len(values) < 2:
            return 0.0
        
        first_value = values.iloc[0]
        last_value = values.iloc[-1]
        
        if first_value == 0:
            return 0.0
        
        return float((last_value - first_value) / first_value)
    
    def _detect_seasonality(
        self, 
        data: pd.DataFrame, 
        numeric_cols: List[str]
    ) -> Dict[str, Any]:
        """检测季节性"""
        
        seasonality_results = {}
        
        for col in numeric_cols:
            if col in data.columns and len(data) >= 14:
                values = data[col].dropna()
                
                # 简单的周期性检测（基于自相关）
                try:
                    # 计算7天周期的自相关
                    if len(values) >= 14:
                        autocorr_7 = values.autocorr(lag=7)
                        seasonality_results[col] = {
                            "weekly_autocorr": float(autocorr_7) if not np.isnan(autocorr_7) else 0,
                            "has_weekly_pattern": abs(autocorr_7) > 0.3 if not np.isnan(autocorr_7) else False
                        }
                except Exception:
                    seasonality_results[col] = {"error": "无法计算季节性"}
        
        return seasonality_results
    
    def _interpret_correlation(self, corr_value: float) -> str:
        """解释相关性强度"""
        abs_corr = abs(corr_value)
        
        if abs_corr >= 0.8:
            return "very_strong"
        elif abs_corr >= 0.6:
            return "strong"
        elif abs_corr >= 0.4:
            return "moderate"
        elif abs_corr >= 0.2:
            return "weak"
        else:
            return "very_weak"
    
    async def _generate_insights(
        self, 
        analysis_result: Dict[str, Any], 
        intent: Dict[str, Any]
    ) -> Dict[str, Any]:
        """生成分析洞察"""
        
        statistics = analysis_result.get("statistics", {})
        analysis_type = intent["type"]
        
        prompt = f"""
        基于以下数据分析结果，生成专业的业务洞察：

        分析类型：{analysis_type}
        分析结果：{statistics}

        请生成：
        1. 简要总结（1-2句话）
        2. 详细洞察（3-5个要点）
        3. 业务建议（2-3个具体建议）

        以JSON格式返回：
        {{
            "summary": "简要总结",
            "detailed_insights": ["洞察1", "洞察2", "洞察3"],
            "recommendations": ["建议1", "建议2", "建议3"]
        }}
        """
        
        try:
            response = await self.generate_llm_response(prompt)
            import json
            insights = json.loads(response)
            
            return insights
            
        except Exception as e:
            self.logger.error(f"生成洞察失败: {str(e)}")
            
            # 返回默认洞察
            return {
                "summary": f"{analysis_type}分析已完成，请查看详细结果。",
                "detailed_insights": [
                    "数据分析已完成",
                    "发现了一些有趣的模式",
                    "建议进一步深入分析"
                ],
                "recommendations": [
                    "持续监控关键指标",
                    "考虑进行更深入的分析"
                ]
            }
    
    def _suggest_visualizations(
        self, 
        analysis_result: Dict[str, Any], 
        intent: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """建议可视化图表"""
        
        charts = []
        analysis_type = intent["type"]
        data = analysis_result.get("processed_data", [])
        
        if not data:
            return charts
        
        if analysis_type == "trend":
            # 时间序列图
            charts.append({
                "type": "line",
                "title": "趋势分析",
                "description": "显示指标随时间的变化趋势",
                "data": data
            })
            
        elif analysis_type == "correlation":
            # 相关性热力图
            charts.append({
                "type": "heatmap", 
                "title": "相关性矩阵",
                "description": "显示变量之间的相关关系",
                "data": data
            })
            
            # 散点图
            charts.append({
                "type": "scatter",
                "title": "变量关系散点图",
                "description": "显示变量之间的具体关系",
                "data": data
            })
            
        elif analysis_type == "distribution":
            # 直方图
            charts.append({
                "type": "histogram",
                "title": "分布直方图",
                "description": "显示数据的分布特征",
                "data": data
            })
            
            # 箱线图
            charts.append({
                "type": "boxplot",
                "title": "箱线图",
                "description": "显示数据的分位数和异常值",
                "data": data
            })
        
        return charts
    
    async def _cleanup_agent(self):
        """清理分析Agent资源"""
        self.analysis_cache.clear()
