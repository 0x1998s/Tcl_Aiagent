"""
开源框架集成服务
支持EDA工具和高级分析框架
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import pandas as pd
import numpy as np

try:
    # 尝试导入专业EDA库
    import sweetviz as sv
    import pandas_profiling as pp
    ADVANCED_EDA_AVAILABLE = True
except ImportError:
    ADVANCED_EDA_AVAILABLE = False

try:
    # 尝试导入统计分析库
    import statsmodels.api as sm
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    # 尝试导入机器学习解释库
    import shap
    import lime
    import eli5
    EXPLAINABILITY_AVAILABLE = True
except ImportError:
    EXPLAINABILITY_AVAILABLE = False

from utils.logger import get_logger
from core.config import Settings

logger = get_logger(__name__)


class FrameworkService:
    """开源框架集成服务"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        
    async def initialize(self):
        """初始化框架服务"""
        logger.info("初始化开源框架集成服务...")
        
        # 检查可用框架
        frameworks = []
        if ADVANCED_EDA_AVAILABLE:
            frameworks.append("SweetViz + Pandas Profiling")
        if STATSMODELS_AVAILABLE:
            frameworks.append("StatsModels")
        if EXPLAINABILITY_AVAILABLE:
            frameworks.append("SHAP + LIME + ELI5")
        
        logger.info(f"可用框架: {', '.join(frameworks)}")
        logger.info("开源框架集成服务初始化完成")
    
    async def advanced_eda_report(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None,
        report_type: str = "sweetviz"
    ) -> Dict[str, Any]:
        """生成高级EDA报告"""
        
        try:
            if not ADVANCED_EDA_AVAILABLE:
                return await self._basic_eda_report(data, target_column)
            
            if report_type == "sweetviz":
                return await self._sweetviz_report(data, target_column)
            elif report_type == "pandas_profiling":
                return await self._pandas_profiling_report(data)
            else:
                return await self._basic_eda_report(data, target_column)
                
        except Exception as e:
            logger.error(f"EDA报告生成失败: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def _sweetviz_report(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """SweetViz EDA报告"""
        
        try:
            # 生成SweetViz报告
            if target_column and target_column in data.columns:
                report = sv.analyze(data, target_feat=target_column)
            else:
                report = sv.analyze(data)
            
            # 保存报告
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = f"./reports/sweetviz_report_{timestamp}.html"
            report.show_html(report_path, open_browser=False)
            
            return {
                "status": "success",
                "report_type": "sweetviz",
                "report_path": report_path,
                "summary": {
                    "rows": len(data),
                    "columns": len(data.columns),
                    "target_column": target_column,
                    "missing_values": data.isnull().sum().sum(),
                    "duplicate_rows": data.duplicated().sum()
                }
            }
            
        except Exception as e:
            logger.error(f"SweetViz报告生成失败: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def _pandas_profiling_report(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Pandas Profiling报告"""
        
        try:
            # 生成Pandas Profiling报告
            profile = pp.ProfileReport(
                data,
                title="数据探索报告",
                explorative=True,
                minimal=False
            )
            
            # 保存报告
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = f"./reports/profiling_report_{timestamp}.html"
            profile.to_file(report_path)
            
            return {
                "status": "success",
                "report_type": "pandas_profiling",
                "report_path": report_path,
                "summary": profile.get_description()
            }
            
        except Exception as e:
            logger.error(f"Pandas Profiling报告生成失败: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def _basic_eda_report(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """基础EDA报告"""
        
        try:
            # 基础统计信息
            basic_stats = {
                "shape": data.shape,
                "columns": list(data.columns),
                "dtypes": data.dtypes.to_dict(),
                "missing_values": data.isnull().sum().to_dict(),
                "duplicate_rows": int(data.duplicated().sum()),
                "memory_usage": float(data.memory_usage(deep=True).sum() / 1024 / 1024)  # MB
            }
            
            # 数值列统计
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                basic_stats["numeric_summary"] = data[numeric_columns].describe().to_dict()
            
            # 分类列统计
            categorical_columns = data.select_dtypes(include=['object', 'category']).columns
            if len(categorical_columns) > 0:
                basic_stats["categorical_summary"] = {}
                for col in categorical_columns:
                    basic_stats["categorical_summary"][col] = {
                        "unique_count": int(data[col].nunique()),
                        "top_values": data[col].value_counts().head(5).to_dict()
                    }
            
            # 目标列分析
            if target_column and target_column in data.columns:
                if data[target_column].dtype in [np.number]:
                    basic_stats["target_analysis"] = {
                        "type": "numeric",
                        "stats": data[target_column].describe().to_dict()
                    }
                else:
                    basic_stats["target_analysis"] = {
                        "type": "categorical",
                        "distribution": data[target_column].value_counts().to_dict()
                    }
            
            return {
                "status": "success",
                "report_type": "basic",
                "summary": basic_stats
            }
            
        except Exception as e:
            logger.error(f"基础EDA报告生成失败: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def time_series_analysis(
        self,
        data: pd.DataFrame,
        date_column: str,
        value_column: str,
        analysis_type: str = "decomposition"
    ) -> Dict[str, Any]:
        """时间序列分析"""
        
        try:
            if not STATSMODELS_AVAILABLE:
                return {"status": "error", "error": "StatsModels库不可用"}
            
            # 数据预处理
            ts_data = data[[date_column, value_column]].copy()
            ts_data[date_column] = pd.to_datetime(ts_data[date_column])
            ts_data = ts_data.set_index(date_column).sort_index()
            
            results = {}
            
            if analysis_type == "decomposition":
                # 时间序列分解
                decomposition = seasonal_decompose(
                    ts_data[value_column],
                    model='additive',
                    period=min(12, len(ts_data) // 2)  # 自动选择周期
                )
                
                results["decomposition"] = {
                    "trend": decomposition.trend.dropna().to_dict(),
                    "seasonal": decomposition.seasonal.dropna().to_dict(),
                    "residual": decomposition.resid.dropna().to_dict()
                }
            
            elif analysis_type == "arima":
                # ARIMA模型
                model = ARIMA(ts_data[value_column], order=(1, 1, 1))
                fitted_model = model.fit()
                
                # 预测
                forecast = fitted_model.forecast(steps=12)
                
                results["arima"] = {
                    "model_summary": str(fitted_model.summary()),
                    "aic": float(fitted_model.aic),
                    "bic": float(fitted_model.bic),
                    "forecast": forecast.to_dict()
                }
            
            elif analysis_type == "stationarity":
                # 平稳性检验
                from statsmodels.tsa.stattools import adfuller
                
                adf_result = adfuller(ts_data[value_column].dropna())
                
                results["stationarity"] = {
                    "adf_statistic": float(adf_result[0]),
                    "p_value": float(adf_result[1]),
                    "critical_values": {k: float(v) for k, v in adf_result[4].items()},
                    "is_stationary": adf_result[1] < 0.05
                }
            
            return {
                "status": "success",
                "analysis_type": analysis_type,
                "data_points": len(ts_data),
                "date_range": {
                    "start": ts_data.index.min().isoformat(),
                    "end": ts_data.index.max().isoformat()
                },
                "results": results
            }
            
        except Exception as e:
            logger.error(f"时间序列分析失败: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def model_explainability_analysis(
        self,
        model: Any,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        method: str = "shap"
    ) -> Dict[str, Any]:
        """模型可解释性分析"""
        
        try:
            if not EXPLAINABILITY_AVAILABLE:
                return {"status": "error", "error": "模型解释库不可用"}
            
            results = {}
            
            if method == "shap":
                # SHAP分析
                explainer = shap.Explainer(model)
                shap_values = explainer(X_test.head(100))  # 限制样本数量
                
                results["shap"] = {
                    "feature_importance": {
                        feature: float(np.abs(shap_values.values[:, i]).mean())
                        for i, feature in enumerate(X_test.columns)
                    },
                    "summary": "SHAP值计算完成，可用于特征重要性分析"
                }
            
            elif method == "lime":
                # LIME分析
                from lime.lime_tabular import LimeTabularExplainer
                
                explainer = LimeTabularExplainer(
                    X_train.values,
                    feature_names=X_train.columns,
                    mode='regression'  # 或 'classification'
                )
                
                # 解释单个样本
                explanation = explainer.explain_instance(
                    X_test.iloc[0].values,
                    model.predict,
                    num_features=len(X_test.columns)
                )
                
                results["lime"] = {
                    "local_explanation": explanation.as_list(),
                    "summary": "LIME局部解释完成"
                }
            
            elif method == "eli5":
                # ELI5分析
                import eli5
                
                # 权重解释
                explanation = eli5.explain_weights(
                    model,
                    feature_names=X_train.columns
                )
                
                results["eli5"] = {
                    "explanation": str(explanation),
                    "summary": "ELI5权重解释完成"
                }
            
            return {
                "status": "success",
                "method": method,
                "model_type": type(model).__name__,
                "features": list(X_test.columns),
                "results": results
            }
            
        except Exception as e:
            logger.error(f"模型可解释性分析失败: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def advanced_statistical_tests(
        self,
        data: pd.DataFrame,
        test_type: str,
        **kwargs
    ) -> Dict[str, Any]:
        """高级统计检验"""
        
        try:
            if not STATSMODELS_AVAILABLE:
                return {"status": "error", "error": "StatsModels库不可用"}
            
            results = {}
            
            if test_type == "regression_diagnostics":
                # 回归诊断
                y_col = kwargs.get("y_column")
                x_cols = kwargs.get("x_columns", [])
                
                if y_col and x_cols:
                    y = data[y_col]
                    X = data[x_cols]
                    X = sm.add_constant(X)  # 添加截距项
                    
                    model = sm.OLS(y, X).fit()
                    
                    results["regression_diagnostics"] = {
                        "r_squared": float(model.rsquared),
                        "adj_r_squared": float(model.rsquared_adj),
                        "f_statistic": float(model.fvalue),
                        "f_pvalue": float(model.f_pvalue),
                        "aic": float(model.aic),
                        "bic": float(model.bic),
                        "coefficients": model.params.to_dict(),
                        "p_values": model.pvalues.to_dict(),
                        "confidence_intervals": model.conf_int().to_dict()
                    }
            
            elif test_type == "normality_tests":
                # 正态性检验
                from scipy import stats
                
                column = kwargs.get("column")
                if column and column in data.columns:
                    values = data[column].dropna()
                    
                    # Shapiro-Wilk检验
                    shapiro_stat, shapiro_p = stats.shapiro(values[:5000])  # 限制样本量
                    
                    # Kolmogorov-Smirnov检验
                    ks_stat, ks_p = stats.kstest(values, 'norm')
                    
                    results["normality_tests"] = {
                        "shapiro_wilk": {
                            "statistic": float(shapiro_stat),
                            "p_value": float(shapiro_p),
                            "is_normal": shapiro_p > 0.05
                        },
                        "kolmogorov_smirnov": {
                            "statistic": float(ks_stat),
                            "p_value": float(ks_p),
                            "is_normal": ks_p > 0.05
                        }
                    }
            
            elif test_type == "correlation_matrix":
                # 相关性矩阵和显著性检验
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                corr_matrix = data[numeric_cols].corr()
                
                # 计算p值
                from scipy.stats import pearsonr
                p_values = np.zeros((len(numeric_cols), len(numeric_cols)))
                
                for i, col1 in enumerate(numeric_cols):
                    for j, col2 in enumerate(numeric_cols):
                        if i != j:
                            _, p_val = pearsonr(data[col1].dropna(), data[col2].dropna())
                            p_values[i, j] = p_val
                
                results["correlation_analysis"] = {
                    "correlation_matrix": corr_matrix.to_dict(),
                    "p_values": pd.DataFrame(
                        p_values, 
                        index=numeric_cols, 
                        columns=numeric_cols
                    ).to_dict(),
                    "significant_correlations": [
                        {
                            "var1": col1,
                            "var2": col2,
                            "correlation": float(corr_matrix.loc[col1, col2]),
                            "p_value": float(p_values[i, j])
                        }
                        for i, col1 in enumerate(numeric_cols)
                        for j, col2 in enumerate(numeric_cols)
                        if i < j and p_values[i, j] < 0.05 and abs(corr_matrix.loc[col1, col2]) > 0.3
                    ]
                }
            
            return {
                "status": "success",
                "test_type": test_type,
                "sample_size": len(data),
                "results": results
            }
            
        except Exception as e:
            logger.error(f"统计检验失败: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def feature_engineering_suggestions(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """特征工程建议"""
        
        try:
            suggestions = []
            
            # 分析数据类型
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
            datetime_cols = data.select_dtypes(include=['datetime64']).columns.tolist()
            
            # 缺失值处理建议
            missing_cols = data.columns[data.isnull().any()].tolist()
            if missing_cols:
                suggestions.append({
                    "type": "missing_values",
                    "description": "处理缺失值",
                    "columns": missing_cols,
                    "suggestions": [
                        "数值列：均值/中位数填充",
                        "分类列：众数填充或新增'未知'类别",
                        "考虑删除缺失率>50%的列"
                    ]
                })
            
            # 异常值检测建议
            if numeric_cols:
                outlier_cols = []
                for col in numeric_cols:
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = ((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))).sum()
                    if outliers > len(data) * 0.05:  # 超过5%的异常值
                        outlier_cols.append(col)
                
                if outlier_cols:
                    suggestions.append({
                        "type": "outliers",
                        "description": "异常值处理",
                        "columns": outlier_cols,
                        "suggestions": [
                            "使用IQR方法检测异常值",
                            "考虑对数变换或Box-Cox变换",
                            "使用稳健的统计方法"
                        ]
                    })
            
            # 特征变换建议
            if numeric_cols:
                skewed_cols = []
                for col in numeric_cols:
                    skewness = abs(data[col].skew())
                    if skewness > 1:  # 高度偏斜
                        skewed_cols.append(col)
                
                if skewed_cols:
                    suggestions.append({
                        "type": "feature_transformation",
                        "description": "特征变换",
                        "columns": skewed_cols,
                        "suggestions": [
                            "对数变换减少偏度",
                            "平方根变换",
                            "Box-Cox变换"
                        ]
                    })
            
            # 特征组合建议
            if len(numeric_cols) >= 2:
                suggestions.append({
                    "type": "feature_combination",
                    "description": "特征组合",
                    "columns": numeric_cols[:5],  # 限制数量
                    "suggestions": [
                        "创建比率特征",
                        "特征相乘/相加",
                        "多项式特征"
                    ]
                })
            
            # 编码建议
            if categorical_cols:
                high_cardinality_cols = [
                    col for col in categorical_cols 
                    if data[col].nunique() > 10
                ]
                
                suggestions.append({
                    "type": "encoding",
                    "description": "分类变量编码",
                    "columns": categorical_cols,
                    "suggestions": [
                        "低基数：独热编码",
                        "高基数：目标编码或嵌入",
                        f"高基数列: {high_cardinality_cols}"
                    ]
                })
            
            # 时间特征建议
            if datetime_cols:
                suggestions.append({
                    "type": "datetime_features",
                    "description": "时间特征提取",
                    "columns": datetime_cols,
                    "suggestions": [
                        "提取年、月、日、星期",
                        "计算时间差",
                        "创建周期性特征（sin/cos）"
                    ]
                })
            
            return {
                "status": "success",
                "data_overview": {
                    "numeric_columns": len(numeric_cols),
                    "categorical_columns": len(categorical_cols),
                    "datetime_columns": len(datetime_cols),
                    "total_features": len(data.columns)
                },
                "suggestions": suggestions,
                "priority_actions": [
                    s["type"] for s in suggestions[:3]  # 前3个优先级最高
                ]
            }
            
        except Exception as e:
            logger.error(f"特征工程建议生成失败: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def get_available_frameworks(self) -> Dict[str, Any]:
        """获取可用框架列表"""
        
        return {
            "eda_frameworks": {
                "sweetviz": ADVANCED_EDA_AVAILABLE,
                "pandas_profiling": ADVANCED_EDA_AVAILABLE,
                "basic_eda": True
            },
            "statistical_frameworks": {
                "statsmodels": STATSMODELS_AVAILABLE,
                "scipy_stats": True,
                "basic_stats": True
            },
            "explainability_frameworks": {
                "shap": EXPLAINABILITY_AVAILABLE,
                "lime": EXPLAINABILITY_AVAILABLE,
                "eli5": EXPLAINABILITY_AVAILABLE
            },
            "capabilities": {
                "advanced_eda": ADVANCED_EDA_AVAILABLE,
                "time_series_analysis": STATSMODELS_AVAILABLE,
                "model_explainability": EXPLAINABILITY_AVAILABLE,
                "statistical_tests": STATSMODELS_AVAILABLE,
                "feature_engineering": True
            }
        }
