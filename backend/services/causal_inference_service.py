"""
因果推断服务
实现因果关系发现、因果效应估计和反事实推理
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

try:
    # 尝试导入专业因果推断库
    import causalml
    from causalml.inference.meta import LRSRegressor, XGBTRegressor
    from causalml.inference.tree import UpliftTreeClassifier, UpliftRandomForestClassifier
    CAUSALML_AVAILABLE = True
except ImportError:
    CAUSALML_AVAILABLE = False

try:
    import dowhy
    from dowhy import CausalModel
    DOWHY_AVAILABLE = True
except ImportError:
    DOWHY_AVAILABLE = False

from utils.logger import get_logger
from core.config import Settings

logger = get_logger(__name__)


class CausalInferenceService:
    """因果推断服务"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.models = {}
        self.causal_graphs = {}
        
    async def initialize(self):
        """初始化因果推断服务"""
        logger.info("初始化因果推断服务...")
        
        if CAUSALML_AVAILABLE:
            logger.info("CausalML库可用，支持高级因果推断")
        else:
            logger.warning("CausalML库不可用，使用基础因果推断方法")
            
        if DOWHY_AVAILABLE:
            logger.info("DoWhy库可用，支持因果图分析")
        else:
            logger.warning("DoWhy库不可用，使用简化因果分析")
        
        logger.info("因果推断服务初始化完成")
    
    async def discover_causal_relationships(
        self,
        data: pd.DataFrame,
        target_variable: str,
        potential_causes: List[str],
        method: str = "correlation"
    ) -> Dict[str, Any]:
        """发现因果关系"""
        
        logger.info(f"开始因果关系发现: 目标变量={target_variable}")
        
        try:
            if method == "correlation":
                return await self._correlation_based_discovery(data, target_variable, potential_causes)
            elif method == "granger":
                return await self._granger_causality_test(data, target_variable, potential_causes)
            elif method == "pc_algorithm":
                return await self._pc_algorithm(data, target_variable, potential_causes)
            else:
                return await self._correlation_based_discovery(data, target_variable, potential_causes)
                
        except Exception as e:
            logger.error(f"因果关系发现失败: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "relationships": []
            }
    
    async def _correlation_based_discovery(
        self,
        data: pd.DataFrame,
        target_variable: str,
        potential_causes: List[str]
    ) -> Dict[str, Any]:
        """基于相关性的因果关系发现"""
        
        relationships = []
        
        for cause in potential_causes:
            if cause in data.columns and target_variable in data.columns:
                # 计算相关系数
                correlation = data[cause].corr(data[target_variable])
                
                # 计算显著性
                n = len(data)
                t_stat = correlation * np.sqrt((n - 2) / (1 - correlation**2))
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
                
                # 时间滞后分析
                lag_correlation = await self._analyze_time_lag(data, cause, target_variable)
                
                relationships.append({
                    "cause": cause,
                    "target": target_variable,
                    "correlation": float(correlation),
                    "p_value": float(p_value),
                    "significant": p_value < 0.05,
                    "strength": self._classify_correlation_strength(abs(correlation)),
                    "lag_analysis": lag_correlation
                })
        
        # 按相关性强度排序
        relationships.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        
        return {
            "status": "success",
            "method": "correlation",
            "relationships": relationships,
            "summary": {
                "total_variables": len(potential_causes),
                "significant_relationships": sum(1 for r in relationships if r["significant"]),
                "strong_relationships": sum(1 for r in relationships if r["strength"] in ["strong", "very_strong"])
            }
        }
    
    async def _granger_causality_test(
        self,
        data: pd.DataFrame,
        target_variable: str,
        potential_causes: List[str]
    ) -> Dict[str, Any]:
        """格兰杰因果检验"""
        
        try:
            from statsmodels.tsa.stattools import grangercausalitytests
            
            relationships = []
            
            for cause in potential_causes:
                if cause in data.columns and target_variable in data.columns:
                    # 准备时间序列数据
                    ts_data = data[[target_variable, cause]].dropna()
                    
                    if len(ts_data) > 20:  # 需要足够的数据点
                        try:
                            # 进行格兰杰因果检验
                            result = grangercausalitytests(ts_data, maxlag=5, verbose=False)
                            
                            # 提取最佳滞后期的结果
                            best_lag = min(result.keys())
                            p_value = result[best_lag][0]['ssr_ftest'][1]
                            f_stat = result[best_lag][0]['ssr_ftest'][0]
                            
                            relationships.append({
                                "cause": cause,
                                "target": target_variable,
                                "f_statistic": float(f_stat),
                                "p_value": float(p_value),
                                "significant": p_value < 0.05,
                                "best_lag": best_lag,
                                "causality_strength": "strong" if p_value < 0.01 else "moderate" if p_value < 0.05 else "weak"
                            })
                            
                        except Exception as e:
                            logger.warning(f"格兰杰检验失败 {cause}: {str(e)}")
            
            return {
                "status": "success",
                "method": "granger",
                "relationships": relationships,
                "note": "格兰杰因果检验基于时间序列数据"
            }
            
        except ImportError:
            logger.warning("statsmodels不可用，无法进行格兰杰因果检验")
            return await self._correlation_based_discovery(data, target_variable, potential_causes)
    
    async def _pc_algorithm(
        self,
        data: pd.DataFrame,
        target_variable: str,
        potential_causes: List[str]
    ) -> Dict[str, Any]:
        """PC算法进行因果发现"""
        
        # 简化版本的PC算法实现
        variables = [target_variable] + potential_causes
        available_vars = [v for v in variables if v in data.columns]
        
        if len(available_vars) < 2:
            return {
                "status": "error",
                "error": "可用变量不足，无法进行PC算法分析"
            }
        
        # 计算条件独立性
        relationships = []
        
        for cause in potential_causes:
            if cause in data.columns:
                # 简化的条件独立性测试
                independence_score = await self._test_conditional_independence(
                    data, cause, target_variable, available_vars
                )
                
                relationships.append({
                    "cause": cause,
                    "target": target_variable,
                    "independence_score": independence_score,
                    "likely_causal": independence_score < 0.05
                })
        
        return {
            "status": "success",
            "method": "pc_algorithm",
            "relationships": relationships,
            "note": "简化版本的PC算法"
        }
    
    async def _analyze_time_lag(
        self,
        data: pd.DataFrame,
        cause: str,
        target: str,
        max_lag: int = 5
    ) -> Dict[str, Any]:
        """分析时间滞后效应"""
        
        lag_correlations = {}
        
        for lag in range(1, max_lag + 1):
            if len(data) > lag:
                lagged_cause = data[cause].shift(lag)
                correlation = lagged_cause.corr(data[target])
                
                if not np.isnan(correlation):
                    lag_correlations[lag] = float(correlation)
        
        if lag_correlations:
            best_lag = max(lag_correlations.keys(), key=lambda k: abs(lag_correlations[k]))
            return {
                "lag_correlations": lag_correlations,
                "best_lag": best_lag,
                "best_correlation": lag_correlations[best_lag]
            }
        
        return {"lag_correlations": {}, "best_lag": 0, "best_correlation": 0.0}
    
    async def _test_conditional_independence(
        self,
        data: pd.DataFrame,
        var1: str,
        var2: str,
        conditioning_vars: List[str]
    ) -> float:
        """测试条件独立性"""
        
        # 简化的条件独立性测试
        # 使用偏相关系数
        
        try:
            # 选择conditioning变量
            control_vars = [v for v in conditioning_vars if v not in [var1, var2]][:3]  # 限制控制变量数量
            
            if not control_vars:
                return data[var1].corr(data[var2])
            
            # 计算偏相关系数
            subset = data[[var1, var2] + control_vars].dropna()
            
            if len(subset) < 10:
                return data[var1].corr(data[var2])
            
            # 使用线性回归计算偏相关
            X_control = subset[control_vars]
            
            # var1对控制变量回归的残差
            reg1 = LinearRegression().fit(X_control, subset[var1])
            residuals1 = subset[var1] - reg1.predict(X_control)
            
            # var2对控制变量回归的残差
            reg2 = LinearRegression().fit(X_control, subset[var2])
            residuals2 = subset[var2] - reg2.predict(X_control)
            
            # 残差的相关系数就是偏相关系数
            partial_corr = np.corrcoef(residuals1, residuals2)[0, 1]
            
            return float(partial_corr) if not np.isnan(partial_corr) else 0.0
            
        except Exception as e:
            logger.warning(f"条件独立性测试失败: {str(e)}")
            return data[var1].corr(data[var2])
    
    def _classify_correlation_strength(self, correlation: float) -> str:
        """分类相关性强度"""
        
        abs_corr = abs(correlation)
        
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
    
    async def estimate_treatment_effect(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        covariates: List[str],
        method: str = "propensity_score"
    ) -> Dict[str, Any]:
        """估计处理效应（因果效应）"""
        
        logger.info(f"开始估计处理效应: {treatment_col} -> {outcome_col}")
        
        try:
            if method == "propensity_score":
                return await self._propensity_score_matching(data, treatment_col, outcome_col, covariates)
            elif method == "instrumental_variable":
                return await self._instrumental_variable_estimation(data, treatment_col, outcome_col, covariates)
            elif method == "difference_in_differences":
                return await self._difference_in_differences(data, treatment_col, outcome_col, covariates)
            elif method == "uplift_modeling" and CAUSALML_AVAILABLE:
                return await self._uplift_modeling(data, treatment_col, outcome_col, covariates)
            else:
                return await self._simple_treatment_effect(data, treatment_col, outcome_col, covariates)
                
        except Exception as e:
            logger.error(f"处理效应估计失败: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def _propensity_score_matching(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        covariates: List[str]
    ) -> Dict[str, Any]:
        """倾向性得分匹配"""
        
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import NearestNeighbors
        
        # 准备数据
        available_covariates = [c for c in covariates if c in data.columns]
        subset = data[[treatment_col, outcome_col] + available_covariates].dropna()
        
        if len(subset) < 50:
            return {
                "status": "error",
                "error": "数据量不足，无法进行倾向性得分匹配"
            }
        
        # 估计倾向性得分
        X = subset[available_covariates]
        y = subset[treatment_col]
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        ps_model = LogisticRegression()
        ps_model.fit(X_scaled, y)
        propensity_scores = ps_model.predict_proba(X_scaled)[:, 1]
        
        # 匹配
        treated = subset[subset[treatment_col] == 1]
        control = subset[subset[treatment_col] == 0]
        
        treated_ps = propensity_scores[subset[treatment_col] == 1]
        control_ps = propensity_scores[subset[treatment_col] == 0]
        
        # 使用最近邻匹配
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(control_ps.reshape(-1, 1))
        
        distances, indices = nn.kneighbors(treated_ps.reshape(-1, 1))
        
        # 计算平均处理效应
        treated_outcomes = treated[outcome_col].values
        matched_control_outcomes = control[outcome_col].iloc[indices.flatten()].values
        
        ate = np.mean(treated_outcomes - matched_control_outcomes)
        ate_se = np.std(treated_outcomes - matched_control_outcomes) / np.sqrt(len(treated_outcomes))
        
        return {
            "status": "success",
            "method": "propensity_score_matching",
            "average_treatment_effect": float(ate),
            "standard_error": float(ate_se),
            "confidence_interval": [
                float(ate - 1.96 * ate_se),
                float(ate + 1.96 * ate_se)
            ],
            "sample_size": {
                "treated": len(treated),
                "control": len(control),
                "matched": len(treated_outcomes)
            },
            "propensity_score_stats": {
                "treated_mean": float(np.mean(treated_ps)),
                "control_mean": float(np.mean(control_ps)),
                "overlap": float(np.sum((treated_ps >= np.min(control_ps)) & 
                                      (treated_ps <= np.max(control_ps))) / len(treated_ps))
            }
        }
    
    async def _uplift_modeling(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        covariates: List[str]
    ) -> Dict[str, Any]:
        """Uplift建模"""
        
        if not CAUSALML_AVAILABLE:
            return await self._simple_treatment_effect(data, treatment_col, outcome_col, covariates)
        
        # 准备数据
        available_covariates = [c for c in covariates if c in data.columns]
        subset = data[[treatment_col, outcome_col] + available_covariates].dropna()
        
        X = subset[available_covariates]
        y = subset[outcome_col]
        treatment = subset[treatment_col]
        
        # 使用CausalML的S-Learner
        s_learner = LRSRegressor()
        s_learner.fit(X=X, treatment=treatment, y=y)
        
        # 预测uplift
        uplift = s_learner.predict(X=X)
        
        # 计算平均uplift效应
        average_uplift = np.mean(uplift)
        uplift_std = np.std(uplift)
        
        return {
            "status": "success",
            "method": "uplift_modeling",
            "average_uplift_effect": float(average_uplift),
            "uplift_standard_deviation": float(uplift_std),
            "uplift_distribution": {
                "min": float(np.min(uplift)),
                "max": float(np.max(uplift)),
                "median": float(np.median(uplift)),
                "q25": float(np.percentile(uplift, 25)),
                "q75": float(np.percentile(uplift, 75))
            },
            "sample_size": len(subset)
        }
    
    async def _simple_treatment_effect(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        covariates: List[str]
    ) -> Dict[str, Any]:
        """简单处理效应估计"""
        
        subset = data[[treatment_col, outcome_col]].dropna()
        
        treated = subset[subset[treatment_col] == 1][outcome_col]
        control = subset[subset[treatment_col] == 0][outcome_col]
        
        if len(treated) == 0 or len(control) == 0:
            return {
                "status": "error",
                "error": "处理组或对照组为空"
            }
        
        # 计算平均处理效应
        ate = treated.mean() - control.mean()
        
        # t检验
        t_stat, p_value = stats.ttest_ind(treated, control)
        
        return {
            "status": "success",
            "method": "simple_difference",
            "average_treatment_effect": float(ate),
            "treated_mean": float(treated.mean()),
            "control_mean": float(control.mean()),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
            "sample_size": {
                "treated": len(treated),
                "control": len(control)
            }
        }
    
    async def _difference_in_differences(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        covariates: List[str]
    ) -> Dict[str, Any]:
        """双重差分估计"""
        
        # 简化版本的双重差分
        # 假设数据包含时间维度
        
        if "time_period" not in data.columns:
            return {
                "status": "error",
                "error": "双重差分需要时间维度数据"
            }
        
        # 创建交互项
        data_copy = data.copy()
        data_copy['treatment_time'] = data_copy[treatment_col] * data_copy['time_period']
        
        # 回归分析
        from sklearn.linear_model import LinearRegression
        
        X = data_copy[[treatment_col, 'time_period', 'treatment_time']].values
        y = data_copy[outcome_col].values
        
        reg = LinearRegression()
        reg.fit(X, y)
        
        # 双重差分系数就是处理效应
        did_effect = reg.coef_[2]  # treatment_time的系数
        
        return {
            "status": "success",
            "method": "difference_in_differences",
            "did_effect": float(did_effect),
            "coefficients": {
                "treatment": float(reg.coef_[0]),
                "time": float(reg.coef_[1]),
                "did": float(reg.coef_[2])
            },
            "intercept": float(reg.intercept_),
            "r_squared": float(reg.score(X, y))
        }
    
    async def _instrumental_variable_estimation(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        covariates: List[str]
    ) -> Dict[str, Any]:
        """工具变量估计"""
        
        # 简化的工具变量估计
        # 假设第一个协变量是工具变量
        
        if not covariates:
            return {
                "status": "error",
                "error": "工具变量估计需要至少一个工具变量"
            }
        
        instrument = covariates[0]
        
        if instrument not in data.columns:
            return {
                "status": "error",
                "error": f"工具变量 {instrument} 不存在"
            }
        
        subset = data[[treatment_col, outcome_col, instrument]].dropna()
        
        # 两阶段最小二乘法
        from sklearn.linear_model import LinearRegression
        
        # 第一阶段：工具变量对处理变量回归
        stage1 = LinearRegression()
        stage1.fit(subset[[instrument]], subset[treatment_col])
        treatment_fitted = stage1.predict(subset[[instrument]])
        
        # 第二阶段：结果变量对拟合的处理变量回归
        stage2 = LinearRegression()
        stage2.fit(treatment_fitted.reshape(-1, 1), subset[outcome_col])
        
        iv_effect = stage2.coef_[0]
        
        return {
            "status": "success",
            "method": "instrumental_variable",
            "iv_effect": float(iv_effect),
            "instrument": instrument,
            "first_stage_r_squared": float(stage1.score(subset[[instrument]], subset[treatment_col])),
            "second_stage_r_squared": float(stage2.score(treatment_fitted.reshape(-1, 1), subset[outcome_col])),
            "sample_size": len(subset)
        }
    
    async def counterfactual_analysis(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        covariates: List[str],
        scenario: Dict[str, Any]
    ) -> Dict[str, Any]:
        """反事实分析"""
        
        logger.info("开始反事实分析...")
        
        try:
            # 构建预测模型
            available_covariates = [c for c in covariates if c in data.columns]
            features = [treatment_col] + available_covariates
            
            subset = data[features + [outcome_col]].dropna()
            
            X = subset[features]
            y = subset[outcome_col]
            
            # 使用随机森林进行预测
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            # 生成反事实场景
            counterfactual_data = X.copy()
            
            # 应用场景变化
            for feature, new_value in scenario.items():
                if feature in counterfactual_data.columns:
                    counterfactual_data[feature] = new_value
            
            # 预测反事实结果
            counterfactual_outcomes = model.predict(counterfactual_data)
            original_outcomes = model.predict(X)
            
            # 计算平均处理效应
            average_effect = np.mean(counterfactual_outcomes - original_outcomes)
            
            return {
                "status": "success",
                "scenario": scenario,
                "average_counterfactual_effect": float(average_effect),
                "original_mean_outcome": float(np.mean(original_outcomes)),
                "counterfactual_mean_outcome": float(np.mean(counterfactual_outcomes)),
                "effect_distribution": {
                    "min": float(np.min(counterfactual_outcomes - original_outcomes)),
                    "max": float(np.max(counterfactual_outcomes - original_outcomes)),
                    "std": float(np.std(counterfactual_outcomes - original_outcomes)),
                    "median": float(np.median(counterfactual_outcomes - original_outcomes))
                },
                "feature_importance": dict(zip(features, model.feature_importances_)),
                "sample_size": len(subset)
            }
            
        except Exception as e:
            logger.error(f"反事实分析失败: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def build_causal_graph(
        self,
        variables: List[str],
        relationships: List[Dict[str, Any]],
        graph_name: str = "default"
    ) -> Dict[str, Any]:
        """构建因果图"""
        
        logger.info(f"构建因果图: {graph_name}")
        
        # 简化的因果图表示
        causal_graph = {
            "nodes": variables,
            "edges": [],
            "adjacency_matrix": np.zeros((len(variables), len(variables)))
        }
        
        var_to_idx = {var: idx for idx, var in enumerate(variables)}
        
        for rel in relationships:
            if rel.get("significant", False) or rel.get("likely_causal", False):
                cause = rel.get("cause")
                target = rel.get("target")
                
                if cause in var_to_idx and target in var_to_idx:
                    cause_idx = var_to_idx[cause]
                    target_idx = var_to_idx[target]
                    
                    causal_graph["edges"].append({
                        "from": cause,
                        "to": target,
                        "strength": rel.get("correlation", rel.get("independence_score", 0)),
                        "type": "causal"
                    })
                    
                    causal_graph["adjacency_matrix"][cause_idx][target_idx] = 1
        
        # 保存因果图
        self.causal_graphs[graph_name] = causal_graph
        
        return {
            "status": "success",
            "graph_name": graph_name,
            "nodes_count": len(variables),
            "edges_count": len(causal_graph["edges"]),
            "graph": causal_graph
        }
    
    async def get_causal_insights(
        self,
        data: pd.DataFrame,
        target_variable: str,
        analysis_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """获取因果洞察"""
        
        logger.info(f"生成因果洞察: {target_variable}")
        
        try:
            insights = {
                "target_variable": target_variable,
                "analysis_timestamp": datetime.now().isoformat(),
                "insights": []
            }
            
            # 获取所有可能的因变量
            potential_causes = [col for col in data.columns if col != target_variable]
            
            if analysis_type == "comprehensive":
                # 发现因果关系
                relationships = await self.discover_causal_relationships(
                    data, target_variable, potential_causes
                )
                
                # 分析强因果关系
                strong_causes = [
                    r for r in relationships.get("relationships", [])
                    if r.get("significant", False) and abs(r.get("correlation", 0)) > 0.3
                ]
                
                if strong_causes:
                    insights["insights"].append({
                        "type": "strong_predictors",
                        "message": f"发现 {len(strong_causes)} 个强预测因子",
                        "details": strong_causes[:5]  # 只显示前5个
                    })
                
                # 时间滞后分析
                lag_insights = []
                for cause in potential_causes[:5]:  # 限制分析数量
                    lag_analysis = await self._analyze_time_lag(data, cause, target_variable)
                    if lag_analysis.get("best_correlation", 0) > 0.3:
                        lag_insights.append({
                            "variable": cause,
                            "best_lag": lag_analysis["best_lag"],
                            "correlation": lag_analysis["best_correlation"]
                        })
                
                if lag_insights:
                    insights["insights"].append({
                        "type": "temporal_patterns",
                        "message": "发现时间滞后效应",
                        "details": lag_insights
                    })
            
            return insights
            
        except Exception as e:
            logger.error(f"生成因果洞察失败: {str(e)}")
            return {
                "target_variable": target_variable,
                "error": str(e),
                "insights": []
            }
    
    async def sensitivity_analysis(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        confounders: List[str],
        gamma_range: Tuple[float, float] = (1.0, 2.0),
        steps: int = 10
    ) -> Dict[str, Any]:
        """敏感性分析 - 评估未观测混淆变量的影响"""
        
        try:
            logger.info("开始敏感性分析...")
            
            # 基础处理效应估计
            base_effect = await self._simple_treatment_effect(
                data, treatment_col, outcome_col, confounders
            )
            
            # 不同Gamma值下的效应估计
            gamma_values = np.linspace(gamma_range[0], gamma_range[1], steps)
            sensitivity_results = []
            
            for gamma in gamma_values:
                # 模拟未观测混淆变量的影响
                # Rosenbaum bounds方法的简化版本
                
                # 计算调整后的处理效应
                adjusted_effect = base_effect.get("average_treatment_effect", 0) * (1 / gamma)
                
                # 计算置信区间
                se = base_effect.get("average_treatment_effect", 0) * 0.1  # 简化的标准误
                ci_lower = adjusted_effect - 1.96 * se
                ci_upper = adjusted_effect + 1.96 * se
                
                sensitivity_results.append({
                    "gamma": float(gamma),
                    "adjusted_effect": float(adjusted_effect),
                    "ci_lower": float(ci_lower),
                    "ci_upper": float(ci_upper),
                    "significant": ci_lower > 0 or ci_upper < 0
                })
            
            # 找到临界Gamma值
            critical_gamma = None
            for result in sensitivity_results:
                if not result["significant"]:
                    critical_gamma = result["gamma"]
                    break
            
            return {
                "status": "success",
                "base_effect": base_effect,
                "sensitivity_results": sensitivity_results,
                "critical_gamma": critical_gamma,
                "interpretation": self._interpret_sensitivity_analysis(critical_gamma)
            }
            
        except Exception as e:
            logger.error(f"敏感性分析失败: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def _interpret_sensitivity_analysis(self, critical_gamma: Optional[float]) -> str:
        """解释敏感性分析结果"""
        
        if critical_gamma is None:
            return "处理效应在所有测试的Gamma值下都保持显著，结果较为稳健"
        elif critical_gamma <= 1.2:
            return "结果对未观测混淆变量非常敏感，需要谨慎解释"
        elif critical_gamma <= 1.5:
            return "结果对未观测混淆变量中度敏感，建议进一步验证"
        else:
            return "结果对未观测混淆变量相对稳健"
    
    async def advanced_causal_discovery(
        self,
        data: pd.DataFrame,
        algorithm: str = "pc",
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """高级因果发现算法"""
        
        try:
            if algorithm == "pc":
                return await self._pc_algorithm_advanced(data, alpha)
            elif algorithm == "lingam":
                return await self._lingam_algorithm(data)
            elif algorithm == "ges":
                return await self._ges_algorithm(data)
            else:
                return {"status": "error", "error": f"不支持的算法: {algorithm}"}
                
        except Exception as e:
            logger.error(f"高级因果发现失败: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def _pc_algorithm_advanced(
        self,
        data: pd.DataFrame,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """高级PC算法实现"""
        
        variables = list(data.columns)
        n_vars = len(variables)
        
        # 初始化完全图
        adjacency_matrix = np.ones((n_vars, n_vars)) - np.eye(n_vars)
        
        # Phase 1: 边删除
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if adjacency_matrix[i, j] == 1:
                    # 测试条件独立性
                    for conditioning_set_size in range(min(n_vars - 2, 5)):  # 限制条件集大小
                        # 选择条件变量
                        other_vars = [k for k in range(n_vars) if k != i and k != j]
                        
                        if len(other_vars) >= conditioning_set_size:
                            from itertools import combinations
                            for conditioning_set in combinations(other_vars, conditioning_set_size):
                                # 条件独立性测试
                                p_value = await self._conditional_independence_test(
                                    data, variables[i], variables[j], 
                                    [variables[k] for k in conditioning_set]
                                )
                                
                                if p_value > alpha:
                                    # 条件独立，删除边
                                    adjacency_matrix[i, j] = 0
                                    adjacency_matrix[j, i] = 0
                                    break
                        
                        if adjacency_matrix[i, j] == 0:
                            break
        
        # Phase 2: 边定向（简化版本）
        directed_edges = []
        for i in range(n_vars):
            for j in range(n_vars):
                if adjacency_matrix[i, j] == 1:
                    # 简化的定向规则
                    if self._should_direct_edge(i, j, adjacency_matrix):
                        directed_edges.append((variables[i], variables[j]))
        
        return {
            "status": "success",
            "algorithm": "PC",
            "variables": variables,
            "adjacency_matrix": adjacency_matrix.tolist(),
            "directed_edges": directed_edges,
            "undirected_edges": [(variables[i], variables[j]) 
                                for i in range(n_vars) for j in range(i+1, n_vars)
                                if adjacency_matrix[i, j] == 1 and 
                                (variables[i], variables[j]) not in directed_edges and
                                (variables[j], variables[i]) not in directed_edges]
        }
    
    async def _conditional_independence_test(
        self,
        data: pd.DataFrame,
        var1: str,
        var2: str,
        conditioning_vars: List[str]
    ) -> float:
        """条件独立性测试"""
        
        try:
            if not conditioning_vars:
                # 无条件相关性测试
                corr, p_value = stats.pearsonr(data[var1], data[var2])
                return p_value
            
            # 偏相关测试
            subset_data = data[[var1, var2] + conditioning_vars].dropna()
            
            if len(subset_data) < 30:
                return 0.5  # 样本量不足，假设独立
            
            # 使用线性回归计算偏相关
            X_control = subset_data[conditioning_vars]
            
            # var1对控制变量的回归残差
            from sklearn.linear_model import LinearRegression
            reg1 = LinearRegression().fit(X_control, subset_data[var1])
            residuals1 = subset_data[var1] - reg1.predict(X_control)
            
            # var2对控制变量的回归残差
            reg2 = LinearRegression().fit(X_control, subset_data[var2])
            residuals2 = subset_data[var2] - reg2.predict(X_control)
            
            # 残差的相关性测试
            if len(residuals1) > 3:
                corr, p_value = stats.pearsonr(residuals1, residuals2)
                return p_value
            else:
                return 0.5
                
        except Exception as e:
            logger.warning(f"条件独立性测试失败: {str(e)}")
            return 0.5
    
    def _should_direct_edge(self, i: int, j: int, adj_matrix: np.ndarray) -> bool:
        """简化的边定向规则"""
        
        # 基于度数的简单启发式规则
        degree_i = np.sum(adj_matrix[i, :])
        degree_j = np.sum(adj_matrix[j, :])
        
        # 度数高的变量更可能是原因
        return degree_i > degree_j
    
    async def _lingam_algorithm(self, data: pd.DataFrame) -> Dict[str, Any]:
        """LiNGAM算法（线性非高斯无环模型）"""
        
        # 这是一个简化的LiNGAM实现
        # 实际应该使用专业的LiNGAM库
        
        variables = list(data.columns)
        n_vars = len(variables)
        
        # 独立成分分析
        from sklearn.decomposition import FastICA
        
        try:
            ica = FastICA(n_components=n_vars, random_state=42)
            S = ica.fit_transform(data.values)
            W = ica.mixing_
            
            # 基于混合矩阵估计因果顺序
            causal_order = []
            remaining_vars = list(range(n_vars))
            
            while remaining_vars:
                # 找到最外生的变量（简化启发式）
                min_sum_idx = min(remaining_vars, key=lambda i: np.sum(np.abs(W[i, :])))
                causal_order.append(variables[min_sum_idx])
                remaining_vars.remove(min_sum_idx)
            
            return {
                "status": "success",
                "algorithm": "LiNGAM",
                "causal_order": causal_order,
                "mixing_matrix": W.tolist()
            }
            
        except Exception as e:
            return {"status": "error", "error": f"LiNGAM算法失败: {str(e)}"}
    
    async def _ges_algorithm(self, data: pd.DataFrame) -> Dict[str, Any]:
        """GES算法（贪婪等价搜索）"""
        
        # GES算法的简化实现
        variables = list(data.columns)
        
        return {
            "status": "success",
            "algorithm": "GES",
            "message": "GES算法需要专业库支持，这里提供框架",
            "variables": variables
        }
