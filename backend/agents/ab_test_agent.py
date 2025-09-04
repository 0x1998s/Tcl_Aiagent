"""
A/B测试Agent - 负责实验设计、执行和分析
实现"A/B全流程"核心功能
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from scipy import stats
import math

from agents.base_agent import BaseAgent
from services.llm_service import LLMService
from services.data_service import DataService
from core.config import Settings


class ABTestAgent(BaseAgent):
    """A/B测试Agent - 处理实验设计和分析"""
    
    def __init__(
        self, 
        llm_service: LLMService,
        data_service: DataService,
        settings: Settings
    ):
        super().__init__(llm_service, data_service, settings, "ABTestAgent")
        
        # 实验存储
        self.experiments = {}
        
        # 统计方法
        self.statistical_tests = {
            "t_test": self._perform_t_test,
            "chi_square": self._perform_chi_square_test,
            "mann_whitney": self._perform_mann_whitney_test,
            "proportions_test": self._perform_proportions_test
        }
        
        # 默认配置
        self.default_config = {
            "confidence_level": 0.95,
            "statistical_power": 0.8,
            "minimum_effect_size": 0.05,
            "minimum_sample_size": 100
        }
    
    async def _initialize_agent(self):
        """初始化A/B测试Agent"""
        
        # 加载现有实验
        await self._load_experiments()
        
        self.logger.info("A/B测试Agent初始化完成")
    
    async def _process_request(
        self, 
        query: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """处理A/B测试请求"""
        
        # 解析请求类型
        request_type = self._parse_ab_request(query, context)
        
        if request_type == "design_experiment":
            return await self._design_experiment(query, context)
        elif request_type == "analyze_experiment":
            return await self._analyze_experiment(context)
        elif request_type == "calculate_sample_size":
            return await self._calculate_sample_size_request(context)
        elif request_type == "get_experiment_status":
            return await self._get_experiment_status(context)
        else:
            return await self._list_experiments()
    
    def _parse_ab_request(self, query: str, context: Dict[str, Any]) -> str:
        """解析A/B测试请求类型"""
        
        query_lower = query.lower()
        
        if any(keyword in query_lower for keyword in ["设计", "创建", "实验"]):
            return "design_experiment"
        elif any(keyword in query_lower for keyword in ["分析", "结果", "效果"]):
            return "analyze_experiment"
        elif any(keyword in query_lower for keyword in ["样本量", "样本数", "计算"]):
            return "calculate_sample_size"
        elif any(keyword in query_lower for keyword in ["状态", "进度"]):
            return "get_experiment_status"
        else:
            return "list_experiments"
    
    async def _load_experiments(self):
        """加载现有实验"""
        
        # 模拟一些现有实验
        mock_experiments = {
            "exp_001": {
                "name": "首页按钮颜色测试",
                "description": "测试红色vs蓝色按钮对转化率的影响",
                "hypothesis": "红色按钮将提高转化率",
                "status": "running",
                "created_at": "2024-01-10T00:00:00",
                "variants": [
                    {"name": "控制组", "description": "蓝色按钮", "traffic": 0.5},
                    {"name": "测试组", "description": "红色按钮", "traffic": 0.5}
                ],
                "success_metric": "conversion_rate",
                "sample_size": 2000,
                "current_sample_size": 1300,
                "confidence_level": 0.95,
                "statistical_power": 0.8
            },
            "exp_002": {
                "name": "推荐算法优化",
                "description": "测试新推荐算法的效果",
                "hypothesis": "新算法将提高点击率",
                "status": "completed",
                "created_at": "2024-01-01T00:00:00",
                "end_date": "2024-01-08T00:00:00",
                "variants": [
                    {"name": "算法V1", "description": "原有算法", "traffic": 0.5},
                    {"name": "算法V2", "description": "新算法", "traffic": 0.5}
                ],
                "success_metric": "click_through_rate",
                "sample_size": 5000,
                "current_sample_size": 5000,
                "winner": "算法V2"
            }
        }
        
        self.experiments.update(mock_experiments)
        self.logger.info(f"加载了 {len(self.experiments)} 个实验")
    
    async def _design_experiment(
        self, 
        query: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """设计A/B实验"""
        
        # 解析实验设计需求
        experiment_spec = await self._parse_experiment_requirements(query, context)
        
        # 计算样本量
        sample_size_result = await self._calculate_required_sample_size(experiment_spec)
        
        # 生成实验设计
        experiment_design = await self._generate_experiment_design(
            experiment_spec, sample_size_result
        )
        
        # 生成实验ID并保存
        experiment_id = f"exp_{int(datetime.now().timestamp())}"
        experiment_design["experiment_id"] = experiment_id
        experiment_design["status"] = "draft"
        experiment_design["created_at"] = datetime.now().isoformat()
        
        self.experiments[experiment_id] = experiment_design
        
        return {
            "response": f"实验设计完成: {experiment_design['name']}",
            "experiment_id": experiment_id,
            "experiment_design": experiment_design,
            "sample_size_calculation": sample_size_result,
            "next_steps": [
                "审核实验设计",
                "准备实验环境",
                "启动实验",
                "监控实验进展"
            ]
        }
    
    async def _parse_experiment_requirements(
        self, 
        query: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """解析实验需求"""
        
        prompt = f"""
        请分析以下A/B实验设计需求，提取关键信息：

        用户需求：{query}
        上下文：{context}

        请识别以下信息并以JSON格式返回：
        {{
            "name": "实验名称",
            "hypothesis": "实验假设",
            "success_metric": "成功指标",
            "baseline_rate": 0.05,
            "minimum_effect_size": 0.01,
            "variants": [
                {{"name": "控制组", "description": "描述"}},
                {{"name": "测试组", "description": "描述"}}
            ],
            "target_audience": "目标用户群体",
            "duration_estimate": 14
        }}
        """
        
        try:
            response = await self.generate_llm_response(prompt)
            import json
            experiment_spec = json.loads(response)
            
            # 验证和标准化
            experiment_spec = self._validate_experiment_spec(experiment_spec)
            
            return experiment_spec
            
        except Exception as e:
            self.logger.error(f"解析实验需求失败: {str(e)}")
            
            # 返回默认实验规格
            return {
                "name": "A/B测试实验",
                "hypothesis": "测试组将优于控制组",
                "success_metric": "conversion_rate",
                "baseline_rate": 0.05,
                "minimum_effect_size": 0.01,
                "variants": [
                    {"name": "控制组", "description": "当前版本"},
                    {"name": "测试组", "description": "新版本"}
                ],
                "target_audience": "全部用户",
                "duration_estimate": 14
            }
    
    def _validate_experiment_spec(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """验证实验规格"""
        
        # 设置默认值
        spec.setdefault("name", "A/B测试实验")
        spec.setdefault("hypothesis", "测试组将优于控制组")
        spec.setdefault("success_metric", "conversion_rate")
        spec.setdefault("baseline_rate", 0.05)
        spec.setdefault("minimum_effect_size", 0.01)
        spec.setdefault("target_audience", "全部用户")
        spec.setdefault("duration_estimate", 14)
        
        # 验证variants
        if "variants" not in spec or len(spec["variants"]) < 2:
            spec["variants"] = [
                {"name": "控制组", "description": "当前版本"},
                {"name": "测试组", "description": "新版本"}
            ]
        
        # 验证数值范围
        spec["baseline_rate"] = max(0.001, min(0.999, spec["baseline_rate"]))
        spec["minimum_effect_size"] = max(0.001, min(0.5, spec["minimum_effect_size"]))
        
        return spec
    
    async def _calculate_required_sample_size(
        self, 
        experiment_spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        """计算所需样本量"""
        
        baseline_rate = experiment_spec["baseline_rate"]
        minimum_effect_size = experiment_spec["minimum_effect_size"]
        confidence_level = self.default_config["confidence_level"]
        statistical_power = self.default_config["statistical_power"]
        
        # 使用双比例Z检验的样本量计算公式
        try:
            alpha = 1 - confidence_level
            beta = 1 - statistical_power
            
            # Z值
            z_alpha_2 = stats.norm.ppf(1 - alpha / 2)
            z_beta = stats.norm.ppf(1 - beta)
            
            # 效果大小
            p1 = baseline_rate
            p2 = baseline_rate * (1 + minimum_effect_size)
            
            # 合并方差
            p_pooled = (p1 + p2) / 2
            
            # 样本量计算
            numerator = (z_alpha_2 * math.sqrt(2 * p_pooled * (1 - p_pooled)) + 
                        z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
            denominator = (p2 - p1) ** 2
            
            sample_size_per_group = math.ceil(numerator / denominator)
            total_sample_size = sample_size_per_group * len(experiment_spec["variants"])
            
            # 估算持续时间
            estimated_duration = self._estimate_experiment_duration(
                total_sample_size, 
                experiment_spec.get("expected_daily_traffic", 1000)
            )
            
            return {
                "sample_size_per_group": sample_size_per_group,
                "total_sample_size": total_sample_size,
                "estimated_duration_days": estimated_duration,
                "confidence_level": confidence_level,
                "statistical_power": statistical_power,
                "minimum_detectable_effect": minimum_effect_size,
                "baseline_rate": baseline_rate
            }
            
        except Exception as e:
            self.logger.error(f"样本量计算失败: {str(e)}")
            
            # 返回默认值
            return {
                "sample_size_per_group": 1000,
                "total_sample_size": 2000,
                "estimated_duration_days": 14,
                "confidence_level": confidence_level,
                "statistical_power": statistical_power,
                "error": str(e)
            }
    
    def _estimate_experiment_duration(
        self, 
        total_sample_size: int, 
        daily_traffic: int = 1000
    ) -> int:
        """估算实验持续时间"""
        
        if daily_traffic <= 0:
            daily_traffic = 1000  # 默认值
        
        duration_days = math.ceil(total_sample_size / daily_traffic)
        
        # 最少7天，最多90天
        return max(7, min(90, duration_days))
    
    async def _generate_experiment_design(
        self, 
        experiment_spec: Dict[str, Any],
        sample_size_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """生成实验设计"""
        
        # 分配流量
        num_variants = len(experiment_spec["variants"])
        traffic_per_variant = 1.0 / num_variants
        
        for variant in experiment_spec["variants"]:
            variant["traffic"] = traffic_per_variant
        
        experiment_design = {
            "name": experiment_spec["name"],
            "description": f"测试 {experiment_spec['hypothesis']}",
            "hypothesis": experiment_spec["hypothesis"],
            "success_metric": experiment_spec["success_metric"],
            "variants": experiment_spec["variants"],
            "target_audience": experiment_spec["target_audience"],
            "sample_size": sample_size_result["total_sample_size"],
            "estimated_duration": sample_size_result["estimated_duration_days"],
            "confidence_level": sample_size_result["confidence_level"],
            "statistical_power": sample_size_result["statistical_power"],
            "minimum_effect_size": experiment_spec["minimum_effect_size"],
            "baseline_rate": experiment_spec["baseline_rate"]
        }
        
        return experiment_design
    
    async def _analyze_experiment(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """分析实验结果"""
        
        experiment_id = context.get("experiment_id", "exp_001")
        
        if experiment_id not in self.experiments:
            return {
                "response": f"实验 {experiment_id} 不存在",
                "error": "实验不存在"
            }
        
        experiment = self.experiments[experiment_id]
        
        # 获取实验数据
        experiment_data = await self._fetch_experiment_data(experiment_id, experiment)
        
        # 执行统计分析
        statistical_results = await self._perform_statistical_analysis(
            experiment_data, experiment
        )
        
        # 生成结论和建议
        conclusion = await self._generate_experiment_conclusion(
            statistical_results, experiment
        )
        
        return {
            "response": f"实验 {experiment['name']} 分析完成",
            "experiment_id": experiment_id,
            "experiment_name": experiment["name"],
            "statistical_results": statistical_results,
            "conclusion": conclusion,
            "recommendation": conclusion.get("recommendation", "需要更多数据"),
            "next_steps": conclusion.get("next_steps", [])
        }
    
    async def _fetch_experiment_data(
        self, 
        experiment_id: str, 
        experiment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """获取实验数据"""
        
        # 生成模拟实验数据
        np.random.seed(42)
        
        variants_data = {}
        
        for i, variant in enumerate(experiment["variants"]):
            variant_name = variant["name"]
            
            # 模拟样本量
            if experiment.get("current_sample_size"):
                sample_size = experiment["current_sample_size"] // len(experiment["variants"])
            else:
                sample_size = 1000
            
            # 根据实验类型生成不同的数据
            if experiment["success_metric"] == "conversion_rate":
                # 转化率数据
                if "控制" in variant_name or "control" in variant_name.lower():
                    conversion_rate = 0.042  # 基线转化率
                else:
                    conversion_rate = 0.048  # 测试组转化率（提升）
                
                conversions = np.random.binomial(sample_size, conversion_rate)
                
                variants_data[variant_name] = {
                    "sample_size": sample_size,
                    "conversions": conversions,
                    "conversion_rate": conversions / sample_size,
                    "metric_type": "binary"
                }
                
            elif experiment["success_metric"] == "click_through_rate":
                # 点击率数据
                if "V1" in variant_name or "控制" in variant_name:
                    ctr = 0.035
                else:
                    ctr = 0.042
                
                clicks = np.random.binomial(sample_size, ctr)
                
                variants_data[variant_name] = {
                    "sample_size": sample_size,
                    "clicks": clicks,
                    "click_through_rate": clicks / sample_size,
                    "metric_type": "binary"
                }
                
            else:
                # 连续型指标（如收入、时长等）
                if "控制" in variant_name:
                    mean_value = 100
                else:
                    mean_value = 105  # 5%提升
                
                values = np.random.normal(mean_value, 20, sample_size)
                
                variants_data[variant_name] = {
                    "sample_size": sample_size,
                    "values": values.tolist(),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "metric_type": "continuous"
                }
        
        return variants_data
    
    async def _perform_statistical_analysis(
        self, 
        experiment_data: Dict[str, Any], 
        experiment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行统计分析"""
        
        variants = list(experiment_data.keys())
        
        if len(variants) != 2:
            return {"error": "目前只支持两组对比分析"}
        
        control_group = variants[0]  # 假设第一个是控制组
        test_group = variants[1]
        
        control_data = experiment_data[control_group]
        test_data = experiment_data[test_group]
        
        # 根据指标类型选择统计检验方法
        metric_type = control_data.get("metric_type", "binary")
        
        if metric_type == "binary":
            # 比例检验
            test_result = await self._perform_proportions_test(control_data, test_data)
        else:
            # t检验
            test_result = await self._perform_t_test(control_data, test_data)
        
        # 计算效果大小和置信区间
        effect_size = self._calculate_effect_size(control_data, test_data, metric_type)
        confidence_interval = self._calculate_confidence_interval(
            control_data, test_data, metric_type
        )
        
        return {
            "control_group": {
                "name": control_group,
                "data": control_data
            },
            "test_group": {
                "name": test_group,
                "data": test_data
            },
            "statistical_test": test_result,
            "effect_size": effect_size,
            "confidence_interval": confidence_interval,
            "is_significant": test_result.get("p_value", 1.0) < 0.05,
            "confidence_level": experiment.get("confidence_level", 0.95)
        }
    
    async def _perform_proportions_test(
        self, 
        control_data: Dict[str, Any], 
        test_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行比例检验"""
        
        try:
            # 获取数据
            if "conversions" in control_data:
                control_successes = control_data["conversions"]
                test_successes = test_data["conversions"]
            else:
                control_successes = control_data["clicks"]
                test_successes = test_data["clicks"]
            
            control_n = control_data["sample_size"]
            test_n = test_data["sample_size"]
            
            # 计算比例
            control_prop = control_successes / control_n
            test_prop = test_successes / test_n
            
            # Z检验
            pooled_prop = (control_successes + test_successes) / (control_n + test_n)
            se = math.sqrt(pooled_prop * (1 - pooled_prop) * (1/control_n + 1/test_n))
            
            if se == 0:
                return {"error": "标准误差为0，无法进行检验"}
            
            z_stat = (test_prop - control_prop) / se
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            
            return {
                "test_type": "proportions_z_test",
                "z_statistic": z_stat,
                "p_value": p_value,
                "control_proportion": control_prop,
                "test_proportion": test_prop,
                "difference": test_prop - control_prop,
                "relative_lift": (test_prop - control_prop) / control_prop if control_prop > 0 else 0
            }
            
        except Exception as e:
            return {"error": f"比例检验失败: {str(e)}"}
    
    async def _perform_t_test(
        self, 
        control_data: Dict[str, Any], 
        test_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行t检验"""
        
        try:
            control_values = np.array(control_data["values"])
            test_values = np.array(test_data["values"])
            
            # 独立样本t检验
            t_stat, p_value = stats.ttest_ind(test_values, control_values)
            
            control_mean = np.mean(control_values)
            test_mean = np.mean(test_values)
            
            return {
                "test_type": "independent_t_test",
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "control_mean": float(control_mean),
                "test_mean": float(test_mean),
                "difference": float(test_mean - control_mean),
                "relative_lift": float((test_mean - control_mean) / control_mean) if control_mean > 0 else 0
            }
            
        except Exception as e:
            return {"error": f"t检验失败: {str(e)}"}
    
    async def _perform_chi_square_test(
        self, 
        control_data: Dict[str, Any], 
        test_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行卡方检验"""
        
        try:
            # 构造列联表
            control_success = control_data.get("conversions", 0)
            control_total = control_data.get("total", 0)
            control_failure = control_total - control_success
            
            test_success = test_data.get("conversions", 0)
            test_total = test_data.get("total", 0)
            test_failure = test_total - test_success
            
            # 2x2列联表
            observed = np.array([[control_success, control_failure],
                               [test_success, test_failure]])
            
            # 卡方检验
            chi2_stat, p_value, dof, expected = stats.chi2_contingency(observed)
            
            control_rate = control_success / control_total if control_total > 0 else 0
            test_rate = test_success / test_total if test_total > 0 else 0
            
            return {
                "test_type": "chi_square_test",
                "chi2_statistic": float(chi2_stat),
                "p_value": float(p_value),
                "degrees_of_freedom": int(dof),
                "control_rate": float(control_rate),
                "test_rate": float(test_rate),
                "difference": float(test_rate - control_rate),
                "relative_lift": float((test_rate - control_rate) / control_rate) if control_rate > 0 else 0
            }
            
        except Exception as e:
            return {"error": f"卡方检验失败: {str(e)}"}
    
    async def _perform_mann_whitney_test(
        self, 
        control_data: Dict[str, Any], 
        test_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行Mann-Whitney U检验（非参数检验）"""
        
        try:
            control_values = np.array(control_data["values"])
            test_values = np.array(test_data["values"])
            
            # Mann-Whitney U检验
            u_stat, p_value = stats.mannwhitneyu(
                test_values, control_values, 
                alternative='two-sided'
            )
            
            control_median = np.median(control_values)
            test_median = np.median(test_values)
            
            return {
                "test_type": "mann_whitney_u_test",
                "u_statistic": float(u_stat),
                "p_value": float(p_value),
                "control_median": float(control_median),
                "test_median": float(test_median),
                "difference": float(test_median - control_median),
                "relative_lift": float((test_median - control_median) / control_median) if control_median > 0 else 0
            }
            
        except Exception as e:
            return {"error": f"Mann-Whitney检验失败: {str(e)}"}
    
    def _calculate_effect_size(
        self, 
        control_data: Dict[str, Any], 
        test_data: Dict[str, Any], 
        metric_type: str
    ) -> Dict[str, Any]:
        """计算效果大小"""
        
        try:
            if metric_type == "binary":
                if "conversions" in control_data:
                    control_rate = control_data["conversion_rate"]
                    test_rate = test_data["conversion_rate"]
                else:
                    control_rate = control_data["click_through_rate"]
                    test_rate = test_data["click_through_rate"]
                
                # Cohen's h for proportions
                h = 2 * (math.asin(math.sqrt(test_rate)) - math.asin(math.sqrt(control_rate)))
                
                return {
                    "cohens_h": h,
                    "interpretation": self._interpret_effect_size(abs(h), "h")
                }
            else:
                control_values = np.array(control_data["values"])
                test_values = np.array(test_data["values"])
                
                # Cohen's d
                pooled_std = math.sqrt(((len(control_values) - 1) * np.var(control_values, ddof=1) + 
                                      (len(test_values) - 1) * np.var(test_values, ddof=1)) / 
                                     (len(control_values) + len(test_values) - 2))
                
                if pooled_std == 0:
                    return {"error": "合并标准差为0"}
                
                cohens_d = (np.mean(test_values) - np.mean(control_values)) / pooled_std
                
                return {
                    "cohens_d": float(cohens_d),
                    "interpretation": self._interpret_effect_size(abs(cohens_d), "d")
                }
                
        except Exception as e:
            return {"error": f"效果大小计算失败: {str(e)}"}
    
    def _interpret_effect_size(self, effect_size: float, effect_type: str) -> str:
        """解释效果大小"""
        
        if effect_type == "d":  # Cohen's d
            if effect_size < 0.2:
                return "negligible"
            elif effect_size < 0.5:
                return "small"
            elif effect_size < 0.8:
                return "medium"
            else:
                return "large"
        elif effect_type == "h":  # Cohen's h
            if effect_size < 0.2:
                return "negligible"
            elif effect_size < 0.5:
                return "small"
            elif effect_size < 0.8:
                return "medium"
            else:
                return "large"
        else:
            return "unknown"
    
    def _calculate_confidence_interval(
        self, 
        control_data: Dict[str, Any], 
        test_data: Dict[str, Any], 
        metric_type: str,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """计算置信区间"""
        
        try:
            alpha = 1 - confidence_level
            z_critical = stats.norm.ppf(1 - alpha / 2)
            
            if metric_type == "binary":
                if "conversions" in control_data:
                    control_rate = control_data["conversion_rate"]
                    test_rate = test_data["conversion_rate"]
                else:
                    control_rate = control_data["click_through_rate"]
                    test_rate = test_data["click_through_rate"]
                
                control_n = control_data["sample_size"]
                test_n = test_data["sample_size"]
                
                # 差异的标准误差
                se_diff = math.sqrt(
                    (control_rate * (1 - control_rate) / control_n) + 
                    (test_rate * (1 - test_rate) / test_n)
                )
                
                diff = test_rate - control_rate
                margin_of_error = z_critical * se_diff
                
                return {
                    "difference": diff,
                    "lower_bound": diff - margin_of_error,
                    "upper_bound": diff + margin_of_error,
                    "confidence_level": confidence_level
                }
            else:
                control_values = np.array(control_data["values"])
                test_values = np.array(test_data["values"])
                
                # 使用t分布
                pooled_se = math.sqrt(
                    np.var(control_values, ddof=1) / len(control_values) + 
                    np.var(test_values, ddof=1) / len(test_values)
                )
                
                df = len(control_values) + len(test_values) - 2
                t_critical = stats.t.ppf(1 - alpha / 2, df)
                
                diff = np.mean(test_values) - np.mean(control_values)
                margin_of_error = t_critical * pooled_se
                
                return {
                    "difference": float(diff),
                    "lower_bound": float(diff - margin_of_error),
                    "upper_bound": float(diff + margin_of_error),
                    "confidence_level": confidence_level
                }
                
        except Exception as e:
            return {"error": f"置信区间计算失败: {str(e)}"}
    
    async def _generate_experiment_conclusion(
        self, 
        statistical_results: Dict[str, Any], 
        experiment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """生成实验结论"""
        
        is_significant = statistical_results.get("is_significant", False)
        test_result = statistical_results.get("statistical_test", {})
        effect_size = statistical_results.get("effect_size", {})
        
        # 基本结论
        if is_significant:
            if test_result.get("relative_lift", 0) > 0:
                conclusion = "测试组显著优于控制组"
                recommendation = "建议采用测试组方案"
            else:
                conclusion = "控制组显著优于测试组"
                recommendation = "建议保持当前方案"
        else:
            conclusion = "两组之间无显著差异"
            recommendation = "可以选择任一方案，或考虑其他优化方向"
        
        # 详细分析
        detailed_analysis = {
            "statistical_significance": is_significant,
            "p_value": test_result.get("p_value", 1.0),
            "effect_size": effect_size.get("cohens_d") or effect_size.get("cohens_h", 0),
            "effect_interpretation": effect_size.get("interpretation", "unknown"),
            "relative_lift": test_result.get("relative_lift", 0),
            "practical_significance": self._assess_practical_significance(
                test_result, experiment
            )
        }
        
        # 下一步建议
        next_steps = []
        if is_significant:
            if detailed_analysis["practical_significance"]:
                next_steps = [
                    "准备全量上线计划",
                    "监控上线后的关键指标",
                    "记录实验学习和最佳实践"
                ]
            else:
                next_steps = [
                    "评估是否值得实施",
                    "考虑成本效益分析",
                    "寻找更大影响的优化点"
                ]
        else:
            next_steps = [
                "分析无显著差异的可能原因",
                "考虑延长实验时间或增加样本量",
                "探索其他优化方向"
            ]
        
        return {
            "conclusion": conclusion,
            "recommendation": recommendation,
            "detailed_analysis": detailed_analysis,
            "next_steps": next_steps,
            "confidence": "high" if is_significant and detailed_analysis["practical_significance"] else "medium"
        }
    
    def _assess_practical_significance(
        self, 
        test_result: Dict[str, Any], 
        experiment: Dict[str, Any]
    ) -> bool:
        """评估实际意义"""
        
        relative_lift = abs(test_result.get("relative_lift", 0))
        minimum_effect_size = experiment.get("minimum_effect_size", 0.01)
        
        return relative_lift >= minimum_effect_size
    
    async def _calculate_sample_size_request(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """处理样本量计算请求"""
        
        baseline_rate = context.get("baseline_rate", 0.05)
        minimum_effect = context.get("minimum_effect", 0.01)
        confidence_level = context.get("confidence_level", 0.95)
        statistical_power = context.get("statistical_power", 0.8)
        
        # 计算样本量
        sample_size_result = await self._calculate_required_sample_size({
            "baseline_rate": baseline_rate,
            "minimum_effect_size": minimum_effect
        })
        
        return {
            "response": "样本量计算完成",
            "sample_size_calculation": sample_size_result,
            "parameters": {
                "baseline_rate": baseline_rate,
                "minimum_effect": minimum_effect,
                "confidence_level": confidence_level,
                "statistical_power": statistical_power
            }
        }
    
    async def _get_experiment_status(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """获取实验状态"""
        
        experiment_id = context.get("experiment_id")
        
        if experiment_id and experiment_id in self.experiments:
            experiment = self.experiments[experiment_id]
            
            # 计算进度
            current_sample = experiment.get("current_sample_size", 0)
            target_sample = experiment.get("sample_size", 1000)
            progress = min(1.0, current_sample / target_sample) if target_sample > 0 else 0
            
            return {
                "response": f"实验 {experiment['name']} 状态查询",
                "experiment_id": experiment_id,
                "experiment": experiment,
                "progress": progress,
                "status": experiment.get("status", "unknown")
            }
        else:
            return await self._list_experiments()
    
    async def _list_experiments(self) -> Dict[str, Any]:
        """列出所有实验"""
        
        experiments_list = []
        
        for exp_id, exp in self.experiments.items():
            experiments_list.append({
                "experiment_id": exp_id,
                "name": exp.get("name", "未命名实验"),
                "status": exp.get("status", "unknown"),
                "created_at": exp.get("created_at", ""),
                "success_metric": exp.get("success_metric", "")
            })
        
        return {
            "response": f"共有 {len(experiments_list)} 个实验",
            "experiments": experiments_list,
            "total_count": len(experiments_list)
        }
    
    async def _cleanup_agent(self):
        """清理A/B测试Agent资源"""
        self.experiments.clear()
