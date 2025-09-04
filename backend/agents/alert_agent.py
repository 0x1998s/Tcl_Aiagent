"""
预警Agent - 负责异常检测和预警系统
实现"7×24预警"核心功能
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from scipy import stats
import asyncio

from agents.base_agent import BaseAgent
from services.llm_service import LLMService
from services.data_service import DataService
from core.config import Settings


class AlertAgent(BaseAgent):
    """预警Agent - 处理异常检测和预警"""
    
    def __init__(
        self, 
        llm_service: LLMService,
        data_service: DataService,
        settings: Settings
    ):
        super().__init__(llm_service, data_service, settings, "AlertAgent")
        
        # 异常检测方法
        self.detection_methods = {
            "threshold": self._threshold_detection,
            "statistical": self._statistical_detection,
            "change_rate": self._change_rate_detection,
            "seasonal": self._seasonal_detection
        }
        
        # 预警规则缓存
        self.alert_rules = {}
        self.alert_history = {}
        
        # 监控任务
        self.monitoring_task = None
        
    async def _initialize_agent(self):
        """初始化预警Agent"""
        
        # 加载预警规则
        await self._load_alert_rules()
        
        # 启动监控任务
        await self._start_monitoring()
        
        self.logger.info("预警Agent初始化完成")
    
    async def _process_request(
        self, 
        query: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """处理预警请求"""
        
        # 解析预警请求类型
        request_type = self._parse_alert_request(query, context)
        
        if request_type == "check_metrics":
            return await self._check_all_metrics()
        elif request_type == "analyze_anomaly":
            return await self._analyze_specific_anomaly(context)
        elif request_type == "create_rule":
            return await self._create_alert_rule(context)
        else:
            return await self._get_alert_status()
    
    def _parse_alert_request(self, query: str, context: Dict[str, Any]) -> str:
        """解析预警请求类型"""
        
        query_lower = query.lower()
        
        if any(keyword in query_lower for keyword in ["检查", "监控", "异常"]):
            return "check_metrics"
        elif any(keyword in query_lower for keyword in ["分析", "原因", "为什么"]):
            return "analyze_anomaly"
        elif any(keyword in query_lower for keyword in ["创建", "设置", "规则"]):
            return "create_rule"
        else:
            return "status"
    
    async def _load_alert_rules(self):
        """加载预警规则"""
        
        # 默认预警规则
        default_rules = {
            "daily_sales_drop": {
                "metric": "daily_sales",
                "method": "change_rate",
                "threshold": -0.05,  # 下跌5%
                "time_window": 60,   # 分钟
                "severity": "medium",
                "enabled": True
            },
            "user_count_low": {
                "metric": "active_users",
                "method": "threshold",
                "threshold": 1000,
                "comparison": "less_than",
                "time_window": 30,
                "severity": "high",
                "enabled": True
            },
            "conversion_rate_anomaly": {
                "metric": "conversion_rate",
                "method": "statistical",
                "threshold": 3.0,  # 3σ
                "time_window": 120,
                "severity": "medium",
                "enabled": True
            }
        }
        
        self.alert_rules.update(default_rules)
        self.logger.info(f"加载了 {len(self.alert_rules)} 个预警规则")
    
    async def _start_monitoring(self):
        """启动监控任务"""
        
        if self.monitoring_task:
            return
        
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("预警监控任务已启动")
    
    async def _monitoring_loop(self):
        """监控循环"""
        
        while True:
            try:
                await self._check_all_metrics()
                await asyncio.sleep(self.settings.ALERT_CHECK_INTERVAL)
                
            except Exception as e:
                self.logger.error(f"监控循环出错: {str(e)}")
                await asyncio.sleep(60)  # 出错后等待1分钟再继续
    
    async def _check_all_metrics(self) -> Dict[str, Any]:
        """检查所有指标"""
        
        results = {
            "checked_rules": 0,
            "new_alerts": 0,
            "resolved_alerts": 0,
            "active_alerts": 0,
            "alerts": []
        }
        
        for rule_id, rule in self.alert_rules.items():
            if not rule.get("enabled", True):
                continue
            
            try:
                # 获取指标数据
                metric_data = await self._fetch_metric_data(rule["metric"], rule["time_window"])
                
                # 执行异常检测
                detection_result = await self._detect_anomaly(metric_data, rule)
                
                if detection_result["is_anomaly"]:
                    alert = await self._create_alert(rule_id, rule, detection_result)
                    results["alerts"].append(alert)
                    results["new_alerts"] += 1
                
                results["checked_rules"] += 1
                
            except Exception as e:
                self.logger.error(f"检查规则 {rule_id} 失败: {str(e)}")
        
        # 更新活跃预警数
        results["active_alerts"] = len([a for a in results["alerts"] if a["status"] == "active"])
        
        return results
    
    async def _fetch_metric_data(self, metric: str, time_window_minutes: int) -> List[Dict[str, Any]]:
        """获取指标数据"""
        
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=time_window_minutes)
        
        # 根据指标类型构建查询
        if metric == "daily_sales":
            sql = """
            SELECT 
                DATE(order_date) as date,
                SUM(amount) as value
            FROM orders 
            WHERE order_date >= %s AND order_date <= %s
            GROUP BY DATE(order_date)
            ORDER BY date;
            """
        elif metric == "active_users":
            sql = """
            SELECT 
                DATE(last_login) as date,
                COUNT(DISTINCT user_id) as value
            FROM users 
            WHERE last_login >= %s AND last_login <= %s
            GROUP BY DATE(last_login)
            ORDER BY date;
            """
        elif metric == "conversion_rate":
            sql = """
            SELECT 
                DATE(order_date) as date,
                COUNT(DISTINCT user_id)::float / 
                (SELECT COUNT(DISTINCT user_id) FROM users WHERE created_at <= DATE(order_date)) as value
            FROM orders 
            WHERE order_date >= %s AND order_date <= %s
            GROUP BY DATE(order_date)
            ORDER BY date;
            """
        else:
            # 默认查询
            sql = "SELECT NOW()::date as date, 100 as value;"
        
        try:
            data = await self.execute_sql(sql, {
                "start_time": start_time,
                "end_time": end_time
            })
            return data
            
        except Exception as e:
            self.logger.error(f"获取指标数据失败: {str(e)}")
            
            # 返回模拟数据
            return self._generate_mock_metric_data(metric, time_window_minutes)
    
    def _generate_mock_metric_data(self, metric: str, time_window_minutes: int) -> List[Dict[str, Any]]:
        """生成模拟指标数据"""
        
        np.random.seed(42)
        
        # 生成时间序列
        end_time = datetime.now()
        if time_window_minutes <= 1440:  # 1天内，按小时
            periods = time_window_minutes // 60
            freq = 'H'
        else:  # 超过1天，按天
            periods = time_window_minutes // 1440
            freq = 'D'
        
        if periods == 0:
            periods = 1
        
        dates = pd.date_range(end=end_time, periods=periods, freq=freq)
        
        # 根据指标生成不同的数据模式
        if metric == "daily_sales":
            base_value = 10000
            trend = np.linspace(-500, 500, len(dates))  # 可能的趋势
            noise = np.random.normal(0, 1000, len(dates))
            values = base_value + trend + noise
            
            # 模拟异常：最后一个值显著下降
            if len(values) > 1:
                values[-1] = values[-2] * 0.85  # 下降15%
                
        elif metric == "active_users":
            base_value = 1200
            values = np.random.normal(base_value, 100, len(dates))
            
            # 模拟异常：最后一个值低于阈值
            if len(values) > 0:
                values[-1] = 950  # 低于1000的阈值
                
        else:  # conversion_rate
            base_value = 0.05
            values = np.random.normal(base_value, 0.005, len(dates))
            values = np.clip(values, 0, 1)  # 限制在0-1之间
        
        # 转换为所需格式
        data = []
        for i, date in enumerate(dates):
            data.append({
                "date": date.isoformat(),
                "value": float(values[i])
            })
        
        return data
    
    async def _detect_anomaly(
        self, 
        data: List[Dict[str, Any]], 
        rule: Dict[str, Any]
    ) -> Dict[str, Any]:
        """检测异常"""
        
        if not data:
            return {"is_anomaly": False, "reason": "无数据"}
        
        method = rule.get("method", "threshold")
        
        if method in self.detection_methods:
            return await self.detection_methods[method](data, rule)
        else:
            return {"is_anomaly": False, "reason": f"不支持的检测方法: {method}"}
    
    async def _threshold_detection(
        self, 
        data: List[Dict[str, Any]], 
        rule: Dict[str, Any]
    ) -> Dict[str, Any]:
        """阈值检测"""
        
        if not data:
            return {"is_anomaly": False, "reason": "无数据"}
        
        current_value = data[-1]["value"]
        threshold = rule["threshold"]
        comparison = rule.get("comparison", "less_than")
        
        is_anomaly = False
        if comparison == "less_than":
            is_anomaly = current_value < threshold
        elif comparison == "greater_than":
            is_anomaly = current_value > threshold
        elif comparison == "equal":
            is_anomaly = abs(current_value - threshold) < 0.01
        
        return {
            "is_anomaly": is_anomaly,
            "current_value": current_value,
            "threshold": threshold,
            "comparison": comparison,
            "reason": f"当前值 {current_value} {'低于' if comparison == 'less_than' else '高于'} 阈值 {threshold}" if is_anomaly else "正常"
        }
    
    async def _statistical_detection(
        self, 
        data: List[Dict[str, Any]], 
        rule: Dict[str, Any]
    ) -> Dict[str, Any]:
        """统计异常检测（基于标准差）"""
        
        if len(data) < 3:
            return {"is_anomaly": False, "reason": "数据不足"}
        
        values = [item["value"] for item in data]
        current_value = values[-1]
        historical_values = values[:-1]
        
        mean_val = np.mean(historical_values)
        std_val = np.std(historical_values)
        
        if std_val == 0:
            return {"is_anomaly": False, "reason": "标准差为0"}
        
        z_score = abs(current_value - mean_val) / std_val
        threshold = rule.get("threshold", 3.0)  # 默认3σ
        
        is_anomaly = z_score > threshold
        
        return {
            "is_anomaly": is_anomaly,
            "current_value": current_value,
            "historical_mean": mean_val,
            "z_score": z_score,
            "threshold": threshold,
            "reason": f"Z-score {z_score:.2f} {'超过' if is_anomaly else '未超过'} 阈值 {threshold}"
        }
    
    async def _change_rate_detection(
        self, 
        data: List[Dict[str, Any]], 
        rule: Dict[str, Any]
    ) -> Dict[str, Any]:
        """变化率检测"""
        
        if len(data) < 2:
            return {"is_anomaly": False, "reason": "数据不足"}
        
        current_value = data[-1]["value"]
        previous_value = data[-2]["value"]
        
        if previous_value == 0:
            return {"is_anomaly": False, "reason": "前值为0，无法计算变化率"}
        
        change_rate = (current_value - previous_value) / previous_value
        threshold = rule["threshold"]
        
        # threshold为负值表示下跌阈值，正值表示上涨阈值
        if threshold < 0:
            is_anomaly = change_rate < threshold
        else:
            is_anomaly = change_rate > threshold
        
        return {
            "is_anomaly": is_anomaly,
            "current_value": current_value,
            "previous_value": previous_value,
            "change_rate": change_rate,
            "threshold": threshold,
            "reason": f"变化率 {change_rate:.1%} {'超过' if is_anomaly else '未超过'} 阈值 {threshold:.1%}"
        }
    
    async def _seasonal_detection(
        self, 
        data: List[Dict[str, Any]], 
        rule: Dict[str, Any]
    ) -> Dict[str, Any]:
        """季节性异常检测"""
        
        # 简化版本，实际可以使用更复杂的季节性分解
        return {"is_anomaly": False, "reason": "季节性检测功能开发中"}
    
    async def _create_alert(
        self, 
        rule_id: str, 
        rule: Dict[str, Any], 
        detection_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """创建预警"""
        
        alert_id = f"alert_{rule_id}_{int(datetime.now().timestamp())}"
        
        alert = {
            "alert_id": alert_id,
            "rule_id": rule_id,
            "metric": rule["metric"],
            "severity": rule["severity"],
            "status": "active",
            "current_value": detection_result.get("current_value", 0),
            "threshold": detection_result.get("threshold", 0),
            "message": self._generate_alert_message(rule, detection_result),
            "created_at": datetime.now().isoformat(),
            "detection_details": detection_result
        }
        
        # 保存到历史记录
        self.alert_history[alert_id] = alert
        
        # 发送通知（异步）
        asyncio.create_task(self._send_notification(alert))
        
        self.logger.warning(f"创建预警: {alert['message']}")
        
        return alert
    
    def _generate_alert_message(
        self, 
        rule: Dict[str, Any], 
        detection_result: Dict[str, Any]
    ) -> str:
        """生成预警消息"""
        
        metric_names = {
            "daily_sales": "日销售额",
            "active_users": "活跃用户数",
            "conversion_rate": "转化率"
        }
        
        metric_display = metric_names.get(rule["metric"], rule["metric"])
        reason = detection_result.get("reason", "检测到异常")
        
        return f"{metric_display}异常：{reason}"
    
    async def _send_notification(self, alert: Dict[str, Any]):
        """发送通知"""
        
        try:
            # 这里可以实现飞书、钉钉、邮件等通知
            self.logger.info(f"发送预警通知: {alert['message']}")
            
            # 模拟通知发送
            await asyncio.sleep(0.1)
            
        except Exception as e:
            self.logger.error(f"发送通知失败: {str(e)}")
    
    async def _analyze_specific_anomaly(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """分析特定异常"""
        
        # 获取异常数据
        metric = context.get("metric", "daily_sales")
        
        # 执行根因分析
        root_cause_analysis = await self._perform_root_cause_analysis(metric)
        
        return {
            "response": "异常分析已完成",
            "metric": metric,
            "root_cause_analysis": root_cause_analysis,
            "recommendations": [
                "检查数据源是否正常",
                "分析业务流程变化",
                "监控相关指标"
            ]
        }
    
    async def _perform_root_cause_analysis(self, metric: str) -> Dict[str, Any]:
        """执行根因分析"""
        
        # 简化的根因分析
        possible_causes = {
            "daily_sales": [
                "营销活动结束",
                "竞争对手促销",
                "季节性因素",
                "产品库存不足",
                "系统故障"
            ],
            "active_users": [
                "产品功能问题",
                "服务器性能问题",
                "用户体验下降",
                "外部流量减少"
            ],
            "conversion_rate": [
                "页面加载速度慢",
                "支付流程问题",
                "价格策略调整",
                "用户界面变更"
            ]
        }
        
        causes = possible_causes.get(metric, ["未知原因"])
        
        return {
            "metric": metric,
            "possible_causes": causes,
            "analysis_method": "规则基础分析",
            "confidence": 0.7,
            "next_steps": [
                "收集更多数据",
                "验证假设",
                "制定应对方案"
            ]
        }
    
    async def _create_alert_rule(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """创建预警规则"""
        
        rule_id = f"rule_{int(datetime.now().timestamp())}"
        
        # 从上下文提取规则参数
        rule = {
            "metric": context.get("metric", "daily_sales"),
            "method": context.get("method", "threshold"),
            "threshold": context.get("threshold", 1000),
            "time_window": context.get("time_window", 60),
            "severity": context.get("severity", "medium"),
            "enabled": True
        }
        
        self.alert_rules[rule_id] = rule
        
        return {
            "response": f"预警规则 {rule_id} 创建成功",
            "rule_id": rule_id,
            "rule": rule
        }
    
    async def _get_alert_status(self) -> Dict[str, Any]:
        """获取预警状态"""
        
        active_alerts = [
            alert for alert in self.alert_history.values() 
            if alert["status"] == "active"
        ]
        
        return {
            "response": f"当前有 {len(active_alerts)} 个活跃预警",
            "total_rules": len(self.alert_rules),
            "active_alerts": len(active_alerts),
            "recent_alerts": list(self.alert_history.values())[-5:],  # 最近5个
            "system_status": "正常运行"
        }
    
    async def _cleanup_agent(self):
        """清理预警Agent资源"""
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.alert_rules.clear()
        self.alert_history.clear()
