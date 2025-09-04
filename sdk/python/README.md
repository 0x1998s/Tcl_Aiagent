# TCL AI Agent Python SDK

企业级数据分析智能助手的Python客户端SDK，提供便捷的API接口访问TCL AI Agent的所有功能。

## 🚀 快速开始

### 安装

```bash
pip install tcl-ai-agent-sdk
```

### 基本使用

```python
import asyncio
from tcl_ai_agent_sdk import TCLAIAgentSDK, AnalysisRequest

async def main():
    async with TCLAIAgentSDK(base_url="http://localhost:8000") as client:
        # 聊天对话
        response = await client.chat("分析一下用户增长趋势")
        print(response.response)
        
        # 自然语言查询
        result = await client.natural_language_query("过去30天的订单总数")
        print(result)

asyncio.run(main())
```

### 同步使用

```python
from tcl_ai_agent_sdk import TCLAIAgentClient

with TCLAIAgentClient() as client:
    # 快速洞察
    insight = client.quick_insight("今天的核心数据如何？")
    print(insight)
    
    # 健康检查
    health = client.health_check()
    print(health)
```

## 📖 功能特性

### 🤖 智能对话
- **自然语言交互**: 用自然语言描述分析需求
- **流式响应**: 支持实时流式对话
- **上下文记忆**: 保持会话上下文

```python
async with TCLAIAgentSDK() as client:
    # 普通对话
    response = await client.chat("帮我分析销售数据")
    
    # 流式对话
    async for chunk in client.stream_chat("详细解释一下趋势变化"):
        print(chunk, end="")
```

### 📊 数据分析
- **趋势分析**: 时间序列趋势分析
- **相关性分析**: 指标间相关性分析
- **细分对比**: 不同维度的对比分析

```python
# 趋势分析
trend = await client.trend_analysis(
    metrics=["revenue", "orders", "users"],
    time_range={"start": "2024-01-01", "end": "2024-01-31"},
    dimensions=["channel", "region"]
)

# 相关性分析
correlation = await client.correlation_analysis(
    metrics=["ad_spend", "revenue", "conversion_rate"],
    filters={"channel": "online"}
)

# 细分对比
comparison = await client.compare_segments(
    metric="conversion_rate",
    segment_dimension="age_group",
    segments=["18-25", "26-35", "36-45"],
    time_range={"start": "2024-01-01", "end": "2024-01-31"}
)
```

### 🔍 智能查询
- **SQL查询**: 直接执行SQL查询
- **自然语言查询**: 将自然语言转换为SQL
- **查询缓存**: 自动缓存提升性能

```python
# SQL查询
sql_result = await client.query(
    sql="SELECT COUNT(*) FROM orders WHERE date >= '2024-01-01'",
    use_cache=True
)

# 自然语言查询
nl_result = await client.natural_language_query(
    question="过去7天每天的订单数量和总金额",
    use_cache=True
)
```

### 🧪 A/B实验
- **实验设计**: 创建和管理A/B测试
- **统计分析**: 自动进行统计显著性检验
- **结果监控**: 实时监控实验结果

```python
from tcl_ai_agent_sdk import ExperimentRequest

# 创建A/B实验
experiment = await client.create_experiment(
    ExperimentRequest(
        experiment_name="新按钮颜色测试",
        control_group={"button_color": "blue"},
        test_group={"button_color": "red"},
        metric="click_through_rate",
        hypothesis="红色按钮能提高点击率",
        confidence_level=0.95
    )
)

# 获取实验结果
results = await client.get_experiment_results(experiment["experiment_id"])
```

### 🚨 智能预警
- **指标监控**: 监控关键业务指标
- **异常检测**: 自动检测数据异常
- **多渠道通知**: 支持飞书、钉钉、邮件通知

```python
# 创建预警规则
alert = await client.create_alert(
    alert_name="收入下降预警",
    metric="daily_revenue",
    condition="decrease",
    threshold=0.1,  # 下降10%
    notification_channels=["feishu", "email"]
)

# 获取预警列表
alerts = await client.get_alerts(status="active")
```

### 📄 自动报告
- **PPT生成**: 自动生成数据分析PPT
- **多种模板**: 商业报告、数据分析等模板
- **定制化**: 支持自定义报告内容

```python
# 生成业务报告
report = await client.generate_report(
    report_type="business_summary",
    time_range={"start": "2024-01-01", "end": "2024-01-31"},
    metrics=["revenue", "users", "orders"],
    format="ppt"
)

# 获取报告列表
reports = await client.get_reports()
```

### 📈 KPI仪表板
- **实时数据**: 实时KPI监控
- **可视化**: 自动生成图表
- **自定义**: 灵活配置KPI指标

```python
# 获取KPI仪表板
dashboard = await client.get_kpi_dashboard(
    kpis=["revenue", "orders", "users", "conversion_rate"],
    time_range={"start": "2024-01-01", "end": "2024-01-31"}
)
```

## 🔧 高级配置

### 认证配置

```python
# 使用API密钥
client = TCLAIAgentSDK(
    base_url="https://api.tcl-ai-agent.com",
    api_key="your-api-key-here",
    timeout=60
)
```

### 错误处理

```python
import httpx

try:
    response = await client.chat("分析数据")
except httpx.HTTPStatusError as e:
    print(f"HTTP错误: {e.response.status_code}")
except httpx.RequestError as e:
    print(f"请求错误: {e}")
except Exception as e:
    print(f"未知错误: {e}")
```

### 自定义配置

```python
# 自定义HTTP客户端配置
client = TCLAIAgentSDK(
    base_url="http://localhost:8000",
    timeout=120,  # 2分钟超时
)

# 设置会话ID
client.session_id = "custom-session-id"
```

## 🔌 集成示例

### Jupyter Notebook集成

```python
# 在Jupyter中使用
%pip install tcl-ai-agent-sdk[jupyter]

from tcl_ai_agent_sdk import TCLAIAgentClient
import pandas as pd

# 创建同步客户端（适合Jupyter）
client = TCLAIAgentClient()

# 获取数据并转换为DataFrame
result = client.natural_language_query("过去30天的用户数据")
df = pd.DataFrame(result["data"])
df.head()
```

### Pandas集成

```python
# 查询数据并直接转换为DataFrame
result = await client.query("SELECT * FROM user_metrics LIMIT 100")
df = pd.DataFrame(result["data"])

# 使用DataFrame进行进一步分析
summary = df.describe()
```

### 定时任务集成

```python
import schedule
import time

def daily_report():
    with TCLAIAgentClient() as client:
        report = client.generate_report(
            report_type="daily_summary",
            format="ppt"
        )
        print(f"日报生成完成: {report['file_path']}")

# 每天上午9点生成报告
schedule.every().day.at("09:00").do(daily_report)

while True:
    schedule.run_pending()
    time.sleep(60)
```

## 📚 API参考

### 主要类

- **TCLAIAgentSDK**: 异步客户端主类
- **TCLAIAgentClient**: 同步客户端类
- **AgentResponse**: 聊天响应数据类
- **AnalysisRequest**: 分析请求数据类
- **ExperimentRequest**: 实验请求数据类

### 主要方法

#### 聊天相关
- `chat(message, user_id, context)`: 发送聊天消息
- `stream_chat(message, user_id, context)`: 流式聊天
- `quick_insight(question)`: 快速洞察

#### 分析相关
- `analyze(request)`: 执行分析
- `trend_analysis(metrics, time_range, dimensions)`: 趋势分析
- `correlation_analysis(metrics, filters)`: 相关性分析
- `compare_segments(metric, segment_dimension, segments)`: 细分对比

#### 查询相关
- `query(sql, use_cache)`: SQL查询
- `natural_language_query(question, use_cache)`: 自然语言查询

#### 实验相关
- `create_experiment(request)`: 创建实验
- `get_experiment_results(experiment_id)`: 获取实验结果
- `list_experiments(status)`: 列出实验

#### 预警相关
- `create_alert(alert_name, metric, condition, threshold)`: 创建预警
- `get_alerts(status)`: 获取预警列表

#### 报告相关
- `generate_report(report_type, time_range, metrics, format)`: 生成报告
- `get_reports()`: 获取报告列表

#### 系统相关
- `health_check()`: 健康检查
- `get_system_stats()`: 获取系统统计

## 🛠️ 开发指南

### 环境设置

```bash
git clone https://github.com/0x1998s/Tcl_Aiagent.git
cd ai-agent-sdk/sdk/python
pip install -e .[dev]
```

### 运行测试

```bash
pytest tests/
```

### 代码格式化

```bash
black tcl_ai_agent_sdk/
flake8 tcl_ai_agent_sdk/
```

### 类型检查

```bash
mypy tcl_ai_agent_sdk/
```

## 📄 许可证

MIT License

