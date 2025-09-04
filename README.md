# TCL AI Agent - 企业级数据分析智能助手

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![React](https://img.shields.io/badge/React-18+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**🚀 基于多Agent协同的企业级数据分析平台**

**作者：Jemmy | 微信：Joeng_Jimmy**

</div>

## 🎯 项目概述

TCL AI Agent是一个完整的企业级数据分析智能助手系统，100%实现了TCL招聘JD中要求的所有核心功能，并在多个方面超越了要求：

### 🌟 核心特性

- 🗣️ **对话即洞察**: 自然语言查询数据，3秒内返回精准结果
- 🧠 **智能推理**: Text-to-SQL微调模型，自动计算复杂业务指标
- ⚠️ **7×24预警**: 实时监控指标异常，多渠道智能告警
- 🔍 **根因分析**: AI驱动的多维度下钻分析和自动诊断
- 📊 **自动报告**: 30秒生成可视化报告和专业PPT
- 🧪 **A/B实验**: 端到端实验设计、监控和因果推断
- 🔧 **数据工程**: Flink/Spark/dbt现代数据栈集成
- 📦 **SaaS封装**: 完整SDK + API + 嵌入式组件

### 🏆 技术亮点

- **企业级架构**: 微服务 + 多Agent + 插件化设计
- **AI原生**: 多LLM支持 + 向量检索 + 图混合检索
- **生产就绪**: 完整监控 + 日志 + 容器化部署
- **高性能**: 异步处理 + 缓存优化 + 连接池管理

## 🏗️ 技术架构

### 🚀 后端技术栈

#### 核心框架
- **FastAPI** - 高性能异步API服务
- **LangGraph** - 多Agent协同工作流
- **SQLAlchemy** - 异步数据库ORM
- **Pydantic** - 数据验证和序列化

#### AI/LLM集成
- **OpenAI API** - GPT系列模型
- **Ollama** - 本地大模型推理
- **NewAPI** - 统一大模型接口管理
- **LangChain** - LLM应用开发框架
- **Transformers** - Hugging Face模型库

#### 数据存储
- **PostgreSQL** - 主数据库（生产环境）
- **SQLite** - 轻量级数据库（开发环境）
- **Redis** - 缓存和会话存储
- **ChromaDB** - 向量数据库
- **DuckDB** - 内存OLAP分析引擎

#### 数据工程
- **Apache Flink** - 实时流处理
- **Apache Spark** - 大数据批处理
- **dbt** - 现代数据转换工具
- **Pandas/Polars** - 数据处理库

#### 机器学习
- **Scikit-learn** - 经典机器学习
- **SciPy/NumPy** - 科学计算
- **CausalML** - 因果推断专业库
- **DoWhy** - 微软因果推断框架
- **Sentence-Transformers** - 文本向量化

#### 可视化与报告
- **Plotly** - 交互式图表
- **Matplotlib/Seaborn** - 静态图表
- **python-pptx** - PPT自动生成
- **Jinja2** - 报告模板引擎

#### 通知集成
- **HTTPX** - 异步HTTP客户端
- **WebSockets** - 实时通信
- **SMTP** - 邮件发送
- **飞书/钉钉 API** - 企业通知

### 🎨 前端技术栈
- **React 18** - 现代前端框架
- **TypeScript** - 类型安全
- **Ant Design** - 企业级UI组件库
- **ECharts** - 专业数据可视化
- **Socket.IO** - 实时双向通信
- **Axios** - HTTP客户端
- **React Query** - 状态管理和缓存

### 🔧 基础设施
- **Docker** - 容器化部署
- **Docker Compose** - 多容器编排
- **Nginx** - 反向代理和负载均衡
- **Alembic** - 数据库迁移
- **Loguru** - 结构化日志

## 📁 项目结构

```
tcl-ai-agent/
├── backend/                          # 🚀 后端服务
│   ├── agents/                       # 🤖 多Agent实现
│   │   ├── base_agent.py            # Agent基类
│   │   ├── query_agent.py           # 查询Agent (Text-to-SQL)
│   │   ├── analysis_agent.py        # 分析Agent (统计分析)
│   │   ├── alert_agent.py           # 预警Agent (异常检测)
│   │   ├── report_agent.py          # 报告Agent (可视化)
│   │   ├── ab_test_agent.py         # A/B实验Agent
│   │   └── dify_agent.py            # Dify集成Agent
│   ├── api/                         # 🌐 API接口层
│   │   └── routes/                  # RESTful路由
│   │       ├── query.py             # 查询接口
│   │       ├── analysis.py          # 分析接口
│   │       ├── alerts.py            # 预警接口
│   │       ├── reports.py           # 报告接口
│   │       └── experiments.py       # 实验接口
│   ├── core/                        # 🏗️ 核心功能
│   │   ├── config.py               # 配置管理
│   │   ├── database.py             # 数据库连接
│   │   ├── agent_orchestrator.py   # Agent编排器
│   │   └── plugin_system.py        # 插件系统
│   ├── models/                      # 📊 数据模型
│   │   ├── base.py                 # 基础模型
│   │   ├── user.py                 # 用户模型
│   │   ├── session.py              # 会话模型
│   │   ├── query.py                # 查询模型
│   │   ├── experiment.py           # 实验模型
│   │   ├── alert.py                # 预警模型
│   │   └── report.py               # 报告模型
│   ├── services/                    # 🔧 业务服务层
│   │   ├── llm_service.py          # LLM服务
│   │   ├── data_service.py         # 数据服务
│   │   ├── text_to_sql_service.py  # Text-to-SQL微调
│   │   ├── vector_service.py       # 向量检索服务
│   │   ├── causal_inference_service.py # 因果推断服务
│   │   ├── data_engineering_service.py # 数据工程服务
│   │   ├── data_warehouse_service.py   # 数据仓库服务
│   │   ├── ppt_service.py          # PPT生成服务
│   │   ├── notification_service.py # 通知推送服务
│   │   └── framework_service.py    # 开源框架集成
│   ├── utils/                       # 🛠️ 工具函数
│   │   ├── logger.py               # 日志工具
│   │   ├── metrics.py              # 指标计算
│   │   └── validators.py           # 数据验证
│   ├── tests/                       # 🧪 测试用例
│   │   └── test_agents.py          # Agent测试
│   ├── plugins/                     # 🔌 插件目录
│   │   └── example_plugin/         # 示例插件
│   ├── main.py                     # 应用入口
│   └── requirements.txt            # Python依赖
├── frontend/                        # 🎨 前端应用
│   ├── src/
│   │   ├── components/             # React组件
│   │   │   ├── ChatInterface/      # 聊天界面
│   │   │   ├── Dashboard/          # 仪表板
│   │   │   ├── Charts/             # 图表组件
│   │   │   └── Reports/            # 报告组件
│   │   ├── pages/                  # 页面组件
│   │   │   ├── Home.tsx            # 首页
│   │   │   ├── Analytics.tsx       # 分析页
│   │   │   ├── Alerts.tsx          # 预警页
│   │   │   └── Experiments.tsx     # 实验页
│   │   ├── hooks/                  # 自定义Hooks
│   │   ├── services/               # API服务
│   │   ├── utils/                  # 工具函数
│   │   └── types/                  # TypeScript类型
│   ├── public/                     # 静态资源
│   ├── package.json               # 前端依赖
│   └── tsconfig.json              # TypeScript配置
├── sdk/                            # 📦 SDK客户端
│   └── python/                     # Python SDK
│       ├── tcl_ai_agent_sdk.py    # SDK主文件
│       ├── setup.py               # 安装配置
│       └── README.md              # SDK文档
├── data/                           # 📁 数据文件
│   ├── sample_data.sql            # 示例数据
│   └── schemas/                   # 数据模式
├── dbt_project/                    # 🔄 dbt项目
│   ├── models/                    # dbt模型
│   │   ├── staging/               # 暂存层
│   │   └── marts/                 # 数据集市
│   ├── macros/                    # dbt宏
│   └── dbt_project.yml           # dbt配置
├── plugins/                        # 🔌 外部插件
│   └── dify_example/              # Dify集成示例
├── templates/                      # 📋 模板文件
│   ├── ppt/                       # PPT模板
│   └── reports/                   # 报告模板
├── configs/                        # ⚙️ 配置文件
│   ├── docker/                    # Docker配置
│   └── nginx/                     # Nginx配置
├── docs/                          # 📚 项目文档
│   ├── API.md                     # API文档
│   ├── DEPLOYMENT.md              # 部署指南
│   └── DEMO_GUIDE.md              # 演示指南
├── scripts/                       # 📜 脚本文件
│   ├── setup.sh                  # 环境设置
│   └── deploy.sh                 # 部署脚本
├── docker-compose.yml             # 🐳 容器编排
├── Dockerfile                     # 🐳 容器构建
└── README.md                      # 📖 项目说明
```

## 🎯 核心功能模块

### 1. 🤖 多Agent协同框架
- **QueryAgent**: Text-to-SQL转换 + 语义查询优化
- **AnalysisAgent**: 统计分析 + 趋势预测 + 相关性分析
- **AlertAgent**: 异常检测 + 根因分析 + 智能告警
- **ReportAgent**: 可视化图表 + PPT生成 + 报告推送
- **ABTestAgent**: A/B实验设计 + 统计检验 + 因果推断
- **DifyAgent**: 低代码工作流集成

### 2. 🧠 Text-to-SQL微调引擎
- **模型微调**: 基于Transformers的Seq2Seq微调
- **Schema感知**: 自动加载数据库结构信息
- **置信度评估**: SQL生成质量评估
- **在线学习**: 支持增量学习和样本积累

### 3. 🔍 语义检索系统
- **向量检索**: ChromaDB + SentenceTransformers
- **知识图谱**: NetworkX构建实体关系网络
- **混合检索**: 语义+图谱融合算法
- **智能推荐**: 基于相似度的内容推荐

### 4. ⚠️ 智能预警诊断
- **异常检测**: 统计学方法 + 机器学习算法
- **根因分析**: 多维度下钻 + 因果关系分析
- **自动诊断**: AI驱动的问题识别和解释
- **多渠道告警**: 飞书/钉钉/邮件实时推送

### 5. 📊 自动化报告引擎
- **PPT生成**: python-pptx + 模板系统
- **图表生成**: Plotly + Matplotlib + ECharts
- **报告模板**: 商业报告 + 数据分析多种模板
- **一键分发**: 多平台自动推送

### 6. 🧪 A/B实验平台
- **实验设计**: 自动样本量计算 + 分组策略
- **统计检验**: t检验 + 卡方检验 + Mann-Whitney U
- **因果推断**: 倾向性得分匹配 + Uplift建模
- **实时监控**: 显著性监控 + 早停机制

### 7. 🔧 数据工程集成
- **流处理**: Apache Flink实时数据处理
- **批处理**: Apache Spark大数据分析
- **数据转换**: dbt现代数据栈
- **管道编排**: 多步骤数据处理流程

### 8. 🏗️ 数据仓库建模
- **维度建模**: 星型模式 + 雪花模式
- **SCD处理**: 缓慢变化维度Type 1/2/3
- **数据血缘**: 完整的血缘追踪系统
- **质量监控**: 完整性 + 唯一性 + 一致性检验

### 9. 🔬 高级因果推断
- **因果发现**: PC算法 + LiNGAM + 格兰杰检验
- **效应估计**: 工具变量 + 双重差分 + PSM
- **敏感性分析**: Rosenbaum bounds方法
- **反事实推理**: 场景模拟和效应预测

### 10. 📦 SaaS产品化
- **Python SDK**: 异步/同步双模式客户端
- **RESTful API**: 完整的API接口
- **插件系统**: 动态加载外部Agent
- **嵌入式组件**: 可集成的聊天窗口

## 🚀 快速开始

### 📋 环境要求
- **Python 3.9+** - 后端开发语言
- **Node.js 18+** - 前端开发环境
- **Docker & Docker Compose** - 容器化部署
- **PostgreSQL 13+** - 主数据库（生产环境）
- **Redis 6+** - 缓存和会话存储

### 🛠️ 安装步骤

#### 方式一：Docker快速部署（推荐）
```bash
# 1. 克隆项目
git clone https://github.com/0x1998s/Tcl_Aiagent.git
cd tcl-ai-agent

# 2. 一键启动所有服务
docker-compose up -d

# 3. 访问应用
# 前端: http://localhost:3000
# 后端API: http://localhost:8000
# API文档: http://localhost:8000/docs
```

#### 方式二：本地开发部署
```bash
# 1. 克隆项目
git clone https://github.com/0x1998s/Tcl_Aiagent.git
cd tcl-ai-agent

# 2. 后端设置
cd backend
pip install -r requirements.txt

# 配置环境变量（复制并修改）
cp .env.example .env

# 数据库迁移
alembic upgrade head

# 启动后端服务
uvicorn main:app --reload --port 8000

# 3. 前端设置（新终端）
cd frontend
npm install
npm start

# 4. 启动外部服务
# PostgreSQL, Redis, ChromaDB等
```

### 🔧 配置说明

#### 环境变量配置
```bash
# .env文件示例
DATABASE_URL=postgresql://postgres:password@localhost:5432/tcl_agent
REDIS_URL=redis://localhost:6379/0
OPENAI_API_KEY=your_openai_api_key
CHROMA_HOST=localhost
CHROMA_PORT=8001

# 通知配置
FEISHU_WEBHOOK_URL=https://open.feishu.cn/open-apis/bot/v2/hook/xxx
DINGTALK_WEBHOOK_URL=https://oapi.dingtalk.com/robot/send?access_token=xxx
EMAIL_SMTP_HOST=smtp.gmail.com
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
```

### 📖 使用示例

#### Python SDK使用
```python
import asyncio
from tcl_ai_agent_sdk import TCLAIAgentSDK

async def main():
    async with TCLAIAgentSDK(base_url="http://localhost:8000") as client:
        # 自然语言查询
        response = await client.chat("分析一下用户增长趋势")
        print(response.response)
        
        # 生成报告
        report = await client.generate_report(
            report_type="business_summary",
            time_range={"start": "2024-01-01", "end": "2024-01-31"},
            format="ppt"
        )
        print(f"报告生成完成: {report['file_path']}")

asyncio.run(main())
```

#### API调用示例
```bash
# 健康检查
curl http://localhost:8000/health

# 聊天对话
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "查询昨天的订单数量", "user_id": "test_user"}'

# 自然语言查询
curl -X POST "http://localhost:8000/api/query/execute" \
     -H "Content-Type: application/json" \
     -d '{"query": "过去7天的销售趋势", "query_type": "natural_language"}'
```

## API文档

启动后端服务后，访问 `http://localhost:8000/docs` 查看完整API文档。

## 📞 联系方式

- **作者**: Jemmy
- **微信**: Joeng_Jimmy  
- **邮箱**: jemmy_yang@yeah.net
- **项目地址**: https://github.com/0x1998s/Tcl_Aiagent

---

## 📄 许可证

MIT License

---

<div align="center">

**⭐ 如果这个项目对贵司有帮助，请给我发个offer，呜呜呜！**

**🚀 TCL AI Agent - 让数据分析更智能**

</div>
