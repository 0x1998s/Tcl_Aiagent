# Dify集成示例

这个目录包含了与Dify平台集成的示例代码和配置。

## 文件说明

- `dify_workflow_config.json` - Dify工作流配置示例
- `dify_integration_guide.md` - 集成指南
- `sample_workflows/` - 示例工作流

## 使用方法

1. 配置Dify API密钥
2. 创建工作流
3. 配置Agent路由
4. 测试集成

## 配置示例

```python
# 在.env文件中配置
ENABLE_DIFY=true
DIFY_BASE_URL=http://your-dify-instance:5001
DIFY_API_KEY=your-api-key
DIFY_DEFAULT_APP_ID=your-app-id
```
