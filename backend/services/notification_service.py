"""
通知推送服务
支持飞书、钉钉、邮件等多种通知方式的模块化集成
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from abc import ABC, abstractmethod
import httpx
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

from utils.logger import get_logger
from core.config import Settings

logger = get_logger(__name__)


class NotificationChannel(ABC):
    """通知渠道抽象基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = self.__class__.__name__
        
    @abstractmethod
    async def send(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """发送通知"""
        pass
    
    @abstractmethod
    async def test_connection(self) -> bool:
        """测试连接"""
        pass


class FeishuChannel(NotificationChannel):
    """飞书通知渠道"""
    
    def __init__(self, webhook_url: str, secret: Optional[str] = None):
        super().__init__({"webhook_url": webhook_url, "secret": secret})
        self.webhook_url = webhook_url
        self.secret = secret
    
    async def send(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """发送飞书消息"""
        
        try:
            # 构造飞书消息格式
            feishu_message = await self._build_feishu_message(message)
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.webhook_url,
                    json=feishu_message,
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
                
                if response.status_code == 200:
                    logger.info("飞书消息发送成功")
                    return {"status": "success", "channel": "feishu"}
                else:
                    logger.error(f"飞书消息发送失败: {response.status_code} - {response.text}")
                    return {"status": "error", "error": response.text}
                    
        except Exception as e:
            logger.error(f"飞书消息发送异常: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def _build_feishu_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """构造飞书消息格式"""
        
        msg_type = message.get("type", "text")
        
        if msg_type == "text":
            return {
                "msg_type": "text",
                "content": {
                    "text": message.get("content", "")
                }
            }
        elif msg_type == "rich_text":
            return {
                "msg_type": "post",
                "content": {
                    "post": {
                        "zh_cn": {
                            "title": message.get("title", "通知"),
                            "content": await self._build_rich_content(message)
                        }
                    }
                }
            }
        elif msg_type == "card":
            return await self._build_card_message(message)
        else:
            return {
                "msg_type": "text",
                "content": {
                    "text": message.get("content", "")
                }
            }
    
    async def _build_rich_content(self, message: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
        """构造富文本内容"""
        
        content = []
        
        # 添加文本内容
        if "content" in message:
            content.append([{
                "tag": "text",
                "text": message["content"]
            }])
        
        # 添加指标信息
        if "metrics" in message:
            metrics_text = "关键指标:\n"
            for key, value in message["metrics"].items():
                metrics_text += f"• {key}: {value}\n"
            
            content.append([{
                "tag": "text",
                "text": metrics_text
            }])
        
        # 添加链接
        if "link" in message:
            content.append([{
                "tag": "a",
                "text": "查看详情",
                "href": message["link"]
            }])
        
        return content
    
    async def _build_card_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """构造卡片消息"""
        
        return {
            "msg_type": "interactive",
            "card": {
                "config": {
                    "wide_screen_mode": True
                },
                "elements": [
                    {
                        "tag": "div",
                        "text": {
                            "content": message.get("content", ""),
                            "tag": "lark_md"
                        }
                    }
                ],
                "header": {
                    "title": {
                        "content": message.get("title", "通知"),
                        "tag": "plain_text"
                    },
                    "template": message.get("color", "blue")
                }
            }
        }
    
    async def test_connection(self) -> bool:
        """测试飞书连接"""
        
        test_message = {
            "msg_type": "text",
            "content": {
                "text": "TCL AI Agent 连接测试"
            }
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.webhook_url,
                    json=test_message,
                    timeout=10
                )
                return response.status_code == 200
        except:
            return False


class DingTalkChannel(NotificationChannel):
    """钉钉通知渠道"""
    
    def __init__(self, webhook_url: str, secret: Optional[str] = None):
        super().__init__({"webhook_url": webhook_url, "secret": secret})
        self.webhook_url = webhook_url
        self.secret = secret
    
    async def send(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """发送钉钉消息"""
        
        try:
            # 构造钉钉消息格式
            dingtalk_message = await self._build_dingtalk_message(message)
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.webhook_url,
                    json=dingtalk_message,
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("errcode") == 0:
                        logger.info("钉钉消息发送成功")
                        return {"status": "success", "channel": "dingtalk"}
                    else:
                        logger.error(f"钉钉消息发送失败: {result}")
                        return {"status": "error", "error": result.get("errmsg", "未知错误")}
                else:
                    logger.error(f"钉钉消息发送失败: {response.status_code}")
                    return {"status": "error", "error": f"HTTP {response.status_code}"}
                    
        except Exception as e:
            logger.error(f"钉钉消息发送异常: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def _build_dingtalk_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """构造钉钉消息格式"""
        
        msg_type = message.get("type", "text")
        
        if msg_type == "text":
            return {
                "msgtype": "text",
                "text": {
                    "content": message.get("content", "")
                }
            }
        elif msg_type == "markdown":
            return {
                "msgtype": "markdown",
                "markdown": {
                    "title": message.get("title", "通知"),
                    "text": message.get("content", "")
                }
            }
        elif msg_type == "actionCard":
            return {
                "msgtype": "actionCard",
                "actionCard": {
                    "title": message.get("title", "通知"),
                    "text": message.get("content", ""),
                    "singleTitle": "查看详情",
                    "singleURL": message.get("link", "")
                }
            }
        else:
            return {
                "msgtype": "text",
                "text": {
                    "content": message.get("content", "")
                }
            }
    
    async def test_connection(self) -> bool:
        """测试钉钉连接"""
        
        test_message = {
            "msgtype": "text",
            "text": {
                "content": "TCL AI Agent 连接测试"
            }
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.webhook_url,
                    json=test_message,
                    timeout=10
                )
                if response.status_code == 200:
                    result = response.json()
                    return result.get("errcode") == 0
                return False
        except:
            return False


class EmailChannel(NotificationChannel):
    """邮件通知渠道"""
    
    def __init__(
        self, 
        smtp_host: str,
        smtp_port: int,
        username: str,
        password: str,
        from_email: Optional[str] = None
    ):
        super().__init__({
            "smtp_host": smtp_host,
            "smtp_port": smtp_port,
            "username": username,
            "password": password,
            "from_email": from_email or username
        })
    
    async def send(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """发送邮件"""
        
        try:
            # 构造邮件
            msg = MIMEMultipart()
            msg['From'] = self.config["from_email"]
            msg['To'] = message.get("to", "")
            msg['Subject'] = message.get("subject", "TCL AI Agent 通知")
            
            # 添加邮件正文
            body = message.get("content", "")
            msg.attach(MIMEText(body, 'html' if message.get("html", False) else 'plain', 'utf-8'))
            
            # 添加附件
            if "attachments" in message:
                for attachment_path in message["attachments"]:
                    await self._add_attachment(msg, attachment_path)
            
            # 发送邮件
            server = smtplib.SMTP(self.config["smtp_host"], self.config["smtp_port"])
            server.starttls()
            server.login(self.config["username"], self.config["password"])
            
            text = msg.as_string()
            server.sendmail(self.config["from_email"], message.get("to", ""), text)
            server.quit()
            
            logger.info("邮件发送成功")
            return {"status": "success", "channel": "email"}
            
        except Exception as e:
            logger.error(f"邮件发送异常: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def _add_attachment(self, msg: MIMEMultipart, file_path: str):
        """添加邮件附件"""
        
        try:
            with open(file_path, "rb") as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
            
            encoders.encode_base64(part)
            part.add_header(
                'Content-Disposition',
                f'attachment; filename= {file_path.split("/")[-1]}'
            )
            
            msg.attach(part)
            
        except Exception as e:
            logger.error(f"添加附件失败: {str(e)}")
    
    async def test_connection(self) -> bool:
        """测试邮件连接"""
        
        try:
            server = smtplib.SMTP(self.config["smtp_host"], self.config["smtp_port"])
            server.starttls()
            server.login(self.config["username"], self.config["password"])
            server.quit()
            return True
        except:
            return False


class NotificationService:
    """通知服务管理器"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.channels: Dict[str, NotificationChannel] = {}
        
    async def initialize(self):
        """初始化通知服务"""
        logger.info("初始化通知服务...")
        
        # 初始化飞书渠道
        if self.settings.FEISHU_WEBHOOK_URL:
            self.channels["feishu"] = FeishuChannel(self.settings.FEISHU_WEBHOOK_URL)
            logger.info("飞书通知渠道已配置")
        
        # 初始化钉钉渠道
        if self.settings.DINGTALK_WEBHOOK_URL:
            self.channels["dingtalk"] = DingTalkChannel(self.settings.DINGTALK_WEBHOOK_URL)
            logger.info("钉钉通知渠道已配置")
        
        # 初始化邮件渠道
        if all([
            self.settings.EMAIL_SMTP_HOST,
            self.settings.EMAIL_USERNAME,
            self.settings.EMAIL_PASSWORD
        ]):
            self.channels["email"] = EmailChannel(
                self.settings.EMAIL_SMTP_HOST,
                self.settings.EMAIL_SMTP_PORT,
                self.settings.EMAIL_USERNAME,
                self.settings.EMAIL_PASSWORD
            )
            logger.info("邮件通知渠道已配置")
        
        # 测试连接
        await self._test_all_connections()
        
        logger.info(f"通知服务初始化完成，可用渠道: {list(self.channels.keys())}")
    
    async def _test_all_connections(self):
        """测试所有通知渠道连接"""
        
        for channel_name, channel in self.channels.items():
            try:
                is_connected = await channel.test_connection()
                if is_connected:
                    logger.info(f"{channel_name} 连接测试成功")
                else:
                    logger.warning(f"{channel_name} 连接测试失败")
            except Exception as e:
                logger.error(f"{channel_name} 连接测试异常: {str(e)}")
    
    async def send_notification(
        self,
        message: Dict[str, Any],
        channels: Optional[List[str]] = None,
        priority: str = "normal"
    ) -> Dict[str, Any]:
        """发送通知"""
        
        if not channels:
            channels = list(self.channels.keys())
        
        results = {}
        
        # 并行发送到所有指定渠道
        tasks = []
        for channel_name in channels:
            if channel_name in self.channels:
                task = self._send_to_channel(channel_name, message)
                tasks.append(task)
        
        if tasks:
            channel_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(channel_results):
                channel_name = channels[i] if i < len(channels) else f"channel_{i}"
                if isinstance(result, Exception):
                    results[channel_name] = {"status": "error", "error": str(result)}
                else:
                    results[channel_name] = result
        
        # 统计发送结果
        success_count = sum(1 for r in results.values() if r.get("status") == "success")
        total_count = len(results)
        
        logger.info(f"通知发送完成: {success_count}/{total_count} 成功")
        
        return {
            "success_count": success_count,
            "total_count": total_count,
            "results": results,
            "message_id": f"msg_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
    
    async def _send_to_channel(self, channel_name: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """发送消息到指定渠道"""
        
        channel = self.channels.get(channel_name)
        if not channel:
            return {"status": "error", "error": f"渠道不存在: {channel_name}"}
        
        try:
            return await channel.send(message)
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def send_alert(
        self,
        alert_type: str,
        title: str,
        content: str,
        metrics: Optional[Dict[str, Any]] = None,
        severity: str = "medium",
        channels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """发送预警通知"""
        
        # 根据严重程度选择合适的消息格式
        if severity in ["high", "critical"]:
            message = {
                "type": "card" if "feishu" in (channels or []) else "markdown",
                "title": f"🚨 {title}",
                "content": content,
                "metrics": metrics,
                "color": "red"
            }
        else:
            message = {
                "type": "rich_text" if "feishu" in (channels or []) else "text",
                "title": f"⚠️ {title}",
                "content": content,
                "metrics": metrics
            }
        
        return await self.send_notification(message, channels, "high")
    
    async def send_report(
        self,
        report_title: str,
        report_content: str,
        attachments: Optional[List[str]] = None,
        recipients: Optional[List[str]] = None,
        channels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """发送报告通知"""
        
        message = {
            "type": "markdown",
            "title": f"📊 {report_title}",
            "content": report_content,
            "attachments": attachments or [],
            "to": recipients[0] if recipients else ""  # 邮件需要收件人
        }
        
        return await self.send_notification(message, channels or ["email"], "normal")
    
    async def send_experiment_result(
        self,
        experiment_name: str,
        results: Dict[str, Any],
        recommendation: str,
        channels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """发送A/B实验结果通知"""
        
        content = f"""
        实验名称: {experiment_name}
        实验结果: {results.get('conclusion', 'N/A')}
        置信度: {results.get('confidence', 'N/A')}
        建议: {recommendation}
        """
        
        message = {
            "type": "card",
            "title": f"🧪 A/B实验结果: {experiment_name}",
            "content": content,
            "color": "green"
        }
        
        return await self.send_notification(message, channels, "normal")
    
    def get_available_channels(self) -> List[str]:
        """获取可用的通知渠道"""
        return list(self.channels.keys())
    
    async def get_channel_status(self) -> Dict[str, Dict[str, Any]]:
        """获取所有渠道状态"""
        
        status = {}
        
        for channel_name, channel in self.channels.items():
            try:
                is_connected = await channel.test_connection()
                status[channel_name] = {
                    "connected": is_connected,
                    "config": {k: "***" if "password" in k.lower() or "secret" in k.lower() 
                             else v for k, v in channel.config.items()}
                }
            except Exception as e:
                status[channel_name] = {
                    "connected": False,
                    "error": str(e)
                }
        
        return status
