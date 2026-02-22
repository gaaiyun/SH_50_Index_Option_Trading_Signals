#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PushPlus 推送模块
用于中证50期权策略的实时通知
"""

import requests
import json
from datetime import datetime
from typing import Optional, Dict


class PushPlus:
    """PushPlus 消息推送类"""
    
    def __init__(self, token: str, secret: str = ""):
        """
        初始化PushPlus
        
        Args:
            token: 用户Token
            secret: 签名密钥(可选)
        """
        self.token = token
        self.secret = secret
        self.api_url = "http://www.pushplus.plus/send"
    
    def send(self, 
             title: str, 
             content: str, 
             channel: str = "wechat",
             template: str = "markdown",
             ) -> Dict:
        """
        发送消息
        
        Args:
            title: 消息标题
            content: 消息内容
            channel: 发送渠道 (wechat/webhook/cp/mail)
            template: 模板类型 (html/txt/json/markdown)
        
        Returns:
            Dict: 发送结果
        """
        data = {
            "token": self.token,
            "title": title,
            "content": content,
            "template": template,
            "channel": channel
        }
        
        if self.secret:
            data["signature"] = self._generate_signature()
        
        try:
            response = requests.post(
                self.api_url,
                json=data,
                timeout=10
            )
            result = response.json()
            
            return {
                "success": result.get("code") == 200,
                "code": result.get("code"),
                "msg": result.get("msg"),
                "data": result.get("data")
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _generate_signature(self) -> str:
        """生成签名"""
        import hashlib
        import time
        
        timestamp = int(time.time() * 1000)
        message = f"{self.token}{timestamp}{self.secret}"
        
        return hashlib.sha256(message.encode()).hexdigest()
    
    def send_signal(self, signal_type: str, message: str) -> Dict:
        """
        发送交易信号通知
        
        Args:
            signal_type: 信号类型 (建仓/止损/平仓)
            message: 消息内容
        
        Returns:
            Dict: 发送结果
        """
        emoji_map = {
            "建仓": "[建仓信号]",
            "止损": "[止损信号]",
            "平仓": "[平仓信号]",
            "警告": "[风险警告]"
        }
        
        title = f"50ETF期权策略 {emoji_map.get(signal_type, signal_type)}"
        
        content = f"""
## {signal_type}信号

**时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**内容**: {message}

---
*自动发送 - 中证50期权策略看板*
"""
        
        return self.send(title, content, template="markdown")
    
    def send_alert(self, alert_type: str, content: str) -> Dict:
        """
        发送警报
        
        Args:
            alert_type: 警报类型
            content: 警报内容
        """
        title = f"[{alert_type}] 期权策略监控"
        
        message = f"""
## {alert_type}

**时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{content}

---
*自动发送 - 中证50期权策略看板*
"""
        
        return self.send(title, message, template="markdown")


# ==================== 配置 ====================
# PushPlus 配置
PUSHPLUS_TOKEN = "3660eb1e0b364a78b3beed2f349b29f8"
PUSHPLUS_SECRET = "ddff31dda80446cc878c163b2410bc5b"

# 创建推送实例
push_client = PushPlus(PUSHPLUS_TOKEN, PUSHPLUS_SECRET)


# ==================== 测试函数 ====================
def test_push():
    """测试推送"""
    result = push_client.send_signal("建仓", "BSADF检验触发，建议卖出虚值期权")
    print(result)
    return result


if __name__ == "__main__":
    test_push()
