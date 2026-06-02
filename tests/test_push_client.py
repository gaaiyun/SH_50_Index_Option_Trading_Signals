# -*- coding: utf-8 -*-
"""VolGuard Pro — 推送模块单元测试（mock requests.post）"""

import pytest
import os
from unittest.mock import patch, MagicMock

import sys
import os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))

from push_client import PushPlus, get_push_client


class TestPushPlusSend:
    """PushPlus.send 行为"""

    def test_success_when_code_200(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"code": 200, "msg": "ok", "data": None}
        mock_resp.raise_for_status = MagicMock()
        with patch("push_client.requests.post", return_value=mock_resp):
            client = PushPlus("test_token")
            out = client.send("标题", "内容")
        assert out["success"] is True
        assert out.get("code") == 200

    def test_failure_when_code_not_200(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"code": 400, "msg": "token invalid"}
        mock_resp.raise_for_status = MagicMock()
        with patch("push_client.requests.post", return_value=mock_resp):
            client = PushPlus("test_token")
            out = client.send("标题", "内容")
        assert out["success"] is False
        assert out.get("code") == 400

    def test_secret_does_not_add_unsupported_send_fields(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"code": 200, "msg": "ok", "data": None}
        mock_resp.raise_for_status = MagicMock()
        with patch("push_client.requests.post", return_value=mock_resp) as mpost:
            client = PushPlus("test_token", "test_secret")
            client.send("标题", "内容")
        payload = mpost.call_args[1]["json"]
        assert "sign" not in payload
        assert "timestamp" not in payload

    def test_send_signal_builds_title_and_content(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"code": 200, "msg": "ok"}
        mock_resp.raise_for_status = MagicMock()
        with patch("push_client.requests.post", return_value=mock_resp) as mpost:
            client = PushPlus("t")
            client.send_signal("建仓", "测试消息")
        call_kw = mpost.call_args[1]
        payload = call_kw["json"]
        assert "[建仓信号]" in payload["title"]
        assert "建仓信号" in payload["content"]
        assert "测试消息" in payload["content"]

    def test_send_alert_builds_title_and_content(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"code": 200, "msg": "ok"}
        mock_resp.raise_for_status = MagicMock()
        with patch("push_client.requests.post", return_value=mock_resp) as mpost:
            client = PushPlus("t")
            client.send_alert("波动率预警", "IV 突破 30%")
        call_kw = mpost.call_args[1]
        payload = call_kw["json"]
        assert "波动率预警" in payload["title"]
        assert "IV 突破 30%" in payload["content"]


class TestPushPlusInit:
    def test_empty_token_raises(self):
        with pytest.raises(ValueError, match="token cannot be empty"):
            PushPlus("")


class TestGetPushClient:
    """get_push_client 优先级与无 token 行为"""

    def test_with_param_token_creates_client(self):
        with patch.dict(os.environ, {}, clear=False):
            client = get_push_client(token="param_token")
        assert client is not None
        assert client.token == "param_token"

    def test_with_env_token_creates_client(self):
        with patch.dict(os.environ, {"PUSHPLUS_TOKEN": "env_token", "PUSHPLUS_SECRET": ""}, clear=False):
            with patch("streamlit.secrets", MagicMock(get=lambda k, default="": default)):
                client = get_push_client()
        assert client is not None
        assert client.token == "env_token"

    def test_no_token_returns_none(self):
        with patch.dict(os.environ, {"PUSHPLUS_TOKEN": "", "PUSHPLUS_SECRET": ""}, clear=False):
            with patch("streamlit.secrets", MagicMock(get=lambda k, default="": "")):
                client = get_push_client()
        assert client is None
