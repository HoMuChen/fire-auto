"""
Telegram 通知模組

用法：
    from notify import send

    send("訊息內容")                    # 純文字
    send("*粗體* `code`", md=True)     # Markdown
"""

import os
import urllib.request
import urllib.parse
import json
from pathlib import Path

ENV_PATH = Path(__file__).parent / ".env.local"
API_BASE = "https://api.telegram.org/bot"


def _load_config() -> tuple[str, str]:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if token and chat_id:
        return token, chat_id
    for line in ENV_PATH.read_text().splitlines():
        if line.startswith("TELEGRAM_BOT_TOKEN="):
            token = line.split("=", 1)[1].strip()
        elif line.startswith("TELEGRAM_CHAT_ID="):
            chat_id = line.split("=", 1)[1].strip()
    if not token or not chat_id:
        raise RuntimeError("找不到 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID")
    return token, chat_id


def send(text: str, md: bool = False) -> bool:
    """發送 Telegram 訊息，失敗回傳 False（不拋例外）"""
    try:
        token, chat_id = _load_config()
        payload = {
            "chat_id": chat_id,
            "text": text,
        }
        if md:
            payload["parse_mode"] = "Markdown"
        data = urllib.parse.urlencode(payload).encode()
        req = urllib.request.Request(
            f"{API_BASE}{token}/sendMessage",
            data=data,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
        return result.get("ok", False)
    except Exception as e:
        print(f"  [Telegram] 發送失敗：{e}")
        return False
