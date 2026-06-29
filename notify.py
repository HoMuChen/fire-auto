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


def _parse_ids(raw: str | None) -> list[str]:
    """逗號分隔的 chat id 字串 -> list"""
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def _load_config() -> tuple[str, str | None, str | None]:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    intraday = os.environ.get("TELEGRAM_CHAT_ID_INTRADAY")
    if not token or not chat_id or intraday is None:
        for line in ENV_PATH.read_text().splitlines():
            if line.startswith("TELEGRAM_BOT_TOKEN=") and not token:
                token = line.split("=", 1)[1].strip()
            elif line.startswith("TELEGRAM_CHAT_ID=") and not chat_id:
                chat_id = line.split("=", 1)[1].strip()
            elif line.startswith("TELEGRAM_CHAT_ID_INTRADAY=") and not intraday:
                intraday = line.split("=", 1)[1].strip()
    if not token or not chat_id:
        raise RuntimeError("找不到 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID")
    return token, chat_id, intraday


def _post(token: str, chat_id: str, text: str, md: bool) -> bool:
    payload = {"chat_id": chat_id, "text": text}
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


def send(text: str, md: bool = False, intraday: bool = False) -> bool:
    """發送 Telegram 訊息，失敗回傳 False（不拋例外）

    intraday=True 時，除了預設 TELEGRAM_CHAT_ID 外，
    額外送給 TELEGRAM_CHAT_ID_INTRADAY 的盤中收件人。
    """
    try:
        token, chat_id, intraday_ids = _load_config()
        ids = _parse_ids(chat_id)
        if intraday:
            ids += _parse_ids(intraday_ids)
        # 去重保序
        seen: set[str] = set()
        targets = [i for i in ids if not (i in seen or seen.add(i))]
        ok_all = True
        for cid in targets:
            if not _post(token, cid, text, md):
                ok_all = False
        return ok_all
    except Exception as e:
        print(f"  [Telegram] 發送失敗：{e}")
        return False
