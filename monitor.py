"""
盤中監測器 — 每 10 分鐘檢查 watchlist_conditions.json 裡的條件是否達到

用法：
    python3 monitor.py           # 單次執行（由 crontab 呼叫）

Crontab（台股 09:00–13:30）：
    */10 9-13 * * 1-5 cd /Users/mu/fire-auto && python3 monitor.py >> data/monitor.log 2>&1

資料來源：FinMind TaiwanStockPrice（盤中即時快照，約有 15 分鐘延遲）
"""

import json
import os
import sys
import urllib.request
import urllib.parse
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent
WATCHLIST_PATH = BASE_DIR / "data" / "watchlist_conditions.json"
ENV_PATH = BASE_DIR / ".env.local"
API_URL = "https://api.finmindtrade.com/api/v4/data"

# 台股交易時間
MARKET_OPEN = (9, 0)
MARKET_CLOSE = (13, 30)


def _load_token() -> str:
    val = os.environ.get("FINMIND_API_TOKEN")
    if val:
        return val
    for line in ENV_PATH.read_text().splitlines():
        if line.startswith("FINMIND_API_TOKEN="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("找不到 FINMIND_API_TOKEN")


def _is_market_hours() -> bool:
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    t = (now.hour, now.minute)
    return MARKET_OPEN <= t <= MARKET_CLOSE


def _fetch_current(stock_id: str, date: str, token: str) -> dict | None:
    """抓單股今日 OHLCV 快照"""
    params = urllib.parse.urlencode({
        "dataset": "TaiwanStockPrice",
        "data_id": stock_id,
        "start_date": date,
        "token": token,
    })
    try:
        req = urllib.request.Request(f"{API_URL}?{params}")
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        if data.get("status") != 200 or not data.get("data"):
            return None
        row = data["data"][-1]  # 取最新一筆
        if row["date"] != date:
            return None  # 非今日資料
        return {
            "open": float(row["open"]),
            "high": float(row["max"]),
            "low": float(row["min"]),
            "close": float(row["close"]),
            "volume_张": int(row["Trading_turnover"]),  # 注意大寫 T
        }
    except Exception:
        return None


def _check_squeeze(stock: dict, cur: dict) -> tuple[bool, list[str]]:
    """檢查波動率擠壓觸發條件，回傳 (triggered, status_lines)"""
    lines = []
    triggered_conditions = 0
    total_conditions = 0

    # 價格突破上軌
    price_above = stock.get("price_above")
    if price_above:
        total_conditions += 1
        if cur["close"] >= price_above:
            lines.append(f"  ✓ 突破上軌 {price_above:.1f}（現 {cur['close']:.1f}）")
            triggered_conditions += 1
        else:
            dist = (price_above - cur["close"]) / price_above * 100
            lines.append(f"  · 未突破上軌 {price_above:.1f}（現 {cur['close']:.1f}，差 {dist:.1f}%）")

    # ADX（歷史條件，盤中不變）
    if not stock.get("adx_ok", True):
        lines.append(f"  ✗ ADX 不足（歷史條件未達）")
        total_conditions += 1
    else:
        triggered_conditions += 1
        total_conditions += 1

    # 量能
    vol_min = stock.get("vol_min_张", 0)
    if vol_min:
        total_conditions += 1
        if cur["volume_张"] >= vol_min:
            lines.append(f"  ✓ 成交量 {cur['volume_张']:,}張 ≥ {vol_min:,}張")
            triggered_conditions += 1
        else:
            lines.append(f"  · 量能不足 {cur['volume_张']:,}張 < {vol_min:,}張")

    # 5日動量
    mom5_ref = stock.get("mom5_ref")
    if mom5_ref:
        total_conditions += 1
        if cur["close"] > mom5_ref:
            lines.append(f"  ✓ 5日動量正向（現 {cur['close']:.1f} > {mom5_ref:.1f}）")
            triggered_conditions += 1
        else:
            lines.append(f"  · 5日動量不足（現 {cur['close']:.1f} ≤ {mom5_ref:.1f}）")

    all_met = triggered_conditions == total_conditions and total_conditions > 0
    return all_met, lines


def _check_oversold(stock: dict, cur: dict) -> tuple[bool, list[str]]:
    """檢查超跌反彈觸發條件，回傳 (triggered, status_lines)"""
    lines = []
    missing = stock.get("missing", [])
    new_met = []

    # 紅K（盤中可觀察）
    if stock.get("need_red_candle"):
        if cur["close"] > cur["open"]:
            lines.append(f"  ✓ 紅K 成立（現 {cur['close']:.1f} > 開 {cur['open']:.1f}）")
            new_met.append("紅K")
        else:
            lines.append(f"  · 尚未紅K（現 {cur['close']:.1f} ≤ 開 {cur['open']:.1f}）")

    # 量縮
    vol_max = stock.get("vol_max_张", 0)
    if vol_max and "量縮" in missing:
        if cur["volume_张"] < vol_max:
            lines.append(f"  ✓ 量縮 {cur['volume_张']:,}張 < {vol_max:,}張")
            new_met.append("量縮")
        else:
            lines.append(f"  · 量未縮 {cur['volume_张']:,}張 ≥ {vol_max:,}張")

    # KD 超賣（需價格繼續下跌）
    kd_price_max = stock.get("kd_price_max")
    if kd_price_max and "短線超賣" in missing:
        if cur["close"] <= kd_price_max:
            lines.append(f"  ✓ 超賣達標（現 {cur['close']:.1f} ≤ {kd_price_max:.1f}）")
            new_met.append("短線超賣")
        else:
            lines.append(f"  · 超賣未達（現 {cur['close']:.1f} > {kd_price_max:.1f}）")

    # 已滿足的歷史條件
    already_met = [c for c in ["60均之上", "5日跌5%", "短線超賣"] if c not in missing]
    remaining = [c for c in missing if c not in new_met]

    all_met = len(remaining) == 0
    return all_met, lines


def _check_ad(stock: dict, cur: dict) -> tuple[bool, list[str]]:
    """檢查 AD 背離觸發條件，回傳 (triggered, status_lines)"""
    lines = []
    missing = stock.get("missing", [])
    new_met = []

    # 創 20 日新低
    price_low_threshold = stock.get("price_low_threshold")
    if price_low_threshold and "20日新低" in missing:
        if cur["close"] <= price_low_threshold:
            lines.append(f"  ✓ 創新低（現 {cur['close']:.1f} ≤ {price_low_threshold:.1f}）")
            new_met.append("20日新低")
        else:
            lines.append(f"  · 未創新低（現 {cur['close']:.1f} > {price_low_threshold:.1f}）")

    # RSI 轉弱
    rsi_price_max = stock.get("rsi_price_max")
    if rsi_price_max and "動能轉弱" in missing:
        if cur["close"] <= rsi_price_max:
            lines.append(f"  ✓ 動能轉弱（現 {cur['close']:.1f} ≤ {rsi_price_max:.1f}）")
            new_met.append("動能轉弱")
        else:
            lines.append(f"  · 動能仍強（現 {cur['close']:.1f} > {rsi_price_max:.1f}）")

    remaining = [c for c in missing if c not in new_met]
    all_met = len(remaining) == 0
    return all_met, lines


def main():
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M")

    if not _is_market_hours():
        print(f"[{time_str}] 非交易時間，略過")
        return

    if not WATCHLIST_PATH.exists():
        print(f"[{time_str}] 找不到 {WATCHLIST_PATH}，請先執行 scan.py")
        return

    watchlist = json.loads(WATCHLIST_PATH.read_text())
    stocks = watchlist.get("stocks", [])
    data_date = watchlist.get("data_date", "")
    generated_at = watchlist.get("generated_at", "")

    if not stocks:
        print(f"[{time_str}] Watchlist 為空")
        return

    token = _load_token()

    print(f"\n{'='*60}")
    print(f"  盤中監測  {today} {time_str}")
    print(f"  Watchlist 來源：{generated_at}（資料日期 {data_date}）")
    print(f"{'='*60}")

    triggered_all = []
    near_all = []

    for stock in stocks:
        sid = stock["stock_id"]
        name = stock["name"]
        strategy = stock["strategy"]
        strategy_label = {"squeeze": "波動率擠壓", "oversold": "超跌反彈", "ad_divergence": "AD背離"}.get(strategy, strategy)

        cur = _fetch_current(sid, today, token)
        if cur is None:
            continue

        if strategy == "squeeze":
            triggered, lines = _check_squeeze(stock, cur)
        elif strategy == "oversold":
            triggered, lines = _check_oversold(stock, cur)
        elif strategy == "ad_divergence":
            triggered, lines = _check_ad(stock, cur)
        else:
            continue

        entry = {
            "stock_id": sid, "name": name,
            "strategy_label": strategy_label,
            "close": cur["close"], "open": cur["open"],
            "volume_张": cur["volume_张"],
            "lines": lines, "triggered": triggered,
        }
        if triggered:
            triggered_all.append(entry)
        else:
            near_all.append(entry)

    from notify import send

    # 已觸發 → 每檔個別發 Telegram
    if triggered_all:
        print(f"\n  🔔 已觸發條件（{len(triggered_all)} 檔）")
        for e in triggered_all:
            print(f"\n  ★ {e['stock_id']} {e['name']}【{e['strategy_label']}】現價 {e['close']:.1f}  量 {e['volume_张']:,}張")
            for line in e["lines"]:
                print(line)
            # Telegram 通知
            detail = "\n".join(l.strip() for l in e["lines"])
            send(
                f"🔔 {e['stock_id']} {e['name']}【{e['strategy_label']}】\n"
                f"現價 {e['close']:.1f}  量 {e['volume_张']:,}張\n"
                f"{detail}"
            )
    else:
        print(f"\n  目前無股票觸發")

    # 接近觸發 → 只印 log，不發通知
    if near_all:
        print(f"\n  觀察中（{len(near_all)} 檔）")
        for e in near_all:
            print(f"\n  {e['stock_id']} {e['name']}【{e['strategy_label']}】現價 {e['close']:.1f}  量 {e['volume_张']:,}張")
            for line in e["lines"]:
                print(line)

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
