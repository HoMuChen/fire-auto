"""
盤中監測器 — 每 10 分鐘全市場掃描三策略即時信號

架構：
  1. 一個 API call 取得全市場即時快照
  2. 讀本地 CSV 歷史資料（無需額外 API call）
  3. 把今日即時價格接在歷史後面，跑完整策略邏輯
  → 可偵測「當日才達成條件」的信號（如 3176 基亞類型）

用法：
    python3 monitor.py           # 單次執行（由 crontab 呼叫）

Crontab（台股 09:00–13:30）：
    */10 9-13 * * 1-5 cd /Users/mu/fire-auto && python3 monitor.py >> data/monitor.log 2>&1
"""

import csv
import json
import os
import sys
import urllib.request
import urllib.parse
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "stock_prices"
INDIVIDUAL_STOCKS_PATH = BASE_DIR / "individual_stocks.json"
FILTERED_LISTS_PATH = BASE_DIR / "strategies" / "filtered_stock_lists.json"
ENV_PATH = BASE_DIR / ".env.local"
FINMIND_SNAPSHOT_URL = "https://api.finmindtrade.com/api/v4/taiwan_stock_tick_snapshot"

MARKET_OPEN = (9, 0)
MARKET_CLOSE = (13, 30)

sys.path.insert(0, str(BASE_DIR))
from backtest import (
    calc_sma, calc_bollinger, calc_keltner, calc_adx,
    calc_kd, calc_rsi, calc_ad_line,
)


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


def _fetch_market_snapshot(token: str) -> dict:
    """FinMind 即時快照（台股全市場，1 call），回傳 {stock_id: {open,high,low,close,volume_张}}"""
    params = urllib.parse.urlencode({"token": token})
    try:
        req = urllib.request.Request(f"{FINMIND_SNAPSHOT_URL}?{params}")
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        if data.get("status") != 200:
            print(f"  [FinMind] 快照失敗：{data.get('msg')}")
            return {}
        result = {}
        for row in data.get("data", []):
            sid = row.get("stock_id", "")
            if not sid or len(sid) != 4:  # 排除指數（3碼）
                continue
            try:
                o = float(row["open"])
                c = float(row["close"])
                if o <= 0 or c <= 0:
                    continue
                result[sid] = {
                    "open": o,
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": c,
                    "volume_张": int(row.get("total_volume", 0)),  # 今日累計張數
                }
            except (ValueError, KeyError, TypeError):
                continue
        return result
    except Exception as e:
        print(f"  [FinMind] 快照失敗：{e}")
        return {}


def _load_history(stock_id: str, n: int = 85) -> list[dict] | None:
    """讀取本地 CSV 最後 n 筆，格式與 backtest.py 相容"""
    csv_path = DATA_DIR / f"{stock_id}.csv"
    if not csv_path.exists():
        return None
    try:
        rows = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                c = float(row.get("close") or 0)
                if c > 0:
                    rows.append({
                        "date": row["date"],
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": c,
                        "volume": float(row["volume"]),
                    })
        return rows[-n:] if len(rows) >= 65 else None
    except Exception:
        return None


# ── 策略條件檢查（傳入已含今日即時價格的 prices 列表）─────────────

def _check_squeeze(prices: list[dict]) -> tuple[str | None, list[str]]:
    closes = [p["close"] for p in prices]
    vols = [p["volume"] for p in prices]
    upper_bb, lower_bb, _ = calc_bollinger(closes, 20, 2)
    upper_kc, lower_kc, _ = calc_keltner(prices, 20, 1.5)
    adx = calc_adx(prices, 14)
    vol_ma = calc_sma([float(v) for v in vols], 20)
    n = len(prices)
    i = n - 1

    if i < 6 or any(v is None for v in [upper_bb[i], upper_kc[i], adx[i], vol_ma[i]]):
        return None, []

    in_sq = []
    for j in range(n):
        vals = [upper_bb[j], lower_bb[j], upper_kc[j], lower_kc[j]]
        in_sq.append(
            all(v is not None for v in vals)
            and upper_bb[j] < upper_kc[j] and lower_bb[j] > lower_kc[j]
        )
    sq_count = [0] * n
    for j in range(1, n):
        sq_count[j] = sq_count[j - 1] + 1 if in_sq[j] else 0

    just_released = not in_sq[i] and sq_count[i - 1] >= 5
    currently_squeezing = in_sq[i] and sq_count[i] >= 3

    bb_breakout = closes[i] > upper_bb[i]
    adx_ok = adx[i] > 18
    vol_ok = vols[i] > vol_ma[i] * 1.4
    mom5 = (closes[i] - closes[i - 5]) / closes[i - 5] if closes[i - 5] > 0 else 0
    mom_ok = mom5 > 0
    vol_ratio = vols[i] / vol_ma[i] if vol_ma[i] > 0 else 0

    if just_released and bb_breakout and adx_ok and vol_ok and mom_ok:
        return "triggered", [
            f"  ✓ 脫離擠壓（前{sq_count[i-1]}天）突破 {upper_bb[i]:.1f}",
            f"  ✓ ADX {adx[i]:.0f}  量比 {vol_ratio:.1f}x  動量 +{mom5*100:.1f}%",
        ]

    if just_released:
        missing = []
        if not bb_breakout:
            missing.append(f"突破上軌 {upper_bb[i]:.1f}（現 {closes[i]:.1f}）")
        if not adx_ok:
            missing.append(f"ADX {adx[i]:.0f}<18")
        if not vol_ok:
            missing.append(f"量比 {vol_ratio:.1f}x<1.4x")
        if not mom_ok:
            missing.append(f"動量 {mom5*100:.1f}%")
        return "near", [f"  · 脫離擠壓，差：{', '.join(missing)}"]

    if currently_squeezing:
        dist = (upper_bb[i] / closes[i] - 1) * 100 if upper_bb[i] else 0
        return "near", [f"  · 擠壓中第{sq_count[i]}天，距上軌 {dist:.1f}%  ADX {adx[i]:.0f}"]

    return None, []


def _intraday_vol_fraction() -> float:
    """盤中已過交易時間比例（9:00=0.0, 13:30=1.0），用於修正量縮門檻"""
    now = datetime.now()
    elapsed = (now.hour * 60 + now.minute) - 9 * 60  # 距開盤分鐘數
    return min(max(elapsed / 270, 0.05), 1.0)  # 270 = 13:30-9:00，至少 5% 避免除以零


def _check_oversold(prices: list[dict]) -> tuple[str | None, list[str]]:
    closes = [p["close"] for p in prices]
    opens = [p["open"] for p in prices]
    vols = [p["volume"] for p in prices]
    sma60 = calc_sma(closes, 60)
    k, _ = calc_kd(prices, 9)
    vol_ma = calc_sma([float(v) for v in vols], 20)
    i = len(prices) - 1

    if any(v is None for v in [sma60[i], k[i], vol_ma[i]]) or vol_ma[i] == 0:
        return None, []

    above_ma = closes[i] > sma60[i]
    if not above_ma:
        return None, []

    drop5 = (closes[i - 5] - closes[i - 1]) / closes[i - 5] if closes[i - 5] > 0 else 0
    kd_ok = k[i] < 30
    is_red = closes[i] > opens[i]
    # 量縮門檻依盤中已過時間比例調整：若只過了 45% 交易時間，門檻也只用 45%
    t_frac = _intraday_vol_fraction()
    vol_threshold = vol_ma[i] * 0.8 * t_frac
    vol_shrink = vols[i] < vol_threshold
    vol_ratio = vols[i] / vol_ma[i]
    # 顯示用：預估全天量 = 今日累積 / 已過時間比例
    proj_ratio = vol_ratio / t_frac
    vol_max_张 = round(vol_ma[i] * 0.8 / 1000)  # 收盤門檻（顯示用）

    if drop5 < 0.03 and k[i] >= 40:
        return None, []

    conditions = {
        "5日跌5%": drop5 >= 0.05,
        "短線超賣": kd_ok,
        "紅K": is_red,
        "量縮": vol_shrink,
    }
    met = sum(conditions.values())

    if met == 4:
        return "triggered", [
            f"  ✓ 60MA之上  5日跌{drop5*100:.1f}%  KD{k[i]:.0f}",
            f"  ✓ 紅K（{closes[i]:.1f}>{opens[i]:.1f}）  量估{proj_ratio:.2f}x（需<{vol_max_张:,}張）",
        ]

    if met >= 2:
        lines = []
        lines.append(f"  {'✓' if drop5>=0.05 else '·'} 5日跌{drop5*100:.1f}%  "
                     f"{'✓' if kd_ok else '·'} KD{k[i]:.0f}  "
                     f"{'✓' if is_red else '·'} {'紅' if is_red else '黑'}K  "
                     f"{'✓' if vol_shrink else '·'} 量估{proj_ratio:.2f}x（需<{vol_max_张:,}張）")
        return "near", lines

    return None, []


def _check_ad(prices: list[dict]) -> tuple[str | None, list[str]]:
    closes = [p["close"] for p in prices]
    sma60 = calc_sma(closes, 60)
    rsi14 = calc_rsi(closes, 14)
    ad = calc_ad_line(prices)
    i = len(prices) - 1

    if any(v is None for v in [sma60[i], rsi14[i]]) or i < 25:
        return None, []

    above_ma = closes[i] > sma60[i]
    ma_rising = sma60[i - 5] is not None and sma60[i] > sma60[i - 5]
    if not (above_ma and ma_rising):
        return None, []

    price_min20 = min(closes[i - 20:i])
    price_dist = (closes[i] / price_min20 - 1) if price_min20 > 0 else 999
    price_at_low = closes[i] <= price_min20

    if price_dist > 0.02 and not price_at_low:
        return None, []
    if rsi14[i] >= 50:
        return None, []

    ad_min20 = min(ad[i - 20:i])
    ad_not_low = ad[i] > ad_min20
    rsi_ok = rsi14[i] < 45

    conditions = {"20日新低": price_at_low, "量價背離": ad_not_low, "動能轉弱": rsi_ok}
    met = sum(conditions.values())

    if met == 3:
        return "triggered", [
            f"  ✓ 60MA上升  20日新低 {closes[i]:.1f}",
            f"  ✓ AD線未新低  RSI{rsi14[i]:.0f}<45",
        ]

    if met >= 1:
        parts = []
        parts.append(f"{'✓' if price_at_low else '·'} 新低（差{price_dist*100:.1f}%）")
        parts.append(f"{'✓' if ad_not_low else '·'} AD背離")
        parts.append(f"{'✓' if rsi_ok else '·'} RSI{rsi14[i]:.0f}")
        return "near", [f"  {', '.join(parts)}"]

    return None, []


# ─────────────────────────────────────────────────────────────

def main():
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M")

    if not _is_market_hours():
        print(f"[{time_str}] 非交易時間，略過")
        return

    token = _load_token()

    # 1. 股票清單
    names = {s["stock_id"]: s["stock_name"]
             for s in json.loads(INDIVIDUAL_STOCKS_PATH.read_text())}
    raw = json.loads(FILTERED_LISTS_PATH.read_text())["strategies"]
    filtered = {
        "squeeze":  {s["stock_id"] for s in raw["squeeze"]["kept"]},
        "oversold": {s["stock_id"] for s in raw["oversold"]["kept"]},
        "ad":       {s["stock_id"] for s in raw["ad_divergence"]["kept"]},
    }
    all_ids = filtered["squeeze"] | filtered["oversold"] | filtered["ad"]

    # 2. FinMind 即時快照（1 API call，全市場）
    snapshot = _fetch_market_snapshot(token)
    if not snapshot:
        print(f"[{time_str}] 快照失敗，略過")
        return

    triggered_all = []
    near_all = []

    # 3. 逐檔掃描
    for sid in sorted(all_ids):
        snap = snapshot.get(sid)
        if not snap:
            continue
        history = _load_history(sid)
        if not history:
            continue

        # 避免今日資料重複（若 CSV 已含今日）
        if history and history[-1]["date"] == today:
            history = history[:-1]

        today_row = {
            "date": today,
            "open": snap["open"],
            "high": snap["high"],
            "low": snap["low"],
            "close": snap["close"],
            "volume": snap["volume_张"] * 1000,
        }
        prices = history + [today_row]
        name = names.get(sid, "")

        for strategy_key, label, checker in [
            ("squeeze",  "波動率擠壓", _check_squeeze),
            ("oversold", "超跌反彈",   _check_oversold),
            ("ad",       "AD背離",     _check_ad),
        ]:
            if sid not in filtered[strategy_key]:
                continue
            status, lines = checker(prices)
            if not status:
                continue
            entry = {
                "stock_id": sid, "name": name,
                "strategy_label": label,
                "close": snap["close"], "open": snap["open"],
                "volume_张": snap["volume_张"],
                "lines": lines,
            }
            if status == "triggered":
                triggered_all.append(entry)
            else:
                near_all.append(entry)

    from notify import send

    print(f"\n{'='*60}")
    print(f"  盤中監測  {today} {time_str}  掃描 {len(all_ids)} 檔")
    print(f"{'='*60}")

    if triggered_all:
        print(f"\n  🔔 已觸發（{len(triggered_all)} 檔）")
        for e in triggered_all:
            print(f"\n  ★ {e['stock_id']} {e['name']}【{e['strategy_label']}】"
                  f"現價 {e['close']:.1f}  量 {e['volume_张']:,}張")
            for line in e["lines"]:
                print(line)
            detail = "\n".join(l.strip() for l in e["lines"])
            send(
                f"🔔 {e['stock_id']} {e['name']}【{e['strategy_label']}】\n"
                f"現價 {e['close']:.1f}  量 {e['volume_张']:,}張\n{detail}"
            )
    else:
        print(f"\n  目前無股票觸發")

    if near_all:
        print(f"\n  觀察中（{len(near_all)} 檔）")
        for e in near_all:
            print(f"\n  {e['stock_id']} {e['name']}【{e['strategy_label']}】"
                  f"現價 {e['close']:.1f}  量 {e['volume_张']:,}張")
            for line in e["lines"]:
                print(line)

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
