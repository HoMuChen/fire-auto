"""
小型台積電期貨網格 — 盤中即時買賣點通知（純提醒版，狀態由使用者回報）

設計：有狀態的部位管理器（非無狀態掃描）。狀態存 data/futures_grid_state.json，
每 30 分 K 收盤跑一次，比對「當前價 / 近10日高 / 你手上的批」，有買賣點就發 Telegram。

用法：
  python3 futures_grid_monitor.py init                 # 初始化狀態（用預設設定）
  python3 futures_grid_monitor.py                       # 監控一次（cron 每30分跑）
  python3 futures_grid_monitor.py buy 2400 [口數]       # 回報：我買進了 1 口 @2400
  python3 futures_grid_monitor.py sell 2400            # 回報：我把進場@2400那口賣了
  python3 futures_grid_monitor.py status               # 看目前狀態/下一格買賣位

Cron（對齊30分K收盤，日盤平日）：
  15,45 9-13 * * 1-5  cd /Users/mu/fire-auto && /usr/local/bin/python3 futures_grid_monitor.py >> data/futures_monitor.log 2>&1
"""
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "research"))
import futures_cache as fc          # reuse fetch_ticks / bars_for_day / token（在 research/）
import notify

BASE = Path(__file__).parent
STATE_PATH = BASE / "data" / "futures_grid_state.json"
LOCAL_BARS = BASE / "data" / "futures" / "QFF_30min_day.csv"
MULTIPLIER = 100                    # 1 口 = 100 股；1 點 = NT$100
H = 100                            # 近10交易日高（100 根 30 分 K）
SESSION_START, SESSION_END = "08:45", "13:45"

DEFAULT_CONFIG = {
    "capital": 500000, "step": 0.007, "take": 0.007, "max_lots": 50,
    "max_leverage": 3.0, "derisk_lev": 3.5, "trail_tp": 0.0,
}


# ─────────── 狀態 ───────────

def load_state():
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return None


def save_state(s):
    STATE_PATH.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")


def cmd_init():
    s = {"config": dict(DEFAULT_CONFIG), "lots": [], "realized": 0.0,
         "alerts": {"date": "", "last_buy_level": None}}
    save_state(s)
    c = s["config"]
    print(f"已初始化：本金{c['capital']:,} / 格{c['step']:.1%} / 停利+{c['take']:.1%} / "
          f"買進上限{c['max_leverage']}x / derisk{c['derisk_lev']}x")


# ─────────── 部位計算 ───────────

def notional(lots, price):
    return sum(price * MULTIPLIER * l["contracts"] for l in lots)


def unreal(lots, price):
    return sum((price - l["entry"]) * MULTIPLIER * l["contracts"] for l in lots)


def equity(s, price):
    return s["config"]["capital"] + s.get("realized", 0.0) + unreal(s["lots"], price)


def leverage(s, price):
    eq = equity(s, price)
    return notional(s["lots"], price) / eq if eq > 0 else float("inf")


# ─────────── 價格 / 參考 ───────────

def local_recent_closes(n):
    """本地 QFF 30分K 最後 n 根收盤（近月連續，raw）。"""
    if not LOCAL_BARS.exists():
        return []
    rows = list(csv.DictReader(open(LOCAL_BARS, encoding="utf-8")))
    return [float(r["close"]) for r in rows[-n:]]


def today_bars(tok):
    """抓今天日盤 tick → 近月連續 30 分 K（排除價差單）。回傳 list of bar dict。"""
    day = datetime.now().strftime("%Y-%m-%d")
    ticks = fc.fetch_ticks(day, tok)
    return fc.bars_for_day(ticks)


def current_price_and_ref(tok):
    """回傳 (最後一根收盤價, 近10日高, 是否上漲bar, 最後bar時間) 或 None。"""
    tb = today_bars(tok)
    if not tb:
        return None
    last = tb[-1]
    price = last["close"]
    is_up = last["close"] >= last["open"]
    # 近10日高 = 本地(到昨天) + 今天，取最後 H 根收盤的最大
    closes = local_recent_closes(H) + [b["close"] for b in tb]
    ref = max(closes[-H:]) if closes else price
    return price, ref, is_up, last["datetime"]


# ─────────── 信號引擎（與回測同邏輯：固定停利/無加碼版）───────────

def signals(s, price, ref, is_up):
    c = s["config"]
    out = []
    # 賣：每口漲到 +take 就提醒（各口獨立），用 sell_armed 去重（回落後重置）
    for lot in s["lots"]:
        hit = price >= lot["entry"] * (1 + c["take"])
        if hit and not lot.get("sell_armed"):
            gain_pt = price - lot["entry"]
            out.append(("賣", f"進場@{lot['entry']:.0f} 那口 → 賣 @~{price:.0f}"
                              f"（+{gain_pt:.0f}點 ≈ +{gain_pt*MULTIPLIER*lot['contracts']:,.0f}元）"))
            lot["sell_armed"] = True
        elif not hit and lot.get("sell_armed"):
            lot["sell_armed"] = False          # 回落，解除警戒（之後再破再提醒）
    # 買：跌破近10日高×(1-step)、±step 內無持倉、批數/槓桿夠
    buy_level = ref * (1 - c["step"])
    near = any(abs(l["entry"] / price - 1) < c["step"] for l in s["lots"])
    lots_ok = len(s["lots"]) < c["max_lots"]
    lev_ok = notional(s["lots"], price) + price * MULTIPLIER <= equity(s, price) * c["max_leverage"]
    last_lv = s["alerts"].get("last_buy_level")
    fresh = last_lv is None or price <= last_lv * (1 - c["step"]) or price >= last_lv * (1 + c["step"])
    if price <= buy_level and not near and lots_ok and lev_ok and fresh:
        out.append(("買", f"跌破近10日高({ref:.0f})的{c['step']:.1%} → 買 @~{price:.0f}"
                          f"（下一格參考 {buy_level:.0f}）"))
        s["alerts"]["last_buy_level"] = price
    elif not (price <= buy_level and not near and lots_ok):
        s["alerts"]["last_buy_level"] = None if not near else last_lv
    # 減碼：槓桿 > derisk 且反彈
    lev = leverage(s, price)
    if lev > c["derisk_lev"] and is_up and s["lots"]:
        lowest = min(s["lots"], key=lambda l: l["entry"])
        out.append(("減碼", f"槓桿{lev:.1f}x > {c['derisk_lev']}x 且反彈 → "
                            f"賣最低成本那口(進場@{lowest['entry']:.0f})降槓桿"))
    return out


# ─────────── 指令 ───────────

def cmd_monitor():
    s = load_state()
    if s is None:
        print("尚未初始化，先跑 init"); return
    now = datetime.now()
    hm = now.strftime("%H:%M")
    if now.weekday() >= 5 or not (SESSION_START <= hm <= SESSION_END):
        print(f"[{now:%Y-%m-%d %H:%M}] 非日盤時段，跳過"); return
    # 換日重置買進提醒
    today = now.strftime("%Y-%m-%d")
    if s["alerts"].get("date") != today:
        s["alerts"] = {"date": today, "last_buy_level": None}
    tok = fc.token()
    res = current_price_and_ref(tok)
    if res is None:
        print(f"[{hm}] 尚無今日 K 棒"); save_state(s); return
    price, ref, is_up, bar_dt = res
    sig = signals(s, price, ref, is_up)
    lev = leverage(s, price)
    if sig:
        lines = [f"⚡ 台積電期貨網格訊號  {bar_dt}", f"現價 {price:.0f}｜近10日高 {ref:.0f}｜"
                 f"持倉 {len(s['lots'])}批/{sum(l['contracts'] for l in s['lots'])}口｜槓桿 {lev:.1f}x", ""]
        for typ, detail in sig:
            icon = {"買": "🟢 建議買進", "賣": "🔴 建議賣出", "減碼": "⚠️ 建議減碼"}[typ]
            lines.append(f"{icon}：{detail}")
        lines += ["", "（成交後回報：buy <價> / sell <進場價>）"]
        msg = "\n".join(lines)
        notify.send(msg, intraday=True)
        print(f"[{hm}] 發送 {len(sig)} 個訊號")
    else:
        print(f"[{hm}] 現價{price:.0f} 近高{ref:.0f} 槓桿{lev:.1f}x — 無訊號")
    save_state(s)


def cmd_buy(price, contracts=1):
    s = load_state()
    s["lots"].append({"entry": float(price), "contracts": int(contracts),
                      "time": datetime.now().strftime("%Y-%m-%d %H:%M"), "sell_armed": False})
    save_state(s)
    print(f"已記錄買進：{contracts}口 @{price}｜目前 {len(s['lots'])}批/"
          f"{sum(l['contracts'] for l in s['lots'])}口")


def cmd_sell(entry):
    s = load_state()
    entry = float(entry)
    match = min(s["lots"], key=lambda l: abs(l["entry"] - entry), default=None)
    if match is None:
        print("目前無持倉"); return
    s["lots"].remove(match)
    save_state(s)
    print(f"已移除進場@{match['entry']:.0f} 那口｜剩 {len(s['lots'])}批")


def cmd_status():
    s = load_state()
    if s is None:
        print("尚未初始化"); return
    c = s["config"]
    print(f"設定：本金{c['capital']:,} / 格{c['step']:.1%} / 停利+{c['take']:.1%} / "
          f"買進上限{c['max_leverage']}x / derisk{c['derisk_lev']}x")
    print(f"持倉 {len(s['lots'])}批 / {sum(l['contracts'] for l in s['lots'])}口"
          f"｜已實現 {s.get('realized',0):+,.0f}")
    for l in sorted(s["lots"], key=lambda x: x["entry"]):
        print(f"  進場@{l['entry']:.0f} × {l['contracts']}口  ({l.get('time','')})"
              f"  →停利賣點 {l['entry']*(1+c['take']):.0f}")


def main():
    if len(sys.argv) < 2:
        cmd_monitor(); return
    cmd = sys.argv[1]
    if cmd == "init":
        cmd_init()
    elif cmd == "buy":
        cmd_buy(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else 1)
    elif cmd == "sell":
        cmd_sell(sys.argv[2])
    elif cmd == "status":
        cmd_status()
    else:
        print(f"未知指令：{cmd}")


if __name__ == "__main__":
    main()
