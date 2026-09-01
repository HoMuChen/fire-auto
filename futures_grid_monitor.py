"""
小型期貨網格 — 盤中即時買賣點通知（純提醒版，狀態由使用者回報）

支援兩檔（--symbol 切換，預設 QFF）：
  QFF = 小型台積電期貨   SEF = 小型智邦期貨

設計：有狀態的部位管理器（非無狀態掃描）。各商品獨立狀態檔（QFF→data/futures_grid_state.json、
SEF→data/futures_grid_state_SEF.json），每 30 分 K 收盤跑一次，比對「當前價 / 近10日高 /
你手上的批」，有買賣點就發 Telegram。

用法（加 --symbol SEF 即操作小型智邦，省略＝QFF 小型台積）：
  python3 futures_grid_monitor.py init [--symbol SEF]        # 初始化狀態（用預設設定）
  python3 futures_grid_monitor.py [--symbol SEF]              # 監控一次（cron 每30分跑）
  python3 futures_grid_monitor.py buy 2400 [口數] [--symbol SEF]   # 回報：我買進了 1 口 @2400
  python3 futures_grid_monitor.py sell 2421 [--symbol SEF]   # 回報：成交價 2421 賣一口（自動判斷平哪口）
  python3 futures_grid_monitor.py status [--symbol SEF]      # 看目前狀態/下一格買賣位

即時價來源：永豐 shioaji（.env.local 的 SJ_API_KEY/SJ_SEC_KEY），需用 .venv/bin/python 跑（含 shioaji）。
近10日高：本地 {SYMBOL}_30min_day.csv 最後100根收盤 + 今日盤中最高。

Cron（對齊30分K收盤，日盤平日；用 .venv 的 python）：
  15,45 9-13 * * 1-5  cd /Users/mu/fire-auto && .venv/bin/python futures_grid_monitor.py >> data/futures_monitor.log 2>&1
  # 要同時盤中跟小型智邦，再加一行 --symbol SEF：
  15,45 9-13 * * 1-5  cd /Users/mu/fire-auto && .venv/bin/python futures_grid_monitor.py --symbol SEF >> data/futures_monitor_sef.log 2>&1
"""
import csv
import json
import sys
from datetime import datetime
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import notify

BASE = Path(__file__).parent
MULTIPLIER = 100                    # 1 口 = 100 股；1 點 = NT$100（小型台積/小型智邦皆同）
H = 100                            # 近10交易日高（100 根 30 分 K）
SESSION_START, SESSION_END = "08:45", "13:45"

# 可切換商品：state 檔各自獨立（QFF 沿用舊路徑保留現有部位）
SYMBOLS = {
    "QFF": {"label": "小型台積電", "state": "futures_grid_state.json"},
    "SEF": {"label": "小型智邦",   "state": "futures_grid_state_SEF.json"},
}
SYMBOL = "QFF"
LABEL = SYMBOLS[SYMBOL]["label"]
STATE_PATH = BASE / "data" / SYMBOLS[SYMBOL]["state"]
LOCAL_BARS = BASE / "data" / "futures" / f"{SYMBOL}_30min_day.csv"


def configure(symbol):
    """依商品設定全域：狀態檔、本地K棒檔、shioaji合約碼、顯示名稱。"""
    global SYMBOL, LABEL, STATE_PATH, LOCAL_BARS
    symbol = symbol.upper()
    if symbol not in SYMBOLS:
        raise SystemExit(f"未知商品：{symbol}（可用 {'/'.join(SYMBOLS)}）")
    SYMBOL = symbol
    LABEL = SYMBOLS[symbol]["label"]
    STATE_PATH = BASE / "data" / SYMBOLS[symbol]["state"]
    LOCAL_BARS = BASE / "data" / "futures" / f"{symbol}_30min_day.csv"

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
    print(f"[{LABEL} {SYMBOL}] 已初始化：本金{c['capital']:,} / 格{c['step']:.1%} / 停利+{c['take']:.1%} / "
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


def load_sj_env():
    env = {}
    for line in (BASE / ".env.local").read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def sj_price():
    """用永豐 shioaji 取當前商品(SYMBOL)近月即時價（唯讀快照）。回傳 float 或 None。"""
    import shioaji as sj
    env = load_sj_env()
    api = sj.Shioaji()
    try:
        api.login(env["SJ_API_KEY"], env["SJ_SEC_KEY"])
        time.sleep(2)                              # 等合約載入
        fut = getattr(api.Contracts.Futures, SYMBOL)
        near = sorted([c for c in fut], key=lambda c: c.delivery_month)[0]
        snap = api.snapshots([near])[0]
        return float(snap.close) if snap and snap.close else None
    finally:
        try:
            api.logout()
        except Exception:
            pass


def current_price_and_ref(state):
    """shioaji 即時價 + 本地近10日高(含今日盤中高)。回傳 (price, ref, is_up) 或 None。"""
    price = sj_price()
    if not price or price <= 0:
        return None
    # 近10日高 = 本地30分K最後H根收盤，並納入今日盤中看到的最高
    today_high = max(state["alerts"].get("today_high", 0) or 0, price)
    state["alerts"]["today_high"] = today_high
    closes = local_recent_closes(H)
    ref = max(closes + [today_high]) if closes else today_high
    # 反彈判斷：現價 vs 上次現價
    last_price = state["alerts"].get("last_price")
    is_up = last_price is None or price >= last_price
    state["alerts"]["last_price"] = price
    return price, ref, is_up


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
    # 減碼：槓桿 > derisk 且反彈（derisk_lev 為 0/None 視為關閉）
    lev = leverage(s, price)
    if c.get("derisk_lev") and lev > c["derisk_lev"] and is_up and s["lots"]:
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
    # 換日重置（買進提醒、今日高、上次價）
    today = now.strftime("%Y-%m-%d")
    if s["alerts"].get("date") != today:
        s["alerts"] = {"date": today, "last_buy_level": None,
                       "today_high": 0, "last_price": None}
    try:
        res = current_price_and_ref(s)
    except Exception as e:
        print(f"[{hm}] shioaji 取價失敗：{e}"); save_state(s); return
    if res is None:
        print(f"[{hm}] 取不到即時價"); save_state(s); return
    price, ref, is_up = res
    sig = signals(s, price, ref, is_up)
    lev = leverage(s, price)
    if sig:
        lines = [f"⚡ {LABEL}期貨網格訊號  {now:%H:%M}", f"現價 {price:.0f}｜近10日高 {ref:.0f}｜"
                 f"持倉 {len(s['lots'])}批/{sum(l['contracts'] for l in s['lots'])}口｜槓桿 {lev:.1f}x", ""]
        for typ, detail in sig:
            icon = {"買": "🟢 建議買進", "賣": "🔴 建議賣出", "減碼": "⚠️ 建議減碼"}[typ]
            lines.append(f"{icon}：{detail}")
        lines += ["", "（成交後回報：buy <價> / sell <成交價>）"]
        msg = "\n".join(lines)
        notify.send(msg)          # 只送主收件人(早上八點那個 TELEGRAM_CHAT_ID)，不送 intraday
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


def cmd_sell(sell_price):
    """回報成交價，自動判斷平哪口：優先平「已達停利中成本最高」那口；否則平成本最接近的。"""
    s = load_state()
    if not s["lots"]:
        print("目前無持倉"); return
    sp = float(sell_price)
    c = s["config"]
    tp_hit = [l for l in s["lots"] if sp >= l["entry"] * (1 + c["take"])]
    lot = (max(tp_hit, key=lambda l: l["entry"]) if tp_hit
           else min(s["lots"], key=lambda l: abs(l["entry"] - sp)))
    s["lots"].remove(lot)
    pnl = (sp - lot["entry"]) * MULTIPLIER * lot["contracts"]
    s["realized"] = s.get("realized", 0.0) + pnl
    save_state(s)
    print(f"已平：進場@{lot['entry']:.0f} → 賣@{sp:.0f}（{pnl:+,.0f}元）｜剩 {len(s['lots'])}批"
          f"｜累計已實現 {s['realized']:+,.0f}")


def cmd_status():
    s = load_state()
    if s is None:
        print("尚未初始化"); return
    c = s["config"]
    print(f"[{LABEL} {SYMBOL}] 設定：本金{c['capital']:,} / 格{c['step']:.1%} / 停利+{c['take']:.1%} / "
          f"買進上限{c['max_leverage']}x / derisk{c['derisk_lev']}x")
    print(f"持倉 {len(s['lots'])}批 / {sum(l['contracts'] for l in s['lots'])}口"
          f"｜已實現 {s.get('realized',0):+,.0f}")
    for l in sorted(s["lots"], key=lambda x: x["entry"]):
        print(f"  進場@{l['entry']:.0f} × {l['contracts']}口  ({l.get('time','')})"
              f"  →停利賣點 {l['entry']*(1+c['take']):.0f}")


def main():
    args = list(sys.argv[1:])
    # 解析 --symbol（可放在任意位置），預設 QFF
    symbol = "QFF"
    if "--symbol" in args:
        i = args.index("--symbol")
        if i + 1 >= len(args):
            raise SystemExit("--symbol 後面要接商品代碼（QFF/SEF）")
        symbol = args[i + 1]
        del args[i:i + 2]
    configure(symbol)

    if not args:
        cmd_monitor(); return
    cmd = args[0]
    if cmd == "init":
        cmd_init()
    elif cmd == "buy":
        cmd_buy(args[1], args[2] if len(args) > 2 else 1)
    elif cmd == "sell":
        cmd_sell(args[1])
    elif cmd == "status":
        cmd_status()
    else:
        print(f"未知指令：{cmd}")


if __name__ == "__main__":
    main()
