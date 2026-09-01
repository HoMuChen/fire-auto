"""
QFF 小型台積電 網格 —— 真實下單執行器（Phase 1，預設關閉，需明確開啟）

與純提醒版 (futures_grid_monitor.py) 完全分離。這支會真的下單，務必先讀懂再開。

★ 安全機制（每一項都是刻意的，不要拿掉）：
  1. Opt-in 旗標：必須存在 data/LIVE_QFF_ENABLED 檔才會下單；不在＝只看不下。
  2. Kill switch：存在 data/AUTO_OFF 檔 → 立即全停（緊急煞車）。
  3. 真實部位對帳：每輪先 api.list_positions() 讀券商真實口數，跟本地帳本比對；
     對不上（你手動下過單/漏成交/部分成交）→ 發 Telegram + 停機，絕不在不一致狀態下單。
  4. 真實保證金檢查：下買單前 api.margin() 查可用保證金，不足就不下（比 config 的 30萬更硬）。
  5. 口數硬上限 HARD_MAX_LOTS：防程式暴走的絕對天花板，跟策略/保證金無關。
  6. 每輪最多下 1 張單：限制單輪風險。
  7. 限價 + 未成交取消：LMT 掛當下價，LMT_TIMEOUT_S 秒內沒成交就撤單。
  8. 每一筆（下單/成交/撤單/錯誤）都 Telegram 回報。
  9. 出任何例外 → 停機並通知，不盲目重試。

帳本：data/live_ledger_QFF.json（逐口 entry 價，驅動 +take 賣出；與提醒版 state 分開）。
Cron（上線後才加；記得把 QFF 的『提醒』cron 停掉避免雙頭）：
  15,45 9-13 * * 1-5  /Users/mu/fire-auto/.venv/bin/python /Users/mu/fire-auto/futures_live.py >> data/futures_live.log 2>&1

環境變數（.env.local）：SJ_API_KEY / SJ_SEC_KEY / SJ_CA_PATH / SJ_CA_PASSWD / SJ_PERSON_ID（下單需憑證，身分證字號）
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import futures_grid_monitor as gm
import notify

BASE = Path(__file__).parent
SYMBOL = "QFF"
LABEL = "小型台積電"
LEDGER_PATH = BASE / "data" / "live_ledger_QFF.json"
ENABLE_FLAG = BASE / "data" / "LIVE_QFF_ENABLED"     # 必須存在才下單
KILL_FLAG = BASE / "data" / "AUTO_OFF"               # 存在即全停
LOCAL_BARS = BASE / "data" / "futures" / "QFF_30min_day.csv"

MULT = gm.MULTIPLIER
H = 100                          # 近10日高（100根30分K）
SESSION_START, SESSION_END = "08:45", "13:45"
HARD_MAX_LOTS = 12               # 絕對口數天花板（防暴走；正常由 3x 保證金閘先限住）
LMT_TIMEOUT_S = 180              # 限價未成交 → 撤單（3分鐘）
POLL_EVERY_S = 10
INIT_MARGIN_RATE = 0.135         # 原始保證金估（買單前的粗略門檻，實際以 api.margin 可用數為準）
MARGIN_BUFFER = 1.10             # 下單後可用保證金要留 10% 緩衝


def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def tg(msg):
    notify.send(f"🤖【{LABEL}自動】{msg}")


# ─────────── 帳本 ───────────

def default_ledger():
    # config 沿用提醒版 QFF 的設定（30萬/2%/3x/固定停利）
    cfg = {"capital": 300000, "step": 0.02, "take": 0.02,
           "max_leverage": 3.0}
    return {"config": cfg, "lots": [], "realized": 0.0,
            "alerts": {"date": "", "today_high": 0.0, "last_buy_level": None}}


def load_ledger():
    if LEDGER_PATH.exists():
        return json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    return default_ledger()


def save_ledger(l):
    LEDGER_PATH.write_text(json.dumps(l, ensure_ascii=False, indent=2), encoding="utf-8")


def net_lots(l):
    return sum(x["contracts"] for x in l["lots"])


# ─────────── 參考價 ───────────

def local_recent_closes(n):
    import csv
    if not LOCAL_BARS.exists():
        return []
    rows = list(csv.DictReader(open(LOCAL_BARS, encoding="utf-8")))
    return [float(r["close"]) for r in rows[-n:]]


# ─────────── shioaji ───────────

def sj_connect():
    """登入 + 啟用憑證（下單必須）。回傳 (api, contract)。"""
    import shioaji as sj
    env = gm.load_sj_env()
    for k in ("SJ_API_KEY", "SJ_SEC_KEY", "SJ_CA_PATH", "SJ_CA_PASSWD", "SJ_PERSON_ID"):
        if not env.get(k):
            raise RuntimeError(f"缺少環境變數 {k}（下單需要，請補進 .env.local）")
    api = sj.Shioaji()
    api.login(env["SJ_API_KEY"], env["SJ_SEC_KEY"])
    time.sleep(3)
    ok = api.activate_ca(ca_path=env["SJ_CA_PATH"], ca_passwd=env["SJ_CA_PASSWD"],
                         person_id=env["SJ_PERSON_ID"])
    if not ok:
        raise RuntimeError("憑證啟用失敗 activate_ca=False")
    fut = getattr(api.Contracts.Futures, SYMBOL)
    contract = sorted([c for c in fut], key=lambda c: c.delivery_month)[0]
    return api, contract


def broker_net_position(api, contract):
    """券商真實近月淨口數（多為正、空為負）。"""
    net = 0
    for p in api.list_positions(api.futopt_account):
        if getattr(p, "code", "") == contract.code:
            q = int(p.quantity)
            d = str(getattr(p, "direction", "")).lower()
            net += q if "buy" in d or d in ("", "long") else -q
    return net


def available_margin(api):
    m = api.margin(api.futopt_account)
    return float(getattr(m, "available_margin", 0.0))


def snapshot_price(api, contract):
    s = api.snapshots([contract])[0]
    return float(s.close) if s and s.close else None


def place_and_confirm(api, contract, action, price, octype):
    """下限價單，輪詢至成交或逾時撤單。回傳 'filled' / 'cancelled' / 'error'。"""
    import shioaji as sj
    order = api.Order(action=action, price=price, quantity=1,
                      price_type="LMT", order_type="ROD", octype=octype,
                      account=api.futopt_account)
    trade = api.place_order(contract, order)
    tg(f"下單 {action} {octype} 1口 @{price:.0f}(限價)")
    log(f"place_order {action} {octype} @{price} -> {getattr(trade.status,'status','?')}")
    waited = 0
    while waited < LMT_TIMEOUT_S:
        time.sleep(POLL_EVERY_S); waited += POLL_EVERY_S
        api.update_status(api.futopt_account)
        st = str(trade.status.status)
        if "Filled" in st:
            return "filled"
        if st in ("Cancelled", "Failed", "Inactive"):
            return "cancelled"
    # 逾時未成交 → 撤單
    try:
        api.cancel_order(trade); api.update_status(api.futopt_account)
    except Exception as e:
        log(f"cancel_order 失敗: {e}")
    return "cancelled"


# ─────────── 決策（與提醒版同邏輯，但用真實保證金/口數）───────────

def decide(l, price, ref, avail_margin):
    """回傳單一動作 dict 或 None。優先賣（停利），其次買。"""
    c = l["config"]
    # 賣：任一口漲到 entry×(1+take) → 平該口（成本最高的先平，貼近停利）
    tp = [x for x in l["lots"] if price >= x["entry"] * (1 + c["take"])]
    if tp:
        lot = max(tp, key=lambda x: x["entry"])
        return {"action": "SELL", "lot": lot, "price": price}
    # 買：跌破近10日高×(1-step)、±step內無持倉、口數/保證金夠、防重複
    buy_level = ref * (1 - c["step"])
    near = any(abs(x["entry"] / price - 1) < c["step"] for x in l["lots"])
    lots_ok = net_lots(l) < HARD_MAX_LOTS
    need_margin = price * MULT * INIT_MARGIN_RATE * MARGIN_BUFFER
    margin_ok = avail_margin >= need_margin
    last_lv = l["alerts"].get("last_buy_level")
    fresh = last_lv is None or price <= last_lv * (1 - c["step"]) or price >= last_lv * (1 + c["step"])
    if price <= buy_level and not near and lots_ok and margin_ok and fresh:
        return {"action": "BUY", "price": price, "margin_ok": True}
    if not margin_ok and price <= buy_level and not near and lots_ok:
        return {"action": "BLOCK_MARGIN", "need": need_margin, "have": avail_margin}
    return None


# ─────────── 主流程 ───────────

def run():
    now = datetime.now()
    hm = now.strftime("%H:%M")

    # (1) 交易時段
    if now.weekday() >= 5 or not (SESSION_START <= hm <= SESSION_END):
        log("非日盤時段，跳過"); return
    # (2) kill switch
    if KILL_FLAG.exists():
        log("AUTO_OFF 存在 → 全停"); return
    # (1) opt-in
    if not ENABLE_FLAG.exists():
        log(f"未開啟（缺 {ENABLE_FLAG.name}）→ 只看不下"); return

    l = load_ledger()
    today = now.strftime("%Y-%m-%d")
    if l["alerts"].get("date") != today:
        l["alerts"] = {"date": today, "today_high": 0.0, "last_buy_level": None}

    api = None
    try:
        api, contract = sj_connect()

        # (3) 真實部位對帳
        broker = broker_net_position(api, contract)
        book = net_lots(l)
        if broker != book:
            save_ledger(l)
            tg(f"⛔ 對帳不符：券商 {broker} 口 vs 帳本 {book} 口 → 停機。"
               f"請人工檢查（可能手動下過單/部分成交）。修正後再開。")
            log(f"RECONCILE MISMATCH broker={broker} book={book} -> HALT")
            return

        price = snapshot_price(api, contract)
        if not price or price <= 0:
            log("取不到即時價"); save_ledger(l); return
        l["alerts"]["today_high"] = max(l["alerts"].get("today_high", 0.0), price)
        closes = local_recent_closes(H)
        ref = max(closes + [l["alerts"]["today_high"]]) if closes else l["alerts"]["today_high"]
        avail = available_margin(api)

        # (4)(5)(6) 決策：每輪最多一個動作
        act = decide(l, price, ref, avail)
        if act is None:
            log(f"現價{price:.0f} 近高{ref:.0f} 持倉{book}口 可用保證金{avail:,.0f} — 無動作")
            save_ledger(l); return

        if act["action"] == "BLOCK_MARGIN":
            tg(f"⚠️ 想買但保證金不足：需 ~{act['need']:,.0f}、可用 {act['have']:,.0f}。"
               f"（入金或此輪跳過）")
            save_ledger(l); return

        if act["action"] == "BUY":
            r = place_and_confirm(api, contract, "Buy", price, "New")
            if r == "filled":
                l["lots"].append({"entry": price, "contracts": 1,
                                  "time": now.strftime("%Y-%m-%d %H:%M"), "sell_armed": False})
                l["alerts"]["last_buy_level"] = price
                tg(f"✅ 買進成交 1口 @{price:.0f}｜持倉 {net_lots(l)}口")
            else:
                tg(f"↩️ 買單未成交已撤（@{price:.0f}）")
            save_ledger(l); return

        if act["action"] == "SELL":
            lot = act["lot"]
            r = place_and_confirm(api, contract, "Sell", price, "Cover")
            if r == "filled":
                pnl = (price - lot["entry"]) * MULT * lot["contracts"]
                l["lots"].remove(lot)
                l["realized"] = l.get("realized", 0.0) + pnl
                tg(f"✅ 賣出成交 1口 @{price:.0f}（平 {lot['entry']:.0f}那口 {pnl:+,.0f}元）"
                   f"｜持倉 {net_lots(l)}口｜累計已實現 {l['realized']:+,.0f}")
            else:
                tg(f"↩️ 賣單未成交已撤（@{price:.0f}）")
            save_ledger(l); return

    except Exception as e:
        log(f"例外 → 停機：{e}")
        tg(f"⛔ 執行例外，本輪停止：{e}")
    finally:
        if api is not None:
            try:
                api.logout()
            except Exception:
                pass


if __name__ == "__main__":
    run()
