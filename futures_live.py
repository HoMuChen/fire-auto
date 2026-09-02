"""
QFF 小型台積電 網格 —— 真實下單執行器（Phase 1，預設關閉，需明確開啟）

與純提醒版 (futures_grid_monitor.py) 完全分離。這支會真的下單，務必先讀懂再開。

★ 兩種模式（由旗標切換）：
  - 提醒模式（無 data/LIVE_QFF_ENABLED）：連線、算訊號、發 Telegram 建議，但【不下單】，
    帳本由你手動 `buy/sell` 回報維持。＝等同純提醒版，關掉旗標也不會失聯。
  - 下單模式（有 data/LIVE_QFF_ENABLED）：真實 place/cancel，帳本由成交自動更新。

★ 安全機制（每一項都是刻意的，不要拿掉）：
  1. Opt-in 旗標：必須存在 data/LIVE_QFF_ENABLED 才進入【下單模式】；不在＝提醒模式。
  2. Kill switch：存在 data/AUTO_OFF 檔 → 立即全停（連提醒都不發，緊急煞車）。
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
    cfg = {"capital": 300000, "step": 0.015, "take": 0.015,
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

def sj_connect(need_ca=True):
    """登入；下單模式(need_ca=True)才啟用憑證。回傳 (api, contract)。"""
    import shioaji as sj
    env = gm.load_sj_env()
    req = ["SJ_API_KEY", "SJ_SEC_KEY"] + (["SJ_CA_PATH", "SJ_CA_PASSWD"] if need_ca else [])
    for k in req:
        if not env.get(k):
            raise RuntimeError(f"缺少環境變數 {k}（請補進 .env.local）")
    api = sj.Shioaji()
    api.login(env["SJ_API_KEY"], env["SJ_SEC_KEY"])
    time.sleep(3)
    if need_ca:
        # 憑證密碼即身分證字號 → person_id 沿用 SJ_CA_PASSWD（可用 SJ_PERSON_ID 覆寫）
        person_id = env.get("SJ_PERSON_ID") or env["SJ_CA_PASSWD"]
        ok = api.activate_ca(ca_path=env["SJ_CA_PATH"], ca_passwd=env["SJ_CA_PASSWD"],
                             person_id=person_id)
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
        st = str(trade.status.status)      # 例："OrderStatus.Filled"（含前綴，用子字串比對）
        if "Filled" in st:
            return "filled"
        if any(x in st for x in ("Cancelled", "Failed", "Inactive")):
            return "cancelled"
    # 逾時未成交 → 撤單
    try:
        api.cancel_order(trade); api.update_status(api.futopt_account)
    except Exception as e:
        log(f"cancel_order 失敗: {e}")
    return "cancelled"


# ─────────── 決策（與提醒版同邏輯，但用真實保證金/口數）───────────

def decide(l, price, ref, avail_margin, equity):
    """回傳單一動作 dict 或 None。優先賣（停利），其次買。"""
    c = l["config"]
    # 賣：任一口漲到 entry×(1+take) → 平該口（成本最高的先平，貼近停利）
    tp = [x for x in l["lots"] if price >= x["entry"] * (1 + c["take"])]
    if tp:
        return {"action": "SELL", "lot": max(tp, key=lambda x: x["entry"]), "price": price}
    # 買：跌破近10日高×(1-step)、±step內無持倉、口數上限、防重複
    buy_level = ref * (1 - c["step"])
    near = any(abs(x["entry"] / price - 1) < c["step"] for x in l["lots"])
    lots_ok = net_lots(l) < HARD_MAX_LOTS
    last_lv = l["alerts"].get("last_buy_level")
    fresh = last_lv is None or price <= last_lv * (1 - c["step"]) or price >= last_lv * (1 + c["step"])
    if not (price <= buy_level and not near and lots_ok and fresh):
        return None
    # 兩道硬限：① 真實可用保證金夠 ② 買後真實槓桿 ≤ max_leverage（對真實權益）
    need_margin = price * MULT * INIT_MARGIN_RATE * MARGIN_BUFFER
    notional_after = (net_lots(l) + 1) * price * MULT
    if avail_margin < need_margin:
        return {"action": "BLOCK", "reason": "保證金不足",
                "detail": f"需~{need_margin:,.0f}、可用{avail_margin:,.0f}"}
    if equity <= 0 or notional_after > equity * c["max_leverage"]:
        lev = notional_after / max(equity, 1)
        return {"action": "BLOCK", "reason": "槓桿上限",
                "detail": f"再買一口→{lev:.1f}x 超過 {c['max_leverage']}x（權益{equity:,.0f}）"}
    return {"action": "BUY", "price": price}


# ─────────── 主流程 ───────────

def run():
    now = datetime.now()
    hm = now.strftime("%H:%M")

    # 交易時段
    if now.weekday() >= 5 or not (SESSION_START <= hm <= SESSION_END):
        log("非日盤時段，跳過"); return
    # kill switch：連提醒都停
    if KILL_FLAG.exists():
        log("AUTO_OFF 存在 → 全停（連提醒都不發）"); return

    live = ENABLE_FLAG.exists()          # 有旗標＝下單模式；無＝提醒模式
    mode = "下單" if live else "提醒"

    l = load_ledger()
    today = now.strftime("%Y-%m-%d")
    if l["alerts"].get("date") != today:
        l["alerts"] = {"date": today, "today_high": 0.0, "last_buy_level": None}

    api = None
    try:
        api, contract = sj_connect(need_ca=live)   # 提醒模式不啟用憑證

        # 對帳：下單模式不符即停機；提醒模式只警告（你手動 buy/sell 回報維持帳本）
        broker = broker_net_position(api, contract)
        book = net_lots(l)
        if broker != book:
            if live:
                save_ledger(l)
                tg(f"⛔ 對帳不符：券商 {broker} 口 vs 帳本 {book} 口 → 停機。"
                   f"請人工檢查（可能手動下過單/部分成交），修正後再開。")
                log(f"RECONCILE MISMATCH broker={broker} book={book} -> HALT")
                return
            tg(f"⚠️【提醒模式】對帳不符：券商 {broker} 口 vs 帳本 {book} 口。"
               f"成交後記得 `futures_live.py buy/sell` 回報，帳本才會準。")

        price = snapshot_price(api, contract)
        if not price or price <= 0:
            log("取不到即時價"); save_ledger(l); return
        l["alerts"]["today_high"] = max(l["alerts"].get("today_high", 0.0), price)
        closes = local_recent_closes(H)
        ref = max(closes + [l["alerts"]["today_high"]]) if closes else l["alerts"]["today_high"]
        mg = api.margin(api.futopt_account)
        avail = float(getattr(mg, "available_margin", 0.0))
        equity = float(getattr(mg, "equity_amount", avail))

        act = decide(l, price, ref, avail, equity)      # 每輪最多一個動作
        if act is None:
            log(f"[{mode}] 現價{price:.0f} 近高{ref:.0f} 持倉{book}口 "
                f"權益{equity:,.0f} 可用{avail:,.0f} — 無動作")
            save_ledger(l); return
        if act["action"] == "BLOCK":
            tg(f"⚠️ 想買但{act['reason']}：{act['detail']}。（{mode}模式，此輪跳過）")
            l["alerts"]["last_buy_level"] = price       # 去重，避免每根重覆發
            save_ledger(l); return

        # ── 提醒模式：只發 Telegram 建議，不下單、不動帳本（由你手動回報）──
        if not live:
            if act["action"] == "BUY":
                tg(f"🟢 建議買進 1口 @~{price:.0f}（跌破近10日高{ref:.0f}的"
                   f"{l['config']['step']:.0%}）｜持倉{book}口\n成交後回報：futures_live.py buy {price:.0f}")
                l["alerts"]["last_buy_level"] = price
            elif act["action"] == "SELL":
                lot = act["lot"]
                tg(f"🔴 建議賣出 1口 @~{price:.0f}（平進場@{lot['entry']:.0f}那口，"
                   f"+{price-lot['entry']:.0f}點）\n成交後回報：futures_live.py sell {price:.0f}")
            log(f"[提醒] 發送建議 {act['action']}")
            save_ledger(l); return

        # ── 下單模式：真實 place/cancel ──
        if act["action"] == "BUY":
            r = place_and_confirm(api, contract, "Buy", price, "New")
            if r == "filled":
                l["lots"].append({"entry": price, "contracts": 1,
                                  "time": now.strftime("%Y-%m-%d %H:%M"), "sell_armed": False})
                l["alerts"]["last_buy_level"] = price
                tg(f"✅ 買進成交 1口 @{price:.0f}｜持倉 {net_lots(l)}口")
            else:
                tg(f"↩️ 買單未成交已撤（@{price:.0f}）")
        elif act["action"] == "SELL":
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
        save_ledger(l)

    except Exception as e:
        log(f"例外 → 停機：{e}")
        tg(f"⛔ 執行例外，本輪停止：{e}")
    finally:
        if api is not None:
            try:
                api.logout()
            except Exception:
                pass


# ─────────── 手動回報指令（提醒模式維持帳本用）───────────

def cmd_buy(price, contracts=1):
    l = load_ledger()
    l["lots"].append({"entry": float(price), "contracts": int(contracts),
                      "time": datetime.now().strftime("%Y-%m-%d %H:%M"), "sell_armed": False})
    save_ledger(l)
    print(f"已記錄買進 {contracts}口 @{price}｜持倉 {net_lots(l)}口")


def cmd_sell(sell_price):
    l = load_ledger()
    if not l["lots"]:
        print("目前無持倉"); return
    sp = float(sell_price); c = l["config"]
    tp = [x for x in l["lots"] if sp >= x["entry"] * (1 + c["take"])]
    lot = max(tp, key=lambda x: x["entry"]) if tp else min(l["lots"], key=lambda x: abs(x["entry"] - sp))
    l["lots"].remove(lot)
    pnl = (sp - lot["entry"]) * MULT * lot["contracts"]
    l["realized"] = l.get("realized", 0.0) + pnl
    save_ledger(l)
    print(f"已平 進場@{lot['entry']:.0f}→賣@{sp:.0f}（{pnl:+,.0f}元）｜剩 {net_lots(l)}口"
          f"｜累計已實現 {l['realized']:+,.0f}")


def cmd_status():
    l = load_ledger(); c = l["config"]
    mode = "下單 🔴" if ENABLE_FLAG.exists() else "提醒 🟡"
    kill = "｜AUTO_OFF 全停中 ⛔" if KILL_FLAG.exists() else ""
    print(f"[QFF 下單器] 模式：{mode}{kill}")
    print(f"  設定：本金{c['capital']:,}/格{c['step']:.0%}/停利+{c['take']:.0%}/上限{c['max_leverage']}x")
    print(f"  持倉 {net_lots(l)}口｜已實現 {l.get('realized', 0):+,.0f}")
    for x in sorted(l["lots"], key=lambda z: z["entry"]):
        print(f"    進場@{x['entry']:.0f} ×{x['contracts']}  →停利賣點 {x['entry']*(1+c['take']):.0f}")


def main():
    a = sys.argv[1:]
    if not a:
        run()
    elif a[0] == "buy":
        cmd_buy(a[1], a[2] if len(a) > 2 else 1)
    elif a[0] == "sell":
        cmd_sell(a[1])
    elif a[0] == "status":
        cmd_status()
    else:
        print(f"未知指令：{a[0]}（可用 buy/sell/status，或無參數＝跑一輪）")


if __name__ == "__main__":
    main()
