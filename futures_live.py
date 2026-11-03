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

★ 選用功能（帳本 config 設定，預設關閉＝行為與舊版完全相同；回測依據見 strategies/tsmc_grid_income.md）：
  - trail_tp（移動停利，0＝關）：漲到 entry×(1+take) 後不賣、改追蹤最高價（lot["peak"]），
    從高點回落 trail_tp 才賣。例 "trail_tp": 0.012。
  - lev_gate（槓桿隨跌幅放大，null＝關）：買進用的槓桿上限＝lo +（max_leverage − lo）× 跌幅 ÷ full_drop，
    跌幅＝1 − 現價 ÷ 近一年最高價（2500 根 30 分 K + 今日高），最多到 max_leverage。
    例 "lev_gate": {"lo": 2.0, "full_drop": 0.30}＝平常 2x，距一年高點跌 30% 才放到 5x。
    只擋新買單，不會因為上限變低而賣出既有部位。
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
HIGH_1Y_BARS = 2500              # lev_gate 的「近一年高」：250 交易日 × 10 根 30 分 K


def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def tg(msg):
    notify.send(f"🤖【{LABEL}自動】{msg}")


# ─────────── 帳本 ───────────

def default_ledger():
    # config 沿用提醒版 QFF 的設定（30萬/2%/3x/固定停利）
    cfg = {"capital": 300000, "step": 0.012, "take": 0.012,
           "max_leverage": 5.0, "intraday_once": True,
           "trail_tp": 0.0, "lev_gate": None}          # 選用功能預設關閉（見檔頭）
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


# ─────────── 選用：移動停利 / 槓桿隨跌幅放大 ───────────

def trail_on(c):
    return float(c.get("trail_tp") or 0) > 0


def update_trail(l, price):
    """移動停利開啟時：漲到 entry×(1+take) 的口標 sell_armed，之後記錄最高價 peak（會寫進帳本）。"""
    c = l["config"]
    if not trail_on(c):
        return
    for x in l["lots"]:
        if price >= x["entry"] * (1 + c["take"]):
            x["sell_armed"] = True
        if x.get("sell_armed"):
            x["peak"] = max(float(x.get("peak") or price), price)


def tp_lots(l, price):
    """達出場條件的口。固定停利：price ≥ entry×(1+take)；移動停利：已啟動且 price ≤ peak×(1−trail_tp)。
    移動停利要先呼叫 update_trail(l, price)。"""
    c = l["config"]
    if trail_on(c):
        t = float(c["trail_tp"])
        return [x for x in l["lots"]
                if x.get("sell_armed") and price <= float(x.get("peak") or price) * (1 - t)]
    return [x for x in l["lots"] if price >= x["entry"] * (1 + c["take"])]


def trail_tag(lot):
    """賣出訊息用：移動停利出場時標註高點；固定停利回傳空字串（訊息與舊版相同）。"""
    return f"｜移動停利：高點{lot['peak']:.0f}回落" if lot.get("sell_armed") and lot.get("peak") else ""


def lev_cap(c, price, hi_1y):
    """買進用的槓桿上限 → (上限, 距一年高跌幅或 None)。lev_gate 關閉＝max_leverage。"""
    hi = float(c["max_leverage"])
    g = c.get("lev_gate")
    if not g:
        return hi, None
    lo = min(float(g["lo"]), hi)
    fd = float(g["full_drop"])
    drop = max(0.0, 1 - price / hi_1y) if hi_1y else 0.0   # 沒有一年高資料 → 當作沒跌（保守用 lo）
    return min(hi, lo + (hi - lo) * drop / fd), drop


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


def place_and_confirm(api, contract, action, price, octype, market=False):
    """下單，輪詢至成交或逾時撤單。market=True 用市價(轉倉確保成交)。回傳 'filled'/'cancelled'。"""
    import shioaji as sj
    if market:
        order = api.Order(action=action, price=0, quantity=1,
                          price_type="MKP", order_type="IOC", octype=octype,
                          account=api.futopt_account)
    else:
        order = api.Order(action=action, price=price, quantity=1,
                          price_type="LMT", order_type="ROD", octype=octype,
                          account=api.futopt_account)
    trade = api.place_order(contract, order)
    tg(f"下單 {action} {octype} 1口 @{price:.0f}（{'市價' if market else '限價'}）")
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


# ─────────── 轉倉（跨月）───────────

ROLL_DAYS_BEFORE = 2             # 近月結算日 ≤ 此天數(日曆天) → 觸發轉倉

def month_contracts(api):
    """每個結算月挑成交量最大的合約，回傳 [(month, contract), ...] 依月份排序。
    處理同月雙碼(如 QFFI6/QFFR1)——挑量大的那個才是真正流動的近月。"""
    from collections import defaultdict
    g = defaultdict(list)
    for c in getattr(api.Contracts.Futures, SYMBOL):
        g[c.delivery_month].append(c)
    out = []
    for m in sorted(g):
        cs = g[m]
        if len(cs) > 1:
            snaps = api.snapshots(cs)
            cs = [max(zip(cs, snaps),
                      key=lambda z: float(getattr(z[1], "total_volume", 0) or 0))[0]]
        out.append((m, cs[0]))
    return out


def active_contract(api, ledger):
    """依帳本 active_month 選合約；沒設就用最近月並寫回。回傳 (contract, months)。"""
    months = month_contracts(api)
    am = ledger.get("active_month")
    for m, c in months:
        if m == am:
            return c, months
    ledger["active_month"] = months[0][0]          # 初始化/校正為最近月
    return months[0][1], months


def days_to_settle(contract):
    from datetime import date
    try:
        return (date.fromisoformat(str(contract.delivery_date)) - date.today()).days
    except Exception:
        return 99


def roll_position(api, ledger, near_c, next_c, dry=False):
    """把近月部位轉到次月：先平近月、再開次月，帳本 entry 加價差。回傳 True/False。"""
    n = net_lots(ledger)
    near_px = snapshot_price(api, near_c)
    next_px = snapshot_price(api, next_c)
    if not near_px or not next_px:
        tg("⛔ 轉倉中止：取不到近月/次月即時價"); return False
    spread = next_px - near_px
    head = (f"{near_c.delivery_month}→{next_c.delivery_month}｜{n}口｜"
            f"近月{near_px:.0f} 次月{next_px:.0f} 價差{spread:+.0f}")
    if dry:
        print(f"[DRY 轉倉預覽] {head}")
        for lot in ledger["lots"]:
            print(f"  進場 {lot['entry']:.0f} → 調整後 {lot['entry']+spread:.0f}"
                  f"（停利點 {(lot['entry']+spread)*(1+ledger['config']['take']):.0f}）")
        print(f"  將：市價平近月 {n}口、市價開次月 {n}口、active_month→{next_c.delivery_month}")
        return True
    tg(f"🔄 開始轉倉 {head}（市價：先平近月、再開次月）")
    # ① 先平近月（先平最安全：萬一次月開失敗頂多變空手）
    #    市價 MKP/IOC：避免限價平到一半卡住（部分平倉→帳本/券商對不上）
    for i in range(n):
        if place_and_confirm(api, near_c, "Sell", near_px, "Cover", market=True) != "filled":
            tg(f"⛔ 轉倉：平近月第{i+1}口失敗（已平 {i} 口、帳本未動）→ 停止，請人工檢查"); return False
    # ② 再開次月（市價 MKP/IOC：近月已全平，次月必須確實補回、不能像限價追不到留缺口）
    filled = 0
    for _ in range(n):
        if place_and_confirm(api, next_c, "Buy", next_px, "New", market=True) == "filled":
            filled += 1
        else:
            tg(f"⚠️ 轉倉：開次月第{filled+1}口失敗（近月已全平）→ 現缺 {n-filled} 口，請人工補")
            break
    # ③ 帳本：只保留成功開的口，entry 加價差，切換月份
    ledger["lots"] = ledger["lots"][:filled]
    for lot in ledger["lots"]:
        lot["entry"] = lot["entry"] + spread
        if lot.get("peak"):                         # 移動停利追蹤中的高點也要一起移到次月尺度
            lot["peak"] = lot["peak"] + spread
    ledger["active_month"] = next_c.delivery_month
    tg(f"✅ 轉倉完成 → {next_c.delivery_month}，持倉 {filled} 口，entry 已 {spread:+.0f} 調整"
       f"（停利點同步移動）")
    return True


def cmd_roll(dry=False):
    """手動轉倉（dry=預覽不下單）。"""
    l = load_ledger()
    if net_lots(l) == 0:
        # 空手：直接把 active_month 切到最近月即可（不需下單）
        api, _ = sj_connect(need_ca=not dry)
        try:
            months = month_contracts(api)
            l["active_month"] = months[0][0]
            save_ledger(l)
            print(f"空手，active_month 設為最近月 {months[0][0]}（無需下單）")
        finally:
            api.logout()
        return
    api, _ = sj_connect(need_ca=not dry)
    try:
        near_c, months = active_contract(api, l)
        nxt = next((c for m, c in months if m > l["active_month"]), None)
        if nxt is None:
            print("找不到次月合約"); return
        print(f"近月 {near_c.delivery_month}（結算 {near_c.delivery_date}，剩 {days_to_settle(near_c)} 天）"
              f" → 次月 {nxt.delivery_month}")
        ok = roll_position(api, l, near_c, nxt, dry=dry)
        if ok and not dry:
            save_ledger(l)
    finally:
        api.logout()


# ─────────── 決策（與提醒版同邏輯，但用真實保證金/口數）───────────

def decide(l, price, ref, avail_margin, equity, timing_ok=True, dedup=True, lev_max=None, gate_drop=None):
    """回傳單一動作 dict 或 None。優先賣（停利），其次買。timing_ok=False 時不買（一天限次用）。
    dedup=False（提醒模式用）：關閉『每格只提醒一次』，只要在買點之下每輪都回報。
    lev_max：買進用槓桿上限（lev_gate 算好傳入；None＝config max_leverage）。gate_drop 只用於訊息。
    移動停利開啟時，呼叫前要先 update_trail(l, price)。"""
    c = l["config"]
    # 賣：固定停利＝任一口漲到 entry×(1+take)；移動停利＝已啟動且自高點回落 trail_tp → 平該口（成本最高的先平）
    tp = tp_lots(l, price)
    if tp:
        return {"action": "SELL", "lot": max(tp, key=lambda x: x["entry"]), "price": price}
    # 買：跌破近10日高×(1-step)、±step內無持倉、口數上限、防重複、當日時機
    buy_level = ref * (1 - c["step"])
    near = any(abs(x["entry"] / price - 1) < c["step"] for x in l["lots"])
    lots_ok = net_lots(l) < HARD_MAX_LOTS
    last_lv = l["alerts"].get("last_buy_level")
    fresh = (not dedup) or last_lv is None or price <= last_lv * (1 - c["step"]) or price >= last_lv * (1 + c["step"])
    if not (price <= buy_level and not near and lots_ok and fresh and timing_ok):
        return None
    # 兩道硬限：① 真實可用保證金夠 ② 買後真實槓桿 ≤ max_leverage（對真實權益）
    need_margin = price * MULT * INIT_MARGIN_RATE * MARGIN_BUFFER
    notional_after = (net_lots(l) + 1) * price * MULT
    if avail_margin < need_margin:
        return {"action": "BLOCK", "reason": "保證金不足",
                "detail": f"需~{need_margin:,.0f}、可用{avail_margin:,.0f}"}
    cap = float(c["max_leverage"]) if lev_max is None else lev_max
    if equity <= 0 or notional_after > equity * cap:
        lev = notional_after / max(equity, 1)
        gate = "" if gate_drop is None else f"；槓桿閘：距一年高 −{gate_drop:.1%} → 上限 {cap:.1f}x"
        return {"action": "BLOCK", "reason": "槓桿上限",
                "detail": f"再買一口→{lev:.1f}x 超過 {cap:.1f}x（權益{equity:,.0f}{gate}）"}
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
        l["alerts"] = {"date": today, "today_high": 0.0, "last_buy_level": None,
                       "bought_today": False}

    api = None
    try:
        api, _ = sj_connect(need_ca=live)          # 提醒模式不啟用憑證
        contract, months = active_contract(api, l)  # 依 active_month 選合約（支援轉倉後鎖次月）

        # ── 轉倉檢查：近月將結算 ──
        dts = days_to_settle(contract)
        if dts <= ROLL_DAYS_BEFORE:
            nxt = next((c for m, c in months if m > l["active_month"]), None)
            if nxt is not None:
                # 提醒模式：持倉就叫你手動轉，不自動
                if not live and net_lots(l) > 0:
                    tg(f"⚠️ 近月 {contract.delivery_month} 剩 {dts} 天結算、持倉 {net_lots(l)}口"
                       f" → 請跑 `futures_live.py roll` 轉倉（提醒模式不自動轉）")
                    save_ledger(l); return
                # ① 先讓「已達停利」的口正常停利出場（別花轉倉成本 carry 正要平的倉）
                #    移動停利開啟時：照移動停利規則（已啟動且自高點回落才賣），未回落的 runner 照常轉倉
                if live and net_lots(l) > 0:
                    near_px = snapshot_price(api, contract)
                    tp = []
                    if near_px:
                        update_trail(l, near_px)
                        tp = tp_lots(l, near_px)
                    for lot in tp:
                        if place_and_confirm(api, contract, "Sell", near_px, "Cover") == "filled":
                            pnl = (near_px - lot["entry"]) * MULT * lot["contracts"]
                            l["lots"].remove(lot); l["realized"] = l.get("realized", 0.0) + pnl
                            tg(f"✅ 轉倉日先停利 賣@{near_px:.0f}（平 {lot['entry']:.0f}那口 "
                               f"{pnl:+,.0f}元{trail_tag(lot)}）｜剩 {net_lots(l)}口")
                # ② 停利後若空手 → 直接切次月、本輪結束（次月買點下一根再判）
                if net_lots(l) == 0:
                    l["active_month"] = nxt.delivery_month
                    save_ledger(l)
                    log(f"轉倉日：空手 → active 切至 {nxt.delivery_month}")
                    return
                # ③ 還有沒到停利的口 → 轉倉到次月
                if roll_position(api, l, contract, nxt):
                    contract = nxt; save_ledger(l)
                else:
                    save_ledger(l); return

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
        # 選用功能：移動停利更新各口高點；槓桿閘依「距近一年高」算這輪的買進槓桿上限
        update_trail(l, price)
        cap, gate_drop = lev_cap(l["config"], price,
                                 max(local_recent_closes(HIGH_1Y_BARS) + [l["alerts"]["today_high"]]))
        mg = api.margin(api.futopt_account)
        avail = float(getattr(mg, "available_margin", 0.0))
        equity = float(getattr(mg, "equity_amount", avail))

        # 一天最多買 2 次：當天買過後，只剩「收盤前最後可成交那根 13:15」能再買
        # （13:45 是收盤瞬間、掛單來不及成交，故用 13:15）
        is_close_run = (hm == "13:15")
        bought_today = l["alerts"].get("bought_today", False)
        # 提醒模式：不限時機、不去重 → 只要在買點之下每輪都叫；下單模式維持一天2次+每格去重
        timing_ok = True if not live else (
            (not l["config"].get("intraday_once")) or (not bought_today) or is_close_run)
        act = decide(l, price, ref, avail, equity, timing_ok, dedup=live, lev_max=cap, gate_drop=gate_drop)
        if act is None:
            gate = "" if gate_drop is None else f" 槓桿閘−{gate_drop:.1%}→上限{cap:.1f}x"
            log(f"[{mode}] 現價{price:.0f} 近高{ref:.0f} 持倉{book}口 "
                f"權益{equity:,.0f} 可用{avail:,.0f}{gate} — 無動作")
            save_ledger(l); return
        if act["action"] == "BLOCK":
            tg(f"⚠️ 想買但{act['reason']}：{act['detail']}。（{mode}模式，此輪跳過）")
            l["alerts"]["last_buy_level"] = price       # 去重，避免每根重覆發
            save_ledger(l); return

        # ── 提醒模式：只發 Telegram 建議，不下單、不動帳本（由你手動回報）──
        if not live:
            if act["action"] == "BUY":
                drop = (ref - price) / ref * 100
                tg(f"🟢 買點！現價 {price:.0f}｜近10日高 {ref:.0f}（−{drop:.1f}%）｜持倉{book}口\n"
                   f"仍在買點，每根K持續提醒｜要接就下：futures_live.py buy {price:.0f}")
                # 提醒模式不設 last_buy_level / bought_today（decide dedup=False）→ 在買點之下每輪都叫。
            elif act["action"] == "SELL":
                lot = act["lot"]
                tg(f"🔴 建議賣出 1口 @~{price:.0f}（平進場@{lot['entry']:.0f}那口，"
                   f"{price-lot['entry']:+.0f}點{trail_tag(lot)}）\n成交後回報：futures_live.py sell {price:.0f}")
            log(f"[提醒] 發送建議 {act['action']}")
            save_ledger(l); return

        # ── 下單模式：真實 place/cancel ──
        if act["action"] == "BUY":
            r = place_and_confirm(api, contract, "Buy", price, "New")
            if r == "filled":
                l["lots"].append({"entry": price, "contracts": 1,
                                  "time": now.strftime("%Y-%m-%d %H:%M"), "sell_armed": False})
                l["alerts"]["last_buy_level"] = price
                l["alerts"]["bought_today"] = True
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
                tg(f"✅ 賣出成交 1口 @{price:.0f}（平 {lot['entry']:.0f}那口 {pnl:+,.0f}元{trail_tag(lot)}）"
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
    tp = tp_lots(l, sp)                         # 與提醒時同一套規則（固定停利 / 移動停利）
    if not tp and trail_on(c):                  # 移動停利：成交價跟提醒價略有出入時，退而找已啟動的口
        tp = [x for x in l["lots"] if x.get("sell_armed")]
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
    print(f"  設定：本金{c['capital']:,}/格{c['step']:.1%}/停利+{c['take']:.1%}/上限{c['max_leverage']}x")
    t = float(c.get("trail_tp") or 0)
    print(f"  移動停利：{'開，自高點回落 %.1f%% 賣' % (t * 100) if t > 0 else '關（固定停利）'}")
    g = c.get("lev_gate")
    if g:
        closes = local_recent_closes(HIGH_1Y_BARS)
        cap, drop = lev_cap(c, closes[-1], max(closes)) if closes else (float(g["lo"]), None)
        now = f"｜以最後一根收盤算：距一年高 −{drop:.1%} → 上限 {cap:.1f}x" if drop is not None else ""
        print(f"  槓桿閘：開，平常 {g['lo']}x，距一年高跌 {float(g['full_drop']):.0%} 放大到 {c['max_leverage']}x{now}")
    else:
        print("  槓桿閘：關（固定上限）")
    print(f"  持倉 {net_lots(l)}口｜已實現 {l.get('realized', 0):+,.0f}")
    for x in sorted(l["lots"], key=lambda z: z["entry"]):
        if t > 0 and x.get("sell_armed") and x.get("peak"):
            exit_ = f"移動停利中：高點 {x['peak']:.0f} → 跌破 {x['peak']*(1-t):.0f} 賣"
        elif t > 0:
            exit_ = f"漲到 {x['entry']*(1+c['take']):.0f} 啟動移動停利"
        else:
            exit_ = f"停利賣點 {x['entry']*(1+c['take']):.0f}"
        print(f"    進場@{x['entry']:.0f} ×{x['contracts']}  →{exit_}")


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
    elif a[0] == "roll":
        cmd_roll(dry=("--dry" in a or "dry" in a))
    else:
        print(f"未知指令：{a[0]}（可用 buy/sell/status/roll [--dry]，或無參數＝跑一輪）")


if __name__ == "__main__":
    main()
