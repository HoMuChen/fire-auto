"""盤中訂單簿錄製器（永豐 Shioaji，只看行情、不啟用憑證、不下單）。

錄什麼：QFF（小型台積電期）最近兩個月、TXF（台指期）近月、2330 現貨的
  - tick：逐筆成交（價、量、內外盤 tick_type 1=外盤買 / 2=內盤賣）
  - bidask：五檔（每次變動一筆，含五檔價量）
存哪：data/orderbook/{YYYY-MM-DD}/{kind}_{code}.csv（盤中每 30 秒 append），收盤後 `pack` 轉 parquet。

用法（需 .venv/bin/python）：
  .venv/bin/python research/orderbook_recorder.py record            # 錄到 13:50 自動結束（夜盤時段則錄到 --until）
  .venv/bin/python research/orderbook_recorder.py record --until 23:59 --only TXF   # 測試用
  .venv/bin/python research/orderbook_recorder.py backfill --days 60 # 用 api.ticks 回補 2330 與現存 QFF/TXF 合約的歷史逐筆
  .venv/bin/python research/orderbook_recorder.py pack              # 把 CSV 轉 parquet（已轉的略過）
  .venv/bin/python research/orderbook_recorder.py status            # 各日檔案筆數

建議 cron（平日 08:40 啟動，程式 13:50 自行結束；由使用者自行加入）：
  40 8 * * 1-5 cd /Users/mu/fire-auto && .venv/bin/python research/orderbook_recorder.py record >> data/orderbook_recorder.log 2>&1 && .venv/bin/python research/orderbook_recorder.py pack >> data/orderbook_recorder.log 2>&1
"""
import argparse
import sys
import threading
import time
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data" / "orderbook"
FLUSH_SEC = 30

_buf = defaultdict(list)
_lock = threading.Lock()


def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def connect():
    import shioaji as sj
    import futures_grid_monitor as gm  # 與 futures_live.py 相同的 .env.local 讀法
    env = gm.load_sj_env()
    api = sj.Shioaji()
    api.login(env["SJ_API_KEY"], env["SJ_SEC_KEY"])   # 不 activate_ca：純行情
    time.sleep(3)
    return api


def front_contracts(api, symbol, n):
    """每個結算月挑成交量最大的合約（處理同月雙碼），回傳最近 n 個月。"""
    g = defaultdict(list)
    for c in getattr(api.Contracts.Futures, symbol):
        if c.delivery_month:
            g[c.delivery_month].append(c)
    out = []
    for m in sorted(g)[:n]:
        cs = g[m]
        if len(cs) > 1:
            snaps = api.snapshots(cs)
            cs = [max(zip(cs, snaps), key=lambda z: float(getattr(z[1], "total_volume", 0) or 0))[0]]
        out.append(cs[0])
    return out


def targets(api, only=None):
    t = {"QFF": front_contracts(api, "QFF", 2), "TXF": front_contracts(api, "TXF", 1),
         "2330": [api.Contracts.Stocks["2330"]]}
    if only:
        t = {k: v for k, v in t.items() if k in only}
    return [c for v in t.values() for c in v]


def _row(obj):
    d = obj.to_dict() if hasattr(obj, "to_dict") else dict(obj.__dict__)
    out = {}
    for k, v in d.items():
        if isinstance(v, (list, tuple)):
            for i, x in enumerate(v[:5], 1):
                out[f"{k}{i}"] = float(x) if hasattr(x, "__float__") else x
        elif hasattr(v, "__float__") and not isinstance(v, (int, bool)):
            out[k] = float(v)
        else:
            out[k] = v
    out["recv"] = datetime.now().isoformat(timespec="microseconds")
    return out


def _push(kind, obj):
    try:
        r = _row(obj)
        with _lock:
            _buf[(kind, str(r.get("code", "")))].append(r)
    except Exception as e:  # 回呼不可丟例外
        log(f"callback error {kind}: {e}")


def flush():
    with _lock:
        items = list(_buf.items()); _buf.clear()
    day = OUT / f"{date.today():%Y-%m-%d}"
    day.mkdir(parents=True, exist_ok=True)
    n = 0
    for (kind, code), rows in items:
        if not rows:
            continue
        f = day / f"{kind}_{code}.csv"
        pd.DataFrame(rows).to_csv(f, mode="a", header=not f.exists(), index=False)
        n += len(rows)
    return n


def _session(stop, only):
    """登入訂閱直到 stop 或斷線；回傳寫入筆數與是否斷線。"""
    import shioaji as sj
    api = connect()
    api.set_on_tick_fop_v1_callback(lambda ex, t: _push("tick", t))
    api.set_on_bidask_fop_v1_callback(lambda ex, b: _push("bidask", b))
    api.set_on_tick_stk_v1_callback(lambda ex, t: _push("tick", t))
    api.set_on_bidask_stk_v1_callback(lambda ex, b: _push("bidask", b))
    down = threading.Event()
    api.set_session_down_callback(lambda: down.set())
    cs = targets(api, only)
    for c in cs:
        for qt in (sj.QuoteType.Tick, sj.QuoteType.BidAsk):
            api.subscribe(c, quote_type=qt, version=sj.QuoteVersion.v1)
    log("訂閱：" + ", ".join(c.code for c in cs) + f"；錄到 {stop:%H:%M}")
    total, last = 0, time.time()
    try:
        while datetime.now() < stop and not down.is_set():
            time.sleep(1)
            if time.time() - last >= FLUSH_SEC:
                k = flush(); total += k; last = time.time()
                if k == 0:
                    log("30 秒內無資料")
    finally:
        total += flush()
        try:
            api.logout()
        except Exception:
            pass
    return total, down.is_set()


def record(until="13:50", only=None):
    stop = datetime.combine(date.today(), datetime.strptime(until, "%H:%M").time())
    if datetime.now() >= stop:
        log(f"已過結束時間 {until}，不錄"); return
    total, tries = 0, 0
    while datetime.now() < stop and tries < 20:
        try:
            k, dropped = _session(stop, only)
            total += k
            if not dropped:
                break
            log("session down → 10 秒後重新登入")
        except Exception as e:
            log(f"錯誤：{e} → 30 秒後重試")
            time.sleep(20)
        tries += 1
        time.sleep(10)
    log(f"結束，共寫入 {total:,} 筆")


def backfill(days=60, only=None):
    """api.ticks 回補：每個交易日一檔一檔抓；已存在的日檔略過。每筆含當下買一/賣一與內外盤。
    期貨的「交易日」含前一晚夜盤（15:00/17:25 起）；休市前一晚的夜盤會遺失（API 歸到休市日），可接受。
    只抓得到目前仍掛牌的期貨合約；已到期合約要靠每日 record 累積。"""
    api = connect()
    cs = targets(api, only)
    # 交易日：本地加權指數日資料（data/v2/futures/taiex.parquet）＋最近 30 天以 2330 K 棒補齊。
    # 休市日 api.ticks 會回傳最近一次夜盤的重複資料，不可只看週末；一次查多年 kbars 會卡住，只查近期。
    t0 = date.today() - timedelta(days=days)
    tx = pd.read_parquet(ROOT / "data" / "v2" / "futures" / "taiex.parquet")
    tx = tx[(tx.close > 0) & (tx.Trading_Volume > 0)]
    tset = set(pd.to_datetime(tx.date).dt.strftime("%Y-%m-%d"))
    kb = pd.DataFrame({**api.kbars(contract=api.Contracts.Stocks["2330"],
                                   start=f"{max(t0, date.today() - timedelta(days=30)):%Y-%m-%d}",
                                   end=f"{date.today() - timedelta(days=1):%Y-%m-%d}")})
    if len(kb):
        tset |= set(pd.to_datetime(kb.ts).dt.strftime("%Y-%m-%d"))
    tdays = sorted(d for d in tset if f"{t0:%Y-%m-%d}" <= d < f"{date.today():%Y-%m-%d}")
    log(f"交易日 {len(tdays)} 天：{tdays[0] if tdays else '-'} ~ {tdays[-1] if tdays else '-'}")
    for ds in tdays:
        for c in cs:
            f = OUT / "hist" / f"{ds}_{c.code}.parquet"
            if f.exists():
                continue
            try:
                t = api.ticks(contract=c, date=ds)
            except Exception as e:
                log(f"{ds} {c.code} 失敗：{e}"); continue
            df = pd.DataFrame({**t})
            if df.empty or "ts" not in df:
                continue
            df["ts"] = pd.to_datetime(df.ts)
            f.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(f)
            log(f"{ds} {c.code} {len(df):,} 筆")
            time.sleep(0.3)
    try:
        log(f"usage: {api.usage()}")
    except Exception:
        pass
    api.logout()


def pack():
    for day in sorted(OUT.glob("20*")):
        for f in day.glob("*.csv"):
            p = f.with_suffix(".parquet")
            if p.exists() and p.stat().st_mtime >= f.stat().st_mtime:
                continue
            df = pd.read_csv(f, low_memory=False)
            df.to_parquet(p)
            log(f"pack {p.relative_to(ROOT)} {len(df):,} 筆")


def status():
    for day in sorted(OUT.glob("20*")):
        print(day.name, {f.name: sum(1 for _ in open(f)) - 1 for f in sorted(day.glob("*.csv"))})
    h = sorted((OUT / "hist").glob("*.parquet"))
    if h:
        print(f"hist：{len(h)} 檔，{h[0].stem} ~ {h[-1].stem}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["record", "backfill", "pack", "status"])
    ap.add_argument("--until", default="13:50")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--days", type=int, default=60)
    a = ap.parse_args()
    {"record": lambda: record(a.until, a.only), "backfill": lambda: backfill(a.days, a.only),
     "pack": pack, "status": status}[a.cmd]()
