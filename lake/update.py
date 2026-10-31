"""data/v2 研究資料每日增量更新（FinMind 依日期一次抓全市場，取代逐檔重抓）。

規則（2026-10-31 以 10/20 bulk vs 逐檔比對定下）：
  日資料 raw/adj/per/inst/margin/mv：從各資料集「全市場最後日期 − OVERLAP_DAYS」起逐個平日抓 bulk；
    該資料集當天有資料就以新值覆蓋同日舊值（FinMind 會事後修正，例如成交量）；股價當天 0 筆＝休市或尚未公布 → 跳過該日。
  adj：重疊日的新舊收盤不一致（除權息後整段重新還原）→ 該檔整段重抓 TaiwanStockPriceAdj。
  停牌後復牌：已存檔最後日早於視窗 → 逐檔補抓缺口。
  revenue 近 3 個月（逐月）、fin/bs/cf 最近 3 個季底、div 近 120 天、hold 近 21 天逐個平日（公布日不固定）。
  universe：新上市普通股加入並逐檔抓完整歷史；下市股標 delist_date / type=delisted（資料保留）。
  大盤 / 期權（futures/、taiex_tr）：從最後日期所在月 / 年重抓到今天、以日期覆蓋；TXO 月資料一次拆出全序列、週選、未平倉；
    期交所最後結算價抓今年；margin_x / raw_x（融資維持率的 ETF 等）跟著日資料 bulk 更新；
    自建 txo_vix / ad / breadth / mkt_maint 整段重算。
寫檔一律 tmp → rename（研究程式同時讀也不會讀到半個檔）；內容沒變不改寫。

用法：python3 lake/update.py [--dry-run] [--rate 60]
"""
import argparse
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from build_data_v2 import COLS, NON_STOCK, OUT as V2, RateLimiter, api  # noqa: E402

OVERLAP_DAYS = 10
PRICE_COLS = ["date", "open", "high", "low", "close", "volume", "money", "turnover"]
DAILY = {  # kind → (dataset, 同日多列時的鍵)
    "raw": ("TaiwanStockPrice", None),
    "adj": ("TaiwanStockPriceAdj", None),
    "per": ("TaiwanStockPER", None),
    "inst": ("TaiwanStockInstitutionalInvestorsBuySell", "name"),
    "margin": ("TaiwanStockMarginPurchaseShortSale", None),
    "mv": ("TaiwanStockMarketValue", None),
}
PERIODIC = {"revenue": "TaiwanStockMonthRevenue", "fin": "TaiwanStockFinancialStatements", "bs": "TaiwanStockBalanceSheet",
            "cf": "TaiwanStockCashFlowsStatement", "div": "TaiwanStockDividend", "hold": "TaiwanStockHoldingSharesPer"}
FULL_START = {"raw": "2010-01-01", "adj": "2010-01-01"}   # 其餘 2009-01-01（同 fetch_fundamentals）
DATASET = {**{k: v[0] for k, v in DAILY.items()}, **PERIODIC}


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# ─────────── 檔案 I/O ───────────
def read(kind, sid):
    f = V2 / kind / f"{sid}.parquet"
    try:
        return pd.read_parquet(f)
    except Exception:
        return pd.DataFrame()


def shape(kind, df):
    """FinMind 回傳 → v2 檔案格式（raw/adj 改欄名、date 轉 datetime、無 stock_id；其餘保留原欄位）。"""
    if kind in ("raw", "adj"):
        df = df.rename(columns=COLS)[PRICE_COLS].copy()
        df["date"] = pd.to_datetime(df.date)
    elif kind == "raw_x":                                    # 融資維持率用的上市非普通股（ETF 等）；同 maint.fetch 格式
        df = df.rename(columns=COLS)
    return df.reset_index(drop=True)


def _like(new, old):
    """新資料欄位順序與 dtype 對齊舊檔。"""
    if not len(old):
        return new
    for c in old.columns:
        if c not in new:
            new[c] = pd.NA
    new = new[list(old.columns)]
    for c in old.columns:
        try:
            new[c] = new[c].astype(old[c].dtype)
        except (TypeError, ValueError):
            pass
    return new


def save(kind, sid, df, old, dry):
    """內容沒變不寫；寫檔 tmp → rename。回傳是否有變動。"""
    if len(old) == len(df) and len(df) and df.reset_index(drop=True).equals(old.reset_index(drop=True)):
        return False
    if not dry:
        f = V2 / kind / f"{sid}.parquet"
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_name(f".{f.name}.tmp")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, f)
    return True


def upsert(kind, sid, new, drop_mask_fn, sort_keys, dry):
    """old 去掉 drop_mask_fn(old) 為真的列 → 接上 new → 排序。"""
    old = read(kind, sid)
    new = _like(shape(kind, new), old)
    keep = old[~drop_mask_fn(old)] if len(old) else old
    out = pd.concat([keep, new], ignore_index=True) if len(keep) else new
    keys = list(dict.fromkeys(k for k in sort_keys if k in out.columns))
    out = out.sort_values(keys, kind="stable").reset_index(drop=True)
    return save(kind, sid, out, old, dry)


def _dates(df):
    return pd.to_datetime(df["date"]).dt.normalize() if len(df) else pd.Series(dtype="datetime64[ns]")


# ─────────── universe ───────────
def update_universe(lim, dry):
    u = pd.read_csv(V2 / "universe.csv", dtype=str)
    info, dl = api(lim, dataset="TaiwanStockInfo"), api(lim, dataset="TaiwanStockDelisting")
    if info is None or dl is None or not len(info):
        raise RuntimeError("TaiwanStockInfo / Delisting 抓取失敗")
    is_stock = info.stock_id.str.fullmatch(r"[1-9]\d{3}") & ~info.industry_category.isin(NON_STOCK)
    cur = info[is_stock].copy()
    cur["rank"] = (cur.industry_category == "電子工業").astype(int)
    cur = cur.sort_values(["stock_id", "rank"]).drop_duplicates("stock_id")
    known = set(u.stock_id)
    new = cur[~cur.stock_id.isin(known)]
    add = pd.DataFrame({"stock_id": new.stock_id, "name": new.stock_name, "industry": new.industry_category,
                        "type": new.type, "delist_date": None})
    # 下市：出現在 Delisting（≥2010）且已不在現存清單
    dl = dl[dl.stock_id.str.fullmatch(r"[1-9]\d{3}") & (dl.date >= "2010-01-01")].sort_values("date").drop_duplicates("stock_id", keep="last")
    gone = dl[dl.stock_id.isin(u[u.delist_date.isna()].stock_id) & ~dl.stock_id.isin(cur.stock_id)]
    for sid, d in zip(gone.stock_id, gone.date):
        u.loc[u.stock_id == sid, ["type", "delist_date"]] = ["delisted", d]
    # 上市別變動（興櫃 → 上櫃 / 上市）
    moved = u.merge(cur[["stock_id", "type"]], on="stock_id", suffixes=("", "_now"))
    moved = moved[moved.delist_date.isna() & (moved.type != moved.type_now)]
    for sid, t in zip(moved.stock_id, moved.type_now):
        u.loc[u.stock_id == sid, "type"] = t
    out = pd.concat([u, add], ignore_index=True).sort_values("stock_id")
    if len(add) or len(gone) or len(moved):
        log(f"universe：新增 {len(add)} {list(add.stock_id)[:10]}、下市 {len(gone)} {list(gone.stock_id)}、上市別變動 {len(moved)}")
        if not dry:
            tmp = V2 / ".universe.csv.tmp"
            out.to_csv(tmp, index=False)
            os.replace(tmp, V2 / "universe.csv")
    return out, list(add.stock_id)


def fetch_full(lim, sid, kinds, dry):
    """新股 / 需重抓者：逐檔抓完整歷史（覆蓋）。"""
    for k in kinds:
        df = api(lim, dataset=DATASET[k], data_id=sid, start_date=FULL_START.get(k, "2009-01-01"))
        if df is None:
            log(f"  {sid} {k} 抓取失敗")
            continue
        df = shape(k, df) if len(df) else df
        if not dry:
            f = V2 / k / f"{sid}.parquet"
            tmp = f.with_name(f".{f.name}.tmp")
            df.to_parquet(tmp, index=False)
            os.replace(tmp, f)


# ─────────── 日資料 ───────────
def last_date(kind, ids):
    """全市場最後日期（現存股的最大值）。"""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    m = None
    for s in ids:
        f = V2 / kind / f"{s}.parquet"
        try:
            t = pq.read_table(f, columns=["date"])
        except Exception:
            continue
        if t.num_rows:
            v = pd.Timestamp(str(pc.max(t["date"]).as_py())[:10])
            m = v if m is None or v > m else m
    return m


def weekdays(a, b):
    d, out = a, []
    while d <= b:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def update_daily(lim, ids, active, dry, today):
    stats = {}
    start = min(last_date(k, active) for k in DAILY) - timedelta(days=OVERLAP_DAYS)
    days = weekdays(start.date(), today)
    log(f"日資料視窗 {days[0]} ~ {days[-1]}（{len(days)} 個平日）")
    bulk = {k: [] for k in DAILY}
    for d in days:
        ds = d.isoformat()
        raw = api(lim, dataset="TaiwanStockPrice", start_date=ds, end_date=ds)
        if raw is None or not len(raw):
            continue                                         # 休市或尚未公布
        bulk["raw"].append(raw)
        for k in DAILY:
            if k == "raw":
                continue
            df = api(lim, dataset=DAILY[k][0], start_date=ds, end_date=ds)
            if df is not None and len(df):
                bulk[k].append(df)
    readjust = []
    for k, src in (("raw_x", "raw"), ("margin_x", "margin")):              # mkt_maint 的分子要含 ETF 等
        have = {f.stem for f in (V2 / k).glob("*.parquet")}
        if not bulk[src] or not have:
            continue
        b = pd.concat(bulk[src], ignore_index=True)
        b = b[b.stock_id.isin(have)]
        n = sum(upsert(k, sid, g, lambda o, gd=set(pd.to_datetime(g.date)): _dates(o).isin(gd), ["date"], dry)
                for sid, g in b.groupby("stock_id"))
        stats[k] = f"{n} 檔變動"
    for k, (dsname, key) in DAILY.items():
        if not bulk[k]:
            stats[k] = "無新資料"
            continue
        b = pd.concat(bulk[k], ignore_index=True)
        b = b[b.stock_id.isin(ids)]
        got_days = sorted(pd.to_datetime(b.date).dt.normalize().unique())
        changed = gap = 0
        for sid, g in b.groupby("stock_id"):
            gd = set(pd.to_datetime(g.date).dt.normalize())
            old = read(k, sid)
            if len(old):
                od = _dates(old)
                if od.max() < min(gd) - pd.Timedelta(days=OVERLAP_DAYS + 5):   # 停牌後復牌 → 補缺口
                    fill = api(lim, dataset=dsname, data_id=sid, start_date=(od.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                               end_date=(min(gd) - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
                    if fill is not None and len(fill):
                        g = pd.concat([fill, g], ignore_index=True)
                        gd = set(pd.to_datetime(g.date).dt.normalize())
                        gap += 1
                if k == "adj":                                                  # 除權息 → 整段重新還原
                    s = shape(k, g).set_index("date").close
                    o = old.assign(date=od).set_index("date").close
                    ov = s.index.intersection(o.index)
                    if len(ov) and not np.allclose(s[ov].values, o[ov].values, rtol=1e-6):
                        readjust.append(sid)
                        continue
            g = g.drop(columns=["stock_id"]) if k in ("raw", "adj") else g
            changed += upsert(k, sid, g, lambda o, gd=gd: _dates(o).isin(gd), ["date", key or "date"], dry)
        stats[k] = f"{len(got_days)} 天 {got_days[0]:%m-%d}~{got_days[-1]:%m-%d}，{changed} 檔變動" + (f"，補缺口 {gap}" if gap else "")
    if readjust:
        log(f"adj 重新還原 {len(readjust)} 檔：{readjust[:15]}{' …' if len(readjust) > 15 else ''}")
        for sid in readjust:
            fetch_full(lim, sid, ["adj"], dry)
        stats["adj"] += f"，整段重抓 {len(readjust)}"
    return stats


# ─────────── 月 / 季 / 週資料 ───────────
def quarter_ends(today, n):
    q = pd.Timestamp(today).to_period("Q")
    return [(q - i).end_time.normalize() for i in range(1, n + 1)]


def _upsert_bulk(kind, b, ids, mask_fn, sort_keys, dry):
    b = b[b.stock_id.isin(ids)]
    n = 0
    for sid, g in b.groupby("stock_id"):
        n += upsert(kind, sid, g, lambda o, g=g: mask_fn(o, g), sort_keys, dry)
    return n


def update_periodic(lim, ids, dry, today):
    stats = {}
    # 月營收：bulk 只回 start_date 那個月（date＝公布月 1 日）→ 逐月查最近 3 個月；依 (revenue_year, revenue_month) 覆蓋
    def m(o, g):
        k = set(zip(g.revenue_year.astype(int), g.revenue_month.astype(int)))
        return pd.Series([(y, mo) in k for y, mo in zip(o.revenue_year.astype(int), o.revenue_month.astype(int))], index=o.index)
    n, got = 0, []
    for i in range(3):
        ms = (pd.Timestamp(today).to_period("M") - i).start_time.strftime("%Y-%m-%d")
        b = api(lim, dataset=PERIODIC["revenue"], start_date=ms, end_date=ms)
        if b is not None and len(b):
            got.append(f"{ms[:7]}:{b.stock_id.nunique()}")
            n += _upsert_bulk("revenue", b, ids, m, ["date"], dry)
    stats["revenue"] = f"{' '.join(got)}，{n} 檔變動"
    # 季報：最近 3 個季底（以季底日期覆蓋）。FinMind 最新一季常先給「年初累計」、之後改成單季
    # （2026-10-31 實測：v2 存的 2026Q2 有 997 檔是 H1 累計）→ 每天重抓才會跟到修正
    for k in ("fin", "bs", "cf"):
        n, qs = 0, []
        for q in quarter_ends(today, 3):
            b = api(lim, dataset=PERIODIC[k], start_date=q.strftime("%Y-%m-%d"), end_date=q.strftime("%Y-%m-%d"))
            if b is not None and len(b):
                qs.append(f"{q:%Y-%m}:{b.stock_id.nunique()}")
                n += _upsert_bulk(k, b, ids, lambda o, g: _dates(o).isin(set(_dates(g))), ["date", "type"], dry)
        stats[k] = f"{' '.join(qs)}，{n} 檔變動"
    # 股利：近 120 天，依 (date, year) 覆蓋
    b = api(lim, dataset=PERIODIC["div"], start_date=(today - timedelta(days=120)).isoformat(), end_date=today.isoformat())
    if b is not None and len(b):
        def m(o, g):
            k = set(zip(g.date, g.year))
            return pd.Series([(d, y) in k for d, y in zip(o.date, o.year)], index=o.index)
        stats["div"] = f"{_upsert_bulk('div', b, ids, m, ['date'], dry)} 檔變動"
    # 集保：公布日不固定、bulk 只能單日 → 固定回看 21 天逐個平日查（某天查詢失敗下次仍會補到）
    n, got = 0, []
    for d in weekdays(today - timedelta(days=21), today):
        b = api(lim, dataset=PERIODIC["hold"], start_date=d.isoformat(), end_date=d.isoformat())
        if b is not None and len(b):
            got.append(f"{d:%m-%d}")
            n += _upsert_bulk("hold", b, ids, lambda o, g: _dates(o).isin(set(_dates(g))), ["date", "HoldingSharesLevel"], dry)
    stats["hold"] = f"{','.join(got) or '無新資料'}，{n} 檔變動"
    return stats


# ─────────── 大盤 / 期權（data/v2/futures 與 taiex_tr）───────────
FD = V2 / "futures"
MARKET = {  # 檔 → (dataset, data_id, 抓取單位 M=月 / Y=年, 篩選)
    FD / "MTX_daily.parquet": ("TaiwanFuturesDaily", "MTX", "M", lambda d: d[d.contract_date.str.fullmatch(r"\d{6}")]),
    FD / "TX_large_traders.parquet": ("TaiwanFuturesOpenInterestLargeTraders", "TX", "M", lambda d: d[d.contract_type == "all"]),
    FD / "mkt_margin.parquet": ("TaiwanStockTotalMarginPurchaseShortSale", None, "Y", None),
    FD / "mkt_inst.parquet": ("TaiwanStockTotalInstitutionalInvestors", None, "Y", None),
    FD / "taiex.parquet": ("TaiwanStockPrice", "TAIEX", "Y", None),
    FD / "gspc.parquet": ("USStockPrice", "^GSPC", "Y", None),
    V2 / "taiex_tr.parquet": ("TaiwanStockTotalReturnIndex", "TAIEX", "Y", None),
}


def _periods(last, today, unit):
    """從 last 所在的月 / 年到今天，回傳 (start, end) 字串。"""
    last = pd.Timestamp(last)
    if unit == "Y":
        return [(f"{y}-01-01", f"{y}-12-31") for y in range(last.year, today.year + 1)]
    ms = pd.date_range(last.to_period("M").start_time, pd.Timestamp(today), freq="MS")
    return [(f"{m:%Y-%m-%d}", f"{m + pd.offsets.MonthEnd(0):%Y-%m-%d}") for m in ms]


def _merge_dates(path, new, dry, index_date=False):
    """以日期覆蓋：舊檔去掉 new 有的日期 → 接上 new → 依日期穩定排序。內容沒變不寫。"""
    old = pd.read_parquet(path)
    if index_date:
        old, new = old.reset_index(), new.reset_index()
    new = _like(new, old)
    nd = set(pd.to_datetime(new.date).dt.normalize())
    out = pd.concat([old[~_dates(old).isin(nd)], new], ignore_index=True)
    out = out.iloc[np.argsort(pd.to_datetime(out.date).values, kind="stable")].reset_index(drop=True)
    if index_date:
        old, out = old.set_index("date"), out.set_index("date")
    if len(out) == len(old) and out.equals(old):
        return 0
    if not dry:
        tmp = path.with_name(f".{path.name}.tmp")
        out.to_parquet(tmp)
        os.replace(tmp, path)
    return len(nd)


def update_market(lim, dry, today):
    stats = {}
    for path, (ds, did, unit, flt) in MARKET.items():
        old = pd.read_parquet(path)
        parts = []
        for a, b in _periods(_dates(old).max(), today, unit):
            kw = dict(dataset=ds, start_date=a, end_date=b, **({"data_id": did} if did else {}))
            d = api(lim, **kw)
            if d is not None and len(d):
                parts.append(flt(d) if flt else d)
        if parts:
            new = pd.concat(parts, ignore_index=True)
            n = _merge_dates(path, new, dry)
            stats[path.stem] = f"至 {pd.to_datetime(new.date).max():%m-%d}，{n} 天覆蓋"
    # TXO：一次月資料 → 全序列 / 週選 / 買賣權未平倉（同 mtx.fetch_aux2、options.fetch_weekly、mtx.fetch_aux 的篩法）
    chain_p = FD / "TXO_chain.parquet"
    parts = []
    for a, b in _periods(_dates(pd.read_parquet(chain_p, columns=["date"])).max(), today, "M"):
        o = api(lim, dataset="TaiwanOptionDaily", data_id="TXO", start_date=a, end_date=b)
        if o is not None and len(o):
            parts.append(o[o.trading_session == "position"])
    if parts:
        o = pd.concat(parts, ignore_index=True)
        mo = o[o.contract_date.str.fullmatch(r"\d{6}")]
        wk = o[o.contract_date.str.fullmatch(r"\d{6}W\d")]
        n1 = _merge_dates(chain_p, mo[["date", "contract_date", "strike_price", "call_put", "settlement_price"]], dry)
        n2 = _merge_dates(FD / "TXO_weekly.parquet",
                          wk[["date", "contract_date", "strike_price", "call_put", "settlement_price", "volume", "open_interest"]], dry)
        oi = mo.groupby(["date", "call_put"]).open_interest.sum().unstack()
        n3 = _merge_dates(FD / "TXO_oi.parquet", oi, dry, index_date=True)
        stats["TXO"] = f"至 {o.date.max()[5:]}，chain {n1} / weekly {n2} / oi {n3} 天覆蓋"
    # 期交所最後結算價：只抓今年
    if not dry:
        sys.path.insert(0, str(ROOT / "research" / "v2"))
        import options
        before = len(pd.read_parquet(FD / "TXO_final_settle.parquet"))
        after = len(options.fetch_fsp(years=[today.year]))
        stats["TXO_final_settle"] = f"{after - before:+d} 筆"
    return stats


def rebuild_derived(dry):
    """自建指標整段重算（都由 v2 檔計算，幾秒到一分鐘）。"""
    if dry:
        return {"derived": "dry-run 略過"}
    sys.path.insert(0, str(ROOT / "research" / "v2"))
    import contextlib
    import io
    import breadth
    import maint
    import mtx
    import signals
    out = {}
    for name, fn in (("txo_vix", mtx.build_vix), ("ad", signals.build_ad), ("breadth", breadth.build), ("mkt_maint", maint.build)):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            fn()
        out[name] = buf.getvalue().strip().splitlines()[-1][:90]
    return out


def run(dry=False, rate=60, today=None):
    t0 = time.time()
    today = today or date.today()
    lim = RateLimiter(rate)
    u, added = update_universe(lim, dry)
    ids = set(u.stock_id)
    active = sorted(u[u.delist_date.isna()].stock_id)
    for sid in added:
        log(f"新股 {sid}：逐檔抓完整歷史")
        fetch_full(lim, sid, list(DATASET), dry)
    stats = update_daily(lim, ids, active, dry, today)
    stats.update(update_periodic(lim, ids, dry, today))
    stats.update(update_market(lim, dry, today))
    stats.update(rebuild_derived(dry))
    for k, v in stats.items():
        log(f"  {k:8s} {v}")
    log(f"完成{'（dry-run，未寫檔）' if dry else ''}，{(time.time() - t0) / 60:.1f} 分")
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rate", type=int, default=60)
    a = ap.parse_args()
    run(a.dry_run, a.rate)
