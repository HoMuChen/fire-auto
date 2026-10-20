"""策略研究 harness v2（2026-10-20）— 防前視、防倖存者偏誤、留封存期。

資料：data/v2（含下市股、還原權息、2010 起；見 AGENTS.md）

期間（固定，不可為了結果改）：
  dev      2010-01-01 ~ 2019-12-31   開發：想法、邏輯、除錯都在這
  val      2020-01-01 ~ 2023-12-31   驗證：dev 過關且規則凍結後才跑
  holdout  2024-01-01 ~             封存：最後只看一次，需 unseal=True，會記錄在實驗紀錄

防漏規則（harness 強制）：
  ① 期間截斷：載入資料直接切到該期間最後一天，策略碰不到之後的 bar
  ② 前視檢查：每個策略先跑 leak_check() — 截斷資料重算訊號須與全期訊號前段一致
  ③ 進場股票池 point-in-time：訊號日近 60 日均量 ≥ 200 張、上市櫃（不含現為興櫃者）、已有 ≥120 根 K
  ④ 成交：訊號日收盤判斷 → 次一交易日開盤成交；一字漲停買不到（訊號作廢）、一字跌停賣不掉（順延）
  ⑤ 下市：資料斷了就以最後收盤出場（標 delisted）
  ⑥ 成本：買 0.1425%、賣 0.1425% + 稅 0.3%
  ⑦ 分群（vol 三分位 × 趨勢 R²>0.3）用「進場年前一年底」往回 250 根計算，只當描述欄位

策略介面：fn(bars) -> list，與 backtest.py 相同（bars 為 dict 列表，含 date/open/high/low/close/volume，
價格為還原權息價；另附 raw_close 供需要絕對價位時用 — 還原價的絕對水準含未來除權息資訊，不可拿來設門檻）。
訊號值 "buy" / "sell" / None。

評估：每筆淨報酬、相對 pit 等權大盤（開盤到開盤）的超額、相對「隨機進場對照組」的優勢；
t 值以進場月份 cluster（月平均超額 → t）。所有跑過的結果自動追加到 research/v2/experiments.jsonl。
"""
import hashlib
import inspect
import json
import time
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
D = BASE / "data/v2"
LOG = Path(__file__).resolve().parent / "experiments.jsonl"
PERIODS = {"dev": ("2010-01-01", "2019-12-31"), "val": ("2020-01-01", "2023-12-31"), "holdout": ("2024-01-01", "2099-12-31")}
LIQ_SHARES, LIQ_WIN, MIN_BARS = 200_000, 60, 120
BUY_FEE, SELL_FEE, SELL_TAX = 0.001425, 0.001425, 0.003
LIMIT_CHANGE = pd.Timestamp("2015-06-01")      # 漲跌幅 7% → 10%


def universe():
    u = pd.read_csv(D / "universe.csv", dtype=str)
    return u[u.type != "emerging"].stock_id.tolist()


def load_stock(sid, end):
    """合併 raw/adj，去掉停牌（close<=0）列，截到 end。"""
    r = pd.read_parquet(D / "raw" / f"{sid}.parquet")
    a = pd.read_parquet(D / "adj" / f"{sid}.parquet")
    if not len(r) or not len(a):
        return None
    m = a[["date", "open", "high", "low", "close"]].merge(
        r[["date", "open", "high", "low", "close", "volume", "money"]].rename(
            columns={"open": "raw_open", "high": "raw_high", "low": "raw_low", "close": "raw_close"}), on="date")
    m = m[(m.raw_close > 0) & (m.close > 0) & (m.open > 0) & (m.date <= pd.Timestamp(end))].reset_index(drop=True)
    return m if len(m) else None


def to_bars(df):
    return df[["date", "open", "high", "low", "close", "volume", "raw_close"]].assign(
        date=df.date.dt.strftime("%Y-%m-%d")).to_dict("records")


def _limit_flags(df):
    """一字漲停 / 一字跌停（開=高=低，且相對前收漲跌接近上限）。"""
    prev = df.raw_close.shift()
    lim = np.where(df.date < LIMIT_CHANGE, 0.07, 0.10) - 0.005
    flat = (df.raw_high == df.raw_low)
    up = flat & (df.raw_open >= prev * (1 + lim))
    dn = flat & (df.raw_open <= prev * (1 - lim))
    return up.fillna(False).values, dn.fillna(False).values


def simulate(df, sig, stop_pct=None, max_hold=None, exit_on_sell=True, period=("1900", "2100"), data_end=None):
    """單檔模擬，回傳交易列表。sig 與 df 等長。只有訊號日落在 period 內才進場。"""
    n = len(df)
    o, c, v = df.open.values, df.close.values, df.volume.values
    liq = (df.volume.rolling(LIQ_WIN).mean() >= LIQ_SHARES).values
    up, dn = _limit_flags(df)
    dates = df.date.values
    p0, p1 = np.datetime64(period[0]), np.datetime64(period[1])
    out, i = [], MIN_BARS
    while i < n - 1:
        if sig[i] != "buy" or not liq[i] or not (p0 <= dates[i] <= p1):
            i += 1
            continue
        e = i + 1
        if up[e] or v[e] <= 0:              # 一字漲停/無量：買不到
            i += 1
            continue
        entry, peak, j, reason = o[e], o[e], e, None
        while True:
            peak = max(peak, c[j])
            if stop_pct is not None and c[j] <= peak * (1 - stop_pct):
                reason = "stop"
            elif exit_on_sell and sig[j] == "sell" and j > i:
                reason = "signal"
            elif max_hold is not None and j - e + 1 >= max_hold:
                reason = "hold"
            if reason or j == n - 1:
                break
            j += 1
        x = j + 1
        while x < n and (dn[x] or v[x] <= 0):  # 一字跌停：順延
            x += 1
        if x >= n:                          # 資料斷：下市或期間結束，用最後收盤
            px, xd = c[n - 1], dates[n - 1]
            reason = "delisted" if data_end is not None and dates[n - 1] < np.datetime64(data_end) - np.timedelta64(10, "D") else "eop"
        else:
            px, xd = o[x], dates[x]
        ret = px * (1 - SELL_FEE - SELL_TAX) / (entry * (1 + BUY_FEE)) - 1
        out.append((dates[e], xd, ret, reason or "eop", x - e))
        i = x if x < n else n
    return out


# ───────── worker ─────────
_W = {}


def _work(sid):
    df = load_stock(sid, _W["end"])
    if df is None or len(df) < MIN_BARS + 2:
        return sid, []
    bars = to_bars(df)
    res = []
    for name, spec in _W["specs"].items():
        try:
            sig = spec["fn"](bars) if spec.get("fn") else _random_sig(sid, len(bars))
        except Exception:  # noqa: BLE001
            continue
        for t in simulate(df, sig, spec.get("stop"), spec.get("max_hold"), spec.get("exit_on_sell", True),
                          _W["period"], _W["data_end"]):
            res.append((name, sid) + t)
    return sid, res


def _random_sig(sid, n):
    off = int(hashlib.md5(sid.encode()).hexdigest(), 16) % 20
    return ["buy" if (k + off) % 20 == 0 else None for k in range(n)]


def leak_check(fn, sids=("2330", "2603", "1101", "3481", "6550"), cuts=(0.4, 0.7, 0.9), end="2019-12-31"):
    """截斷資料重算的訊號必須與全期訊號前段一致，否則策略偷看未來。"""
    bad = []
    for sid in sids:
        df = load_stock(sid, end)
        if df is None:
            continue
        bars = to_bars(df)
        full = fn(bars)
        for q in cuts:
            k = int(len(bars) * q)
            part = fn(bars[:k])
            diff = [d for d in range(k) if (part[d] or None) != (full[d] or None)]
            if diff:
                bad.append((sid, q, len(diff), bars[diff[0]]["date"]))
    return bad


# ───────── 大盤與分群 ─────────
def market_and_groups(sids, end):
    O, C, V = {}, {}, {}
    for s in sids:
        df = load_stock(s, end)
        if df is None:
            continue
        df = df.set_index("date")
        O[s], C[s], V[s] = df.open, df.close, df.volume
    O, C, V = (pd.DataFrame(x).sort_index() for x in (O, C, V))
    liq = (V.rolling(LIQ_WIN, min_periods=LIQ_WIN).mean() >= LIQ_SHARES)
    r = (O / O.shift() - 1).where(liq.shift().fillna(False)).clip(-0.5, 0.5)
    mkt = (1 + r.mean(axis=1).fillna(0)).cumprod()
    groups = {}
    for y in range(C.index[0].year + 1, C.index[-1].year + 2):
        hist = C.loc[:f"{y-1}-12-31"].iloc[-250:]
        ok = liq.loc[:f"{y-1}-12-31"].iloc[-1] & (hist.notna().sum() >= 200)
        h = hist.loc[:, ok]
        vol = np.log(h).diff().std() * np.sqrt(252)
        t = np.arange(len(h))
        r2 = h.apply(lambda s: np.corrcoef(t[s.notna().values], s.dropna().values)[0, 1] ** 2)
        q1, q2 = vol.quantile([1 / 3, 2 / 3])
        vl = np.where(vol <= q1, "low_vol", np.where(vol <= q2, "med_vol", "high_vol"))
        groups[y] = pd.Series([f"{a}/{'trending' if b > 0.3 else 'sideways'}" for a, b in zip(vl, r2)], index=h.columns)
    return mkt, groups


# ───────── 主流程 ─────────
def run(specs, period="dev", unseal=False, procs=8, note=""):
    """specs: {name: {"fn", "stop", "max_hold", "exit_on_sell", "desc"}}；自動加入隨機對照組。"""
    if period == "holdout" and not unseal:
        raise PermissionError("holdout 封存中：確定規則凍結後才 run(..., period='holdout', unseal=True)")
    p0, p1 = PERIODS[period]
    end = min(pd.Timestamp(p1), pd.Timestamp.today()).strftime("%Y-%m-%d")
    for name, sp in specs.items():
        if sp.get("fn"):
            bad = leak_check(sp["fn"], end=end)
            if bad:
                raise RuntimeError(f"{name} 前視檢查失敗：{bad[:3]}")
    stops = sorted({sp.get("stop") for sp in specs.values()}, key=lambda x: (x is None, x))
    allspecs = dict(specs)
    for s in stops:
        for mh in sorted({sp.get("max_hold") for sp in specs.values() if sp.get("stop") == s}, key=lambda x: (x is None, x)):
            allspecs[_base_name(s, mh)] = {"fn": None, "stop": s, "max_hold": mh, "exit_on_sell": False}
    sids = universe()
    t0 = time.time()
    _W.update(specs=allspecs, end=end, period=(p0, p1), data_end=end)
    with mp.get_context("fork").Pool(procs) as pool:   # fork：讓 worker 繼承 specs（含 lambda）
        res = pool.map(_work, sids, chunksize=16)
    T = pd.DataFrame([t for _, ts in res for t in ts], columns=["name", "sid", "buy", "sell", "ret", "reason", "bars"])
    mkt, groups = market_and_groups(sids, end)
    T["mret"] = mkt.reindex(T.sell).values / mkt.reindex(T.buy).values - 1
    T["ex"] = T.ret - T.mret
    T["year"] = T.buy.dt.year
    T["group"] = [groups.get(y, pd.Series(dtype=str)).get(s) for y, s in zip(T.year, T.sid)]
    S = summarize(T, specs)
    _log(S, specs, period, note)
    print(f"[{period}] {len(sids)} 檔、{len(T)} 筆，{time.time()-t0:.0f}s")
    return T, S


def _base_name(stop, mh):
    return f"RANDOM(stop={stop},hold={mh})"


def summarize(T, specs):
    rows = []
    for name, sp in specs.items():
        x = T[T.name == name]
        b = T[T.name == _base_name(sp.get("stop"), sp.get("max_hold"))]
        if not len(x):
            rows.append(dict(strategy=name, n=0))
            continue
        m = x.groupby(x.buy.dt.to_period("M")).ex.mean()
        bex = b.ex.mean()
        yr = x.groupby("year").ex.mean() - b.groupby("year").ex.mean().reindex(x.groupby("year").ex.mean().index)
        rows.append(dict(strategy=name, n=len(x), per_yr=len(x) / x.year.nunique(), win=(x.ret > 0).mean(),
                         avg=x.ret.mean(), ex=x.ex.mean(), edge=x.ex.mean() - bex,
                         t_month=(m - bex).mean() / (m - bex).std() * np.sqrt(len(m)),
                         yrs_pos=f"{int((yr > 0).sum())}/{len(yr)}", hold=x.bars.mean(),
                         delisted=int((x.reason == "delisted").sum())))
    return pd.DataFrame(rows)


def _log(S, specs, period, note):
    with open(LOG, "a") as f:
        for r in S.to_dict("records"):
            sp = specs[r["strategy"]]
            src = inspect.getsource(sp["fn"]) if sp.get("fn") else ""
            f.write(json.dumps(dict(ts=time.strftime("%Y-%m-%d %H:%M:%S"), period=period, note=note,
                                    code=hashlib.md5(src.encode()).hexdigest()[:10], desc=sp.get("desc", ""),
                                    stop=sp.get("stop"), max_hold=sp.get("max_hold"),
                                    **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}),
                               ensure_ascii=False, default=str) + "\n")


def by_group(T, specs):
    """策略 × 分群：每筆超額與相對對照組優勢（對照組同群組）。"""
    rows = []
    for name, sp in specs.items():
        bn = _base_name(sp.get("stop"), sp.get("max_hold"))
        for g, x in T[T.name == name].groupby("group"):
            b = T[(T.name == bn) & (T.group == g)]
            rows.append(dict(strategy=name, group=g, n=len(x), avg=x.ret.mean() * 100, ex=x.ex.mean() * 100,
                             edge=(x.ex.mean() - b.ex.mean()) * 100))
    return pd.DataFrame(rows)
