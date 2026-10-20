"""橫斷面（排名換股）策略評估 — v2 harness 的姊妹模組。

score_fn(P) -> DataFrame（日期 × 股票），第 t 列只能用 t 日收盤以前的資訊；
換股日 d 用 d 前一交易日的分數，d 開盤成交。P 為寬表 dict：
  open/high/low/close（還原）、raw_close、volume、money、liq（pit 流動性 bool）、nbars（累計 K 數），
  以及可選的 per/pbr/dy（日）、rev_yoy 等基本面（已依可得日對齊）。
"""
import hashlib
import inspect
import json
import time

import numpy as np
import pandas as pd

import harness as H

COST = H.BUY_FEE + H.SELL_FEE + H.SELL_TAX


def load_panel(end, fundamentals=False):
    sids = H.universe()
    cols = {k: {} for k in ("open", "high", "low", "close", "raw_close", "volume", "money")}
    for s in sids:
        df = H.load_stock(s, end)
        if df is None:
            continue
        df = df.set_index("date")
        for k in cols:
            cols[k][s] = df[k]
    P = {k: pd.DataFrame(v).sort_index() for k, v in cols.items()}
    P["liq"] = P["volume"].rolling(H.LIQ_WIN, min_periods=40).mean() >= H.LIQ_SHARES
    P["nbars"] = P["close"].notna().cumsum()
    if fundamentals:
        _add_fundamentals(P, end)
    return P


def _add_fundamentals(P, end):
    idx, cols = P["close"].index, P["close"].columns
    per, pbr, dy, yoy = {}, {}, {}, {}
    for s in cols:
        f = H.D / "per" / f"{s}.parquet"
        if f.exists():
            d = pd.read_parquet(f)
            if len(d):
                d = d.assign(date=pd.to_datetime(d.date)).drop_duplicates("date").set_index("date")
                per[s], pbr[s], dy[s] = d.PER, d.PBR, d.dividend_yield
        f = H.D / "revenue" / f"{s}.parquet"
        if f.exists():
            d = pd.read_parquet(f)
            if len(d):
                # 營收月份 m：次月 10 日法定截止，保守視為 11 日起可得（換股日 ≥12 日）
                d["ym"] = pd.PeriodIndex.from_fields(year=d.revenue_year.astype(int), month=d.revenue_month.astype(int), freq="M")
                d = d.drop_duplicates("ym", keep="last").set_index("ym").revenue.astype(float)
                y = d / d.shift(12, freq="M").reindex(d.index) - 1
                avail = (d.index + 1).to_timestamp() + pd.Timedelta(days=10)
                y = pd.Series(y.values, index=avail).replace([np.inf, -np.inf], np.nan).dropna()
                y = y[~y.index.duplicated(keep="last")].sort_index()
                if len(y):
                    u = y.index.union(idx)
                    v = y.reindex(u).ffill()
                    last = pd.Series(y.index, index=y.index).reindex(u).ffill()
                    v = v.where((u - pd.DatetimeIndex(last)) <= pd.Timedelta(days=75))   # 超過 75 天沒新營收視為缺
                    yoy[s] = v.reindex(idx)
    P["per"], P["pbr"], P["dy"], P["rev_yoy"] = (pd.DataFrame(x).reindex(index=idx, columns=cols)
                                                for x in (per, pbr, dy, yoy))


def rebalance_dates(idx, p0, p1):
    s = pd.Series(idx, index=idx)
    s = s[(s >= p0) & (s <= p1) & (s.dt.day >= 12)]
    return list(s.groupby(s.dt.to_period("M")).first())


def leak_check_xs(fn, P, cuts=("2013-06-28", "2016-03-31", "2018-09-28")):
    full = fn(P)
    bad = []
    for c in cuts:
        Pt = {k: v.loc[:c] for k, v in P.items()}
        part = fn(Pt).iloc[-1]
        f = full.loc[part.name]
        m = ~(((part - f).abs() <= 1e-9 * (1 + f.abs())) | (part.isna() & f.isna()))
        if m.sum():
            bad.append((c, int(m.sum())))
    return bad


def backtest(score, P, p0, p1, nq=5, topn=30):
    O, C = P["open"], P["close"]
    Cf = C.ffill()
    idx = O.index
    rd = rebalance_dates(idx, pd.Timestamp(p0), pd.Timestamp(p1))
    ss = score.shift(1)
    elig = P["liq"].shift(1).fillna(False).astype(bool) & (P["nbars"].shift(1) >= 250) & O.notna()
    rows, prev = [], {}
    for d, d2 in zip(rd[:-1], rd[1:]):
        e = elig.loc[d]
        s = ss.loc[d][e].dropna()
        k2 = idx.get_loc(d2)
        px2 = O.loc[d2].fillna(Cf.iloc[k2 - 1])
        r = (px2 / O.loc[d] - 1)[e].clip(-0.95, 10)
        bench = r.mean()
        row = dict(date=d, bench=bench, n=len(s))
        if len(s) >= nq * 10:
            q = pd.qcut(s.rank(method="first"), nq, labels=False) + 1
            for k in range(1, nq + 1):
                names = q.index[q == k]
                row[f"Q{k}"] = r[names].mean()
            for lab, names in (("top_q", q.index[q == nq]), ("top30", s.nlargest(topn).index)):
                held = set(names)
                turn = 1 - len(held & prev.get(lab, set())) / max(len(held), 1)
                row[lab] = r[list(names)].mean() - turn * COST
                row[f"turn_{lab}"] = turn
                prev[lab] = held
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


def stats(R):
    out = {}
    yrs = (R.index[-1] - R.index[0]).days / 365.25
    for col in ("top_q", "top30"):
        x = R[col].dropna()
        ex = x - R.bench.reindex(x.index)
        eq = (1 + x).cumprod()
        by = ex.groupby(ex.index.year).sum()
        out[col] = dict(CAGR=eq.iloc[-1] ** (1 / yrs) - 1, bench_CAGR=(1 + R.bench).prod() ** (1 / yrs) - 1,
                        ex_ann=ex.mean() * 12, t=ex.mean() / ex.std() * np.sqrt(len(ex)),
                        yrs_pos=f"{int((by > 0).sum())}/{len(by)}", MDD=(eq / eq.cummax() - 1).min(),
                        turnover=R[f"turn_{col}"].mean())
    out["Q_ex_ann"] = {f"Q{k}": round((R[f"Q{k}"] - R.bench).mean() * 12, 4) for k in range(1, 6) if f"Q{k}" in R}
    return out


def run(fns, period="dev", P=None, unseal=False, note=""):
    if period == "holdout" and not unseal:
        raise PermissionError("holdout 封存中")
    p0, p1 = H.PERIODS[period]
    end = min(pd.Timestamp(p1), pd.Timestamp.today()).strftime("%Y-%m-%d")
    if P is None:
        P = load_panel(end, fundamentals=True)
    res = {}
    for name, (fn, desc) in fns.items():
        bad = leak_check_xs(fn, P) if period == "dev" else []
        if bad:
            raise RuntimeError(f"{name} 前視檢查失敗 {bad}")
        R = backtest(fn(P), P, p0, p1)
        st = stats(R)
        res[name] = (R, st)
        with open(H.LOG, "a") as f:
            f.write(json.dumps(dict(ts=time.strftime("%Y-%m-%d %H:%M:%S"), kind="xs", period=period, note=note,
                                    strategy=name, desc=desc,
                                    code=hashlib.md5(inspect.getsource(fn).encode()).hexdigest()[:10], **st),
                               ensure_ascii=False, default=lambda v: round(float(v), 5)) + "\n")
    return res, P
