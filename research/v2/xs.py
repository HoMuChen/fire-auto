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
    per, pbr, dy = {}, {}, {}
    rev = {k: {} for k in ("rev_yoy", "rev3_high", "rev_pos6", "rev_yoy3")}
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
                d = d.reindex(pd.period_range(d.index.min(), d.index.max(), freq="M"))   # 缺月補 NaN，shift 才對齊
                y = d / d.shift(12) - 1
                r3 = d.rolling(3).sum()
                feats = {"rev_yoy": y,
                         "rev3_high": r3 / r3.rolling(12, min_periods=12).max(),
                         "rev_pos6": (y > 0).astype(float).where(y.notna()).rolling(6, min_periods=6).sum(),
                         "rev_yoy3": y.rolling(3, min_periods=3).mean()}
                avail = (d.index + 1).to_timestamp() + pd.Timedelta(days=10)
                for k, f_ in feats.items():
                    v = _align(pd.Series(f_.values, index=avail), idx)
                    if v is not None:
                        rev[k][s] = v
    P["per"], P["pbr"], P["dy"] = (pd.DataFrame(x).reindex(index=idx, columns=cols) for x in (per, pbr, dy))
    for k, x in rev.items():
        P[k] = pd.DataFrame(x).reindex(index=idx, columns=cols)


def _align(y, idx, stale=75):
    """可得日序列 → 交易日（ffill；超過 stale 天沒新值視為缺）。"""
    y = y.replace([np.inf, -np.inf], np.nan).dropna()
    y = y[~y.index.duplicated(keep="last")].sort_index()
    if not len(y):
        return None
    u = y.index.union(idx)
    v = y.reindex(u).ffill()
    last = pd.Series(y.index, index=y.index).reindex(u).ffill()
    return v.where((u - pd.DatetimeIndex(last)) <= pd.Timedelta(days=stale)).reindex(idx)


_AVAIL = {3: (5, 16, 0), 6: (8, 15, 0), 9: (11, 15, 0), 12: (4, 1, 1)}   # 季底月 → (月, 日, 跨年)


def _q_avail(qidx):
    return pd.DatetimeIndex([pd.Timestamp(q.year + _AVAIL[q.month][2], _AVAIL[q.month][0], _AVAIL[q.month][1]) for q in qidx])


def add_fin(P):
    """季報特徵（單季值）：roe_ttm、gpa_ttm、sue；依法定截止 +1 天對齊。"""
    idx, cols = P["close"].index, P["close"].columns
    out = {k: {} for k in ("roe_ttm", "gpa_ttm", "sue")}
    for s in cols:
        f, b = H.D / "fin" / f"{s}.parquet", H.D / "bs" / f"{s}.parquet"
        if not f.exists():
            continue
        d = pd.read_parquet(f)
        if not len(d):
            continue
        q = d.pivot_table(index="date", columns="type", values="value", aggfunc="last")
        q.index = pd.to_datetime(q.index)
        q = q.reindex(pd.date_range(q.index.min(), q.index.max(), freq="QE"))
        feats = {}
        if "EPS" in q:
            e = q.EPS
            dif = e - e.shift(4)
            feats["sue"] = dif / dif.rolling(8, min_periods=6).std()
        ni = q.get("IncomeAfterTaxes")
        gp = q.get("GrossProfit")
        if b.exists():
            bb = pd.read_parquet(b)
            if len(bb):
                bs = bb.pivot_table(index="date", columns="type", values="value", aggfunc="last")
                bs.index = pd.to_datetime(bs.index)
                bs = bs.reindex(q.index)
                if ni is not None and "Equity" in bs:
                    eq = bs.Equity.where(bs.Equity > 0)
                    feats["roe_ttm"] = ni.rolling(4, min_periods=4).sum() / eq
                if gp is not None and "TotalAssets" in bs:
                    feats["gpa_ttm"] = gp.rolling(4, min_periods=4).sum() / bs.TotalAssets.where(bs.TotalAssets > 0)
        av = _q_avail(q.index)
        for k, v in feats.items():
            a = _align(pd.Series(v.values, index=av), idx, stale=200)
            if a is not None:
                out[k][s] = a
    for k, x in out.items():
        P[k] = pd.DataFrame(x).reindex(index=idx, columns=cols)


def add_chips(P):
    """三大法人淨買超（股）與融資餘額（張）。法人資料 2012-05 起；之前為 NaN，之後無列＝0。"""
    idx, cols = P["close"].index, P["close"].columns
    fam = {"Foreign_Investor": "f", "Foreign_Dealer_Self": "f", "Investment_Trust": "t",
           "Dealer": "d", "Dealer_self": "d", "Dealer_Hedging": "d"}
    net = {k: {} for k in "ftd"}
    mb = {}
    for s in cols:
        f = H.D / "inst" / f"{s}.parquet"
        if f.exists():
            d = pd.read_parquet(f)
            if len(d):
                d["fam"] = d.name.map(fam)
                d["net"] = d.buy.astype(float) - d.sell.astype(float)
                g = d.dropna(subset=["fam"]).groupby(["date", "fam"]).net.sum().unstack()
                g.index = pd.to_datetime(g.index)
                for k in "ftd":
                    if k in g:
                        net[k][s] = g[k]
        f = H.D / "margin" / f"{s}.parquet"
        if f.exists():
            d = pd.read_parquet(f)
            if len(d):
                mb[s] = d.assign(date=pd.to_datetime(d.date)).drop_duplicates("date").set_index("date").MarginPurchaseTodayBalance.astype(float)
    t0 = pd.Timestamp("2012-05-02")   # 現存股法人資料自此起（部分下市股有更早資料，不用以免覆蓋不一致）
    after = np.asarray(idx >= t0)[:, None]
    for k in "ftd":
        x = pd.DataFrame(net[k]).reindex(index=idx, columns=cols)
        x = x.mask(after & x.isna().values, 0).where(np.broadcast_to(after, x.shape)).where(P["close"].notna())
        P[f"inst_{k}"] = x
    P["margin_bal"] = pd.DataFrame(mb).reindex(index=idx, columns=cols).where(P["close"].notna())


def _pivot(f):
    if not f.exists():
        return None
    d = pd.read_parquet(f)
    if not len(d):
        return None
    q = d.pivot_table(index="date", columns="type", values="value", aggfunc="last")
    q.index = pd.to_datetime(q.index)
    return q


INT32_MAX = 2_147_483_647


def add_v16(P):
    """第十六批：mv（日市值）、big_chg / holders_chg（集保：換股月前一月底 vs 前兩月底快照）、
    季報衍生 ta_g、cfo_ttm、accr、fscore（現金流量表為年初累計 → 拆單季；依法定截止 +1 天對齊）。
    mv 修正：FinMind 股數有 int32 溢位（2,147,483,647；2010 初大型股最多，2330 市值錯 19 天），隱含股數（市值 / 原始收盤）
    與集保總股數（7 天前以前最近一次有效快照）差超過 2 倍時改用集保股數 × 原始收盤；兩邊都無效 → 缺值。"""
    idx, cols = P["close"].index, P["close"].columns
    per = idx.to_period("M")
    mv, big, hol, tsh = {}, {}, {}, {}
    out = {k: {} for k in ("ta_g", "cfo_ttm", "accr", "fscore")}
    for s in cols:
        f = H.D / "mv" / f"{s}.parquet"
        if f.exists():
            d = pd.read_parquet(f)
            if len(d):
                mv[s] = d.assign(date=pd.to_datetime(d.date)).drop_duplicates("date").set_index("date").market_value.astype(float)
        f = H.D / "hold" / f"{s}.parquet"
        if f.exists():
            d = pd.read_parquet(f)
            if len(d):
                d["date"] = pd.to_datetime(d.date)
                t = d[d.HoldingSharesLevel == "total"].drop_duplicates("date").set_index("date").unit.astype(float).sort_index()
                t = t[t != INT32_MAX]
                if len(t):
                    t.index = t.index + pd.Timedelta(days=7)
                    tsh[s] = t.reindex(t.index.union(idx)).ffill().reindex(idx)
                for lev, col, dst in (("more than 1,000,001", "percent", big), ("total", "people", hol)):
                    x = d[d.HoldingSharesLevel == lev].drop_duplicates("date").set_index("date")[col].astype(float).sort_index()
                    if len(x):
                        m = x.groupby(x.index.to_period("M")).last()            # 每月最後一個快照
                        m = m.reindex(pd.period_range(m.index.min(), m.index.max(), freq="M"))
                        chg = m - m.shift(1) if col == "percent" else m / m.shift(1) - 1
                        dst[s] = pd.Series(chg.shift(1).reindex(per).values, index=idx)   # 月 M 的交易日用 (M−1) vs (M−2)
        q, b, c = (_pivot(H.D / k / f"{s}.parquet") for k in ("fin", "bs", "cf"))
        if q is None or b is None:
            continue
        qi = pd.date_range(min(q.index.min(), b.index.min()), max(q.index.max(), b.index.max()), freq="QE")
        q, b = q.reindex(qi), b.reindex(qi)
        g = lambda df, k: df[k] if k in df else pd.Series(np.nan, index=qi)   # noqa: E731
        ta = g(b, "TotalAssets").where(lambda x: x > 0)
        ttm = lambda x: x.rolling(4, min_periods=4).sum()                      # noqa: E731
        ni, gp, rev = ttm(g(q, "IncomeAfterTaxes")), ttm(g(q, "GrossProfit")), ttm(g(q, "Revenue"))
        feats = {"ta_g": ta / ta.shift(4) - 1}
        cfo = None
        if c is not None:
            cum = c["CashFlowsFromOperatingActivities"] if "CashFlowsFromOperatingActivities" in c else g(c, "NetCashInflowFromOperatingActivities")
            cum = cum.reindex(qi)
            one = cum - cum.shift(1)
            cfo = ttm(cum.where(qi.quarter == 1, one))                      # Q1 = 累計值，其餘 = 累計差
            feats["cfo_ttm"] = cfo
            feats["accr"] = (ni - cfo) / ((ta + ta.shift(4)) / 2)
            roa = ni / ta.shift(4)
            ltd = g(b, "LongtermBorrowings").fillna(0).where(ta.notna())
            lev, cr = ltd / ta, g(b, "CurrentAssets") / g(b, "CurrentLiabilities").where(lambda x: x > 0)
            cs = g(b, "CapitalStock")
            gm, at = gp / rev.where(rev > 0), rev / ta.shift(4)
            parts = [(roa, roa > 0), (cfo, cfo > 0), (roa.shift(4), roa > roa.shift(4)), (ni, cfo > ni),
                     (lev.shift(4), lev <= lev.shift(4)), (cr.shift(4), cr > cr.shift(4)), (cs.shift(4), cs <= cs.shift(4)),
                     (gm.shift(4), gm > gm.shift(4)), (at.shift(4), at > at.shift(4))]
            ok = pd.concat([x.notna() & cfo.notna() & roa.notna() for x, _ in parts], axis=1).all(axis=1)
            feats["fscore"] = sum(v.astype(float) for _, v in parts).where(ok)
        av = _q_avail(qi)
        for k, v in feats.items():
            a_ = _align(pd.Series(v.values, index=av), idx, stale=200)
            if a_ is not None:
                out[k][s] = a_
    rc = P["raw_close"].where(P["raw_close"] > 0)
    mv = pd.DataFrame(mv).reindex(index=idx, columns=cols)
    ts = pd.DataFrame(tsh).reindex(index=idx, columns=cols).where(lambda x: x > 0)
    z = mv / rc / ts
    ovf = ((mv / rc / INT32_MAX - 1).abs() < 0.02) & ts.isna()
    P["mv"] = mv.mask((z < 0.5) | (z > 2), ts * rc).mask(ovf).where(P["close"].notna())
    P["big_chg"] = pd.DataFrame(big).reindex(columns=cols)
    P["holders_chg"] = pd.DataFrame(hol).reindex(columns=cols)
    for k, x in out.items():
        P[k] = pd.DataFrame(x).reindex(index=idx, columns=cols)


def add_v17(P):
    """第十七批：sur（月營收驚喜）、margin3（三率三升）、ni_high（盈餘創新高）、inv_chg（存貨變動）、turn60（換手率）、tr_r252（加權報酬指數近 252 日報酬）。"""
    idx, cols = P["close"].index, P["close"].columns
    out = {k: {} for k in ("sur", "margin3", "ni_high", "inv_chg")}
    for s in cols:
        f = H.D / "revenue" / f"{s}.parquet"
        if f.exists():
            d = pd.read_parquet(f)
            if len(d):
                d["ym"] = pd.PeriodIndex.from_fields(year=d.revenue_year.astype(int), month=d.revenue_month.astype(int), freq="M")
                d = d.drop_duplicates("ym", keep="last").set_index("ym").revenue.astype(float)
                d = d.reindex(pd.period_range(d.index.min(), d.index.max(), freq="M"))
                dif = d - d.shift(12)
                h = dif.shift(1).rolling(24, min_periods=12)
                avail = (d.index + 1).to_timestamp() + pd.Timedelta(days=10)
                v = _align(pd.Series(((dif - h.mean()) / h.std()).values, index=avail), idx)
                if v is not None:
                    out["sur"][s] = v
        q, b = _pivot(H.D / "fin" / f"{s}.parquet"), _pivot(H.D / "bs" / f"{s}.parquet")
        feats = {}
        if q is not None:
            qi = pd.date_range(q.index.min(), q.index.max(), freq="QE")
            q = q.reindex(qi)
            g = lambda df, k: df[k] if k in df else pd.Series(np.nan, index=df.index)   # noqa: E731
            rev = g(q, "Revenue").where(lambda x: x > 0)
            ni = g(q, "IncomeAfterTaxes").combine_first(g(q, "NetIncome"))     # 2013 前（舊制）淨利欄位叫 NetIncome
            dm = [(x / rev).diff(4) for x in (g(q, "GrossProfit"), g(q, "OperatingIncome"), ni)]
            ok = pd.concat([x.notna() for x in dm], axis=1).all(axis=1)
            feats["margin3"] = (sum((x > 0).astype(float) for x in dm) + 0.1 * dm[1].clip(-1, 1)).where(ok)
            ttm = ni.rolling(4, min_periods=4).sum()
            hi = ttm.rolling(12, min_periods=8).max()
            gr = (ttm / ttm.shift(4).where(lambda x: x > 0) - 1).clip(-1, 10).fillna(0)
            feats["ni_high"] = (ttm / hi + 1e-4 * gr).where((ttm > 0) & (hi > 0))
        if b is not None and "Inventories" in b:
            qi = pd.date_range(b.index.min(), b.index.max(), freq="QE")
            b = b.reindex(qi)
            ta = b["TotalAssets"].where(lambda x: x > 0) if "TotalAssets" in b else pd.Series(np.nan, index=qi)
            inv = b["Inventories"]
            feats["inv_chg"] = (inv - inv.shift(4)) / ((ta + ta.shift(4)) / 2)
        for k, v in feats.items():
            a_ = _align(pd.Series(v.values, index=_q_avail(v.index)), idx, stale=200)
            if a_ is not None:
                out[k][s] = a_
    for k, x in out.items():
        P[k] = pd.DataFrame(x).reindex(index=idx, columns=cols)
    if "mv" not in P:
        add_v16(P)
    sh = P["mv"] / P["raw_close"].where(P["raw_close"] > 0)
    P["turn60"] = P["volume"].astype(float).rolling(60, min_periods=40).mean() / sh.where(sh > 0)
    t = pd.read_parquet(H.D / "taiex_tr.parquet")
    t = t.assign(date=pd.to_datetime(t.date)).drop_duplicates("date").set_index("date").price.astype(float)
    P["tr_r252"] = (t / t.shift(252) - 1).reindex(t.index.union(idx)).ffill().reindex(idx)   # 2005 起算，2010 即有值


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
        m = ~(((part - f).abs() <= 1e-9 * (1 + f.abs())) | (part == f) | (part.isna() & f.isna()))
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
            for lab, names in (("top_q", q.index[q == nq]), ("topn", s.nlargest(topn).index)):
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
    for col in ("top_q", "topn"):
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


def run(fns, period="dev", P=None, unseal=False, note="", topn=30, start=None, chips=False, fin=False, v16=False, v17=False):
    if period == "holdout" and not unseal:
        raise PermissionError("holdout 封存中")
    p0, p1 = H.PERIODS[period]
    p0 = max(p0, start) if start else p0
    end = min(pd.Timestamp(p1), pd.Timestamp.today()).strftime("%Y-%m-%d")
    if P is None:
        P = load_panel(end, fundamentals=True)
        if chips:
            add_chips(P)
        if fin:
            add_fin(P)
        if v16:
            add_v16(P)
        if v17:
            add_v17(P)
    res = {}
    for name, (fn, desc) in fns.items():
        bad = leak_check_xs(fn, P) if period == "dev" else []
        if bad:
            raise RuntimeError(f"{name} 前視檢查失敗 {bad}")
        R = backtest(fn(P), P, p0, p1, topn=topn)
        st = stats(R)
        res[name] = (R, st)
        with open(H.LOG, "a") as f:
            f.write(json.dumps(dict(ts=time.strftime("%Y-%m-%d %H:%M:%S"), kind="xs", period=period, note=note,
                                    strategy=name, desc=desc, n_hold=topn,
                                    code=hashlib.md5(inspect.getsource(fn).encode()).hexdigest()[:10], **st),
                               ensure_ascii=False, default=lambda v: round(float(v), 5)) + "\n")
    return res, P
