"""第四十二批 FS5：涂敏峰（超越巔峰）影片清單裡還沒測、現有資料能做的特徵，加上借券。規則見 HYPOTHESES.md 第四十二批。

特徵函數 FNS[代號] = (fn(P), 說明)，介面同 hypotheses.FNS：回傳日頻分數（日期 × 股票），評估時取換股日前一交易日的值。
先呼叫 add_tu(P)（需已跑 xs.add_chips、xs.add_v16）載入額外資料：借券、千張大戶比例、負債比、預估 EPS、大盤日報酬。
PEER_EY / PEER_BP 要用池與產業標籤，在換股日矩陣上算：peer_rank(基礎矩陣, 池, feature_industry.labels(欄位))。
評估併入 feature_library.py（同一套用途標籤與安慰劑）。

費氏時間轉折是類別事件，另外檢定：.venv/bin/python research/v2/feature_tu.py fib
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402
import xs  # noqa: E402

GAP_WIN, RS_WIN, RS_MIN = 20, 60, 10
FIB = (5, 8, 13, 21, 34, 55)
FIB_ALL = {1, 2, 3, 5, 8, 13, 21, 34, 55, 89}
FIB_CTRL = tuple(sorted({m + k for m in FIB for k in (-2, -1, 1, 2)} - FIB_ALL))
FIB_LOOK, FIB_MIN = 60, 20


def _frames(kind, cols):
    for s in cols:
        f = H.D / kind / f"{s}.parquet"
        if f.exists():
            d = pd.read_parquet(f)
            if len(d):
                yield s, d


def _ym(d):
    d = d.copy()
    d["ym"] = pd.PeriodIndex.from_fields(year=d.revenue_year.astype(int), month=d.revenue_month.astype(int), freq="M")
    r = d.drop_duplicates("ym", keep="last").set_index("ym").revenue.astype(float)
    return r.reindex(pd.period_range(r.index.min(), r.index.max(), freq="M"))


def nowcast_eps(eps, rev):
    """eps：單季 EPS（index＝季 Period）；rev：月營收（index＝月 Period）。回傳 {可得日: 年化前的預估單季 EPS}。

    預估＝最近已公布 EPS × 最近完整一季營收 ÷ 該 EPS 季營收；EPS ≤ 0 不乘比例；營收季已有 EPS 用實際值。"""
    g = rev.groupby(rev.index.asfreq("Q"))
    rq = g.sum(min_count=3).where(g.count() == 3)
    eps = eps.dropna()
    ea = pd.Series(xs._q_avail(eps.index.to_timestamp(how="end").normalize()), index=eps.index)
    rq = rq.dropna()
    ra = pd.Series([(q.asfreq("M", "end") + 1).to_timestamp() + pd.Timedelta(days=10) for q in rq.index], index=rq.index)
    out = {}
    for t in sorted(set(ea) | set(ra)):
        qe = ea.index[ea <= t]
        if not len(qe):
            continue
        qe = qe.max()
        e = eps[qe]
        qr = ra.index[ra <= t]
        qr = qr.max() if len(qr) else None
        if qr is not None and qr > qe and e > 0 and qe in rq.index and rq[qe] > 0:
            e = e * rq[qr] / rq[qe]
        out[t] = e
    return out


def add_tu(P):
    idx, cols = P["close"].index, P["close"].columns
    per = idx.to_period("M")
    sbl, big, debt, fwd = {}, {}, {}, {}
    for s, d in _frames("sbl", cols):
        d = d.assign(date=pd.to_datetime(d.date)).drop_duplicates("date", keep="last").set_index("date")
        sbl[s] = d.SBLShortSalesCurrentDayBalance.astype(float)
    for s, d in _frames("hold", cols):
        x = d[d.HoldingSharesLevel == "more than 1,000,001"].assign(date=lambda z: pd.to_datetime(z.date))
        x = x.drop_duplicates("date").set_index("date").percent.astype(float).sort_index()
        if len(x):
            m = x.groupby(x.index.to_period("M")).last()
            m = m.reindex(pd.period_range(m.index.min(), m.index.max(), freq="M"))
            big[s] = pd.Series(m.shift(1).reindex(per).values, index=idx)      # 月 M 用 M−1 最後一個快照
    for s in cols:
        b = xs._pivot(H.D / "bs" / f"{s}.parquet")
        if b is not None and "Liabilities" in b and "TotalAssets" in b:
            qi = pd.date_range(b.index.min(), b.index.max(), freq="QE")
            b = b.reindex(qi)
            lr = b.Liabilities / b.TotalAssets.where(b.TotalAssets > 0)
            a = xs._align(pd.Series((lr - lr.shift(4)).values, index=xs._q_avail(qi)), idx, stale=200)
            if a is not None:
                debt[s] = a
        q = xs._pivot(H.D / "fin" / f"{s}.parquet")
        f = H.D / "revenue" / f"{s}.parquet"
        if q is None or "EPS" not in q or not f.exists():
            continue
        r = pd.read_parquet(f)
        if not len(r):
            continue
        e = q.EPS.copy()
        e.index = e.index.to_period("Q")
        nc = nowcast_eps(e[~e.index.duplicated(keep="last")], _ym(r))
        if nc:
            a = xs._align(pd.Series(nc), idx, stale=200)
            if a is not None:
                fwd[s] = a
    P["sbl_bal"] = pd.DataFrame(sbl).reindex(index=idx, columns=cols).where(P["close"].notna())
    P["big_lvl"] = pd.DataFrame(big).reindex(index=idx, columns=cols)
    P["debt_chg"] = pd.DataFrame(debt).reindex(index=idx, columns=cols)
    P["eps_nc"] = pd.DataFrame(fwd).reindex(index=idx, columns=cols)
    t = pd.read_parquet(H.D / "taiex_tr.parquet")
    t = t.assign(date=pd.to_datetime(t.date)).drop_duplicates("date").set_index("date").price.astype(float)
    P["mkt_r"] = t.pct_change().reindex(idx)


# ───────── 價格 ─────────
def _gaps(P, up):
    Hh, L, C = P["high"], P["low"], P["close"]
    Hp, Lp, Cp = Hh.shift(1), L.shift(1), C.shift(1)
    if up:
        hit, lev, size = L > Hp, Hp, (L - Hp) / Cp
    else:
        hit, lev, size = Hh < Lp, Lp, (Lp - Hh) / Cp
    n = pd.DataFrame(0.0, index=C.index, columns=C.columns)
    opn = n.copy()
    for k in range(GAP_WIN):
        h, lv, sz = hit.shift(k, fill_value=False), lev.shift(k), size.shift(k)
        n += h.astype(float)
        if k == 0:
            still = h
        elif up:
            still = h & (L.rolling(k, min_periods=1).min() > lv)
        else:
            still = h & (Hh.rolling(k, min_periods=1).max() < lv)
        opn += sz.where(still, 0.0).fillna(0.0)
    ok = C.notna() & (P["nbars"] > GAP_WIN)
    return n.where(ok), opn.where(ok)


def gap_dn_n(P):
    return _gaps(P, up=False)[0]


def gap_dn_open(P):
    return _gaps(P, up=False)[1]


def gap_up_open(P):
    return _gaps(P, up=True)[1]


def rs_down(P):
    m = P["mkt_r"]
    ex = P["close"].pct_change(fill_method=None).sub(m, axis=0)
    ex = ex.where(np.broadcast_to((m < 0).values[:, None], ex.shape))
    return ex.rolling(RS_WIN, min_periods=RS_MIN).mean()


def _ma(P, n):
    return P["close"].rolling(n, min_periods=n).mean()


def ma60_slope(P):
    m = _ma(P, 60)
    return m / m.shift(20) - 1


def granville(P):
    m = _ma(P, 60)
    return (-(P["close"] / m - 1)).where(m > m.shift(5))


def ma_align(P):
    C = P["close"]
    s, ok = 0, None
    for n in (5, 20, 60, 120, 240):
        m = _ma(P, n)
        s = s + (C > m).astype(float) + (m > m.shift(5)).astype(float)
        ok = m.notna() & m.shift(5).notna() if ok is None else ok & m.notna() & m.shift(5).notna()
    return s.where(ok)


def upper_shadow(P):
    up = (P["high"] - np.maximum(P["open"], P["close"])) / P["close"]
    return up.rolling(20, min_periods=16).mean()


def kd(P, n=9):
    """日 KD(9,3,3)，起始 50；RSV 缺值（停牌等）時 K、D 沿用前值。"""
    C = P["close"]
    hh, ll = P["high"].rolling(n, min_periods=n).max(), P["low"].rolling(n, min_periods=n).min()
    rsv = ((C - ll) / (hh - ll).where(hh > ll) * 100).values
    K, D = np.full(rsv.shape, np.nan), np.full(rsv.shape, np.nan)
    k, d = np.full(rsv.shape[1], 50.0), np.full(rsv.shape[1], 50.0)
    for t in range(len(rsv)):
        ok = np.isfinite(rsv[t])
        k = np.where(ok, k * 2 / 3 + np.nan_to_num(rsv[t]) / 3, k)
        d = np.where(ok, d * 2 / 3 + k / 3, d)
        K[t], D[t] = np.where(ok, k, np.nan), np.where(ok, d, np.nan)
    return pd.DataFrame(K, index=C.index, columns=C.columns), pd.DataFrame(D, index=C.index, columns=C.columns)


def kd_div(P):
    K, _ = kd(P)
    C = P["close"]
    r = lambda X, f: getattr(X.rolling(20, min_periods=16), f)()   # noqa: E731
    pl2, pl1, kl2, kl1 = r(C, "min"), r(C, "min").shift(20), r(K, "min"), r(K, "min").shift(20)
    ph2, ph1, kh2, kh1 = r(C, "max"), r(C, "max").shift(20), r(K, "max"), r(K, "max").shift(20)
    bull = ((kl2 - kl1) / 100).where((pl2 < pl1) & (kl2 > kl1) & (kl1 < 20), 0.0)
    bear = ((kh1 - kh2) / 100).where((ph2 > ph1) & (kh2 < kh1) & (kh1 > 80), 0.0)
    return (bull - bear).where(pl1.notna() & kl1.notna() & pl2.notna() & kl2.notna())


def vol_price(P):
    from numpy.lib.stride_tricks import sliding_window_view as sw
    C, v5 = P["close"], P["volume"].astype(float).rolling(5, min_periods=4).mean()
    cur = v5.rolling(20, min_periods=16).max()
    out = np.full(C.shape, np.nan)
    cv, vv = C.values, v5.values
    for j in range(C.shape[1]):
        c = cv[:, j]
        if np.isfinite(c).sum() < 200:
            continue
        w = sw(np.where(np.isfinite(c), c, -np.inf), 252)
        cnt = sw(np.isfinite(c).astype(np.int16), 252).sum(axis=1)
        am = w.argmax(axis=1) + np.arange(len(w))
        vh = vv[am, j]
        out[251:, j] = np.where(cnt >= 200, vh, np.nan)
    return cur / pd.DataFrame(out, index=C.index, columns=C.columns).where(lambda x: x > 0)


def macd_m(P):
    C = P["close"]
    per = C.index.to_period("M")
    mc = C.groupby(per).last()
    e12, e26 = mc.ewm(span=12, adjust=False).mean(), mc.ewm(span=26, adjust=False).mean()
    dif = e12 - e26
    hist = (dif - dif.ewm(span=9, adjust=False).mean()) / mc
    hist = hist.where(mc.notna().cumsum() >= 35)
    hist = hist.reindex(pd.period_range(hist.index.min(), hist.index.max(), freq="M"))
    return pd.DataFrame(hist.shift(1).reindex(per).values, index=C.index, columns=C.columns)   # M 月用 M−1 月


def low_price(P):
    return P["raw_close"].where(P["raw_close"] > 0)


# ───────── 估值、籌碼、集保、財報 ─────────
def fwd_ey(P):
    raw = P["raw_close"].where(P["raw_close"] > 0)
    return 4 * P["eps_nc"] / raw


def trust_streak(P):
    x = P["inst_t"]
    b, v = (x > 0).values, x.notna().values
    out = np.zeros(b.shape)
    for t in range(1, len(out)):
        out[t] = np.where(b[t], out[t - 1] + 1, 0.0)
    out[0] = b[0]
    return pd.DataFrame(np.where(v, out, np.nan), index=x.index, columns=x.columns)


def big_lvl(P):
    return P["big_lvl"]


def debt_chg(P):
    return P["debt_chg"]


def _shares(P):
    sh = P["mv"] / P["raw_close"].where(P["raw_close"] > 0)
    return sh.where(sh > 0)


def margin_lvl(P):
    return P["margin_bal"] * 1000 / _shares(P)


def sbl_mr(P):
    m = P["margin_bal"]
    return (P["sbl_bal"] / 1000 / m).where(m >= 100)


def sbl_lvl(P):
    return P["sbl_bal"] / _shares(P)


def sbl_chg(P):
    s = P["sbl_bal"]
    return (s - s.shift(20)) / _shares(P)


FNS = {"GAP_DN_N": (gap_dn_n, "近20日向下缺口數"), "GAP_DN_OPEN": (gap_dn_open, "未補向下缺口"),
       "GAP_UP_OPEN": (gap_up_open, "未補向上缺口"), "RS_DOWN": (rs_down, "大盤跌時抗跌"),
       "MA60_SLOPE": (ma60_slope, "季線斜率"), "GRANVILLE": (granville, "季線上揚回測"), "MA_ALIGN": (ma_align, "多頭排列"),
       "FWD_EY": (fwd_ey, "預估盈餘殖利率"), "TRUST_STREAK": (trust_streak, "投信連買天數"), "BIG_LVL": (big_lvl, "千張大戶比例"),
       "DEBT_CHG": (debt_chg, "負債比變化"), "SBL_MR": (sbl_mr, "借券÷融資"), "SBL_LVL": (sbl_lvl, "借券餘額比"),
       "SBL_CHG": (sbl_chg, "借券20日增量"), "UPPER_SHADOW": (upper_shadow, "上影線"), "KD_DIV": (kd_div, "KD背離／擴散"),
       "VOL_PRICE": (vol_price, "量過價必過"), "MACD_M": (macd_m, "月MACD柱狀體"), "LOW_PRICE": (low_price, "股價高低"),
       "MARGIN_LVL": (margin_lvl, "融資水準")}
SBL = ("SBL_MR", "SBL_LVL", "SBL_CHG")
PEER = {"PEER_EY": ("H11", "同業內盈餘殖利率"), "PEER_BP": ("H5", "同業內淨值股價比")}     # 基礎特徵（hypotheses.FNS）
COMBO = {"THREE_LOW": ([("H11", 1), ("H5", 1), ("MARGIN_LVL", -1), ("BIG_LVL", 1)], 3, None, "三低一高"),
         "FOUR_TRAITS": ([("H2", -1), ("H22", 1)], 2, "H11", "四特徵（底部、賺錢、投信買）")}   # (成分, 最少有值數, 篩選, 說明)
CAT = {**{f: "價格" for f in ("GAP_DN_N", "GAP_DN_OPEN", "GAP_UP_OPEN", "RS_DOWN", "MA60_SLOPE", "GRANVILLE", "MA_ALIGN")},
       "FWD_EY": "估值", "PEER_EY": "估值", "PEER_BP": "估值", "TRUST_STREAK": "籌碼", "BIG_LVL": "集保", "DEBT_CHG": "財報",
       **{f: "借券" for f in ("SBL_MR", "SBL_LVL", "SBL_CHG")},
       "UPPER_SHADOW": "價格", "KD_DIV": "價格", "VOL_PRICE": "價格", "MACD_M": "價格", "LOW_PRICE": "價格",
       "MARGIN_LVL": "籌碼", "THREE_LOW": "組合", "FOUR_TRAITS": "組合"}
HIS = {"GAP_DN_N": 1, "GAP_DN_OPEN": 1, "GAP_UP_OPEN": 1, "RS_DOWN": 1, "MA60_SLOPE": 1, "GRANVILLE": 1, "MA_ALIGN": 1,
       "FWD_EY": 1, "PEER_EY": 1, "PEER_BP": 1, "TRUST_STREAK": 1, "BIG_LVL": 1, "DEBT_CHG": -1,
       "SBL_MR": 1, "SBL_LVL": -1, "SBL_CHG": -1, "UPPER_SHADOW": -1, "KD_DIV": 1, "VOL_PRICE": 1, "MACD_M": 1,
       "LOW_PRICE": -1, "MARGIN_LVL": -1, "THREE_LOW": 1, "FOUR_TRAITS": 1}       # 他講的方向（只描述）
NAMES = list(FNS) + list(PEER) + list(COMBO)


def peer_rank(X, E, codes, min_peers=5):
    """同產業（池內、有值）百分位；同業（不含自己）少於 min_peers 家為缺值。"""
    V = E & np.isfinite(X)
    out = np.full(X.shape, np.nan)
    for g in np.unique(codes[codes >= 0]):
        c = np.flatnonzero(codes == g)
        sub = pd.DataFrame(np.where(V[:, c], X[:, c], np.nan))
        r = sub.rank(axis=1, pct=True).values
        n = V[:, c].sum(axis=1, keepdims=True)
        out[:, c] = np.where(V[:, c] & (n - 1 >= min_peers), r, np.nan)
    return out


def combo(mats, E, spec):
    """池內百分位（依成分方向）的平均；有值成分數 ≥ need；filt 有值才算（有賺錢＝本益比 > 0）。"""
    parts, need, filt, _ = spec
    pct = [pd.DataFrame(np.where(E & np.isfinite(mats[f]), sg * mats[f], np.nan)).rank(axis=1, pct=True).values for f, sg in parts]
    st = np.stack(pct)
    n = np.isfinite(st).sum(axis=0)
    with np.errstate(invalid="ignore"):
        out = np.where(n >= need, np.nanmean(st, axis=0), np.nan)
    if filt is not None:
        out = np.where(np.isfinite(mats[filt]), out, np.nan)
    return out


# ───────── 費氏時間轉折（類別事件） ─────────
def fib_months(P, dates):
    """換股日 d（在 M 月）：M−1 月往前 60 個月最低月收盤距今月數；資料不足 60 個月為缺值。"""
    C = P["close"]
    mc = C.groupby(C.index.to_period("M")).last()
    cnt = C.notna().groupby(C.index.to_period("M")).sum()
    mc = mc.where(cnt > 0)
    arr = mc.values
    pos = {p: i for i, p in enumerate(mc.index)}
    out = np.full((len(dates), C.shape[1]), np.nan)
    for t, d in enumerate(dates):
        i = pos.get(d.to_period("M") - 1)
        if i is None or i + 1 < FIB_LOOK:
            continue
        w = arr[i + 1 - FIB_LOOK:i + 1]
        ok = np.isfinite(w).sum(axis=0) >= FIB_LOOK - 3
        am = np.nanargmin(np.where(np.isfinite(w), w, np.inf), axis=0)
        out[t] = np.where(ok, FIB_LOOK - 1 - am, np.nan)
    return out


def fib_test():
    import feature_library as FL
    t0 = time.time()
    P = xs.load_panel(FL.END)
    dates, Rdf, Edf = FL.forward(P, FL.PER["A"][0], FL.END)
    cols = P["close"].columns
    R = Rdf.reindex(index=dates, columns=cols).values
    E = Edf.reindex(index=dates, columns=cols).fillna(False).values.astype(bool)
    M = fib_months(P, dates)
    diff, by_m = np.full(len(dates), np.nan), {m: [] for m in range(FIB_LOOK)}
    for t in range(len(dates)):
        ok = E[t] & np.isfinite(M[t]) & np.isfinite(R[t])
        if ok.sum() < 100:
            continue
        mu = R[t, ok].mean()
        for m in range(FIB_LOOK):
            k = ok & (M[t] == m)
            if k.sum() >= 5:
                by_m[m].append(R[t, k].mean() - mu)
        f, c = ok & np.isin(M[t], FIB), ok & np.isin(M[t], FIB_CTRL)
        if f.sum() >= FIB_MIN and c.sum() >= FIB_MIN:
            diff[t] = R[t, f].mean() - R[t, c].mean()
    st = FL.pstats(diff, dates)
    dev = np.isfinite(diff) & (dates <= pd.Timestamp(FL.PER["B"][1]))
    x = diff[dev]
    st["dev"] = dict(n=len(x), mean=x.mean(), t=x.mean() / x.std(ddof=1) * np.sqrt(len(x)))
    ok_dev = st["dev"]["mean"] > 0 and st["dev"]["t"] >= 2 and st["A"]["mean"] * st["B"]["mean"] > 0
    verdict = ("✓" if st["V"]["mean"] > 0 else "✗（val 反向）") if ok_dev else "✗（dev）"
    print(f"FIB：費氏組 {FIB} vs 對照 {FIB_CTRL}  [{time.time()-t0:.0f}s]")
    for p in ("A", "B", "dev", "V"):
        s = st[p]
        print(f"  {p:<3} 月差 {s['mean']*100:+.3f}%  t {s['t']:+.2f}  ({s['n']} 月)")
    print(f"  判定：{verdict}")
    prof = {m: (np.mean(v) * 100, len(v)) for m, v in by_m.items() if v}
    print("  距 60 個月低點月數 m 的次月超額（%，全期 A+B+V）：")
    print("  " + "  ".join(f"{m}{'*' if m in FIB else ''}:{v:+.2f}" for m, (v, n) in prof.items()))
    with open(H.LOG, "a") as fh:
        fh.write(json.dumps(dict(ts=time.strftime("%Y-%m-%d %H:%M:%S"), kind="fib_event", period="dev（A/B）＋ val（V）",
                                 strategy="FS5 FIB 費氏時間轉折", note="第四十二批 事前登記", verdict=verdict,
                                 stats={p: {k: round(float(v), 5) for k, v in s.items()} for p, s in st.items()},
                                 profile={int(m): round(float(v), 4) for m, (v, n) in prof.items()}), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    if sys.argv[1:2] == ["fib"]:
        fib_test()
