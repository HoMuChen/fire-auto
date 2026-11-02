"""第二十八批 H128~H130：涂敏峰（超越巔峰）影片口訣，規則見 HYPOTHESES.md 第二十八批。

  H128 / H128b  H29 + 5 日乖離 > 20% / > 13% 排到最後（橫斷面，對照 H29）
  H129          量價不同時到頂：最大量月 K 的高點之後 12 個月內被突破的機率 vs 同月同 σ 距離基準
  H130          月 KD(9,3,3) 50 以上黃金交叉，持有 60 日（harness 事件型）＋「聯一光第二」梯度 L2~L5

用法：python research/v2/batch28_tu.py dev|val [h128 h129 h130]
"""
import hashlib
import inspect
import json
import multiprocessing as mp
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402
import hypotheses as Hy  # noqa: E402
import xs  # noqa: E402

LEAK_SIDS = ("2330", "2603", "1101", "3481", "6550")
H129_T = {"dev": ("2011-01", "2018-12"), "val": ("2020-01", "2022-12")}
HS = (3, 6, 12)
HOLD = 60
BASE = H._base_name(None, HOLD)
warnings.filterwarnings("ignore", category=RuntimeWarning)   # nanmax / nanmean 遇到全 NaN 片段（停牌月）


# ───────── 共用 ─────────
def months(df):
    """日 K（load_stock 輸出，index 0..n−1）→ 日曆月 K（沒有成交的月為 NaN）。
    high / low / close 為還原價，raw_* 為原始價，vol 為當月原始成交量合計，first / last 為該月第一 / 最後一根日 K 的位置。"""
    per = df.date.dt.to_period("M")
    pos = pd.Series(np.arange(len(df)), index=df.index)
    g = df.groupby(per)
    M = pd.DataFrame({"high": g.high.max(), "low": g.low.min(), "close": g.close.last(),
                      "raw_high": g.raw_high.max(), "raw_close": g.raw_close.last(),
                      "vol": g.volume.sum().astype(float),
                      "first": pos.groupby(per).first(), "last": pos.groupby(per).last()})
    return M.reindex(pd.period_range(M.index.min(), M.index.max(), freq="M"))


def nw_t(x, lag):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return np.nan
    e = x - x.mean()
    s = e @ e / n
    for k in range(1, min(lag, n - 1) + 1):
        s += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    return x.mean() / np.sqrt(s / n)


def _log(rec):
    with open(H.LOG, "a") as f:
        f.write(json.dumps(dict(ts=time.strftime("%Y-%m-%d %H:%M:%S"), **rec), ensure_ascii=False,
                           default=lambda v: round(float(v), 5)) + "\n")


def _pool(fn, sids, **kw):
    with mp.get_context("fork").Pool(8, initializer=_W.update, initargs=(kw,)) as pool:
        return pool.map(fn, sids, chunksize=16)


_W = {}


# ───────── H128：H29 + 乖離排除 ─────────
def _bias5(P):
    C = P["close"]
    return C / C.rolling(5, min_periods=5).mean() - 1


def h128_bias20(P):
    s = Hy.h29_double_high(P)
    return s.mask(_bias5(P) > 0.20, s - 2)


def h128b_bias13(P):
    s = Hy.h29_double_high(P)
    return s.mask(_bias5(P) > 0.13, s - 2)


def _cmp(Ra, Rb):
    a, b = Ra["topn"].dropna(), Rb["topn"].dropna()
    ix = a.index.intersection(b.index)
    d = a[ix] - b[ix]
    by = d.groupby(d.index.year).sum()
    return dict(diff_m=d.mean(), t=d.mean() / d.std() * np.sqrt(len(d)) if d.std() > 0 else np.nan,
                yrs_won=f"{int((by > 0).sum())}/{len(by)}", months_diff=int((d.abs() > 1e-12).sum()), n=len(d))


def diag128(P, p0, p1):
    """每個換股日 × pit 池：前一日乖離、是否在 H29 top20、換股日開盤起次 5 日與次月超額（減 pit 等權）。"""
    O, C = P["open"], P["close"]
    Cf = C.ffill()
    idx = O.index
    rd = xs.rebalance_dates(idx, pd.Timestamp(p0), pd.Timestamp(p1))
    s29, b = Hy.h29_double_high(P).shift(1), _bias5(P).shift(1)
    elig = P["liq"].shift(1).fillna(False).astype(bool) & (P["nbars"].shift(1) >= 250) & O.notna()
    rows = []
    for d, d2 in zip(rd[:-1], rd[1:]):
        e = elig.loc[d]
        names = e.index[e]
        k, k2 = idx.get_loc(d), idx.get_loc(d2)
        k5 = min(k + 5, len(idx) - 1)
        o0 = O.loc[d, names]
        r1 = (O.loc[d2, names].fillna(Cf.iloc[k2 - 1][names]) / o0 - 1).clip(-0.95, 10)
        r5 = (O.iloc[k5][names].fillna(Cf.iloc[k5 - 1][names]) / o0 - 1).clip(-0.95, 10)
        top = set(s29.loc[d][e].dropna().nlargest(20).index)
        rows.append(pd.DataFrame({"date": d, "sid": names, "bias": b.loc[d, names].values,
                                  "ex1": (r1 - r1.mean()).values, "ex5": (r5 - r5.mean()).values,
                                  "top": [s in top for s in names]}))
    D = pd.concat(rows, ignore_index=True)
    D["top"] = D["top"].astype(bool)
    return D


def _grp(x, lab):
    if not len(x):
        return f"  {lab:<26} n=0"
    m1, m5 = x.groupby("date").ex1.mean(), x.groupby("date").ex5.mean()
    t1 = m1.mean() / m1.std() * np.sqrt(len(m1)) if len(m1) > 2 else np.nan
    t5 = m5.mean() / m5.std() * np.sqrt(len(m5)) if len(m5) > 2 else np.nan
    return (f"  {lab:<26} n={len(x):>7} 月={x.date.nunique():>3}  次5日超額 {x.ex5.mean()*100:+5.2f}%（t {t5:+.2f}）"
            f"  次月超額 {x.ex1.mean()*100:+5.2f}%（t {t1:+.2f}）")


def run128(period):
    p0, p1 = H.PERIODS[period]
    fns = {"H128": (h128_bias20, "H29+5日乖離>20%排最後"), "H128b": (h128b_bias13, "H29+5日乖離>13%排最後"),
           "H29": Hy.FNS["H29"]}
    res, P = xs.run(fns, period, note="第二十八批 涂敏峰 乖離排除（事前，對照 H29）", topn=20)
    print(f"\n== H128 [{period}] top20（年化、%）")
    for k, (R, st) in res.items():
        s = st["topn"]
        print(f"  {k:<6} CAGR {s['CAGR']*100:5.1f}  超額 {s['ex_ann']*100:5.1f}  t {s['t']:.2f}  正年 {s['yrs_pos']}"
              f"  MDD {s['MDD']*100:5.1f}  換手 {s['turnover']*100:4.1f}")
    out = {}
    for k in ("H128", "H128b"):
        c = _cmp(res[k][0], res["H29"][0])
        out[k] = c
        ok = c["diff_m"] > 0 and c["t"] >= 2
        print(f"  {k} − H29：月差 {c['diff_m']*100:+.3f}%  t {c['t']:+.2f}  年勝 {c['yrs_won']}"
              f"  （{c['months_diff']}/{c['n']} 個月持股不同）→ {'✓' if ok else '✗'}")
    Dg = diag128(P, p0, p1)
    top = Dg[Dg["top"]]
    print("\nH29 原始 top20：每月超過門檻的檔數")
    for thr in (0.20, 0.13):
        n = top.assign(f=top.bias > thr).groupby("date").f.sum()
        print(f"  > {thr:.0%}：平均 {n.mean():.2f} 檔／月，{int((n > 0).sum())}/{len(n)} 個月至少 1 檔")
    print("H29 top20 內：被剔除 vs 留下")
    print(_grp(top[top.bias > 0.20], "乖離 > 20%（H128 剔除）"))
    print(_grp(top[(top.bias > 0.13) & (top.bias <= 0.20)], "乖離 13~20%"))
    print(_grp(top[top.bias <= 0.13], "乖離 ≤ 13%"))
    print("pit 池全體（描述）")
    print(_grp(Dg[Dg.bias > 0.20], "乖離 > 20%"))
    print(_grp(Dg[(Dg.bias > 0.13) & (Dg.bias <= 0.20)], "乖離 13~20%"))
    print(_grp(Dg[Dg.bias <= 0.13], "乖離 ≤ 13%"))
    _log(dict(kind="xs_cmp", period=period, strategy="H128 vs H29", note="第二十八批", **{k: v for k, v in out.items()}))
    return Dg


# ───────── H128x（事後描述，不判定）：逐日「5 日乖離首次超過門檻」之後的走勢 ─────────
# H128 在月換股日幾乎碰不到乖離 > 20%（約要連 6 根漲停），沒有測到「乖離大會被迫回檔」本身 → 看過 H128 結果後補的描述。
def _bias_first(bars, thr):
    c = pd.Series([b["close"] for b in bars], dtype=float)
    b = c / c.rolling(5).mean() - 1
    x = (b > thr) & ~(b.shift() > thr)
    return ["buy" if v else None for v in x.values]


def h128x_20(bars):
    return _bias_first(bars, 0.20)


def h128x_13(bars):
    return _bias_first(bars, 0.13)


SPECS128X = {
    "H128x 乖離>20%首日 持有5日": {"fn": h128x_20, "max_hold": 5, "exit_on_sell": False, "desc": "事後描述：5日乖離首次>20%，隔日開盤買，持有5日"},
    "H128x 乖離>20%首日 持有20日": {"fn": h128x_20, "max_hold": 20, "exit_on_sell": False, "desc": "事後描述：同上，持有20日"},
    "H128x 乖離>13%首日 持有5日": {"fn": h128x_13, "max_hold": 5, "exit_on_sell": False, "desc": "事後描述：5日乖離首次>13%，持有5日"},
    "H128x 乖離>13%首日 持有20日": {"fn": h128x_13, "max_hold": 20, "exit_on_sell": False, "desc": "事後描述：同上，持有20日"},
}


def run128x(period):
    T, S = H.run(SPECS128X, period, note="第二十八批 H128 事後描述（不判定）")
    f = S.copy()
    for c in ("win", "avg", "ex", "edge"):
        f[c] = (f[c] * 100).round(2)
    pd.set_option("display.width", 250)
    print(f"\n== H128x [{period}]（事後描述，不判定；ex＝減 pit 等權、edge＝再減同持有期隨機進場）")
    print(f.round(2).to_string(index=False))
    return T, S


# ───────── H129：最大量月 K 高點會不會被突破 ─────────
def h129_stock(sid):
    """回傳 (pool rows, event rows)。pool：月底 t 的 pit 股（σ、之後 h 月最高 / C_t、12 月報酬）；event：最大量月高點事件。"""
    df = H.load_stock(sid, _W["end"])
    if df is None or len(df) < 260:
        return [], []
    M = months(df)
    liq = (df.volume.rolling(H.LIQ_WIN).mean() >= H.LIQ_SHARES).values
    sig = np.log(df.close).diff().rolling(250, min_periods=200).std().values
    hi, cl, vol, rh = M.high.values, M.close.values, M.vol.values, M.raw_high.values
    last = M["last"].values
    per = M.index
    t0, t1 = pd.Period(_W["t0"], "M"), pd.Period(_W["t1"], "M")
    n = len(M)
    pool, ev, used = [], [], set()

    def fmax(a, k, h):
        w = a[k + 1:k + 1 + h]
        return np.nanmax(w) if np.isfinite(w).any() else np.nan

    for k in range(n):
        if not (t0 <= per[k] <= t1) or not np.isfinite(last[k]):
            continue
        p = int(last[k])
        if p < 249 or not liq[p] or not np.isfinite(sig[p]):
            continue
        fm = {h: fmax(hi, k, h) for h in HS}
        fc = cl[k + 1:k + 13]
        fc = fc[np.isfinite(fc)]
        r12 = fc[-1] / cl[k] - 1 if len(fc) else np.nan
        pool.append((per[k], sid, sig[p], *(fm[h] / cl[k] for h in HS), r12))
        if k < 11:
            continue
        w = vol[k - 11:k + 1]
        if not np.isfinite(w).any():
            continue
        v = k - int(np.nanargmax(w[::-1]))          # 同量取較近的月
        hv = hi[v]
        if v in used or not (cl[k] < hv) or (v < k and np.nanmax(hi[v + 1:k + 1]) > hv):
            continue
        used.add(v)
        rmax = fmax(rh, k, 12)
        ev.append((per[k], sid, per[v], hv / cl[k], vol[v] / np.nanmean(w), sig[p],
                   *(bool(fm[h] > hv) for h in HS), bool(rmax > rh[v]) if np.isfinite(rmax) else False, r12))
    return pool, ev


def leak129(period):
    """事件特徵（v、距離、量比）只能用到 t 以前：截斷資料重算須一致。"""
    bad = []
    for sid in LEAK_SIDS:
        _W.update(end=H.PERIODS[period][1], t0=H129_T[period][0], t1=H129_T[period][1])
        _, full = h129_stock(sid)
        f = {(e[0], e[2]): e[3:5] for e in full}
        for cut in ("2013-06", "2016-03", "2018-09"):
            _W.update(end=(pd.Period(cut, "M") + 1).to_timestamp() - pd.Timedelta(days=1), t1=cut)
            _, part = h129_stock(sid)
            for e in part:
                g = f.get((e[0], e[2]))
                if g is None or not np.allclose(g, e[3:5], equal_nan=True):
                    bad.append((sid, cut, str(e[0])))
    return bad


def run129(period):
    t0, t1 = H129_T[period]
    end = H.PERIODS[period][1]
    if period == "dev":
        bad = leak129(period)
        if bad:
            raise RuntimeError(f"H129 前視檢查失敗 {bad[:5]}")
    res = _pool(h129_stock, H.universe(), end=end, t0=t0, t1=t1)
    Pl = pd.DataFrame([r for p, _ in res for r in p], columns=["t", "sid", "sig", *(f"m{h}" for h in HS), "r12"])
    E = pd.DataFrame([r for _, e in res for r in e],
                     columns=["t", "sid", "v", "dist", "vratio", "sig", *(f"hit{h}" for h in HS), "hit12_raw", "r12"])
    for h in HS:
        Pl[f"z{h}"] = np.log(Pl[f"m{h}"].fillna(0).clip(lower=1e-12)) / Pl.sig
    E["zd"] = np.log(E.dist) / E.sig
    for h in HS:
        E[f"b{h}"], E[f"bp{h}"] = np.nan, np.nan
    for t, g in Pl.groupby("t"):
        ii = E.index[E.t == t]
        if not len(ii):
            continue
        for h in HS:
            z, m = np.sort(g[f"z{h}"].values), np.sort(g[f"m{h}"].fillna(0).values)
            E.loc[ii, f"b{h}"] = 1 - np.searchsorted(z, E.loc[ii, "zd"].values, side="right") / len(z)
            E.loc[ii, f"bp{h}"] = 1 - np.searchsorted(m, E.loc[ii, "dist"].values, side="right") / len(m)
    pm = Pl.groupby("t").r12.mean()
    E["ex12"] = E.r12 - E.t.map(pm)
    E["year"] = E.t.dt.year
    print(f"\n== H129 [{period}] 事件月 {t0}~{t1}：{len(E):,} 個事件（{E.sid.nunique()} 檔），pit 池 {len(Pl):,} 檔月")
    print(f"  最大量月＝當月（v = t）{(E.v == E.t).mean()*100:.0f}%；距離 Hv/C 中位 {(E.dist.median()-1)*100:.1f}%"
          f"（{(E.zd.median()):.1f} 個日 σ）；量比中位 {E.vratio.median():.1f} 倍")
    stats = {}
    for h in HS:
        d = E[f"hit{h}"].astype(float) - E[f"b{h}"]
        m = d.groupby(E.t).mean()
        yr = d.groupby(E.year).mean()
        st = dict(n=len(E), hit=E[f"hit{h}"].mean(), base=E[f"b{h}"].mean(), base_plain=E[f"bp{h}"].mean(),
                  diff=d.mean(), nw_t=nw_t(m.values, h), yrs_pos=f"{int((yr > 0).sum())}/{len(yr)}")
        stats[f"h{h}"] = st
        print(f"  h={h:>2} 月：過 {st['hit']*100:5.1f}%  基準（同 σ 距離）{st['base']*100:5.1f}%  差 {st['diff']*100:+5.1f}pp"
              f"  NW t {st['nw_t']:+.2f}  正年 {st['yrs_pos']}   ｜不除 σ 的同距離基準 {st['base_plain']*100:5.1f}%")
    s = stats["h12"]
    need = "7/8" if period == "dev" else "2/3"
    ny, ty = (int(x) for x in s["yrs_pos"].split("/"))
    ok = s["diff"] >= 0.10 and s["nw_t"] >= 2 and ny >= int(need.split("/")[0]) * ty / int(need.split("/")[1]) - 1e-9
    print(f"  主判定 h=12：差 {s['diff']*100:+.1f}pp（門檻 +10pp）、NW t {s['nw_t']:+.2f}（≥ 2）、正年 {s['yrs_pos']}（≥ {need}）→ {'✓' if ok else '✗'}")
    print("  描述：")
    print(f"    他說「90% 會過」：12 個月內實際 {s['hit']*100:.1f}%；用原始價 {E.hit12_raw.mean()*100:.1f}%")
    for lab, x in (("量比 ≥ 2 倍", E[E.vratio >= 2]), ("量比 < 2 倍", E[E.vratio < 2]),
                   ("v = t（最大量就是本月）", E[E.v == E.t]), ("v < t（最大量在之前）", E[E.v < E.t])):
        dd = x.hit12.astype(float) - x.b12
        print(f"    {lab:<20} n={len(x):>6}  過 {x.hit12.mean()*100:5.1f}%  基準 {x.b12.mean()*100:5.1f}%  差 {dd.mean()*100:+5.1f}pp")
    m = E.groupby("t").ex12.mean()
    print(f"    事件股之後 12 個月報酬 − 同月池平均：{E.ex12.mean()*100:+.2f}%（月平均 NW t {nw_t(m.values, 12):+.2f}；窗重疊、只描述）")
    print("    年別 h=12 差（pp）：" + "  ".join(f"{y} {v*100:+.1f}" for y, v in
                                              (E.hit12.astype(float) - E.b12).groupby(E.year).mean().items()))
    _log(dict(kind="event_prob", period=period, strategy="H129 量價不同時到頂", note="第二十八批 事前登記",
              events=len(E), pass_=ok, **stats))
    return E, Pl


# ───────── H130：月 KD 50 以上金叉 ＋ 聯一光第二梯度 ─────────
def _pbr_eps(sid, dates):
    pbr = np.full(len(dates), np.nan)
    f = H.D / "per" / f"{sid}.parquet"
    if f.exists():
        d = pd.read_parquet(f)
        if len(d):
            s = d.assign(date=pd.to_datetime(d.date)).drop_duplicates("date", keep="last").set_index("date").PBR.astype(float).sort_index()
            pbr = s.reindex(dates, method="ffill", tolerance=pd.Timedelta(days=7)).values
    eps = np.full(len(dates), np.nan)
    q = xs._pivot(H.D / "fin" / f"{sid}.parquet")
    if q is not None and "EPS" in q:
        qi = pd.date_range(q.index.min(), q.index.max(), freq="QE")
        ttm = q.EPS.reindex(qi).rolling(4, min_periods=4).sum()
        a = xs._align(pd.Series(ttm.values, index=xs._q_avail(qi)), dates, stale=200)
        if a is not None:
            eps = a.values
    return pbr, eps


def feats130(sid, df):
    """日 K 長度的布林陣列：月 m 的條件成立 → 在 m+1 月第一根日 K 為 True（隔天開盤進場）。"""
    n = len(df)
    M = months(df)
    A = M.dropna(subset=["close"])
    hh, ll = A.high.rolling(9).max(), A.low.rolling(9).min()
    rg = hh - ll
    rsv = ((A.close - ll) / rg * 100).where(rg > 0, 50.0).where(hh.notna()).values
    K, D = np.full(len(A), np.nan), np.full(len(A), np.nan)
    kp = dp = 50.0
    start = None
    for i, r in enumerate(rsv):
        if np.isnan(r):
            continue
        start = i if start is None else start
        kp = kp * 2 / 3 + r / 3
        dp = dp * 2 / 3 + kp / 3
        K[i], D[i] = kp, dp
    valid = np.zeros(len(A), bool)
    if start is not None:
        valid[start + 12:] = True
    Kp, Dp = np.r_[np.nan, K[:-1]], np.r_[np.nan, D[:-1]]
    with np.errstate(invalid="ignore"):
        cross = valid & (Kp <= Dp) & (K > D)
        maxvol = (M.vol >= M.vol.rolling(12, min_periods=1).max()).reindex(A.index).fillna(False).values
        above60 = (A.close > A.close.rolling(60).mean()).values
    nxt = M["first"].shift(-1).reindex(A.index).values           # m+1 月第一根日 K（m+1 月沒成交 → NaN，不出訊號）
    dates = pd.DatetimeIndex(df.date)
    pbr, eps = _pbr_eps(sid, dates)
    out = {k: np.zeros(n, bool) for k in ("H130", "H130c", "H130d", "L2", "L3", "L4", "L5")}
    for i in np.flatnonzero(cross & np.isfinite(nxt)):
        s = int(nxt[i])
        gc50 = D[i] >= 50
        out["H130d"][s] = True
        out["H130c"][s] = not gc50
        if not gc50:
            continue
        out["H130"][s] = True
        l2 = bool(maxvol[i])
        l3 = l2 and 0 < pbr[s] < 1
        l4 = l3 and eps[s] > 0
        out["L2"][s], out["L3"][s], out["L4"][s] = l2, l3, l4
        out["L5"][s] = l4 and bool(above60[i])
    return out


SPECS130 = {
    "H130 月KD≥50金叉 持有60日": ("H130", "月KD(9,3,3)金叉且D≥50，m+1月第一根K訊號、隔日開盤買，持有60日"),
    "H130c 月KD<50金叉 60日": ("H130c", "月KD金叉且D<50，持有60日（描述）"),
    "H130d 任何月KD金叉 60日": ("H130d", "任何月KD金叉，持有60日（描述）"),
    "H130-L2 +最大量": ("L2", "H130且金叉月為近12月最大量"),
    "H130-L3 +PBR<1": ("L3", "L2且訊號日PBR<1"),
    "H130-L4 +近4季EPS>0": ("L4", "L3且近4季EPS合計>0"),
    "H130-L5 +站上60月線": ("L5", "L4且月收>60月均線"),
}


def h130_stock(sid):
    df = H.load_stock(sid, _W["end"])
    if df is None or len(df) < H.MIN_BARS + 2:
        return []
    F = feats130(sid, df)
    out = []
    for name, (key, _) in SPECS130.items():
        sig = ["buy" if b else None for b in F[key]]
        out += [(name, sid) + t for t in H.simulate(df, sig, None, HOLD, False, _W["period"], _W["end"])]
    out += [(BASE, sid) + t for t in H.simulate(df, H._random_sig(sid, len(df)), None, HOLD, False, _W["period"], _W["end"])]
    return out


def leak130(end):
    bad = []
    for sid in LEAK_SIDS:
        df = H.load_stock(sid, end)
        if df is None:
            continue
        full = feats130(sid, df)
        for q in (0.4, 0.7, 0.9):
            k = int(len(df) * q)
            part = feats130(sid, df.iloc[:k].copy())
            for key in full:
                if (part[key] != full[key][:k]).any():
                    bad.append((sid, q, key))
    return bad


def run130(period):
    p0, p1 = H.PERIODS[period]
    end = min(pd.Timestamp(p1), pd.Timestamp.today()).strftime("%Y-%m-%d")
    if period == "dev":
        bad = leak130(end)
        if bad:
            raise RuntimeError(f"H130 前視檢查失敗 {bad[:5]}")
    sids = H.universe()
    t_ = time.time()
    res = _pool(h130_stock, sids, end=end, period=(p0, p1))
    T = pd.DataFrame([t for ts in res for t in ts], columns=["name", "sid", "buy", "sell", "ret", "reason", "bars"])
    mkt, groups = H.market_and_groups(sids, end)
    T["mret"] = mkt.reindex(T.sell).values / mkt.reindex(T.buy).values - 1
    T["ex"] = T.ret - T.mret
    T["year"] = T.buy.dt.year
    T["group"] = [groups.get(y, pd.Series(dtype=str)).get(s) for y, s in zip(T.year, T.sid)]
    specs = {name: {"fn": feats130, "max_hold": HOLD, "exit_on_sell": False, "desc": desc} for name, (_, desc) in SPECS130.items()}
    S = H.summarize(T, specs)
    H._log(S, specs, period, "第二十八批 涂敏峰 月KD≥50金叉＋聯一光第二梯度（事前登記）")
    b = T[T.name == BASE]
    print(f"\n== H130 [{period}] {len(sids)} 檔、{len(T):,} 筆，{time.time()-t_:.0f}s；隨機對照（持有 {HOLD} 日）"
          f"n={len(b):,} 每筆超額 {b.ex.mean()*100:+.2f}%")
    f = S.copy()
    for c in ("win", "avg", "ex", "edge"):
        f[c] = (f[c] * 100).round(2)
    pd.set_option("display.width", 250)
    print(f.round(2).to_string(index=False))

    def judge(r, tmin):
        ny, ty = (int(x) for x in r["yrs_pos"].split("/"))
        return r["n"] > 0 and r["ex"] > 0 and r["edge"] > 0 and r["t_month"] >= tmin and ny >= 0.7 * ty

    r = S.set_index("strategy").loc["H130 月KD≥50金叉 持有60日"]
    print(f"\n主判定 H130：每筆超額 {r['ex']*100:+.2f}%、edge {r['edge']*100:+.2f}pp、月 t {r['t_month']:.2f}、正年 {r['yrs_pos']}"
          f" → {'✓' if judge(r, 2) else '✗'}")
    lad = [k for k in SPECS130 if k.startswith("H130-L")]
    deep = [k for k in lad if S.set_index("strategy").loc[k, "n"] >= 100]
    if deep:
        k = deep[-1]
        r = S.set_index("strategy").loc[k]
        print(f"梯度最深層（n ≥ 100）{k}：n {int(r['n'])}、超額 {r['ex']*100:+.2f}%、edge {r['edge']*100:+.2f}pp、t {r['t_month']:.2f}"
              f"、正年 {r['yrs_pos']} → {'進 val' if judge(r, 2.5) else '✗'}")
    out = H.D / "results"
    out.mkdir(exist_ok=True)
    T.to_parquet(out / f"b28_h130_{period}.parquet")
    return T, S


if __name__ == "__main__":
    period = sys.argv[1] if len(sys.argv) > 1 else "dev"
    if period == "holdout":
        raise PermissionError("holdout 封存中")
    want = [a.lower() for a in sys.argv[2:]] or ["h128", "h129", "h130"]
    for w in want:
        {"h128": run128, "h128x": run128x, "h129": run129, "h130": run130}[w](period)
