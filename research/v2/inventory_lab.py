"""第六十七批：存貨三個方向（H176 存貨週轉天數、H177 同產業庫存循環、H178 全市場庫存擇時）。規則見 HYPOTHESES.md 第六十七批（事前登記）。
選股引擎沿用第六十六批（combo_search.Eng、快取 combo_{period}.pkl）。

用法：.venv/bin/python research/v2/inventory_lab.py leak     # 防前視：截斷季資料重算，月特徵要一致
      .venv/bin/python research/v2/inventory_lab.py dev
      .venv/bin/python research/v2/inventory_lab.py val
      .venv/bin/python research/v2/inventory_lab.py coverage；holdout --unseal（使用者同意後；只測 val 過的 3 個）
"""
import itertools
import json
import math
import multiprocessing as mp
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bpool_lab as BL  # noqa: E402
import combo_search as C  # noqa: E402
import feature_industry as FI  # noqa: E402
import harness as H  # noqa: E402
import strategy_lab as SL  # noqa: E402
import xs  # noqa: E402

RES = H.D / "results"
NEW = {"DIO_LVL": -1, "DIO_CHG": -1, "REL_DIO": -1, "REL_DIO_CHG": -1, "IND_DIO_CHG": -1, "IND_DIO_TURN": -1, "IND_DSDI": 1}
IND_FEATS = ("IND_DIO_CHG", "IND_DIO_TURN", "IND_DSDI")
STALE, MINF, TOPQ, MIN_M, POS_MIN, NSEL = 200, 5, 0.2, 60, 5 / 6, 10
DEV0 = "2014-06-01"
C.SEG["d67"] = (DEV0, "2019-12-31")
BUY_E, SELL_E, FUT = 0.001425, 0.002425, 0.0005
NAME = "第六十七批 存貨三方向"
TRIALS = 1701277
SIGNALS = {"DSDI": ("DSDI", 1), "DIO_CHG": ("DIO_CHG", -1), "DIO_TURN": ("DIO_TURN", -1)}   # 方向：值 × 方向 > 0 → 做多


# ═════════ 季資料與特徵 ═════════
def load_raw():
    f = RES / "inv_raw.pkl"
    if f.exists():
        with open(f, "rb") as fh:
            return pickle.load(fh)
    ids = pd.read_csv(H.D / "universe.csv", dtype=str).stock_id.tolist()
    qi = pd.date_range("2009-03-31", "2026-12-31", freq="QE")
    inv, cogs, rev = {}, {}, {}
    for s in ids:
        q, b = xs._pivot(H.D / "fin" / f"{s}.parquet"), xs._pivot(H.D / "bs" / f"{s}.parquet")
        if q is None or b is None or "Inventories" not in b or "CostOfGoodsSold" not in q:
            continue
        inv[s] = b["Inventories"].reindex(qi)
        cogs[s] = q["CostOfGoodsSold"].reindex(qi)
        rev[s] = q["Revenue"].reindex(qi) if "Revenue" in q else pd.Series(np.nan, index=qi)
    raw = dict(INV=pd.DataFrame(inv), COGS=pd.DataFrame(cogs), REV=pd.DataFrame(rev))
    with open(f, "wb") as fh:
        pickle.dump(raw, fh)
    return raw


def _agg(INV, COGS, REV, cols):
    """同樣本加總：週轉天數年變化、拐點、營收成長 − 存貨成長（≥ MINF 家）。"""
    I, Cg, Rv = INV[cols], COGS[cols], REV[cols]
    ok = I.notna() & Cg.notna() & I.shift(4).notna() & Cg.shift(4).notna()
    now = I.where(ok).sum(axis=1) / Cg.where(ok).sum(axis=1) * 365
    prev = I.shift(4).where(ok).sum(axis=1) / Cg.shift(4).where(ok).sum(axis=1) * 365
    chg = (now - prev).where(ok.sum(axis=1) >= MINF)
    ok2 = Rv.notna() & Rv.shift(4).notna() & I.notna() & I.shift(4).notna()
    dsdi = ((Rv.where(ok2).sum(axis=1) / Rv.shift(4).where(ok2).sum(axis=1) - 1)
            - (I.where(ok2).sum(axis=1) / I.shift(4).where(ok2).sum(axis=1) - 1)).where(ok2.sum(axis=1) >= MINF)
    return dict(DIO_CHG=chg, DIO_TURN=chg - chg.shift(1), DSDI=dsdi)


def derive(raw, cut=None):
    """季 × 股的 7 個特徵與大盤序列。cut：只用可得日 ≤ cut 的季資料（防前視檢查用）。"""
    INV, COGSq, REVq = raw["INV"].copy(), raw["COGS"].copy(), raw["REV"].copy()
    A = xs._q_avail(INV.index)
    if cut is not None:
        late = A > pd.Timestamp(cut)
        for X in (INV, COGSq, REVq):
            X.loc[late] = np.nan
    INVp = INV.where(INV > 0)
    COGS = COGSq.rolling(4, min_periods=4).sum()
    COGS = COGS.where(COGS > 0)
    REV = REVq.rolling(4, min_periods=4).sum()
    REV = REV.where(REV > 0)
    DIO = INVp / COGS * 365
    F = {"DIO_LVL": DIO, "DIO_CHG": DIO - DIO.shift(4)}
    cols = DIO.columns
    lab = pd.read_csv(H.D / "universe.csv", dtype=str).set_index("stock_id").industry.reindex(cols).replace(FI.MERGE)
    codes = FI.labels(cols)
    for k in ("REL_DIO", "REL_DIO_CHG", *IND_FEATS):
        F[k] = pd.DataFrame(np.nan, index=DIO.index, columns=cols)
    for g in np.unique(codes[codes >= 0]):
        gc = cols[codes == g]
        for src, dst in ((F["DIO_LVL"], "REL_DIO"), (F["DIO_CHG"], "REL_DIO_CHG")):
            x = src[gc]
            med = x.median(axis=1).where(x.notna().sum(axis=1) >= MINF)
            F[dst][gc] = x.sub(med, axis=0)
        a = _agg(INVp, COGS, REV, gc)
        for k, s in (("IND_DIO_CHG", "DIO_CHG"), ("IND_DIO_TURN", "DIO_TURN"), ("IND_DSDI", "DSDI")):
            F[k][gc] = np.repeat(a[s].values[:, None], len(gc), axis=1)
    mkt = {"全市場": _agg(INVp, COGS, REV, cols), "半導體業": _agg(INVp, COGS, REV, cols[(lab == "半導體業").values])}
    return F, mkt


def monthly(M, dates, cols, days):
    """季矩陣（index＝季底）→ 換股日矩陣：用換股日前一個交易日已可得的最新一季；超過 STALE 天沒更新視為缺。"""
    A = xs._q_avail(M.index).values
    V = M.reindex(columns=cols)
    L = pd.DataFrame(np.where(V.notna().values, A[:, None], np.datetime64("NaT")), index=M.index, columns=cols).ffill().values
    Vf = V.ffill().values
    out = np.full((len(dates), len(cols)), np.nan)
    dv = days.values
    for t, d in enumerate(dates):
        j = np.searchsorted(dv, np.datetime64(d), side="left") - 1
        if j < 0:
            continue
        dp = dv[j]
        k = np.searchsorted(A, dp, side="right") - 1
        if k < 0:
            continue
        ok = (dp - L[k]) <= np.timedelta64(STALE, "D")
        out[t] = np.where(ok, Vf[k], np.nan)
    return out


def trading_days():
    tr = pd.read_parquet(H.D / "taiex_tr.parquet")
    return pd.DatetimeIndex(pd.to_datetime(tr.date)).drop_duplicates().sort_values()


def leak():
    raw, days = load_raw(), trading_days()
    F, mkt = derive(raw)
    cols = F["DIO_LVL"].columns
    for cut in ("2015-06-30", "2017-09-29", "2019-03-29"):
        Ft, mt = derive(raw, cut)
        dates = pd.DatetimeIndex(xs.rebalance_dates(days, pd.Timestamp("2013-01-01"), pd.Timestamp(cut)))
        bad = []
        for k in NEW:
            a, b = monthly(F[k], dates, cols, days), monthly(Ft[k], dates, cols, days)
            same = (np.isclose(a, b, rtol=1e-9, atol=1e-12) | (np.isnan(a) & np.isnan(b)))
            if not same.all():
                bad.append((k, int((~same).sum())))
        for nm in mkt:
            for s in SIGNALS:
                a = monthly(mkt[nm][s].to_frame("x"), dates, ["x"], days)
                b = monthly(mt[nm][s].to_frame("x"), dates, ["x"], days)
                if not (np.isclose(a, b) | (np.isnan(a) & np.isnan(b))).all():
                    bad.append((nm + s, 1))
        print(f"截斷 {cut}：{len(dates)} 個換股日；{'OK' if not bad else bad}", flush=True)
    yr = F["DIO_LVL"].index.year
    for k in NEW:
        n = F[k].notna().sum(axis=1)
        print(f"  {k}：每季有值檔數 2013 {int(n[yr == 2013].median())}、2015 {int(n[yr == 2015].median())}、2019 {int(n[yr == 2019].median())}"
              f"；2018 分位 p1/p50/p99 {np.nanpercentile(F[k][yr == 2018], 1):.3g}/{np.nanpercentile(F[k][yr == 2018], 50):.3g}/{np.nanpercentile(F[k][yr == 2018], 99):.3g}")


# ═════════ 選股（H176／H177）═════════
def setup(period):
    B = C.load(period)
    lib = list(B["X"])
    days = trading_days()
    F, mkt = derive(load_raw())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for f in NEW:
            B["X"][f] = np.where(B["E"], monthly(F[f], B["dates"], B["cols"], days), np.nan).astype(np.float32)
    eng = C.Eng(B)
    eng.sign.update(NEW)
    return B, eng, lib, mkt, days


_G = {}


def _stats(ex, net, m, yrs, min_m=MIN_M):
    k = m & np.isfinite(ex)
    e, x, y = ex[k], net[k], yrs[k]
    if len(e) < min_m:
        return (np.nan, np.nan, np.nan, np.nan, len(e))
    yr = np.bincount(y, weights=e)
    has = np.bincount(y, minlength=len(yr)) > 0
    return (e.mean() / e.std(ddof=1) * np.sqrt(len(e)), x.mean() / x.std(ddof=1) * np.sqrt(12), e.mean() * 12,
            (yr[has] > 0).sum() / has.sum(), len(e))


def _work(chunk):
    eng, m, yrs = _G["eng"], _G["m"], _G["yrs"]
    out = np.full((len(chunk) * len(C.NS), 5), np.nan, np.float32)
    for i, (parts, need) in enumerate(chunk):
        for j, (net, ex) in enumerate(eng.run_all(eng.score(parts, need))):
            out[i * len(C.NS) + j] = _stats(ex, net, m, yrs)
    return out


def scan(eng, specs, m, yrs, procs=9, size=400):
    _G.update(eng=eng, m=m, yrs=yrs)
    chunks = [specs[i:i + size] for i in range(0, len(specs), size)]
    with mp.get_context("fork").Pool(procs) as pool:
        return np.vstack(pool.map(_work, chunks, chunksize=1))


def ok_v(r):
    with np.errstate(invalid="ignore"):
        return np.isfinite(r[:, 0]) & (r[:, 0] >= 2) & (r[:, 1] >= 1.0) & (r[:, 3] >= POS_MIN - 1e-9) & (r[:, 4] >= MIN_M)


def specs(eng, lib):
    new = list(NEW)
    sg = lambda f: (f, eng.sign[f])   # noqa: E731
    singles = [(((f, s),), 1) for f in new for s in (1, -1)]
    pairs = [((sg(a), sg(b)), 2) for a in new for b in lib] + [((sg(a), sg(b)), 2) for a, b in itertools.combinations(new, 2)]
    allf = lib + new
    triples = [(tuple(sg(f) for f in c), 2) for c in itertools.combinations(allf, 3) if any(f in NEW for f in c)]
    return singles, pairs, triples


def sub_of(parts):
    rest = tuple(p for p in parts if p[0] not in NEW)
    return rest if 0 < len(rest) < len(parts) else None


def rotation(B, eng, f, sign):
    """產業輪動：有值池內股 ≥ MINF 檔的產業依分數排序，持有前 TOPQ 產業的全部池內股（等權）。"""
    X, E, T = B["X"][f], eng.E, eng.T
    codes = FI.labels(B["cols"])
    W = np.zeros((T, eng.U))
    for t in range(T):
        v = E[t] & np.isfinite(X[t]) & (codes >= 0)
        g, cnt = np.unique(codes[v], return_counts=True)
        g = g[cnt >= MINF]
        if len(g) < 5:
            continue
        val = np.array([X[t][v & (codes == x)][0] for x in g]) * sign
        top = g[np.argsort(-val, kind="stable")[:math.ceil(TOPQ * len(g))]]
        mem = v & np.isin(codes, top)
        W[t, mem] = 1 / mem.sum()
    df = SL.weights_portfolio(W, np.nan_to_num(B["R"]), E, bench=eng.bench)
    return df.net.values, df.ex.values, df.n.values


# ═════════ 擇時（H178）═════════
def tr_monthly(dates):
    """加權報酬指數的月窗報酬：換股日前一天收盤 → 下次換股日前一天收盤。"""
    tr = pd.read_parquet(H.D / "taiex_tr.parquet")
    tr["date"] = pd.to_datetime(tr.date)
    px = tr.drop_duplicates("date").set_index("date").price.astype(float).sort_index()
    rd = pd.DatetimeIndex(xs.rebalance_dates(px.index, pd.Timestamp("2010-01-01"), px.index[-1]))
    nxt = dict(zip(rd[:-1], rd[1:]))
    prev = lambda d: px[px.index < d].iloc[-1]   # noqa: E731
    return np.array([prev(nxt[d]) / prev(d) - 1 for d in dates])


def timing(B, mkt, days, seg):
    dates = B["dates"]
    r_tr = tr_monthly(dates)
    r_mx = B["mtx"]
    m = C.mask(dates, seg)
    out = []
    for nm, a in mkt.items():
        for s, (key, dr) in SIGNALS.items():
            x = monthly(a[key].to_frame("x"), dates, ["x"], days)[:, 0]
            on = np.where(np.isfinite(x), (x * dr > 0).astype(float), np.nan)
            for veh, r in (("V1 加權報酬指數 多／空手", r_tr), ("V2 小台指 多／空", r_mx)):
                pos = on if veh.startswith("V1") else np.where(np.isfinite(on), 2 * on - 1, np.nan)
                p0 = np.concatenate([[np.nan], pos[:-1]])
                dpos = np.nan_to_num(pos - p0)
                cost = (np.where(dpos > 0, BUY_E * dpos, -SELL_E * dpos) if veh.startswith("V1") else FUT * np.abs(dpos))
                s_ret = pos * r - cost
                k = m & np.isfinite(s_ret) & np.isfinite(r)
                at, a0, b1 = BL.alpha_t(s_ret, r, k, 3)
                sh = s_ret[k].mean() / s_ret[k].std(ddof=1) * np.sqrt(12)
                bh = r[k].mean() / r[k].std(ddof=1) * np.sqrt(12)
                out.append(dict(name=f"{nm} {s} {veh}", months=int(k.sum()), alpha_t=at, alpha_ann=a0 * 12, beta=b1, sharpe=sh, bh_sharpe=bh,
                                on_frac=float(np.nanmean(on[k])), switches=int((np.abs(dpos[k]) > 0).sum()),
                                verdict="✓" if at >= 2 and sh > bh else "✗"))
    return out


def show_timing(rows):
    for r in rows:
        print(f"  {r['name']}：擇時 alpha t {r['alpha_t']:+.2f}（年化 {r['alpha_ann']*100:+.1f}%，β {r['beta']:.2f}）Sharpe {r['sharpe']:.2f} vs 買進持有 {r['bh_sharpe']:.2f}"
              f"｜做多比例 {r['on_frac']:.0%}、換邊 {r['switches']} 次、{r['months']} 月 → {r['verdict']}")


# ═════════ 指令 ═════════
def dev():
    t0 = time.time()
    B, eng, lib, mkt, days = setup("dev")
    m = C.mask(eng.dates, "d67")
    yrs = (eng.dates.year - 2014).values.clip(0)
    singles, pairs, triples = specs(eng, lib)
    lib_s = [(((f, eng.sign[f]),), 1) for f in lib]
    lib_p = [(tuple((f, eng.sign[f]) for f in c), 2) for c in itertools.combinations(lib, 2)]
    print(f"準備 {time.time()-t0:.0f}s；dev {C.SEG['d67']}（{m.sum()} 個換股月）；單一 {len(singles)}、兩兩 {len(pairs)}、三三 {len(triples)} × N {C.NS}", flush=True)
    for f in NEW:
        cov = np.isfinite(B["X"][f][m])[eng.E[m]].mean()
        print(f"  {f} 池內有值比例 {cov:.0%}", flush=True)
    res = {}
    for nm, sp in (("單一", singles), ("兩兩", pairs), ("三三", triples), ("_庫單一", lib_s), ("_庫兩兩", lib_p)):
        t1 = time.time()
        res[nm] = scan(eng, sp, m, yrs)
        if not nm.startswith("_"):
            r = res[nm]
            print(f"  {nm}：{len(r)} 個（{time.time()-t1:.0f}s）；可評 {np.isfinite(r[:, 0]).sum()}；t 最高 {np.nanmax(r[:, 0]):.2f}；過 dev 四條 {ok_v(r).sum()}", flush=True)
    base = {}
    for sp, key in ((lib_s, "_庫單一"), (lib_p, "_庫兩兩")):
        for i, (parts, _) in enumerate(sp):
            for j, n in enumerate(C.NS):
                base[(frozenset(p[0] for p in parts), n)] = float(res[key][i * len(C.NS) + j, 0])
    rows = []
    for nm, sp in (("單一", singles), ("兩兩", pairs), ("三三", triples)):
        r = res[nm]
        for k in np.flatnonzero(ok_v(r)):
            parts, need = sp[k // len(C.NS)]
            n = C.NS[k % len(C.NS)]
            sub = sub_of(parts)
            tb = base.get((frozenset(p[0] for p in sub), n), np.nan) if sub else np.nan
            rows.append((nm, parts, need, n, *r[k], sub, tb))
    T = pd.DataFrame(rows, columns=["kind", "parts", "need", "n", "t", "sharpe", "ex_ann", "pos", "months", "sub", "t_sub"])
    T["contrib"] = T["sub"].isna() | ~(T.t_sub >= T.t)
    T = T.sort_values("t", ascending=False)
    print(f"\n過 dev 四條共 {len(T)} 個（單一 {(T.kind == '單一').sum()}、兩兩 {(T.kind == '兩兩').sum()}、三三 {(T.kind == '三三').sum()}）；"
          f"其中存貨有貢獻（拿掉後 t 變低）{T.contrib.sum()} 個", flush=True)
    for f in NEW:
        k = T[T.parts.apply(lambda p: any(x[0] == f for x in p)) & T.contrib]
        print(f"  含 {f}（有貢獻）：{len(k)} 個；最好 {C.spec_name(k.parts.iloc[0], k.n.iloc[0]) + f' t {k.t.iloc[0]:.2f}（拿掉後 {k.t_sub.iloc[0]:.2f}）' if len(k) else '—'}", flush=True)
    h = C.h136_series("dev", eng.dates)
    sel, sel_net, n_clone = [], [h], 0
    for _, r in T[T.contrib].iterrows():
        net, _ = eng.run_all(eng.score(r.parts, r.need))[C.NS.index(r.n)]
        rhos = [C.corr(net, z, m) for z in sel_net]
        if max(rhos) > C.RHO_MAX:
            n_clone += rhos[0] > C.RHO_MAX
            continue
        sel.append((r.kind, r.parts, r.need, r.n, r["sub"]))
        sel_net.append(net)
        if len(sel) == NSEL:
            break
    print(f"去重：H136 分身跳過 {n_clone} 個；選進 val {len(sel)} 個", flush=True)
    extra = []
    for kind, parts, need, n, sub in sel:
        d, net, nl, el = C.detail(eng, B, parts, need, n, "d67", ())
        d.update(kind=kind, rho_h136=C.corr(nl, h, m))
        if sub:
            ds = C.detail(eng, B, sub, 1 if len(sub) == 1 else 2, n, "d67", ())[0]
            d["t_sub_lim"] = ds["t_lim"]
        extra.append(d)
        print(f"  [{kind}] {d['name']}：超額 t {d['t_lim']:.2f}（拿掉存貨後 {d.get('t_sub_lim', float('nan')):.2f}）年化超額 {d['ex_ann_lim']*100:+.1f}% 逐年 {d['by_year']}"
              f"｜Sharpe {d['sharpe_lim']:.2f}｜換手 {d['turnover']:.0%}｜和 H136 {d['rho_h136']:+.2f}", flush=True)
    print("\n產業輪動（前 20% 產業、池內等權）：", flush=True)
    rot = []
    for f in IND_FEATS:
        net, ex, nn = rotation(B, eng, f, NEW[f])
        st = _stats(ex, net, m, yrs)
        ok = bool(ok_v(np.array([st]))[0])
        rot.append(dict(name=f, t=st[0], sharpe=st[1], ex_ann=st[2], pos=st[3], months=st[4], n_med=float(np.median(nn[m])), ok=ok))
        print(f"  {f}：超額 t {st[0]:.2f} 年化超額 {st[2]*100:+.1f}% Sharpe {st[1]:.2f} 正年 {st[3]:.0%} 持股中位 {np.median(nn[m]):.0f} 檔 → {'進 val' if ok else '✗'}", flush=True)
    print("\n擇時（dev）：", flush=True)
    tim = timing(B, mkt, days, "d67")
    show_timing(tim)
    # 描述：半導體業的庫存拐點
    codes = FI.labels(B["cols"])
    lab = pd.read_csv(H.D / "universe.csv", dtype=str).set_index("stock_id").industry.reindex(B["cols"]).replace(FI.MERGE).values
    semi = eng.E & (lab == "半導體業")[None, :]
    r_semi = np.array([np.nanmean(np.nan_to_num(B["R"])[t][semi[t]]) if semi[t].any() else np.nan for t in range(eng.T)]) - eng.bench
    x = monthly(mkt["半導體業"]["DIO_TURN"].to_frame("x"), eng.dates, ["x"], days)[:, 0]
    k = m & np.isfinite(x) & np.isfinite(r_semi)
    print(f"\n描述：半導體業池內等權對全池的月超額，拐點向下（DIO_TURN < 0）{np.mean(r_semi[k & (x < 0)])*100:+.2f}%（{(k & (x < 0)).sum()} 月）"
          f"、向上 {np.mean(r_semi[k & (x >= 0)])*100:+.2f}%（{(k & (x >= 0)).sum()} 月）", flush=True)
    allt = np.concatenate([res[k][:, 0] for k in ("單一", "兩兩", "三三")])
    SL.log(dict(kind="strategy", period="dev", strategy=NAME, note="第六十七批 事前登記", dev_window=C.SEG["d67"],
                n_specs=int(sum(len(res[k]) for k in ("單一", "兩兩", "三三"))), n_eval=int(np.isfinite(allt).sum()), best_t=float(np.nanmax(allt)),
                n_pass=len(T), n_contrib=int(T.contrib.sum()), n_clone_h136=int(n_clone),
                selected=[dict(kind=k, parts=[list(p) for p in parts], need=need, n=int(n), sub=[list(p) for p in sub] if sub else None)
                          for k, parts, need, n, sub in sel],
                extra=extra, rotation=rot, timing=tim, trials_cum=TRIALS))


def val():
    rec = [json.loads(l) for l in open(H.LOG) if f'"{NAME}"' in l and '"period": "dev"' in l][-1]
    B, eng, lib, mkt, days = setup("val")
    m = C.mask(eng.dates, "val")
    yrs = (eng.dates.year - 2020).values.clip(0)
    h = C.h136_series("val", eng.dates)
    print(f"val 基準年化 {np.nanmean(eng.bench[m])*12*100:+.1f}%；選股 {len(rec['selected'])} 個、產業輪動 {sum(r['ok'] for r in rec['rotation'])} 個", flush=True)
    out = []
    for s in rec["selected"]:
        parts = tuple((f, int(g)) for f, g in s["parts"])
        d, net, nl, el = C.detail(eng, B, parts, s["need"], s["n"], "val", ())
        d.update(kind=s["kind"], rho_h136=C.corr(nl, h, m))
        if s["sub"]:
            sub = tuple((f, int(g)) for f, g in s["sub"])
            d["t_sub_lim"] = C.detail(eng, B, sub, 1 if len(sub) == 1 else 2, s["n"], "val", ())[0]["t_lim"]
        d["verdict"] = "✓" if d["t_lim"] >= 2 and d["sharpe_lim"] >= 1.0 else "✗"
        out.append(d)
        print(f"  [{s['kind']}] {d['name']}：超額 t {d['t_lim']:.2f}（拿掉存貨後 {d.get('t_sub_lim', float('nan')):.2f}）年化超額 {d['ex_ann_lim']*100:+.1f}%"
              f" Sharpe {d['sharpe_lim']:.2f}｜逐年 {d['by_year']}｜和 H136 {d['rho_h136']:+.2f} → {d['verdict']}", flush=True)
    rot = []
    for r in rec["rotation"]:
        if not r["ok"]:
            continue
        net, ex, nn = rotation(B, eng, r["name"], NEW[r["name"]])
        st = _stats(ex, net, m, yrs, min_m=24)
        v = "✓" if st[0] >= 2 and st[1] >= 1.0 else "✗"
        rot.append(dict(name=r["name"], t=st[0], sharpe=st[1], ex_ann=st[2], verdict=v))
        print(f"  產業輪動 {r['name']}：超額 t {st[0]:.2f} 年化超額 {st[2]*100:+.1f}% Sharpe {st[1]:.2f} → {v}", flush=True)
    sel_tim = [r["name"] for r in rec["timing"] if r["verdict"] == "✓"]
    tim = [r for r in timing(B, mkt, days, "val") if r["name"] in sel_tim]
    if tim:
        print("擇時（val）：")
        show_timing(tim)
    else:
        print("擇時：dev 沒有過的，依登記不跑 val")
    SL.log(dict(kind="strategy", period="val", strategy=NAME, note="第六十七批 dev 選出的", result=out, rotation=rot, timing=tim))


def _holdout_specs():
    """val 過的（依 val 紀錄），預期 3 個。"""
    rec = [json.loads(l) for l in open(H.LOG) if f'"{NAME}"' in l and '"period": "val"' in l][-1]
    dv = [json.loads(l) for l in open(H.LOG) if f'"{NAME}"' in l and '"period": "dev"' in l][-1]
    ok = {r["name"] for r in rec["result"] if r["verdict"] == "✓"}
    out = []
    for s in dv["selected"]:
        parts = tuple((f, int(g)) for f, g in s["parts"])
        if C.spec_name(parts, s["n"]) in ok:
            out.append((parts, s["need"], s["n"], tuple((f, int(g)) for f, g in s["sub"])))
    assert len(out) == 3, out
    return out


def coverage():
    """開封前的資料檢查：holdout 的新特徵池內有值比例（不看報酬）。"""
    B = C.load("holdout")
    days = trading_days()
    F, _ = derive(load_raw())
    yr = pd.DatetimeIndex(B["dates"]).year
    for f in ("DIO_CHG", "REL_DIO_CHG"):
        X = monthly(F[f], B["dates"], B["cols"], days)
        print(f"  {f} 池內有值比例：" + "、".join(f"{y} {np.isfinite(X[yr == y])[B['E'][yr == y]].mean():.0%}" for y in sorted(set(yr))))
    for nm, a in (("最後一季有存貨的公司數", F["DIO_LVL"]),):
        n = a.notna().sum(axis=1)
        print(f"  {nm}：" + "、".join(f"{q.date()} {int(v)}" for q, v in n[n > 0].tail(6).items()))


def holdout():
    if "--unseal" not in sys.argv:
        raise PermissionError("holdout 封存中：使用者同意後才加 --unseal")
    sp = _holdout_specs()
    B, eng, lib, mkt, days = setup("holdout")
    m = C.mask(eng.dates, "holdout")
    r_tr = tr_monthly(eng.dates)
    k = m & np.isfinite(r_tr)
    tr_ann, tr_sh = r_tr[k].mean() * 12, r_tr[k].mean() / r_tr[k].std(ddof=1) * np.sqrt(12)
    print(f"holdout {eng.dates[m][0].date()}～{eng.dates[m][-1].date()}（{m.sum()} 個月）；pit 池等權年化 {np.nanmean(eng.bench[m])*12*100:+.1f}%；"
          f"加權報酬指數 年化 {tr_ann*100:+.1f}% Sharpe {tr_sh:.2f}")
    d, _, h, _ = C.detail(eng, B, *C.H136, "holdout", ())
    print(f"  對照 H136（同引擎）：超額 t {d['t_lim']:.2f}（年化 {d['ex_ann_lim']*100:+.1f}%）年化 {d['ann']*100:+.1f}% Sharpe {d['sharpe_lim']:.2f}")
    out, npass = [], 0
    for parts, need, n, sub in sp:
        d, net, nl, el = C.detail(eng, B, parts, need, n, "holdout", ())
        ds = C.detail(eng, B, sub, 2, n, "holdout", ())[0]
        d.update(rho_h136=C.corr(nl, h, m), t_sub_lim=ds["t_lim"], sharpe_sub_lim=ds["sharpe_lim"], tr_ann=tr_ann, tr_sharpe=tr_sh)
        d["verdict"] = "✓" if d["t_lim"] >= 2 and d["sharpe_lim"] >= 1.0 else "✗"
        npass += d["verdict"] == "✓"
        out.append(d)
        print(f"  {d['name']}：超額 t {d['t_lim']:.2f}（拿掉存貨後 {ds['t_lim']:.2f}）年化超額 {d['ex_ann_lim']*100:+.1f}% 年化 {d['ann']*100:+.1f}% "
              f"Sharpe {d['sharpe_lim']:.2f}（拿掉存貨後 {ds['sharpe_lim']:.2f}）｜逐年超額 {d['by_year']}｜和 H136 {d['rho_h136']:+.2f}｜換手 {d['turnover']:.0%} → {d['verdict']}")
    print(f"holdout 通過 {npass}／{len(sp)}")
    SL.log(dict(kind="strategy", period="holdout", strategy=NAME, note="第六十七批 holdout（使用者 2026-11-05 同意；val 過的 3 個）", unseal=True,
                result=out, n_pass=npass))


if __name__ == "__main__":
    {"leak": leak, "dev": dev, "val": val, "coverage": coverage, "holdout": holdout}[sys.argv[1] if len(sys.argv) > 1 else "dev"]()
