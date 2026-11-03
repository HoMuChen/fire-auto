"""第四十八批：H142 集中投組「篩選 → 排序」（S1）、H143 寬投組「市場減垃圾＋排序加權」（S2）。規則見 HYPOTHESES.md 第四十八批（事前登記）。

用法：.venv/bin/python research/v2/batch48_s1s2.py dev     # 菜單、條件檢查、安慰劑、deflated Sharpe、選 val
      .venv/bin/python research/v2/batch48_s1s2.py val     # 只跑 dev 選出的各 1 個
"""
import itertools
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402
import strategy_lab as SL  # noqa: E402

RANK = {"R1": ["H2"], "R2": ["H2", "H10", "H87", "H86"]}
FILT = ["NOA_GR", "PPEINV", "NET_ISSUE", "SBL_LVL"]
EXCL = ["H10", "H87", "MA_ALIGN", "NOA_GR", "PPEINV", "SBL_LVL", "NET_ISSUE"]
TILT = ["EBITDA_EV", "FWD_EY", "NI_VOL", "CF_VOL", "H3", "H13", "HL_SPREAD", "SM_RATIO"]
SIGN = {"H2": 1, "H10": 1, "H87": 1, "H86": 1, "MA_ALIGN": 1, "NOA_GR": -1, "PPEINV": -1, "NET_ISSUE": -1, "SBL_LVL": -1,
        "EBITDA_EV": 1, "FWD_EY": 1, "NI_VOL": -1, "CF_VOL": -1, "H3": 1, "H13": 1, "HL_SPREAD": -1, "SM_RATIO": -1}
P1 = ("2011-01-01", "2019-12-31")       # H142 dev 判定
P2 = ("2015-01-01", "2019-12-31")       # H143 dev 判定（B 段）
VAL = ("2020-01-01", "2023-12-31")
TOP100, CHK_MIN, NPL, TOPMV = 100, 20, 100, 150
MENU1 = list(itertools.product(("R1", "R2"), (False, True), (20, 50)))
MENU2 = list(itertools.product(("all", "top150"), (0.2, 0.3), (False, True)))
OUT = H.D / "results"


def name1(v):
    return f"H142[{v[0]}{'+篩' if v[1] else ''}/{v[2]}]"


def name2(v):
    return f"H143[{'全池' if v[0] == 'all' else '前150'}/剔{int(v[1]*100)}%{'/傾斜' if v[2] else ''}]"


def pct(X, M):
    """池 M 內的百分位（越大越好）。"""
    return pd.DataFrame(np.where(M & np.isfinite(X), X, np.nan)).rank(axis=1, pct=True).values


def check_filter(mats, R, E, dates, comps):
    """R2 前 100 名裡，各成分最差 20% 減其他的下月報酬差（dev 平均）；< 0 才留下。"""
    r2 = SL.combo([(mats[f], SIGN[f]) for f in RANK["R2"]], E, 3)
    dev = (dates >= pd.Timestamp(P1[0])) & (dates <= pd.Timestamp(P1[1]))
    res = {}
    for f in comps:
        x = SIGN[f] * mats[f]
        d = []
        for t in np.flatnonzero(dev):
            top = np.argsort(np.where(np.isfinite(r2[t]), -r2[t], np.inf), kind="stable")[:TOP100]
            top = top[np.isfinite(r2[t, top]) & np.isfinite(x[t, top])]
            if len(top) < CHK_MIN:
                continue
            q = pd.Series(x[t, top]).rank(pct=True).values
            bad = q <= 0.2
            d.append(R[t, top][bad].mean() - R[t, top][~bad].mean())
        d = np.array(d)
        res[f] = dict(months=len(d), diff=d.mean() if len(d) else np.nan,
                      t=d.mean() / d.std(ddof=1) * np.sqrt(len(d)) if len(d) > 2 else np.nan)
    return res


def run1(mats, R, E, dates, lim, comps, v):
    rk, use_f, n = v
    S = SL.combo([(mats[f], SIGN[f]) for f in RANK[rk]], E, 1 if rk == "R1" else 3)
    filt = None
    if use_f and comps:
        q = pct(SL.combo([(mats[f], SIGN[f]) for f in comps], E, min(2, len(comps))), E)
        filt = ~(q <= 0.2)
    return S, SL.portfolio(S, R, E, n, buffer=2, filt=filt, limits=lim)


def run2(mats, R, E, U150, v):
    uni, cut, tilt = v
    U = E if uni == "all" else U150
    q = pct(SL.combo([(mats[f], SIGN[f]) for f in EXCL], U, 4), U)
    keep = U & ~(q <= cut)
    w = np.where(keep, 1.0, 0.0)
    if tilt:
        tq = pct(SL.combo([(mats[f], SIGN[f]) for f in TILT], U, 4), U)
        w = np.where(keep, np.where(np.isfinite(tq), 1 + tq - 0.5, 1.0), 0.0)
    s = w.sum(axis=1, keepdims=True)
    Wm = np.divide(w, s, out=np.zeros_like(w), where=s > 0)
    bench = np.array([R[t][U[t]].mean() if U[t].any() else np.nan for t in range(len(R))])
    return SL.weights_portfolio(Wm, R, E, bench=bench), U


def quint(S, R, E, dates, p):
    m = (dates >= pd.Timestamp(p[0])) & (dates <= pd.Timestamp(p[1]))
    q = pct(S, E)
    out = np.full((m.sum(), 5), np.nan)
    for i, t in enumerate(np.flatnonzero(m)):
        ok = np.isfinite(q[t])
        if ok.sum() < SL.MIN_VALID:
            continue
        b = np.minimum((q[t, ok] * 5 - 1e-12).astype(int), 4)
        mu = R[t][E[t]].mean()
        out[i] = [R[t, ok][b == k].mean() - mu if (b == k).any() else np.nan for k in range(5)]
    qm = np.nanmean(out, axis=0) * 12
    rho = pd.Series(qm).rank().corr(pd.Series(np.arange(5)), method="spearman")
    return qm, rho


def setup(end):
    lab = SL.Lab(end)
    dates, R, E = lab.monthly("2010-01-01", end)
    mats = {f: lab.feat(f, dates, E) for f in SIGN}
    mv = lab.P["mv"].shift(1).reindex(index=dates, columns=lab.cols).values
    U150 = E & (pct(mv, E) * E.sum(axis=1, keepdims=True) > E.sum(axis=1, keepdims=True) - TOPMV)
    money20 = lab.P["money"].astype(float).rolling(20, min_periods=15).mean().shift(1).reindex(index=dates, columns=lab.cols).values
    cost_pool = {k: SL.weights_portfolio(np.divide(M, M.sum(axis=1, keepdims=True), out=np.zeros(M.shape), where=M.sum(axis=1, keepdims=True) > 0),
                                         R, E, bench=np.zeros(len(R))) for k, M in (("all", E), ("top150", U150))}
    return lab, dates, R, E, mats, U150, money20, cost_pool


def check_signs():
    L = pd.read_csv(H.D / "results" / "feature_library.csv").set_index("H")
    bad = {f: (s, L.loc[f, "sign"]) for f, s in SIGN.items() if float(L.loc[f, "sign"]) != s}
    if bad:
        raise RuntimeError(f"方向和特徵庫不一致：{bad}")


def fmt(s):
    return (f"超額 {s['ex']*100:+6.1f}% t {s['t']:5.2f} 正年 {s['yrs']:>4} Sharpe {s['sharpe']:.2f} MDD {s['mdd']*100:5.1f}% "
            f"換手 {s['turn']*100:4.0f}% 檔數 {s['hold']:.0f}")


def dev():
    check_signs()
    t0 = time.time()
    lab, dates, R, E, mats, U150, money20, cpool = setup(H.PERIODS["dev"][1])
    lim = lab.limits(dates)
    print(f"載入 {time.time()-t0:.0f}s；{len(dates)} 個換股日", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        chk = check_filter(mats, R, E, dates, FILT)
        comps = [f for f in FILT if chk[f]["diff"] < 0]
        print("\n== H142 篩選成分的條件檢查（R2 前 100 名內，最差 20% − 其他，下月報酬）")
        for f, r in chk.items():
            print(f"  {f:<10} {r['months']:>3} 個月  差 {r['diff']*100:+.2f}%/月（t {r['t']:+.2f}）→ {'留' if f in comps else '拿掉'}")
        rows, series = [], {}
        print(f"\n== H142 dev {P1[0]} ~ {P1[1]}（對 pit 池等權；成本已扣、一字漲跌停開啟）  篩選成分：{comps}")
        for v in MENU1:
            S, df = run1(mats, R, E, dates, lim, comps, v)
            st = SL.stats(df, dates, *P1)
            qm, rho = quint(S, R, E, dates, P1)
            ca = SL.stats(df.assign(ex=df.net - cpool["all"].net, bench=cpool["all"].net), dates, *P1)
            ok = st["t"] >= 2 and int(st["yrs"].split("/")[0]) >= 7 and qm[4] == np.nanmax(qm) and rho >= 0.8
            rows.append(dict(h="H142", var=name1(v), spec=v, **{f"dev_{k}": x for k, x in st.items()}, q=np.round(qm * 100, 1).tolist(),
                             rho=rho, ex_costpool=ca["ex"], t_costpool=ca["t"], cap=SL.capacity(df, money20), pass_=ok))
            series[name1(v)] = df.ex.values
            print(f"  {name1(v):<18} {fmt(st)}｜分位 {np.round(qm*100,1)} ρ {rho:.2f}｜對扣成本池 {ca['ex']*100:+.1f}%｜容量 {rows[-1]['cap']/1e8:.1f} 億 → {'過' if ok else '✗'}")
        print(f"\n== H143 dev B 段 {P2[0]} ~ {P2[1]}（全池對 pit 池、前 150 對前 150 等權；A 段只描述）")
        for v in MENU2:
            df, U = run2(mats, R, E, U150, v)
            st, sa = SL.stats(df, dates, *P2), SL.stats(df, dates, P1[0], "2014-12-31")
            ca = SL.stats(df.assign(ex=df.net - cpool[v[0]].net, bench=cpool[v[0]].net), dates, *P2)
            ok = st["t"] >= 2 and int(st["yrs"].split("/")[0]) >= 4
            rows.append(dict(h="H143", var=name2(v), spec=v, **{f"dev_{k}": x for k, x in st.items()}, A_ex=sa.get("ex"), A_t=sa.get("t"),
                             ex_costpool=ca["ex"], t_costpool=ca["t"], cap=SL.capacity(df, money20), pass_=ok))
            series[name2(v)] = df.ex.values
            print(f"  {name2(v):<22} {fmt(st)}｜A 段 {sa.get('ex', np.nan)*100:+.1f}%｜對扣成本池 {ca['ex']*100:+.1f}%（t {ca['t']:.2f}）｜容量 {rows[-1]['cap']/1e8:.0f} 億 → {'過' if ok else '✗'}")
        D = pd.DataFrame(rows)

        # 安慰劑：同一個排列打亂所有成分矩陣
        def best1(m):
            c = [f for f in FILT if check_filter(m, R, E, dates, [f])[f]["diff"] < 0]
            return max(SL.stats(run1(m, R, E, dates, lim, c, v)[1], dates, *P1)["t"] for v in MENU1)

        def best2(m):
            return max(SL.stats(run2(m, R, E, U150, v)[0], dates, *P2)["t"] for v in MENU2)
        t1 = time.time()
        pl1 = SL.placebo(best1, mats, n=NPL)
        pl2 = SL.placebo(best2, mats, n=NPL, seed=2027)
        print(f"\n== 安慰劑（{NPL} 次，菜單最佳 t；{time.time()-t1:.0f}s）")
        for h, pl, sub in (("H142", pl1, D[D.h == "H142"]), ("H143", pl2, D[D.h == "H143"])):
            best = sub.dev_t.max()
            print(f"  {h}：真實最佳 t {best:.2f}｜安慰劑 50／95 分位 {np.percentile(pl, 50):.2f}／{np.percentile(pl, 95):.2f}｜真實 > 95 分位：{best > np.percentile(pl, 95)}")
            D.loc[D.h == h, "placebo_p95"] = np.percentile(pl, 95)
        # deflated Sharpe（試驗數＝累計 16）
        msk = {"H142": (dates >= pd.Timestamp(P1[0])) & (dates <= pd.Timestamp(P1[1])),
               "H143": (dates >= pd.Timestamp(P2[0])) & (dates <= pd.Timestamp(P2[1]))}
        srs = [np.nanmean(series[k][msk[k[:4]]]) / np.nanstd(series[k][msk[k[:4]]], ddof=1) for k in series]
        D["dsr"] = [SL.deflated_sharpe(series[k][msk[k[:4]]], srs, 16) for k in D["var"]]
        sel = {}
        for h in ("H142", "H143"):
            sub = D[(D.h == h) & D.pass_ & (D.dev_t > D.placebo_p95)]
            sel[h] = sub.sort_values("dev_t", ascending=False).iloc[0]["var"] if len(sub) else None
        D["selected"] = D["var"].isin([v for v in sel.values() if v])
        print("\n== deflated Sharpe（試驗數 16）與 val 選擇")
        for _, r in D.iterrows():
            print(f"  {r['var']:<22} DSR {r.dsr:.3f}{'  ← 進 val' if r.selected else ''}")
        print(f"  選出：{sel}")
    D.drop(columns="spec").to_csv(OUT / "batch48_dev.csv", index=False)
    SL.log(dict(kind="strategy", period="dev", strategy="第四十八批 H142/H143", note="第四十八批 事前登記", filter_check=chk, filter_comps=comps,
                rows=D.drop(columns="spec").to_dict("records"), placebo_p95={"H142": float(np.percentile(pl1, 95)), "H143": float(np.percentile(pl2, 95))},
                selected=sel, trials_cum=16))


def val():
    D = pd.read_csv(OUT / "batch48_dev.csv")
    sel = D[D.selected]
    if not len(sel):
        print("dev 沒有選出任何變體 → 依登記不跑 val")
        return
    t0 = time.time()
    lab, dates, R, E, mats, U150, money20, cpool = setup(VAL[1])
    lim = lab.limits(dates)
    print(f"載入 {time.time()-t0:.0f}s", flush=True)
    rec = [json.loads(l) for l in open(H.LOG) if '"第四十八批 H142/H143"' in l and '"period": "dev"' in l][-1]
    comps = rec["filter_comps"]
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for _, r in sel.iterrows():
            if r.h == "H142":
                v = next(v for v in MENU1 if name1(v) == r["var"])
                S, df = run1(mats, R, E, dates, lim, comps, v)
                cp = cpool["all"]
            else:
                v = next(v for v in MENU2 if name2(v) == r["var"])
                df, _ = run2(mats, R, E, U150, v)
                cp = cpool[v[0]]
            st = SL.stats(df, dates, *VAL)
            ca = SL.stats(df.assign(ex=df.net - cp.net, bench=cp.net), dates, *VAL)
            ok = st["t"] >= 2 and st["yrs"] == "4/4"
            yr = df.ex[(dates >= pd.Timestamp(VAL[0])) & (dates <= pd.Timestamp(VAL[1]))].groupby(dates[(dates >= pd.Timestamp(VAL[0])) & (dates <= pd.Timestamp(VAL[1]))].year).sum()
            out[r["var"]] = dict(**st, ex_costpool=ca["ex"], t_costpool=ca["t"], by_year=(yr * 100).round(1).to_dict(), verdict="✓" if ok else "✗")
            print(f"  {r['var']:<22} val {fmt(st)}｜對扣成本池 {ca['ex']*100:+.1f}%（t {ca['t']:.2f}）｜逐年 {out[r['var']]['by_year']} → {out[r['var']]['verdict']}")
    SL.log(dict(kind="strategy", period="val", strategy="第四十八批 H142/H143", note="第四十八批 dev 選出的各 1 個", result=out))


def slip():
    """描述（不改規則）：H142 選出的變體在每次進出多付零股價差 0／0.2／0.5／1% 時，dev、val 的超額。"""
    D = pd.read_csv(OUT / "batch48_dev.csv")
    var = D[D.selected & (D.h == "H142")]["var"].iloc[0]
    v = next(v for v in MENU1 if name1(v) == var)
    rec = [json.loads(l) for l in open(H.LOG) if '"第四十八批 H142/H143"' in l and '"period": "dev"' in l][-1]
    t0 = time.time()
    lab, dates, R, E, mats, U150, money20, cpool = setup(VAL[1])
    lim = lab.limits(dates)
    print(f"載入 {time.time()-t0:.0f}s；{var}", flush=True)
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        S = SL.combo([(mats[f], SIGN[f]) for f in RANK[v[0]]], E, 3)
        q = pct(SL.combo([(mats[f], SIGN[f]) for f in rec["filter_comps"]], E, 2), E)
        for extra in (0.0, 0.002, 0.005, 0.01):
            df = SL.portfolio(S, R, E, v[2], buffer=2, filt=~(q <= 0.2), limits=lim, cost=SL.COST + extra)
            d, va = SL.stats(df, dates, *P1), SL.stats(df, dates, *VAL)
            out[f"{extra*100:.1f}%"] = dict(dev=d, val=va)
            print(f"  多付 {extra*100:.1f}%／次進出：dev {d['ex']*100:+5.1f}%（t {d['t']:.2f}，{d['yrs']}）｜val {va['ex']*100:+5.1f}%（t {va['t']:.2f}，{va['yrs']}）｜月換手 {d['turn']*100:.0f}%", flush=True)
    SL.log(dict(kind="strategy", period="dev+val", strategy=f"第四十八批 {var} 零股價差敏感度", note="描述（不改規則）", result=out))


if __name__ == "__main__":
    {"dev": dev, "val": val, "slip": slip}[sys.argv[1] if len(sys.argv) > 1 else "dev"]()
