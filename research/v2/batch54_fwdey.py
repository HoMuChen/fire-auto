"""第五十四批：H160 預估盈餘殖利率（FWD_EY）集中投組。篩選（無／有）× 檔數（20／50）× 換股（每月／每週）。
規則見 HYPOTHESES.md 第五十四批（事前登記）。

用法：.venv/bin/python research/v2/batch54_fwdey.py dev|val
"""
import itertools
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch48_s1s2 as B48  # noqa: E402
import batch49_concentration as B49  # noqa: E402
import feature_tracks as FT  # noqa: E402
import harness as H  # noqa: E402
import strategy_lab as SL  # noqa: E402

FILT = ["NOA_GR", "PPEINV", "NET_ISSUE", "SBL_LVL"]
SIGN = {"FWD_EY": 1, **{f: -1 for f in FILT}}
MENU = list(itertools.product((False, True), (20, 50), ("月", "週")))
P1, VAL, NPL, TRIALS, TOP100 = B48.P1, B48.VAL, 100, 87, 100
PPY = {"月": 12, "週": 52}
OUT = H.D / "results"


def nm(v):
    return f"H160[{'篩/' if v[0] else ''}{v[1]}檔/{v[2]}換]"


def load(end):
    lab = SL.Lab(end)
    md, R, E = lab.monthly("2010-01-01", end)
    wd, Rw, Ew = FT.weekly(lab.P, lab.cols)
    Ew = Ew & np.isfinite(Rw)
    G = {"月": dict(d=md, R=R, E=E, lim=lab.limits(md)), "週": dict(d=wd, R=Rw, E=Ew, lim=lab.limits(wd))}
    mats = {k: {f: lab.feat(f, g["d"], g["E"]) for f in SIGN} for k, g in G.items()}
    return lab, G, mats


def check(mats, g, comps):
    S = B48.pct(mats["FWD_EY"], g["E"])
    dev = (g["d"] >= pd.Timestamp(P1[0])) & (g["d"] <= pd.Timestamp(P1[1]))
    res = {}
    for f in comps:
        x = SIGN[f] * mats[f]
        d = []
        for t in np.flatnonzero(dev):
            top = np.argsort(np.where(np.isfinite(S[t]), -S[t], np.inf), kind="stable")[:TOP100]
            top = top[np.isfinite(S[t, top]) & np.isfinite(x[t, top])]
            if len(top) < 20:
                continue
            q = pd.Series(x[t, top]).rank(pct=True).values
            d.append(g["R"][t, top][q <= 0.2].mean() - g["R"][t, top][q > 0.2].mean())
        d = np.array(d)
        res[f] = dict(months=len(d), diff=d.mean() if len(d) else np.nan, t=d.mean() / d.std(ddof=1) * np.sqrt(len(d)) if len(d) > 2 else np.nan)
    return res


def run(mats, G, comps, v):
    filt_on, n, fq = v
    g, m = G[fq], mats[fq]
    S = SL.combo([(m["FWD_EY"], 1)], g["E"], 1)
    filt = None
    if filt_on and comps:
        q = B48.pct(SL.combo([(m[f], SIGN[f]) for f in comps], g["E"], min(2, len(comps))), g["E"])
        filt = ~(q <= 0.2)
    return SL.portfolio(S, g["R"], g["E"], n, buffer=2, filt=filt, limits=g["lim"])


def fmt(s, fq):
    return (f"超額 {s['ex']*100:+6.1f}% t {s['t']:5.2f} 正年 {s['yrs']:>4} Sharpe {s['sharpe']:.2f} MDD {s['mdd']*100:6.1f}% "
            f"換手 {s['turn']*100:3.0f}%/{fq}（年成本 {s['turn']*PPY[fq]*SL.COST*100:4.1f}%）")


def h142_ex(lab, g):
    """同一個面板算 H142（50 檔、留任 3N）的月超額，不重新載入。"""
    comps = B49.filt_comps()
    m = {f: lab.feat(f, g["d"], g["E"]) for f in set(B48.RANK["R2"]) | set(comps)}
    S, filt = B49.s142(m, g["E"], comps)
    return g["d"], SL.portfolio(S, g["R"], g["E"], 50, buffer=3, filt=filt, limits=g["lim"]).ex.values


def dev():
    t0 = time.time()
    lab, G, mats = load(H.PERIODS["dev"][1])
    print(f"載入 {time.time()-t0:.0f}s", flush=True)
    rows, ser = [], {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        chk = check(mats["月"], G["月"], FILT)
        comps = [f for f in FILT if chk[f]["diff"] < 0]
        print("\n== 篩選成分的條件檢查（FWD_EY 前 100 名內，最差 20% − 其他，下月報酬）")
        for f, r in chk.items():
            print(f"  {f:<10} {r['months']:>3} 個月 差 {r['diff']*100:+.2f}%/月（t {r['t']:+.2f}）→ {'留' if f in comps else '拿掉'}")
        hd, hx = h142_ex(lab, G["月"])
        print(f"\n== dev {P1[0]} ~ {P1[1]}（對 pit 池等權；扣成本、一字漲跌停開啟）  篩選成分 {comps}")
        for v in MENU:
            g = G[v[2]]
            df = run(mats, G, comps, v)
            st = SL.stats(df, g["d"], *P1, ppy=PPY[v[2]])
            ok = st["t"] >= 2 and int(st["yrs"].split("/")[0]) >= 7
            rho = np.nan
            if v[2] == "月":
                m = (g["d"] >= pd.Timestamp(P1[0])) & (g["d"] <= pd.Timestamp(P1[1]))
                hxx = pd.Series(hx, index=hd).reindex(g["d"]).values
                k = m & np.isfinite(df.ex.values) & np.isfinite(hxx)
                rho = np.corrcoef(df.ex.values[k], hxx[k])[0, 1]
            rows.append(dict(var=nm(v), filt=v[0], n=v[1], fq=v[2], **{f"dev_{k}": x for k, x in st.items()}, rho_h142=rho, pass_=ok))
            ser[nm(v)] = (df.ex.values, g["d"], v[2])
            print(f"  {nm(v):<18} {fmt(st, v[2])}{'' if np.isnan(rho) else f'｜和 H142 相關 {rho:+.2f}'} → {'過' if ok else '✗'}", flush=True)
        D = pd.DataFrame(rows)

        def best(m_):
            mm = {fq: {f: m_[f"{fq}|{f}"] for f in SIGN} for fq in G}
            c = [f for f in FILT if check(mm["月"], G["月"], [f])[f]["diff"] < 0]
            return max(SL.stats(run(mm, G, c, v), G[v[2]]["d"], *P1, ppy=PPY[v[2]])["t"] for v in MENU)
        flat = {f"{fq}|{f}": mats[fq][f] for fq in G for f in SIGN}
        t1 = time.time()
        pl = SL.placebo(best, flat, n=NPL, seed=2034)
        p95 = np.percentile(pl, 95)
        print(f"\n== 安慰劑（{NPL} 次，菜單最佳 dev t；{time.time()-t1:.0f}s）：50／95 分位 {np.percentile(pl, 50):.2f}／{p95:.2f}；真實 {D.dev_t.max():.2f}")
        srs, dsr = [], []
        for k, (x, d, fq) in ser.items():
            m = (d >= pd.Timestamp(P1[0])) & (d <= pd.Timestamp(P1[1])) & np.isfinite(x)
            srs.append(x[m].mean() / x[m].std(ddof=1) * np.sqrt(PPY[fq] / 12))   # 換成月尺度再比
        for k, (x, d, fq) in ser.items():
            m = (d >= pd.Timestamp(P1[0])) & (d <= pd.Timestamp(P1[1])) & np.isfinite(x)
            dsr.append(SL.deflated_sharpe(x[m], np.array(srs) / np.sqrt(PPY[fq] / 12), TRIALS))
        D["dsr"], D["placebo_p95"] = dsr, p95
        okd = D[D.pass_ & (D.dev_t > p95)]
        sel = okd.sort_values("dev_t", ascending=False).iloc[0]["var"] if len(okd) else None
        D["selected"] = D["var"] == sel
        print(f"\n== deflated Sharpe（試驗數 {TRIALS}）與 val 選擇")
        for _, r in D.iterrows():
            print(f"  {r['var']:<18} DSR {r.dsr:.3f}{'  ← 進 val' if r.selected else ''}")
    D.to_csv(OUT / "batch54_dev.csv", index=False)
    SL.log(dict(kind="strategy", period="dev", strategy="第五十四批 H160 FWD_EY 集中投組", note="第五十四批 事前登記",
                filter_check=chk, filter_comps=comps, rows=D.to_dict("records"), placebo_p95=p95, selected=sel, trials_cum=TRIALS))


def val():
    import json
    D = pd.read_csv(OUT / "batch54_dev.csv")
    sel = D[D.selected]
    if not len(sel):
        print("dev 沒有選出 → 依登記不跑 val")
        return
    r = sel.iloc[0]
    v = (bool(r.filt), int(r.n), r.fq)
    rec = [json.loads(l) for l in open(H.LOG) if '"第五十四批 H160 FWD_EY 集中投組"' in l and '"period": "dev"' in l][-1]
    t0 = time.time()
    lab, G, mats = load(VAL[1])
    print(f"載入 {time.time()-t0:.0f}s", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        g = G[v[2]]
        df = run(mats, G, rec["filter_comps"], v)
        st = SL.stats(df, g["d"], *VAL, ppy=PPY[v[2]])
        ok = st["t"] >= 2 and st["yrs"] == "4/4"
        m = (g["d"] >= pd.Timestamp(VAL[0])) & (g["d"] <= pd.Timestamp(VAL[1]))
        yr = (df.ex[m].groupby(g["d"][m].year).sum() * 100).round(1).to_dict()
        print(f"\n== val {VAL[0]} ~ {VAL[1]}\n  {r['var']:<18} {fmt(st, v[2])}｜逐年 {yr} → {'✓' if ok else '✗'}")
    SL.log(dict(kind="strategy", period="val", strategy="第五十四批 H160 FWD_EY 集中投組", note="第五十四批 dev 選出的",
                result=dict(var=r["var"], **st, by_year=yr, verdict="✓" if ok else "✗")))


if __name__ == "__main__":
    {"dev": dev, "val": val}[sys.argv[1] if len(sys.argv) > 1 else "dev"]()
