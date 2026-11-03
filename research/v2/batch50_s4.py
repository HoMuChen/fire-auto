"""第五十批（STRATEGY_PLAN S4）：持有期與換股頻率。H146＝H142 50 檔、H147＝H144 10 檔／產業 ≤ 2，
各測 緩衝 1.5N／緩衝 3N／每季換／60 日分 3 批。規則見 HYPOTHESES.md 第五十批（事前登記）。

用法：.venv/bin/python research/v2/batch50_s4.py dev|val
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch48_s1s2 as B48  # noqa: E402
import batch49_concentration as B49  # noqa: E402
import harness as H  # noqa: E402
import strategy_lab as SL  # noqa: E402

P1, VAL, NPL, TRIALS, TOL = B48.P1, B48.VAL, 100, 30, 0.01
FAM = {"H146": dict(n=50, cap=False), "H147": dict(n=10, cap=True)}
MENU = ["緩衝1.5N", "緩衝3N", "每季換", "分3批"]
OUT = H.D / "results"


def run(mats, R, E, lim, comps, codes, fam, v):
    S, filt = B49.s142(mats, E, comps)
    f = FAM[fam]
    kw = dict(filt=filt, limits=lim, group=codes if f["cap"] else None, cap=B49.CAP if f["cap"] else None)
    t = np.arange(len(S))
    if v == "原版":
        return SL.portfolio(S, R, E, f["n"], buffer=2, **kw)
    if v == "緩衝1.5N":
        return SL.portfolio(S, R, E, f["n"], buffer=1.5, **kw)
    if v == "緩衝3N":
        return SL.portfolio(S, R, E, f["n"], buffer=3, **kw)
    if v == "每季換":
        return SL.portfolio(S, R, E, f["n"], buffer=2, rebal=(t % 3 == 0), **kw)
    return SL.staggered(S, R, E, f["n"], k=3, buffer=2, **kw)


def fmt(s, tev):
    return (f"超額 {s['ex']*100:+6.1f}% t {s['t']:5.2f} 正年 {s['yrs']:>4} Sharpe {s['sharpe']:.2f} MDD {s['mdd']*100:6.1f}% "
            f"追蹤誤差 {tev*100:4.1f}% 月換手 {s['turn']*100:3.0f}%（年成本 {s['turn']*12*SL.COST*100:4.1f}%） 平均檔數 {s['hold']:.0f}")


def dev():
    comps = B49.filt_comps()
    t0 = time.time()
    lab, dates, R, E, mats, lim, codes = B49.load(H.PERIODS["dev"][1])
    print(f"載入 {time.time()-t0:.0f}s", flush=True)
    rows, ser = [], {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        print(f"\n== dev {P1[0]} ~ {P1[1]}（對 pit 池等權；扣成本、一字漲跌停開啟）")
        for fam in FAM:
            base = SL.stats(run(mats, R, E, lim, comps, codes, fam, "原版"), dates, *P1)
            print(f"  {fam}（{FAM[fam]['n']} 檔{'／產業≤2' if FAM[fam]['cap'] else ''}）")
            df0 = run(mats, R, E, lim, comps, codes, fam, "原版")
            print(f"    {'原版（對照）':<12} {fmt(base, B49.te(df0, dates, P1))}")
            for v in MENU:
                df = run(mats, R, E, lim, comps, codes, fam, v)
                st = SL.stats(df, dates, *P1)
                ok = st["t"] >= 2 and int(st["yrs"].split("/")[0]) >= 7
                keep = st["ex"] >= base["ex"] - TOL
                rows.append(dict(h=fam, var=f"{fam}[{v}]", v=v, **{f"dev_{k}": x for k, x in st.items()}, base_ex=base["ex"],
                                 te=B49.te(df, dates, P1), pass_=ok, keep_ex=keep))
                ser[f"{fam}[{v}]"] = df.ex.values
                print(f"    {v:<12} {fmt(st, rows[-1]['te'])} → {'過' if ok else '✗'}{'、超額沒掉' if keep else '、超額少 >1%'}")
        D = pd.DataFrame(rows)
        t1 = time.time()
        print(f"\n== 安慰劑（{NPL} 次，菜單最佳 dev t）")
        for fam, seed in (("H146", 2030), ("H147", 2031)):
            pl = SL.placebo(lambda m, fam=fam: max(SL.stats(run(m, R, E, lim, comps, codes, fam, v), dates, *P1)["t"] for v in MENU),
                            mats, n=NPL, seed=seed)
            D.loc[D.h == fam, "placebo_p95"] = np.percentile(pl, 95)
            print(f"  {fam}：真實最佳 t {D[D.h == fam].dev_t.max():.2f}｜安慰劑 50／95 分位 {np.percentile(pl, 50):.2f}／{np.percentile(pl, 95):.2f}  [{time.time()-t1:.0f}s]", flush=True)
        m = (dates >= pd.Timestamp(P1[0])) & (dates <= pd.Timestamp(P1[1]))
        srs = [np.nanmean(ser[k][m]) / np.nanstd(ser[k][m], ddof=1) for k in ser]
        D["dsr"] = [SL.deflated_sharpe(ser[k][m], srs, TRIALS) for k in D["var"]]
        sel = []
        for fam in FAM:
            sub = D[(D.h == fam) & D.pass_ & D.keep_ex & (D.dev_t > D.placebo_p95)]
            if len(sub):
                sel.append(sub.sort_values(["dev_turn", "dev_t"], ascending=[True, False]).iloc[0]["var"])
        D["selected"] = D["var"].isin(sel)
        print(f"\n== deflated Sharpe（試驗數 {TRIALS}）與 val 選擇（超額沒掉的裡面取換手最低）")
        for _, r in D.iterrows():
            print(f"  {r['var']:<16} DSR {r.dsr:.3f}{'  ← 進 val' if r.selected else ''}")
    D.to_csv(OUT / "batch50_dev.csv", index=False)
    SL.log(dict(kind="strategy", period="dev", strategy="第五十批 H146/H147 S4 持有期", note="第五十批 事前登記",
                rows=D.to_dict("records"), selected=sel, trials_cum=TRIALS))


def val():
    D = pd.read_csv(OUT / "batch50_dev.csv")
    sel = D[D.selected]
    if not len(sel):
        print("dev 沒有選出任何變體 → 依登記不跑 val，維持原版")
        return
    comps = B49.filt_comps()
    t0 = time.time()
    lab, dates, R, E, mats, lim, codes = B49.load(VAL[1])
    print(f"載入 {time.time()-t0:.0f}s", flush=True)
    out = {}
    m = (dates >= pd.Timestamp(VAL[0])) & (dates <= pd.Timestamp(VAL[1]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        print(f"\n== val {VAL[0]} ~ {VAL[1]}")
        for _, r in sel.iterrows():
            df0 = run(mats, R, E, lim, comps, codes, r.h, "原版")
            print(f"  {r.h + '[原版]（對照）':<20} {fmt(SL.stats(df0, dates, *VAL), B49.te(df0, dates, VAL))}")
            df = run(mats, R, E, lim, comps, codes, r.h, r.v)
            st = SL.stats(df, dates, *VAL)
            ok = st["t"] >= 2 and st["yrs"] == "4/4"
            yr = (df.ex[m].groupby(dates[m].year).sum() * 100).round(1).to_dict()
            out[r["var"]] = dict(**st, te=B49.te(df, dates, VAL), by_year=yr, verdict="✓" if ok else "✗")
            print(f"  {r['var']:<20} {fmt(st, out[r['var']]['te'])}｜逐年 {yr} → {out[r['var']]['verdict']}")
    SL.log(dict(kind="strategy", period="val", strategy="第五十批 H146/H147 S4 持有期", note="第五十批 dev 選出的", result=out))


if __name__ == "__main__":
    {"dev": dev, "val": val}[sys.argv[1] if len(sys.argv) > 1 else "dev"]()
