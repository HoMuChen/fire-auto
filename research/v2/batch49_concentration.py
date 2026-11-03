"""第四十九批：降低檔數（事後）。H144＝H142 規則凍結改 10／5 檔（± 同產業上限 2），H145＝H136 規則凍結改 10／5 檔。
規則見 HYPOTHESES.md 第四十九批（事前登記）。

用法：.venv/bin/python research/v2/batch49_concentration.py dev|val
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch48_s1s2 as B48  # noqa: E402
import harness as H  # noqa: E402
import strategy_lab as SL  # noqa: E402

P1, VAL, NPL, CAP, TRIALS = B48.P1, B48.VAL, 100, 2, 22
H144 = [(10, False), (10, True), (5, False), (5, True)]
H145 = [10, 5]
OUT = H.D / "results"


def filt_comps():
    rec = [json.loads(l) for l in open(H.LOG) if '"第四十八批 H142/H143"' in l and '"period": "dev"' in l][-1]
    return rec["filter_comps"]


def s142(mats, E, comps):
    S = SL.combo([(mats[f], B48.SIGN[f]) for f in B48.RANK["R2"]], E, 3)
    q = B48.pct(SL.combo([(mats[f], B48.SIGN[f]) for f in comps], E, 2), E)
    return S, ~(q <= 0.2)


def s136(mats, E):
    return SL.combo([(mats[f], 1) for f in ("H2", "H87", "H97")], E, 2)


def run144(mats, R, E, lim, comps, codes, n, cap):
    S, filt = s142(mats, E, comps)
    return SL.portfolio(S, R, E, n, buffer=2, filt=filt, limits=lim, group=codes if cap else None, cap=CAP if cap else None)


def run136(mats, R, E, lim, n):
    return SL.portfolio(s136(mats, E), R, E, n, limits=lim)


def nm144(v):
    return f"H144[{v[0]}檔{'/產業≤2' if v[1] else ''}]"


def nm145(n):
    return f"H145[{n}檔]"


def te(df, dates, p):
    m = (dates >= pd.Timestamp(p[0])) & (dates <= pd.Timestamp(p[1]))
    return df.ex[m].std(ddof=1) * np.sqrt(12)


def fmt(s, tev):
    return (f"超額 {s['ex']*100:+6.1f}% t {s['t']:5.2f} 正年 {s['yrs']:>4} Sharpe {s['sharpe']:.2f} MDD {s['mdd']*100:6.1f}% "
            f"追蹤誤差 {tev*100:4.1f}% 換手 {s['turn']*100:3.0f}%")


def load(end):
    lab, dates, R, E, mats, *_ = B48.setup(end)
    mats["H97"] = lab.feat("H97", dates, E)
    return lab, dates, R, E, mats, lab.limits(dates), np.asarray(lab.src.codes)


def dev():
    comps = filt_comps()
    t0 = time.time()
    lab, dates, R, E, mats, lim, codes = load(H.PERIODS["dev"][1])
    print(f"載入 {time.time()-t0:.0f}s；篩選成分 {comps}", flush=True)
    rows, ser = [], {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        print(f"\n== dev {P1[0]} ~ {P1[1]}（對 pit 池等權；扣成本、一字漲跌停開啟）")
        print("  對照（不算試驗）：")
        for nm, df in (("H142[50檔]", run144(mats, R, E, lim, comps, codes, 50, False)),
                       ("H142[20檔]", run144(mats, R, E, lim, comps, codes, 20, False)),
                       ("H136[20檔]", run136(mats, R, E, lim, 20))):
            st = SL.stats(df, dates, *P1)
            print(f"    {nm:<16} {fmt(st, te(df, dates, P1))}")
        for fam, vs, fn, nm in (("H144", H144, lambda v: run144(mats, R, E, lim, comps, codes, v[0], v[1]), nm144),
                                ("H145", H145, lambda v: run136(mats, R, E, lim, v), nm145)):
            print(f"  {fam}：")
            for v in vs:
                df = fn(v)
                st = SL.stats(df, dates, *P1)
                ok = st["t"] >= 2 and int(st["yrs"].split("/")[0]) >= 7
                rows.append(dict(h=fam, var=nm(v), n=v[0] if fam == "H144" else v, **{f"dev_{k}": x for k, x in st.items()},
                                 te=te(df, dates, P1), pass_=ok))
                ser[nm(v)] = df.ex.values
                print(f"    {nm(v):<16} {fmt(st, rows[-1]['te'])} → {'過' if ok else '✗'}")
        D = pd.DataFrame(rows)

        def best(fam):
            if fam == "H144":
                return lambda m: max(SL.stats(run144(m, R, E, lim, comps, codes, *v), dates, *P1)["t"] for v in H144)
            return lambda m: max(SL.stats(run136(m, R, E, lim, v), dates, *P1)["t"] for v in H145)
        t1 = time.time()
        print(f"\n== 安慰劑（{NPL} 次，菜單最佳 dev t）")
        for fam, seed in (("H144", 2028), ("H145", 2029)):
            pl = SL.placebo(best(fam), mats, n=NPL, seed=seed)
            p95 = np.percentile(pl, 95)
            D.loc[D.h == fam, "placebo_p95"] = p95
            print(f"  {fam}：真實最佳 t {D[D.h == fam].dev_t.max():.2f}｜安慰劑 50／95 分位 {np.percentile(pl, 50):.2f}／{p95:.2f}  [{time.time()-t1:.0f}s]", flush=True)
        m = (dates >= pd.Timestamp(P1[0])) & (dates <= pd.Timestamp(P1[1]))
        srs = [np.nanmean(ser[k][m]) / np.nanstd(ser[k][m], ddof=1) for k in ser]
        D["dsr"] = [SL.deflated_sharpe(ser[k][m], srs, TRIALS) for k in D["var"]]
        ok = D.pass_ & (D.dev_t > D.placebo_p95)
        sel = []
        for n in (10, 5):
            sub = D[ok & (D.h == "H144") & (D.n == n)]
            if len(sub):
                sel.append(sub.sort_values("dev_t", ascending=False).iloc[0]["var"])
        sel += D[ok & (D.h == "H145")]["var"].tolist()
        D["selected"] = D["var"].isin(sel)
        print(f"\n== deflated Sharpe（試驗數 {TRIALS}）與 val 選擇")
        for _, r in D.iterrows():
            print(f"  {r['var']:<16} DSR {r.dsr:.3f}{'  ← 進 val' if r.selected else ''}")
    D.to_csv(OUT / "batch49_dev.csv", index=False)
    SL.log(dict(kind="strategy", period="dev", strategy="第四十九批 H144/H145 降低檔數", note="第四十九批 事前登記（事後動機）",
                rows=D.to_dict("records"), selected=sel, trials_cum=TRIALS))


def val():
    D = pd.read_csv(OUT / "batch49_dev.csv")
    sel = D[D.selected]
    if not len(sel):
        print("dev 沒有選出任何變體 → 依登記不跑 val")
        return
    comps = filt_comps()
    t0 = time.time()
    lab, dates, R, E, mats, lim, codes = load(VAL[1])
    print(f"載入 {time.time()-t0:.0f}s", flush=True)
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        print(f"\n== val {VAL[0]} ~ {VAL[1]}")
        refs = (("H142[50檔]（對照）", run144(mats, R, E, lim, comps, codes, 50, False)),
                ("H136[20檔]（對照）", run136(mats, R, E, lim, 20)))
        for nm, df in refs:
            print(f"  {nm:<20} {fmt(SL.stats(df, dates, *VAL), te(df, dates, VAL))}")
        for _, r in sel.iterrows():
            v = next(v for v in H144 if nm144(v) == r["var"]) if r.h == "H144" else int(r.n)
            df = run144(mats, R, E, lim, comps, codes, *v) if r.h == "H144" else run136(mats, R, E, lim, v)
            st = SL.stats(df, dates, *VAL)
            ok = st["t"] >= 2 and st["yrs"] == "4/4"
            m = (dates >= pd.Timestamp(VAL[0])) & (dates <= pd.Timestamp(VAL[1]))
            yr = (df.ex[m].groupby(dates[m].year).sum() * 100).round(1).to_dict()
            out[r["var"]] = dict(**st, te=te(df, dates, VAL), by_year=yr, verdict="✓" if ok else "✗")
            print(f"  {r['var']:<20} {fmt(st, out[r['var']]['te'])}｜逐年 {yr} → {out[r['var']]['verdict']}")
    SL.log(dict(kind="strategy", period="val", strategy="第四十九批 H144/H145 降低檔數", note="第四十九批 dev 選出的", result=out))


if __name__ == "__main__":
    {"dev": dev, "val": val}[sys.argv[1] if len(sys.argv) > 1 else "dev"]()
