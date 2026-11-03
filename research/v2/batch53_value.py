"""第五十三批：H159「價值＋紀律」寬投組，以及和 H142（50 檔、留任 3N）的 50／50、70／30 組合。
規則見 HYPOTHESES.md 第五十三批（事前登記）。

用法：.venv/bin/python research/v2/batch53_value.py dev|val
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

COMP = {"EBITDA_EV": 1, "SALE_ME": 1, "NETDEBT_ME": -1, "MISP_MGMT": 1}
MENU = ["前20%", "前30%", "全池傾斜"]
MIX = [0.5, 0.7]                                   # H142 的資金比例
P2, VAL, NPL, TRIALS = B48.P2, B48.VAL, 100, 79
OUT = H.D / "results"


def check_signs():
    L = pd.read_csv(H.D / "results" / "feature_library.csv").set_index("H")
    bad = {f: s for f, s in COMP.items() if float(L.loc[f, "sign"]) != s}
    if bad:
        raise RuntimeError(f"方向和特徵庫不一致：{bad}")


def load(end):
    lab, dates, R, E, mats, lim, codes = B49.load(end)
    for f in COMP:
        if f not in mats:
            mats[f] = lab.feat(f, dates, E)
    return lab, dates, R, E, mats, lim


def h142(mats, R, E, lim, comps):
    S, filt = B49.s142(mats, E, comps)
    return SL.portfolio(S, R, E, 50, buffer=3, filt=filt, limits=lim)


def h159(mats, R, E, v):
    q = B48.pct(SL.combo([(mats[f], s) for f, s in COMP.items()], E, 3), E)
    if v == "全池傾斜":
        w = np.where(E, np.where(np.isfinite(q), 1 + q - 0.5, 1.0), 0.0)
    else:
        w = np.where(E & (q > (0.8 if v == "前20%" else 0.7)), 1.0, 0.0)
    s = w.sum(axis=1, keepdims=True)
    return SL.weights_portfolio(np.divide(w, s, out=np.zeros_like(w), where=s > 0), R, E)


def mix(a, b, wa):
    net = wa * a.net.values + (1 - wa) * b.net.values
    return pd.DataFrame(dict(net=net, bench=a.bench.values, ex=net - a.bench.values, turn=np.nan,
                             n=a.n.values + b.n.values))


def jk(x, y):
    """Jobson-Korkie（Memmel 修正）：x、y 月報酬的 Sharpe 差 z。"""
    k = np.isfinite(x) & np.isfinite(y)
    x, y = x[k], y[k]
    s1, s2 = x.mean() / x.std(ddof=1), y.mean() / y.std(ddof=1)
    r = np.corrcoef(x, y)[0, 1]
    v = (2 * (1 - r) + 0.5 * (s1 ** 2 + s2 ** 2 - 2 * s1 * s2 * r ** 2)) / len(x)
    return (s1 - s2) / np.sqrt(v)


def corr(a, b, dates, p):
    m = (dates >= pd.Timestamp(p[0])) & (dates <= pd.Timestamp(p[1])) & np.isfinite(a) & np.isfinite(b)
    return np.corrcoef(a[m], b[m])[0, 1]


def fmt(s):
    return (f"超額 {s['ex']*100:+6.1f}% t {s['t']:5.2f} 正年 {s['yrs']:>4} Sharpe {s['sharpe']:.2f} MDD {s['mdd']*100:6.1f}% "
            f"年報酬 {s['ret']*100:+5.1f}%")


def combos(base, v, dates, p):
    m = (dates >= pd.Timestamp(p[0])) & (dates <= pd.Timestamp(p[1]))
    sb = SL.stats(base, dates, *p)
    out = {}
    for wa in MIX:
        c = mix(base, v, wa)
        sc = SL.stats(c, dates, *p)
        z = jk(c.net.values[m], base.net.values[m])
        ok = sc["sharpe"] > sb["sharpe"] and sc["mdd"] > sb["mdd"] and sc["t"] >= 2
        out[f"{int(wa*100)}/{int(round((1-wa)*100))}"] = dict(**sc, jk_z=z, pass_=ok)
    return sb, out


def dev():
    check_signs()
    comps = B49.filt_comps()
    t0 = time.time()
    lab, dates, R, E, mats, lim = load(H.PERIODS["dev"][1])
    print(f"載入 {time.time()-t0:.0f}s", flush=True)
    rows = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        base = h142(mats, R, E, lim, comps)
        cp = SL.weights_portfolio(E / np.maximum(E.sum(axis=1, keepdims=True), 1), R, E, bench=np.zeros(len(R)))
        print(f"\n== dev B 段 {P2[0]} ~ {P2[1]}（對 pit 池等權；扣成本）")
        print(f"  H142（50 檔、留任 3N，對照） {fmt(SL.stats(base, dates, *P2))}")
        V = {}
        for v in MENU:
            df = h159(mats, R, E, v)
            V[v] = df
            st = SL.stats(df, dates, *P2)
            ca = SL.stats(df.assign(ex=df.net - cp.net), dates, *P2)
            rho = corr(df.ex.values, base.ex.values, dates, P2)
            ok = st["t"] >= 2 and int(st["yrs"].split("/")[0]) >= 4
            rows[v] = dict(var=f"H159[{v}]", **{f"B_{k}": x for k, x in st.items()}, ex_costpool=ca["ex"], rho_h142=rho, pass_=ok)
            print(f"  H159[{v}]  {fmt(st)} 換手 {st['turn']*100:3.0f}% 檔數 {st['hold']:.0f}｜對扣成本池 {ca['ex']*100:+.1f}%｜和 H142 相關 {rho:+.2f} → {'過' if ok else '✗'}", flush=True)
        D = pd.DataFrame(rows.values())

        def best(m_):
            return max(SL.stats(h159(m_, R, E, v), dates, *P2)["t"] for v in MENU)
        sub = {f: mats[f] for f in COMP}
        pl = SL.placebo(lambda m_: best({**mats, **m_}), sub, n=NPL, seed=2033)
        p95 = np.percentile(pl, 95)
        print(f"\n== 安慰劑（{NPL} 次，菜單最佳 B 段 t）：50／95 分位 {np.percentile(pl, 50):.2f}／{p95:.2f}；真實 {D.B_t.max():.2f}")
        m = (dates >= pd.Timestamp(P2[0])) & (dates <= pd.Timestamp(P2[1]))
        srs = [np.nanmean(V[v].ex.values[m]) / np.nanstd(V[v].ex.values[m], ddof=1) for v in MENU]
        D["dsr"] = [SL.deflated_sharpe(V[v].ex.values[m], srs, TRIALS) for v in MENU]
        D["placebo_p95"] = p95
        okd = D[D.pass_ & (D.B_t > p95)]
        sel = okd.sort_values("B_t", ascending=False).iloc[0]["var"] if len(okd) else None
        D["selected"] = D["var"] == sel
        for _, r in D.iterrows():
            print(f"  {r['var']:<16} DSR（試驗數 {TRIALS}）{r.dsr:.3f}{'  ← 選出' if r.selected else ''}")
        res = {}
        if sel:
            v = sel[5:-1]
            sb, res = combos(base, V[v], dates, P2)
            print(f"\n== 組合（B 段；H142 單獨 Sharpe {sb['sharpe']:.2f}、MDD {sb['mdd']*100:.1f}%）")
            for k, s in res.items():
                print(f"  H142／H159 {k}  {fmt(s)}｜JK z {s['jk_z']:+.2f} → {'過' if s['pass_'] else '✗'}")
    D.to_csv(OUT / "batch53_dev.csv", index=False)
    SL.log(dict(kind="strategy", period="dev", strategy="第五十三批 H159 價值＋紀律／與 H142 組合", note="第五十三批 事前登記（成分看過 dev 相關性）",
                rows=D.to_dict("records"), placebo_p95=p95, selected=sel, combos=res, trials_cum=TRIALS))


def val():
    D = pd.read_csv(OUT / "batch53_dev.csv")
    sel = D[D.selected]
    if not len(sel):
        print("dev 沒有選出 → 依登記不跑 val")
        return
    v = sel.iloc[0]["var"][5:-1]
    comps = B49.filt_comps()
    t0 = time.time()
    lab, dates, R, E, mats, lim = load(VAL[1])
    print(f"載入 {time.time()-t0:.0f}s", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        base, df = h142(mats, R, E, lim, comps), h159(mats, R, E, v)
        st = SL.stats(df, dates, *VAL)
        rho = corr(df.ex.values, base.ex.values, dates, VAL)
        ok = st["t"] >= 2 and st["yrs"] == "4/4"
        m = (dates >= pd.Timestamp(VAL[0])) & (dates <= pd.Timestamp(VAL[1]))
        yr = (df.ex[m].groupby(dates[m].year).sum() * 100).round(1).to_dict()
        print(f"\n== val {VAL[0]} ~ {VAL[1]}")
        print(f"  H142（對照）      {fmt(SL.stats(base, dates, *VAL))}")
        print(f"  H159[{v}]  {fmt(st)}｜和 H142 相關 {rho:+.2f}｜逐年 {yr} → {'✓' if ok else '✗'}")
        res = {}
        print("  組合 dev 沒過（Sharpe 下降）→ 依登記不跑 val")
    SL.log(dict(kind="strategy", period="val", strategy="第五十三批 H159 價值＋紀律／與 H142 組合", note="第五十三批 dev 選出的",
                result=dict(var=f"H159[{v}]", **st, rho_h142=rho, by_year=yr, verdict="✓" if ok else "✗", combos=res)))


if __name__ == "__main__":
    {"dev": dev, "val": val}[sys.argv[1] if len(sys.argv) > 1 else "dev"]()
