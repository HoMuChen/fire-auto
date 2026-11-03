"""第五十五批（STRATEGY_PLAN S8）：H161 做多 H146＋空小台指、H162 個股多空（前 150 大近似個股期貨）。
規則見 HYPOTHESES.md 第五十五批（事前登記）。

用法：.venv/bin/python research/v2/batch55_s8.py dev|val
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

MOM = {"H2": 1, "H10": 1, "H87": 1, "H86": 1}
VAL_ = {"EBITDA_EV": 1, "SALE_ME": 1, "NETDEBT_ME": -1, "MISP_MGMT": 1}
SCORES = {"動能營收": MOM, "價值紀律": VAL_, "合成": {**MOM, **VAL_}}
NEED = {"動能營收": 3, "價值紀律": 3, "合成": 5}
P1, P2, VAL = B48.P1, B48.P2, B48.VAL
PER = {"動能營收": P1, "價值紀律": P2, "合成": P2}
K, TOPV, SSF_COST, FUT_COST, MARGIN, NPL, TRIALS = 20, 150, 0.001, 0.0002, 0.15, 100, 93
OUT = H.D / "results"


def mtx_chain(days):
    """小台指日盤近月連續（開盤到開盤）：每天持有結算日還沒到的最近月；回傳以交易日為索引的累積指數。"""
    d = pd.read_parquet(H.D / "futures" / "MTX_daily.parquet")
    d = d[(d.trading_session == "position") & (d.open > 0)].copy()
    d["date"] = pd.to_datetime(d.date)
    d["c"] = d.contract_date.astype(str)
    last = d.groupby("c").date.max()                                     # 結算日（最後出現的那天）
    op = d.pivot_table(index="date", columns="c", values="open", aggfunc="last")
    op = op.reindex(days)
    cs = sorted(last.index)
    lastv = last.reindex(cs).values
    r = np.zeros(len(days))
    for i in range(1, len(days)):
        t0 = days[i - 1]
        k = next((j for j, c in enumerate(cs) if lastv[j] > t0 and c in op.columns and np.isfinite(op.at[t0, c])), None)
        if k is None:
            continue
        c = cs[k]
        a, b = op.at[t0, c], op.at[days[i], c]
        if np.isfinite(a) and np.isfinite(b) and a > 0:
            r[i] = b / a - 1
    return pd.Series(np.cumprod(1 + r), index=days)


def load(end):
    lab = SL.Lab(end)
    dates, R, E = lab.monthly("2010-01-01", end)
    lim = lab.limits(dates)
    feats = set(MOM) | set(VAL_) | set(B49.filt_comps())
    mats = {f: lab.feat(f, dates, E) for f in feats}
    m60 = lab.P["money"].astype(float).rolling(60, min_periods=40).mean().shift(1).reindex(index=dates, columns=lab.cols).values
    U = E & np.isfinite(m60) & (pd.DataFrame(np.where(E, m60, np.nan)).rank(axis=1, ascending=False).values <= TOPV)
    idx = lab.P["open"].index
    ch = mtx_chain(idx)
    k = [idx.get_loc(d) for d in dates]
    nxt = k[1:] + [None]
    mtx = np.array([ch.iloc[b] / ch.iloc[a] - 1 if b is not None else np.nan for a, b in zip(k, nxt)])
    return lab, dates, R, E, lim, mats, U, mtx


def h146(mats, R, E, lim):
    S, filt = B49.s142(mats, E, B49.filt_comps())
    return SL.portfolio(S, R, E, 50, buffer=3, filt=filt, limits=lim)


def as_df(ret, n=None):
    return pd.DataFrame(dict(net=ret, bench=0.0, ex=ret, turn=np.nan, n=np.nan if n is None else n))


def h161(base, mtx, h):
    beta = np.full(len(base), np.nan)
    if h == "β":
        x, y = mtx, base.net.values
        for t in range(len(base)):
            a = max(0, t - 24)
            k = np.isfinite(x[a:t]) & np.isfinite(y[a:t])
            if k.sum() >= 12:
                b = np.polyfit(x[a:t][k], y[a:t][k], 1)[0]
                beta[t] = np.clip(b, 0.5, 1.5)
    hr = np.ones(len(base)) if h == "1" else beta
    ret = (1 - MARGIN) * base.net.values - (1 - MARGIN) * hr * mtx - FUT_COST
    return as_df(ret), hr


def h162(mats, R, E, U, lim, name):
    S = SL.combo([(mats[f], s) for f, s in SCORES[name].items()], U, NEED[name])
    lo = SL.portfolio(S, R, E, K, buffer=2, filt=U, limits=lim, cost=SSF_COST)
    sh = SL.portfolio(-S, R, E, K, buffer=2, filt=U, limits=(lim[1], lim[0]), cost=SSF_COST)
    ret = lo.net.values - (sh.net.values + 2 * sh.turn.values * SSF_COST)
    return as_df(ret)


def corr(a, b, dates, p):
    m = (dates >= pd.Timestamp(p[0])) & (dates <= pd.Timestamp(p[1])) & np.isfinite(a) & np.isfinite(b)
    return np.corrcoef(a[m], b[m])[0, 1]


def fmt(s):
    return f"年化 {s['ex']*100:+6.1f}% t {s['t']:5.2f} 正年 {s['yrs']:>4} Sharpe {s['sharpe']:.2f} MDD {s['mdd']*100:6.1f}%"


def yrs_ok(s, p):
    a, b = map(int, s["yrs"].split("/"))
    return a >= (7 if p == P1 else 4) if b >= 5 else a == b


def dev():
    t0 = time.time()
    lab, dates, R, E, lim, mats, U, mtx = load(H.PERIODS["dev"][1])
    print(f"載入 {time.time()-t0:.0f}s；前 150 大池平均 {U.sum(axis=1)[U.sum(axis=1) > 0].mean():.0f} 檔", flush=True)
    rows, ser = [], {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        base = h146(mats, R, E, lim)
        sb = SL.stats(as_df(base.net.values), dates, *P1)
        sm = SL.stats(as_df(mtx), dates, *P1)
        print(f"\n== dev 對現金  H146（只做多，對照）{fmt(sb)}｜小台指近月連續 {fmt(sm)}")
        for h in ("1", "β"):
            df, hr = h161(base, mtx, h)
            st = SL.stats(df, dates, *P1)
            ok = st["t"] >= 2 and yrs_ok(st, P1)
            rows.append(dict(h="H161", var=f"H161[避險{h}]", **{f"dev_{k}": x for k, x in st.items()}, rho_mkt=corr(df.ex.values, mtx, dates, P1),
                             rho_h146=corr(df.ex.values, base.net.values, dates, P1), hr=np.nanmean(hr), pass_=ok))
            ser[rows[-1]["var"]] = (df.ex.values, P1)
            print(f"  {rows[-1]['var']:<14} {fmt(st)}｜和大盤相關 {rows[-1]['rho_mkt']:+.2f}｜平均避險比例 {rows[-1]['hr']:.2f} → {'過' if ok else '✗'}", flush=True)
        for name in SCORES:
            p = PER[name]
            df = h162(mats, R, E, U, lim, name)
            st = SL.stats(df, dates, *p)
            ok = st["t"] >= 2 and yrs_ok(st, p)
            rows.append(dict(h="H162", var=f"H162[{name}]", **{f"dev_{k}": x for k, x in st.items()}, period=f"{p[0][:4]}-{p[1][:4]}",
                             rho_mkt=corr(df.ex.values, mtx, dates, p), rho_h146=corr(df.ex.values, base.net.values, dates, p), pass_=ok))
            ser[rows[-1]["var"]] = (df.ex.values, p)
            print(f"  {rows[-1]['var']:<14} {fmt(st)}（{p[0][:4]}–{p[1][:4]}）｜和大盤相關 {rows[-1]['rho_mkt']:+.2f}｜和 H146 相關 {rows[-1]['rho_h146']:+.2f} → {'過' if ok else '✗'}", flush=True)
        D = pd.DataFrame(rows)

        def best(m_):
            mm = {**mats, **m_}
            return max(SL.stats(h162(mm, R, E, U, lim, n), dates, *PER[n])["t"] for n in SCORES)
        sub = {f: mats[f] for f in set(MOM) | set(VAL_)}
        t1 = time.time()
        pl = SL.placebo(best, sub, n=NPL, seed=2035)
        p95 = np.percentile(pl, 95)
        print(f"\n== H162 安慰劑（{NPL} 次，菜單最佳 t；{time.time()-t1:.0f}s）：50／95 分位 {np.percentile(pl, 50):.2f}／{p95:.2f}")
        srs = []
        for k, (x, p) in ser.items():
            m = (dates >= pd.Timestamp(p[0])) & (dates <= pd.Timestamp(p[1])) & np.isfinite(x)
            srs.append(x[m].mean() / x[m].std(ddof=1))
        D["dsr"] = [SL.deflated_sharpe(x[(dates >= pd.Timestamp(p[0])) & (dates <= pd.Timestamp(p[1])) & np.isfinite(x)], srs, TRIALS)
                    for x, p in ser.values()]
        D["placebo_p95"] = np.where(D.h == "H162", p95, np.nan)
        sel = {}
        for h in ("H161", "H162"):
            sub_ = D[(D.h == h) & D.pass_ & ((D.h == "H161") | (D.dev_t > D.placebo_p95))]
            sel[h] = sub_.sort_values("dev_t", ascending=False).iloc[0]["var"] if len(sub_) else None
        D["selected"] = D["var"].isin([v for v in sel.values() if v])
        ov = None
        if sel["H162"]:
            name = sel["H162"][5:-1]
            p = PER[name]
            ls = h162(mats, R, E, U, lim, name)
            o = as_df((1 - MARGIN) * base.net.values + 0.5 * ls.ex.values)
            so, sb2 = SL.stats(o, dates, *p), SL.stats(as_df(base.net.values), dates, *p)
            ok = so["sharpe"] > sb2["sharpe"] and so["mdd"] >= sb2["mdd"]
            ov = dict(name=name, **so, base_sharpe=sb2["sharpe"], base_mdd=sb2["mdd"], pass_=ok)
            print(f"\n== 疊加（{p[0][:4]}–{p[1][:4]}）：85% H146 ＋ 0.5 倍 {name} 多空  {fmt(so)}｜H146 單獨 Sharpe {sb2['sharpe']:.2f} MDD {sb2['mdd']*100:.1f}% → {'過' if ok else '✗'}")
        print(f"\n== deflated Sharpe（試驗數 {TRIALS}）與 val 選擇：{sel}")
        for _, r in D.iterrows():
            print(f"  {r['var']:<14} DSR {r.dsr:.3f}{'  ← 進 val' if r.selected else ''}")
    D.to_csv(OUT / "batch55_dev.csv", index=False)
    SL.log(dict(kind="strategy", period="dev", strategy="第五十五批 H161/H162 S8 避險與多空", note="第五十五批 事前登記",
                rows=D.to_dict("records"), placebo_p95=p95, selected=sel, overlay=ov, trials_cum=TRIALS))


def val():
    D = pd.read_csv(OUT / "batch55_dev.csv")
    sel = D[D.selected]
    if not len(sel):
        print("dev 沒有選出 → 依登記不跑 val")
        return
    t0 = time.time()
    lab, dates, R, E, lim, mats, U, mtx = load(VAL[1])
    print(f"載入 {time.time()-t0:.0f}s", flush=True)
    out = {}
    m = (dates >= pd.Timestamp(VAL[0])) & (dates <= pd.Timestamp(VAL[1]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        base = h146(mats, R, E, lim)
        sb = SL.stats(as_df(base.net.values), dates, *VAL)
        print(f"\n== val {VAL[0]} ~ {VAL[1]}（對現金）  H146（只做多，對照）{fmt(sb)}｜小台指 {fmt(SL.stats(as_df(mtx), dates, *VAL))}")
        for _, r in sel.iterrows():
            if r.h == "H161":
                df, _ = h161(base, mtx, r["var"][7:-1])
            else:
                df = h162(mats, R, E, U, lim, r["var"][5:-1])
            st = SL.stats(df, dates, *VAL)
            ok = st["t"] >= 2 and st["yrs"] == "4/4"
            yr = (df.ex[m].groupby(dates[m].year).sum() * 100).round(1).to_dict()
            out[r["var"]] = dict(**st, rho_mkt=corr(df.ex.values, mtx, dates, VAL), rho_h146=corr(df.ex.values, base.net.values, dates, VAL),
                                 by_year=yr, verdict="✓" if ok else "✗")
            print(f"  {r['var']:<14} {fmt(st)}｜和大盤相關 {out[r['var']]['rho_mkt']:+.2f}｜和 H146 相關 {out[r['var']]['rho_h146']:+.2f}｜逐年 {yr} → {out[r['var']]['verdict']}")
            if r.h == "H162":
                o = as_df((1 - MARGIN) * base.net.values + 0.5 * df.ex.values)
                so = SL.stats(o, dates, *VAL)
                ok2 = so["sharpe"] > sb["sharpe"] and so["mdd"] >= sb["mdd"]
                out["overlay"] = dict(**so, pass_=ok2)
                print(f"  疊加 85% H146 ＋ 0.5 倍多空  {fmt(so)}｜H146 單獨 Sharpe {sb['sharpe']:.2f} MDD {sb['mdd']*100:.1f}% → {'過' if ok2 else '✗'}")
    SL.log(dict(kind="strategy", period="val", strategy="第五十五批 H161/H162 S8 避險與多空", note="第五十五批 dev 選出的", result=out))


if __name__ == "__main__":
    {"dev": dev, "val": val}[sys.argv[1] if len(sys.argv) > 1 else "dev"]()
