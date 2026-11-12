"""第七十八批 H191：台指期每日主動買賣（tick rule Delta）擇時小台。規則見 HYPOTHESES.md 第七十八批（事前登記）。
資料：data/v2/futures/tx1md（每分鐘主動買賣）、MTX_daily（mtx.continuous）。

用法：.venv/bin/python research/v2/dflow_lab.py dev|val
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402
import mtx as M  # noqa: E402
import tx1md  # noqa: E402

NAME = "第七十八批 H191 台指期每日 Delta 擇時小台"
PER = {"dev": ("2012-01-01", "2019-12-31"), "val": ("2020-01-01", "2023-12-31")}
SEGS = {"A": ("2011-01-01", "2014-12-31"), "B": ("2015-01-01", "2019-12-31")}
WIN, MINW = 250, 120


def daily_dr(end):
    T = tx1md.load("2011-01-01", end)
    g = T.groupby("day")[["buy", "sell"]].sum()
    dr = (g.buy - g.sell) / (g.buy + g.sell)
    dr.index = pd.to_datetime(dr.index)
    return dr


def build(end):
    F = M.continuous(end)
    dr = daily_dr(end).reindex(F.index)
    S = F.S
    So = S.shift(1) * (1 + F.gap)                       # 開盤時的指數
    X = {}
    for N in (1, 5, 20):
        drn = dr.rolling(N).mean()
        rn = S.pct_change(N)
        e = pd.Series(np.nan, index=F.index)
        a, b = drn.values, rn.values
        for t in range(len(F)):
            lo = max(0, t - WIN + 1)
            x, y = b[lo:t + 1], a[lo:t + 1]
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < MINW or not np.isfinite(a[t]) or not np.isfinite(b[t]):
                continue
            beta, alpha = np.polyfit(x[ok], y[ok], 1)
            e.iloc[t] = a[t] - (alpha + beta * b[t])
        fwd = S.shift(-N) / So.shift(-1) - 1            # t+1 開盤 → t+N 收盤
        X[N] = dict(dr=drn, r=rn, e=e, fwd=fwd)
    return F, dr, X


def resid_delta(end, N=5):
    """特徵 RESID_DELTA_N（MARKET_FEATURES.md）：收盤 t 可得的殘差 Delta（日期索引；2011 年前半暖機為 NaN）。"""
    _, _, X = build(end)
    return X[N]["e"].rename(f"RESID_DELTA_{N}")


def nw_ols(y, Xm, lag):
    ok = np.isfinite(y) & np.all(np.isfinite(Xm), axis=1)
    y, Xm = y[ok], Xm[ok]
    n, k = Xm.shape
    XtX = np.linalg.inv(Xm.T @ Xm)
    b = XtX @ Xm.T @ y
    u = y - Xm @ b
    Z = Xm * u[:, None]
    Sm = Z.T @ Z
    for L in range(1, lag + 1):
        w = 1 - L / (lag + 1)
        G = Z[L:].T @ Z[:-L]
        Sm += w * (G + G.T)
    V = XtX @ Sm @ XtX
    return b, b / np.sqrt(np.diag(V)), n


def info(X, p0, p1):
    out = {}
    for N, x in X.items():
        df = pd.DataFrame(dict(y=x["fwd"], dr=x["dr"], r=x["r"])).loc[p0:p1].dropna()
        sd = df.dr.std()
        res = {}
        for lab, cols in (("dr+r", ["dr", "r"]), ("dr", ["dr"]), ("r", ["r"])):
            Xm = np.column_stack([np.ones(len(df))] + [df[c].values for c in cols])
            b, t, n = nw_ols(df.y.values, Xm, max(N, 1))
            res[lab] = (b[1], t[1])
        seg = {}
        for s, (a0, a1) in SEGS.items():
            d = df.loc[max(a0, p0):min(a1, p1)]
            if len(d) < 100:
                continue
            Xm = np.column_stack([np.ones(len(d)), d.dr.values, d.r.values])
            b, t, _ = nw_ols(d.y.values, Xm, max(N, 1))
            seg[s] = (b[1], t[1])
        out[N] = dict(n=len(df), sd=sd, **{k: v for k, v in res.items()}, seg=seg)
    return out


def signals(F, X):
    e5 = X[5]["e"]
    sig = {"T1 多／空手": (e5 > 0).astype(float), "T2 多空": np.sign(e5).fillna(0),
           "T1 N=1": (X[1]["e"] > 0).astype(float), "T2 N=1": np.sign(X[1]["e"]).fillna(0),
           "T1 N=20": (X[20]["e"] > 0).astype(float), "T2 N=20": np.sign(X[20]["e"]).fillna(0)}
    S = F.S
    low20 = S <= S.rolling(20).min()
    for lab, cond in (("T3 日線吸收", low20 & (e5 > 0)), ("對照 新低就買", low20)):
        s = np.zeros(len(S))
        for t in np.nonzero(cond.values)[0]:
            s[t:t + 5] = 1
        sig[lab] = pd.Series(s, index=S.index)
    return sig


def leak(period):
    end = PER[period][1]
    cut = "2015-06-30"
    Fa, _, Xa = build(end)
    Fb, _, Xb = build(cut)
    sa, sb = signals(Fa, Xa), signals(Fb, Xb)
    k = len(Fb)
    return all(np.allclose(np.asarray(sa[h])[:k], np.asarray(sb[h])[:k]) for h in sa)


def main(period):
    t0 = time.time()
    p0, p1 = PER[period]
    F, dr, X = build(p1)
    print(f"{NAME}｜{period}：{dr.notna().sum()} 天有 Delta（{time.time()-t0:.0f}s）；DR 平均 {dr.mean():+.3f}、標準差 {dr.std():.3f}")
    print(f"  每日 DR 和當天報酬相關 {dr.corr(F.S.pct_change()):.2f}")
    ip0 = "2011-01-01" if period == "dev" else p0
    I = info(X, ip0, p1)
    print("\n== A. 資訊檢定：未來 N 日報酬（t+1 開 → t+N 收）～ DR_N ＋ r_N（係數＝每 1σ DR 的 bp）")
    ok_info = {}
    for N, v in I.items():
        b, t = v["dr+r"]
        seg = "  ".join(f"{s} {bb*v['sd']*1e4:+.1f}bp（t {tt:+.2f}）" for s, (bb, tt) in v["seg"].items())
        ok = t >= 2 and all(bb > 0 for bb, _ in v["seg"].values()) if period == "dev" else t >= 2
        ok_info[f"N={N}"] = bool(ok)
        print(f"  N={N:<3} n={v['n']}  DR｜控制價格 {b*v['sd']*1e4:+.1f}bp t {t:+.2f} {'✓' if ok else '✗'}"
              f"｜只放 DR {v['dr'][0]*v['sd']*1e4:+.1f}bp t {v['dr'][1]:+.2f}｜只放 r 係數 t {v['r'][1]:+.2f}｜{seg}")
    tr = pd.read_parquet(H.D / "taiex_tr.parquet").assign(date=lambda x: pd.to_datetime(x.date)).set_index("date").price
    rtr = tr.pct_change().loc[p0:p1]
    rb, _ = M.run_pos(F, *M.strategy("BH", F))
    rb = rb.loc[p0:p1]
    mb, mt = M.metrics(rb), M.metrics(rtr)
    bar = max(mb["Sharpe"], mt["Sharpe"])
    print(f"\n== B. 交易（{p0}～{p1}）基準：小台做多 Sharpe {mb['Sharpe']:.2f} CAGR {mb['CAGR']*100:+.1f}% MDD {mb['MDD']*100:.1f}%"
          f"｜加權報酬指數 Sharpe {mt['Sharpe']:.2f} → 門檻 {bar:.2f}")
    lk = leak(period) if period == "dev" else None
    if lk is not None:
        print(f"  前視檢查（截斷 2015-06-30）：{'OK' if lk else '✗✗'}")
    R, ok_tr = {}, {}
    for h, s in signals(F, X).items():
        n, d = M.from_signal(s.values)
        r, tv = M.run_pos(F, n, d)
        r, tv = r.loc[p0:p1], tv.loc[p0:p1]
        m = M.metrics(r, rb.reindex(r.index))
        yrs = (r.index[-1] - r.index[0]).days / 365.25
        expo = float(np.mean(np.abs(pd.Series(d, index=F.index).loc[p0:p1])))
        ok = m["Sharpe"] > bar
        R[h] = dict(Sharpe=m["Sharpe"], CAGR=m["CAGR"], MDD=m["MDD"], JK_z=m["JK_z"], expo=expo, turn=tv.sum() / yrs,
                    years=m["years"], passed=bool(ok))
        if h in ("T1 多／空手", "T2 多空", "T3 日線吸收"):
            ok_tr[h] = bool(ok)
        print(f"  {h:<12} Sharpe {m['Sharpe']:5.2f} {'✓' if ok else '✗'}  CAGR {m['CAGR']*100:+6.1f}%  MDD {m['MDD']*100:6.1f}%"
              f"  曝險 {expo:.2f}  年交易 {tv.sum()/yrs:5.1f} 邊  JK z {m['JK_z']:+.2f}")
        print("      " + "  ".join(f"{k}:{v*100:+.1f}" for k, v in m["years"].items()))
    # T3 事件層：未來 5 日（t+1 開 → t+5 收）報酬
    e5, S = X[5]["e"], F.S
    low20 = (S <= S.rolling(20).min())
    f5 = X[5]["fwd"]
    for lab, c in (("新低＋殘差 Delta > 0", low20 & (e5 > 0)), ("新低＋殘差 Delta ≤ 0", low20 & (e5 <= 0)), ("所有新低", low20 & e5.notna())):
        v = f5[c].loc[p0:p1].dropna()
        print(f"  事件 {lab:<16} {len(v):4d} 次  未來 5 日平均 {v.mean()*100:+.2f}%  勝率 {(v > 0).mean():.0%}")
    print(f"  所有日子未來 5 日平均 {f5.loc[p0:p1].mean()*100:+.2f}%")
    verdict = "✓" if any(ok_tr.values()) else "✗"
    print(f"\n判定：資訊 {ok_info}｜交易 {ok_tr} → {verdict}")
    with open(H.LOG, "a") as f:
        f.write(json.dumps(dict(ts=time.strftime("%Y-%m-%d %H:%M:%S"), kind="mtx", period=period, strategy=NAME,
                                note="第七十八批 事前登記", bar=bar, info={f"N={k}": v for k, v in I.items()}, trade=R,
                                checks=dict(info=ok_info, trade=ok_tr), leak_ok=lk, verdict=verdict),
                           ensure_ascii=False,
                           default=lambda v: round(float(v), 6) if isinstance(v, (float, np.floating)) else str(v)) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
