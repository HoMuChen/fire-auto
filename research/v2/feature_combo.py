"""第三十三批 FS2：FS1 的 20 群代表特徵隨機組合 → 分布、邊際貢獻、事前規則選出唯一的 C* 進 val。

規則見 HYPOTHESES.md 第三十三批（事前登記）。組合分數＝pit 流動池內各特徵百分位等權平均（H29 法），
有值的特徵數 ≥ ceil(2k/3)；top20 等權月換股，成本同 xs.backtest。H29 對照用同一個引擎的 {H2, H10}。

用法：.venv/bin/python research/v2/feature_combo.py dev|val
"""
import itertools
import json
import math
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

POS = ["H10", "H87", "H38", "H6", "H11", "H13", "H88", "H24", "H12", "H3", "H86", "H97", "H2", "H89", "H1", "H99"]
NEG = ["H35", "H85", "H84", "H36"]
FEATS = POS + NEG
TOPN, MIN_VALID, NBERN, SEED = 20, 50, 2000, 2026
A_END = pd.Timestamp("2014-12-31")
OUT = H.D / "results"


def _log(rec):
    with open(H.LOG, "a") as f:
        f.write(json.dumps(dict(ts=time.strftime("%Y-%m-%d %H:%M:%S"), **rec), ensure_ascii=False,
                           default=lambda v: round(float(v), 5) if isinstance(v, (float, np.floating)) else str(v)) + "\n")


def prep(P, p0, p1, feats=None):
    """回傳換股日、次月報酬 R、可交易 E、特徵百分位 F[特徵, 月, 股]（換股日前一交易日的值，pit 池內排名）。"""
    O, Cf = P["open"], P["close"].ffill()
    idx = O.index
    rd = xs.rebalance_dates(idx, pd.Timestamp(p0), pd.Timestamp(p1))
    elig = P["liq"].shift(1).fillna(False).astype(bool) & (P["nbars"].shift(1) >= 250) & O.notna()
    dates = rd[:-1]
    R = np.vstack([((O.loc[d2].fillna(Cf.iloc[idx.get_loc(d2) - 1]) / O.loc[d] - 1).clip(-0.95, 10)).values
                   for d, d2 in zip(rd[:-1], rd[1:])])
    E = elig.loc[dates].values & np.isfinite(R)
    feats = feats or FEATS
    F = np.full((len(feats), len(dates), O.shape[1]), np.nan)
    for j, f in enumerate(feats):
        S = Hy.FNS[f][0](P).replace([np.inf, -np.inf], np.nan).shift(1).reindex(index=dates, columns=O.columns)
        S = (-S if f in NEG else S).where(E)
        F[j] = S.rank(axis=1, pct=True).values
    return pd.DatetimeIndex(dates), R, E, F


def bt(F, R, E, sub, ret_top=False):
    """回傳每月扣成本報酬 net 與超額 ex（不足 MIN_VALID 的月為 NaN）；ret_top 另回傳每月持股（欄位位置）。"""
    X = F[list(sub)]
    need = math.ceil(2 * len(sub) / 3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        M = np.nanmean(X, axis=0)
    M[(np.isfinite(X).sum(axis=0) < need) | ~E] = np.nan
    T, N = M.shape
    order = np.argsort(np.where(np.isfinite(M), -M, np.inf), axis=1, kind="stable")[:, :TOPN]
    bench = np.array([R[t][E[t]].mean() for t in range(T)])
    net, ex = np.full(T, np.nan), np.full(T, np.nan)
    prev, tops = None, [None] * T
    for t in range(T):
        if np.isfinite(M[t]).sum() < MIN_VALID:
            continue
        top = order[t]
        cur = set(top.tolist())
        turn = 1 - len(cur & prev) / TOPN if prev is not None else 1.0
        net[t] = R[t, top].mean() - turn * xs.COST
        ex[t] = net[t] - bench[t]
        prev = tops[t] = cur
    return (net, ex, tops) if ret_top else (net, ex)


def st(net, ex, m):
    n, e = net[m], ex[m]
    k = np.isfinite(e)
    n, e = n[k], e[k]
    if len(e) < 12:
        return dict(n=len(e), ex=np.nan, t=np.nan, sh=np.nan)
    return dict(n=len(e), ex=e.mean() * 12, t=e.mean() / e.std(ddof=1) * np.sqrt(len(e)), sh=n.mean() / n.std(ddof=1) * np.sqrt(12))


def subsets():
    nf = len(FEATS)
    pairs = [tuple(c) for c in itertools.combinations(range(nf), 2)]
    triples = [tuple(c) for c in itertools.combinations(range(nf), 3)]
    rng, seen, bern = np.random.default_rng(SEED), set(), []
    while len(bern) < NBERN:
        s = tuple(np.flatnonzero(rng.random(nf) < 0.5).tolist())
        if len(s) >= 2 and s not in seen:
            seen.add(s)
            bern.append(s)
    return [("pair", s) for s in pairs] + [("triple", s) for s in triples] + [("bern", s) for s in bern]


def name(sub):
    return "+".join(FEATS[i] for i in sub)


def run_period(P, p0, p1, subs):
    dates, R, E, F = prep(P, p0, p1)
    mA, mB, mAll = dates <= A_END, dates > A_END, np.ones(len(dates), bool)
    rows, series = [], {}
    for kind, sub in subs:
        net, ex = bt(F, R, E, sub)
        series[sub] = (net, ex)
        a, b, al = st(net, ex, mA), st(net, ex, mB), st(net, ex, mAll)
        rows.append(dict(kind=kind, combo=name(sub), k=len(sub), sub=sub, **{f"A_{x}": a[x] for x in a},
                         **{f"B_{x}": b[x] for x in b}, **{f"all_{x}": al[x] for x in al}))
    return pd.DataFrame(rows), series, dates


def marginal(D, col):
    B = D[D.kind == "bern"]
    X = np.zeros((len(B), len(FEATS)))
    for i, s in enumerate(B["sub"]):
        X[i, list(s)] = 1
    X1 = np.column_stack([np.ones(len(B)), X])
    y = B[col].values
    ok = np.isfinite(y)
    beta, *_ = np.linalg.lstsq(X1[ok], y[ok], rcond=None)
    res = y[ok] - X1[ok] @ beta
    s2 = res @ res / (ok.sum() - X1.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X1[ok].T @ X1[ok])))
    return pd.DataFrame(dict(coef=beta[1:], t=beta[1:] / se[1:]), index=FEATS)


def dev_report(D, base):
    print(f"\n== H29（同引擎 {{H2,H10}}）dev：A 超額 {base['A_ex']*100:+.1f}% t {base['A_t']:.2f} | B {base['B_ex']*100:+.1f}% t {base['B_t']:.2f}"
          f" | 全 {base['all_ex']*100:+.1f}% t {base['all_t']:.2f} Sharpe {base['all_sh']:.2f}")
    print("\n== 隨機組合分布（dev 全期年化超額 %：10 / 50 / 90 分位；Sharpe 中位；A、B 都贏 H29 的比例）")
    D = D.assign(win=(D.A_ex > base["A_ex"]) & (D.B_ex > base["B_ex"]))
    for lab, g in [("2 個一組", D[D.kind == "pair"]), ("3 個一組", D[D.kind == "triple"])] + \
            [(f"伯努利 k={a}-{b}", D[(D.kind == "bern") & D.k.between(a, b)]) for a, b in ((2, 7), (8, 9), (10, 11), (12, 13), (14, 20))]:
        if len(g):
            q = (g.all_ex * 100).quantile([0.1, 0.5, 0.9]).values
            print(f"  {lab:<14} {len(g):>5} 組  {q[0]:+6.1f} / {q[1]:+6.1f} / {q[2]:+6.1f}   Sharpe {g.all_sh.median():.2f}   兩段都贏 {g.win.mean()*100:5.1f}%")
    print("\n== dev 前 10 名組合（描述；挑最好的＝過度擬合）")
    for _, r in D.sort_values("all_ex", ascending=False).head(10).iterrows():
        print(f"  {r.combo:<40} 全 {r.all_ex*100:+5.1f}% t {r.all_t:.2f}  A {r.A_ex*100:+5.1f}%  B {r.B_ex*100:+5.1f}%")
    mA, mB = marginal(D, "A_ex"), marginal(D, "B_ex")
    M = pd.DataFrame(dict(A_coef=mA.coef * 100, A_t=mA.t, B_coef=mB.coef * 100, B_t=mB.t))
    M["desc"] = [Hy.FNS[f][1] + ("（反向）" if f in NEG else "") for f in FEATS]
    M = M.sort_values("B_coef", ascending=False)
    print("\n== 邊際貢獻（伯努利 2000 組；放進組合時年化超額平均多幾個百分點）")
    for f, r in M.iterrows():
        mark = "★" if r.A_coef > 0 and r.B_coef > 0 else " "
        print(f"  {mark} {f:<4} {r.desc:<14} A {r.A_coef:+5.2f} (t {r.A_t:+5.1f})   B {r.B_coef:+5.2f} (t {r.B_t:+5.1f})")
    cstar = tuple(i for i, f in enumerate(FEATS) if M.loc[f, "A_coef"] > 0 and M.loc[f, "B_coef"] > 0)
    return M, cstar


def main(period):
    if period not in ("dev", "val"):
        raise PermissionError("只跑 dev / val；holdout 封存中")
    t0 = time.time()
    p1 = H.PERIODS[period][1]
    P = xs.load_panel(p1, fundamentals=True)
    for add in (xs.add_chips, xs.add_fin, xs.add_v16, xs.add_v17):
        add(P)
    print(f"載入 {time.time()-t0:.0f}s；{len(FEATS)} 個特徵", flush=True)
    subs = subsets()
    h29 = (FEATS.index("H2"), FEATS.index("H10"))
    D, Sd, _ = run_period(P, *H.PERIODS["dev"], subs + [("base", h29)])
    base = D[D.kind == "base"].iloc[0]
    D = D[D.kind != "base"]
    print(f"dev {len(D)} 組 [{time.time()-t0:.0f}s]", flush=True)
    M, cstar = dev_report(D, base)
    print(f"\n  C*（A、B 邊際係數都 > 0）＝{name(cstar)}（{len(cstar)} 個）")
    OUT.mkdir(exist_ok=True)
    if period == "dev":
        D.drop(columns="sub").to_csv(OUT / "fs2_dev.csv", index=False)
        M.to_csv(OUT / "fs2_marginal.csv")
        _log(dict(kind="feature_combo", period="dev", strategy="FS2 特徵隨機組合", note="第三十三批 事前登記",
                  n_combos=len(D), h29_all_ex=base.all_ex, combo_median_ex=D.all_ex.median(),
                  frac_beat_h29_both=float(((D.A_ex > base.A_ex) & (D.B_ex > base.B_ex)).mean()),
                  cstar=name(cstar), marginal=M[["A_coef", "B_coef"]].round(3).to_dict("index")))
        return
    best = D.sort_values("all_ex", ascending=False).iloc[0]["sub"]
    V, Sv, vd = run_period(P, *H.PERIODS["val"], subs + [("base", h29), ("cstar", cstar)])
    vb, vc = V[V.kind == "base"].iloc[0], V[V.kind == "cstar"].iloc[0]
    vbest = V[(V.kind != "base") & (V.kind != "cstar") & (V.combo == name(best))].iloc[0]
    d = Sv[cstar][0] - Sv[h29][0]
    d = d[np.isfinite(d)]
    dm, dt = d.mean(), d.mean() / d.std(ddof=1) * np.sqrt(len(d))
    by = pd.Series(d, index=vd[np.isfinite(Sv[cstar][0] - Sv[h29][0])]).groupby(lambda x: x.year).sum()
    v = "✓" if dm > 0 and dt >= 2 else ("方向對、不足以換" if dm > 0 else "✗")
    Vr = V[~V.kind.isin(["base", "cstar"])].set_index("combo")
    Dr = D.set_index("combo")
    rho = Dr.all_ex.rank().corr(Vr.all_ex.reindex(Dr.index).rank())
    print(f"\n== val 2020-23（top20，扣成本；年化超額 / t / Sharpe）")
    print(f"  H29（同引擎）     {vb.all_ex*100:+6.1f}% / {vb.all_t:.2f} / {vb.all_sh:.2f}")
    print(f"  C*（{len(cstar)} 個）       {vc.all_ex*100:+6.1f}% / {vc.all_t:.2f} / {vc.all_sh:.2f}")
    print(f"  dev 第一名（描述） {vbest.all_ex*100:+6.1f}% / {vbest.all_t:.2f} / {vbest.all_sh:.2f}   {name(best)}")
    print(f"  C* − H29 月差 {dm*100:+.3f}%（年化 {dm*1200:+.2f}%）t {dt:+.2f}  贏的年 {int((by > 0).sum())}/{len(by)}  → {v}")
    print(f"  全部 {len(Dr)} 組 dev vs val 年化超額 Spearman {rho:+.2f}（描述）")
    _log(dict(kind="feature_combo", period="val", strategy="FS2 C*", note="第三十三批 C* vs H29（同引擎）",
              cstar=name(cstar), cstar_ex=vc.all_ex, cstar_t=vc.all_t, cstar_sh=vc.all_sh, h29_ex=vb.all_ex, h29_sh=vb.all_sh,
              diff_m=dm, t=dt, yrs_won=f"{int((by > 0).sum())}/{len(by)}", verdict=v,
              best_dev=name(best), best_dev_val_ex=vbest.all_ex, spearman_dev_val=rho))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dev")
