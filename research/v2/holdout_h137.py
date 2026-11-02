"""第三十六批 H137：H136 ＋ M2（ML1 HistGB）各半，對照 H136 降到同波動（w=0.69）＋現金。規則見 HYPOTHESES.md 第三十六批（事前登記）。

同一次跑重算第一階段（2015-01 ~ 2023-11，核對第三十五批）與 holdout（2024-01 ~ 2026-09）。
2023-12 那一窗跨進 holdout，兩段都不算。持股跨段連續（2024-01 的換手照 2023-12 的持股算）。

用法：.venv/bin/python research/v2/holdout_h137.py --unseal     # 使用者 2026-11-02 同意開 holdout
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import feature_combo as FC  # noqa: E402
import ml_walkforward as ML  # noqa: E402
import xs  # noqa: E402

END = "2026-10-20"
YEARS = range(2015, 2027)
W, CASH = 0.69, 0.015
SEGS = {"stage1": ("2015-01-01", "2023-11-30"), "holdout": ("2024-01-01", END)}


def jk_z(a, b):
    a1, a2 = a.mean() / a.std(), b.mean() / b.std()
    rho, T = np.corrcoef(a, b)[0, 1], len(a)
    return (a1 - a2) / np.sqrt((2 - 2 * rho + 0.5 * (a1 * a1 + a2 * a2 - 2 * a1 * a2 * rho * rho)) / T)


def nw_t(x, lag=3):
    x = np.asarray(x, float)
    e, n = x - x.mean(), len(x)
    s = e @ e / n
    for k in range(1, min(lag, n - 1) + 1):
        s += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    return x.mean() / np.sqrt(s / n)


def stats(n):
    eq = (1 + n).cumprod()
    by = n.groupby(n.index.year).apply(lambda x: (1 + x).prod() - 1)
    return dict(CAGR=eq.iloc[-1] ** (12 / len(n)) - 1, Sharpe=n.mean() / n.std(ddof=1) * np.sqrt(12),
                vol=n.std(ddof=1) * np.sqrt(12), MDD=(eq / eq.cummax() - 1).min(), years={str(y): v for y, v in by.items()})


def segment(N, tops136, topsm2, ti):
    """N：H136、M2、池等權的月報酬（該段）。回傳各組統計、月差、相關。"""
    cash = (1 + CASH) ** (1 / 12) - 1
    N = N.assign(組合=0.5 * N["H136"] + 0.5 * N["M2"], 對照=W * N["H136"] + (1 - W) * cash)
    d = N["組合"] - N["對照"]
    by = d.groupby(d.index.year).sum()
    m = N["池"]
    beta = {k: np.cov(N[k], m)[0, 1] / m.var() for k in ("H136", "M2")}
    res = {k: N[k] - beta[k] * m for k in beta}
    ov = [len(tops136[t] & topsm2[t]) for t in ti if tops136[t] and topsm2[t]]
    return dict(
        months=len(N), first=str(N.index[0].date()), last=str(N.index[-1].date()),
        stats={k: stats(N[k]) for k in ("H136", "M2", "組合", "對照", "池")},
        diff=dict(mean=d.mean(), t=d.mean() / d.std(ddof=1) * np.sqrt(len(d)), nw_t=nw_t(d), sd=d.std(ddof=1),
                  yrs_won=f"{int((by > 0).sum())}/{len(by)}", by_year={str(y): v for y, v in by.items()}),
        jk_blend_vs_h136=jk_z(N["組合"], N["H136"]),
        corr=N["H136"].corr(N["M2"]), resid_corr=res["H136"].corr(res["M2"]), beta=beta,
        overlap=float(np.mean(ov)), vol_ratio=N["組合"].std() / N["H136"].std())


def verdict(d):
    if d["mean"] > 0 and d["t"] >= 2:
        return "✓ 顯著"
    return "方向對、不顯著" if d["mean"] > 0 else "✗"


def show(name, s):
    print(f"\n== {name}（{s['first']} ~ {s['last']}，{s['months']} 個月；月報酬算）")
    for k, v in s["stats"].items():
        print(f"  {k:<5} CAGR {v['CAGR']*100:+6.1f}%  Sharpe {v['Sharpe']:.2f}  年化波動 {v['vol']*100:4.1f}%  MDD {v['MDD']*100:6.1f}%  逐年 "
              + " ".join(f"{y} {r*100:+.1f}" for y, r in v["years"].items()))
    d = s["diff"]
    print(f"  月差（組合 − 對照）{d['mean']*100:+.2f}%/月（t {d['t']:+.2f}，NW t {d['nw_t']:+.2f}，標準差 {d['sd']*100:.2f}%，"
          f"贏 {d['yrs_won']} 年）逐年加總 " + " ".join(f"{y} {r*100:+.1f}" for y, r in d["by_year"].items()))
    print(f"  組合 vs H136 Sharpe 差 JK z {s['jk_blend_vs_h136']:+.2f}；H136–M2 相關 {s['corr']:.2f}、殘差相關 {s['resid_corr']:.2f}、"
          f"beta H136 {s['beta']['H136']:.2f} / M2 {s['beta']['M2']:.2f}、共同持股 {s['overlap']:.1f}/20、事後波動比 {s['vol_ratio']:.2f}")


def main():
    if "--unseal" not in sys.argv:
        raise PermissionError("holdout 封存中：使用者同意後加 --unseal")
    t0 = time.time()
    P = xs.load_panel(END, fundamentals=True)
    for add in (xs.add_chips, xs.add_fin, xs.add_v16, xs.add_v17):
        add(P)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dates, R, E, F = FC.prep(P, "2011-01-01", END)
        Y = ML.target(R, E)
        print(f"載入＋特徵 {time.time()-t0:.0f}s；{len(dates)} 個月（至 {dates[-1].date()}）", flush=True)
        pred, info = ML.walk(F, Y, E, dates, YEARS)
        for r in info:
            print(f"  {r['year']} 模型：訓練 {r['train_months']} 個月 / {r['train_rows']:,} 列（最後一窗結束 {r['last_train_end']}）")
        oos = np.array([d.year in YEARS for d in dates])
        Fb = F.copy()
        Fb[:, ~oos] = np.nan
        n136, x136, t136 = FC.bt(Fb, R, E, tuple(FC.FEATS.index(f) for f in ("H2", "H87", "H97")), ret_top=True)
        nm2, _, tm2 = FC.bt(pred["M2"][None], R, E, (0,), ret_top=True)
    out = {}
    for seg, (a, b) in SEGS.items():
        k = (dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b)) & np.isfinite(n136) & np.isfinite(nm2)
        N = pd.DataFrame({"H136": n136[k], "M2": nm2[k], "池": (n136 - x136)[k]}, index=dates[k])
        out[seg] = segment(N, t136, tm2, np.flatnonzero(k))
        show(seg, out[seg])
    v = verdict(out["holdout"]["diff"])
    print(f"\n判定（holdout 月差）：{v}")
    FC._log(dict(kind="blend_holdout", period="holdout", strategy="H137 H136+M2 各半 vs H136×0.69+現金", note="第三十六批 事前登記",
                 W=W, cash=CASH, segments=out, verdict=v))


if __name__ == "__main__":
    main()
