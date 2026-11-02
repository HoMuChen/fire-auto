"""第三十四批 H136：H87+H97+H2 等權百分位 top20，holdout 一次性檢驗（使用者 2026-11-02 同意，只測這一組）。

規則凍結＝FS2 引擎（feature_combo.prep / bt），對照 H29 用同引擎 {H2, H10}。規則見 HYPOTHESES.md 第三十四批。
用法：.venv/bin/python research/v2/holdout_h136.py --unseal
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import feature_combo as FC  # noqa: E402
import harness as H  # noqa: E402
import xs  # noqa: E402

FEATS = ["H2", "H10", "H87", "H97"]
H136, H29 = (0, 2, 3), (0, 1)


def summ(net, ex, dates):
    k = np.isfinite(net)
    n, e, d = net[k], ex[k], dates[k]
    yrs = (d[-1] - d[0]).days / 365.25 + 1 / 12
    eq = np.cumprod(1 + n)
    by = pd.Series(n, index=d).groupby(lambda x: x.year).apply(lambda x: (1 + x).prod() - 1)
    return dict(months=len(n), CAGR=eq[-1] ** (1 / yrs) - 1, ex_ann=e.mean() * 12, t=e.mean() / e.std(ddof=1) * np.sqrt(len(e)),
                Sharpe=n.mean() / n.std(ddof=1) * np.sqrt(12), MDD=(eq / np.maximum.accumulate(eq) - 1).min(),
                years={str(y): v for y, v in by.items()})


def main():
    if "--unseal" not in sys.argv:
        raise PermissionError("holdout 封存中；使用者同意後加 --unseal")
    t0 = time.time()
    p0 = H.PERIODS["holdout"][0]
    p1 = min(pd.Timestamp(H.PERIODS["holdout"][1]), pd.Timestamp.today()).strftime("%Y-%m-%d")
    P = xs.load_panel(p1, fundamentals=True)
    for add in (xs.add_chips, xs.add_fin, xs.add_v16, xs.add_v17):
        add(P)
    dates, R, E, F = FC.prep(P, p0, p1, FEATS)
    print(f"載入 {time.time()-t0:.0f}s；資料最後一天 {P['close'].index[-1].date()}；換股窗 {dates[0].date()} ~ {dates[-1].date()}（{len(dates)} 個）")
    na, ea, ta = FC.bt(F, R, E, H136, ret_top=True)
    nb, eb, tb = FC.bt(F, R, E, H29, ret_top=True)
    a, b = summ(na, ea, dates), summ(nb, eb, dates)
    k = np.isfinite(na) & np.isfinite(nb)
    d = (na - nb)[k]
    dm, dt = d.mean(), d.mean() / d.std(ddof=1) * np.sqrt(len(d))
    byd = pd.Series(d, index=dates[k]).groupby(lambda x: x.year).sum()
    rho = np.corrcoef(na[k], nb[k])[0, 1]
    common = np.mean([len(x & y) for x, y in zip(ta, tb) if x and y])
    print(f"\n== H136 holdout（top20，扣成本；月報酬算）")
    for lab, s in (("H136 H87+H97+H2", a), ("H29（同引擎）", b)):
        print(f"  {lab:<16} CAGR {s['CAGR']*100:+6.1f}%  年化超額 {s['ex_ann']*100:+6.1f}%  t {s['t']:.2f}  Sharpe {s['Sharpe']:.2f}"
              f"  MDD {s['MDD']*100:6.1f}%  逐年 " + "  ".join(f"{y} {v*100:+.1f}%" for y, v in s["years"].items()))
    v_rep = "✓ 可考慮取代 H29" if dm > 0 and dt >= 2 else ("方向對、不足以取代" if dm > 0 else "✗ 不如 H29")
    v_alone = "✓" if a["t"] >= 2 else "✗"
    print(f"\n  H136 − H29 月差 {dm*100:+.3f}%（年化 {dm*1200:+.2f}%）t {dt:+.2f}  贏的年 {int((byd > 0).sum())}/{len(byd)}  → 取代判定：{v_rep}")
    print(f"  單獨有效（超額 t ≥ 2）：{v_alone}")
    print(f"  與 H29 月報酬相關 {rho:.2f}；平均共同持股 {common:.1f} / 20 檔")
    FC._log(dict(kind="feature_combo", period="holdout", strategy="H136 H87+H97+H2", note="第三十四批 holdout 一次性（使用者同意）",
                 data_end=str(P["close"].index[-1].date()), h136=a, h29=b, diff_m=dm, t=dt, yrs_won=f"{int((byd > 0).sum())}/{len(byd)}",
                 verdict_replace=v_rep, verdict_alone=v_alone, rho=rho, common=common))


if __name__ == "__main__":
    main()
