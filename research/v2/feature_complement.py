"""描述（不是策略判定）：特徵庫每個特徵和現有策略（H142 50 檔留任 3N、H136、H143）的互補性。只用 dev 2011–19。

每個特徵依特徵庫方向，算每月「好端 10% 超額」與「多空（前 20% − 後 20%）」，
和三個策略的月超額算相關；另算 H142 最差 1/4 月份裡該特徵好端的平均超額（避險價值）。
輸出 data/v2/results/feature_complement.csv。

用法：.venv/bin/python research/v2/feature_complement.py
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
import feature_library as FL  # noqa: E402
import harness as H  # noqa: E402
import strategy_lab as SL  # noqa: E402

P1 = B48.P1


def tstat(x):
    x = x[np.isfinite(x)]
    return x.mean() / x.std(ddof=1) * np.sqrt(len(x)) if len(x) > 2 else np.nan


def main():
    t0 = time.time()
    comps = B49.filt_comps()
    lab, dates, R, E, mats, lim, codes = B49.load(H.PERIODS["dev"][1])
    m = (dates >= pd.Timestamp(P1[0])) & (dates <= pd.Timestamp(P1[1]))
    L = pd.read_csv(H.D / "results" / "feature_library.csv").set_index("H")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        S, filt = B49.s142(mats, E, comps)
        strat = {"H142": SL.portfolio(S, R, E, 50, buffer=3, filt=filt, limits=lim).ex.values,
                 "H136": B49.run136(mats, R, E, lim, 20).ex.values,
                 "H143": B48.run2(mats, R, E, None, ("all", 0.2, False))[0].ex.values}
        for k, v in strat.items():
            print(f"{k} dev 年化超額 {np.nanmean(v[m])*12*100:+.1f}%", flush=True)
        c = pd.DataFrame({k: v[m] for k, v in strat.items()}).corr()
        print("策略之間的月超額相關：\n" + c.round(2).to_string(), flush=True)
        worst = m & (strat["H142"] <= np.nanpercentile(strat["H142"][m], 25))
        rows = []
        for f in FL.FEATS:
            X = lab.src.build(f, dates, E)
            sgn = float(L.loc[f, "sign"]) if f in L.index else 1.0
            mo = FL.monthly(sgn * X, R, E, full=False)                    # ic, ls, good, bad
            ls, good = mo[:, 1], mo[:, 2]
            ok = m & np.isfinite(good)
            if ok.sum() < 24:
                continue
            r = dict(H=f, name=L.loc[f, "name"] if f in L.index else f, cat=L.loc[f, "cat"] if f in L.index else "",
                     status=L.loc[f, "status"] if f in L.index else "", months=int(ok.sum()),
                     good=np.nanmean(good[ok]) * 12, good_t=tstat(good[ok]), ls=np.nanmean(ls[ok]) * 12, ls_t=tstat(ls[ok]),
                     good_in_h142_worst=np.nanmean(good[worst & np.isfinite(good)]) * 12)
            for k, v in strat.items():
                for nm, x in (("good", good), ("ls", ls)):
                    q = ok & np.isfinite(v) & np.isfinite(x)
                    r[f"rho_{nm}_{k}"] = np.corrcoef(x[q], v[q])[0, 1] if q.sum() > 12 else np.nan
            rows.append(r)
    D = pd.DataFrame(rows)
    D.to_csv(H.D / "results" / "feature_complement.csv", index=False)
    print(f"完成 {len(D)} 個特徵，{time.time()-t0:.0f}s → data/v2/results/feature_complement.csv")
    SL.log(dict(kind="describe", period="dev", strategy="特徵互補性（對 H142／H136／H143）", note="描述，不是策略判定",
                n_feats=len(D), strat_corr=c.round(3).to_dict()))


if __name__ == "__main__":
    main()
