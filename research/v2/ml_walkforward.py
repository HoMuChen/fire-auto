"""第三十五批 ML1：走時間前進，讓資料決定 FS1 20 群代表特徵的權重。規則見 HYPOTHESES.md 第三十五批（事前登記）。

輸入＝feature_combo.prep 的 pit 池內百分位（20 個）；目標＝下個換股窗報酬在 pit 池內的百分位。
每年第一個換股日前重新訓練，只用「下個換股日早於當年第一個換股日」的月份（前一年 12 月那窗未實現 → 空一個月）。
  M1 Fama-MacBeth：每月橫斷面 OLS（缺值補 0.5、含截距），訓練期斜率平均
  M2 HistGradientBoosting：max_depth 3、lr 0.05、300 輪、min_samples_leaf 500、l2 1.0、無 early stopping、原生缺值
投組＝預測最高 20 檔等權月換股（FS2 引擎 bt），對照 H136（實盤）與 H29，同引擎、同樣只算樣本外月份。

用法：
  .venv/bin/python research/v2/ml_walkforward.py stage1                         # 2015~2023（主判定）
  .venv/bin/python research/v2/ml_walkforward.py stage2 --unseal --models M1     # holdout，需第一階段過關＋使用者同意
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import feature_combo as FC  # noqa: E402
import harness as H  # noqa: E402
import xs  # noqa: E402

FEATS = FC.FEATS
STAGES = {"stage1": ("2023-12-31", range(2015, 2024)), "stage2": ("2026-10-20", range(2024, 2027))}
HGB = dict(max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=500, l2_regularization=1.0,
           early_stopping=False, random_state=0)
MIN_ROWS = 100


def target(R, E):
    Y = np.full(R.shape, np.nan)
    for t in range(R.shape[0]):
        m = E[t]
        if m.sum() >= MIN_ROWS:
            Y[t, m] = pd.Series(R[t, m]).rank(pct=True).values
    return Y


def fm_betas(F, Y, E):
    """每月橫斷面 OLS 斜率（缺值補 0.5、含截距）；樣本不足的月為 NaN。"""
    T = Y.shape[0]
    B = np.full((T, F.shape[0] + 1), np.nan)
    for t in range(T):
        m = E[t] & np.isfinite(Y[t])
        if m.sum() < MIN_ROWS:
            continue
        X = np.nan_to_num(F[:, t, m].T, nan=0.5)
        X1 = np.column_stack([np.ones(len(X)), X])
        B[t], *_ = np.linalg.lstsq(X1, Y[t, m], rcond=None)
    return B


def walk(F, Y, E, dates, years):
    T, N = Y.shape
    d2 = list(dates[1:]) + [pd.Timestamp.max]          # 各月換股窗的結束日（最後一月不會被拿來訓練）
    B = fm_betas(F, Y, E)
    pred = {"M1": np.full((T, N), np.nan), "M2": np.full((T, N), np.nan)}
    info = []
    for y in years:
        ty = [t for t in range(T) if dates[t].year == y]
        if not ty:
            continue
        first = dates[ty[0]]
        tr = [t for t in range(T) if d2[t] < first and np.isfinite(Y[t]).any()]
        assert tr and max(d2[t] for t in tr) < first, "訓練資料含未實現的報酬"
        bbar = np.nanmean(B[tr], axis=0)
        rows = [(t, np.flatnonzero(E[t] & np.isfinite(Y[t]))) for t in tr]
        Xtr = np.vstack([F[:, t, ix].T for t, ix in rows])
        ytr = np.concatenate([Y[t, ix] for t, ix in rows])
        mdl = HistGradientBoostingRegressor(**HGB).fit(Xtr, ytr)
        for t in ty:
            ix = np.flatnonzero(E[t])
            X = F[:, t, ix].T
            pred["M1"][t, ix] = np.column_stack([np.ones(len(ix)), np.nan_to_num(X, nan=0.5)]) @ bbar
            pred["M2"][t, ix] = mdl.predict(X)
        info.append(dict(year=y, train_months=len(tr), train_rows=len(ytr), first_train=str(dates[tr[0]].date()),
                         last_train_end=str(max(d2[t] for t in tr).date()), bbar=bbar))
    return pred, info


def turnover(tops):
    tt = [x for x in tops if x]
    return np.mean([1 - len(a & b) / FC.TOPN for a, b in zip(tt[:-1], tt[1:])]) if len(tt) > 1 else np.nan


def summ(net, ex, dates, oos, tops):
    k = oos & np.isfinite(net)
    n, e, d = net[k], ex[k], dates[k]
    eq = np.cumprod(1 + n)
    yrs = len(n) / 12
    by = pd.Series(n, index=d).groupby(lambda x: x.year).apply(lambda x: (1 + x).prod() - 1)
    return dict(months=len(n), CAGR=eq[-1] ** (1 / yrs) - 1, ex_ann=e.mean() * 12, t=e.mean() / e.std(ddof=1) * np.sqrt(len(e)),
                Sharpe=n.mean() / n.std(ddof=1) * np.sqrt(12), MDD=(eq / np.maximum.accumulate(eq) - 1).min(), turn=turnover(tops),
                years={str(y): v for y, v in by.items()})


def diff(a, b, dates, oos):
    k = oos & np.isfinite(a) & np.isfinite(b)
    d = (a - b)[k]
    by = pd.Series(d, index=dates[k]).groupby(lambda x: x.year).sum()
    return dict(diff_m=d.mean(), t=d.mean() / d.std(ddof=1) * np.sqrt(len(d)), yrs_won=f"{int((by > 0).sum())}/{len(by)}", n=len(d))


def ic(pred, R, E, oos):
    v = []
    for t in np.flatnonzero(oos):
        m = E[t] & np.isfinite(pred[t])
        if m.sum() >= MIN_ROWS:
            v.append(pd.Series(pred[t, m]).corr(pd.Series(R[t, m]), method="spearman"))
    v = np.array(v)
    return v.mean(), v.mean() / v.std(ddof=1) * np.sqrt(len(v))


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "stage1"
    if stage not in STAGES:
        raise SystemExit("stage1 | stage2")
    if stage == "stage2" and "--unseal" not in sys.argv:
        raise PermissionError("holdout 封存中：第一階段過關且使用者同意後加 --unseal --models ...")
    models = sys.argv[sys.argv.index("--models") + 1].split(",") if "--models" in sys.argv else ["M1", "M2"]
    end, years = STAGES[stage]
    t0 = time.time()
    P = xs.load_panel(end, fundamentals=True)
    for add in (xs.add_chips, xs.add_fin, xs.add_v16, xs.add_v17):
        add(P)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dates, R, E, F = FC.prep(P, "2011-01-01", end)
        Y = target(R, E)
        print(f"載入＋特徵 {time.time()-t0:.0f}s；{len(dates)} 個月、{len(FEATS)} 個特徵", flush=True)
        pred, info = walk(F, Y, E, dates, years)
        oos = np.array([d.year in years for d in dates])
        print(f"訓練完成 {time.time()-t0:.0f}s；樣本外 {oos.sum()} 個月（{dates[oos][0].date()} ~ {dates[oos][-1].date()}）")
        for r in info:
            print(f"  {r['year']} 模型：訓練 {r['train_months']} 個月 / {r['train_rows']:,} 列（{r['first_train']} 起，最後一窗結束 {r['last_train_end']}）")
        Fb = F.copy()
        Fb[:, ~oos] = np.nan
        res = {}
        for k, sub in (("H136", tuple(FEATS.index(f) for f in ("H2", "H87", "H97"))), ("H29", (FEATS.index("H2"), FEATS.index("H10")))):
            net, ex, tops = FC.bt(Fb, R, E, sub, ret_top=True)
            res[k] = (net, ex, tops)
        for k in models:
            net, ex, tops = FC.bt(pred[k][None], R, E, (0,), ret_top=True)
            res[k] = (net, ex, tops)
    S = {k: summ(*v[:2], dates, oos, v[2]) for k, v in res.items()}
    print(f"\n== {stage}（樣本外，top20 扣成本；月報酬算）")
    for k, s in S.items():
        extra = ""
        if k in models:
            icm, ict = ic(pred[k], R, E, oos)
            s["IC"], s["IC_t"] = icm, ict
            extra = f"  IC {icm:+.3f}（t {ict:+.1f}）"
        print(f"  {k:<5} CAGR {s['CAGR']*100:+6.1f}%  年化超額 {s['ex_ann']*100:+6.1f}%  t {s['t']:.2f}  Sharpe {s['Sharpe']:.2f}"
              f"  MDD {s['MDD']*100:6.1f}%  換手 {s['turn']*100:3.0f}%{extra}")
    print("  逐年（%）：")
    for k, s in S.items():
        print(f"    {k:<5} " + "  ".join(f"{y} {v*100:+6.1f}" for y, v in s["years"].items()))
    print("\n== 對照（月差）")
    out = {}
    for k in models:
        a, b = diff(res[k][0], res["H136"][0], dates, oos), diff(res[k][0], res["H29"][0], dates, oos)
        if stage == "stage1":
            v = "過第一階段（holdout 需使用者同意）" if a["diff_m"] > 0 and a["t"] >= 2 else ("方向對、不足" if a["diff_m"] > 0 else "✗")
        else:
            v = "✓ holdout 確認" if a["diff_m"] > 0 else "✗ holdout 未確認"
        out[k] = dict(vs_h136=a, vs_h29=b, verdict=v)
        print(f"  {k}：對 H136 {a['diff_m']*100:+.2f}%/月（t {a['t']:+.2f}，贏 {a['yrs_won']} 年）；"
              f"對 H29 {b['diff_m']*100:+.2f}%/月（t {b['t']:+.2f}，贏 {b['yrs_won']} 年）→ {v}")
    last = info[-1]["bbar"][1:]
    order = np.argsort(-np.abs(last))
    print(f"\n== M1 {info[-1]['year']} 模型的平均斜率（描述；特徵百分位 0→1 時目標百分位變動）")
    print("  " + "  ".join(f"{FEATS[i]} {last[i]:+.3f}" for i in order))
    FC._log(dict(kind="ml_walkforward", period=stage, strategy="ML1 " + ",".join(models), note="第三十五批 事前登記",
                 years=f"{years.start}-{years.stop - 1}", stats=S, compare=out,
                 m1_bbar_last={FEATS[i]: last[i] for i in range(len(FEATS))}))


if __name__ == "__main__":
    main()
