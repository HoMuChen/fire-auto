"""H33/H34：H2/H29 加絕對門檻（未達門檻的格子留現金）。日線淨值，只跑 dev/val。"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402
import hypotheses as Hy  # noqa: E402
import xs  # noqa: E402

N = 20


def run(P, rank_score, ok, p0, p1):
    """rank_score: 排序分數；ok: 是否達門檻（bool DataFrame，None=不設門檻）。回傳日線淨值與平均持股數。"""
    O, Cf = P["open"], P["close"].ffill()
    elig = P["liq"].shift(1).fillna(False).astype(bool) & (P["nbars"].shift(1) >= 250) & O.notna()
    ss = rank_score.shift(1)
    okk = ok.shift(1).fillna(False).astype(bool) if ok is not None else None
    rd = xs.rebalance_dates(O.index, pd.Timestamp(p0), pd.Timestamp(p1))
    v, vals, prev, ns = 1.0, [], set(), []
    for d, d2 in zip(rd[:-1], rd[1:]):
        s = ss.loc[d][elig.loc[d]].dropna()
        top = list(s.sort_values(ascending=False, kind="stable").head(N).index)   # 先排名取前 20
        if okk is not None:
            top = [x for x in top if okk.at[d, x]]                         # 再用門檻剔除
        held = set(top)
        traded = len(held - prev) + len(prev - held)
        v *= 1 - traded / N / 2 * xs.COST                                   # 買+賣各半邊（COST 為來回）
        seg = Cf.loc[d:d2].iloc[:-1]
        if top:
            path = v * ((N - len(top)) / N + seg[top].div(O.loc[d, top]).sum(axis=1) / N)
        else:
            path = pd.Series(v, index=seg.index)
        vals.append(path)
        v = path.iloc[-1]
        prev = held
        ns.append(len(top))
    s = pd.concat(vals)
    return s[~s.index.duplicated()], np.mean(ns)


def metr(s):
    r = s.pct_change().dropna()
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    return dict(CAGR=(s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1, MDD=(s / s.cummax() - 1).min(), Sharpe=r.mean() / r.std() * np.sqrt(252))


if __name__ == "__main__":
    period = sys.argv[1] if len(sys.argv) > 1 else "dev"
    if period == "holdout":
        raise PermissionError("holdout 已用於基本版；變體不在此跑")
    p0, p1 = H.PERIODS[period]
    p0 = max(p0, "2011-01-01")
    P = xs.load_panel(p1, fundamentals=True)
    h2, rh = Hy.h2_hi52(P), P["rev3_high"]
    h29 = Hy.h29_double_high(P)
    cases = [("H2 原版", h2, None), ("H33 H2≥0.90", h2, h2 >= 0.90), ("H33b H2≥0.95", h2, h2 >= 0.95),
             ("H29 原版", h29, None), ("H34 H29 兩項≥0.90", h29, (h2 >= 0.90) & (rh >= 0.90)),
             ("H34b H29 兩項≥0.95", h29, (h2 >= 0.95) & (rh >= 0.95))]
    rows = []
    for name, sc, ok in cases:
        s, n = run(P, sc, ok, p0, p1)
        m = metr(s)
        by = s.resample("YE").last().pct_change()
        by.iloc[0] = s.resample("YE").last().iloc[0] / s.iloc[0] - 1
        rows.append(dict(策略=name, CAGR=m["CAGR"] * 100, MDD=m["MDD"] * 100, Sharpe=m["Sharpe"], 平均持股=n,
                         **{str(k.year): v * 100 for k, v in by.items()}))
        with open(H.LOG, "a") as f:
            f.write(json.dumps(dict(ts=time.strftime("%Y-%m-%d %H:%M:%S"), kind="threshold", period=period, strategy=name,
                                    note="H2/H29 絕對門檻變體（holdout 已開封，僅 dev/val）", avg_hold=n, **m), ensure_ascii=False, default=float) + "\n")
    pd.set_option("display.width", 250)
    print(f"[{period}]"); print(pd.DataFrame(rows).round(2).to_string(index=False))
