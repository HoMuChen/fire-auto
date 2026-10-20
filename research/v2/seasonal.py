"""H18 春節行情、H19 金融股 11 月 — 舊季節性策略在 dev（2010-2019，對它們是真樣本外）重驗。

H18：封關前第 5 個交易日開盤買 pit 流動股等權 → 開紅盤後第 10 個交易日開盤賣（含成本）
      封關/紅盤由交易日曆推（1/10~2/28 間最大的休市缺口）
      對照：同一年、同持有長度、所有起始日的平均報酬 → 超額
H19：八大金控 11 月首交易日開盤買 → 12 月首交易日開盤賣；對照：同 8 檔同年其他 11 個月的平均月報酬
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402
import xs  # noqa: E402

COST = xs.COST
FIN = ["2882", "2881", "2886", "2884", "2891", "2883", "2885", "5880"]


def cny(P, years):
    O = P["open"]
    idx = O.index
    elig = P["liq"].shift(1).fillna(False).astype(bool) & (P["nbars"].shift(1) >= 250)
    rows = []
    for y in years:
        d = idx[(idx >= f"{y}-01-10") & (idx <= f"{y}-02-28")]
        k = np.argmax(np.diff(d.values).astype("timedelta64[D]").astype(int))
        close_day, open_day = d[k], d[k + 1]
        i0, i1 = idx.get_loc(close_day) - 5, idx.get_loc(open_day) + 10
        L = i1 - i0
        e = elig.iloc[i0]
        r = (O.iloc[i1][e] / O.iloc[i0][e] - 1).clip(-0.95, 10).mean() - COST
        yr = idx[(idx.year == y)]
        base = []
        for j in range(idx.get_loc(yr[0]), idx.get_loc(yr[-1]) - L):
            ee = elig.iloc[j]
            base.append((O.iloc[j + L][ee] / O.iloc[j][ee] - 1).clip(-0.95, 10).mean() - COST)
        rows.append(dict(year=y, 封關=close_day.date(), 紅盤=open_day.date(), 持有日=L, ret=r, base=np.mean(base)))
    return pd.DataFrame(rows)


def fin_nov(P, years):
    O = P["open"][[s for s in FIN if s in P["open"]]]
    idx = O.index
    first = pd.Series(idx, index=idx).groupby(idx.to_period("M")).first()
    rows = []
    for y in years:
        mr = {}
        for m in range(1, 13):
            a, b = pd.Period(f"{y}-{m:02d}", "M"), pd.Period(f"{y}-{m:02d}", "M") + 1
            if a in first.index and b in first.index:
                mr[m] = (O.loc[first[b]] / O.loc[first[a]] - 1).mean() - COST
        rows.append(dict(year=y, nov=mr.get(11), other=np.mean([v for k, v in mr.items() if k != 11])))
    return pd.DataFrame(rows)


def report(df, a, b, name, period):
    ex = df[a] - df[b]
    t = ex.mean() / ex.std() * np.sqrt(len(ex))
    st = dict(mean=df[a].mean(), base=df[b].mean(), ex=ex.mean(), t=t, yrs_pos=f"{int((ex > 0).sum())}/{len(ex)}",
              win=f"{int((df[a] > 0).sum())}/{len(df)}")
    print(f"\n{name}"); print(df.round(4).to_string(index=False))
    print({k: (round(v * 100, 2) if k in ("mean", "base", "ex") else (round(v, 2) if k == "t" else v)) for k, v in st.items()})
    with open(H.LOG, "a") as f:
        f.write(json.dumps(dict(ts=time.strftime("%Y-%m-%d %H:%M:%S"), kind="seasonal", period=period, strategy=name,
                                note="舊季節策略 dev 真樣本外重驗", **st), ensure_ascii=False, default=float) + "\n")


if __name__ == "__main__":
    period = sys.argv[1] if len(sys.argv) > 1 else "dev"
    p0, p1 = H.PERIODS[period]
    P = xs.load_panel(p1)
    years = range(max(2011, int(p0[:4])), int(p1[:4]) + 1)
    report(cny(P, years), "ret", "base", "H18 春節行情", period)
    report(fin_nov(P, years), "nov", "other", "H19 金融股11月", period)
