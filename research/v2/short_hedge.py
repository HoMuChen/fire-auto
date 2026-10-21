"""H40/H41：H2 搭配期貨做空（避險 / 多空）。日線淨值，只跑 dev/val。"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402
import hypotheses as Hy  # noqa: E402
import threshold as TH  # noqa: E402
import xs  # noqa: E402

N, FUT_COST = 20, 0.0005


def leg_daily(P, picks_by_rd, rd, sign=1, cost=xs.COST):
    """等權持有每期名單的日線淨值（多方 sign=1；空方 sign=-1 → 報酬取負，每期重設名目）。"""
    O, Cf = P["open"], P["close"].ffill()
    v, vals, prev = 1.0, [], set()
    for d, d2 in zip(rd[:-1], rd[1:]):
        names = picks_by_rd[d]
        traded = len(set(names) - prev) + len(prev - set(names))
        v *= 1 - traded / max(len(names), 1) / 2 * cost
        seg = Cf.loc[d:d2].iloc[:-1][names].div(O.loc[d, names]).mean(axis=1)
        path = v * (1 + sign * (seg - 1))
        vals.append(path)
        v = path.iloc[-1]
        prev = set(names)
    s = pd.concat(vals)
    return s[~s.index.duplicated()]


def combine(parts, cash_w):
    """每月換股日重新配權：parts = [(日線淨值序列, 權重)]；cash_w 為現金權重。日報酬加權。"""
    r = sum(w * s.pct_change().fillna(0) for s, w in parts)
    return (1 + r).cumprod()


if __name__ == "__main__":
    period = sys.argv[1] if len(sys.argv) > 1 else "dev"
    if period == "holdout":
        raise PermissionError("holdout 不在此跑")
    p0, p1 = H.PERIODS[period]
    p0 = max(p0, "2011-01-01")
    P = xs.load_panel(p1)
    O = P["open"]
    rd = xs.rebalance_dates(O.index, pd.Timestamp(p0), pd.Timestamp(p1))
    elig = P["liq"].shift(1).fillna(False).astype(bool) & (P["nbars"].shift(1) >= 250) & O.notna()
    sc = Hy.h2_hi52(P).shift(1)
    big = P["money"].rolling(60, min_periods=40).mean().shift(1)
    longs, shorts = {}, {}
    for d in rd:
        s = sc.loc[d][elig.loc[d]].dropna()
        longs[d] = list(s.sort_values(ascending=False, kind="stable").head(N).index)
        top150 = big.loc[d][s.index].nlargest(150).index
        shorts[d] = list(s[top150].sort_values(kind="stable").head(N).index)
    L = leg_daily(P, longs, rd)
    S = leg_daily(P, shorts, rd, sign=-1, cost=0.001)
    tr = pd.read_parquet(H.D / "taiex_tr.parquet").assign(date=lambda x: pd.to_datetime(x.date)).set_index("date").price
    tr = tr.reindex(L.index).ffill()
    Fr = -tr.pct_change().fillna(0) - FUT_COST / 21          # 空指數日報酬（成本按月攤）
    Fs = (1 + Fr).cumprod()
    cases = {"對照 H2 100%": combine([(L, 1.0)], 0),
             "H40 H2 80% + 空台指 80%": combine([(L, 0.8), (Fs, 0.8)], 0.2),
             "H41 多 H2 50% + 空弱勢權值 50%": combine([(L, 0.5), (S, 0.5)], 0.5),
             "（參考）加權報酬指數": tr / tr.iloc[0],
             "（參考）空弱勢權值股單腿": S}
    rows = []
    for k, s in cases.items():
        m = TH.metr(s)
        by = s.resample("YE").last()
        by = pd.concat([pd.Series([s.iloc[0]], index=[s.index[0]]), by]).pct_change().dropna()
        rows.append(dict(策略=k, CAGR=m["CAGR"] * 100, MDD=m["MDD"] * 100, Sharpe=m["Sharpe"],
                         **{str(i.year): v * 100 for i, v in by.items()}))
        with open(H.LOG, "a") as f:
            f.write(json.dumps(dict(ts=time.strftime("%Y-%m-%d %H:%M:%S"), kind="short", period=period, strategy=k,
                                    note="期貨做空登記假設", **m), ensure_ascii=False, default=float) + "\n")
    print(f"[{period}] {L.index[0].date()}~{L.index[-1].date()}  long/short 與 long 日報酬相關:",
          round(L.pct_change().corr(S.pct_change()), 2), " H2 對加權指數相關:", round(L.pct_change().corr(tr.pct_change()), 2))
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).round(2).to_string(index=False))
