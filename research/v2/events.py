"""事件型策略（H31/H32）：營收突破事件 → 次一可交易日開盤買、持有固定交易日數。"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402
import xs  # noqa: E402
from hypotheses import h2_hi52  # noqa: E402

HOLD = 60


def monthly_rev(sid):
    f = H.D / "revenue" / f"{sid}.parquet"
    if not f.exists():
        return None
    d = pd.read_parquet(f)
    if not len(d):
        return None
    d["ym"] = pd.PeriodIndex.from_fields(year=d.revenue_year.astype(int), month=d.revenue_month.astype(int), freq="M")
    d = d.drop_duplicates("ym", keep="last").set_index("ym").revenue.astype(float)
    d = d.reindex(pd.period_range(d.index.min(), d.index.max(), freq="M"))
    r3 = d.rolling(3).sum()
    out = pd.DataFrame({"yoy": d / d.shift(12) - 1, "r3h": r3 / r3.rolling(12, min_periods=12).max()})
    out["avail"] = (out.index + 1).to_timestamp() + pd.Timedelta(days=10)
    return out


def events(P, p0, p1, price_confirm=False):
    O, Cf = P["open"], P["close"].ffill()
    idx = O.index
    elig = P["liq"].shift(1).fillna(False).astype(bool) & (P["nbars"].shift(1) >= 250)
    h2 = h2_hi52(P).shift(1) if price_confirm else None
    mkt = (1 + (O / O.shift() - 1).where(P["liq"].shift().fillna(False).astype(bool)).clip(-0.5, 0.5).mean(axis=1).fillna(0)).cumprod()
    rows = []
    for s in O.columns:
        m = monthly_rev(s)
        if m is None:
            continue
        trig = (m.yoy >= 0.30) & (m.r3h >= 0.999)
        prev6 = trig.shift(1).rolling(6, min_periods=1).max().fillna(0).astype(bool)
        ev = m[trig & ~prev6]
        for ym, r in ev.iterrows():
            k = idx.searchsorted(r.avail)
            if k >= len(idx) or not (pd.Timestamp(p0) <= idx[k] <= pd.Timestamp(p1)):
                continue
            d0 = idx[k]
            if not elig.at[d0, s] or np.isnan(O.at[d0, s]):
                continue
            if price_confirm and not (h2.at[d0, s] >= 0.9):
                continue
            k1 = min(k + HOLD, len(idx) - 1)
            px1 = O.iat[k1, O.columns.get_loc(s)]
            if np.isnan(px1):
                px1 = Cf.iat[k1 - 1, O.columns.get_loc(s)]
            ret = px1 * (1 - H.SELL_FEE - H.SELL_TAX) / (O.at[d0, s] * (1 + H.BUY_FEE)) - 1
            rows.append(dict(sid=s, ym=str(ym), buy=d0, sell=idx[k1], ret=ret, mret=mkt.iloc[k1] / mkt.loc[d0] - 1))
    T = pd.DataFrame(rows)
    T["ex"] = T.ret - T.mret
    return T


def summarize(T, name, period):
    m = T.groupby(T.buy.dt.to_period("M")).ex.mean()
    yr = T.groupby(T.buy.dt.year).ex.mean()
    nm = T.groupby(T.buy.dt.to_period("M")).size()
    st = dict(n=len(T), per_month=nm.reindex(pd.period_range(T.buy.min(), T.buy.max(), freq="M"), fill_value=0).mean(),
              avg=T.ret.mean(), ex=T.ex.mean(), t=m.mean() / m.std() * np.sqrt(len(m)), win=(T.ret > 0).mean(),
              yrs_pos=f"{int((yr > 0).sum())}/{len(yr)}")
    with open(H.LOG, "a") as f:
        f.write(json.dumps(dict(ts=time.strftime("%Y-%m-%d %H:%M:%S"), kind="event", period=period, strategy=name,
                                note="事件型登記假設", **st), ensure_ascii=False, default=float) + "\n")
    print(name, {k: (round(v * 100, 2) if k in ("avg", "ex", "win") else (round(v, 2) if isinstance(v, float) else v)) for k, v in st.items()})
    print("  逐年每筆超額%:", (yr * 100).round(2).to_dict())


if __name__ == "__main__":
    period = sys.argv[1] if len(sys.argv) > 1 else "dev"
    p0, p1 = H.PERIODS[period]
    if period == "holdout":
        raise PermissionError("holdout 封存中")
    P = xs.load_panel(p1)
    p0 = max(p0, "2011-01-01")
    summarize(events(P, p0, p1), "H31 營收突破事件", period)
    summarize(events(P, p0, p1, price_confirm=True), "H32 營收突破+股價確認", period)
