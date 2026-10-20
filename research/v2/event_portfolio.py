"""H31/H32 有限格數投組模擬（規則見 HYPOTHESES.md「有限資金投組規則」）。只允許 dev/val。"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import events as E  # noqa: E402
import harness as H  # noqa: E402
import xs  # noqa: E402
from hypotheses import h2_hi52  # noqa: E402


def simulate(T, P, p0, p1, N, seed=None, priority=None):
    """T: events() 交易表（buy/sell/ret）。priority: Series(index=T.index) 越大越優先；None=隨機（seed）。"""
    O, Cf = P["open"], P["close"].ffill()
    days = O.index[(O.index >= p0) & (O.index <= p1)]
    rng = np.random.default_rng(seed)
    T = T.assign(pri=priority if priority is not None else rng.random(len(T)))
    by_day = {d: g.sort_values("pri", ascending=False) for d, g in T.groupby("buy")}
    cash, pos, eq, taken = 1.0, [], [], 0
    prev_eq = 1.0
    for d in days:
        keep = []
        for p in pos:
            if d >= p["sell"]:
                cash += p["alloc"] * (1 + p["ret"])
            else:
                keep.append(p)
        pos = keep
        if d in by_day:
            held = {p["sid"] for p in pos}
            for _, r in by_day[d].iterrows():
                if len(pos) >= N:
                    break
                if r.sid in held:
                    continue
                a = min(prev_eq / N, cash)
                if a <= 0:
                    break
                cash -= a
                pos.append(dict(sid=r.sid, alloc=a, ret=r.ret, sell=r.sell, buy_px=O.at[d, r.sid]))
                held.add(r.sid)
                taken += 1
        v = cash + sum(p["alloc"] * Cf.at[d, p["sid"]] / p["buy_px"] for p in pos)
        eq.append((d, v, len(pos)))
        prev_eq = v
    s = pd.DataFrame(eq, columns=["date", "nav", "n"]).set_index("date")
    return s, taken


def bench_daily(P, p0, p1):
    O = P["open"]
    r = (O / O.shift() - 1).where(P["liq"].shift().fillna(False).astype(bool)).clip(-0.5, 0.5).mean(axis=1).fillna(0)
    r = r[(r.index >= p0) & (r.index <= p1)]
    return (1 + r).cumprod()


def metr(eq, bench):
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    m = eq.resample("ME").last().pct_change().dropna()
    bm = bench.resample("ME").last().pct_change().dropna().reindex(m.index)
    ex = m - bm
    by = (1 + m).groupby(m.index.year).prod() - (1 + bm).groupby(bm.index.year).prod()
    return dict(CAGR=(eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1, MDD=(eq / eq.cummax() - 1).min(),
                t=ex.mean() / ex.std() * np.sqrt(len(ex)), yrs_pos=(by > 0).mean(),
                bench=(bench.iloc[-1] / bench.iloc[0]) ** (1 / yrs) - 1)


if __name__ == "__main__":
    period = sys.argv[1] if len(sys.argv) > 1 else "dev"
    if period == "holdout":
        raise PermissionError("holdout 封存中")
    p0, p1 = H.PERIODS[period]
    p0 = max(p0, "2011-01-01")
    P = xs.load_panel(p1)
    h2 = h2_hi52(P).shift(1)
    B = bench_daily(P, p0, p1)
    rows = []
    for name, pc in (("H31", False), ("H32", True)):
        T = E.events(P, p0, p1, price_confirm=pc)
        pri_h2 = pd.Series([h2.at[b, s] for b, s in zip(T.buy, T.sid)], index=T.index).fillna(0)
        for N in (20, 10):
            res = []
            for sd in range(50):
                eq, k = simulate(T, P, p0, p1, N, seed=sd)
                res.append(dict(**metr(eq.nav, B), util=eq.n.mean() / N, taken=k))
            R = pd.DataFrame(res)
            eq, k = simulate(T, P, p0, p1, N, priority=pri_h2)
            h = metr(eq.nav, B)
            row = dict(策略=name, N=N, 對照CAGR=R.bench.iloc[0] * 100,
                       CAGR中位=R.CAGR.median() * 100, CAGR_5pct=R.CAGR.quantile(0.05) * 100, CAGR_95pct=R.CAGR.quantile(0.95) * 100,
                       MDD中位=R.MDD.median() * 100, t中位=R.t.median(), 正年中位=R.yrs_pos.median() * 100,
                       使用率=R.util.median() * 100, 年交易=R.taken.median() / ((pd.Timestamp(p1) - pd.Timestamp(p0)).days / 365.25),
                       H2優先_CAGR=h["CAGR"] * 100, H2優先_MDD=h["MDD"] * 100, H2優先_t=h["t"])
            rows.append(row)
            with open(H.LOG, "a") as f:
                f.write(json.dumps(dict(ts=time.strftime("%Y-%m-%d %H:%M:%S"), kind="event_portfolio", period=period,
                                        note="有限格數投組（隨機50種子+H2優先）", **row), ensure_ascii=False, default=float) + "\n")
            print(name, N, "done", flush=True)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).round(2).to_string(index=False))
