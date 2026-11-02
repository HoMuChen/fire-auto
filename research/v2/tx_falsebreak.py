"""第二十七批 H127：台指期 1 分 K「180 根區間假跌破收回」做多當沖，9 個變體。規則見 HYPOTHESES.md 第二十七批。

用法：/usr/local/bin/python3 research/v2/tx_falsebreak.py dev|val
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tx1m  # noqa: E402

PERIODS = {"dev": ("2011-01-01", "2019-12-31"), "val": ("2020-01-01", "2023-12-31"), "holdout": ("2024-01-01", "2099-12-31")}
WIN, STOP, SLIP, FEE, TAX = 180, 10, 1, 0.25, 0.00002
ENTRIES, EXITS = ("Ea", "Eb", "Ec"), ("X1", "X2", "X3")
LOG = Path(__file__).resolve().parent / "experiments.jsonl"


def days_arrays(p0, p1):
    """每天：近月合約（當天日盤量最大）的 [前一交易日同合約 K + 當天 K] 陣列與當天起點。"""
    B = tx1m.load(str(pd.Timestamp(p0) - pd.Timedelta(days=10))[:10], p1)
    B = B[B.volume > 0]
    days = sorted(B.day.unique())
    by = {d: g for d, g in B.groupby("day")}
    out = []
    for i, d in enumerate(days):
        if d < p0:
            continue
        g = by[d]
        c = g.groupby("contract").volume.sum().idxmax()
        cur = g[g.contract == c].sort_values("minute")
        prev = by[days[i - 1]] if i else g.iloc[:0]
        prev = prev[prev.contract == c].sort_values("minute")
        x = pd.concat([prev, cur])
        o, h, l, cl = (x[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        L = pd.Series(l).rolling(WIN).min().shift().to_numpy()
        H = pd.Series(h).rolling(WIN).max().shift().to_numpy()
        out.append((d, o, h, l, cl, L, H, len(prev)))
    return out


def targets(x, entry, Lb, Hb):
    mid = (Lb + Hb) / 2
    t1, t2 = {"X1": (entry + 20, mid), "X2": (mid, Hb), "X3": (entry + 50, Hb)}[x]
    return [max(t1, entry + 1), max(t2, entry + 1)]


def manage(pos, t, o, h, l, cl, last):
    """在第 t 根處理持倉；回傳 (是否結束, 結束原因)。pos["fills"] 收集每口 (原始出場價, 成交出場價)。"""
    if l[t] <= pos["stop"]:
        raw = min(o[t], pos["stop"])
        for k in pos["open"]:
            pos["fills"].append((raw, raw - SLIP))
        pos["open"] = []
        return True, "stop"
    for k in list(pos["open"]):
        tp = pos["tp"][k]
        if h[t] >= tp:
            px = max(o[t], tp)
            pos["fills"].append((px, px))
            pos["open"].remove(k)
    if not pos["open"]:
        return True, "target"
    if t == last:
        for k in pos["open"]:
            pos["fills"].append((cl[t], cl[t] - SLIP))
        pos["open"] = []
        return True, "eod"
    return False, None


def close_trade(pos, d):
    e_raw, e_fill = pos["raw"], pos["raw"] + SLIP
    gross = sum(r - e_raw for r, _ in pos["fills"])
    net = sum(f - e_fill - 2 * FEE - (e_fill + f) * TAX for _, f in pos["fills"])
    return dict(day=d, entry=e_fill, gross=gross, net=net, reason=pos["why"])


def run_day(d, o, h, l, cl, L, H, n0, ev, xv):
    last = len(o) - 1
    trades, setup, pos, pend = [], None, None, None
    for t in range(max(n0, WIN), last + 1):
        if pend is not None:
            pos = dict(raw=o[t], stop=o[t] + SLIP - STOP, tp=targets(xv, o[t] + SLIP, *pend), open=[0, 1], fills=[], Lb=pend[0], Hb=pend[1])
            pend = None
        if pos is not None:
            done, why = manage(pos, t, o, h, l, cl, last)
            if not done:
                continue
            pos["why"] = why
            trades.append(close_trade(pos, d))
            if why == "eod":
                break
            if why == "target":
                setup, pos = None, None
                continue
            setup = dict(Lb=pos["Lb"], Hb=pos["Hb"], start=t, phase="re", stop=pos["stop"])
            pos = None
        if setup is None and not np.isnan(L[t]) and l[t] < L[t]:
            setup = dict(Lb=L[t], Hb=H[t], start=t, phase="entry")
        if setup is None:
            continue
        limited = ev == "Ea" or (ev == "Ec" and setup["phase"] == "entry")
        if limited and t - setup["start"] > 10:
            setup = None
            if not np.isnan(L[t]) and l[t] < L[t]:
                setup = dict(Lb=L[t], Hb=H[t], start=t, phase="entry")
            else:
                continue
        level = setup["stop"] if (ev == "Ec" and setup["phase"] == "re") else setup["Lb"]
        if cl[t] > level and t < last:
            pend = (setup["Lb"], setup["Hb"])
    return trades


def random_day(d, o, h, l, cl, L, H, n0, xv, k, rng):
    last, out = len(o) - 1, []
    lo = max(n0, WIN)
    for _ in range(k):
        t = int(rng.integers(lo, last))
        e = o[t + 1]
        pos = dict(raw=e, stop=e + SLIP - STOP, tp=targets(xv, e + SLIP, L[t], H[t]), open=[0, 1], fills=[])
        for u in range(t + 1, last + 1):
            done, why = manage(pos, u, o, h, l, cl, last)
            if done:
                pos["why"] = why
                break
        out.append(close_trade(pos, d)["net"])
    return out


def stats(T, years):
    if not len(T):
        return dict(n=0)
    dd = T.groupby("day").net.sum()
    yr = T.groupby(T.day.str[:4]).net.sum().reindex(years, fill_value=0)
    eq = dd.cumsum()
    return dict(n=len(T), days=len(dd), win=(T.net > 0).mean() * 100, gross=T.gross.mean(), net=T.net.mean(),
                total=T.net.sum(), t_day=dd.mean() / dd.std() * np.sqrt(len(dd)), yrs_pos=f"{int((yr > 0).sum())}/{len(yr)}",
                mdd=(eq - eq.cummax()).min(), stop_pct=(T.reason == "stop").mean() * 100,
                target_pct=(T.reason == "target").mean() * 100)


if __name__ == "__main__":
    period = sys.argv[1] if len(sys.argv) > 1 else "dev"
    if period == "holdout" and "--unseal" not in sys.argv:
        raise PermissionError("holdout 封存中")
    p0, p1 = PERIODS[period]
    t0 = time.time()
    D = days_arrays(p0, p1)
    years = sorted({d[:4] for d, *_ in D})
    print(f"[{period}] {len(D)} 天 {D[0][0]}~{D[-1][0]}，載入 {time.time()-t0:.0f}s", flush=True)
    rows, allT = [], []
    for ev in ENTRIES:
        for xv in EXITS:
            T = pd.DataFrame([tr for day in D for tr in run_day(*day, ev, xv)])
            cnt = T.groupby("day").size() if len(T) else pd.Series(dtype=int)
            rng, rnd = np.random.default_rng(0), []
            for _ in range(20):
                for day in D:
                    k = int(cnt.get(day[0], 0))
                    if k:
                        rnd += random_day(*day, xv, k, rng)
            s = stats(T, years)
            s.update(variant=f"{ev}-{xv}", rand_net=float(np.mean(rnd)) if rnd else np.nan)
            s["edge"] = s.get("net", np.nan) - s["rand_net"]
            rows.append(s)
            allT.append(T.assign(variant=f"{ev}-{xv}"))
            print(f"  {ev}-{xv} {s.get('n')} 筆", flush=True)
    S = pd.DataFrame(rows).set_index("variant")
    pd.set_option("display.width", 250)
    print(S.round(2).to_string())
    A = pd.concat(allT)
    print("\n逐年淨點數（2 口合計）")
    print(A.assign(y=A.day.str[:4]).pivot_table(index="variant", columns="y", values="net", aggfunc="sum").round(0).to_string())
    A.to_parquet(tx1m.OUT.parent.parent / "results" / f"b27_{period}.parquet")
    with open(LOG, "a") as f:
        for v, r in S.iterrows():
            f.write(json.dumps(dict(ts=time.strftime("%Y-%m-%d %H:%M:%S"), period=period, note="第二十七批 H127 TX 1分K 180根假跌破",
                                    strategy=f"H127 {v}", **{k: (round(x, 4) if isinstance(x, float) else x) for k, x in r.items()}),
                               ensure_ascii=False, default=str) + "\n")
