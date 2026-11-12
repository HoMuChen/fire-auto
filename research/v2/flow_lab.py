"""第七十七批 H190：訂單流「泡泡」當過濾（∞FLOW 的用法）。規則見 HYPOTHESES.md 第七十七批（事前登記）。
基礎訊號＝第七十六批 ict_lab.py 的 S1 ICT 主規則、S2 掃了就收回（C1）；Delta＝tx1md 的 tick rule 每分鐘主動買 − 主動賣。

用法：.venv/bin/python research/v2/flow_lab.py dev|val
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402
import ict_lab as I  # noqa: E402
import tx1md  # noqa: E402
from intraday_lab import FEE_MTX, FEE_TMF, tstat  # noqa: E402

NAME = "第七十七批 H190 訂單流泡泡過濾（ICT／掃了就收回）"
BASE_N = 30
FILTERS = ("F1 確認", "F2 沒有反向泡泡", "F3 掃蕩後累積")
NPERM = 1000


def delta_arrays(days, period):
    p0, p1 = I.SEG[period]
    T = tx1md.load(p0, p1)
    T["minute"] = pd.to_datetime(T.minute.astype(str).str[:19])
    T["k"] = (T.minute.dt.hour * 60 + T.minute.dt.minute - (8 * 60 + 45)).astype(int)
    by = {d: g for d, g in T.groupby("day")}
    out = []
    for d in days:
        g = by.get(d["day"].strftime("%Y-%m-%d"))
        if g is None or not len(g) or str(g.contract.iloc[0]) != d["contract"]:
            out.append(None)
            continue
        D = np.zeros(I.NB)
        g = g[(g.k >= 0) & (g.k < I.NB)]
        D[g.k.values] = (g.buy - g.sell).values
        out.append(D)
    return out


def flags(X, side, e, anchor, W, k):
    """回傳 (F1, F2, F3)；只用 e 之前的 K 棒。X＝Delta 或 K 棒實體。"""
    x = side * X
    ax = np.abs(X)
    lo = max(0, e - W)
    same = opp = False
    for t in range(lo, e):
        if t == 0:
            continue
        A = ax[max(0, t - BASE_N):t].mean()
        if A <= 0:
            continue
        same |= x[t] >= k * A
        opp |= x[t] <= -k * A
    f1 = x[lo:e].sum() > 0 and same
    f2 = not opp
    f3 = x[anchor:e].sum() > 0 if e > anchor else False
    return f1, f2, f3


def mark(tr, days, Ds, W, k):
    rows = []
    for _, r in tr.iterrows():
        D = Ds[int(r.i)]
        d = days[int(r.i)]
        if D is None:
            rows.append((np.nan,) * 6)
            continue
        fd = flags(D, r.side, int(r.e), int(r.anchor), W, k)
        fb = flags(d["C"] - d["O"], r.side, int(r.e), int(r.anchor), W, k)
        rows.append(fd + fb)
    m = pd.DataFrame(rows, columns=[f"D{j}" for j in range(3)] + [f"B{j}" for j in range(3)], index=tr.index)
    return m


def welch(a, b):
    if len(a) < 5 or len(b) < 5:
        return np.nan
    return float((a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))


def daily(days, tr):
    T = len(days)
    g, ct, cm = np.zeros(T), np.zeros(T), np.zeros(T)
    for _, r in tr.iterrows():
        i = int(r.i)
        g[i] += r.pts / r.en
        ct[i] += r.cost_t / r.en
        cm[i] += (r.cost_t - 2 * (FEE_TMF - FEE_MTX)) / r.en
    return g, ct, cm


def evaluate(days, tr, m, j, rng):
    ok = tr[m[f"D{j}"].notna()].copy()
    keep = m.loc[ok.index, f"D{j}"].astype(bool)
    kb = m.loc[ok.index, f"B{j}"].astype(bool)
    K, S = ok[keep], ok[~keep]
    seg = lambda x, s: x[(x.day >= I.SEG[s][0]) & (x.day <= I.SEG[s][1])].pts.mean()  # noqa: E731
    perm = np.array([ok.pts.values[rng.permutation(len(ok))[:len(K)]].mean() for _ in range(NPERM)])
    g, ct, cm = daily(days, K)
    st = I.stats(days, g, ct, cm, K)
    return dict(n_all=len(ok), keep=len(K) / max(len(ok), 1), kept_pts=float(K.pts.mean()), skip_pts=float(S.pts.mean()),
                t_ks=welch(K.pts, S.pts), A_kept=seg(K, "A"), A_skip=seg(S, "A"), B_kept=seg(K, "B"), B_skip=seg(S, "B"),
                p_rand=float((perm >= K.pts.mean()).mean()), kline_pts=float(ok[kb].pts.mean()),
                kline_keep=float(kb.mean()), skip_net=float((S.pts - S.cost_t).mean()), trade=st)


def show(lab, r):
    s = r["trade"]
    return (f"  {lab:<22} 留 {r['keep']:.0%}（{s['n']} 筆）｜每筆毛 留 {r['kept_pts']:+5.1f}／跳 {r['skip_pts']:+5.1f}"
            f"（t {r['t_ks']:+.2f}、隨機 p {r['p_rand']:.2f}）A {r['A_kept']:+.1f}/{r['A_skip']:+.1f} B {r['B_kept']:+.1f}/{r['B_skip']:+.1f}"
            f"｜K 線版 留 {r['kline_keep']:.0%} 每筆 {r['kline_pts']:+5.1f}｜留下淨 {s['net_pts']:+.1f} 點 t {s['t']:+.2f}"
            f"（小台 {s['t_mtx']:+.2f}）正年 {s['yrs']}")


def main(period):
    t0 = time.time()
    days = I.load(period)
    Ds = delta_arrays(days, period)
    nod = sum(D is None for D in Ds)
    print(f"{NAME}｜{period}：{len(days)} 天，Delta 合約不同或缺 {nod} 天（{time.time()-t0:.0f}s）")
    bases = {"S1 ICT": I.run(days)[3], "S2 掃了就收回": I.run(days, mode="c1")[3]}
    rng = np.random.default_rng(0)
    R, ROB, ok_all = {}, {}, {}
    for bn, tr in bases.items():
        print(f"\n== {bn}：{len(tr)} 筆，每筆毛 {tr.pts.mean():+.1f} 點")
        m = mark(tr, days, Ds, 5, 2.0)
        print(f"  無 Delta 資料的交易 {int(m.D0.isna().sum())} 筆（不列入）")
        for j, fn in enumerate(FILTERS):
            r = evaluate(days, tr, m, j, rng)
            R[f"{bn}｜{fn}"] = r
            print(show(fn, r))
            s = r["trade"]
            info = (r["t_ks"] >= 2 and r["A_kept"] > r["A_skip"] and r["B_kept"] > r["B_skip"]
                    and r["kept_pts"] > r["kline_pts"])
            trade = s["t"] >= 2 and s["A"] > 0 and s["B"] > 0 and int(s["yrs"].split("/")[0]) >= 7 and s["n"] >= 200
            ok_all[f"{bn}｜{fn}"] = dict(info=bool(info), trade=bool(trade))
        print("  穩健性：")
        for lab, (W, k) in (("W=3", (3, 2.0)), ("W=10", (10, 2.0)), ("k=3", (5, 3.0))):
            m2 = mark(tr, days, Ds, W, k)
            for j, fn in enumerate(FILTERS):
                r = evaluate(days, tr, m2, j, rng)
                ROB[f"{bn}｜{fn}｜{lab}"] = r
                print(f"    {lab:<5} {fn:<10} 留 {r['keep']:.0%} 每筆毛 留 {r['kept_pts']:+5.1f}／跳 {r['skip_pts']:+5.1f}"
                      f"（t {r['t_ks']:+.2f}）K 線版 {r['kline_pts']:+5.1f}｜留下淨 t {r['trade']['t']:+.2f}")
    print("\n判定（info＝過濾有資訊、trade＝留下的能交易）：")
    for k_, v in ok_all.items():
        print(f"  {k_:<26} info {'✓' if v['info'] else '✗'}  trade {'✓' if v['trade'] else '✗'}")
    verdict = "✓" if any(v["trade"] and v["info"] for v in ok_all.values()) else "✗"
    print(f"整體：{verdict}")
    with open(H.LOG, "a") as f:
        f.write(json.dumps(dict(ts=time.strftime("%Y-%m-%d %H:%M:%S"), kind="intraday", period=period, strategy=NAME,
                                note="第七十七批 事前登記", main=R, robust=ROB, checks=ok_all, verdict=verdict),
                           ensure_ascii=False,
                           default=lambda v: round(float(v), 5) if isinstance(v, (float, np.floating)) else str(v)) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
