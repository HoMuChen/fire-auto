"""第七十五批 H188：台指期「賣壓被吸收」反轉（足跡圖 Delta，tick rule 推估）。規則見 HYPOTHESES.md 第七十五批（事前登記）。
資料：data/v2/futures/tx1md（tx1md.py）。

用法：.venv/bin/python research/v2/absorb_lab.py dev|val
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402
import tx1md  # noqa: E402
from intraday_lab import FEE_MTX, FEE_TMF, SLIP, TAX, tstat  # noqa: E402

NB, BASE, LOOK, LAST_IN = 300, 30, 3, 270          # 08:45～13:44；前 30 根基準；確認 3 根；13:15（第 270 根）以後不進場
SEG = {"A": ("2011-01-01", "2014-12-31"), "B": ("2015-01-01", "2019-12-31"), "dev": ("2011-01-01", "2019-12-31"),
       "val": ("2020-01-01", "2023-12-31")}
NAME = "第七十五批 H188 台指期賣壓吸收反轉"


def days_arrays(p0, p1):
    df = tx1md.load(p0, p1)
    out = []
    for day, g in df.groupby("day", sort=True):
        g = g.set_index("minute").sort_index()
        idx = pd.date_range(f"{day} 08:45", periods=NB, freq="min")
        g = g.reindex(idx)
        if g.close.notna().sum() < 250:
            continue
        c = g.close.ffill().bfill()
        o = g.open.fillna(c.shift(1)).fillna(c)
        out.append(dict(day=pd.Timestamp(day), O=o.values, H=g.high.fillna(c).values, L=g.low.fillna(c).values, C=c.values,
                        V=g.volume.fillna(0).values, D=(g.buy - g.sell).fillna(0).values))
    return out


def trades(d, side, mode="delta", k=3.0, hold=15):
    """一天的交易：(進場根, 出場價, 進場價, 確認時已反彈點數)；另回傳 C2（真跌破）後 hold 分鐘毛點數。"""
    O, Hh, L, C, V, D = d["O"], d["H"], d["L"], d["C"], d["V"], d["D"]
    if side < 0:                                   # 做空＝價格、Delta 鏡像
        O, Hh, L, C, D = -O, -L, -Hh, -C, -D
    absD = np.abs(D)
    res, c2, t, free = [], [], BASE, 0
    while t < LAST_IN:
        if t < free:
            t += 1
            continue
        A, Vm = absD[t - BASE:t].mean(), V[t - BASE:t].mean()
        big = (D[t] <= -k * A) if mode == "delta" else (V[t] >= k * Vm)
        if not (A > 0 and big and L[t] <= L[:t].min()):
            t += 1
            continue
        conf = None
        for c in range(t + 1, min(t + LOOK + 1, NB - 1)):
            if L[t + 1:c + 1].min() < L[t]:
                e = c + 1
                if e < NB:
                    x = min(e + hold, NB - 1) if hold != "close" else NB - 1
                    c2.append(((C[NB - 1] if (hold == "close" or e + hold >= NB) else O[x]) - O[e]))
                break
            ok = (D[c] >= k * A) if mode == "delta" else (V[c] >= k * Vm)
            if ok and C[c] > O[c]:
                conf = c
                break
        if conf is None or conf + 1 >= LAST_IN:
            t += 1
            continue
        e = conf + 1
        if hold == "close" or e + hold >= NB:
            x, px = NB - 1, C[NB - 1]
        else:
            x, px = e + hold, O[e + hold]
        res.append((e, side * px, side * O[e], C[conf] - L[t]))
        free, t = x, x
    return res, c2


def run(days, sides=(1, -1), **kw):
    """每日毛、成本（微台、小台）與交易明細。"""
    T = len(days)
    g, ct, cm = np.zeros(T), np.zeros(T), np.zeros(T)
    rows, c2 = [], []
    for i, d in enumerate(days):
        for s in sides:
            tr, cc = trades(d, s, **kw)
            c2 += cc
            for e, px, en, reb in tr:
                pe, pxa = abs(en), abs(px)
                pts = s * (pxa - pe)
                g[i] += pts / pe
                ct[i] += (2 * (SLIP + FEE_TMF) + TAX * (pe + pxa)) / pe
                cm[i] += (2 * (SLIP + FEE_MTX) + TAX * (pe + pxa)) / pe
                rows.append(dict(day=d["day"], side=s, bar=e, pts=pts, reb=reb, cost_t=2 * (SLIP + FEE_TMF) + TAX * (pe + pxa)))
    return g, ct, cm, pd.DataFrame(rows), np.array(c2)


def stats(days, g, ct, cm, tr):
    dd = pd.DatetimeIndex([d["day"] for d in days])
    net = g - ct
    msk = {s: np.asarray((dd >= SEG[s][0]) & (dd <= SEG[s][1])) for s in SEG}
    by = pd.Series(net, index=dd).groupby(dd.year).sum()
    n = len(tr)
    return dict(
        n=n, per_day=n / len(days), win=float((tr.pts - tr.cost_t > 0).mean()) if n else np.nan,
        gross_pts=float(tr.pts.mean()) if n else np.nan, net_pts=float((tr.pts - tr.cost_t).mean()) if n else np.nan,
        reb=float(tr.reb.mean()) if n else np.nan,
        t=tstat(net), t_gross=tstat(g), t_mtx=tstat(g - cm),
        A=float(net[msk["A"]].mean() * 1e4) if msk["A"].any() else np.nan,
        B=float(net[msk["B"]].mean() * 1e4) if msk["B"].any() else np.nan,
        yrs=f"{int((by > 0).sum())}/{len(by)}", years={int(y): round(v * 100, 2) for y, v in by.items()})


def line(lab, s):
    return (f"  {lab:<26} 筆 {s['n']:>5}（{s['per_day']:.2f}/日） 勝率 {s['win']*100:4.0f}%  每筆毛 {s['gross_pts']:+6.1f} 點"
            f" 淨 {s['net_pts']:+6.1f}  t 淨 {s['t']:+.2f}（毛 {s['t_gross']:+.2f}、小台 {s['t_mtx']:+.2f}）"
            f"  A/B {s['A']:+.2f}/{s['B']:+.2f} bp/日  正年 {s['yrs']}  確認時已反彈 {s['reb']:.1f} 點")


def main(period):
    t0 = time.time()
    days = days_arrays(*SEG[period])
    print(f"{NAME}｜{period}：{len(days)} 天，載入 {time.time()-t0:.0f}s")
    R = {}
    print("\n== 主規則（k=3、持有 15 分）")
    for lab, sides in (("做多", (1,)), ("做空", (-1,)), ("合併", (1, -1))):
        g, ct, cm, tr, c2 = run(days, sides)
        R[lab] = (g, ct, stats(days, g, ct, cm, tr), c2, tr)
        print(line(lab, R[lab][2]))
    print("\n== C1 只用價量（同型態，Delta 條件換成量）")
    C1 = {}
    for lab, sides in (("做多", (1,)), ("做空", (-1,)), ("合併", (1, -1))):
        g, ct, cm, tr, _ = run(days, sides, mode="vol")
        C1[lab] = (g, ct, stats(days, g, ct, cm, tr))
        print(line(lab, C1[lab][2]))
    diff = (R["合併"][0] - R["合併"][1]) - (C1["合併"][0] - C1["合併"][1])
    t_diff = tstat(diff)
    print(f"\n  Delta − C1（合併，每日淨損益差）t {t_diff:+.2f}")
    c2l, c2s = R["做多"][3], R["做空"][3]
    print(f"\n== C2 真跌破／真突破後 15 分（毛，點；正＝和原本反轉方向一致）：做多側 {len(c2l)} 筆 平均 {c2l.mean():+.1f}"
          f"｜做空側 {len(c2s)} 筆 平均 {c2s.mean():+.1f}")
    rnd = np.concatenate([d["O"][BASE + 1 + 15:LAST_IN + 15] - d["O"][BASE + 1:LAST_IN] for d in days])
    print(f"== 隨機進場持有 15 分：平均毛 {rnd.mean():+.2f} 點（多方）")
    print("\n== 穩健性（合併）")
    rob = {}
    for lab, kw in (("k=2", dict(k=2.0)), ("k=4", dict(k=4.0)), ("持有 5 分", dict(hold=5)), ("持有 30 分", dict(hold=30)),
                    ("持有到收盤", dict(hold="close"))):
        g, ct, cm, tr, _ = run(days, (1, -1), **kw)
        rob[lab] = stats(days, g, ct, cm, tr)
        print(line(lab, rob[lab]))
    s = R["合併"][2]
    ok = dict(t=s["t"] >= 2, AB=s["A"] > 0 and s["B"] > 0, yrs=int(s["yrs"].split("/")[0]) >= 7, n=s["n"] >= 200,
              c1=s["net_pts"] > C1["合併"][2]["net_pts"] and t_diff >= 1)
    if period == "val":
        ok = dict(t=s["t"] >= 2)
    verdict = "✓" if all(ok.values()) else "✗"
    print(f"\n判定（合併）：{verdict}  " + "  ".join(f"{k} {'✓' if v else '✗'}" for k, v in ok.items()))
    print("逐年（合併淨 %）：" + "  ".join(f"{y} {v:+.1f}" for y, v in s["years"].items()))
    with open(H.LOG, "a") as f:
        f.write(json.dumps(dict(ts=time.strftime("%Y-%m-%d %H:%M:%S"), kind="intraday", period=period, strategy=NAME,
                                note="第七十五批 事前登記", main={k: v[2] for k, v in R.items()},
                                c1={k: v[2] for k, v in C1.items()}, t_diff=t_diff, c2=[float(c2l.mean()), float(c2s.mean())],
                                rnd=float(rnd.mean()), robust=rob, verdict=verdict, checks=ok), ensure_ascii=False,
                           default=lambda v: round(float(v), 5) if isinstance(v, (float, np.floating)) else str(v)) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
