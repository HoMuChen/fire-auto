"""第七十六批 H189：台指期 ICT「掃前日高低 → 結構轉變 MSS → FVG 回測限價進場」。規則見 HYPOTHESES.md 第七十六批（事前登記）。
資料：data/v2/futures/tx1m（tx1m.py）。

用法：.venv/bin/python research/v2/ict_lab.py dev|val|leak
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402
import tx1m  # noqa: E402
from intraday_lab import FEE_MTX, FEE_TMF, SLIP, TAX, tstat  # noqa: E402

NB = 300                                             # 08:45～13:44
SEG = {"A": ("2011-01-01", "2014-12-31"), "B": ("2015-01-01", "2019-12-31"), "dev": ("2011-01-01", "2019-12-31"),
       "val": ("2020-01-01", "2023-12-31")}
NAME = "第七十六批 H189 台指期 ICT 掃流動性＋MSS＋FVG"
RES = H.D / "results"
BASE = dict(n=2, mss_win=60, fill_win=30, last_in=270, entry="top", tgt=2.0, mode="ict", tf=1)
NSEED = 20


# ───────── 資料 ─────────
def build(period):
    p0, p1 = SEG[period]
    B = tx1m.load(str(pd.Timestamp(p0) - pd.Timedelta(days=12))[:10], p1)
    B = B[B.volume > 0].copy()
    B["minute"] = pd.to_datetime(B.minute.astype(str).str[:19])
    B["k"] = (B.minute.dt.hour * 60 + B.minute.dt.minute - (8 * 60 + 45)).astype(int)
    B = B[(B.k >= 0) & (B.k < NB)]
    by = {d: g for d, g in B.groupby("day", sort=True)}
    days = sorted(by)
    out, skip = [], 0
    for i, d in enumerate(days):
        if d < p0 or i == 0:
            continue
        g = by[d]
        vol = g.groupby("contract").volume.sum().sort_values(ascending=False)
        late = g.groupby("contract").k.max()
        ok = [c for c in vol.index if late[c] >= 295]
        c = ok[0] if ok else vol.index[0]
        pv = by[days[i - 1]]
        pv = pv[pv.contract == c]
        x = g[g.contract == c].set_index("k").sort_index()
        if len(pv) < 100 or len(x) < 250:
            skip += 1
            continue
        x = x.reindex(range(NB))
        C = x.close.ffill().bfill().values
        miss = x.close.isna().values
        O = np.where(miss, C, x.open.values)
        Hh = np.where(miss, C, x.high.values)
        L = np.where(miss, C, x.low.values)
        out.append(dict(day=pd.Timestamp(d), O=O.astype(float), H=Hh.astype(float), L=L.astype(float), C=C.astype(float),
                        pdh=float(pv.high.max()), pdl=float(pv.low.min())))
    print(f"建資料 {period}：{len(out)} 天（跳過 {skip} 天：前一天沒有同合約或資料不足）", flush=True)
    return out


def load(period):
    f = RES / f"ict_{period}.pkl"
    if not f.exists():
        with open(f, "wb") as fh:
            pickle.dump(build(period), fh)
    with open(f, "rb") as fh:
        return pickle.load(fh)


def agg5(d):
    k = NB // 5
    r = lambda a: a.reshape(k, 5)  # noqa: E731
    return dict(d, O=r(d["O"])[:, 0], H=r(d["H"]).max(1), L=r(d["L"]).min(1), C=r(d["C"])[:, -1])


# ───────── 規則（做多；做空用鏡像陣列） ─────────
def manage(O, Hh, L, C, f, fill, stop, tgt):
    """持倉管理：成交那根只檢查停損；之後同一根同時碰到停損與目標算停損。"""
    N = len(C)
    if L[f] <= stop:
        return min(O[f], stop), "停損"
    for b in range(f + 1, N):
        if L[b] <= stop:
            return min(O[b], stop), "停損"
        if Hh[b] >= tgt:
            return max(O[b], tgt), "目標"
    return C[N - 1], "收盤"


def setup(O, Hh, L, C, pdl, pdh, p):
    """一天一個方向最多一筆；回傳 dict 或 None。"""
    N, n, last_in = len(C), p["n"], p["last_in"]
    below = np.nonzero(L < pdl)[0]
    if not len(below) or below[0] >= last_in - 1:
        return None
    s = int(below[0])
    sl, slb = L[s], s
    R = p["tgt"]
    if p["mode"] == "c1":                               # 掃了就收回
        for t in range(s, last_in - 1):
            if L[t] < sl:
                sl, slb = L[t], t
            if C[t] > pdl:
                e = t + 1
                en, stop = O[e], sl - 1
                if en <= stop:
                    return None
                tgt = en + R * (en - stop)
                px, how = manage(O, Hh, L, C, e, en, stop, tgt)
                return dict(e=e, en=en, px=px, risk=en - stop, how=how)
        return None
    sh = np.zeros(N, bool)
    for j in range(n, N - n):
        sh[j] = Hh[j] > Hh[j - n:j].max() and Hh[j] > Hh[j + 1:j + n + 1].max()
    shi = np.nonzero(sh)[0]
    for t in range(s, N):
        if t + 1 >= last_in:
            return None
        if L[t] < sl:
            sl, slb = L[t], t
        if t - slb > p["mss_win"]:
            continue
        cand = shi[(shi < slb) & (shi <= t - n)]
        if not len(cand) or C[t] <= Hh[cand[-1]]:
            continue
        m = t                                           # 結構轉變
        stop = sl - 1
        if p["mode"] == "c3":
            e = m + 1
            en = O[e]
            if en <= stop:
                return None
            tgt = pdh if R == "pdh" else en + R * (en - stop)
            px, how = manage(O, Hh, L, C, e, en, stop, max(tgt, en + 1))
            return dict(e=e, en=en, px=px, risk=en - stop, how=how)
        fv = None
        for i in range(m, slb + 1, -1):
            if L[i] > Hh[i - 2]:
                fv = (Hh[i - 2], L[i])
                break
        if fv is None:
            return None
        lim = fv[1] if p["entry"] == "top" else (fv[0] + fv[1]) / 2
        tgt_lim = pdh if R == "pdh" else lim + R * (lim - stop)
        for b in range(m + 1, min(m + 1 + p["fill_win"], last_in)):
            if L[b] < lim:
                en = min(O[b], lim)
                if en <= stop:
                    en = O[b]
                tgt = pdh if R == "pdh" else en + R * (en - stop)
                px, how = manage(O, Hh, L, C, b, en, stop, max(tgt, en + 1))
                return dict(e=b, en=en, px=px, risk=en - stop, how=how, mss=m, fvg=fv[1] - fv[0])
            if Hh[b] >= tgt_lim:
                return None
        return None
    return None


def day_trades(d, side, p):
    O, Hh, L, C, pdl, pdh = d["O"], d["H"], d["L"], d["C"], d["pdl"], d["pdh"]
    if side < 0:
        O, Hh, L, C, pdl, pdh = -O, -L, -Hh, -C, -pdh, -pdl
    r = setup(O, Hh, L, C, pdl, pdh, p)
    if r is None:
        return None
    r["pts"] = r["px"] - r["en"]
    r["en"], r["px"] = abs(r["en"]), abs(r["px"])
    return r


def cost(en, px, fee):
    return 2 * (SLIP + fee) + TAX * (en + px)


def run(days, sides=(1, -1), **kw):
    p = dict(BASE, **kw)
    if p["tf"] == 5:
        p.update(mss_win=p["mss_win"] // 5, fill_win=p["fill_win"] // 5, last_in=p["last_in"] // 5)
    T = len(days)
    g, ct, cm = np.zeros(T), np.zeros(T), np.zeros(T)
    rows = []
    for i, d in enumerate(days):
        dd = agg5(d) if p["tf"] == 5 else d
        for s in sides:
            r = day_trades(dd, s, p)
            if r is None:
                continue
            g[i] += r["pts"] / r["en"]
            ct[i] += cost(r["en"], r["px"], FEE_TMF) / r["en"]
            cm[i] += cost(r["en"], r["px"], FEE_MTX) / r["en"]
            rows.append(dict(i=i, day=d["day"], side=s, e=r["e"] * p["tf"], pts=r["pts"], risk=r["risk"], how=r["how"],
                             cost_t=cost(r["en"], r["px"], FEE_TMF)))
    return g, ct, cm, pd.DataFrame(rows)


def random_ctrl(days, tr, R=2.0):
    """每筆 ICT 交易：同一天隨機一根（30～269）同方向市價進場，同停損距離、同 R 目標；20 種子平均的每日毛、成本。"""
    T = len(days)
    g, ct = np.zeros(T), np.zeros(T)
    rng = np.random.default_rng(0)
    for _, x in tr.iterrows():
        d = days[int(x.i)]
        O, Hh, L, C = d["O"], d["H"], d["L"], d["C"]
        if x.side < 0:
            O, Hh, L, C = -O, -L, -Hh, -C
        for e in rng.integers(30, 270, NSEED):
            en = O[e]
            stop = en - x.risk
            px, _ = manage(O, Hh, L, C, e, en, stop, en + R * x.risk)
            g[int(x.i)] += (px - en) / abs(en) / NSEED
            ct[int(x.i)] += cost(abs(en), abs(px), FEE_TMF) / abs(en) / NSEED
    return g, ct


def stats(days, g, ct, cm, tr):
    dd = pd.DatetimeIndex([d["day"] for d in days])
    net = g - ct
    msk = {s: np.asarray((dd >= SEG[s][0]) & (dd <= SEG[s][1])) for s in SEG}
    by = pd.Series(net, index=dd).groupby(dd.year).sum()
    n = len(tr)
    hw = tr.how.value_counts(normalize=True) if n else pd.Series(dtype=float)
    return dict(
        n=n, per_day=n / len(days), win=float((tr.pts - tr.cost_t > 0).mean()) if n else np.nan,
        gross_pts=float(tr.pts.mean()) if n else np.nan, net_pts=float((tr.pts - tr.cost_t).mean()) if n else np.nan,
        risk=float(tr.risk.mean()) if n else np.nan,
        how={k: round(float(hw.get(k, 0)), 3) for k in ("停損", "目標", "收盤")},
        t=tstat(net), t_gross=tstat(g), t_mtx=tstat(g - cm),
        A=float(net[msk["A"]].mean() * 1e4) if msk["A"].any() else np.nan,
        B=float(net[msk["B"]].mean() * 1e4) if msk["B"].any() else np.nan,
        yrs=f"{int((by > 0).sum())}/{len(by)}", years={int(y): round(v * 100, 2) for y, v in by.items()})


def line(lab, s):
    h = s["how"]
    return (f"  {lab:<22} 筆 {s['n']:>5}（{s['per_day']:.2f}/日） 勝率 {s['win']*100:4.0f}%  每筆毛 {s['gross_pts']:+6.1f} 點"
            f" 淨 {s['net_pts']:+6.1f}  風險 {s['risk']:5.1f}  t 淨 {s['t']:+.2f}（毛 {s['t_gross']:+.2f}、小台 {s['t_mtx']:+.2f}）"
            f"  A/B {s['A']:+.2f}/{s['B']:+.2f} bp/日  正年 {s['yrs']}  停損/目標/收盤 {h['停損']:.0%}/{h['目標']:.0%}/{h['收盤']:.0%}")


# ───────── 防前視 ─────────
def leak(n_days=120):
    days = load("dev")[:n_days]
    rng = np.random.default_rng(1)
    bad = tot = 0
    for kw in (dict(), dict(tf=5), dict(entry="ce"), dict(mode="c3"), dict(mode="c1")):
        base = run(days, **kw)[3]
        for k in (100, 150, 200, 250):
            cut = []
            for d in days:
                x = dict(d)
                for a in ("O", "H", "L", "C"):
                    x[a] = d[a].copy()
                walk = d["C"][k - 1] + np.cumsum(rng.normal(0, 8, NB - k)).round()
                x["O"][k:], x["C"][k:] = walk, walk
                x["H"][k:], x["L"][k:] = walk + 5, walk - 5
                cut.append(x)
            tr = run(cut, **kw)[3]
            a = base[base.e < k][["i", "side", "e"]].reset_index(drop=True)
            b = tr[tr.e < k][["i", "side", "e"]].reset_index(drop=True)
            tot += len(a)
            if not a.equals(b):
                bad += 1
                print(f"  ✗ {kw} k={k}：原 {len(a)} 筆、截斷後 {len(b)} 筆")
    print(f"防前視：{'✓ 全部一致' if not bad else f'✗ {bad} 組不一致'}（比對 {tot} 筆進場）")


def main(period):
    t0 = time.time()
    days = load(period)
    print(f"{NAME}｜{period}：{len(days)} 天，載入 {time.time()-t0:.0f}s")
    R = {}
    print("\n== 主規則（1 分 K、n=2、FVG 上緣限價、2R）")
    for lab, sides in (("做多", (1,)), ("做空", (-1,)), ("合併", (1, -1))):
        g, ct, cm, tr = run(days, sides)
        R[lab] = (g, ct, stats(days, g, ct, cm, tr), tr)
        print(line(lab, R[lab][2]))
    tr = R["合併"][3]
    gr, cr = random_ctrl(days, tr)
    diff = (R["合併"][0] - R["合併"][1]) - (gr - cr)
    t_diff = tstat(diff)
    print(f"\n== 隨機對照（同天、同方向、同風險、2R，{NSEED} 種子）：每日淨 t {tstat(gr - cr):+.2f}（毛 {tstat(gr):+.2f}）"
          f"｜ICT − 隨機 每日淨差 t {t_diff:+.2f}")
    print("\n== 對照（合併）")
    C = {}
    for lab, kw in (("C1 只掃不等結構", dict(mode="c1")), ("C3 不等 FVG", dict(mode="c3"))):
        g, ct, cm, t2 = run(days, **kw)
        C[lab] = stats(days, g, ct, cm, t2)
        print(line(lab, C[lab]))
    print("\n== 穩健性（合併）")
    rob = {}
    for lab, kw in (("n=3", dict(n=3)), ("5 分 K", dict(tf=5)), ("進場 FVG 中點", dict(entry="ce")), ("目標 3R", dict(tgt=3.0)),
                    ("目標 PDH/PDL", dict(tgt="pdh")), ("MSS 窗口 30", dict(mss_win=30))):
        g, ct, cm, t2 = run(days, **kw)
        rob[lab] = stats(days, g, ct, cm, t2)
        print(line(lab, rob[lab]))
    s = R["合併"][2]
    ok = dict(t=s["t"] >= 2, AB=s["A"] > 0 and s["B"] > 0, yrs=int(s["yrs"].split("/")[0]) >= 7, n=s["n"] >= 200,
              rnd=t_diff >= 1)
    if period == "val":
        ok = dict(t=s["t"] >= 2)
    verdict = "✓" if all(ok.values()) else "✗"
    print(f"\n判定（合併）：{verdict}  " + "  ".join(f"{k} {'✓' if v else '✗'}" for k, v in ok.items()))
    print("逐年（合併淨 %）：" + "  ".join(f"{y} {v:+.1f}" for y, v in s["years"].items()))
    with open(H.LOG, "a") as f:
        f.write(json.dumps(dict(ts=time.strftime("%Y-%m-%d %H:%M:%S"), kind="intraday", period=period, strategy=NAME,
                                note="第七十六批 事前登記", main={k: v[2] for k, v in R.items()}, ctrl=C,
                                rnd=dict(t_net=tstat(gr - cr), t_gross=tstat(gr), t_diff=t_diff), robust=rob,
                                verdict=verdict, checks=ok), ensure_ascii=False,
                           default=lambda v: round(float(v), 5) if isinstance(v, (float, np.floating)) else str(v)) + "\n")
    tr.to_csv(RES / f"ict_trades_{period}.csv", index=False)


if __name__ == "__main__":
    if sys.argv[1] == "leak":
        leak()
    else:
        main(sys.argv[1])
