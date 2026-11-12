"""第八十三批 H196：股票池事件型短打（小停利、高勝率、每天最多 1 筆）。規則見 HYPOTHESES.md 第八十三批（事前登記）。
日 K 版：開盤進場，盤中碰停利出、同日碰停利與停損算停損、到期收盤出。個股期貨近似（還原價），每邊 0.1%。
資料截到 2023-12-31（holdout 不讀）。

用法：.venv/bin/python research/v2/scalp_lab.py dev|val|leak
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
import xs  # noqa: E402

END = "2023-12-31"
SEG = {"A": ("2011-01-01", "2014-12-31"), "B": ("2015-01-01", "2019-12-31"), "dev": ("2011-01-01", "2019-12-31"),
       "val": ("2020-01-01", "2023-12-31")}
TOPN, COST, DT_COST = 100, 0.002, 0.0054
HOLDS, TPS, SLM = (0, 1, 3), (0.005, 0.01, 0.02), (None, 2, 4)
NPLAC = 100
RES = H.D / "results"
NAMES = {1: "前一天大跌／大漲 ≥ 5% 反向", 2: "開盤跳空 ≥ 2% 反向", 3: "RSI2 極端＋200 日線", 4: "連 3 天同向反向",
         5: "前一天收漲停／跌停 跟進", 6: "開盤跳空 ≥ 2% 順勢", 7: "20 日新高／新低＋量 2 倍 順勢", 8: "前一天 ≥ 3% 且收在極端反向"}


def panel():
    f = RES / "scalp_panel.pkl"
    if not f.exists():
        P = xs.load_panel(END)
        with open(f, "wb") as fh:
            pickle.dump({k: P[k] for k in ("open", "high", "low", "close", "volume", "money", "liq", "nbars")}, fh)
    with open(f, "rb") as fh:
        return pickle.load(fh)


def events(P, cut=None):
    if cut is not None:
        P = {k: v.loc[:cut] for k, v in P.items()}
    O, Hh, L, C, V = P["open"], P["high"], P["low"], P["close"], P["volume"].astype(float)
    idx = C.index
    lim = pd.Series(np.where(idx < H.LIMIT_CHANGE, 0.07, 0.10), index=idx)
    pc = C.shift(1)
    gap = O / pc - 1
    r1 = C / pc - 1
    mny = P["money"].astype(float).rolling(60, min_periods=40).mean().shift(1)
    ok = P["liq"].shift(1).fillna(False).astype(bool) & (P["nbars"].shift(1) >= H.MIN_BARS + 130)
    rk = mny.where(ok).rank(axis=1, ascending=False)
    pool = rk <= TOPN
    flat = (O == Hh) & (O == L)
    art = gap.abs().gt(lim + 0.01, axis=0)
    upl = flat & gap.ge(lim - 0.006, axis=0)
    dnl = flat & gap.le(-(lim - 0.006), axis=0)
    elig_L = pool & ~art & ~upl & O.notna()
    elig_S = pool & ~art & ~dnl & O.notna()
    dx = C.diff()
    up = dx.clip(lower=0).ewm(alpha=1 / 2, adjust=False).mean()
    dn = (-dx.clip(upper=0)).ewm(alpha=1 / 2, adjust=False).mean()
    rsi = (100 - 100 / (1 + up / dn)).shift(1)
    ma200 = C.rolling(200).mean().shift(1)
    c1 = pc
    p1 = r1.shift(1)
    r3 = (C / C.shift(3) - 1).shift(1)
    d3 = ((r1 < 0) & (r1.shift(1) < 0) & (r1.shift(2) < 0)).shift(1, fill_value=False)
    u3 = ((r1 > 0) & (r1.shift(1) > 0) & (r1.shift(2) > 0)).shift(1, fill_value=False)
    vr = (V / V.rolling(20).mean().shift(1)).shift(1)
    limp = lim.shift(1)
    hi20 = (C >= C.rolling(20).max()).shift(1, fill_value=False)
    lo20 = (C <= C.rolling(20).min()).shift(1, fill_value=False)
    pos = ((C - L) / (Hh - L)).shift(1)
    E = {}
    E[("L", 1)] = (p1 <= -0.05, -p1)
    E[("S", 1)] = (p1 >= 0.05, p1)
    E[("L", 2)] = (gap <= -0.02, -gap)
    E[("S", 2)] = (gap >= 0.02, gap)
    E[("L", 3)] = ((rsi < 5) & (c1 > ma200), 5 - rsi)
    E[("S", 3)] = ((rsi > 95) & (c1 < ma200), rsi - 95)
    E[("L", 4)] = (d3, -r3)
    E[("S", 4)] = (u3, r3)
    E[("L", 5)] = (p1.ge(limp - 0.005, axis=0), vr)
    E[("S", 5)] = (p1.le(-(limp - 0.005), axis=0), vr)
    E[("L", 6)] = (gap >= 0.02, gap)
    E[("S", 6)] = (gap <= -0.02, -gap)
    E[("L", 7)] = (hi20 & (vr >= 2), vr)
    E[("S", 7)] = (lo20 & (vr >= 2), vr)
    E[("L", 8)] = ((p1 <= -0.03) & (pos <= 0.1), -p1)
    E[("S", 8)] = ((p1 >= 0.03) & (pos >= 0.9), p1)
    out = {}
    for (sd, k), (cond, st) in E.items():
        el = elig_L if sd == "L" else elig_S
        m = (cond.fillna(False).astype(bool) & el)
        s = st.where(m)
        out[(sd, k)] = s
    return dict(ev=out, elig=dict(L=elig_L, S=elig_S), idx=idx, cols=C.columns)


def pick(s):
    """每天取強度最大的一檔：回傳 (日 index, 股 index)。"""
    v = s.values
    has = np.isfinite(v).any(axis=1)
    d = np.nonzero(has)[0]
    c = np.nanargmax(np.where(np.isfinite(v[d]), v[d], -np.inf), axis=1)
    return d, c


def windows(P, d, c, side):
    """進場日起 4 天的報酬（相對進場開盤）：Ro、Rh、Rl、Rc（n × 4）。做空時鏡像。"""
    O, Hh, L, C = (P[k].values for k in ("open", "high", "low", "close"))
    T = O.shape[0]
    E = O[d, c]
    out = []
    for A in (O, Hh, L, C):
        w = np.full((len(d), 4), np.nan)
        for k in range(4):
            j = np.minimum(d + k, T - 1)
            w[:, k] = A[j, c]
        out.append(w)
    Cw = pd.DataFrame(out[3].T).ffill().values.T
    for i in range(3):
        out[i] = np.where(np.isfinite(out[i]), out[i], Cw)
    out[3] = Cw
    Ro, Rh, Rl, Rc = (x / E[:, None] - 1 for x in out)
    if side == "S":
        Ro, Rh, Rl, Rc = -Ro, -Rl, -Rh, -Rc
    return Ro, Rh, Rl, Rc


def exits(W, h, tp, slm):
    Ro, Rh, Rl, Rc = W
    n = Ro.shape[0]
    x = np.full(n, np.nan)
    done = np.zeros(n, bool)
    sl = None if slm is None else -slm * tp
    for k in range(h + 1):
        if k > 0:
            m = ~done & (Ro[:, k] >= tp)
            x[m], done[m] = Ro[m, k], True
            if sl is not None:
                m = ~done & (Ro[:, k] <= sl)
                x[m], done[m] = Ro[m, k], True
        if sl is not None:
            m = ~done & (Rl[:, k] <= sl)
            x[m], done[m] = sl, True
        m = ~done & (Rh[:, k] >= tp)
        x[m], done[m] = tp, True
    x[~done] = Rc[~done, h]
    return x


def tstat(x, lag):
    n = len(x)
    m = x.mean()
    y = x - m
    s = (y @ y) / n
    for L in range(1, lag + 1):
        s += 2 * (1 - L / (lag + 1)) * (y[L:] @ y[:-L]) / n
    return float(m / np.sqrt(s / n)) if s > 0 else np.nan


def stats(idx, d, net, h, seg_mask):
    T = len(idx)
    ser = np.zeros(T)
    np.add.at(ser, d, net)
    ser = ser[seg_mask]
    di = idx[seg_mask]
    by = pd.Series(ser, index=di).groupby(di.year).sum()
    A = ser[np.asarray((di >= SEG["A"][0]) & (di <= SEG["A"][1]))]
    B = ser[np.asarray((di >= SEG["B"][0]) & (di <= SEG["B"][1]))]
    w, l_ = net[net > 0], net[net <= 0]
    return dict(n=len(net), per_day=len(net) / len(di), win=float((net > 0).mean()), mean=float(net.mean()),
                payoff=float(w.mean() / -l_.mean()) if len(w) and len(l_) and l_.mean() < 0 else np.nan,
                t=tstat(ser, h + 1), A=float(A.mean() * 1e4) if len(A) else np.nan, B=float(B.mean() * 1e4) if len(B) else np.nan,
                yrs=f"{int((by > 0).sum())}/{len(by)}", years={int(y): round(v * 100, 1) for y, v in by.items()})


def specs():
    return [(h, tp, sl) for h in HOLDS for tp in TPS for sl in SLM]


def spec_name(s):
    h, tp, sl = s
    return f"抱≤{h}天 停利{tp*100:.1f}% 停損{'無' if sl is None else f'{sl*tp*100:.0f}%'}"


def run(period):
    t0 = time.time()
    P = panel()
    X = events(P)
    idx = X["idx"]
    T = len(idx)
    seg = np.asarray((idx >= SEG[period][0]) & (idx <= SEG[period][1]))
    valid_end = np.arange(T) + max(HOLDS) < T
    print(f"載入 {len(X['cols'])} 檔 × {T} 天（{time.time()-t0:.0f}s）", flush=True)
    rng = np.random.default_rng(0)
    elig = {sd: X["elig"][sd].values for sd in ("L", "S")}
    rows, plac_best = [], np.full(NPLAC, -9.0)
    base = {}
    for (sd, k), s in X["ev"].items():
        d, c = pick(s)
        keep = seg[d] & valid_end[d]
        d, c = d[keep], c[keep]
        W = windows(P, d, c, sd)
        # 安慰劑：同一天換成池內隨機一檔
        el = elig[sd]
        PW = []
        for _ in range(NPLAC):
            cc = np.array([rng.choice(np.flatnonzero(el[i])) for i in d])
            PW.append(windows(P, d, cc, sd))
        for sp in specs():
            g = exits(W, *sp)
            net = g - COST
            st = stats(idx, d, net, sp[0], seg)
            st.update(side=sd, ev=k, spec=sp, gross=float(g.mean()))
            if sd == "L" and sp[0] == 0:
                st["dt_mean"] = float((g - DT_COST).mean())
            pts, pwin = [], []
            for p, pw in enumerate(PW):
                gp = exits(pw, *sp) - COST
                tp_ = stats(idx, d, gp, sp[0], seg)["t"]
                plac_best[p] = max(plac_best[p], tp_)
                pts.append(gp.mean())
                pwin.append((gp > 0).mean())
            st.update(rand_mean=float(np.mean(pts)), rand_win=float(np.mean(pwin)))
            rows.append(st)
        print(f"  {sd}{k} {NAMES[k]}：{len(d)} 筆（{time.time()-t0:.0f}s）", flush=True)
    return rows, plac_best, idx


def show(r):
    return (f"{'多' if r['side'] == 'L' else '空'}{r['ev']} {NAMES[r['ev']]:<22} {spec_name(r['spec']):<22} 筆 {r['n']:5d}（{r['per_day']:.2f}/日）"
            f" 勝率 {r['win']*100:4.0f}%（隨機 {r['rand_win']*100:3.0f}%） 每筆淨 {r['mean']*100:+.3f}%（毛 {r['gross']*100:+.3f}%、"
            f"隨機淨 {r['rand_mean']*100:+.3f}%） 盈虧比 {r['payoff']:.2f} t {r['t']:+.2f} A/B {r['A']:+.1f}/{r['B']:+.1f}bp 正年 {r['yrs']}")


def main_dev():
    rows, pb, idx = run("dev")
    thr = max(2.0, float(np.quantile(pb, 0.95)))
    print(f"\n安慰劑（{NPLAC} 次，同日隨機換股）每次 432 個中最高 t：中位 {np.median(pb):.2f}、95 分位 {np.quantile(pb, .95):.2f} → 門檻 {thr:.2f}")
    R = pd.DataFrame(rows)
    print(f"\n== 全部 {len(R)} 個：每筆淨 > 0 的 {int((R['mean'] > 0).sum())} 個；勝率 ≥ 60% 的 {int((R.win >= 0.6).sum())} 個，其中每筆淨 > 0 的 {int(((R.win >= 0.6) & (R['mean'] > 0)).sum())} 個")
    print(f"   勝率 ≥ 60% 的項目：隨機換股時勝率中位 {R[R.win >= 0.6].rand_win.median()*100:.0f}%（＝出場設計本身給的勝率）")
    print("\n== dev t 最高 15 個")
    for _, r in R.sort_values("t", ascending=False).head(15).iterrows():
        print("  " + show(r))
    print("\n== 勝率 ≥ 60% 中 t 最高 10 個")
    for _, r in R[R.win >= 0.6].sort_values("t", ascending=False).head(10).iterrows():
        print("  " + show(r))
    print("\n== 各事件：最好的出場（t）與毛報酬（停利 1%、抱 ≤ 1 天、停損 2%）")
    for (sd, k), g in R.groupby(["side", "ev"]):
        b = g.sort_values("t").iloc[-1]
        ref = g[g.spec.apply(lambda s: s == (1, 0.01, 2))].iloc[0]
        print(f"  {'多' if sd == 'L' else '空'}{k} {NAMES[k]:<22} 最好 t {b.t:+.2f}（{spec_name(b.spec)}）｜參考出場 毛 {ref.gross*100:+.3f}% 隨機毛 {(ref.rand_mean+COST)*100:+.3f}%")
    dtb = R[R.side == "L"].dropna(subset=["dt_mean"]).sort_values("dt_mean").iloc[-1]
    print(f"\n== 現股當沖成本（做多、當天出）最好：{show(dtb)}｜當沖淨 {dtb.dt_mean*100:+.3f}%")
    cand = R[(R.t >= thr) & (R.A > 0) & (R.B > 0) & (R.yrs.str.split('/').str[0].astype(int) >= 7) & (R.per_day >= 0.5) & (R.win >= 0.6)]
    print(f"\n候選（t ≥ {thr:.2f}、A/B > 0、正年 ≥ 7、每天 ≥ 0.5 筆、勝率 ≥ 60%）：{len(cand)} 個")
    sel = []
    for sd in ("L", "S"):
        c = cand[cand.side == sd].sort_values("t", ascending=False)
        if len(c):
            sel.append(dict(side=sd, ev=int(c.iloc[0].ev), spec=list(c.iloc[0].spec)))
            print("  選 val：" + show(c.iloc[0]))
    with open(RES / "scalp_sel.json", "w") as fh:
        json.dump(sel, fh, ensure_ascii=False)
    log(dict(period="dev", thr=thr, placebo=pb.tolist(), n_menu=len(R), n_pos=int((R['mean'] > 0).sum()), n_cand=len(cand),
             top=[{k: (v if not isinstance(v, (np.floating, float)) else round(float(v), 5)) for k, v in r.items()}
                  for r in R.sort_values("t", ascending=False).head(10).to_dict("records")], sel=sel,
             verdict="候選 → val" if sel else "✗"))


def main_val():
    sel = json.load(open(RES / "scalp_sel.json"))
    if not sel:
        print("dev 沒有候選，依登記不跑 val")
        return
    rows, _, _ = run("val")
    R = pd.DataFrame(rows)
    out = []
    for s in sel:
        r = R[(R.side == s["side"]) & (R.ev == s["ev"]) & (R.spec.apply(lambda x: list(x) == s["spec"]))].iloc[0]
        ok = r.t >= 2 and r["mean"] > 0
        print(("✓ " if ok else "✗ ") + show(r))
        out.append(dict(sel=s, t=r.t, mean=r["mean"], win=r.win, ok=bool(ok)))
    log(dict(period="val", res=out))


def log(d):
    with open(H.LOG, "a") as fh:
        fh.write(json.dumps(dict(ts=time.strftime("%Y-%m-%d %H:%M:%S"), kind="scalp", strategy="第八十三批 H196 股票池事件型短打",
                                 note="第八十三批 事前登記", **d), ensure_ascii=False,
                            default=lambda v: round(float(v), 5) if isinstance(v, (float, np.floating)) else str(v)) + "\n")


def leak():
    P = panel()
    A = events(P)
    B = events(P, cut="2016-06-30")
    k = len(B["idx"])
    bad = []
    for key in A["ev"]:
        a, b = A["ev"][key].iloc[:k].values, B["ev"][key].values
        if not np.array_equal(np.isfinite(a), np.isfinite(b)) or not np.allclose(np.nan_to_num(a), np.nan_to_num(b)):
            bad.append(key)
    print(f"防前視（截到 2016-06-30 重算 16 個事件）：{'OK' if not bad else bad}")


if __name__ == "__main__":
    {"dev": main_dev, "val": main_val, "leak": leak}[sys.argv[1]]()
