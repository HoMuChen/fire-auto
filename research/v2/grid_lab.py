"""第六十批 H165：市值前 20 大網格（個股期貨做多），格距 × 層數 × 停止條件 × 濾網。
規則見 HYPOTHESES.md 第六十批（事前登記）。資料用第五十九批的快取（timing_lab.py cache dev|val）。

用法：.venv/bin/python research/v2/grid_lab.py dev
      .venv/bin/python research/v2/grid_lab.py val
"""
import itertools
import json
import multiprocessing as mp
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402
import strategy_lab as SL  # noqa: E402
import timing_lab as TL  # noqa: E402

KS, NS, STOPS = (0.5, 1, 2), (5, 10), ("年線", "無")
FILTS = ("無", "大盤>MA60", "收>MA60", "低波動", "外資20日買")
MENU = list(itertools.product(KS, NS, STOPS, FILTS))
SIDE, ROLL_M, TICK, MIN_STEP, SLOTS, NPL = 0.0005, 0.001, 0.001, 0.003, 20, 100
TRIALS = 4017
SEG, CHIP_START = TL.SEG, TL.CHIP_START


def nm(v):
    return f"k{v[0]}/N{v[1]}/停止{v[2]}/濾網{v[3]}"


class Panel:
    """價格（可換成安慰劑路徑）＋固定的成員、大盤、外資。"""

    def __init__(self, D):
        self.days = D["open"].index
        self.mem = D["member"].values
        self.O, self.H, self.L, self.C = (D[k].values for k in ("open", "high", "low", "close"))
        mk = D["mkt"]
        mma = mk.rolling(60).mean()
        self.mkt_ok = np.broadcast_to(((mk > mma) & mma.notna()).values[:, None], self.C.shape)
        self.f20 = (D["inst_f"].rolling(20, min_periods=20).sum() > 0).values
        self.cols = D["close"].columns

    def shuffled(self, seed):
        """所有股票用同一個排列打亂「日」的順序（開高低收對前一日收盤的比例整組搬），重組價格路徑。
        保留股票之間的同步與每天的分布，只拿掉序列上的結構。沒有資料的日子比例當 1（價格不動、不會成交）。"""
        rng = np.random.default_rng(seed)
        P = object.__new__(Panel)
        P.days, P.mem, P.mkt_ok, P.f20, P.cols = self.days, self.mem, self.mkt_ok, self.f20, self.cols
        T, S = self.C.shape
        pc = self.C[:-1]
        rel = np.stack([self.O[1:] / pc, self.H[1:] / pc, self.L[1:] / pc, self.C[1:] / pc], axis=2)
        rel = np.where(np.isfinite(rel).all(axis=2, keepdims=True), rel, 1.0)
        rel = rel[rng.permutation(T - 1)]
        c0 = np.array([self.C[np.flatnonzero(np.isfinite(self.C[:, j]))[0], j] if np.isfinite(self.C[:, j]).any() else 100.0 for j in range(S)])
        cc = c0[None, :] * np.vstack([np.ones((1, S)), np.cumprod(rel[:, :, 3], axis=0)])
        prev = cc[:-1]
        P.C = cc
        P.O = np.vstack([cc[:1], prev * rel[:, :, 0]])
        P.H = np.vstack([cc[:1], prev * rel[:, :, 1]])
        P.L = np.vstack([cc[:1], prev * rel[:, :, 2]])
        return P


def derived(P):
    Cdf = pd.DataFrame(P.C)
    r = Cdf.pct_change(fill_method=None)
    sd = r.rolling(20, min_periods=15).std().values
    ma60 = Cdf.rolling(60, min_periods=60).mean().values
    ma250 = Cdf.rolling(250, min_periods=250).mean().values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lowvol = (TL.own_pct(pd.DataFrame(sd)) <= 0.5).values
    return dict(sd=sd, above60=P.C > ma60, below250=P.C < ma250, lowvol=lowvol)


def hold_series(P):
    r = P.C[1:] / P.C[:-1] - 1
    r = np.vstack([np.full((1, P.C.shape[1]), np.nan), r])
    m = P.mem
    n = np.maximum(m.sum(axis=1), 1)
    dm = np.abs(np.diff(m.astype(np.int8), axis=0, prepend=0)).sum(axis=1)
    h = (np.nan_to_num(np.where(m, r, 0.0)).sum(axis=1) - SIDE * dm) / n - ROLL_M / 21
    return np.where(m.sum(axis=1) > 0, h, np.nan), np.nan_to_num(np.where(m, r, 0.0))


def simulate(P, dv, v):
    """回傳（每日報酬、每日每股損益、每日曝險、來回次數、每股來回次數）。"""
    k, N, stop, flt = v
    T, S = P.C.shape
    w = 1.0 / (SLOTS * N)
    lay = np.full((S, N), np.nan)
    lst = np.zeros((S, N))
    pnl = np.zeros((T, S))
    expo = np.zeros(T)
    trips = np.zeros(S)
    step = np.maximum(k * np.nan_to_num(dv["sd"], nan=np.inf), MIN_STEP)
    stopf = dv["below250"] if stop == "年線" else np.zeros((T, S), bool)
    fl = {"無": np.ones((T, S), bool), "大盤>MA60": P.mkt_ok, "收>MA60": dv["above60"], "低波動": dv["lowvol"], "外資20日買": P.f20}[flt]
    O, Hh, L, C, mem = P.O, P.H, P.L, P.C, P.mem
    for t in range(1, T):
        o, hi, lo, c, pc = O[t], Hh[t], L[t], C[t], C[t - 1]
        valid = np.isfinite(o) & np.isfinite(hi) & np.isfinite(lo) & np.isfinite(c) & np.isfinite(pc)
        held = np.isfinite(lay)
        out = (stopf[t - 1] | ~mem[t]) & valid
        # 1. 停止／出池：開盤出清
        if out.any() and held[out].any():
            nh = held[out].sum(axis=1)
            pnl[t, out] += w * nh * (o[out] / pc[out] - 1 - SIDE)
            trips[out] += nh
            lay[out] = np.nan
            held = np.isfinite(lay)
        # 2. 停利
        tp = lay * (1 + lst)
        hit = held & valid[:, None] & (hi[:, None] >= tp * (1 + TICK))
        if hit.any():
            fill = np.maximum(tp, o[:, None])
            pnl[t] += w * np.where(hit, fill / pc[:, None] - 1 - SIDE, 0.0).sum(axis=1)
            trips += hit.sum(axis=1)
            lay[hit] = np.nan
            held = np.isfinite(lay)
        # 3. 續抱
        nh = held.sum(axis=1)
        live = valid & (nh > 0)
        pnl[t, live] += w * nh[live] * (c[live] / pc[live] - 1 - ROLL_M / 21)
        # 4. 買進
        can = valid & mem[t] & ~stopf[t - 1] & fl[t - 1] & (nh < N) & ~out
        if can.any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                ref = np.where(nh > 0, np.nanmin(lay, axis=1), pc)
            b = ref * (1 - step[t - 1])
            buy = can & (lo <= b * (1 - TICK))
            if buy.any():
                fill = np.minimum(b, o)
                js = np.flatnonzero(buy)
                slot = np.argmax(~np.isfinite(lay[js]), axis=1)
                lay[js, slot] = fill[js]
                lst[js, slot] = step[t - 1, js]
                pnl[t, js] += w * (c[js] / fill[js] - 1 - SIDE - ROLL_M / 21)
        expo[t] = np.isfinite(lay).sum() * w
    s = pnl.sum(axis=1)
    return s, pnl, expo, trips


def scan(P, menu, m_dev, m_chip):
    dv = derived(P)
    h, _ = hold_series(P)
    out = np.full(len(menu), np.nan)
    for i, v in enumerate(menu):
        s, *_ = simulate(P, dv, v)
        m = (m_chip if v[3] == "外資20日買" else m_dev) & np.isfinite(h)
        out[i] = TL.alpha_t(s, h, m)[0]
    return out


_G = {}


def _pl(seed):
    return float(np.nanmax(scan(_G["P"].shuffled(2060_000 + seed), MENU, _G["md"], _G["mc"])))


def detail(P, v, seg_main, segs):
    dv = derived(P)
    h, rr = hold_series(P)
    s, pnl, expo, trips = simulate(P, dv, v)
    days = P.days
    p0 = max(SEG[seg_main][0], CHIP_START) if v[3] == "外資20日買" else SEG[seg_main][0]
    m = TL.mask(days, p0, SEG[seg_main][1]) & np.isfinite(h)
    t, a, b = TL.alpha_t(s, h, m)
    out = dict(var=nm(v), alpha_t=t, alpha_ann=a * 252, beta=b, sharpe=TL.sharpe(s[m]), sharpe_hold=TL.sharpe(h[m]),
               expo=float(expo[m].mean()), trips=int(trips.sum()))
    for g in segs:
        q0 = max(SEG[g][0], CHIP_START) if v[3] == "外資20日買" else SEG[g][0]
        mg = TL.mask(days, q0, SEG[g][1]) & np.isfinite(h)
        out[f"sharpe_{g}"], out[f"hold_{g}"] = TL.sharpe(s[mg]), TL.sharpe(h[mg])
    pos, tsmc = [], None
    for j in range(P.C.shape[1]):
        mj = m & P.mem[:, j]
        if mj.sum() < 250:
            continue
        X = np.column_stack([np.ones(mj.sum()), rr[mj, j]])
        a_j = np.linalg.lstsq(X, pnl[mj, j] * SLOTS, rcond=None)[0][0]
        pos.append(a_j > 0)
        if P.cols[j] == "2330":
            tt = TL.alpha_t(pnl[:, j] * SLOTS, rr[:, j], mj)
            tsmc = dict(alpha_t=tt[0], alpha_ann=tt[1] * 252, sharpe=TL.sharpe((pnl[:, j] * SLOTS)[mj]), sharpe_hold=TL.sharpe(rr[mj, j]))
    out["breadth"] = float(np.mean(pos)) if pos else np.nan
    out["tsmc"] = tsmc
    yr = pd.Series(s[m] - b * h[m]).groupby(days[m].year).sum().mul(100).round(1).to_dict()
    out["alpha_by_year"] = yr
    return out, s


def run_dev():
    t0 = time.time()
    D = TL.load("dev")
    P = Panel(D)
    days = P.days
    md, mc = TL.mask(days, *SEG["dev"]), TL.mask(days, CHIP_START, SEG["dev"][1])
    real = scan(P, MENU, md, mc)
    print(f"菜單 {len(MENU)} 個（{time.time()-t0:.0f}s）；alpha t ≥ 2 的 {(real >= 2).sum()} 個；最高 {np.nanmax(real):.2f}", flush=True)
    _G.update(P=P, md=md, mc=mc)
    t1 = time.time()
    with mp.get_context("fork").Pool(8) as pool:
        pl = np.array(pool.map(_pl, range(NPL)))
    p95 = float(np.percentile(pl, 95))
    print(f"安慰劑 {NPL} 次（{time.time()-t1:.0f}s）：菜單最佳 alpha t 50／95 分位 {np.percentile(pl, 50):.2f}／{p95:.2f}", flush=True)
    order = np.argsort(-np.nan_to_num(real, nan=-9))
    print("\n  菜單 alpha t 前 12：")
    for i in order[:12]:
        print(f"    {nm(MENU[i]):<36} {real[i]:+.2f}")
    print("  菜單 alpha t 後 5：", ", ".join(f"{nm(MENU[i])} {real[i]:+.2f}" for i in order[-5:]))
    rows = []
    for i in order:
        if not real[i] >= max(2, p95):
            break
        r, s = detail(P, MENU[i], "dev", ("A", "B"))
        r["pass"] = r["sharpe_A"] > r["hold_A"] and r["sharpe_B"] > r["hold_B"] and r["breadth"] >= 0.5 and r["trips"] >= 200
        rows.append(r)
    sel = next((r for r in rows if r["pass"]), None)
    extra = {}
    best_i = order[0]
    rb, sb = detail(P, MENU[best_i], "dev", ("A", "B"))
    print(f"\n  菜單第一名細節（不論是否過門檻）：{rb['var']} alpha t {rb['alpha_t']:.2f}（年化 {rb['alpha_ann']*100:+.1f}%）Sharpe {rb['sharpe']:.2f}"
          f"（持有 {rb['sharpe_hold']:.2f}；A {rb['sharpe_A']:.2f}/{rb['hold_A']:.2f}、B {rb['sharpe_B']:.2f}/{rb['hold_B']:.2f}）廣度 {rb['breadth']:.0%}"
          f" 曝險 {rb['expo']:.0%} 來回 {rb['trips']}｜台積電 {rb['tsmc']}｜逐年 alpha {rb['alpha_by_year']}")
    print(f"\n== 超過門檻（{max(2, p95):.2f}）的 {len(rows)} 個；通過全部條件：{sum(r['pass'] for r in rows)} 個 → 進 val：{sel['var'] if sel else '無'}")
    if sel:
        v = next(x for x in MENU if nm(x) == sel["var"])
        r, s = detail(P, v, "dev", ("A", "B"))
        hc = TL.h136_check(s, type("B", (), {"days": days})(), D, "dev", ("A", "B"))
        m = md & np.isfinite(s)
        srs = real[np.isfinite(real)] / np.sqrt(m.sum())
        extra = dict(**hc, dsr=SL.deflated_sharpe(s[m] - r["beta"] * hold_series(P)[0][m], srs, TRIALS))
        print(f"  和 H136 月相關 {hc['rho_h136']:+.2f}｜關卡 3 {hc['gate3']} {'✓' if hc['gate3_ok'] else '✗'}｜DSR {extra['dsr']:.3f}")
    SL.log(dict(kind="strategy", period="dev", strategy="第六十批 H165 前 20 大網格", note="第六十批 事前登記", menu=len(MENU),
                real=dict(zip([nm(v) for v in MENU], np.round(real, 3).tolist())), placebo_p50=float(np.percentile(pl, 50)), placebo_p95=p95,
                candidates=rows, best=rb, selected=sel["var"] if sel else None, extra=extra, trials_cum=TRIALS))


def run_val():
    rec = [json.loads(l) for l in open(H.LOG) if '"第六十批 H165 前 20 大網格"' in l and '"period": "dev"' in l][-1]
    if not rec["selected"]:
        print("dev 沒有選出 → 依登記不跑 val")
        return
    D = TL.load("val")
    P = Panel(D)
    v = next(x for x in MENU if nm(x) == rec["selected"])
    r, s = detail(P, v, "val", ("val",))
    ok = r["alpha_t"] >= 2 and r["sharpe"] > r["sharpe_hold"]
    hc = TL.h136_check(s, type("B", (), {"days": P.days})(), D, "val", ("val",))
    print(f"  {r['var']}：alpha t {r['alpha_t']:.2f}（年化 {r['alpha_ann']*100:+.1f}%）Sharpe {r['sharpe']:.2f}（持有 {r['sharpe_hold']:.2f}）"
          f" 廣度 {r['breadth']:.0%} 曝險 {r['expo']:.0%}｜逐年 alpha {r['alpha_by_year']}｜台積電 {r['tsmc']}｜和 H136 相關 {hc['rho_h136']:+.2f} → {'✓' if ok else '✗'}")
    SL.log(dict(kind="strategy", period="val", strategy="第六十批 H165 前 20 大網格", note="第六十批 dev 選出的",
                result=dict(**r, **hc, verdict="✓" if ok else "✗")))


if __name__ == "__main__":
    {"dev": run_dev, "val": run_val}[sys.argv[1] if len(sys.argv) > 1 else "dev"]()
