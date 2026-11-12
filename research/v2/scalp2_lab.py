"""第八十四批 H197：股票池短打，多條件組合（1～3 個條件 AND）。規則見 HYPOTHESES.md 第八十四批（事前登記）。
池、成交、成本、27 種出場沿用 H196（scalp_lab）；每天的報酬＝當天所有符合股票的等權平均（沒有就 0）。
資料截到 2023-12-31（holdout 不讀）。

用法：.venv/bin/python research/v2/scalp2_lab.py dev|val|leak
"""
import json
import multiprocessing as mp
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402
import scalp_lab as S  # noqa: E402

END, COST, SEG = S.END, S.COST, S.SEG
SPECS = S.specs()
ACT = 0.5
NPLAC = 20
CHUNK = 500
CN = ["前日≤−5%", "前日−5~−2%", "前日+2~+5%", "前日≥+5%", "跳空≤−2%", "跳空−2~−0.5%", "跳空+0.5~+2%", "跳空≥+2%",
      "收在低20%", "收在高20%", "RSI2<10", "RSI2>90", ">MA200", "<MA200", ">MA20", "<MA20", "連3跌", "連3漲",
      "20日新高", "20日新低", "量≥2倍", "量≤0.5倍", "5日≤−10%", "5日≥+10%", "前日漲停", "前日跌停",
      "大盤前日≤−1%", "大盤前日≥+1%", "振幅≥6%", "振幅≤2%", "週一", "週五"]


def conditions(P, cut=None):
    if cut is not None:
        P = {k: v.loc[:cut] for k, v in P.items()}
    O, Hh, L, C, V = P["open"], P["high"], P["low"], P["close"], P["volume"].astype(float)
    idx = C.index
    lim = pd.Series(np.where(idx < H.LIMIT_CHANGE, 0.07, 0.10), index=idx)
    pc = C.shift(1)
    gap = O / pc - 1
    r1 = C / pc - 1
    p1 = r1.shift(1)
    mny = P["money"].astype(float).rolling(60, min_periods=40).mean().shift(1)
    ok = P["liq"].shift(1).fillna(False).astype(bool) & (P["nbars"].shift(1) >= H.MIN_BARS + 130)
    pool = mny.where(ok).rank(axis=1, ascending=False) <= S.TOPN
    flat = (O == Hh) & (O == L)
    art = gap.abs().gt(lim + 0.01, axis=0)
    upl = flat & gap.ge(lim - 0.006, axis=0)
    dnl = flat & gap.le(-(lim - 0.006), axis=0)
    base = pool & ~art & O.notna()
    dx = C.diff()
    up = dx.clip(lower=0).ewm(alpha=1 / 2, adjust=False).mean()
    dn = (-dx.clip(upper=0)).ewm(alpha=1 / 2, adjust=False).mean()
    rsi = (100 - 100 / (1 + up / dn)).shift(1)
    ma200, ma20 = C.rolling(200).mean().shift(1), C.rolling(20).mean().shift(1)
    vr = (V / V.rolling(20).mean().shift(1)).shift(1)
    pos = ((C - L) / (Hh - L)).shift(1)
    rng_ = ((Hh - L) / C).shift(1)
    r5 = (C / C.shift(5) - 1).shift(1)
    limp = lim.shift(1)
    tr = pd.read_parquet(H.D / "taiex_tr.parquet")
    tr = tr.assign(date=pd.to_datetime(tr.date)).set_index("date").price.loc[:END]
    if cut is not None:
        tr = tr.loc[:cut]
    mk = tr.reindex(idx).pct_change().shift(1)
    sh = lambda x: x.shift(1, fill_value=False)  # noqa: E731
    wd = pd.DataFrame(np.repeat(idx.weekday.values[:, None], C.shape[1], 1), index=idx, columns=C.columns)
    cond = [p1 <= -0.05, (p1 > -0.05) & (p1 <= -0.02), (p1 >= 0.02) & (p1 < 0.05), p1 >= 0.05,
            gap <= -0.02, (gap > -0.02) & (gap <= -0.005), (gap >= 0.005) & (gap < 0.02), gap >= 0.02,
            pos <= 0.2, pos >= 0.8, rsi < 10, rsi > 90, pc > ma200, pc < ma200, pc > ma20, pc < ma20,
            sh((r1 < 0) & (r1.shift(1) < 0) & (r1.shift(2) < 0)), sh((r1 > 0) & (r1.shift(1) > 0) & (r1.shift(2) > 0)),
            sh(C >= C.rolling(20).max()), sh(C <= C.rolling(20).min()), vr >= 2, vr <= 0.5, r5 <= -0.10, r5 >= 0.10,
            p1.ge(limp - 0.005, axis=0), p1.le(-(limp - 0.005), axis=0),
            pd.DataFrame(np.repeat((mk <= -0.01).values[:, None], C.shape[1], 1), index=idx, columns=C.columns),
            pd.DataFrame(np.repeat((mk >= 0.01).values[:, None], C.shape[1], 1), index=idx, columns=C.columns),
            rng_ >= 0.06, rng_ <= 0.02, wd == 0, wd == 4]
    cond = [c.fillna(False).astype(bool) for c in cond]
    return dict(cond=cond, base=base, vL=~upl, vS=~dnl, idx=idx)


def flatten(P, X, period):
    """把期間內的池內股票日攤平成 N 筆（依日期排序），算 27 種出場的多／空淨報酬。"""
    idx = X["idx"]
    T = len(idx)
    inw = np.asarray((idx >= SEG[period][0]) & (idx <= SEG[period][1])) & (np.arange(T) + 3 < T)
    b = X["base"].values & inw[:, None]
    d, c = np.nonzero(b)
    days = np.nonzero(inw)[0]
    di = np.searchsorted(days, d)
    Cm = np.stack([x.values[d, c] for x in X["cond"]], 1)
    R, V = {}, {}
    for sd in ("L", "S"):
        W = S.windows(P, d, c, sd)
        R[sd] = np.stack([S.exits(W, *sp) for sp in SPECS], 1).astype(np.float32) - COST
        V[sd] = X["v" + sd].values[d, c]
    return dict(di=di, nd=len(days), dates=idx[days], C=Cm, R=R, V=V)


def rules():
    n = len(CN)
    out = [(i,) for i in range(n)]
    out += list(combinations(range(n), 2)) + list(combinations(range(n), 3))
    return out


def masks(Cm, rr):
    M = np.ones((len(rr), Cm.shape[0]), bool)
    for j, r in enumerate(rr):
        for k in r:
            M[j] &= Cm[:, k]
    return M


def dmat(w, di, nd):
    """每天合計用的稀疏矩陣（nd × N），值＝權重 w。"""
    from scipy import sparse
    return sparse.csr_matrix((w.astype(np.float32), (di, np.arange(len(di)))), shape=(nd, len(di)))


def nw_t(X, lag):
    T = X.shape[0]
    m = X.mean(0)
    Y = X - m
    s = (Y * Y).sum(0) / T
    for L in range(1, lag + 1):
        s = s + 2 * (1 - L / (lag + 1)) * (Y[L:] * Y[:-L]).sum(0) / T
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(s > 0, m / np.sqrt(s / T), np.nan)


def evaluate(F, rr, perm=None, full=True):
    """回傳頻率合格項目的統計（full=False 只回最高 t）。"""
    di, nd, dates = F["di"], F["nd"], F["dates"]
    yrs = dates.year.values
    segA = np.asarray(dates <= SEG["A"][1])
    rows, best = [], -9.0
    for sd in ("L", "S"):
        R, V = F["R"][sd], F["V"][sd]
        if perm is not None:
            R, V = R[perm], V[perm]
        D1 = dmat(np.ones(len(di)), di, nd)
        DE = [dmat(R[:, e], di, nd) for e in range(len(SPECS))]
        for a in range(0, len(rr), CHUNK):
            ch = rr[a:a + CHUNK]
            Mb = masks(F["C"], ch) & V[None]
            cnt = D1 @ np.ascontiguousarray(Mb.T, dtype=np.float32)
            act = (cnt > 0).mean(0)
            keep = np.flatnonzero(act >= ACT)
            if not len(keep):
                continue
            M = Mb[keep].astype(np.float32)
            MT = np.ascontiguousarray(M.T)
            cnt = cnt[:, keep]
            for e, sp in enumerate(SPECS):
                dm_e = np.where(cnt > 0, (DE[e] @ MT) / np.maximum(cnt, 1), 0)
                t = nw_t(dm_e, sp[0] + 1)
                best = max(best, np.nanmax(t))
                if not full:
                    continue
                ntr = M.sum(1)
                win = (M @ (R[:, e] > 0).astype(np.float32)) / ntr
                tm = (M @ R[:, e]) / ntr
                A, B = dm_e[segA].mean(0), dm_e[~segA].mean(0)
                ys = pd.DataFrame(dm_e).groupby(yrs).sum()
                ypos = (ys.values > 0).sum(0)
                for j, kk in enumerate(keep):
                    rows.append(dict(rule=ch[kk], side=sd, e=e, t=float(t[j]), A=float(A[j] * 1e4), B=float(B[j] * 1e4),
                                     ypos=int(ypos[j]), ny=len(ys), act=float(act[kk]), per_day=float(ntr[j] / nd),
                                     win=float(win[j]), mean=float(tm[j]), dmean=float(dm_e[:, j].mean())))
    return rows, best


def one_per_day(F, rule, sd, e, n=200, seed=0):
    """每天從符合的股票隨機挑 1 檔（使用者要的每天 1 筆）：n 次的 t 中位數、每天淨報酬中位數。"""
    m = masks(F["C"], [tuple(rule)])[0] & F["V"][sd]
    ii = np.flatnonzero(m)
    di = F["di"][ii]
    r = F["R"][sd][ii, e]
    rng = np.random.default_rng(seed)
    ts, mu = [], []
    for _ in range(n):
        o = np.lexsort((rng.random(len(ii)), di))
        first = np.r_[True, np.diff(di[o]) != 0]
        x = np.zeros(F["nd"])
        x[di[o][first]] = r[o][first]
        ts.append(nw_t(x[:, None], SPECS[e][0] + 1)[0])
        mu.append(x.mean())
    return float(np.median(ts)), float(np.median(mu))


def plac(args):
    F, rr, seed = args
    rng = np.random.default_rng(seed)
    perm = np.lexsort((rng.random(len(F["di"])), F["di"]))
    return evaluate(F, rr, perm, full=False)[1]


def show(r):
    return (f"{'多' if r['side'] == 'L' else '空'} {' & '.join(CN[k] for k in r['rule']):<30} {S.spec_name(SPECS[r['e']]):<22}"
            f" 有交易日 {r['act']:.0%}、每天 {r['per_day']:.1f} 檔 勝率 {r['win']*100:3.0f}% 每筆淨 {r['mean']*100:+.3f}%"
            f" t {r['t']:+.2f} A/B {r['A']:+.1f}/{r['B']:+.1f}bp 正年 {r['ypos']}/{r['ny']}")


def main_dev():
    t0 = time.time()
    P = S.panel()
    X = conditions(P)
    F = flatten(P, X, "dev")
    rr = rules()
    print(f"dev：{F['nd']} 天、池內股票日 {len(F['di']):,}；規則 {len(rr):,} × 2 邊 × {len(SPECS)} 出場（{time.time()-t0:.0f}s）", flush=True)
    rows, best = evaluate(F, rr)
    R = pd.DataFrame(rows)
    print(f"頻率合格（有交易日 ≥ {ACT:.0%}）{len(R):,} 項（{R.groupby(['rule','side']).ngroups:,} 個規則×邊）；最高 t {best:.2f}（{time.time()-t0:.0f}s）", flush=True)
    with mp.get_context("fork").Pool(5) as pool:
        pb = np.array(pool.map(plac, [(F, rr, s) for s in range(NPLAC)]))
    thr = max(2.0, float(np.quantile(pb, 0.95)))
    print(f"安慰劑（{NPLAC} 次，當天打亂報酬、整套重跑）最高 t：中位 {np.median(pb):.2f}、95 分位 {np.quantile(pb, .95):.2f} → 門檻 {thr:.2f}（{time.time()-t0:.0f}s）")
    R["nc"] = R.rule.apply(len)
    print("\n== 條件數 → 項目數、最高 t、t ≥ 2 個數")
    for nc, g in R.groupby("nc"):
        print(f"  {nc} 個條件：{len(g):,} 項，最高 t {g.t.max():+.2f}，t ≥ 2：{int((g.t >= 2).sum())}，t ≥ 門檻：{int((g.t >= thr).sum())}")
    print("\n== dev t 最高 20 個")
    for _, r in R.sort_values("t", ascending=False).head(20).iterrows():
        print("  " + show(r))
    hw = R[R.win >= 0.6]
    print(f"\n== 勝率 ≥ 60%：{len(hw):,} 項，其中每筆淨 > 0 的 {int((hw['mean'] > 0).sum()):,}；t 最高 10 個")
    for _, r in hw.sort_values("t", ascending=False).head(10).iterrows():
        print("  " + show(r))
    print("\n== 出場分布（t ≥ 2 的項目）")
    print(R[R.t >= 2].e.map(lambda e: S.spec_name(SPECS[e])).value_counts().head(10).to_string())
    cand = R[(R.t >= thr) & (R.A > 0) & (R.B > 0) & (R.ypos >= 7) & (R.win >= 0.6)]
    print(f"\n候選（t ≥ {thr:.2f}、A/B > 0、正年 ≥ 7、勝率 ≥ 60%）：{len(cand)} 個")
    sel = []
    for sd in ("L", "S"):
        c = cand[cand.side == sd].sort_values("t", ascending=False)
        for _, r in c.head(5).iterrows():
            print("  " + show(r))
        if len(c):
            sel.append(dict(side=sd, rule=list(c.iloc[0].rule), e=int(c.iloc[0].e)))
    for s_ in sel:
        t1, m1 = one_per_day(F, s_["rule"], s_["side"], s_["e"])
        s_["one_t"] = t1
        print(f"  每天隨機挑 1 檔（200 次中位）：{s_['side']} t {t1:+.2f}、每天平均淨 {m1*1e4:+.1f}bp")
    print("選 val：" + ("；".join(f"{s['side']} {[CN[k] for k in s['rule']]} {S.spec_name(SPECS[s['e']])}" for s in sel) or "無"))
    with open(S.RES / "scalp2_sel.json", "w") as fh:
        json.dump(sel, fh, ensure_ascii=False)
    top = R.sort_values("t", ascending=False).head(10)
    log(dict(period="dev", n_items=len(rr) * 2 * len(SPECS), n_freq_ok=len(R), best_t=best, placebo=pb.tolist(), thr=thr,
             n_t2=int((R.t >= 2).sum()), n_cand=len(cand),
             top=[dict(rule=[CN[k] for k in r.rule], side=r.side, exit=S.spec_name(SPECS[r.e]), t=r.t, A=r.A, B=r.B, win=r.win,
                       mean=r.mean, act=r.act) for r in top.itertuples()],
             sel=sel, verdict="候選 → val" if sel else "✗"))
    R.to_pickle(S.RES / "scalp2_dev.pkl")


def main_val():
    sel = json.load(open(S.RES / "scalp2_sel.json"))
    if not sel:
        print("dev 沒有候選，依登記不跑 val")
        return
    P = S.panel()
    X = conditions(P)
    F = flatten(P, X, "val")
    out = []
    for s in sel:
        rows, _ = evaluate(F, [tuple(s["rule"])])
        r = [x for x in rows if x["side"] == s["side"] and x["e"] == s["e"]]
        if not r:
            print(f"✗ {s}：val 有交易日 < {ACT:.0%}")
            out.append(dict(sel=s, ok=False, note="頻率不足"))
            continue
        r = r[0]
        ok = r["t"] >= 2 and r["dmean"] > 0
        print(("✓ " if ok else "✗ ") + show(r))
        t1, m1 = one_per_day(F, s["rule"], s["side"], s["e"])
        print(f"   每天隨機挑 1 檔（200 次中位）：t {t1:+.2f}、每天平均淨 {m1*1e4:+.1f}bp")
        out.append(dict(sel=s, t=r["t"], mean=r["mean"], win=r["win"], A=r["A"], B=r["B"], ypos=r["ypos"], act=r["act"],
                        one_t=t1, ok=bool(ok)))
    log(dict(period="val", res=out))


def log(d):
    with open(H.LOG, "a") as fh:
        fh.write(json.dumps(dict(ts=time.strftime("%Y-%m-%d %H:%M:%S"), kind="scalp", strategy="第八十四批 H197 股票池短打多條件組合",
                                 note="第八十四批 事前登記", **d), ensure_ascii=False,
                            default=lambda v: round(float(v), 5) if isinstance(v, (float, np.floating)) else str(v)) + "\n")


def leak():
    P = S.panel()
    A, B = conditions(P), conditions(P, cut="2016-06-30")
    k = len(B["idx"])
    bad = [i for i in range(len(CN)) if not np.array_equal(A["cond"][i].values[:k], B["cond"][i].values)]
    bad += ["base"] if not np.array_equal(A["base"].values[:k], B["base"].values) else []
    print(f"防前視（截到 2016-06-30 重算 32 個條件與池）：{'OK' if not bad else [CN[i] if isinstance(i, int) else i for i in bad]}")


if __name__ == "__main__":
    {"dev": main_dev, "val": main_val, "leak": leak}[sys.argv[1]]()
