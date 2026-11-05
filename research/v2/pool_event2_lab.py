"""第六十九批：H180 性格指標的延續性（關卡）＋ H181 長持有順勢事件（10 格）。規則見 HYPOTHESES.md 第六十九批（事前登記）。
模擬沿用 pool_event_lab.simulate／evaluate（第六十八批）。

用法：.venv/bin/python research/v2/pool_event2_lab.py persist        # H180：只看資料
      .venv/bin/python research/v2/pool_event2_lab.py cache dev|val
      .venv/bin/python research/v2/pool_event2_lab.py dev
      .venv/bin/python research/v2/pool_event2_lab.py val
"""
import itertools
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import feature_industry as FI  # noqa: E402
import harness as H  # noqa: E402
import pool_event_lab as PE  # noqa: E402
import strategy_lab as SL  # noqa: E402
import vpool_lab as V  # noqa: E402
import xs  # noqa: E402

RES = H.D / "results"
GATE, TOPQ = 0.20, 0.2
MEAS = {"M1": "週報酬自我相關（104 週）", "M2": "變異數比 VR(4)（104 週）", "M3": "月報酬自我相關（36 個月）", "M4": "趨勢效率（52 週）",
        "M5": "產業週報酬自我相關（104 週）"}
ENTRY, EXIT, PRIO = ("T1", "T2", "T4"), ("X2", "X5", "X6", "X7"), ("P1", "P2")
MAXH = {"X2": 250, "X5": 250, "X6": 250, "X7": 250}
NAME = "第六十九批 H180／H181 性格延續＋長持有順勢事件"
TRIALS_FULL, TRIALS_SHORT = 1701625, 1701577


def base(period):
    import feature_library as FL
    end, p0 = H.PERIODS[period][1], H.PERIODS[period][0]
    P = xs.load_panel(end, fundamentals=True)
    cols = P["close"].columns
    dates, Rdf, Edf = FL.forward(P, p0, end)
    R = Rdf.reindex(index=dates, columns=cols).values
    E = Edf.reindex(index=dates, columns=cols).fillna(False).values.astype(bool) & np.isfinite(R)
    fx, trade, _ = V.gap_fix(P, cols)
    return P, cols, dates, E, fx, trade, end


def measures(fx, cols):
    """各指標：(週或月的 DataFrame, 窗長, 頻率)。"""
    Cw = fx["close"].resample("W-FRI").last()
    rw = Cw.pct_change(fill_method=None)
    lw = np.log(Cw.where(Cw > 0)).diff()
    v1 = rw.rolling(104, min_periods=80).var()
    r4 = rw.rolling(4, min_periods=4).sum()
    Cm = fx["close"].resample("ME").last()
    rm = Cm.pct_change(fill_method=None)
    codes = FI.labels(cols)
    ind = pd.DataFrame(np.nan, index=rw.index, columns=cols)
    for g in np.unique(codes[codes >= 0]):
        gc = cols[codes == g]
        ir = rw[gc].mean(axis=1)
        ac = ir.rolling(104, min_periods=80).corr(ir.shift(1))
        ind[gc] = np.repeat(ac.values[:, None], len(gc), axis=1)
    return {"M1": (rw.rolling(104, min_periods=80).corr(rw.shift(1)), 104),
            "M2": (r4.rolling(104, min_periods=80).var() / (4 * v1.where(v1 > 0)), 104),
            "M3": (rm.rolling(36, min_periods=30).corr(rm.shift(1)), 36),
            "M4": (lw.rolling(52, min_periods=45).sum().abs() / lw.abs().rolling(52, min_periods=45).sum(), 52),
            "M5": (ind, 104)}


def at_dates(X, dates):
    """換股日之前最後一個完整的週（或月）的值；回傳 (矩陣, 位置)。"""
    ix = X.index.values
    ks = np.array([np.searchsorted(ix, np.datetime64(d), side="left") - 1 for d in dates])
    out = np.full((len(dates), X.shape[1]), np.nan)
    ok = ks >= 0
    out[ok] = X.values[ks[ok]]
    return out, ks


def persist():
    t0 = time.time()
    P, cols, dates, E, fx, trade, end = base("dev")
    Ms = measures(fx, cols)
    res = {}
    for k, (X, L) in Ms.items():
        now, ks = at_dates(X, dates)
        rs = []
        for t in range(len(dates)):
            if ks[t] < 0 or ks[t] + L >= len(X):
                continue
            fut = X.values[ks[t] + L]
            m = E[t] & np.isfinite(now[t]) & np.isfinite(fut)
            if m.sum() > 100:
                rs.append(pd.Series(now[t][m]).rank().corr(pd.Series(fut[m]).rank()))
        rs = np.array(rs)
        res[k] = dict(mean=float(rs.mean()), median=float(np.median(rs)), pos=float((rs > 0).mean()), months=len(rs))
        print(f"  {k} {MEAS[k]}：延續性（等級相關）平均 {rs.mean():+.3f}、中位 {np.median(rs):+.3f}、> 0 {(rs > 0).mean():.0%}（{len(rs)} 個月）", flush=True)
    best = max(res, key=lambda k: res[k]["mean"])
    chosen = best if res[best]["mean"] >= GATE else None
    print(f"關卡（平均 ≥ {GATE}）：{'選 ' + chosen + ' ' + MEAS[chosen] if chosen else '都沒過 → A′／B′ 這條線結束，H181 只跑 C 池和對照'}（{time.time()-t0:.0f}s）", flush=True)
    SL.log(dict(kind="describe", period="dev", strategy=NAME, note="H180 性格指標延續性（只看資料）", persist=res, chosen=chosen))


def chosen_measure():
    rec = [json.loads(l) for l in open(H.LOG) if f'"{NAME}"' in l and '"H180' in l][-1]
    return rec["chosen"]


def cache(period):
    t0 = time.time()
    ch = chosen_measure()
    P, cols, dates, E, fx, trade, end = base(period)
    PM = {}
    if ch:
        X, _ = measures(fx, cols)[ch]
        now, _ = at_dates(X, dates)
        rk = pd.DataFrame(np.where(E & np.isfinite(now), now, np.nan)).rank(axis=1, pct=True).values
        PM["A2"], PM["B2"] = E & (rk > 1 - TOPQ), E & (rk <= TOPQ)
    pos6 = P["rev_pos6"].shift(1).reindex(index=dates, columns=cols).values
    yoy3 = P["rev_yoy3"].shift(1).reindex(index=dates, columns=cols).values
    with np.errstate(invalid="ignore"):
        PM["C"] = E & (pos6 >= 5) & (yoy3 > 0)
    PM["L"] = E
    U = np.flatnonzero(np.logical_or.reduce([M.any(axis=0) for M in PM.values()]))
    uc = cols[U]
    days = P["close"].index
    kd = np.searchsorted(dates.values, days.values, side="right") - 1
    daily = {}
    for p, M in PM.items():
        x = np.zeros((len(days), len(U)), bool)
        x[kd >= 0] = M[kd[kd >= 0]][:, U]
        daily[p] = x
    LU, LD, _, _ = V.limit_flags(fx)
    O, Hh, Lw, Cc = (fx[k][uc] for k in ("open", "high", "low", "close"))
    tr_ok = trade[uc]
    Cf = Cc.ffill()
    Hf, Lf = Hh.fillna(Cf), Lw.fillna(Cf)
    Vol = P["volume"][uc].astype(float).where(tr_ok).fillna(0.0)
    ma60 = Cf.rolling(60, min_periods=60).mean()
    hi20 = Hf.rolling(20, min_periods=20).max().shift(1)
    mx250 = Cf.rolling(250, min_periods=250).max()
    prevC = Cf.shift(1)
    atr = np.fmax(np.fmax(Hf - Lf, (Hf - prevC).abs()), (Lf - prevC).abs()).rolling(22, min_periods=22).mean()
    brk = Cf > hi20
    ent = {"T1": (Cf >= mx250) & ~(prevC >= mx250.shift(1)), "T2": brk & ~brk.shift(1, fill_value=False)}
    ent["T4"] = ent["T2"] & (Vol > 2 * Vol.rolling(20, min_periods=20).mean().shift(1))
    tr = pd.read_parquet(H.D / "taiex_tr.parquet")
    tr["date"] = pd.to_datetime(tr.date)
    mkt = tr.drop_duplicates("date").set_index("date").price.astype(float).sort_index().reindex(days).ffill()
    lastv = np.array([np.flatnonzero(tr_ok[c].values).max() if tr_ok[c].values.any() else -1 for c in uc])
    ret20 = (Cf / Cf.shift(20) - 1).values
    D = dict(days=days, cols=uc, dates=dates, period=period, chosen=ch,
             O=O.values.astype(np.float64), H=Hf.values, C=Cf.values, trade=tr_ok.values, LU=LU[uc].fillna(False).values, LD=LD[uc].fillna(False).values,
             atr=atr.values, ret20=ret20, lastv=lastv, ent={k: (v & tr_ok).fillna(False).values for k, v in ent.items()},
             ex={"X7": (Cf < ma60).fillna(False).values}, prio={"P1": -ret20, "P2": -P["rev_yoy3"][uc].values},
             mkt=mkt.values, pool=daily, pool_n={p: M.sum(axis=1) for p, M in PM.items()})
    with open(RES / f"pool_event2_{period}.pkl", "wb") as f:
        pickle.dump(D, f)
    print(f"快取完成 {time.time()-t0:.0f}s；性格指標 {ch}；股票 {len(U)} 檔；池每月平均檔數 " + "、".join(f"{p} {M.sum(1).mean():.0f}" for p, M in PM.items()), flush=True)


def load(period):
    with open(RES / f"pool_event2_{period}.pkl", "rb") as f:
        return pickle.load(f)


def pools(D):
    return [p for p in ("A2", "B2", "C", "L") if p in D["pool"]]


def models(D):
    return [dict(pool=p, fam="T", entry=e, exit=x, prio=q, filt="F0", max_hold=MAXH[x])
            for p in pools(D) for e, x, q in itertools.product(ENTRY, EXIT, PRIO)]


def mname(m):
    pl = {"A2": "A′池", "B2": "B′池", "C": "C池", "L": "流動池（對照）"}[m["pool"]]
    return f"{pl} {PE.LABEL[m['entry']]}→{PE.LABEL[m['exit']]}（{PE.LABEL[m['prio']]}）"


def run(D, seg):
    M = models(D)
    rows = PE.run_all(D, M, seg)
    for r, m in zip(rows, M):
        r["name"] = mname(m)
        r["prio"] = m["prio"]
    return M, rows


def dev():
    t0 = time.time()
    D = load("dev")
    M, rows = run(D, "dev")
    print(f"dev {PE.SEG['dev']}；性格指標 {D['chosen']}；模型 {len(M)} 個（{time.time()-t0:.0f}s）", flush=True)
    T = pd.DataFrame(rows)
    g = T.groupby("pool")[["alpha_t", "sharpe"]].mean()
    print("各池平均：\n" + g.round(2).to_string(), flush=True)
    fit = None
    if "A2" in D["pool"]:
        fit = bool(g.loc["A2", "alpha_t"] > g.loc["B2", "alpha_t"])
        print(f"搭配預測（A′ > B′）dev：{'成立' if fit else '不成立'}", flush=True)
    print("依出場平均 alpha t：", T.groupby("exit").alpha_t.mean().round(2).to_dict(), "；依排序：", T.groupby("prio").alpha_t.mean().round(2).to_dict(), flush=True)
    print("\nalpha t 前 15：", flush=True)
    for _, r in T.sort_values("alpha_t", ascending=False).head(15).iterrows():
        PE.show(r)
    ok = [r for r in sorted(rows, key=lambda r: -r["alpha_t"]) if PE.passes(r)]
    print(f"\n過 dev 三條：{len(ok)} 個（" + "、".join(f"{p} {sum(r['pool'] == p for r in ok)}" for p in pools(D)) + "）；對照池不進 val", flush=True)
    sel, ser = [], []
    for r in ok:
        if r["pool"] == "L":
            continue
        m = {k: r[k] for k in ("pool", "fam", "entry", "exit", "prio", "filt", "max_hold")}
        _, s = PE.evaluate(D, m, "dev")
        if any(np.corrcoef(s, z)[0, 1] > 0.8 for z in ser):
            continue
        r["rho_h136"] = PE.monthly_corr_h136(D, s, "dev")
        sel.append(r)
        ser.append(s)
        PE.show(r, f"｜和 H136 月相關 {r['rho_h136']:+.2f}｜逐年超額 {r['by_year']}")
        if len(sel) == 10:
            break
    SL.log(dict(kind="strategy", period="dev", strategy=NAME, note="第六十九批 H181 事前登記", chosen=D["chosen"], pool_mean=g.round(3).to_dict(), fit_dev=fit,
                n_pass=len(ok), selected=[{k: r[k] for k in ("pool", "fam", "entry", "exit", "prio", "filt", "max_hold")} for r in sel], extra=sel,
                all=[{k: r[k] for k in ("name", "alpha_t", "sharpe", "yrs_win")} for r in rows],
                trials_cum=TRIALS_FULL if D["chosen"] else TRIALS_SHORT))


def val():
    rec = [json.loads(l) for l in open(H.LOG) if f'"{NAME}"' in l and '"H181' in l and '"period": "dev"' in l][-1]
    D = load("val")
    out = []
    if rec["fit_dev"] is not None:
        M, rows = run(D, "val")
        g = pd.DataFrame(rows).groupby("pool").alpha_t.mean()
        fit = bool(g["A2"] > g["B2"])
        print(f"搭配預測 val（A′ {g['A2']:.2f} vs B′ {g['B2']:.2f}，全部模型平均；只用來判定搭配）：{'成立' if fit else '不成立'} → 搭配{'成立' if fit and rec['fit_dev'] else '不成立'}")
    if not rec["selected"]:
        print("dev 沒有候選，依登記不跑 val 的策略判定")
        return
    for m in rec["selected"]:
        r, s = PE.evaluate(D, m, "val")
        r["name"] = mname(m)
        r["rho_h136"] = PE.monthly_corr_h136(D, s, "val")
        r["verdict"] = "✓" if r["alpha_t"] >= 2 and r["sharpe"] >= 1.0 else "✗"
        out.append(r)
        PE.show(r, f"｜和 H136 月相關 {r['rho_h136']:+.2f}｜逐年超額 {r['by_year']} → {r['verdict']}")
    print(f"val 通過 {sum(r['verdict'] == '✓' for r in out)}／{len(out)}")
    SL.log(dict(kind="strategy", period="val", strategy=NAME, note="第六十九批 H181 dev 選出的", result=out))


if __name__ == "__main__":
    warnings.simplefilter("ignore", RuntimeWarning)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "persist"
    if cmd == "cache":
        cache(sys.argv[2])
    else:
        {"persist": persist, "dev": dev, "val": val}[cmd]()
