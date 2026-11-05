"""第六十六批 H175：全池特徵組合大搜尋（目標「自己夠好」）。規則見 HYPOTHESES.md 第六十六批（事前登記）。
163 個特徵：單一 × 兩方向、兩兩、三三（方向用特徵庫方向欄）× 前 N ∈ {10, 20} 等權月換股，現股做多。

用法：.venv/bin/python research/v2/combo_search.py cache dev|val
      .venv/bin/python research/v2/combo_search.py dev
      .venv/bin/python research/v2/combo_search.py val
      .venv/bin/python research/v2/combo_search.py cache holdout --unseal；holdout --unseal（使用者同意後；只測 HOLDOUT 的 3 個）
"""
import itertools
import json
import multiprocessing as mp
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bpool_lab as BL  # noqa: E402
import harness as H  # noqa: E402
import portfolio_lab as PL  # noqa: E402
import strategy_lab as SL  # noqa: E402
import vpool_lab as V  # noqa: E402
import xs  # noqa: E402

RES = H.D / "results"
NS, MIN_VALID, MIN_DEV_M, NSEL, RHO_MAX = (10, 20), 50, 72, 20, 0.8
COST = xs.COST
SEG = {"A": ("2011-01-01", "2014-12-31"), "B": ("2015-01-01", "2019-12-31"), "dev": ("2011-01-01", "2019-12-31"),
       "val": ("2020-01-01", "2023-12-31"), "holdout": ("2024-01-01", "2099-12-31")}
# 第六十六批 holdout 登記（使用者 2026-11-05 同意）：只測這 3 個
HOLDOUT = [((("H95", 1), ("H10", 1), ("EQNPO", 1)), 2, 20),
           ((("H2", 1), ("H99", 1), ("H10", 1)), 2, 10),
           ((("H97", 1), ("H10", 1), ("H81", 1)), 2, 20)]
H136 = ((("H2", 1), ("H87", 1), ("H97", 1)), 2, 20)
NAME = "第六十六批 H175 全池特徵組合大搜尋"
TRIALS = 1507152


# ═════════ 資料 ═════════
def cache(period):
    import batch55_s8 as B55
    import feature_library as FL
    end = H.PERIODS[period][1]
    p0 = H.PERIODS[period][0]
    if period == "holdout" and "--unseal" not in sys.argv:
        raise PermissionError("holdout 封存中：使用者同意後才加 --unseal")
    t0 = time.time()
    lab = SL.Lab(end)
    P, cols = lab.P, lab.cols
    dates, R, E = lab.monthly(p0, end)
    R = R.copy()
    days = P["open"].index
    _, _, evm = V.gap_fix(P, cols)
    cpos = {c: i for i, c in enumerate(cols)}
    rd = pd.DatetimeIndex(xs.rebalance_dates(days, pd.Timestamp(p0), pd.Timestamp(end)))
    nfix = 0
    for c, a, b, f in evm:
        t = np.searchsorted(rd.values, b.to_datetime64(), side="left") - 1
        if 0 <= t < len(dates) and dates[t] < b and np.isfinite(R[t, cpos[c]]):
            R[t, cpos[c]] = (1 + R[t, cpos[c]]) / f - 1
            nfix += 1
    U = np.flatnonzero(E.any(axis=0))
    print(f"載入 {time.time()-t0:.0f}s；換股日 {len(dates)}；池內曾出現 {len(U)} 檔；減資停牌修正月報酬 {nfix} 筆", flush=True)
    X, bad = {}, {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i, f in enumerate(FL.FEATS):
            try:
                A = np.asarray(lab.feat(f, dates, E), float)
            except Exception as e:   # noqa: BLE001
                bad[f] = repr(e)[:200]
                continue
            X[f] = np.where(E, A, np.nan)[:, U].astype(np.float32)
            if i % 40 == 0:
                print(f"  特徵 {i+1}/{len(FL.FEATS)}（{time.time()-t0:.0f}s）", flush=True)
    up, dn = lab.limits(dates)
    mtx = B55.mtx_chain(days)
    out = dict(dates=dates, cols=cols[U], E=E[:, U], R=R[:, U], X=X, bad=bad, up=up[:, U], dn=dn[:, U],
               bench=np.array([np.nanmean(R[t][E[t]]) if E[t].any() else np.nan for t in range(len(dates))]),
               mtx=np.array([mtx.loc[d2] / mtx.loc[d] - 1 for d, d2 in zip(rd[:-1], rd[1:])], float))
    with open(RES / f"combo_{period}.pkl", "wb") as f:
        pickle.dump(out, f)
    print(f"快取完成 {time.time()-t0:.0f}s；特徵 {len(X)}；失敗 {list(bad)}", flush=True)
    if period == "holdout":   # 開封前的資料檢查：只看特徵有值的比例，不看報酬
        yr = pd.DatetimeIndex(dates).year
        for f in ("H95", "H10", "EQNPO", "H2", "H99", "H97", "H81", "H87"):
            cov = {int(y): float(np.isfinite(X[f][yr == y])[out["E"][yr == y]].mean()) for y in sorted(set(yr))}
            print(f"  {f} 有值比例：" + "、".join(f"{y} {v:.0%}" for y, v in cov.items()), flush=True)


def load(period):
    with open(RES / f"combo_{period}.pkl", "rb") as f:
        return pickle.load(f)


# ═════════ 引擎 ═════════
class Eng:
    def __init__(self, B):
        self.dates, self.E, self.bench = B["dates"], B["E"], B["bench"]
        self.R = np.nan_to_num(B["R"]).astype(np.float32)
        self.T, self.U = self.E.shape
        self.eps = (np.arange(self.U) * 1e-9).astype(np.float64)   # 同分依代號小→大（cols 已排序）
        sign = BL.lib_sign()
        self.sign = {f: sign.get(f, 1) for f in B["X"]}
        self.P = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for f, A in B["X"].items():
                for s in (1, -1):
                    self.P[(f, s)] = pd.DataFrame(np.where(self.E, s * A, np.nan)).rank(axis=1, pct=True).values.astype(np.float32)

    def score(self, parts, need):
        st = np.stack([self.P[p] for p in parts])
        ok = np.isfinite(st)
        cnt = ok.sum(axis=0)
        S = np.where(ok, st, 0).sum(axis=0, dtype=np.float64) / np.maximum(cnt, 1)
        return np.where((cnt >= need) & self.E, S, np.nan)

    def run(self, S, n):
        key = np.where(np.isfinite(S), S - self.eps, -np.inf)
        valid = np.isfinite(S).sum(axis=1) >= MIN_VALID
        top = np.argpartition(-key, n, axis=1)[:, :n]
        sel = np.zeros((self.T, self.U), bool)
        np.put_along_axis(sel, top, True, axis=1)
        sel[~valid] = False
        prev = np.vstack([np.zeros((1, self.U), bool), sel[:-1]])
        buy = (sel & ~prev).sum(axis=1) / n
        net = (sel * self.R).sum(axis=1) / n - COST * buy
        net = np.where(valid, net, np.nan)
        return net, net - self.bench, sel

    def run_all(self, S):
        """同 run，但一次算完 NS 的每個 N（前 max(N) 名排好序後取前 n）。"""
        key = np.where(np.isfinite(S), S - self.eps, -np.inf)
        valid = np.isfinite(S).sum(axis=1) >= MIN_VALID
        nmax = max(NS)
        top = np.argpartition(-key, nmax, axis=1)[:, :nmax]
        top = np.take_along_axis(top, np.argsort(-np.take_along_axis(key, top, axis=1), axis=1, kind="stable"), axis=1)
        rt = np.take_along_axis(self.R, top, axis=1)
        out = []
        for n in NS:
            tn = top[:, :n]
            sel = np.zeros((self.T, self.U), bool)
            np.put_along_axis(sel, tn, True, axis=1)
            sel[~valid] = False
            prev = np.vstack([np.zeros((1, self.U), bool), sel[:-1]])
            buy = (sel & ~prev).sum(axis=1) / n
            net = np.where(valid, rt[:, :n].mean(axis=1) - COST * buy, np.nan)
            out.append((net, net - self.bench))
        return out


def mask(dates, seg):
    return np.asarray((dates >= pd.Timestamp(SEG[seg][0])) & (dates <= pd.Timestamp(SEG[seg][1])))


def lib_combos(eng, k):
    fs = list(eng.sign)
    return [tuple((f, eng.sign[f]) for f in c) for c in itertools.combinations(fs, k)]


def spec_name(parts, n):
    return "＆".join(f"{f}{'+' if s > 0 else '−'}" for f, s in parts) + f"/N{n}"


_G = {}


def _stats(ex, net, m, yrs):
    e, x = ex[m], net[m]
    k = np.isfinite(e)
    e, x, y = e[k], x[k], yrs[m][k]
    if len(e) < MIN_DEV_M:
        return (np.nan, np.nan, np.nan, np.nan, len(e))
    yr = np.bincount(y, weights=e)
    has = np.bincount(y, minlength=len(yr)) > 0
    pos = (yr[has] > 0).sum() / has.sum()
    return (e.mean() / e.std(ddof=1) * np.sqrt(len(e)), x.mean() / x.std(ddof=1) * np.sqrt(12), e.mean() * 12, pos * 9, len(e))


def _work(chunk):
    eng, m, yrs = _G["eng"], _G["m"], _G["yrs"]
    out = np.full((len(chunk) * len(NS), 5), np.nan, np.float32)
    for i, (parts, need) in enumerate(chunk):
        for j, (net, ex) in enumerate(eng.run_all(eng.score(parts, need))):
            out[i * len(NS) + j] = _stats(ex, net, m, yrs)
    return out


def scan(eng, specs, m, yrs, procs=9, size=400):
    _G.update(eng=eng, m=m, yrs=yrs)
    chunks = [specs[i:i + size] for i in range(0, len(specs), size)]
    with mp.get_context("fork").Pool(procs) as pool:
        parts = pool.map(_work, chunks, chunksize=1)
    return np.vstack(parts)   # 每個 spec × NS 一列：t、Sharpe、年化超額、正年（換算 9 年制）、月數


def all_specs(eng):
    fs = list(eng.sign)
    singles = [(((f, s),), 1) for f in fs for s in (1, -1)]
    pairs = [(c, 2) for c in lib_combos(eng, 2)]
    triples = [(c, 2) for c in lib_combos(eng, 3)]
    return singles, pairs, triples


def passes_v(r):
    with np.errstate(invalid="ignore"):
        return np.isfinite(r[:, 0]) & (r[:, 0] >= 2) & (r[:, 1] >= 1.0) & (r[:, 3] >= 7 - 1e-6) & (r[:, 4] >= MIN_DEV_M)


def detail(eng, B, parts, need, n, seg, segs):
    S = eng.score(parts, need)
    net, ex, sel = eng.run(S, n)
    m = mask(eng.dates, seg) & np.isfinite(ex)
    out = dict(name=spec_name(parts, n), t=float(ex[m].mean() / ex[m].std(ddof=1) * np.sqrt(m.sum())), ex_ann=float(ex[m].mean() * 12),
               sharpe=float(net[m].mean() / net[m].std(ddof=1) * np.sqrt(12)), ann=float(net[m].mean() * 12), months=int(m.sum()))
    yr = pd.Series(ex[m]).groupby(eng.dates[m].year).sum()
    out["yrs"] = f"{int((yr > 0).sum())}/{len(yr)}"
    out["by_year"] = (yr * 100).round(1).to_dict()
    for g in segs:
        mg = mask(eng.dates, g) & np.isfinite(ex)
        out[f"ex_{g}"] = float(ex[mg].mean() * 12)
    out["turnover"] = float((sel[1:] & ~sel[:-1]).sum(axis=1)[m[1:]].mean() / n)
    lim = SL.portfolio(S, np.nan_to_num(B["R"]), eng.E, n, limits=(B["up"], B["dn"]), cost=COST, min_valid=MIN_VALID)
    nl = np.where(np.isfinite(net), lim.net.values, np.nan)
    el = nl - eng.bench
    ml = m & np.isfinite(el)
    out.update(t_lim=float(el[ml].mean() / el[ml].std(ddof=1) * np.sqrt(ml.sum())), ex_ann_lim=float(el[ml].mean() * 12),
               sharpe_lim=float(nl[ml].mean() / nl[ml].std(ddof=1) * np.sqrt(12)))
    mt = B["mtx"]
    mh = ml & np.isfinite(mt)
    bm = np.polyfit(mt[mh], nl[mh], 1)[0]
    hd = nl[mh] - bm * mt[mh]
    out.update(beta=float(bm), hedged_ann=float(hd.mean() * 12), hedged_sharpe=float(hd.mean() / hd.std(ddof=1) * np.sqrt(12)))
    return out, net, nl, el


def h136_series(period, dates):
    led = pd.read_parquet(RES / "sleeves" / period / "H136.parquet")
    return led.ret.reindex(dates).values


def corr(a, b, m):
    k = m & np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[k], b[k])[0, 1]) if k.sum() > 12 else np.nan


# ═════════ 指令 ═════════
def dev():
    t0 = time.time()
    B = load("dev")
    eng = Eng(B)
    m = mask(eng.dates, "dev")
    yrs = (eng.dates.year - 2011).values.clip(0)
    singles, pairs, triples = all_specs(eng)
    print(f"載入＋百分位 {time.time()-t0:.0f}s；特徵 {len(eng.sign)}；單一 {len(singles)}、兩兩 {len(pairs)}、三三 {len(triples)} × N {NS}", flush=True)
    res = {}
    for nm, sp in (("單一", singles), ("兩兩", pairs), ("三三", triples)):
        t1 = time.time()
        res[nm] = scan(eng, sp, m, yrs)
        r = res[nm]
        ok = passes_v(r)
        print(f"  {nm}：{len(r)} 個（{time.time()-t1:.0f}s）；可評 {np.isfinite(r[:, 0]).sum()}；t 最高 {np.nanmax(r[:, 0]):.2f}；"
              f"Sharpe 最高 {np.nanmax(r[:, 1]):.2f}；通過 dev 三條 {ok.sum()}", flush=True)
    rows = []
    for nm, sp in (("單一", singles), ("兩兩", pairs), ("三三", triples)):
        r = res[nm]
        for k in np.flatnonzero(passes_v(r)):
            parts, need = sp[k // len(NS)]
            rows.append((nm, parts, need, NS[k % len(NS)], *r[k]))
    T = pd.DataFrame(rows, columns=["kind", "parts", "need", "n", "t", "sharpe", "ex_ann", "pos9", "months"]).sort_values("t", ascending=False)
    print(f"\n通過 dev 三條（超額 t ≥ 2、正年 ≥ 7/9、Sharpe ≥ 1）共 {len(T)} 個（單一 {(T.kind=='單一').sum()}、兩兩 {(T.kind=='兩兩').sum()}、三三 {(T.kind=='三三').sum()}）", flush=True)
    h = h136_series("dev", eng.dates)
    sel, sel_net, n_clone, best_clone = [], [h], 0, None
    for _, r in T.iterrows():
        S = eng.score(r.parts, r.need)
        net, ex, _ = eng.run(S, r.n)
        rhos = [corr(net, z, m) for z in sel_net]
        if max(rhos) > RHO_MAX:
            if rhos[0] > RHO_MAX:
                n_clone += 1
                best_clone = best_clone or spec_name(r.parts, r.n)
            continue
        sel.append((r.kind, r.parts, r.need, r.n))
        sel_net.append(net)
        if len(sel) == NSEL:
            break
    print(f"去重：H136 分身（ρ > {RHO_MAX}）跳過 {n_clone} 個（最好的是 {best_clone}）；選進 val {len(sel)} 個", flush=True)
    extra = []
    for kind, parts, need, n in sel:
        d, net, nl, el = detail(eng, B, parts, need, n, "dev", ("A", "B"))
        d.update(kind=kind, rho_h136=corr(nl, h, m))
        extra.append(d)
        print(f"  [{kind}] {d['name']}：超額 t {d['t']:.2f}（年化 {d['ex_ann']*100:+.1f}%；漲跌停處理後 {d['t_lim']:.2f}）A {d['ex_A']*100:+.1f}% B {d['ex_B']*100:+.1f}% "
              f"正年 {d['yrs']}｜年化 {d['ann']*100:+.1f}% Sharpe {d['sharpe']:.2f}｜換手 {d['turnover']:.0%}｜和 H136 {d['rho_h136']:+.2f}"
              f"｜避險版 {d['hedged_ann']*100:+.1f}%（Sharpe {d['hedged_sharpe']:.2f}）", flush=True)
    allt = np.concatenate([res[k][:, 0] for k in res])
    top = T.head(15)
    print("\n  dev 超額 t 前 15（不去重）：" + "；".join(f"{spec_name(p, n)} {t:.2f}" for p, n, t in zip(top.parts, top.n, top.t)))
    T.assign(name=[spec_name(p, n) for p, n in zip(T.parts, T.n)]).drop(columns=["parts"]).to_csv(RES / "batch66_dev_pass.csv", index=False)
    SL.log(dict(kind="strategy", period="dev", strategy=NAME, note="第六十六批 事前登記（目標：自己夠好）",
                n_specs=int(sum(len(v) for v in res.values())), n_eval=int(np.isfinite(allt).sum()), best_t=float(np.nanmax(allt)),
                n_pass=len(T), n_clone_h136=n_clone, best_clone=best_clone,
                selected=[dict(kind=k, parts=[list(p) for p in parts], need=need, n=int(n)) for k, parts, need, n in sel],
                extra=extra, trials_cum=TRIALS))


def val():
    rec = [json.loads(l) for l in open(H.LOG) if f'"{NAME}"' in l and '"period": "dev"' in l][-1]
    sel = rec["selected"]
    B = load("val")
    eng = Eng(B)
    m = mask(eng.dates, "val")
    h = h136_series("val", eng.dates)
    bm = np.nanmean(eng.bench[m]) * 12
    print(f"val 基準（pit 池等權）年化 {bm*100:+.1f}%；選進 val 的 {len(sel)} 個")
    out, npass = [], 0
    for s in sel:
        parts = tuple((f, int(g)) for f, g in s["parts"])
        d, net, nl, el = detail(eng, B, parts, s["need"], s["n"], "val", ())
        d.update(kind=s["kind"], rho_h136=corr(nl, h, m))
        d["verdict"] = "✓" if d["t_lim"] >= 2 and d["sharpe_lim"] >= 1.0 else "✗"
        npass += d["verdict"] == "✓"
        out.append(d)
        print(f"  [{s['kind']}] {d['name']}：漲跌停處理後 超額 t {d['t_lim']:.2f}（年化 {d['ex_ann_lim']*100:+.1f}%）Sharpe {d['sharpe_lim']:.2f}"
              f"｜未處理 t {d['t']:.2f}｜逐年 {d['by_year']}｜和 H136 {d['rho_h136']:+.2f}｜避險版 {d['hedged_ann']*100:+.1f}%（{d['hedged_sharpe']:.2f}）→ {d['verdict']}")
    print(f"val 通過 {npass}／{len(sel)}（20 個全是雜訊時，預期約 0.5 個碰巧過）")
    SL.log(dict(kind="strategy", period="val", strategy=NAME, note="第六十六批 dev 選出的", result=out, n_pass=npass))


def holdout():
    if "--unseal" not in sys.argv:
        raise PermissionError("holdout 封存中：使用者同意後才加 --unseal")
    B = load("holdout")
    eng = Eng(B)
    m = mask(eng.dates, "holdout")
    print(f"holdout 換股月 {eng.dates[m][0].date()}～{eng.dates[m][-1].date()}（{m.sum()} 個）；基準（pit 池等權）年化 {np.nanmean(eng.bench[m])*12*100:+.1f}%")
    d, _, h, _ = detail(eng, B, *H136, "holdout", ())
    print(f"  對照 H136（同引擎）：超額 t {d['t_lim']:.2f}（年化 {d['ex_ann_lim']*100:+.1f}%）年化 {d['ann']*100:+.1f}% Sharpe {d['sharpe_lim']:.2f}｜逐年 {d['by_year']}")
    out, npass = [dict(d, kind="對照")], 0
    for parts, need, n in HOLDOUT:
        d, net, nl, el = detail(eng, B, parts, need, n, "holdout", ())
        d.update(rho_h136=corr(nl, h, m))
        d["verdict"] = "✓" if d["t_lim"] >= 2 and d["sharpe_lim"] >= 1.0 else "✗"
        npass += d["verdict"] == "✓"
        out.append(d)
        print(f"  {d['name']}：漲跌停處理後 超額 t {d['t_lim']:.2f}（年化 {d['ex_ann_lim']*100:+.1f}%）年化 {d['ann']*100:+.1f}% Sharpe {d['sharpe_lim']:.2f}"
              f"｜未處理 t {d['t']:.2f}｜逐年 {d['by_year']}｜和 H136 {d['rho_h136']:+.2f}｜換手 {d['turnover']:.0%}"
              f"｜避險版 {d['hedged_ann']*100:+.1f}%（{d['hedged_sharpe']:.2f}）→ {d['verdict']}")
    print(f"holdout 通過 {npass}／{len(HOLDOUT)}")
    SL.log(dict(kind="strategy", period="holdout", strategy=NAME, note="第六十六批 holdout（使用者 2026-11-05 同意；事先指定 3 個）", unseal=True,
                result=out, n_pass=npass))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "dev"
    if cmd == "cache":
        cache(sys.argv[2])
    else:
        {"dev": dev, "val": val, "holdout": holdout}[cmd]()
