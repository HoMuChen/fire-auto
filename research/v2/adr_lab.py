"""第六十二批 H169：ADR 隔夜資訊 → 台股開盤後的延續／反轉（台積電、聯電、日月光、中華電；個股期貨做多）。
規則見 HYPOTHESES.md 第六十二批（事前登記）。資料：research/v2/fetch_adr.py → data/v2/us/。

用法：.venv/bin/python research/v2/adr_lab.py dev | val
"""
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

US = H.D / "us"
SIDE, ROLL_M, NPL = 0.0005, 0.001, 100
SEG = TL.SEG
SLOTS = {"2330": ("TSM", [("2330", None, None, 5)]), "2303": ("UMC", [("2303", None, None, 5)]),
         "ASE": ("ASX", [("2311", None, "2018-04-17", 5), ("3711", "2018-05-02", None, 2)]), "2412": ("CHT", [("2412", None, None, 10)])}
ASE_GAP = ("2018-03-01", "2018-06-30")
HOLDS = ("開→收", "隔天開盤", "3天", "5天")
TRIALS = 31199
NAME = "第六十二批 H169 ADR 隔夜資訊"


def _tw(sid, kind):
    d = pd.read_parquet(H.D / kind / f"{sid}.parquet", columns=["date", "open", "close"])
    d["date"] = pd.to_datetime(d.date)
    return d.set_index("date")[["open", "close"]].astype(float).replace(0, np.nan)


def load(end):
    days = pd.to_datetime(pd.read_parquet(H.D / "adj" / "2330.parquet", columns=["date"]).date)
    days = pd.DatetimeIndex(days[days <= pd.Timestamp(end)])
    fx = pd.read_parquet(US / "usdtwd.parquet")
    fx = fx.assign(date=pd.to_datetime(fx.date)).set_index("date")
    fx = ((fx.spot_buy.astype(float) + fx.spot_sell.astype(float)) / 2).replace(0, np.nan).reindex(days).ffill()
    cols = list(SLOTS)
    O, C, Cr, K, mem = (pd.DataFrame(np.nan, index=days, columns=cols) for _ in range(5))
    G, Ue, Z = (pd.DataFrame(np.nan, index=days, columns=cols) for _ in range(3))
    for c, (adr, parts) in SLOTS.items():
        a = pd.read_parquet(US / f"{adr}.parquet")
        a = a.assign(date=pd.to_datetime(a.date)).set_index("date").sort_index()
        a = a[a.index <= pd.Timestamp(end)]
        ud = a.index.values
        # d：t′ 之前（嚴格）最後一個美股日的位置
        di = np.searchsorted(ud, days.values, side="left") - 1
        la = np.log(a.Adj_Close.astype(float).values)
        lr = np.log(a.Close.astype(float).values)
        for sid, p0, p1, k in parts:
            adj, raw = _tw(sid, "adj"), _tw(sid, "raw")
            seg = (days >= pd.Timestamp(p0 or "1900-01-01")) & (days <= pd.Timestamp(p1 or "2100-01-01"))
            o, cl = adj.open.reindex(days), adj.close.reindex(days)
            O.loc[seg, c], C.loc[seg, c] = o[seg], cl[seg]
            Cr.loc[seg, c] = raw.close.reindex(days)[seg]
            K.loc[seg, c] = k
            mem.loc[seg, c] = (o.notna() & cl.notna())[seg].astype(float)
        if c == "ASE":
            gap = (days >= pd.Timestamp(ASE_GAP[0])) & (days <= pd.Timestamp(ASE_GAP[1]))
            mem.loc[gap, c] = 0.0
        lo, lc = np.log(O[c].values), np.log(C[c].values)
        ok = di >= 1
        g = np.full(len(days), np.nan)
        # 對 t′（第 i 天）：d ＝ di[i]、d0 ＝ di[i-1]；t 的盤中 ＝ lc[i-1] − lo[i-1]
        i = np.arange(1, len(days))
        same = di[i] == di[i - 1]
        gi = la[np.clip(di[i], 0, None)] - la[np.clip(di[i - 1], 0, None)] - (lc[i - 1] - lo[i - 1])
        gi[same | ~ok[i] | ~ok[i - 1]] = np.nan
        gi[np.abs(gi) > 0.2] = np.nan
        g[1:] = gi
        G[c] = g
        u = np.full(len(days), np.nan)
        u[1:] = g[1:] - (lo[1:] - lc[:-1])
        Ue[c] = u
        # S3 溢價水準：ADR 原始收盤_d × 匯率_t ／（股數 × 台股原始收盤_t），t ＝ t′ 前一日
        prem = np.full(len(days), np.nan)
        prem[1:] = lr[np.clip(di[1:], 0, None)] + np.log(fx.values[:-1]) - np.log(K[c].values[:-1] * Cr[c].values[:-1])
        prem[1:][~ok[1:]] = np.nan
        ps = pd.Series(prem, index=days)
        zz = pd.Series(np.nan, index=days)
        for sid, p0, p1, k in parts:   # 日月光換股前後分開算
            seg = (days >= pd.Timestamp(p0 or "1900-01-01")) & (days <= pd.Timestamp(p1 or "2100-01-01"))
            x = ps[seg]
            zz[seg] = (x - x.rolling(60, min_periods=40).mean()) / x.rolling(60, min_periods=40).std()
        Z[c] = zz.values
    mem = mem.fillna(0).astype(bool)
    return dict(days=days, O=O, C=C, mem=mem, G=G, U=Ue, Z=Z)


class Bk:
    def __init__(self, D):
        O, C = D["O"], D["C"]
        Of = O.fillna(C.shift(1)).ffill()
        self.days, self.mem = D["days"], D["mem"].values
        self.roo = np.nan_to_num(np.where(self.mem, (Of.shift(-1) / Of - 1).values, 0.0))
        self.roc = np.nan_to_num(np.where(self.mem, (C / O - 1).values, 0.0))
        self.n = np.maximum(self.mem.sum(axis=1), 1)
        live = self.mem.sum(axis=1) > 0
        self.h = np.where(live, self.roo.sum(axis=1) / self.n - ROLL_M / 21, np.nan)

    def run(self, ent, hold, side=SIDE):
        """ent[t′]：t′ 開盤前成立（t′ 開盤進場）。回傳日報酬、每格持有矩陣。"""
        ent = ent & self.mem
        if hold == "開→收":
            eff = ent
            s = (eff * (self.roc - 2 * side)).sum(axis=1) / self.n
        else:
            n = {"隔天開盤": 1, "3天": 3, "5天": 5, "10天": 10}[hold]
            eff = TL.pos_hold(ent, n) & self.mem
            tr = np.abs(np.diff(eff.astype(np.int8), axis=0, prepend=0)).sum(axis=1)
            s = ((eff * self.roo).sum(axis=1) - side * tr - ROLL_M / 21 * eff.sum(axis=1)) / self.n
        return np.where(np.isfinite(self.h), s, np.nan), eff


def signals(D):
    ent = {}
    for nm, X in (("S1", D["G"]), ("S2", D["U"])):
        p = TL.own_pct(X)
        for thr, hi, lo in (("分位90", p >= 0.9, p <= 0.1), ("分位80", p >= 0.8, p <= 0.2), ("1%", X > 0.01, X < -0.01), ("2%", X > 0.02, X < -0.02)):
            ent[(nm, thr, "高")] = hi.fillna(False).values
            ent[(nm, thr, "低")] = lo.fillna(False).values
    Z = D["Z"]
    for thr in (1.5, 2.0):
        ent[("S3", f"z{thr}", "高")] = (Z >= thr).fillna(False).values
        ent[("S3", f"z{thr}", "低")] = (Z <= -thr).fillna(False).values
    return ent


def menu(ent):
    M = []
    for key in ent:
        holds = HOLDS if key[0] != "S3" else ("隔天開盤", "3天", "5天", "10天")
        M += [(key, h) for h in holds]
    return M


_G = {}


def _scan(seed):
    bk, ent, M, m = (_G[k] for k in ("bk", "ent", "M", "m"))
    T, N = bk.mem.shape
    if seed is not None:
        rng = np.random.default_rng(2062_000 + seed)
        k = rng.integers(TL.WIN, T - TL.WIN, N)
        ix = (np.arange(T)[:, None] - k[None, :]) % T
        ent = {a: b[ix, np.arange(N)[None, :]] for a, b in ent.items()}
    out = np.full(len(M), np.nan)
    for i, (key, hold) in enumerate(M):
        s, _ = bk.run(ent[key], hold)
        out[i] = TL.alpha_t(s, bk.h, m & np.isfinite(s))[0]
    return out


def _pl(seed):
    return float(np.nanmax(_scan(seed)))


def detail(sp, bk, ent, seg_main, segs):
    key, hold = sp
    s, eff = bk.run(ent[key], hold)
    s2, _ = bk.run(ent[key], hold, side=2 * SIDE)
    d = bk.days
    m = TL.mask(d, *SEG[seg_main]) & np.isfinite(s)
    t, a, b = TL.alpha_t(s, bk.h, m)
    out = dict(spec=f"{key[0]}/{key[1]}/{key[2]}/{hold}", alpha_t=t, alpha_ann=a * 252, beta=b, alpha_t_cost2=TL.alpha_t(s2, bk.h, m)[0],
               sharpe=TL.sharpe(s[m]), sharpe_hold=TL.sharpe(bk.h[m]), expo=float(eff[m].sum(axis=1).mean() / bk.n[m].mean()))
    for g in segs:
        mg = TL.mask(d, *SEG[g]) & np.isfinite(s)
        out[f"sharpe_{g}"], out[f"hold_{g}"] = TL.sharpe(s[mg]), TL.sharpe(bk.h[mg])
    per = {}
    for j, c in enumerate(SLOTS):
        mj = m & bk.mem[:, j]
        r = bk.roc[:, j] - 2 * SIDE if hold == "開→收" else bk.roo[:, j]
        y = eff[mj, j] * r[mj]
        X = np.column_stack([np.ones(mj.sum()), bk.roo[mj, j]])
        per[c] = float(np.linalg.lstsq(X, y, rcond=None)[0][0] * 252 * 100)
    out["per_stock_alpha"] = {k: round(v, 1) for k, v in per.items()}
    out["breadth"] = int(sum(v > 0 for v in per.values()))
    st = np.vstack([np.zeros((1, eff.shape[1]), bool), eff[1:] & ~eff[:-1]]) if hold != "開→收" else eff
    out["entries"] = int(st[m].sum())
    out["alpha_by_year"] = (pd.Series(s[m] - b * bk.h[m]).groupby(d[m].year).sum() * 100).round(1).to_dict()
    return out, s


def align_check(D, seg):
    m = TL.mask(D["days"], *SEG[seg])
    out = {}
    for c in SLOTS:
        g = D["G"][c].values
        gap = np.log(D["O"][c] / D["C"][c].shift(1)).values
        k = m & np.isfinite(g) & np.isfinite(gap) & D["mem"][c].values
        b = np.polyfit(g[k], gap[k], 1)[0]
        r2 = np.corrcoef(g[k], gap[k])[0, 1] ** 2
        out[c] = (round(float(b), 2), round(float(r2), 2), int(k.sum()))
    return out


def h136(s, bk, period, segs):
    import portfolio_lab as PL
    led = pd.read_parquet(H.D / "results" / "sleeves" / period / "H136.parquet")
    md = led.index
    k = np.searchsorted(md.values, bk.days.values, side="right") - 1
    g = pd.Series(np.log1p(np.nan_to_num(s))).groupby(k).sum()
    g = g[g.index >= 0]
    mret = np.full(len(md), np.nan)
    mret[g.index.values] = np.expm1(g.values)
    seg = "dev" if period == "dev" else "val"
    g3, ok3 = PL.gate3(mret, led.ret.values, md, segs)
    return dict(rho_h136=PL.corr(mret, led.ret.values, md, seg), gate3=g3, gate3_ok=ok3)


def setup(period):
    D = load(H.PERIODS[period][1])
    bk = Bk(D)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ent = signals(D)
    return D, bk, ent, menu(ent)


def dev():
    t0 = time.time()
    D, bk, ent, M = setup("dev")
    m = TL.mask(bk.days, *SEG["dev"]) & np.isfinite(bk.h)
    print(f"載入 {time.time()-t0:.0f}s；菜單 {len(M)}；對照組 dev Sharpe {TL.sharpe(bk.h[m]):.2f}")
    print(f"對齊檢查（開盤跳空對 g：斜率、R²、天數）：{align_check(D, 'dev')}", flush=True)
    _G.update(bk=bk, ent=ent, M=M, m=m)
    real = _scan(None)
    with mp.get_context("fork").Pool(8) as pool:
        pl = np.array(pool.map(_pl, range(NPL)))
    p95 = float(np.percentile(pl, 95))
    print(f"alpha t ≥ 2 的 {(real >= 2).sum()} 個，最高 {np.nanmax(real):.2f}；安慰劑 50／95 分位 {np.percentile(pl, 50):.2f}／{p95:.2f}")
    o = np.argsort(-np.nan_to_num(real, nan=-9))
    print("前 15：" + "；".join(f"{M[i][0][0]}/{M[i][0][1]}/{M[i][0][2]}/{M[i][1]} {real[i]:.2f}" for i in o[:15]))
    print("後 5：" + "；".join(f"{M[i][0][0]}/{M[i][0][1]}/{M[i][0][2]}/{M[i][1]} {real[i]:.2f}" for i in o[-5:]))
    rows = []
    for i in [i for i in o if real[i] >= max(2, p95)]:
        r, _ = detail(M[i], bk, ent, "dev", ("A", "B"))
        r["family"] = M[i][0][0]
        r["pass"] = r["sharpe_A"] > r["hold_A"] and r["sharpe_B"] > r["hold_B"] and r["breadth"] >= 3 and r["entries"] >= 100
        rows.append(r)
        print(f"  候選 {r['spec']}：alpha t {r['alpha_t']:.2f}（年化 {r['alpha_ann']*100:+.1f}%）成本加倍 {r['alpha_t_cost2']:.2f}｜Sharpe {r['sharpe']:.2f}（持有 {r['sharpe_hold']:.2f}；"
              f"A {r['sharpe_A']:.2f}/{r['hold_A']:.2f}、B {r['sharpe_B']:.2f}/{r['hold_B']:.2f}）｜個股 alpha {r['per_stock_alpha']}｜進場 {r['entries']}｜曝險 {r['expo']:.0%}"
              f"｜逐年 {r['alpha_by_year']} → {'通過' if r['pass'] else '不過'}")
    T = pd.DataFrame(rows)
    sel, extra = [], {}
    if len(T) and T["pass"].any():
        for fam, g in T[T["pass"]].groupby("family"):
            b = g.sort_values("alpha_t", ascending=False).iloc[0]
            sel.append(b.spec)
    for sp in sel:
        spec = next(x for x in M if f"{x[0][0]}/{x[0][1]}/{x[0][2]}/{x[1]}" == sp)
        r, s = detail(spec, bk, ent, "dev", ("A", "B"))
        hc = h136(s, bk, "dev", ("A", "B"))
        extra[sp] = hc
        print(f"  選 val：{sp}｜和 H136 相關 {hc['rho_h136']:+.2f}｜關卡 3 {hc['gate3']} {'✓' if hc['gate3_ok'] else '✗'}")
    SL.log(dict(kind="strategy", period="dev", strategy=NAME, note="第六十二批 事前登記", menu=len(M), align=align_check(D, "dev"),
                n_t2=int((real >= 2).sum()), best=float(np.nanmax(real)), placebo_p50=float(np.percentile(pl, 50)), placebo_p95=p95,
                menu_t={f"{a[0]}/{a[1]}/{a[2]}/{h}": float(v) for (a, h), v in zip(M, real)}, candidates=T.to_dict("records") if len(T) else [],
                selected=sel, extra=extra, trials_cum=TRIALS))


def val():
    rec = [json.loads(l) for l in open(H.LOG) if f'"{NAME}"' in l and '"period": "dev"' in l][-1]
    if not rec["selected"]:
        print("dev 沒有選出 → 依登記不跑 val")
        return
    D, bk, ent, M = setup("val")
    print(f"對齊檢查 val：{align_check(D, 'val')}")
    out = {}
    for sp in rec["selected"]:
        spec = next(x for x in M if f"{x[0][0]}/{x[0][1]}/{x[0][2]}/{x[1]}" == sp)
        r, s = detail(spec, bk, ent, "val", ("val",))
        ok = r["alpha_t"] >= 2 and r["sharpe"] > r["sharpe_hold"]
        hc = h136(s, bk, "val", ("val",))
        out[sp] = dict(**r, **hc, verdict="✓" if ok else "✗")
        print(f"  {sp}：alpha t {r['alpha_t']:.2f}（年化 {r['alpha_ann']*100:+.1f}%）成本加倍 {r['alpha_t_cost2']:.2f}｜Sharpe {r['sharpe']:.2f}（持有 {r['sharpe_hold']:.2f}）"
              f"｜個股 alpha {r['per_stock_alpha']}｜進場 {r['entries']}｜逐年 {r['alpha_by_year']}｜和 H136 相關 {hc['rho_h136']:+.2f}"
              f"｜關卡 3 {hc['gate3']} → {'✓' if ok else '✗'}")
    SL.log(dict(kind="strategy", period="val", strategy=NAME, note="第六十二批 dev 選出的", result=out))


# ═════════ H170（事後）ADR 壞消息的盤中延續 ═════════
NAME170 = "第六十二批之二 H170 ADR 壞消息盤中延續（事後）"
THR = ("分位10", "分位20", "-1%", "-2%")


def low_mask(X, thr):
    p = TL.own_pct(X)
    c = {"分位10": p <= 0.1, "分位20": p <= 0.2, "-1%": X < -0.01, "-2%": X < -0.02}[thr]
    return c.fillna(False).values


def mtx_intraday(days):
    """小台指日盤（position）開盤到收盤：用當天不結算的最近月。"""
    d = pd.read_parquet(H.D / "futures" / "MTX_daily.parquet")
    d = d[(d.trading_session == "position") & (d.open > 0) & (d.close > 0)].copy()
    d["date"] = pd.to_datetime(d.date)
    d["c"] = d.contract_date.astype(str)
    last = d.groupby("c").date.max()
    op = d.pivot_table(index="date", columns="c", values="open", aggfunc="last").reindex(days)
    cl = d.pivot_table(index="date", columns="c", values="close", aggfunc="last").reindex(days)
    cs = sorted(last.index)
    lastv = last.reindex(cs).values
    r = np.full(len(days), np.nan)
    for i, t in enumerate(days):
        k = next((j for j, c in enumerate(cs) if lastv[j] > t and c in op.columns and np.isfinite(op.at[t, c])), None)
        if k is not None:
            r[i] = cl.at[t, cs[k]] / op.at[t, cs[k]] - 1
    return r


def nw_t(x, lag=5):
    x = x[np.isfinite(x)]
    e = x - x.mean()
    S = (e @ e) / len(x)
    for L in range(1, lag + 1):
        S += 2 * (1 - L / (lag + 1)) * (e[L:] @ e[:-L]) / len(x)
    return float(x.mean() / np.sqrt(S / len(x)))


def h170_one(kind, thr, D, bk, rmtx, seg, segs=()):
    d = bk.days
    m = TL.mask(d, *SEG[seg])
    if kind == "a":
        av = low_mask(D["G"], thr) & bk.mem
        Of = D["O"].fillna(D["C"].shift(1)).ffill()
        roo = (Of.shift(-1) / Of - 1).values
        roc = (D["C"] / D["O"] - 1).values
        night = (1 + roo) / (1 + roc) - 1
        slot = np.where(av, np.nan_to_num(night) - 2 * SIDE, bk.roo)
        s = np.where(bk.mem, slot, 0.0).sum(axis=1) / bk.n - ROLL_M / 21
        s = np.where(np.isfinite(bk.h), s, np.nan)
        mm = m & np.isfinite(s)
        t, a, b = TL.alpha_t(s, bk.h, mm)
        out = dict(spec=f"a/{thr}", alpha_t=t, alpha_ann=a * 252, sharpe=TL.sharpe(s[mm]), sharpe_hold=TL.sharpe(bk.h[mm]),
                   days_avoided=int(av[mm].any(axis=1).sum()), slot_days=int(av[mm].sum()),
                   avoided_oc_ann=float(np.nanmean(np.where(av, roc, np.nan)[mm]) * 100),
                   alpha_by_year=(pd.Series(s[mm] - b * bk.h[mm]).groupby(d[mm].year).sum() * 100).round(1).to_dict())
        for g in segs:
            mg = TL.mask(d, *SEG[g]) & np.isfinite(s)
            out[f"sharpe_{g}"], out[f"hold_{g}"] = TL.sharpe(s[mg]), TL.sharpe(bk.h[mg])
        return out, s
    sig = low_mask(D["G"][["2330"]], thr)[:, 0]
    s = np.where(sig & np.isfinite(rmtx), -rmtx - 2 * SIDE, 0.0)
    s = np.where(np.isfinite(rmtx) | ~sig, s, 0.0)
    mm = m
    x = s[mm]
    out = dict(spec=f"b/{thr}", t=nw_t(x), ann=float(x.mean() * 252), sharpe=TL.sharpe(x), n_days=int(sig[mm].sum()),
               per_trade=float(x[sig[mm]].mean() * 100), hit=float((x[sig[mm]] > 0).mean()),
               by_year=(pd.Series(x).groupby(d[mm].year).sum() * 100).round(1).to_dict())
    for g in segs:
        mg = TL.mask(d, *SEG[g])
        out[f"t_{g}"] = nw_t(s[mg])
    return out, s


def h170(period):
    D = load(H.PERIODS[period][1])
    bk = Bk(D)
    rmtx = mtx_intraday(bk.days)
    if period == "dev":
        res, sel = {}, []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for kind in ("a", "b"):
                rows = [h170_one(kind, thr, D, bk, rmtx, "dev", ("A", "B"))[0] for thr in THR]
                for r in rows:
                    res[r["spec"]] = r
                    print("  " + "｜".join(f"{k} {round(v, 3) if isinstance(v, float) else v}" for k, v in r.items()))
                key = "alpha_t" if kind == "a" else "t"
                sel.append(max(rows, key=lambda r: r[key])["spec"])
        print(f"選 val：{sel}")
        SL.log(dict(kind="strategy", period="dev", strategy=NAME170, note="第六十二批之二 事後登記；dev 只當描述", result=res, selected=sel, trials_cum=31207))
        return
    rec = [json.loads(l) for l in open(H.LOG) if f'"{NAME170}"' in l and '"period": "dev"' in l][-1]
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for sp in rec["selected"]:
            kind, thr = sp.split("/")
            r, s = h170_one(kind, thr, D, bk, rmtx, "val")
            hc = h136(s, bk, "val", ("val",))
            ok = (r["alpha_t"] >= 2 and r["sharpe"] > r["sharpe_hold"]) if kind == "a" else r["t"] >= 2
            out[sp] = dict(**r, **hc, verdict="✓" if ok else "✗")
            print("  " + "｜".join(f"{k} {round(v, 3) if isinstance(v, float) else v}" for k, v in out[sp].items()))
    SL.log(dict(kind="strategy", period="val", strategy=NAME170, note="第六十二批之二 val 判定", result=out))


if __name__ == "__main__":
    if sys.argv[1] == "h170":
        h170(sys.argv[2])
    else:
        {"dev": dev, "val": val}[sys.argv[1]]()
