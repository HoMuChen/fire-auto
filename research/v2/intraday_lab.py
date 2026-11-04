"""第六十五批 H174：台指期日內（當天平倉；做多或空微台）。規則見 HYPOTHESES.md 第六十五批（事前登記）。
資料：data/v2/futures/tx1m（tx1m.py；從 ai-dev 同步）。

用法：.venv/bin/python research/v2/intraday_lab.py dev|val
"""
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402
import portfolio_lab as PL  # noqa: E402
import strategy_lab as SL  # noqa: E402
import tx1m  # noqa: E402

RES = H.D / "results"
NB = 300                                      # 08:45～13:44
SLIP, FEE_TMF, FEE_MTX, TAX = 1.0, 1.5, 0.5, 0.00002
NW_LAG, NPL, WIN, MINP = 5, 100, 250, 120
SEG = {"A": ("2011-01-01", "2014-12-31"), "B": ("2015-01-01", "2019-12-31"), "dev": ("2011-01-01", "2019-12-31"),
       "val": ("2020-01-01", "2023-12-31")}
LOAD = {"dev": ("2010-12-15", "2019-12-31"), "val": ("2019-01-01", "2023-12-31")}
NAME = "第六十五批 H174 台指期日內"
TRIALS = 62972


def ix(hhmm):
    h, m = map(int, hhmm.split(":"))
    return h * 60 + m - (8 * 60 + 45)


# ───────── 資料 ─────────
def build(period):
    p0, p1 = LOAD[period]
    B = tx1m.load(p0, p1)
    B = B[B.volume > 0].copy()
    B["minute"] = pd.to_datetime(B.minute.astype(str).str[:19])
    B["k"] = (B.minute.dt.hour * 60 + B.minute.dt.minute - (8 * 60 + 45)).astype(int)
    B = B[(B.k >= 0) & (B.k < NB)]
    days = sorted(B.day.unique())
    last_close = {}
    O, Hh, L, C = (np.full((len(days), NB), np.nan) for _ in range(4))
    C0, con = np.full(len(days), np.nan), []
    for i, (d, g) in enumerate(B.groupby("day", sort=True)):
        vol = g.groupby("contract").volume.sum().sort_values(ascending=False)
        late = g.groupby("contract").k.max()
        ok = [c for c in vol.index if late[c] >= ix("13:40")]
        c = ok[0] if ok else vol.index[0]
        x = g[g.contract == c].sort_values("k")
        for arr, col in ((O, "open"), (Hh, "high"), (L, "low"), (C, "close")):
            arr[i, x.k.values] = x[col].values
        C0[i] = last_close.get(c, np.nan)
        con.append(c)
        last_close = {cc: gg.sort_values("k").close.iloc[-1] for cc, gg in g.groupby("contract")}
    # 缺的分鐘：收盤往後補；開高低用補過的收盤（沒有成交＝價格不動）
    Cf = pd.DataFrame(C).T.ffill().bfill().T.values
    prevC = np.hstack([Cf[:, :1], Cf[:, :-1]])
    O = np.where(np.isfinite(O), O, prevC)
    Hh = np.where(np.isfinite(Hh), Hh, np.maximum(O, Cf))
    L = np.where(np.isfinite(L), L, np.minimum(O, Cf))
    return dict(days=pd.DatetimeIndex(days), O=O, H=Hh, L=L, C=Cf, C0=C0, contract=con)


def load(period):
    f = RES / f"intraday_{period}.pkl"
    if not f.exists():
        t0 = time.time()
        D = build(period)
        with open(f, "wb") as fh:
            pickle.dump(D, fh)
        print(f"建快取 {period}：{len(D['days'])} 天（{time.time()-t0:.0f}s）；C0 缺 {np.isnan(D['C0']).sum()} 天", flush=True)
    with open(f, "rb") as fh:
        return pickle.load(fh)


# ───────── 菜單 ─────────
def past_med(x, q=0.5):
    """自己過去 WIN 日（不含今天）的分位數。"""
    return pd.Series(x).shift(1).rolling(WIN, min_periods=MINP).quantile(q).values


def price(D, t):
    if t == "C0":
        return D["C0"]
    if t == "收盤":
        return D["C"][:, -1]
    return D["O"][:, ix(t)]


F1_PAIRS = [("C0", "09:15", "13:15", "收盤"), ("08:45", "09:15", "13:15", "收盤"), ("C0", "09:15", "09:15", "收盤"),
            ("12:45", "13:15", "13:15", "收盤"), ("C0", "13:15", "13:15", "收盤"), ("08:45", "11:00", "11:00", "收盤"),
            ("C0", "12:00", "12:00", "收盤"), ("08:45", "09:15", "09:15", "10:15")]


def trades(D):
    """回傳 [(名稱, 族, 方向 dir（+1／−1／0）, 進場價, 出場價)]，每個陣列長度＝天數。"""
    out = []
    with np.errstate(invalid="ignore", divide="ignore"):
        # F1 日內動能
        for a, b, c, d in F1_PAIRS:
            sig = price(D, b) / price(D, a) - 1
            big = np.abs(sig) >= past_med(np.abs(sig))
            en, ex = price(D, c), price(D, d)
            for mode, s in (("順勢", 1), ("逆勢", -1)):
                for fl in ("無", "大訊號"):
                    dr = s * np.sign(np.nan_to_num(sig))
                    if fl == "大訊號":
                        dr = np.where(big, dr, 0)
                    out.append((f"F1 {a}→{b}｜{c}→{d}｜{mode}｜{fl}", "F1", dr, en, ex))
        # F2 開盤區間突破
        O, Hh, L, C = D["O"], D["H"], D["L"], D["C"]
        T = len(D["days"])
        cols = np.arange(NB)[None, :]
        last_entry = ix("13:15")
        for N in (5, 15, 30, 60):
            hi, lo = Hh[:, :N].max(axis=1), L[:, :N].min(axis=1)
            narrow = (hi - lo) / hi <= past_med((hi - lo) / hi)
            up = (Hh > hi[:, None]) & (cols >= N) & (cols <= last_entry)
            dn = (L < lo[:, None]) & (cols >= N) & (cols <= last_entry)
            tu = np.where(up.any(axis=1), up.argmax(axis=1), NB)
            td = np.where(dn.any(axis=1), dn.argmax(axis=1), NB)
            for side in ("兩邊", "只做多", "只做空"):
                if side == "兩邊":
                    t = np.minimum(tu, td)
                    dr = np.where(tu < td, 1, np.where(td < tu, -1, 0))
                elif side == "只做多":
                    t, dr = tu, np.where(tu < NB, 1, 0)
                else:
                    t, dr = td, np.where(td < NB, -1, 0)
                tt = np.minimum(t, NB - 1)
                both = up[np.arange(T), tt] & dn[np.arange(T), tt]
                dr = np.where((t < NB) & ~both, dr, 0)
                en = np.where(dr > 0, np.maximum(O[np.arange(T), tt], hi), np.minimum(O[np.arange(T), tt], lo))
                for xn in ("收盤", "停損另一側", "停損中點"):
                    if xn == "收盤":
                        ex = C[:, -1]
                    else:
                        stop = np.where(dr > 0, lo, hi) if xn == "停損另一側" else (hi + lo) / 2
                        after = cols > tt[:, None]
                        hit = np.where(dr[:, None] > 0, L <= stop[:, None], Hh >= stop[:, None]) & after
                        hs = hit.argmax(axis=1)
                        hx = np.where(dr > 0, np.minimum(O[np.arange(T), hs], stop), np.maximum(O[np.arange(T), hs], stop))
                        ex = np.where(hit.any(axis=1), hx, C[:, -1])
                    for fl in ("無", "窄區間"):
                        d2 = np.where(narrow, dr, 0) if fl == "窄區間" else dr
                        out.append((f"F2 {N}分｜{side}｜{xn}｜{fl}", "F2", d2, en, ex))
        # F3 跳空
        g = O[:, 0] / D["C0"] - 1
        ag = np.abs(g)
        q80, q50 = ag >= past_med(ag, 0.8), ag >= past_med(ag, 0.5)
        for e in ("08:45", "09:00"):
            for x in ("09:15", "10:00", "收盤"):
                for mode, s in (("順勢", 1), ("逆勢", -1)):
                    for cn, cond in (("80分位", q80), ("50分位", q50), ("不限", np.ones(T, bool))):
                        dr = np.where(cond & np.isfinite(g), s * np.sign(np.nan_to_num(g)), 0)
                        out.append((f"F3 {e}→{x}｜{mode}｜{cn}", "F3", dr, price(D, e), price(D, x)))
    return out


def pnl(dr, en, ex, fee):
    """每日報酬（名目 1 倍）：毛、成本、淨。"""
    on = (dr != 0) & np.isfinite(en) & np.isfinite(ex) & (en > 0)
    g = np.where(on, dr * (ex - en) / en, 0.0)
    c = np.where(on, (2 * (SLIP + fee) + TAX * (en + ex)) / en, 0.0)
    return g, c, on


def tstat(x):
    n = len(x)
    if n < 50:
        return np.nan
    m = x.mean()
    y = x - m
    s = (y @ y) / n
    for L in range(1, NW_LAG + 1):
        s += 2 * (1 - L / (NW_LAG + 1)) * (y[L:] @ y[:-L]) / n
    return float(m / np.sqrt(s / n)) if s > 0 else np.nan


def mask(days, seg):
    return np.asarray((days >= pd.Timestamp(SEG[seg][0])) & (days <= pd.Timestamp(SEG[seg][1])))


def h136(r, days, period, segs):
    led = pd.read_parquet(RES / "sleeves" / period / "H136.parquet")
    md = led.index
    k = np.searchsorted(md.values, days.values, side="right") - 1
    g = pd.Series(np.log1p(r)).groupby(k).sum()
    g = g[g.index >= 0]
    x = np.full(len(md), np.nan)
    x[g.index.values] = np.expm1(g.values)
    h = led.ret.values
    seg = "dev" if period == "dev" else "val"
    g3, ok = PL.gate3(x, h, md, segs)
    return dict(rho_h136=PL.corr(x, h, md, seg), worst25=PL.worst25(x, h, md, seg),
                gate3={a: (round(b[0], 2), round(b[1], 2)) for a, b in g3.items()}, gate3_ok=ok)


def detail(spec, days, seg, segs):
    nm, fam, dr, en, ex = spec
    g, c, on = pnl(dr, en, ex, FEE_TMF)
    r = g - c
    m = mask(days, seg)
    out = dict(name=nm, family=fam, t=tstat(r[m]), ann=float(r[m].mean() * 252), sharpe=float(r[m].mean() / r[m].std(ddof=1) * np.sqrt(252)),
               trades=int(on[m].sum()), win=float((r[m & on] > 0).mean()) if (m & on).any() else np.nan,
               per_trade=float(r[m & on].mean() * 100) if (m & on).any() else np.nan,
               per_trade_pts=float(np.nanmean((dr * (ex - en))[m & on])) if (m & on).any() else np.nan)
    nav = np.cumprod(1 + r[m])
    out["mdd"] = float((nav / np.maximum.accumulate(nav) - 1).min())
    gm, cm, _ = pnl(dr, en, ex, FEE_MTX)
    out["t_mtx"] = tstat((gm - cm)[m])
    out["t_cost2"] = tstat((g - 2 * c)[m])
    out["t_gross"] = tstat(g[m])
    for sg in segs:
        ms = mask(days, sg)
        out[f"mean_{sg}"] = float(r[ms].mean() * 252)
    yr = pd.Series(r[m]).groupby(days[m].year).sum()
    out["yrs_pos"] = f"{int((yr > 0).sum())}/{len(yr)}"
    out["by_year"] = (yr * 100).round(1).to_dict()
    return out, r


# ───────── 指令 ─────────
def dev():
    t0 = time.time()
    D = load("dev")
    days = D["days"]
    S = trades(D)
    m = mask(days, "dev")
    G = np.array([pnl(dr, en, ex, FEE_TMF)[0] for _, _, dr, en, ex in S])[:, m]
    Cc = np.array([pnl(dr, en, ex, FEE_TMF)[1] for _, _, dr, en, ex in S])[:, m]
    real = np.array([tstat(g - c) for g, c in zip(G, Cc)])
    print(f"dev {m.sum()} 天、菜單 {len(S)} 個（{time.time()-t0:.0f}s）；t ≥ 2 的 {(real >= 2).sum()} 個，最高 {np.nanmax(real):.2f}；"
          f"毛 t 最高 {max(tstat(g) for g in G):.2f}", flush=True)
    rng = np.random.default_rng(2065)
    pl = []
    for _ in range(NPL):
        e = rng.choice([-1.0, 1.0], m.sum())
        pl.append(np.nanmax([tstat(e * g - c) for g, c in zip(G, Cc)]))
    p95 = float(np.percentile(pl, 95))
    print(f"安慰劑 {NPL} 次：菜單最佳 t 的 50／95 分位 {np.percentile(pl, 50):.2f}／{p95:.2f}", flush=True)
    order = np.argsort(-np.nan_to_num(real, nan=-99))
    print("\n  菜單 t 前 15：")
    for i in order[:15]:
        print(f"   {S[i][0]}  t {real[i]:.2f}（毛 {tstat(G[i]):.2f}）")
    print("  各族最佳：" + "；".join(f"{f} {max(real[i] for i in range(len(S)) if S[i][1] == f):.2f}" for f in ("F1", "F2", "F3")))
    cand = [i for i in order if real[i] >= max(2, p95)]
    print(f"\n== 超過門檻 {len(cand)} 個，檢查 A、B 段 > 0、逐年 ≥ 7/9、交易 ≥ 200")
    rows, sel = [], []
    for i in cand:
        r, _ = detail(S[i], days, "dev", ("A", "B"))
        r["pass"] = r["mean_A"] > 0 and r["mean_B"] > 0 and int(r["yrs_pos"].split("/")[0]) >= 7 and r["trades"] >= 200
        rows.append(r)
    T = pd.DataFrame(rows)
    if len(T):
        for fam, g in T[T["pass"]].groupby("family"):
            b = g.sort_values("t", ascending=False).iloc[0]
            sel.append(b["name"])
    extra = {}
    for nm in sel:
        sp = next(s for s in S if s[0] == nm)
        r, rr = detail(sp, days, "dev", ("A", "B"))
        hc = h136(rr, days, "dev", ("A", "B"))
        extra[nm] = dict(**r, **hc)
        print(f"  [{r['family']}] {nm}：t {r['t']:.2f}（小台成本 {r['t_mtx']:.2f}、成本加倍 {r['t_cost2']:.2f}、毛 {r['t_gross']:.2f}）年化 {r['ann']*100:+.1f}% "
              f"Sharpe {r['sharpe']:.2f} MDD {r['mdd']*100:.0f}%｜A {r['mean_A']*100:+.1f}% B {r['mean_B']*100:+.1f}% 逐年 {r['yrs_pos']} {r['by_year']}"
              f"｜{r['trades']} 筆 勝率 {r['win']:.0%} 每筆 {r['per_trade']:+.3f}%（{r['per_trade_pts']:+.1f} 點）"
              f"｜和 H136 相關 {hc['rho_h136']:+.2f}｜H136 最差 25% 月份 {hc['worst25'][0]*100:+.2f}%｜關卡 3 {hc['gate3']} {'✓' if hc['gate3_ok'] else '✗'}")
    if len(T):
        nf = T[~T["pass"]]
        if len(nf):
            print(f"  沒通過的 {len(nf)} 個：" + "；".join(f"{r['name']} t {r['t']:.2f} A {r['mean_A']*100:+.1f} B {r['mean_B']*100:+.1f} {r['yrs_pos']} {r['trades']} 筆"
                                                  for _, r in nf.head(8).iterrows()))
    pd.DataFrame(dict(name=[s[0] for s in S], family=[s[1] for s in S], t=real)).to_csv(RES / "batch65_menu.csv", index=False)
    SL.log(dict(kind="strategy", period="dev", strategy=NAME, note="第六十五批 事前登記", menu=len(S), n_t2=int((real >= 2).sum()),
                best=float(np.nanmax(real)), placebo_p50=float(np.percentile(pl, 50)), placebo_p95=p95, n_cand=len(cand),
                candidates=T.drop(columns=["by_year"]).to_dict("records") if len(T) else [], selected=sel, extra=extra, trials_cum=TRIALS))


def val():
    rec = [json.loads(l) for l in open(H.LOG) if f'"{NAME}"' in l and '"period": "dev"' in l][-1]
    sel = rec["selected"]
    if not sel:
        print("dev 沒有選出 → 依登記不跑 val")
        return
    D = load("val")
    days = D["days"]
    S = trades(D)
    out = {}
    for nm in sel:
        sp = next(s for s in S if s[0] == nm)
        r, rr = detail(sp, days, "val", ())
        hc = h136(rr, days, "val", ("val",))
        r["verdict"] = "✓" if r["t"] >= 2 else "✗"
        out[nm] = dict(**r, **hc)
        print(f"  [{r['family']}] {nm}：t {r['t']:.2f}（小台成本 {r['t_mtx']:.2f}、成本加倍 {r['t_cost2']:.2f}、毛 {r['t_gross']:.2f}）年化 {r['ann']*100:+.1f}% "
              f"Sharpe {r['sharpe']:.2f} MDD {r['mdd']*100:.0f}%｜逐年 {r['by_year']}｜{r['trades']} 筆 勝率 {r['win']:.0%} 每筆 {r['per_trade']:+.3f}%（{r['per_trade_pts']:+.1f} 點）"
              f"｜和 H136 相關 {hc['rho_h136']:+.2f}｜關卡 3 {hc['gate3']} → {r['verdict']}")
    SL.log(dict(kind="strategy", period="val", strategy=NAME, note="第六十五批 dev 選出的", result=out))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        {"dev": dev, "val": val}[sys.argv[1] if len(sys.argv) > 1 else "dev"]()
