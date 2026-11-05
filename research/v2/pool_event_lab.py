"""第六十八批 H179：池 × 事件模型的搭配（10 格資金）。規則見 HYPOTHESES.md 第六十八批（事前登記）。
池：A 趨勢性格（週報酬自我相關高）、B 回歸性格（低）、C 營收成長。模型：順勢族 16、逆勢族 12，× 濾網 3。

用法：.venv/bin/python research/v2/pool_event_lab.py cache dev|val
      .venv/bin/python research/v2/pool_event_lab.py dev
      .venv/bin/python research/v2/pool_event_lab.py val
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
import strategy_lab as SL  # noqa: E402
import vpool_lab as V  # noqa: E402
import xs  # noqa: E402

RES = H.D / "results"
K, BUY, SELL = 10, 0.001425, 0.004425
AC_WIN, AC_MIN, TOPQ = 104, 80, 0.2
SEG = {"dev": ("2011-01-01", "2019-12-31"), "val": ("2020-01-01", "2023-12-31")}
POOLS = ("A", "B", "C")
T_ENTRY, T_EXIT, T_MAX = ("T1", "T2", "T3", "T4"), ("X1", "X2", "X3", "X4"), 120
R_ENTRY, R_EXIT, R_MAX = ("R1", "R2", "R3"), ("Y1", "Y2", "Y3", "Y4"), 20
FILTS = ("F0", "F1", "F2")
HOLD = {"X1": 20, "X2": 60, "Y3": 10, "X5": 120}
LABEL = {"T1": "250日新高", "T2": "20日突破", "T3": "20/60均線黃金交叉", "T4": "20日突破＋量增2倍", "R1": "RSI(2)<10", "R2": "布林下軌",
         "R3": "RSI(14)<30", "X1": "持有20日", "X2": "持有60日", "X3": "跌破20日線", "X4": "吊燈3ATR", "Y1": "站上5日線", "Y2": "RSI(2)>70",
         "Y3": "持有10日", "Y4": "站上20日線", "F0": "無濾網", "F1": "大盤>200日線", "F2": "個股>200日線",
         "X5": "持有120日", "X6": "吊燈3ATR(最長250日)", "X7": "跌破60日線", "P1": "20日報酬排序", "P2": "營收年增排序"}
NAME = "第六十八批 H179 池×事件模型"
TRIALS = 1701529


# ═════════ 資料 ═════════
def _rsi(C, n):
    d = C.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn.where(dn > 0))


def cache(period):
    import feature_library as FL
    end = H.PERIODS[period][1]
    p0 = H.PERIODS[period][0]
    t0 = time.time()
    P = xs.load_panel(end, fundamentals=True)
    cols = P["close"].columns
    dates, Rdf, Edf = FL.forward(P, p0, end)
    R = Rdf.reindex(index=dates, columns=cols).values
    E = Edf.reindex(index=dates, columns=cols).fillna(False).values.astype(bool) & np.isfinite(R)
    fx, trade, _ = V.gap_fix(P, cols)
    LU, LD, _, _ = V.limit_flags(fx)
    days = P["close"].index
    print(f"載入 {time.time()-t0:.0f}s；{len(days)} 個交易日、{len(dates)} 個換股日", flush=True)
    # ── 池 ──
    Cw = fx["close"].resample("W-FRI").last()
    rw = Cw.pct_change(fill_method=None)
    ac = rw.rolling(AC_WIN, min_periods=AC_MIN).corr(rw.shift(1))
    ac52 = rw.rolling(52, min_periods=40).corr(rw.shift(1))
    wk = ac.index.values
    ACm = np.full(E.shape, np.nan)
    persist = []
    for t, d in enumerate(dates):
        k = np.searchsorted(wk, np.datetime64(d), side="left") - 1     # 換股日之前最後一個完整的週
        if k < 0:
            continue
        ACm[t] = ac.iloc[k].values
        if k + 52 < len(wk):                                           # 描述：之後 52 週的自我相關
            fut = ac52.iloc[k + 52].values
            m = E[t] & np.isfinite(ACm[t]) & np.isfinite(fut)
            if m.sum() > 100:
                persist.append((d, pd.Series(ACm[t][m]).rank().corr(pd.Series(fut[m]).rank())))
    rk = pd.DataFrame(np.where(E & np.isfinite(ACm), ACm, np.nan)).rank(axis=1, pct=True).values
    pos6 = P["rev_pos6"].shift(1).reindex(index=dates, columns=cols).values
    yoy3 = P["rev_yoy3"].shift(1).reindex(index=dates, columns=cols).values
    with np.errstate(invalid="ignore"):
        PM = {"A": E & (rk > 1 - TOPQ), "B": E & (rk <= TOPQ), "C": E & (pos6 >= 5) & (yoy3 > 0)}
    U = np.flatnonzero(PM["A"].any(axis=0) | PM["B"].any(axis=0) | PM["C"].any(axis=0))
    uc = cols[U]
    kd = np.searchsorted(dates.values, days.values, side="right") - 1
    daily = {}
    for p, M in PM.items():
        x = np.zeros((len(days), len(U)), bool)
        ok = kd >= 0
        x[ok] = M[kd[ok]][:, U]
        daily[p] = x
    # ── 價格與指標（還原、修正後；指標用 ffill 收盤）──
    O = fx["open"][uc]
    Hh, Lw, Cc = fx["high"][uc], fx["low"][uc], fx["close"][uc]
    tr_ok = trade[uc]
    Cf = Cc.ffill()
    Hf, Lf = Hh.fillna(Cf), Lw.fillna(Cf)
    Vol = P["volume"][uc].astype(float).where(tr_ok).fillna(0.0)
    MA = {n: Cf.rolling(n, min_periods=n).mean() for n in (5, 20, 60, 200)}
    sd20 = Cf.rolling(20, min_periods=20).std(ddof=0)
    rsi2, rsi14 = _rsi(Cf, 2), _rsi(Cf, 14)
    hi20 = Hf.rolling(20, min_periods=20).max().shift(1)
    mx250 = Cf.rolling(250, min_periods=250).max()
    prevC = Cf.shift(1)
    trr = np.fmax(np.fmax(Hf - Lf, (Hf - prevC).abs()), (Lf - prevC).abs())
    atr = trr.rolling(22, min_periods=22).mean()
    brk = Cf > hi20
    ent = {"T1": (Cf >= mx250) & ~(prevC >= mx250.shift(1)),
           "T2": brk & ~brk.shift(1, fill_value=False),
           "T3": (MA[20] > MA[60]) & ~(MA[20].shift(1) > MA[60].shift(1)),
           "R1": rsi2 < 10, "R2": Cf < MA[20] - 2 * sd20, "R3": rsi14 < 30}
    ent["T4"] = ent["T2"] & (Vol > 2 * Vol.rolling(20, min_periods=20).mean().shift(1))
    ex = {"X3": Cf < MA[20], "Y1": Cf > MA[5], "Y2": rsi2 > 70, "Y4": Cf > MA[20]}
    tr = pd.read_parquet(H.D / "taiex_tr.parquet")
    tr["date"] = pd.to_datetime(tr.date)
    mkt = tr.drop_duplicates("date").set_index("date").price.astype(float).sort_index().reindex(days).ffill()
    lastv = np.array([np.flatnonzero(tr_ok[c].values).max() if tr_ok[c].values.any() else -1 for c in uc])
    D = dict(days=days, cols=uc, dates=dates, period=period, persist=persist,
             O=O.values.astype(np.float64), H=Hf.values, C=Cf.values, trade=tr_ok.values, LU=LU[uc].fillna(False).values, LD=LD[uc].fillna(False).values,
             atr=atr.values, ret20=(Cf / Cf.shift(20) - 1).values, rsi2=rsi2.values, lastv=lastv,
             ent={k: (v & tr_ok).fillna(False).values for k, v in ent.items()}, ex={k: v.fillna(False).values for k, v in ex.items()},
             f2=(Cf > MA[200]).values, mkt=mkt.values, f1=(mkt > mkt.rolling(200, min_periods=200).mean()).values, pool=daily,
             pool_n={p: M.sum(axis=1) for p, M in PM.items()}, E_n=E.sum(axis=1))
    with open(RES / f"pool_event_{period}.pkl", "wb") as f:
        pickle.dump(D, f)
    pv = np.array([x for _, x in persist])
    print(f"快取完成 {time.time()-t0:.0f}s；股票 {len(U)} 檔；池每月平均檔數 A {PM['A'].sum(1).mean():.0f}、B {PM['B'].sum(1).mean():.0f}、"
          f"C {PM['C'].sum(1).mean():.0f}（流動池 {E.sum(1).mean():.0f}）；性格延續（等級相關）平均 {pv.mean() if len(pv) else float('nan'):+.3f}"
          f"（{len(pv)} 個月，> 0 的比例 {(pv > 0).mean() if len(pv) else float('nan'):.0%}）", flush=True)


def load(period):
    with open(RES / f"pool_event_{period}.pkl", "rb") as f:
        return pickle.load(f)


# ═════════ 模擬 ═════════
def models():
    out = []
    for p in POOLS:
        for fam, ens, exs, mh in (("T", T_ENTRY, T_EXIT, T_MAX), ("R", R_ENTRY, R_EXIT, R_MAX)):
            for e, x, f in itertools.product(ens, exs, FILTS):
                out.append(dict(pool=p, fam=fam, entry=e, exit=x, filt=f, max_hold=mh))
    return out


def mname(m):
    return f"{m['pool']}池 {m['fam']} {LABEL[m['entry']]}→{LABEL[m['exit']]}（{LABEL[m['filt']]}）"


def simulate(D, m, seg):
    days = D["days"]
    t0 = int(np.searchsorted(days.values, np.datetime64(SEG[seg][0])))
    t1 = int(np.searchsorted(days.values, np.datetime64(SEG[seg][1]), side="right")) - 1
    O, Hh, C, trade, LU, LD, lastv = D["O"], D["H"], D["C"], D["trade"], D["LU"], D["LD"], D["lastv"]
    sig = D["ent"][m["entry"]] & D["pool"][m["pool"]]
    if m.get("filt") == "F1":
        sig = sig & D["f1"][:, None]
    elif m.get("filt") == "F2":
        sig = sig & D["f2"]
    prio = D["prio"][m["prio"]] if "prio" in m else (-D["ret20"] if m["fam"] == "T" else D["rsi2"])
    exs = D["ex"].get(m["exit"])
    hold_n = HOLD.get(m["exit"])
    chand = m["exit"] in ("X4", "X6")
    nk = m.get("K", K)
    cash = np.full(nk, 1.0 / nk)
    held = np.full(nk, -1)
    bpx, bday, hh = np.zeros(nk), np.zeros(nk, int), np.zeros(nk)
    pend = np.zeros(nk, bool)
    cand = np.array([], int)
    nav, inv, trades = np.zeros(t1 - t0 + 1), np.zeros(t1 - t0 + 1), []
    for i, t in enumerate(range(t0, t1 + 1)):
        for k in np.flatnonzero(pend):                                # 1. 開盤賣
            j = held[k]
            if LD[t, j] or not trade[t, j] or not np.isfinite(O[t, j]):
                continue
            r = (1 - BUY) * O[t, j] / bpx[k] * (1 - SELL)
            trades.append((j, bday[k], t, r - 1))
            cash[k] *= r
            held[k], pend[k] = -1, False
        free = list(np.flatnonzero(held < 0))                         # 2. 開盤買
        hs = set(held[held >= 0].tolist())
        for j in cand:
            if not free:
                break
            if j in hs or LU[t, j] or not trade[t, j] or not np.isfinite(O[t, j]) or O[t, j] <= 0:
                continue
            k = free.pop(0)
            held[k], bpx[k], bday[k], hh[k] = j, O[t, j], t, Hh[t, j]
            hs.add(j)
        val = cash.copy()                                             # 3. 收盤計值
        for k in np.flatnonzero(held >= 0):
            j = held[k]
            hh[k] = max(hh[k], Hh[t, j])
            val[k] = cash[k] * (1 - BUY) * C[t, j] / bpx[k]
            if t >= lastv[j]:                                         # 下市（最後一個交易日）：以收盤出場
                r = (1 - BUY) * C[t, j] / bpx[k] * (1 - SELL)
                trades.append((j, bday[k], t, r - 1))
                cash[k] *= r
                val[k] = cash[k]
                held[k], pend[k] = -1, False
        nav[i] = val.sum()
        inv[i] = (val[held >= 0]).sum() / nav[i]
        for k in np.flatnonzero((held >= 0) & ~pend):                 # 4. 收盤判斷出場 → 明天開盤賣
            j, hd = held[k], t + 1 - bday[k]
            if (hold_n and hd >= hold_n) or hd >= m["max_hold"] or (exs is not None and exs[t, j]) \
                    or (chand and np.isfinite(D["atr"][t, j]) and C[t, j] < hh[k] - 3 * D["atr"][t, j]):
                pend[k] = True
        nfree = int((held < 0).sum() + pend.sum())                    # 5. 收盤判斷進場 → 明天開盤買
        if nfree:
            idx = np.flatnonzero(sig[t])
            if len(idx):
                pr = np.nan_to_num(prio[t, idx], nan=np.inf)
                cand = idx[np.lexsort((idx, pr))][: nfree + 20]
            else:
                cand = np.array([], int)
        else:
            cand = np.array([], int)
    d = days[t0:t1 + 1]
    return pd.Series(nav, index=d), pd.Series(inv, index=d), trades


def evaluate(D, m, seg):
    nav, inv, trades = simulate(D, m, seg)
    s = nav.pct_change().fillna(nav.iloc[0] - 1.0).values
    mk = pd.Series(D["mkt"], index=D["days"]).reindex(nav.index)
    h = mk.pct_change().fillna(0.0).values
    k = np.isfinite(s) & np.isfinite(h)
    at, a0, b1 = BL.alpha_t(s, h, k, 5)
    sd = s[k].std(ddof=1)
    sh = s[k].mean() / sd * np.sqrt(252) if sd > 0 else np.nan
    ys = (1 + pd.Series(s, index=nav.index)).groupby(nav.index.year).prod() - 1
    yh = (1 + pd.Series(h, index=nav.index)).groupby(nav.index.year).prod() - 1
    tr = np.array([x[3] for x in trades]) if trades else np.array([])
    hd = np.array([x[2] - x[1] for x in trades]) if trades else np.array([])
    eq = nav / nav.cummax() - 1
    return dict(name=m.get("label") or mname(m), **m, alpha_t=at, alpha_ann=a0 * 252, beta=b1, sharpe=sh, cagr=float(nav.iloc[-1] ** (252 / len(nav)) - 1),
                mdd=float(eq.min()), yrs_win=float((ys > yh).mean()), yrs=f"{int((ys > yh).sum())}/{len(ys)}",
                by_year={int(y): round(float(ys[y] - yh[y]) * 100, 1) for y in ys.index}, exposure=float(inv.mean()),
                n_trades=len(tr), tr_mean=float(tr.mean()) if len(tr) else np.nan, tr_med=float(np.median(tr)) if len(tr) else np.nan,
                win=float((tr > 0).mean()) if len(tr) else np.nan, hold=float(hd.mean()) if len(hd) else np.nan), s


_G = {}


def _work(i):
    r, _ = evaluate(_G["D"], _G["M"][i], _G["seg"])
    return r


def run_all(D, M, seg, procs=9):
    _G.update(D=D, M=M, seg=seg)
    with mp.get_context("fork").Pool(procs) as pool:
        return pool.map(_work, range(len(M)), chunksize=1)


def passes(r):
    return np.isfinite(r["alpha_t"]) and r["alpha_t"] >= 2 and r["sharpe"] >= 1.0 and r["yrs_win"] >= 7 / 9 - 1e-9


def fit_table(rows):
    T = pd.DataFrame(rows)
    g = T.groupby(["pool", "fam"]).alpha_t.mean().unstack()
    return g


def monthly_corr_h136(D, s, period):
    led = pd.read_parquet(RES / "sleeves" / period / "H136.parquet")
    nav = (1 + pd.Series(s, index=D["days"][D["days"].slice_indexer(*SEG[period])])).cumprod()
    px = [nav[nav.index < d].iloc[-1] if (nav.index < d).any() else np.nan for d in led.index]
    mr = pd.Series(px, index=led.index).pct_change().shift(-1)
    k = mr.notna() & led.ret.notna()
    return float(np.corrcoef(mr[k], led.ret[k])[0, 1]) if k.sum() > 12 else np.nan


def show(r, extra=""):
    print(f"  {r['name']}：alpha t {r['alpha_t']:+.2f}（年化 {r['alpha_ann']*100:+.1f}%、β {r['beta']:.2f}）Sharpe {r['sharpe']:.2f} 年化 {r['cagr']*100:+.1f}% MDD {r['mdd']*100:.0f}%"
          f"｜贏大盤年 {r['yrs']}｜持股率 {r['exposure']:.0%}｜{r['n_trades']} 筆 平均 {r['tr_mean']*100:+.2f}% 中位 {r['tr_med']*100:+.2f}% 勝率 {r['win']:.0%} 持有 {r['hold']:.0f} 日{extra}", flush=True)


def dev():
    t0 = time.time()
    D = load("dev")
    M = models()
    print(f"dev {SEG['dev']}；模型 {len(M)} 個；池每月平均檔數 A {D['pool_n']['A'].mean():.0f}、B {D['pool_n']['B'].mean():.0f}、C {D['pool_n']['C'].mean():.0f}", flush=True)
    pv = np.array([x for _, x in D["persist"]])
    print(f"描述：性格延續（過去 104 週自我相關 vs 之後 52 週，橫斷面等級相關）平均 {pv.mean():+.3f}、中位 {np.median(pv):+.3f}、> 0 的月份 {(pv > 0).mean():.0%}（{len(pv)} 個月）", flush=True)
    rows = run_all(D, M, "dev")
    print(f"模擬 {time.time()-t0:.0f}s", flush=True)
    g = fit_table(rows)
    print("搭配表（平均 alpha t）：\n" + g.round(2).to_string(), flush=True)
    fit = bool(g.loc["A", "T"] > g.loc["B", "T"] and g.loc["B", "R"] > g.loc["A", "R"])
    print(f"搭配預測（T：A > B 且 R：B > A）dev：{'成立' if fit else '不成立'}", flush=True)
    T = pd.DataFrame(rows).sort_values("alpha_t", ascending=False)
    print("\nalpha t 前 15：", flush=True)
    for _, r in T.head(15).iterrows():
        show(r)
    ok = [r for r in sorted(rows, key=lambda r: -r["alpha_t"]) if passes(r)]
    print(f"\n過 dev 三條：{len(ok)} 個（各池 A {sum(r['pool'] == 'A' for r in ok)}、B {sum(r['pool'] == 'B' for r in ok)}、C {sum(r['pool'] == 'C' for r in ok)}）", flush=True)
    sel, ser = [], []
    for r in ok:
        _, s = evaluate(D, {k: r[k] for k in ("pool", "fam", "entry", "exit", "filt", "max_hold")}, "dev")
        if any(np.corrcoef(s, z)[0, 1] > 0.8 for z in ser):
            continue
        r["rho_h136"] = monthly_corr_h136(D, s, "dev")
        sel.append(r)
        ser.append(s)
        show(r, f"｜和 H136 月相關 {r['rho_h136']:+.2f}｜逐年超額 {r['by_year']}")
        if len(sel) == 10:
            break
    SL.log(dict(kind="strategy", period="dev", strategy=NAME, note="第六十八批 事前登記", fit_table=g.round(3).to_dict(), fit_dev=fit,
                persist=dict(mean=float(pv.mean()), pos=float((pv > 0).mean())), n_pass=len(ok),
                selected=[{k: r[k] for k in ("pool", "fam", "entry", "exit", "filt", "max_hold")} for r in sel], extra=sel,
                all=[{k: r[k] for k in ("name", "alpha_t", "sharpe", "yrs_win")} for r in rows], trials_cum=TRIALS))


def val():
    rec = [json.loads(l) for l in open(H.LOG) if f'"{NAME}"' in l and '"period": "dev"' in l][-1]
    D = load("val")
    M = models()
    rows = run_all(D, M, "val")
    g = fit_table(rows)
    print("搭配表 val（平均 alpha t，全部 252 個；只用來判定搭配）：\n" + g.round(2).to_string())
    fit = bool(g.loc["A", "T"] > g.loc["B", "T"] and g.loc["B", "R"] > g.loc["A", "R"])
    print(f"搭配預測 val：{'成立' if fit else '不成立'}（dev：{'成立' if rec['fit_dev'] else '不成立'}）→ 搭配{'成立' if fit and rec['fit_dev'] else '不成立'}")
    by = {mname(r): r for r in rows}
    out = []
    print(f"\n選進 val 的 {len(rec['selected'])} 個：")
    for m in rec["selected"]:
        r = by[mname(m)]
        _, s = evaluate(D, m, "val")
        r["rho_h136"] = monthly_corr_h136(D, s, "val")
        r["verdict"] = "✓" if r["alpha_t"] >= 2 and r["sharpe"] >= 1.0 else "✗"
        out.append(r)
        show(r, f"｜和 H136 月相關 {r['rho_h136']:+.2f}｜逐年超額 {r['by_year']} → {r['verdict']}")
    print(f"val 通過 {sum(r['verdict'] == '✓' for r in out)}／{len(out)}")
    SL.log(dict(kind="strategy", period="val", strategy=NAME, note="第六十八批 dev 選出的", fit_table=g.round(3).to_dict(), fit_val=fit,
                result=out, all=[{k: r[k] for k in ("name", "alpha_t", "sharpe")} for r in rows]))


if __name__ == "__main__":
    warnings.simplefilter("ignore", RuntimeWarning)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "dev"
    if cmd == "cache":
        cache(sys.argv[2])
    else:
        {"dev": dev, "val": val}[cmd]()
