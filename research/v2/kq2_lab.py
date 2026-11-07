"""第七十四批 H187：突破前先進場（B 整理中買／C 收斂日買）× 領漲股門檻（L2 報酬前 2%／L10 H2 前 10%）。
型態、部位、出場沿用第七十三批 H186（kq_lab.py）。規則見 HYPOTHESES.md 第七十四批（事前登記）。

用法：.venv/bin/python research/v2/kq2_lab.py leak
      .venv/bin/python research/v2/kq2_lab.py dev
      .venv/bin/python research/v2/kq2_lab.py val
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
import bpool_lab as BL  # noqa: E402
import harness as H  # noqa: E402
import kq_lab as K  # noqa: E402
import strategy_lab as SL  # noqa: E402
import vpool_lab as V  # noqa: E402
import xs  # noqa: E402

RES = H.D / "results"
BUY, SELL, RISK, CAP = K.BUY, K.SELL, K.RISK, K.CAP
LB, BASE_MIN, RUN_LB, RUN_MIN, TOP, H2TOP = K.LB, K.BASE_MIN, K.RUN_LB, K.RUN_MIN, K.TOP, 0.10
SEG = K.SEG
NAME = "第七十四批 H187 突破前先進場"
TRIALS = 1701721
MODELS = ("B_L2", "B_L10", "C_L2", "C_L10")
LAB = {"B": "整理中先買（停損整理低點）", "C": "收斂日買（NR7／內包，停損當日低）", "L2": "報酬前 2%", "L10": "H2 前 10%"}


def setup(P, cut=None):
    if cut is not None:
        P = {k: v.loc[:cut] for k, v in P.items()}
    cols = P["close"].columns
    fx, trade, _ = V.gap_fix(P, cols)
    LU, LD, _, _ = V.limit_flags(fx)
    days = fx["close"].index
    O, Hh, Lw, C = fx["open"], fx["high"], fx["low"], fx["close"]
    Cf = C.ffill()
    Hn, Ln = Hh.where(trade), Lw.where(trade)
    elig = (P["liq"].reindex(index=days, columns=cols).fillna(False).astype(bool)
            & (P["nbars"].reindex(index=days, columns=cols) >= 250) & trade)
    l2 = np.zeros(C.shape, bool)
    for n in (21, 63, 126):
        r = (Cf / Cf.shift(n) - 1).where(elig)
        l2 |= (r.rank(axis=1, ascending=False, pct=True) <= TOP).values
    h2 = (Cf / Cf.rolling(252, min_periods=250).max()).where(elig)
    l10 = (h2.rank(axis=1, ascending=False, pct=True, method="min") <= H2TOP).values
    ma10, ma20 = Cf.rolling(10, min_periods=10).mean(), Cf.rolling(20, min_periods=20).mean()
    dr = Hn / Ln - 1
    adr10 = dr.rolling(10, min_periods=8).mean()
    adr_prev20 = dr.shift(10).rolling(20, min_periods=15).mean()
    adr20 = dr.rolling(20, min_periods=15).mean()
    trend = (Cf > ma20) & (ma10 > ma20) & (ma10 > ma10.shift(5)) & (ma20 > ma20.shift(5)) & (Cf >= ma10)
    pre = (trend & (adr10 < adr_prev20) & elig).values & (l2 | l10)
    rng = (Hn - Ln)
    nr7 = (rng <= rng.rolling(7, min_periods=7).min()).values
    inside = ((Hn <= Hn.shift(1)) & (Ln >= Ln.shift(1))).values
    tight = (nr7 | inside) & trade.values
    Hv, Lv = Hn.values, Ln.values
    Lfill = Ln.ffill().values
    Lline, blow = np.full(C.shape, np.nan), np.full(C.shape, np.nan)
    for t, j in zip(*np.nonzero(pre)):
        if t < LB + RUN_LB:
            continue
        w = Hv[t - LB + 1:t + 1, j]
        if not np.isfinite(w).any():
            continue
        p = t - LB + 1 + int(np.nanargmax(w))
        if t - p < BASE_MIN:
            continue
        L = Hv[p, j]
        lo = np.nanmin(Lfill[p - RUN_LB:p + 1, j])
        if not (np.isfinite(lo) and lo > 0 and L / lo >= RUN_MIN):
            continue
        b = Lv[p + 1:t + 1, j]
        h = len(b) // 2
        if not (np.nanmin(b[h:]) >= np.nanmin(b[:h])):
            continue
        Lline[t, j], blow[t, j] = L, np.nanmin(b)
    tr = pd.read_parquet(H.D / "taiex_tr.parquet")
    tr["date"] = pd.to_datetime(tr.date)
    mkt = tr.drop_duplicates("date").set_index("date").price.astype(float).sort_index().reindex(days).ffill()
    m10, m20 = mkt.rolling(10).mean(), mkt.rolling(20).mean()
    mf = ((m10 > m20) & (m10 > m10.shift(5)) & (m20 > m20.shift(5))).values
    tv = trade.values
    lastv = np.array([np.flatnonzero(tv[:, j]).max() if tv[:, j].any() else -1 for j in range(len(cols))])
    ok = np.isfinite(Lline) & mf[:, None]
    # 訊號：停損價（成立才有值）
    stops = {"B_L2": np.where(ok & l2, blow, np.nan), "B_L10": np.where(ok & l10, blow, np.nan),
             "C_L2": np.where(ok & l2 & tight, Lv, np.nan), "C_L10": np.where(ok & l10 & tight, Lv, np.nan)}
    return dict(days=days, cols=cols, O=O.values, H=Hv, L=Lv, C=Cf.values, trade=tv, LU=LU.fillna(False).values,
                LD=LD.fillna(False).values, ma10=ma10.values, adr20=adr20.values, r63=(Cf / Cf.shift(63) - 1).values,
                Lline=Lline, stops=stops, mkt=mkt.values, lastv=lastv)


def load(period):
    f = RES / f"kq2_{period}.pkl"
    if f.exists():
        return pickle.load(open(f, "rb"))
    D = setup(xs.load_panel(H.PERIODS[period][1]))
    pickle.dump(D, open(f, "wb"))
    return D


def leak():
    P = xs.load_panel(H.PERIODS["dev"][1])
    full = setup(P)
    for cut in ("2013-06-28", "2016-03-31", "2018-09-28"):
        part = setup(P, cut)
        n = len(part["days"])
        msg = []
        for k in MODELS:
            a, b = full["stops"][k][:n], part["stops"][k]
            fa, fb = np.isfinite(a), np.isfinite(b)
            both = fa & fb
            dr = np.abs(a[both] / full["C"][:n][both] - b[both] / part["C"][both]).max() if both.any() else 0.0
            msg.append(f"{k} 不同 {(fa != fb).sum()}／{fb.sum()}、比例差 {dr:.0e}")
        print(f"截斷 {cut}：{'；'.join(msg)}", flush=True)


def simulate(D, name, seg):
    days = D["days"]
    t0 = int(np.searchsorted(days.values, np.datetime64(SEG[seg][0])))
    t1 = int(np.searchsorted(days.values, np.datetime64(SEG[seg][1]), side="right")) - 1
    O, Hh, Lw, C, trade, LU, LD, lastv = D["O"], D["H"], D["L"], D["C"], D["trade"], D["LU"], D["LD"], D["lastv"]
    S, ent = D["stops"][name], name[0]
    cash, nav_prev = 1.0, 1.0
    pos, trades = [], []
    nav, inv = np.zeros(t1 - t0 + 1), np.zeros(t1 - t0 + 1)
    n_sig, n_skip = 0, 0

    def close_out(p, sh, px, t):
        nonlocal cash
        cash += sh * px * (1 - SELL)
        p["pnl"] += sh * (px * (1 - SELL) - p["e"] * (1 + BUY))
        p["sh"] -= sh
        if p["sh"] <= 1e-12:
            fut = Hh[p["t_in"]:min(p["t_in"] + 20, len(days)), p["j"]]
            trades.append(dict(j=p["j"], t_in=p["t_in"], t_out=t, ret=p["pnl"] / (p["sh0"] * p["e"] * (1 + BUY)),
                               R=p["pnl"] / p["risk"] if p["risk"] > 0 else np.nan, dist=(p["e"] - p["stop0"]) / p["e"],
                               brk=bool(np.nanmax(fut) > p["Lb"]) if np.isfinite(fut).any() else False))
            return True
        return False

    for i, t in enumerate(range(t0, t1 + 1)):
        keep = []
        for p in pos:                                                   # 開盤賣單、盤中停損
            j, done = p["j"], False
            if (p["pend"] or p["ph"]) and trade[t, j] and not LD[t, j]:
                if p["pend"]:
                    done = close_out(p, p["sh"], O[t, j], t)
                else:
                    close_out(p, p["sh"] / 2, O[t, j], t)
                    p["half"], p["ph"], p["stop"] = True, False, max(p["stop"], p["e"])
            if not done and trade[t, j] and not LD[t, j] and Lw[t, j] <= p["stop"]:
                done = close_out(p, p["sh"], min(O[t, j], p["stop"]), t)
            if not done:
                keep.append(p)
        pos = keep
        held = {p["j"] for p in pos}
        st_row = S[t - 1]                                               # 昨天收盤的訊號 → 今天開盤買
        idx = np.flatnonzero(np.isfinite(st_row) & trade[t])
        idx = idx[np.lexsort((idx, -np.nan_to_num(D["r63"][t - 1, idx], nan=-9)))]
        for j in idx:
            if j in held or LU[t, j] or not np.isfinite(O[t, j]):
                continue
            n_sig += 1
            e, st = O[t, j], st_row[j]
            if e <= st or (ent == "C" and (e - st) / e > D["adr20"][t - 1, j]):
                n_skip += 1
                continue
            val = min(RISK * nav_prev / ((e - st) / e), CAP * nav_prev, cash / (1 + BUY))
            if val <= 1e-6:
                continue
            sh = val / e
            cash -= val * (1 + BUY)
            p = dict(j=j, sh=sh, sh0=sh, e=e, stop=st, stop0=st, t_in=t, half=False, pend=False, ph=False,
                     risk=sh * (e - st), pnl=0.0, Lb=D["Lline"][t - 1, j])
            held.add(j)
            if Lw[t, j] <= st:
                close_out(p, sh, st, t)
            else:
                pos.append(p)
        keep = []
        for p in pos:
            if t >= lastv[p["j"]] and close_out(p, p["sh"], C[t, p["j"]], t):
                continue
            keep.append(p)
        pos = keep
        mv = sum(p["sh"] * C[t, p["j"]] for p in pos)
        nav[i] = cash + mv
        inv[i] = mv / nav[i]
        nav_prev = nav[i]
        for p in pos:
            j = p["j"]
            if not trade[t, j]:
                continue
            if C[t, j] < D["ma10"][t, j]:
                p["pend"] = True
            elif not p["half"] and t - p["t_in"] >= 3:
                p["ph"] = True
    d = days[t0:t1 + 1]
    return pd.Series(nav, index=d), pd.Series(inv, index=d), trades, n_sig, n_skip


def evaluate(D, name, seg):
    nav, inv, trades, n_sig, n_skip = simulate(D, name, seg)
    s = nav.pct_change().fillna(nav.iloc[0] - 1.0).values
    h = pd.Series(D["mkt"], index=D["days"]).reindex(nav.index).pct_change().fillna(0.0).values
    k = np.isfinite(s) & np.isfinite(h)
    at, a0, b1 = BL.alpha_t(s, h, k, 5)
    sd = s[k].std(ddof=1)
    ys = (1 + pd.Series(s, index=nav.index)).groupby(nav.index.year).prod() - 1
    yh = (1 + pd.Series(h, index=nav.index)).groupby(nav.index.year).prod() - 1
    T = pd.DataFrame(trades)
    R, pnl = T.R.dropna().values, T.ret.values
    srt = np.sort(R)[::-1]
    eq = nav / nav.cummax() - 1
    e, l = name.split("_")
    r = dict(model=name, name=f"{name} {LAB[e]}・{LAB[l]}", alpha_t=at, alpha_ann=a0 * 252, beta=b1,
             sharpe=s[k].mean() / sd * np.sqrt(252), mkt_sharpe=h[k].mean() / h[k].std(ddof=1) * np.sqrt(252),
             cagr=float(nav.iloc[-1] ** (252 / len(nav)) - 1), mdd=float(eq.min()), yrs_win=float((ys > yh).mean()),
             yrs=f"{int((ys > yh).sum())}/{len(ys)}", by_year={int(y): round(float(ys[y]) * 100, 1) for y in ys.index},
             mkt_year={int(y): round(float(yh[y]) * 100, 1) for y in yh.index}, exposure=float(inv.mean()),
             n_trades=len(T), n_sig=n_sig, skip=n_skip / n_sig if n_sig else np.nan, win=float((pnl > 0).mean()),
             tr_mean=float(pnl.mean()), tr_med=float(np.median(pnl)), R_mean=float(R.mean()), R_med=float(np.median(R)),
             R_q90=float(np.quantile(R, 0.9)), R_max=float(R.max()),
             top10_share=float(srt[: max(1, len(srt) // 10)].sum() / srt[srt > 0].sum()) if (srt > 0).any() else np.nan,
             hold=float((T.t_out - T.t_in).mean()), brk20=float(T.brk.mean()), dist=float(T.dist.mean()))
    return r, s, T


def show(r):
    K.show(r)
    print(f"  20 日內真的突破 {r['brk20']:.0%}｜平均停損距離 {r['dist']*100:.1f}%", flush=True)


def run(period):
    D = load(period)
    rows = []
    for name in MODELS:
        r, s, T = evaluate(D, name, period)
        r["rho_h136"] = K.monthly_corr_h136(D, s, period)
        r["pass"] = K.passes(r)
        rows.append(r)
        show(r)
        T.assign(stock=D["cols"][T.j.values], d_in=D["days"][T.t_in.values]).to_csv(RES / f"kq2_{period}_{name}_trades.csv", index=False)
    print(f"大盤年報酬 {rows[0]['mkt_year']}")
    print("判定：" + "、".join(f"{r['model']} {'✓' if r['pass'] else '✗'}" for r in rows))
    SL.log(dict(kind="strategy", period=period, strategy=NAME, note="第七十四批 事前登記",
                passed=[r["model"] for r in rows if r["pass"]], result=rows, trials_cum=TRIALS))
    return rows


if __name__ == "__main__":
    warnings.simplefilter("ignore", RuntimeWarning)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "dev"
    if cmd == "leak":
        leak()
    elif cmd == "val":
        rec = [json.loads(l) for l in open(H.LOG) if f'"{NAME}"' in l and '"period": "dev"' in l]
        if not rec or not rec[-1]["passed"]:
            sys.exit("dev 沒有通過的，依登記不跑 val")
        run("val")
    else:
        run("dev")
