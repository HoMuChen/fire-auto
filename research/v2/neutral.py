"""第九批：期貨多空中性（H42 距離法配對 / H43 共整合配對 / H44 產業內週反轉 / H45 產業中性 52 週高 / H46 產業中性營收新高）。

權重 W 於收盤 t 決定、次日開盤生效：持有期報酬 = 還原開盤 t+1 → t+2。成本每腳單邊 0.1%（換手 × 0.1%）。
用法：python3 research/v2/neutral.py dev|val [H42 H43 ...]
"""
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402
import hypotheses as Hy  # noqa: E402
import xs  # noqa: E402

LEG_COST, TOPV, FORM, TRADE, NPAIR, NPORT = 0.001, 150, 252, 126, 20, 6
IND_MAP = {"其他電子類": "其他電子業", "金融業": "金融保險", "觀光餐旅": "觀光事業"}
NO_PAIR = {"其他", "其他電子業", ""}


def industries():
    u = pd.read_csv(H.D / "universe.csv", dtype=str).fillna("")
    return u.set_index("stock_id").industry.replace(IND_MAP)


def tradable(P):
    """收盤 t 可交易：pit 流動、≥250 根、當日有收盤、近 60 日均成交金額前 150。"""
    ok = P["liq"].fillna(False).astype(bool) & (P["nbars"] >= 250) & P["close"].notna()
    mv = P["money"].rolling(60, min_periods=40).mean().where(ok)
    return ok & (mv.rank(axis=1, ascending=False) <= TOPV)


def nav(W, P, p0, p1):
    """W：收盤決定的權重（index 交易日）。回傳 p0~p1 日淨值與換手。"""
    O = P["open"].ffill()
    R = (O.shift(-2) / O.shift(-1) - 1).reindex(columns=W.columns).fillna(0)      # 收盤 t 決定 → 開盤 t+1~t+2
    W = W.reindex(O.index).fillna(0)
    pnl = (W * R).sum(axis=1)
    turn = W.diff().abs().sum(axis=1).fillna(W.abs().sum(axis=1))
    r = (pnl - turn * LEG_COST).loc[p0:p1]
    return (1 + r).cumprod(), turn.loc[p0:p1]


# ---------- 配對（H42 / H43） ----------
_G = {}


def _form(i):
    """第 i 個交易日開盤開始交易的一個組合：回傳 [(A, B, 權重A, 權重B, 訊號序列)]。"""
    C, T, ind, method = _G["C"], _G["T"], _G["ind"], _G["method"]
    idx = C.index
    win = C.iloc[i - FORM:i]
    names = [s for s in T.columns[T.iloc[i - 1].values] if ind.get(s, "") not in NO_PAIR and win[s].notna().all()]
    groups = pd.Series({s: ind[s] for s in names}).groupby(lambda s: ind[s]).groups
    cand = []
    for g, ss in groups.items():
        ss = list(ss)
        if len(ss) < 2:
            continue
        if method == "dist":
            nz = (win[ss] / win[ss].iloc[0]).values
            for a in range(len(ss)):
                d = ((nz[:, a:a + 1] - nz[:, a + 1:]) ** 2).sum(axis=0)
                cand += [(d[k], ss[a], ss[a + 1 + k]) for k in range(len(d))]
        else:
            lg = np.log(win[ss].values)
            for a in range(len(ss)):
                for b in range(len(ss)):
                    if a == b:
                        continue
                    y, x = lg[:, a], lg[:, b]
                    beta, alpha = np.polyfit(x, y, 1)
                    if not 0.2 < beta < 5:
                        continue
                    e = y - alpha - beta * x
                    p = adfuller(e, maxlag=1, regression="c", autolag=None)[1]
                    if p < 0.05:
                        cand.append((p, ss[a], ss[b], alpha, beta, e.mean(), e.std()))
    cand.sort(key=lambda t: t[0])
    if method == "coint":                                  # A~B 與 B~A 只留一個
        seen, c2 = set(), []
        for t in cand:
            k = frozenset(t[1:3])
            if k not in seen:
                seen.add(k)
                c2.append(t)
        cand = c2
    out, seg = [], C.iloc[i - 1:i - 1 + TRADE]               # 決策收盤 i-1 … i+124
    for t in cand[:NPAIR]:
        a, b = t[1], t[2]
        pa, pb = seg[a].values, seg[b].values
        if method == "dist":
            sig = (win[a] / win[a].iloc[0] - win[b] / win[b].iloc[0]).std()
            z = (pa / pa[0] - pb / pb[0]) / sig
            wa, wb, stop = 1.0, 1.0, np.inf
        else:
            _, _, _, al, be, mu, sd = t
            z = (np.log(pa) - al - be * np.log(pb) - mu) / sd
            wa, wb, stop = 2 / (1 + be), 2 * be / (1 + be), 4.0
        pos, s, dead = 0, np.zeros(len(z)), False
        for k in range(len(z) - 1):                        # 最後一根強制平倉
            if not np.isfinite(z[k]) or dead:
                pos, dead = 0, True
            elif pos == 0:
                pos = -1 if z[k] > 2 else (1 if z[k] < -2 else 0)
            elif abs(z[k]) > stop:
                pos, dead = 0, True
            elif (pos == -1 and z[k] <= 0) or (pos == 1 and z[k] >= 0):
                pos = 0
            s[k] = pos
        out.append((a, b, wa, wb, pd.Series(s, index=seg.index)))
    return i, out


def pairs_W(P, method, p0, p1):
    C = P["close"]
    idx = C.index
    _G.update(C=C, T=tradable(P), ind=industries(), method=method)
    # 形成日：每月換股日；開始交易前一收盤即可決策
    starts = [i for i in (idx.get_loc(d) for d in xs.rebalance_dates(idx, pd.Timestamp(p0) - pd.DateOffset(months=NPORT - 1), pd.Timestamp(p1))) if i >= FORM]
    with mp.get_context("fork").Pool(8) as pool:
        res = pool.map(_form, starts)
    W = pd.DataFrame(0.0, index=idx, columns=C.columns)
    w = 1 / NPAIR / NPORT / 2                              # 每對總名目 1/120 → 多方合計最多 0.5、空方 0.5（共 1 倍）
    n_open = []
    for i, prs in res:
        for a, b, wa, wb, s in prs:
            W.loc[s.index, a] += s.values * wa * w
            W.loc[s.index, b] -= s.values * wb * w
        n_open.append(sum((s != 0).mean() for *_, s in prs))
    return W, np.mean(n_open)


# ---------- 產業中性（H44 / H45 / H46） ----------
def neutral_W(P, score, dates):
    T, ind = tradable(P), industries()
    W = pd.DataFrame(0.0, index=P["close"].index, columns=P["close"].columns)
    for d in dates:                                        # d = 決策收盤日
        s = score.loc[d][T.loc[d]].dropna()
        g = s.groupby(lambda x: ind.get(x, ""))
        L, S = [], []
        for name, x in g:
            if name in NO_PAIR or len(x) < 5:
                continue
            k = max(1, int(len(x) * 0.3))
            x = x.sort_values(kind="stable")
            S += list(x.index[:k]); L += list(x.index[-k:])
        row = pd.Series(0.0, index=W.columns)
        if L:
            row[L], row[S] = 0.5 / len(L), -0.5 / len(S)  # 多 0.5 + 空 0.5 = 名目 1 倍
        W.loc[d] = row
    W[~W.index.isin(dates)] = np.nan                       # 換股日之間持有
    return W.ffill().fillna(0)


def report(name, s, turn, tr, period, extra=None):
    r = s.pct_change().fillna(s.iloc[0] - 1)
    m = r.groupby(r.index.to_period("M")).apply(lambda x: (1 + x).prod() - 1)
    y = r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    wk = pd.concat([r, tr.pct_change()], axis=1, sort=True).dropna().resample("W").apply(lambda x: (1 + x).prod() - 1)
    out = dict(CAGR=s.iloc[-1] ** (1 / yrs) - 1, MDD=(s / s.cummax() - 1).min(), Sharpe=r.mean() / r.std() * np.sqrt(252),
               t=m.mean() / m.std() * np.sqrt(len(m)), pos_years=f"{(y > 0).sum()}/{len(y)}",
               rho=wk.corr().iloc[0, 1], turn_yr=turn.sum() / yrs, **(extra or {}))
    print(f"{name:22s} CAGR {out['CAGR']*100:+6.1f}%  MDD {out['MDD']*100:6.1f}%  Sharpe {out['Sharpe']:5.2f}  "
          f"t {out['t']:5.2f}  正年 {out['pos_years']}  ρ {out['rho']:+.2f}  年換手 {out['turn_yr']:.0f}x"
          + (f"  平均開倉對數 {extra['open_pairs']:.1f}" if extra else ""))
    print("   " + "  ".join(f"{k}:{v*100:+.1f}" for k, v in y.items()))
    with open(H.LOG, "a") as f:
        f.write(json.dumps(dict(ts=time.strftime("%Y-%m-%d %H:%M:%S"), kind="neutral", period=period, strategy=name,
                                note="期貨多空中性 第九批", **out), ensure_ascii=False, default=float) + "\n")


if __name__ == "__main__":
    period = sys.argv[1] if len(sys.argv) > 1 else "dev"
    if period == "holdout":
        raise PermissionError("holdout 未開封")
    want = sys.argv[2:] or ["H42", "H43", "H44", "H45", "H46"]
    p0, p1 = H.PERIODS[period]
    p0 = max(p0, "2011-01-01")
    P = xs.load_panel(p1, fundamentals="H46" in want)
    idx = P["close"].index
    tr = pd.read_parquet(H.D / "taiex_tr.parquet").assign(date=lambda x: pd.to_datetime(x.date)).set_index("date").price
    for h in want:
        t0 = time.time()
        if h in ("H42", "H43"):
            W, no = pairs_W(P, "dist" if h == "H42" else "coint", p0, p1)
            s, turn = nav(W, P, p0, p1)
            report(f"{h} {'距離法' if h == 'H42' else '共整合'}配對", s, turn, tr, period, {"open_pairs": no})
        else:
            if h == "H44":
                ind = industries()
                rel = P["close"] / P["close"].shift(5) - 1
                T = tradable(P)
                g = rel.where(T).T.groupby(lambda x: ind.get(x, "")).transform("mean").T
                sc = -(rel - g)                                          # 近 5 日相對同業弱 → 分數高 → 做多
                pos = np.arange(idx.get_indexer([pd.Timestamp(p0)], method="bfill")[0] - 1, idx.get_loc(idx[idx <= p1][-1]), 5)
                dates = list(idx[pos])
            else:
                sc = Hy.h2_hi52(P) if h == "H45" else P["rev3_high"]
                dates = [idx[idx.get_loc(d) - 1] for d in xs.rebalance_dates(idx, pd.Timestamp(p0), pd.Timestamp(p1))]
            W = neutral_W(P, sc, dates)
            s, turn = nav(W, P, p0, p1)
            report({"H44": "H44 產業內週反轉", "H45": "H45 產業中性52週高", "H46": "H46 產業中性營收新高"}[h], s, turn, tr, period)
        print(f"   ({time.time()-t0:.0f}s)")
