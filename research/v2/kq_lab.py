"""第七十三批 H186：Kullamägi 領漲股整理突破（大盤濾網 → 領漲股 → 整理型態 → 突破買 → 當日低點停損、按風險算部位
→ 第 3 天後賣一半、跌破 10 日線出場）。規則見 HYPOTHESES.md 第七十三批（事前登記）。

用法：.venv/bin/python research/v2/kq_lab.py leak
      .venv/bin/python research/v2/kq_lab.py dev
      .venv/bin/python research/v2/kq_lab.py val
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
import strategy_lab as SL  # noqa: E402
import vpool_lab as V  # noqa: E402
import xs  # noqa: E402

RES = H.D / "results"
BUY, SELL = 0.001425, 0.004425
TOP, LB, BASE_MIN, RUN_LB, RUN_MIN = 0.02, 40, 10, 60, 1.30
RISK, CAP = 0.005, 0.20
SEG = {"dev": ("2011-01-01", "2019-12-31"), "val": ("2020-01-01", "2023-12-31")}
NAME = "第七十三批 H186 Kullamägi 整理突破"
TRIALS = 1701717
MODELS = {f"{f}{v}": dict(fill=f, var=v) for f in ("I", "N") for v in range(4)}
VLAB = {0: "主規則", 1: "無大盤濾網", 2: "20日線出場", 3: "無領漲股條件"}


# ═════════ 訊號 ═════════
def setup(P, cut=None):
    """回傳 dict：L（突破線，設定成立才有值，否則 NaN；兩版：有／無領漲股條件）、指標陣列。所有值只用該列（含）以前的資料。"""
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
    # 領漲股：21／63／126 日報酬任一項池內前 2%
    lead = np.zeros(C.shape, bool)
    for n in (21, 63, 126):
        r = (Cf / Cf.shift(n) - 1).where(elig)
        lead |= (r.rank(axis=1, ascending=False, pct=True) <= TOP).values
    ma10, ma20 = Cf.rolling(10, min_periods=10).mean(), Cf.rolling(20, min_periods=20).mean()
    dr = Hn / Ln - 1
    adr10 = dr.rolling(10, min_periods=8).mean()
    adr_prev20 = dr.shift(10).rolling(20, min_periods=15).mean()
    adr20 = dr.rolling(20, min_periods=15).mean()
    trend = (Cf > ma20) & (ma10 > ma20) & (ma10 > ma10.shift(5)) & (ma20 > ma20.shift(5))
    pre = (trend & (adr10 < adr_prev20) & elig).values
    Hv, Lv = Hn.values, Ln.values
    Lfill = Ln.ffill().values
    Lline = np.full(C.shape, np.nan)
    ti, ji = np.nonzero(pre)
    for t, j in zip(ti, ji):
        if t < LB + RUN_LB:
            continue
        w = Hv[t - LB + 1:t + 1, j]
        if not np.isfinite(w).any():
            continue
        k = int(np.nanargmax(w))
        p = t - LB + 1 + k
        if t - p < BASE_MIN:                       # 最高價要在 10 個交易日以前
            continue
        L = Hv[p, j]
        lo = np.nanmin(Lfill[p - RUN_LB:p + 1, j])
        if not (np.isfinite(lo) and lo > 0 and L / lo >= RUN_MIN):
            continue
        b = Lv[p + 1:t + 1, j]
        h = len(b) // 2
        if not (np.nanmin(b[h:]) >= np.nanmin(b[:h])):  # 後半最低 ≥ 前半最低
            continue
        Lline[t, j] = L
    tr = pd.read_parquet(H.D / "taiex_tr.parquet")
    tr["date"] = pd.to_datetime(tr.date)
    mkt = tr.drop_duplicates("date").set_index("date").price.astype(float).sort_index().reindex(days).ffill()
    m10, m20 = mkt.rolling(10).mean(), mkt.rolling(20).mean()
    mf = ((m10 > m20) & (m10 > m10.shift(5)) & (m20 > m20.shift(5))).values
    tv = trade.values
    lastv = np.array([np.flatnonzero(tv[:, j]).max() if tv[:, j].any() else -1 for j in range(len(cols))])
    return dict(days=days, cols=cols, O=O.values, H=Hv, L=Lv, C=Cf.values, trade=tv, LU=LU.fillna(False).values,
                LD=LD.fillna(False).values, ma10=ma10.values, ma20=ma20.values, adr20=adr20.values,
                r63=(Cf / Cf.shift(63) - 1).values, Lall=Lline, Llead=np.where(lead, Lline, np.nan), lead=lead,
                mf=mf, mkt=mkt.values, lastv=lastv)


def cache(period):
    t0 = time.time()
    P = xs.load_panel(H.PERIODS[period][1])
    D = setup(P)
    with open(RES / f"kq_{period}.pkl", "wb") as f:
        pickle.dump(D, f)
    print(f"快取 {period} {time.time()-t0:.0f}s；設定成立（含領漲股）{np.isfinite(D['Llead']).sum()} 格、全池 {np.isfinite(D['Lall']).sum()} 格", flush=True)
    return D


def load(period):
    f = RES / f"kq_{period}.pkl"
    return pickle.load(open(f, "rb")) if f.exists() else cache(period)


def leak():
    P = xs.load_panel(H.PERIODS["dev"][1])
    full = setup(P)
    for cut in ("2013-06-28", "2016-03-31", "2018-09-28"):
        part = setup(P, cut)
        n = len(part["days"])
        msg = []
        for k in ("Llead", "Lall"):                  # 減資修正會整體縮放價位，所以比遮罩和「突破線 ÷ 收盤」
            a, b = full[k][:n], part[k]
            fa, fb = np.isfinite(a), np.isfinite(b)
            both = fa & fb
            dr = np.abs(a[both] / full["C"][:n][both] - b[both] / part["C"][both]).max() if both.any() else 0.0
            msg.append(f"{k} 不同 {(fa != fb).sum()}／{fb.sum()} 格、比例差 {dr:.1e}")
        print(f"截斷 {cut}：{'；'.join(msg)}；大盤濾網不同 {(full['mf'][:n] != part['mf']).sum()} 天", flush=True)


# ═════════ 模擬 ═════════
def simulate(D, m, seg):
    days = D["days"]
    t0 = int(np.searchsorted(days.values, np.datetime64(SEG[seg][0])))
    t1 = int(np.searchsorted(days.values, np.datetime64(SEG[seg][1]), side="right")) - 1
    O, Hh, Lw, C, trade, LU, LD, lastv = D["O"], D["H"], D["L"], D["C"], D["trade"], D["LU"], D["LD"], D["lastv"]
    v, fill = m["var"], m["fill"]
    Lset = D["Lall"] if v == 3 else D["Llead"]
    trail = D["ma20"] if v == 2 else D["ma10"]
    use_mf = v != 1
    cash, nav_prev = 1.0, 1.0
    pos = []        # dict(j, sh, sh0, e, stop, t_in, half, pend, ph, risk, pnl)
    nav = np.zeros(t1 - t0 + 1)
    inv = np.zeros(t1 - t0 + 1)
    trades, n_sig, n_skip = [], 0, 0

    def close_out(p, sh, px, t):
        nonlocal cash
        cash += sh * px * (1 - SELL)
        p["pnl"] += sh * (px * (1 - SELL) - p["e"] * (1 + BUY))
        p["sh"] -= sh
        if p["sh"] <= 1e-12:
            trades.append(dict(j=p["j"], t_in=p["t_in"], t_out=t, ret=p["pnl"] / (p["sh0"] * p["e"] * (1 + BUY)),
                               R=p["pnl"] / p["risk"] if p["risk"] > 0 else np.nan))
            return True
        return False

    def buy(j, e, stop, t):
        nonlocal cash
        d = (e - stop) / e
        val = min(RISK * nav_prev / max(d, 1e-9), CAP * nav_prev, cash / (1 + BUY))
        if val <= 1e-6:
            return None
        sh = val / e
        cash -= val * (1 + BUY)
        p = dict(j=j, sh=sh, sh0=sh, e=e, stop=stop, t_in=t, half=False, pend=False, ph=False, risk=sh * (e - stop), pnl=0.0)
        pos.append(p)
        return p

    for i, t in enumerate(range(t0, t1 + 1)):
        keep = []
        for p in pos:                                                   # 1. 開盤：昨天收盤決定的賣單
            j = p["j"]
            done = False
            if (p["pend"] or p["ph"]) and trade[t, j] and not LD[t, j]:
                if p["pend"]:
                    done = close_out(p, p["sh"], O[t, j], t)
                else:
                    close_out(p, p["sh"] / 2, O[t, j], t)
                    p["half"], p["ph"], p["stop"] = True, False, max(p["stop"], p["e"])
            if not done and trade[t, j] and not LD[t, j] and Lw[t, j] <= p["stop"]:   # 2. 盤中停損
                done = close_out(p, p["sh"], min(O[t, j], p["stop"]), t)
            if not done:
                keep.append(p)
        pos = keep
        held = {p["j"] for p in pos}
        # 3. 進場
        if fill == "I":
            Ls = Lset[t - 1]
            idx = np.flatnonzero(np.isfinite(Ls) & (Hh[t] > Ls) & trade[t])
            if use_mf:
                idx = idx if D["mf"][t - 1] else idx[:0]
            idx = idx[np.lexsort((idx, -np.nan_to_num(D["r63"][t - 1, idx], nan=-9)))]
            for j in idx:
                if j in held or LU[t, j]:
                    continue
                n_sig += 1
                e = max(O[t, j], Ls[j])
                st = Lw[t, j]
                if (e - st) / e > D["adr20"][t - 1, j]:
                    n_skip += 1
                    continue
                p = buy(j, e, st, t)
                if p is None:
                    continue
                held.add(j)
                if C[t, j] < e:                                         # 假突破：當天以最低價停損
                    close_out(p, p["sh"], st, t)
                    pos.remove(p)
        else:
            Ls = Lset[t - 2]
            idx = np.flatnonzero(np.isfinite(Ls) & (C[t - 1] > Ls) & trade[t - 1] & trade[t])
            if use_mf:
                idx = idx if D["mf"][t - 1] else idx[:0]
            idx = idx[np.lexsort((idx, -np.nan_to_num(D["r63"][t - 1, idx], nan=-9)))]
            for j in idx:
                if j in held or LU[t, j] or not np.isfinite(O[t, j]):
                    continue
                n_sig += 1
                e, st = O[t, j], Lw[t - 1, j]
                if e <= st or (e - st) / e > D["adr20"][t - 1, j]:
                    n_skip += 1
                    continue
                p = buy(j, e, st, t)
                if p is None:
                    continue
                held.add(j)
                if Lw[t, j] <= st:                                      # 當天就停損
                    close_out(p, p["sh"], st, t)
                    pos.remove(p)
        # 4. 收盤計值、下市
        keep = []
        for p in pos:
            j = p["j"]
            if t >= lastv[j] and close_out(p, p["sh"], C[t, j], t):
                continue
            keep.append(p)
        pos = keep
        mv = sum(p["sh"] * C[t, p["j"]] for p in pos)
        nav[i] = cash + mv
        inv[i] = mv / nav[i]
        nav_prev = nav[i]
        # 5. 收盤判斷 → 明天開盤
        for p in pos:
            j = p["j"]
            if not trade[t, j]:
                continue
            if C[t, j] < trail[t, j]:
                p["pend"] = True
            elif not p["half"] and t - p["t_in"] >= 3:
                p["ph"] = True
    d = days[t0:t1 + 1]
    return pd.Series(nav, index=d), pd.Series(inv, index=d), trades, n_sig, n_skip


def evaluate(D, name, seg):
    m = MODELS[name]
    nav, inv, trades, n_sig, n_skip = simulate(D, m, seg)
    s = nav.pct_change().fillna(nav.iloc[0] - 1.0).values
    h = pd.Series(D["mkt"], index=D["days"]).reindex(nav.index).pct_change().fillna(0.0).values
    k = np.isfinite(s) & np.isfinite(h)
    at, a0, b1 = BL.alpha_t(s, h, k, 5)
    sd = s[k].std(ddof=1)
    sh = s[k].mean() / sd * np.sqrt(252) if sd > 0 else np.nan
    hd = h[k].std(ddof=1)
    ys = (1 + pd.Series(s, index=nav.index)).groupby(nav.index.year).prod() - 1
    yh = (1 + pd.Series(h, index=nav.index)).groupby(nav.index.year).prod() - 1
    T = pd.DataFrame(trades)
    R = T.R.dropna().values if len(T) else np.array([])
    pnl = T.ret.values if len(T) else np.array([])
    srt = np.sort(R)[::-1]
    top10 = srt[: max(1, len(srt) // 10)].sum() / srt[srt > 0].sum() if len(srt) and (srt > 0).any() else np.nan
    eq = nav / nav.cummax() - 1
    r = dict(model=name, name=f"{name} {VLAB[m['var']]}（{'盤中突破' if m['fill'] == 'I' else '收盤確認、隔天開盤'}）", alpha_t=at, alpha_ann=a0 * 252, beta=b1,
             sharpe=sh, mkt_sharpe=h[k].mean() / hd * np.sqrt(252), cagr=float(nav.iloc[-1] ** (252 / len(nav)) - 1), mdd=float(eq.min()),
             yrs_win=float((ys > yh).mean()), yrs=f"{int((ys > yh).sum())}/{len(ys)}",
             by_year={int(y): round(float(ys[y]) * 100, 1) for y in ys.index}, mkt_year={int(y): round(float(yh[y]) * 100, 1) for y in yh.index},
             exposure=float(inv.mean()), n_trades=len(T), n_sig=n_sig, skip=n_skip / n_sig if n_sig else np.nan,
             win=float((pnl > 0).mean()) if len(pnl) else np.nan, tr_mean=float(pnl.mean()) if len(pnl) else np.nan,
             tr_med=float(np.median(pnl)) if len(pnl) else np.nan, R_mean=float(R.mean()) if len(R) else np.nan,
             R_med=float(np.median(R)) if len(R) else np.nan, R_q90=float(np.quantile(R, 0.9)) if len(R) else np.nan,
             R_max=float(R.max()) if len(R) else np.nan, top10_share=float(top10),
             hold=float((T.t_out - T.t_in).mean()) if len(T) else np.nan)
    return r, s, T


def passes(r):
    return np.isfinite(r["alpha_t"]) and r["alpha_t"] >= 2 and r["sharpe"] >= 1.0 and r["yrs_win"] >= 7 / 9 - 1e-9


def monthly_corr_h136(D, s, period):
    led = pd.read_parquet(RES / "sleeves" / period / "H136.parquet")
    nav = (1 + pd.Series(s, index=D["days"][D["days"].slice_indexer(*SEG[period])])).cumprod()
    px = [nav[nav.index < d].iloc[-1] if (nav.index < d).any() else np.nan for d in led.index]
    mr = pd.Series(px, index=led.index).pct_change().shift(-1)
    k = mr.notna() & led.ret.notna()
    return float(np.corrcoef(mr[k], led.ret[k])[0, 1]) if k.sum() > 12 else np.nan


def show(r):
    print(f"{r['name']}\n  alpha t {r['alpha_t']:+.2f}（年化 {r['alpha_ann']*100:+.1f}%、β {r['beta']:.2f}）Sharpe {r['sharpe']:.2f}（大盤 {r['mkt_sharpe']:.2f}）"
          f" 年化 {r['cagr']*100:+.1f}% MDD {r['mdd']*100:.0f}%｜贏大盤年 {r['yrs']}｜持股率 {r['exposure']:.0%}｜和 H136 月相關 {r.get('rho_h136', np.nan):+.2f}\n"
          f"  訊號 {r['n_sig']}（停損太遠放棄 {r['skip']:.0%}）→ {r['n_trades']} 筆｜勝率 {r['win']:.0%}｜每筆 平均 {r['tr_mean']*100:+.2f}% 中位 {r['tr_med']*100:+.2f}%"
          f"｜R 平均 {r['R_mean']:+.2f} 中位 {r['R_med']:+.2f} 90分位 {r['R_q90']:+.1f} 最大 {r['R_max']:+.1f}｜前 10% 交易佔獲利 {r['top10_share']:.0%}｜持有 {r['hold']:.1f} 日\n"
          f"  逐年 {r['by_year']}", flush=True)


def run(period):
    D = load(period)
    print(f"{period} {SEG[period]}；大盤濾網開的日子 {D['mf'][D['days'].slice_indexer(*SEG[period])].mean():.0%}", flush=True)
    rows, ex = [], {}
    for name in MODELS:
        r, s, T = evaluate(D, name, period)
        r["rho_h136"] = monthly_corr_h136(D, s, period)
        r["pass"] = passes(r)
        rows.append(r)
        ex[name] = T
        show(r)
    print(f"大盤年報酬 {rows[0]['mkt_year']}")
    main_ok = all(r["pass"] for r in rows if r["model"] in ("I0", "N0"))
    by = {r["model"]: r for r in rows}
    for f in "IN":
        print(f"預測 {f}：濾網 Sharpe {by[f+'0']['sharpe']:.2f} vs 無 {by[f+'1']['sharpe']:.2f} → {'成立' if by[f+'0']['sharpe'] > by[f+'1']['sharpe'] else '不成立'}；"
              f"領漲股 alpha t {by[f+'0']['alpha_t']:+.2f} vs 無 {by[f+'3']['alpha_t']:+.2f} → {'成立' if by[f+'0']['alpha_t'] > by[f+'3']['alpha_t'] else '不成立'}")
    print(f"\n判定（I0、N0 都過三條）：{'✓ 過' if main_ok else '✗'}")
    T = ex["I0"]
    if len(T):
        T = T.assign(stock=D["cols"][T.j.values], d_in=D["days"][T.t_in.values], d_out=D["days"][T.t_out.values])
        print("I0 R 最大 10 筆：\n" + T.sort_values("R", ascending=False).head(10)[["stock", "d_in", "d_out", "ret", "R"]].round(3).to_string(index=False))
        T.to_csv(RES / f"kq_{period}_I0_trades.csv", index=False)
    SL.log(dict(kind="strategy", period=period, strategy=NAME, note="第七十三批 事前登記", passed=main_ok, result=rows, trials_cum=TRIALS))
    return rows


if __name__ == "__main__":
    warnings.simplefilter("ignore", RuntimeWarning)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "dev"
    if cmd == "leak":
        leak()
    elif cmd == "cache":
        cache(sys.argv[2])
    elif cmd == "val":
        rec = [json.loads(l) for l in open(H.LOG) if f'"{NAME}"' in l and '"period": "dev"' in l]
        if not rec or not rec[-1]["passed"]:
            sys.exit("dev 沒過，依登記不跑 val")
        run("val")
    else:
        run("dev")
