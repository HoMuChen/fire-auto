"""第五十一批（STRATEGY_PLAN S3a）：H148＝H140 的有限資金版（K 格 × 同日先後）。規則見 HYPOTHESES.md 第五十一批（事前登記）。

訊號照第三十八批（batch38_breakout.signals 的 E），成交交給 harness.simulate（持有 20 日）；資金照 strategy_lab.slot_book。
用法：.venv/bin/python research/v2/batch51_s3a.py dev|val
"""
import itertools
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch38_breakout as B38  # noqa: E402
import harness as H  # noqa: E402
import strategy_lab as SL  # noqa: E402
import xs  # noqa: E402

DEV, VAL = (B38.P0, H.PERIODS["dev"][1]), ("2020-01-01", "2023-12-31")
MENU = list(itertools.product((10, 30), ("sur", "code")))
HOLD, NPL, TRIALS = 20, 100, 34
OUT = H.D / "results"


def nm(v):
    return f"H148[{v[0]}格/{'依SUR' if v[1] == 'sur' else '依代號'}]"


def load(p0, p1):
    P = xs.load_panel(p1)
    sig = B38.signals(P, p0, p1)
    sur = B38.sur_panel(P)
    C, O = P["close"].ffill(), P["open"]
    return P, sig, sur, C, O, P["close"].index


def key_of(T, sur, v):
    if v[1] == "code":
        return np.zeros(len(T))
    s = sur.shift(1)                                                # 訊號日（進場前一個交易日）的 SUR
    return -np.array([s.at[b, c] if c in s.columns else np.nan for b, c in zip(T.buy, T.sid)], float)


def monthly(nav, inv, mkt, p0, p1):
    """月淨值變化 vs 全曝險池、曝險相同的池。回傳 DataFrame（給 SL.stats）。"""
    m = (nav.index >= pd.Timestamp(p0)) & (nav.index <= pd.Timestamp(p1))
    nav, inv, mk = nav[m], inv[m], mkt.reindex(nav.index[m])
    rm = mk / mk.shift(1) - 1
    match = (1 + (inv.shift(1).fillna(0) * rm).fillna(0)).cumprod()
    me = lambda x: x.groupby(x.index.to_period("M")).last()                        # noqa: E731
    first = lambda x: x.iloc[0]                                                     # noqa: E731
    N, M, X = me(nav), me(mk), me(match)
    net = N.pct_change().fillna(N.iloc[0] / first(nav) - 1)
    bench = M.pct_change().fillna(M.iloc[0] / first(mk) - 1)
    mat = X.pct_change().fillna(X.iloc[0] / first(match) - 1)
    df = pd.DataFrame(dict(net=net.values, bench=bench.values, ex=(net - bench).values, turn=np.nan,
                           n=inv.groupby(inv.index.to_period("M")).mean().values, ex_match=(net - mat).values))
    return df, pd.DatetimeIndex(N.index.to_timestamp())


def approx_trades(sigm, O, C, days, elig, p0, p1):
    """近似交易（安慰劑用）：訊號隔日開盤 → 20 個交易日後開盤，扣成本；同一檔持有中不重複進場。"""
    S = sigm & elig
    Ov, Cv = O.values, C.values
    D = len(days)
    p0i, p1i = days.searchsorted(pd.Timestamp(p0)), days.searchsorted(pd.Timestamp(p1), side="right") - 1
    rows = []
    for j in np.flatnonzero(S.any(axis=0)):
        nxt = -1
        for i in np.flatnonzero(S[:, j]):
            if i < p0i or i > p1i or i < nxt or i + 1 >= D:
                continue
            e = i + 1
            o = Ov[e, j]
            if not np.isfinite(o) or o <= 0:
                continue
            x = min(e + HOLD, D - 1)
            px = Ov[x, j] if np.isfinite(Ov[x, j]) and Ov[x, j] > 0 else Cv[x, j]
            rows.append((C.columns[j], days[e], days[x], px * (1 - H.SELL_FEE - H.SELL_TAX) / (o * (1 + H.BUY_FEE)) - 1))
            nxt = x
    return pd.DataFrame(rows, columns=["sid", "buy", "sell", "ret"])


def run_book(T, K, rule, sur, C, O, days, mkt, p):
    key = key_of(T, sur, (K, rule)) if len(T) else None
    nav, inv, took = SL.slot_book(T, K, C, O, days, key=key)
    df, md = monthly(nav, inv, mkt, *p)
    return df, md, took


def fmt(s):
    return (f"月超額年化 {s['ex']*100:+6.1f}% t {s['t']:5.2f} 正年 {s['yrs']:>4} Sharpe {s['sharpe']:.2f} MDD {s['mdd']*100:6.1f}% "
            f"平均曝險 {s['hold']*100:3.0f}%")


def evaluate(period):
    p = DEV if period == "dev" else VAL
    t0 = time.time()
    P, sig, sur, C, O, days = load(*p)
    mkt = SL.mkt_index(P)
    T = SL.event_trades(P, sig["E"], p[0], p[1], p[1], max_hold=HOLD)
    TC = SL.event_trades(P, sig["CH"], p[0], p[1], p[1], max_hold=HOLD)
    print(f"載入＋模擬 {time.time()-t0:.0f}s；H140 訊號成交 {len(T)} 筆、每筆超額 {T.ex.mean()*100:+.2f}%（中位 {T.ex.median()*100:+.2f}%）", flush=True)
    return P, sig, sur, C, O, days, mkt, T, TC, p


def dev():
    P, sig, sur, C, O, days, mkt, T, TC, p = evaluate("dev")
    rows, ser = [], {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        print(f"\n== dev {p[0]} ~ {p[1]}（月超額 vs 全曝險 pit 池等權；成本已扣）")
        for v in MENU:
            df, md, took = run_book(T, v[0], v[1], sur, C, O, days, mkt, p)
            st = SL.stats(df, md, *p)
            sm = SL.stats(df.assign(ex=df.ex_match), md, *p)
            ok = st["t"] >= 2 and int(st["yrs"].split("/")[0]) >= 7
            rows.append(dict(var=nm(v), K=v[0], rule=v[1], **{f"dev_{k}": x for k, x in st.items()}, ex_match=sm["ex"], t_match=sm["t"],
                             n_took=len(took), n_sig=len(T), took_ex=took.ex.mean(), took_med=took.ex.median(), pass_=ok))
            ser[nm(v)] = df.ex.values
            print(f"  {nm(v):<18} {fmt(st)}｜對曝險相同池 {sm['ex']*100:+.1f}%（t {sm['t']:.2f}）｜成交 {len(took)}/{len(T)} 筆、"
                  f"每筆超額 {took.ex.mean()*100:+.2f}%（中位 {took.ex.median()*100:+.2f}%）→ {'過' if ok else '✗'}", flush=True)
        for K in (10, 30):
            df, md, took = run_book(TC, K, "sur", sur, C, O, days, mkt, p)
            st = SL.stats(df, md, *p)
            print(f"  描述 CH（月曆進場）{K} 格/依SUR    {fmt(st)}｜成交 {len(took)}/{len(TC)} 筆")
        D = pd.DataFrame(rows)

        # 安慰劑：近似交易，訊號與 SUR 的股票用同一個排列打亂
        sigm, surm = sig["E"].values, sur.values
        elig = sig["elig"].values

        def best(sm_, su_):
            Ta = approx_trades(sm_, O, C, days, elig, *p)
            sup = pd.DataFrame(su_, index=days, columns=C.columns)
            return max(SL.stats(*run_book(Ta, K, r, sup, C, O, days, mkt, p)[:2], *p)["t"] for K, r in MENU)
        t1 = time.time()
        real = best(sigm, surm)
        rng = np.random.default_rng(2032)
        pl = []
        for _ in range(NPL):
            perm = rng.permutation(sigm.shape[1])
            pl.append(best(sigm[:, perm], surm[:, perm]))
        pl = np.array(pl)
        p95 = np.percentile(pl, 95)
        print(f"\n== 安慰劑（{NPL} 次，近似交易，菜單最佳 dev t；{time.time()-t1:.0f}s）")
        print(f"  真實（近似交易）{real:.2f}｜安慰劑 50／95 分位 {np.percentile(pl, 50):.2f}／{p95:.2f} → 真實 > 95 分位：{real > p95}")
        m = np.isfinite(np.vstack(list(ser.values()))).all(axis=0)
        srs = [np.nanmean(x[m]) / np.nanstd(x[m], ddof=1) for x in ser.values()]
        D["dsr"] = [SL.deflated_sharpe(ser[k][m], srs, TRIALS) for k in D["var"]]
        D["placebo_p95"], D["real_approx"] = p95, real
        sub = D[D.pass_] if real > p95 else D.iloc[0:0]
        sel = sub.sort_values("dev_t", ascending=False).iloc[0]["var"] if len(sub) else None
        D["selected"] = D["var"] == sel
        print(f"\n== deflated Sharpe（試驗數 {TRIALS}）與 val 選擇")
        for _, r in D.iterrows():
            print(f"  {r['var']:<18} DSR {r.dsr:.3f}{'  ← 進 val' if r.selected else ''}")
    D.to_csv(OUT / "batch51_dev.csv", index=False)
    SL.log(dict(kind="strategy", period="dev", strategy="第五十一批 H148 H140 有限資金版", note="第五十一批 事前登記",
                rows=D.to_dict("records"), placebo_p95=p95, real_approx=real, selected=sel, trials_cum=TRIALS))


def val():
    D = pd.read_csv(OUT / "batch51_dev.csv")
    sel = D[D.selected]
    if not len(sel):
        print("dev 沒有選出任何變體 → 依登記不跑 val")
        return
    r = sel.iloc[0]
    P, sig, sur, C, O, days, mkt, T, TC, p = evaluate("val")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        df, md, took = run_book(T, int(r.K), r.rule, sur, C, O, days, mkt, p)
        st = SL.stats(df, md, *p)
        sm = SL.stats(df.assign(ex=df.ex_match), md, *p)
        ok = st["t"] >= 2 and st["yrs"] == "4/4"
        yr = (df.ex.groupby(md.year).sum() * 100).round(1).to_dict()
        print(f"\n== val {p[0]} ~ {p[1]}")
        print(f"  {r['var']:<18} {fmt(st)}｜對曝險相同池 {sm['ex']*100:+.1f}%（t {sm['t']:.2f}）｜成交 {len(took)}/{len(T)} 筆、"
              f"每筆超額 {took.ex.mean()*100:+.2f}%（中位 {took.ex.median()*100:+.2f}%）｜逐年 {yr} → {'✓' if ok else '✗'}")
        dc, mc, tc = run_book(TC, int(r.K), "sur", sur, C, O, days, mkt, p)
        print(f"  描述 CH（月曆進場）同規則 {fmt(SL.stats(dc, mc, *p))}")
    SL.log(dict(kind="strategy", period="val", strategy="第五十一批 H148 H140 有限資金版", note="第五十一批 dev 選出的",
                result=dict(var=r["var"], **st, ex_match=sm["ex"], by_year=yr, n_took=len(took), verdict="✓" if ok else "✗")))


if __name__ == "__main__":
    {"dev": dev, "val": val}[sys.argv[1] if len(sys.argv) > 1 else "dev"]()
