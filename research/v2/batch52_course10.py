"""第五十二批：「散戶逆襲」課程 10 個選股模型（依名稱猜因子）H149～H158。規則見 HYPOTHESES.md 第五十二批（事前登記）。

訊號＝特徵庫特徵（當日收盤值）在當天 pit 池內的百分位／自然門檻條件全部成立 → 隔日開盤買；
成交、成本、漲跌停、下市交給 harness.simulate（strategy_lab.event_trades）；投組用 strategy_lab.slot_book（10 格）。
用法：.venv/bin/python research/v2/batch52_course10.py leak|dev|val
"""
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch51_s3a as B51  # noqa: E402
import feature_library as FL  # noqa: E402
import harness as H  # noqa: E402
import strategy_lab as SL  # noqa: E402

DEV, VAL = ("2011-01-01", "2019-12-31"), ("2020-01-01", "2023-12-31")
SEG = {"A": ("2011-01-01", "2014-12-31"), "B": ("2015-01-01", "2019-12-31")}
QS = (0.2, 0.1)
EXITS = {"6日": dict(stop=None, max_hold=6), "移動10%": dict(stop=0.10, max_hold=20)}
MODELS = {"H149": "#1 大戶宰羊奔向多頭", "H150": "#2 主力掃貨利多集結", "H151": "#3 大戶法人合流鎖籌",
          "H152": "#4 巨鯨卡位散戶衝浪", "H153": "#5 豹式順流集火破空", "H154": "#6 順風放量破曉起飛",
          "H155": "#7 賺錢金雞延長續航", "H156": "#8 法人定調熱錢追隨", "H157": "#9 鯨群集結積極預備",
          "H158": "#10 法人新歡破冰早鳥"}
FEATS = ["H86", "H85", "MA_ALIGN", "H78", "H87", "H25", "DT_RATIO", "NI_INC8Q", "H2", "H97", "H26", "H24",
         "BIG_LVL", "TRUST_STREAK", "H22"]
PCT = ["H86", "H85", "H78", "H87", "H25", "DT_RATIO", "H97", "H26", "H2", "H24", "BIG_LVL"]
K, NPL, TRIALS, MIN_N, MIN_A = 10, 100, 74, 100, 30
EPS = 1e-12
OUT = H.D / "results"


def nm(h, q, ex):
    return f"{h}[q{q}/{ex}]"


# ───────── 訊號 ─────────
def prep(P):
    """特徵（不 shift，當日收盤值）、當天 pit 池、池內百分位。"""
    F = {f: FL.feature_fn(f)(P).replace([np.inf, -np.inf], np.nan) for f in FEATS}
    el = P["liq"].fillna(False).astype(bool) & (P["nbars"] >= 250)
    pc = {f: F[f].where(el).rank(axis=1, pct=True) for f in PCT}
    return F, el, pc


def signals(F, el, pc, q):
    def top(f):
        return pc[f] >= 1 - q

    def bot(f):
        return pc[f] <= q
    hi = F["H2"] >= 1 - EPS
    ma = F["MA_ALIGN"] >= 10
    up = F["H85"] > 0
    M = {
        "H149": top("H86") & up & ma,
        "H150": top("H78") & top("H87") & top("H86"),
        "H151": up & top("H25") & top("H86"),
        "H152": top("H85") & top("DT_RATIO"),
        "H153": (F["NI_INC8Q"] >= 2) & top("H86") & ma & hi,
        "H154": top("H87") & top("H78") & top("H97"),
        "H155": top("H26") & ma & top("H2"),
        "H156": top("H25") & bot("H24"),
        "H157": top("BIG_LVL") & up & (F["H2"] >= 0.9) & ~hi,
        "H158": (F["TRUST_STREAK"] == 1) & (F["H22"].shift(1) <= 0) & bot("H2"),
    }
    return {k: (v & el).fillna(False).astype(bool) for k, v in M.items()}


def leak():
    """截斷資料重算的訊號必須和全期訊號一致（q=0.2 與 0.1）。"""
    P = FL.load(DEV[1])
    full = {q: signals(*prep(P), q) for q in QS}
    bad = []
    for c in ("2013-06-28", "2016-03-31", "2018-09-28"):
        Pc = {k: (v.loc[:c] if isinstance(v, (pd.DataFrame, pd.Series)) and isinstance(v.index, pd.DatetimeIndex) else v)
              for k, v in P.items()}
        for q in QS:
            part = signals(*prep(Pc), q)
            for h in MODELS:
                a, b = part[h], full[q][h].loc[part[h].index, part[h].columns]
                n = int((a != b).to_numpy().sum())
                if n:
                    bad.append((c, q, h, n))
        print(f"  截斷 {c} 檢查完", flush=True)
    print("leak_check：" + ("全部一致" if not bad else f"不一致 {bad}"))
    SL.log(dict(kind="leak_check", period="dev", strategy="第五十二批 散戶逆襲 10 模型", ok=not bad, bad=bad))


# ───────── 統計 ─────────
def seg_stats(T, p):
    m = (T.buy >= pd.Timestamp(p[0])) & (T.buy <= pd.Timestamp(p[1]))
    return SL.event_stats(T[m]) if m.sum() > 1 else dict(n=int(m.sum()), ex=np.nan, med=np.nan, t=np.nan)


def describe(T, sig, p):
    d = (sig.index >= pd.Timestamp(p[0])) & (sig.index <= pd.Timestamp(p[1]))
    days = int(d.sum())
    w, lo = T.ret[T.ret > 0], T.ret[T.ret <= 0]
    rs = T.reason.value_counts(normalize=True).round(2).to_dict()
    return dict(sig_day=float(sig[d].sum(axis=1).mean()), entry_day=len(T) / max(days, 1), bars=T.bars.mean() if len(T) else np.nan,
                avg_win=w.mean(), avg_loss=lo.mean(), reasons=rs)


def pass1(st, sa, sb):
    if st["n"] < MIN_N or not (st["ex"] > 0 and st["t"] >= 2 and st["med"] > 0):
        return False
    if sa["n"] >= MIN_A:
        return bool(np.sign(sa["ex"]) == np.sign(sb["ex"]))
    return bool(sb["ex"] > 0)


def book(T, C, O, days, mkt, p):
    """10 格、同日依代號；評估期間從第一筆進場的月份開始。"""
    nav, inv, took = SL.slot_book(T, K, C, O, days)
    p0 = max(pd.Timestamp(p[0]), T.buy.min().to_period("M").to_timestamp())
    df, md = B51.monthly(nav, inv, mkt, str(p0.date()), p[1])
    return df, md, took, (str(p0.date()), p[1])


def pass2(st):
    if "yrs" not in st:
        return False, 0
    k, n = (int(x) for x in st["yrs"].split("/"))
    return bool(st["t"] >= 2 and k >= math.ceil(n * 7 / 9)), math.ceil(n * 7 / 9)


def placebo(P, sig, el, p, ex, n=NPL, seed=2052):
    """每天的訊號數隨機分給同一天池內的股票，出場規則相同；回傳逐筆 t 的分布。"""
    rng = np.random.default_rng(seed)
    S, E = sig.values, el.reindex(index=sig.index, columns=sig.columns).fillna(False).values
    cnt = S.sum(axis=1)
    out = []
    for _ in range(n):
        R = np.zeros_like(S)
        for i in np.flatnonzero(cnt):
            pool = np.flatnonzero(E[i])
            if len(pool):
                R[i, rng.choice(pool, size=min(cnt[i], len(pool)), replace=False)] = True
        T = SL.event_trades(P, pd.DataFrame(R, index=sig.index, columns=sig.columns), p[0], p[1], p[1], **EXITS[ex])
        out.append(SL.event_stats(T)["t"])
    return np.array(out)


def fmt(st):
    return f"{st['n']:6d} 筆 超額 {st['ex']*100:+6.2f}% 中位 {st['med']*100:+6.2f}% t {st['t']:5.2f} 勝率 {st['win']*100:4.1f}%"


# ───────── dev ─────────
def dev():
    t0 = time.time()
    P = FL.load(DEV[1])
    F, el, pc = prep(P)
    sigs = {q: signals(F, el, pc, q) for q in QS}
    C, O, days = P["close"].ffill(), P["open"], P["close"].index
    mkt = SL.mkt_index(P)
    print(f"載入＋特徵 {time.time()-t0:.0f}s", flush=True)
    rows, ser, trades = [], {}, {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        print(f"\n== dev {DEV[0]} ~ {DEV[1]}：逐筆（超額 vs 同持有窗 pit 池等權；成本已扣）")
        for h, title in MODELS.items():
            print(f"  {h} {title}")
            for q in QS:
                for ex in EXITS:
                    v = nm(h, q, ex)
                    sig = sigs[q][h]
                    T = SL.event_trades(P, sig, DEV[0], DEV[1], DEV[1], **EXITS[ex])
                    trades[v] = T
                    st = SL.event_stats(T) if len(T) > 1 else dict(n=len(T), ex=np.nan, med=np.nan, win=np.nan, t=np.nan, clusters=0)
                    sa, sb = seg_stats(T, SEG["A"]), seg_stats(T, SEG["B"])
                    ds = describe(T, sig, DEV) if len(T) else {}
                    ok1 = pass1(st, sa, sb) if len(T) > 1 else False
                    rows.append(dict(h=h, var=v, q=q, exit=ex, **{f"dev_{k}": x for k, x in st.items()},
                                     A_n=sa["n"], A_ex=sa["ex"], A_t=sa["t"], B_n=sb["n"], B_ex=sb["ex"], B_t=sb["t"], **ds, pass1=ok1))
                    print(f"    q{q} {ex:<6} {fmt(st)}｜A {sa['ex']*100:+.2f}%（{sa['n']}）B {sb['ex']*100:+.2f}%（{sb['n']}）｜"
                          f"每天訊號 {ds.get('sig_day', 0):5.1f} 檔、新進場 {ds.get('entry_day', 0):4.1f} 筆、持有 {ds.get('bars', np.nan):4.1f} 天、"
                          f"平均賺 {ds.get('avg_win', np.nan)*100:+.1f}% 賠 {ds.get('avg_loss', np.nan)*100:+.1f}% → {'過' if ok1 else '✗'}"
                          f"  [{time.time()-t0:.0f}s]", flush=True)
        D = pd.DataFrame(rows)

        print("\n== 第二步 投組（10 格、同日依代號；只做第一步通過的）")
        D["pass2"], D["placebo_p95"], D["placebo_ok"] = False, np.nan, False
        for i, r in D[D.pass1].iterrows():
            df, md, took, pe = book(trades[r["var"]], C, O, days, mkt, DEV)
            st = SL.stats(df, md, *pe)
            sm = SL.stats(df.assign(ex=df.ex_match), md, *pe)
            ok2, need = pass2(st)
            ser[r["var"]] = (df.ex.values, md)
            for k, x in st.items():
                D.loc[i, f"bk_{k}"] = x
            D.loc[i, "bk_ex_match"], D.loc[i, "bk_period"], D.loc[i, "pass2"] = sm["ex"], f"{pe[0]}~{pe[1]}", ok2
            print(f"  {r['var']:<22} 月超額年化 {st['ex']*100:+6.1f}% t {st['t']:5.2f} 正年 {st['yrs']}（需 {need}）Sharpe {st['sharpe']:.2f} "
                  f"MDD {st['mdd']*100:6.1f}% 曝險 {st['hold']*100:3.0f}%｜對同曝險池 {sm['ex']*100:+.1f}%（t {sm['t']:.2f}）｜"
                  f"成交 {len(took)}/{len(trades[r['var']])} 筆 → {'過' if ok2 else '✗'}", flush=True)

        if D.pass1.any():
            print(f"\n== 安慰劑（每天的訊號數隨機分給池內股票，{NPL} 次，同出場；逐筆 t）")
            for i, r in D[D.pass1].iterrows():
                t1 = time.time()
                pl = placebo(P, sigs[r.q][r.h], el, DEV, r.exit)
                p95 = np.nanpercentile(pl, 95)
                D.loc[i, "placebo_p95"], D.loc[i, "placebo_ok"] = p95, r.dev_t > p95
                print(f"  {r['var']:<22} 真實 t {r.dev_t:.2f}｜安慰劑 50／95 分位 {np.nanpercentile(pl, 50):.2f}／{p95:.2f} → "
                      f"{'超過' if r.dev_t > p95 else '沒超過'}  [{time.time()-t1:.0f}s]", flush=True)

        sel = []
        for h in MODELS:
            sub = D[(D.h == h) & D.pass1 & D.pass2 & D.placebo_ok]
            if len(sub):
                sel.append(sub.sort_values("dev_t", ascending=False).iloc[0]["var"])
        D["selected"] = D["var"].isin(sel)
        if ser:
            keys = list(ser)
            srs = [np.nanmean(ser[k][0]) / np.nanstd(ser[k][0], ddof=1) for k in keys]
            for k in keys:
                D.loc[D["var"] == k, "dsr"] = SL.deflated_sharpe(ser[k][0], srs, TRIALS)
        print(f"\n== 選進 val：{sel if sel else '無'}")
    D.to_csv(OUT / "batch52_dev.csv", index=False)
    SL.log(dict(kind="strategy", period="dev", strategy="第五十二批 散戶逆襲 10 模型", note="第五十二批 事前登記",
                rows=D.to_dict("records"), selected=sel, trials_cum=TRIALS))


# ───────── val ─────────
def val():
    D = pd.read_csv(OUT / "batch52_dev.csv")
    sel = D[D.selected]
    if not len(sel):
        print("dev 沒有選出任何變體 → 依登記不跑 val")
        return
    t0 = time.time()
    P = FL.load(VAL[1])
    F, el, pc = prep(P)
    C, O, days = P["close"].ffill(), P["open"], P["close"].index
    mkt = SL.mkt_index(P)
    print(f"載入＋特徵 {time.time()-t0:.0f}s", flush=True)
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        print(f"\n== val {VAL[0]} ~ {VAL[1]}")
        for _, r in sel.iterrows():
            sig = signals(F, el, pc, r.q)[r.h]
            T = SL.event_trades(P, sig, VAL[0], VAL[1], VAL[1], **EXITS[r.exit])
            st = SL.event_stats(T)
            df, md, took, pe = book(T, C, O, days, mkt, VAL)
            bs = SL.stats(df, md, *pe)
            ok = st["ex"] > 0 and st["t"] >= 2 and st["med"] > 0 and bs["t"] >= 2 and bs["yrs"] == "4/4"
            yr = (df.ex.groupby(md.year).sum() * 100).round(1).to_dict()
            out[r["var"]] = dict(trades=st, book=bs, by_year=yr, verdict="✓" if ok else "✗")
            print(f"  {r['var']:<22} 逐筆 {fmt(st)}｜投組 月超額年化 {bs['ex']*100:+.1f}% t {bs['t']:.2f} 正年 {bs['yrs']} "
                  f"Sharpe {bs['sharpe']:.2f} MDD {bs['mdd']*100:.1f}%｜逐年 {yr} → {out[r['var']]['verdict']}", flush=True)
    SL.log(dict(kind="strategy", period="val", strategy="第五十二批 散戶逆襲 10 模型", note="第五十二批 dev 選出的", result=out))


if __name__ == "__main__":
    {"leak": leak, "dev": dev, "val": val}[sys.argv[1] if len(sys.argv) > 1 else "dev"]()
