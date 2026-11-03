"""探索（使用者要求，未事前登記，不算證據）：用券商分點資料重做「散戶逆襲」課程的籌碼面。

期間：2021-07 ～ 資料最後一天（使用者 2026-11-03 同意含 holdout）；分 P1 2021-07～2023-12、P2 2024-01～ 兩段報告。
股票：舊個股清單 1,439 檔（data/broker 只有這些），且當天在 pit 流動池（60 日均量 ≥ 200 張、≥ 250 根 K）。

第一部分：課程「籌碼面」打分表裡能用公開資料做的 7 項（C1～C3、C7～C10），逐項和總分 → 次日開盤進、持有 6／20 日的超額（毛，未扣成本）；
          另測連續特徵（籌碼集中度 5／20／60 日、買方集中度、大量買超分點數）的五分位。
第二部分：籌碼類模型的分點版（事件型，同第五十二批：隔日開盤進、6 日或移動 10%，harness 扣成本）。

用法：.venv/bin/python research/v2/explore_broker_chips.py
"""
import multiprocessing as mp
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import feature_library as FL  # noqa: E402
import feature_tracks as FT  # noqa: E402
import harness as H  # noqa: E402
import strategy_lab as SL  # noqa: E402

BROKER = H.BASE / "data" / "broker"
CACHE = H.D / "results" / "broker_feats.pkl"
END = "2026-11-02"
P0 = "2021-07-01"
PERS = {"P1 2021-07～2023-12": ("2021-07-01", "2023-12-31"), "P2 2024-01～2026-10": ("2024-01-01", "2026-10-31")}
HS = (6, 20)
LOT = 1000
SMALL = ("1-999", "1,000-5,000", "5,001-10,000")          # 羊群：持股 ≤ 10 張
BIG = "more than 1,000,001"                                 # 大戶：> 1,000 張
EXITS = {"6日": dict(stop=None, max_hold=6), "移動10%": dict(stop=0.10, max_hold=20)}


# ───────── 分點 → 每檔每日特徵 ─────────
def _roll(x, n):
    c = np.cumsum(np.vstack([np.zeros((1,) + x.shape[1:]), x]), axis=0)
    out = c[n:] - c[:-n]
    return np.vstack([np.full((n - 1,) + x.shape[1:], np.nan), out])


def stock_feats(sid):
    f = BROKER / f"{sid}.parquet"
    if not f.exists():
        return sid, None
    b = pd.read_parquet(f, columns=["date", "securities_trader_id", "buy", "sell"])
    if not len(b):
        return sid, None
    b = b.groupby(["date", "securities_trader_id"], as_index=False)[["buy", "sell"]].sum()
    di, dates = pd.factorize(b.date, sort=True)
    bi, _ = pd.factorize(b.securities_trader_id)
    D, B = len(dates), bi.max() + 1
    buy, sell = np.zeros((D, B)), np.zeros((D, B))
    buy[di, bi], sell[di, bi] = b.buy.values, b.sell.values
    net, gross = buy - sell, buy + sell
    vol = buy.sum(axis=1)
    k = min(15, B)
    srt = np.sort(net, axis=1)
    main = srt[:, -k:].clip(min=0).sum(axis=1) + srt[:, :k].clip(max=0).sum(axis=1)       # 前 15 大買超 − 前 15 大賣超
    t10 = np.argpartition(-gross, min(10, B - 1), axis=1)[:, :10]
    top10d = np.take_along_axis(net, t10, axis=1).sum(axis=1)                                 # 當日前 10 大交易分點淨買
    g20, n20 = _roll(gross, 20), _roll(net, 20)
    top10_20 = np.full(D, np.nan)
    ok = np.isfinite(g20[:, 0])
    i20 = np.argpartition(-np.nan_to_num(g20[ok]), min(10, B - 1), axis=1)[:, :10]
    top10_20[ok] = np.take_along_axis(n20[ok], i20, axis=1).sum(axis=1)                       # 20 日前 10 大交易分點淨買
    top5buy = np.sort(buy, axis=1)[:, -5:].sum(axis=1) / np.where(vol > 0, vol, np.nan)        # 買方集中度
    n5 = _roll(net, 5)
    nbig5 = np.where(np.isfinite(n5[:, 0]), (n5 >= 100 * LOT).sum(axis=1), np.nan)            # 5 日淨買 ≥ 100 張的分點數
    out = pd.DataFrame(dict(vol=vol, main=main, maxnet=srt[:, -1], top10d=top10d, top10_20=top10_20,
                            top5buy=top5buy, nbig5=nbig5), index=pd.to_datetime(dates))
    return sid, out


def hold_weekly(sid, idx):
    """集保每週：大戶（> 1,000 張）比例變化、羊群（≤ 10 張）比例變化；快照日 +3 天（下週一）起可用。"""
    f = H.D / "hold" / f"{sid}.parquet"
    if not f.exists():
        return None
    d = pd.read_parquet(f)
    if not len(d) or not {"date", "HoldingSharesLevel", "percent"} <= set(d.columns):
        return None
    d["date"] = pd.to_datetime(d.date)
    p = d.pivot_table(index="date", columns="HoldingSharesLevel", values="percent", aggfunc="last").sort_index()
    if BIG not in p:
        return None
    big = p[BIG]
    small = p[[c for c in SMALL if c in p]].sum(axis=1, min_count=1)
    w = pd.DataFrame(dict(big_chg=big.diff(), big_chg1=big.diff().shift(1), small_chg=small.diff()))
    w.index = w.index + pd.Timedelta(days=3)
    return w.reindex(w.index.union(idx)).ffill().reindex(idx)


def build(P):
    if CACHE.exists():
        return pd.read_pickle(CACHE)
    sids = [s for s in P["close"].columns if (BROKER / f"{s}.parquet").exists()]
    t0 = time.time()
    with mp.get_context("fork").Pool(8) as pool:
        res = dict(pool.map(stock_feats, sids, chunksize=8))
    idx, cols = P["close"].index, P["close"].columns
    F = {k: pd.DataFrame({s: v[k] for s, v in res.items() if v is not None}).reindex(index=idx, columns=cols)
         for k in ("vol", "main", "maxnet", "top10d", "top10_20", "top5buy", "nbig5")}
    hw = {s: hold_weekly(s, idx) for s in sids}
    for k in ("big_chg", "big_chg1", "small_chg"):
        F[k] = pd.DataFrame({s: v[k] for s, v in hw.items() if v is not None}).reindex(index=idx, columns=cols)
    pd.to_pickle(F, CACHE)
    print(f"分點特徵 {len(sids)} 檔 {time.time()-t0:.0f}s", flush=True)
    return F


# ───────── 課程籌碼項目 ─────────
def items(P, F):
    vol = F["vol"]
    conc = {n: F["main"].rolling(n, min_periods=n).sum() / vol.rolling(n, min_periods=n).sum().where(lambda x: x > 0)
            for n in (5, 20, 60)}
    inst = P["inst_f"] + P["inst_t"] + P["inst_d"]
    has = vol.notna() & (vol > 0)
    C = {
        "C1 近3日主力分點單日買超≥1000張": F["maxnet"].rolling(3, min_periods=1).max() >= 1000 * LOT,
        "C2 5日籌碼集中度>0": conc[5] > 0,
        "C3 外資投信同時連買2天": (P["inst_f"] > 0) & (P["inst_t"] > 0) & (P["inst_f"].shift(1) > 0) & (P["inst_t"].shift(1) > 0),
        "C7 法人或主力大買且量增": (((F["main"] / vol) >= 0.10) | ((inst / P["volume"]) >= 0.10))
                              & (vol > vol.rolling(20, min_periods=15).mean().shift(1)),
        "C8 前10大交易分點20日淨買>0": F["top10_20"] > 0,
        "C9 前10大交易分點曾連3日淨買": ((F["top10d"] > 0).astype(float).rolling(3).sum() >= 3).astype(float).rolling(20, min_periods=1).max() >= 1,
        "C10 近1週大戶加碼且羊群減碼": (F["big_chg"] > 0) & (F["small_chg"] < 0),
    }
    C = {k: v.where(has) for k, v in C.items()}                     # 沒有分點資料的日子不算
    score = sum(v.astype(float).fillna(0) for v in C.values()).where(has)
    cont = {"CONC5 籌碼集中度5日": conc[5], "CONC20 籌碼集中度20日": conc[20], "CONC60 籌碼集中度60日": conc[60],
            "TOP5BUY 買方集中度": F["top5buy"], "NBIG5 5日大買分點數": F["nbig5"] / np.sqrt(vol.rolling(20).mean() / LOT),
            "MAIN1 當日主力買賣超/量": F["main"] / vol}
    return C, score, cont, conc


def fwd(P, h):
    O = P["open"]
    return (O.shift(-(1 + h)) / O.shift(-1) - 1).clip(-0.9, 5)


def xs_eval(X, R, E, h, p, kind):
    """kind='bin'：True 減 False；'score'：各分數桶；'q'：五分位（Q1 低～Q5 高）與 Q5−Q1。回傳 dict。"""
    m = (X.index >= pd.Timestamp(p[0])) & (X.index <= pd.Timestamp(p[1]))
    Xv, Rv, Ev = X.values[m], R.values[m], E.values[m]
    ex = Rv - np.nanmean(np.where(Ev, Rv, np.nan), axis=1, keepdims=True)
    if kind == "bin":
        a = np.where(Ev & (Xv == 1), ex, np.nan)
        b = np.where(Ev & (Xv == 0), ex, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            d = np.nanmean(a, axis=1) - np.nanmean(b, axis=1)
            share = np.nanmean(np.nansum(Ev & (Xv == 1), axis=1) / np.maximum(Ev.sum(axis=1), 1))
            ta = np.nanmean(a, axis=1)
        d, ta = d[np.isfinite(d)], ta[np.isfinite(ta)]
        return dict(diff=d.mean(), t=FT.tstat(d, h), true_ex=ta.mean(), share=share)
    if kind == "score":
        out = {}
        for s in range(0, 8):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                v = np.nanmean(np.where(Ev & (Xv == s), ex, np.nan), axis=1)
                share = np.nanmean(np.nansum(Ev & (Xv == s), axis=1) / np.maximum(Ev.sum(axis=1), 1))
            v = v[np.isfinite(v)]
            out[s] = (v.mean() if len(v) > 20 else np.nan, FT.tstat(v, h) if len(v) > 20 else np.nan, share)
        return out
    q = pd.DataFrame(np.where(Ev & np.isfinite(Xv), Xv, np.nan)).rank(axis=1, pct=True).values
    b = np.minimum((q * 5 - 1e-12).astype(float) // 1, 4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        qm = np.stack([np.nanmean(np.where(b == k, ex, np.nan), axis=1) for k in range(5)], axis=1)
    ok = np.isfinite(qm).all(axis=1)
    sp = qm[ok, 4] - qm[ok, 0]
    return dict(q=np.nanmean(qm[ok], axis=0), spread=sp.mean(), t=FT.tstat(sp, h), top=qm[ok, 4].mean(), t_top=FT.tstat(qm[ok, 4], h))


# ───────── 第二部分：籌碼模型的分點版 ─────────
def models(P, F, C, conc, el):
    G = {f: FL.feature_fn(f)(P).replace([np.inf, -np.inf], np.nan) for f in ("MA_ALIGN", "H87", "H2", "DT_RATIO", "NI_INC8Q")}
    has = F["vol"].notna() & (F["vol"] > 0)
    pool = el & has

    def top(X, q=0.2):
        return X.where(pool).rank(axis=1, pct=True) >= 1 - q
    ma, hi = G["MA_ALIGN"] >= 10, G["H2"] >= 1 - 1e-12
    big2 = (F["big_chg"] > 0) & (F["big_chg1"] > 0)
    M = {
        "#1 大戶宰羊奔向多頭": C["C10 近1週大戶加碼且羊群減碼"].fillna(False).astype(bool) & (conc[5] > 0) & ma,
        "#2 主力掃貨利多集結": C["C1 近3日主力分點單日買超≥1000張"].fillna(False).astype(bool) & top(G["H87"]) & (conc[20] > 0),
        "#3 大戶法人合流鎖籌": C["C10 近1週大戶加碼且羊群減碼"].fillna(False).astype(bool) & C["C3 外資投信同時連買2天"].fillna(False).astype(bool) & top(conc[20]),
        "#4 巨鯨卡位散戶衝浪": big2 & top(F["big_chg"] + F["big_chg1"]) & top(G["DT_RATIO"]),
        "#5 豹式順流集火破空": (G["NI_INC8Q"] >= 2) & (conc[5] > 0) & (conc[5] > conc[20]) & ma & hi,
        "#9 鯨群集結積極預備": top(F["nbig5"] / np.sqrt(F["vol"].rolling(20).mean() / LOT)) & (conc[20] > 0) & (G["H2"] >= 0.9) & ~hi,
    }
    return {k: (v & pool).fillna(False).astype(bool) for k, v in M.items()}


def main():
    t0 = time.time()
    P = FL.load(END)
    print(f"載入 {time.time()-t0:.0f}s", flush=True)
    F = build(P)
    el = P["liq"].fillna(False).astype(bool) & (P["nbars"] >= 250)
    C, score, cont, conc = items(P, F)
    has = F["vol"].notna() & (F["vol"] > 0)
    rec = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for h in HS:
            R = fwd(P, h)
            E = el & has & R.notna()
            print(f"\n================ 持有 {h} 日（次日開盤進、{h} 日後開盤出；超額 vs 同池等權，毛、未扣 0.585% 成本）================")
            for pn, p in PERS.items():
                print(f"\n── {pn}")
                print("  [課程籌碼項目] 成立的股票超額／成立減不成立（t）／當天成立比例")
                for k, X in C.items():
                    r = xs_eval(X.astype(float), R, E, h, p, "bin")
                    rec[(h, pn, k)] = r
                    print(f"    {k:<26} 成立 {r['true_ex']*100:+.2f}%｜差 {r['diff']*100:+.2f}%（t {r['t']:+.2f}）｜成立比例 {r['share']*100:4.1f}%")
                sc = xs_eval(score, R, E, h, p, "score")
                rec[(h, pn, "score")] = sc
                print("  [7 項總分] 分數：超額（t）／比例")
                print("    " + "  ".join(f"{s}分 {v[0]*100:+.2f}%（{v[1]:+.1f}）/{v[2]*100:.0f}%" for s, v in sc.items() if np.isfinite(v[0])))
                print("  [連續特徵] 五分位 Q1→Q5 超額｜Q5−Q1（t）｜Q5（t）")
                for k, X in cont.items():
                    r = xs_eval(X, R, E, h, p, "q")
                    rec[(h, pn, k)] = r
                    print(f"    {k:<20} " + " ".join(f"{v*100:+.2f}" for v in r["q"]) +
                          f"｜{r['spread']*100:+.2f}%（t {r['t']:+.2f}）｜Q5 {r['top']*100:+.2f}%（t {r['t_top']:+.2f}）", flush=True)

        print("\n================ 第二部分：籌碼模型的分點版（事件，harness 扣成本；超額 vs pit 池等權開盤指數）================")
        M = models(P, F, C, conc, el)
        for k, sig in M.items():
            for ex in EXITS:
                T = SL.event_trades(P, sig, P0, END, END, **EXITS[ex])
                line = []
                for pn, p in PERS.items():
                    m = (T.buy >= pd.Timestamp(p[0])) & (T.buy <= pd.Timestamp(p[1]))
                    st = SL.event_stats(T[m]) if m.sum() > 1 else dict(n=int(m.sum()), ex=np.nan, med=np.nan, t=np.nan, win=np.nan)
                    rec[("ev", k, ex, pn)] = st
                    line.append(f"{pn[:2]} {st['n']:5d} 筆 {st['ex']*100:+.2f}%（t {st['t']:+.2f}）中位 {st['med']*100:+.2f}% 勝 {st['win']*100:.0f}%")
                d = (sig.index >= pd.Timestamp(P0))
                print(f"  {k:<14} {ex:<6} 每天訊號 {sig[d].sum(axis=1).mean():4.1f}｜" + "｜".join(line), flush=True)
    SL.log(dict(kind="explore", period="2021-07~2026-10（含 holdout，使用者同意）", strategy="探索：分點版課程籌碼面",
                note="使用者要求，未事前登記，不算證據", result={str(k): v for k, v in rec.items()}))
    print(f"\n總時間 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
