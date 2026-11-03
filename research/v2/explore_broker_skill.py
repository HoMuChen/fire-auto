"""探索（使用者要求，未事前登記，不算證據）：課程籌碼面 10 項裡不透明的部分。

A. 分點選股能力有沒有延續性（point-in-time，每月 M 只用 M 之前已實現的結果）：
   事件＝某分點某日在某檔淨買 ≥ max(30 張, 當日量 2%)；結果＝隔日開盤進、h 日後開盤出的「動能調整超額」。
   - 全市場（最威券商）：過去 6 個月（h=5）／12 個月（h=20）事件的平均調整超額排十分位 → 下個月事件的表現。
   - 個股（股懂券商）：過去 12 個月在同一檔的事件（≥ 5 筆）平均最高的 3 家 → 下個月在該股的事件 vs 其他分點。
B. 用上面的分點清單重做第 4～6 項，和其他 7 項一起，看成立 vs 不成立的超額（原始／動能調整）。
C. 第 1、7、9 項的模糊門檻各試幾組。

動能調整：每天把池內股票依「近 20 日報酬」×「52 週高接近度」各分五組（25 格），超額＝自己 − 同格平均。
期間：P1 2022-01～2023-12、P2 2024-01～2026-10（第一年 2021-07～12 只當回看）。含 holdout，使用者 2026-11-03 同意。
用法：.venv/bin/python research/v2/explore_broker_skill.py
"""
import multiprocessing as mp
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import explore_broker_chips as X  # noqa: E402
import feature_tracks as FT  # noqa: E402
import harness as H  # noqa: E402
import hypotheses as Hy  # noqa: E402
import strategy_lab as SL  # noqa: E402
import xs  # noqa: E402

END, LOT = X.END, X.LOT
PERS = {"P1 2022-01～2023-12": ("2022-01-01", "2023-12-31"), "P2 2024-01～2026-10": ("2024-01-01", "2026-10-31")}
HS = (5, 20)
EV_MIN, EV_SHARE = 30 * LOT, 0.02
SKILL = {5: dict(W=126, nmin=50, sW=252, snmin=5), 20: dict(W=252, nmin=30, sW=252, snmin=5)}
TOPG, TOPS = 20, 3
_G = {}


# ───────── 報酬與動能調整 ─────────
def excess(P, el, h):
    O, Cl = P["open"], P["close"].ffill()
    R = (O.shift(-(1 + h)) / O.shift(-1) - 1).clip(-0.9, 5).values
    E = el.values & np.isfinite(R)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        raw = R - np.nanmean(np.where(E, R, np.nan), axis=1, keepdims=True)
        q = lambda Z: np.floor(np.clip(pd.DataFrame(np.where(E, Z, np.nan)).rank(axis=1, pct=True).values * 5 - 1e-9, 0, 4))  # noqa: E731
        r20 = (Cl / Cl.shift(20) - 1).values
        bk = q(r20) * 5 + q(Hy.h2_hi52(P).values)
        adj = np.full(R.shape, np.nan)
        for b in range(25):
            M = E & (bk == b)
            mu = np.where(M, R, 0).sum(axis=1) / np.maximum(M.sum(axis=1), 1)
            adj = np.where(M, R - mu[:, None], adj)
    return raw.astype(np.float32), adj.astype(np.float32), E & np.isfinite(adj)


# ───────── 第 1 輪：分點大買事件 ─────────
def _load_b(sid):
    f = X.BROKER / f"{sid}.parquet"
    if not f.exists():
        return None
    b = pd.read_parquet(f, columns=["date", "securities_trader_id", "buy", "sell"])
    if not len(b):
        return None
    b = b.groupby(["date", "securities_trader_id"], as_index=False)[["buy", "sell"]].sum()
    u, inv = np.unique(b.date.values, return_inverse=True)
    pu = np.array([_G["pos"].get(x, -1) for x in pd.to_datetime(u)])
    b["d"] = pu[inv]
    b = b[b.d >= 0]
    b["d"] = b.d.astype(np.int32)
    b["net"] = (b.buy - b.sell).astype(np.float64)
    return b


def ev_work(sid):
    b = _load_b(sid)
    if b is None:
        return None
    vol = b.groupby("d").buy.transform("sum")
    e = b[(b.net >= EV_MIN) & (b.net >= EV_SHARE * vol)]
    return pd.DataFrame(dict(j=np.int32(_G["col"][sid]), d=e.d.values, bid=e.securities_trader_id.values.astype("S8"),
                             net=e.net.values.astype(np.float32)))


# ───────── 能力延續性 ─────────
def months(idx, p0, p1):
    ms = pd.period_range(p0, p1, freq="M")
    per = idx.to_period("M")
    out = []
    for m in ms:
        w = np.flatnonzero(per == m)
        if len(w):
            out.append((m, w[0], w[-1] + 1))
    return out


def skill(EV, NB, idx):
    """回傳 rec（延續性統計）、glob[h][month]＝全市場前 TOPG 名分點、stock[h][month]＝{j: [前 TOPS 名分點]}。"""
    rec, glob, stock = {}, {h: {} for h in HS}, {h: {} for h in HS}
    d, b, j = EV.d.values, EV.b.values, EV.j.values
    for h in HS:
        a = EV[f"adj{h}"].values
        fin = np.isfinite(a)
        cfg = SKILL[h]
        dec, sp, dong = [], [], []
        for m, s0, s1 in months(idx, "2022-01", "2026-10"):
            tr = fin & (d >= s0 - cfg["W"]) & (d + 1 + h < s0)
            cnt = np.bincount(b[tr], minlength=NB)
            mu = np.bincount(b[tr], weights=a[tr], minlength=NB) / np.maximum(cnt, 1)
            ok = cnt >= cfg["nmin"]
            nx = fin & (d >= s0) & (d < s1)
            if ok.sum() >= 50:
                r = pd.Series(mu[ok]).rank(pct=True).values
                dc = np.full(NB, -1)
                dc[np.flatnonzero(ok)] = np.minimum((r * 10 - 1e-9).astype(int), 9)
                k = nx & (dc[b] >= 0)
                c2 = np.bincount(dc[b[k]], minlength=10)
                m2 = np.where(c2 > 0, np.bincount(dc[b[k]], weights=a[k], minlength=10) / np.maximum(c2, 1), np.nan)
                dec.append(m2)
                glob[h][m] = np.flatnonzero(ok)[np.argsort(-mu[ok], kind="stable")[:TOPG]]
            # 個股：過去 sW 日同一檔 ≥ snmin 筆
            st = fin & (d >= s0 - cfg["sW"]) & (d + 1 + h < s0)
            g = pd.DataFrame(dict(j=j[st], b=b[st], a=a[st])).groupby(["j", "b"]).a.agg(["mean", "count"])
            g = g[(g["count"] >= cfg["snmin"]) & (g["mean"] > 0)].reset_index().sort_values(["j", "mean"], ascending=[True, False])
            top = g.groupby("j").head(TOPS)
            stock[h][m] = top.groupby("j").b.apply(list).to_dict()
            if len(top):
                nj = nx & np.isin(j, top.j.unique())
                isd = np.isin(j[nj].astype(np.int64) * NB + b[nj], top.j.values.astype(np.int64) * NB + top.b.values)
                if isd.any() and (~isd).any():
                    dong.append((a[nj][isd].mean(), a[nj][~isd].mean(), isd.sum()))
        dec = np.array(dec)
        top_bot = dec[:, 9] - dec[:, 0]
        dg = np.array(dong)
        rec[h] = dict(dec=np.nanmean(dec, axis=0), tb=np.nanmean(top_bot), tb_t=FT.tstat(top_bot[np.isfinite(top_bot)], 0),
                      months=len(dec), dong=dg[:, 0].mean(), other=dg[:, 1].mean(),
                      dong_t=FT.tstat(dg[:, 0] - dg[:, 1], 0), dong_n=int(dg[:, 2].sum()))
    return rec, glob, stock


# ───────── 第 2 輪：第 4～6 項 ─────────
def item_work(sid):
    b = _load_b(sid)
    if b is None:
        return sid, None
    jj = _G["col"][sid]
    ub = _G["ub"]
    want = set()
    for h in HS:
        for m, v in _G["stock"][h].items():
            want.update(v.get(jj, []))
    for v in _G["glob"][5].values():
        want.update(v.tolist())
    days = np.unique(b.d.values)
    b = b[b.securities_trader_id.isin({ub[i] for i in want})]
    b = b.assign(b=b.securities_trader_id.map({ub[i]: i for i in want}))
    D = len(_G["idx"])
    out = {k: np.full(D, np.nan) for k in ("I4", "I5", "I6")}
    for k in out:
        out[k][days] = 0.0
    if not len(b):
        return sid, out
    pv = b.pivot_table(index="d", columns="b", values="net", aggfunc="sum").reindex(np.arange(D)).fillna(0)
    pos = (pv > 0).values
    c2 = pos & np.vstack([np.zeros((1, pos.shape[1]), bool), pos[:-1]])                    # 連 2 日淨買
    ever = pd.DataFrame(c2.astype(float)).rolling(20, min_periods=1).max().values > 0       # 近 20 日曾連續買超
    colix = {c: i for i, c in enumerate(pv.columns)}
    for m, s0, s1 in _G["months"]:
        sl = slice(s0, s1)
        dd = days[(days >= s0) & (days < s1)]
        if not len(dd):
            continue
        for key, mat, sets in (("I4", c2, _G["stock"][20].get(m, {}).get(jj, [])),
                               ("I5", ever, _G["glob"][5].get(m, np.array([], int)).tolist()),
                               ("I6", ever, _G["stock"][5].get(m, {}).get(jj, []))):
            ci = [colix[c] for c in sets if c in colix]
            if ci:
                v = mat[sl][:, ci].any(axis=1)
                out[key][dd] = v[dd - s0].astype(float)
    return sid, out


# ───────── 評估 ─────────
def bin_eval(Xv, Ex, E, idx, h, p):
    m = (idx >= pd.Timestamp(p[0])) & (idx <= pd.Timestamp(p[1]))
    Xv, Ex, E = Xv[m], Ex[m], E[m]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        a = np.nanmean(np.where(E & (Xv == 1), Ex, np.nan), axis=1)
        b = np.nanmean(np.where(E & (Xv == 0), Ex, np.nan), axis=1)
        share = np.nanmean(np.nansum(E & (Xv == 1), axis=1) / np.maximum(np.nansum(E & np.isfinite(Xv), axis=1), 1))
    dlt = (a - b)[np.isfinite(a - b)]
    return dlt.mean() if len(dlt) else np.nan, FT.tstat(dlt, h) if len(dlt) > 20 else np.nan, share


def show(name, Xdf, EXS, idx, cols):
    Xv = Xdf.reindex(index=idx, columns=cols).astype(float).values
    parts = []
    for h in HS:
        raw, adj, E = EXS[h]
        for pn, p in PERS.items():
            r0, t0, sh = bin_eval(Xv, raw, E, idx, h, p)
            r1, t1, _ = bin_eval(Xv, adj, E, idx, h, p)
            parts.append(f"{h}日{pn[:2]} 原 {r0*100:+.2f}（{t0:+.1f}）調 {r1*100:+.2f}（{t1:+.1f}）")
    print(f"    {name:<30} 成立 {sh*100:4.1f}%｜" + "｜".join(parts), flush=True)


def main():
    t0 = time.time()
    P = xs.load_panel(END)
    xs.add_chips(P)
    idx, cols = P["close"].index, P["close"].columns
    el = P["liq"].fillna(False).astype(bool) & (P["nbars"] >= 250)
    F = {k: v.reindex(index=idx, columns=cols) for k, v in pd.read_pickle(X.CACHE).items()}
    has = F["vol"].notna() & (F["vol"] > 0)
    EXS = {h: excess(P, el & has, h) for h in HS}
    print(f"載入＋報酬 {time.time()-t0:.0f}s", flush=True)

    sids = [s for s in cols if (X.BROKER / f"{s}.parquet").exists()]
    _G.update(pos={d: i for i, d in enumerate(idx)}, col={s: i for i, s in enumerate(cols)}, idx=idx)
    with mp.get_context("fork").Pool(8) as pool:
        EV = pd.concat([x for x in pool.map(ev_work, sids, chunksize=8) if x is not None], ignore_index=True)
    ub, EV["b"] = np.unique(EV.bid.values, return_inverse=True)
    NB = len(ub)
    for h in HS:
        raw, adj, E = EXS[h]
        dd, jj = EV.d.values, EV.j.values
        EV[f"adj{h}"] = np.where(E[dd, jj], adj[dd, jj], np.nan)
        EV[f"raw{h}"] = np.where(E[dd, jj], raw[dd, jj], np.nan)
    print(f"分點大買事件 {len(EV):,} 筆、{NB} 個分點 {time.time()-t0:.0f}s", flush=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        print("\n== A. 事件本身（全部分點大買，下 h 日超額）")
        for h in HS:
            for pn, p in PERS.items():
                m = (idx[EV.d] >= pd.Timestamp(p[0])) & (idx[EV.d] <= pd.Timestamp(p[1]))
                print(f"  {h} 日 {pn}：{m.sum():,} 筆｜原始 {np.nanmean(EV[f'raw{h}'][m])*100:+.3f}%｜動能調整 {np.nanmean(EV[f'adj{h}'][m])*100:+.3f}%")

        rec, glob, stock = skill(EV, NB, idx)
        print("\n== A. 分點能力的延續性（每月用過去已實現的結果排名 → 下個月事件的動能調整超額；2022-01～2026-10）")
        for h in HS:
            r = rec[h]
            print(f"  h={h:<2}（全市場排名，回看 {SKILL[h]['W']} 日、≥ {SKILL[h]['nmin']} 筆）十分位 D1→D10：" +
                  " ".join(f"{v*100:+.2f}" for v in r["dec"]) + f"｜D10−D1 {r['tb']*100:+.2f}%（t {r['tb_t']:+.2f}，{r['months']} 個月）")
            print(f"        （個股「股懂」：同檔過去 12 個月 ≥ 5 筆、平均最高 3 家）下月在該股：股懂 {r['dong']*100:+.2f}%  其他分點 {r['other']*100:+.2f}%"
                  f"  差 t {r['dong_t']:+.2f}（股懂事件 {r['dong_n']:,} 筆）", flush=True)

        _G.update(ub=[x.decode() for x in ub], glob=glob, stock=stock, months=months(idx, "2022-01", "2026-10"))
        with mp.get_context("fork").Pool(8) as pool:
            res = dict(pool.map(item_work, sids, chunksize=8))
        I = {k: pd.DataFrame({s: v[k] for s, v in res.items() if v is not None}, index=idx).reindex(columns=cols) for k in ("I4", "I5", "I6")}
        print(f"\n第 4～6 項訊號 {time.time()-t0:.0f}s")

        C, score, cont, conc = X.items(P, F)
        names = {"I4": "4 近2日中長線股懂券商連續買超", "I5": "5 近20日短期最威券商曾連續買超", "I6": "6 近20日短期股懂券商曾連續買超"}
        print("\n== B. 10 項：成立 − 不成立的超額（原＝對池平均、調＝動能調整；%，括號 t）")
        allC = {**{k.split(" ")[0] + " " + k.split(" ")[1]: v for k, v in C.items()}, **{names[k]: v for k, v in I.items()}}
        order = ["C1", "C2", "C3", "4", "5", "6", "C7", "C8", "C9", "C10"]
        for o in order:
            k = next(n for n in allC if n.split(" ")[0] == o)
            show(k, allC[k], EXS, idx, cols)
        sc10 = sum(v.reindex(index=idx, columns=cols).astype(float).fillna(0) for v in allC.values()).where(has)
        print("  10 項總分（重做版）：分數桶的動能調整超額（20 日）")
        raw, adj, E = EXS[20]
        for pn, p in PERS.items():
            m = (idx >= pd.Timestamp(p[0])) & (idx <= pd.Timestamp(p[1]))
            sv = sc10.values[m]
            line = []
            for s in range(11):
                v = np.nanmean(np.where(E[m] & (sv == s), adj[m], np.nan), axis=1)
                v = v[np.isfinite(v)]
                if len(v) > 20:
                    line.append(f"{s}分 {v.mean()*100:+.2f}（{FT.tstat(v, 20):+.1f}）")
            print(f"    {pn}：" + " ".join(line), flush=True)

        print("\n== C. 模糊門檻（第 1、7、9 項）：成立 − 不成立")
        mx3 = lambda Z, L: Z.rolling(L, min_periods=1).max()   # noqa: E731
        vol, inst = F["vol"], P["inst_f"] + P["inst_t"] + P["inst_d"]
        for T in (500, 1000, 2000):
            for L in (1, 3, 5):
                show(f"1 近{L}日單一分點買超≥{T}張", (mx3(F["maxnet"], L) >= T * LOT).where(has), EXS, idx, cols)
        for s in (0.05, 0.10, 0.20):
            show(f"1' 近3日單一分點買超≥當日量{int(s*100)}%", (mx3(F["maxnet"] / vol, 3) >= s).where(has), EXS, idx, cols)
        for s in (0.05, 0.10, 0.20):
            for v in (1.0, 1.5, 2.0):
                x = (((F["main"] / vol) >= s) | ((inst / P["volume"]) >= s)) & (vol > v * vol.rolling(20, min_periods=15).mean().shift(1))
                show(f"7 主力或法人買≥量{int(s*100)}%且量>{v}倍", x.where(has), EXS, idx, cols)
        for k in (2, 3, 5):
            x = ((F["top10d"] > 0).astype(float).rolling(k).sum() >= k).astype(float).rolling(20, min_periods=1).max() >= 1
            show(f"9 前10大分點近20日曾連{k}日淨買", x.where(has), EXS, idx, cols)
    SL.log(dict(kind="explore", period="2022-01~2026-10（含 holdout，使用者同意）", strategy="探索：分點能力延續性與課程不透明項目",
                note="使用者要求，未事前登記，不算證據", skill={h: {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()} for h, r in rec.items()}))
    print(f"\n總時間 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
