"""探索（未登記，不算證據）：explore_broker_skill.py 的追查——全市場分點能力延續性（D10−D1）在 P1／P2 各自成立嗎？
排名延後 1／3／6 個月、同檔同月去均值（只剩買點時機）、D10／D1 常客、月對月留存、個股訊號「近 N 日 D10 分點大買」，
最後用更嚴的 81 格控制（R20×R120×H2×成交值）重算。用法：.venv/bin/python research/v2/explore_broker_skill2.py
"""
import multiprocessing as mp
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import explore_broker_skill as S  # noqa: E402
import explore_broker_chips as X  # noqa: E402
import feature_tracks as FT  # noqa: E402
import xs  # noqa: E402

warnings.simplefilter("ignore", RuntimeWarning)
P = xs.load_panel(S.END)
idx, cols = P["close"].index, P["close"].columns
el = P["liq"].fillna(False).astype(bool) & (P["nbars"] >= 250)
F = {k: v.reindex(index=idx, columns=cols) for k, v in pd.read_pickle(X.CACHE).items()}
has = F["vol"].notna() & (F["vol"] > 0)
EXS = {h: S.excess(P, el & has, h) for h in S.HS}
sids = [s for s in cols if (X.BROKER / f"{s}.parquet").exists()]
S._G.update(pos={d: i for i, d in enumerate(idx)}, col={s: i for i, s in enumerate(cols)}, idx=idx)
with mp.get_context("fork").Pool(8) as pool:
    EV = pd.concat([x for x in pool.map(S.ev_work, sids, chunksize=8) if x is not None], ignore_index=True)
ub, EV["b"] = np.unique(EV.bid.values, return_inverse=True)
NB = len(ub)
d, j, b = EV.d.values, EV.j.values, EV.b.values
for h in S.HS:
    raw, adj, E = EXS[h]
    EV[f"adj{h}"] = np.where(E[d, j], adj[d, j], np.nan)
names = {}
for s in ("2330", "2317", "2454", "2603", "3231", "2382", "1101", "6505"):
    t = pd.read_parquet(X.BROKER / f"{s}.parquet", columns=["securities_trader_id", "securities_trader"]).drop_duplicates()
    names.update(dict(zip(t.securities_trader_id, t.securities_trader)))
ubs = [x.decode() for x in ub]
MS = S.months(idx, "2022-01", "2026-10")
mon = np.full(len(idx), -1)
for k, (m, s0, s1) in enumerate(MS):
    mon[s0:s1] = k
em = mon[d]

for h in S.HS:
    a = EV[f"adj{h}"].values
    fin = np.isfinite(a)
    cfg = S.SKILL[h]
    DEC = {lag: np.full(len(EV), -1, np.int8) for lag in (1, 3, 6)}      # lag＝用幾個月前的排名
    ranks = []
    for k, (m, s0, s1) in enumerate(MS):
        tr = fin & (d >= s0 - cfg["W"]) & (d + 1 + h < s0)
        cnt = np.bincount(b[tr], minlength=NB)
        mu = np.bincount(b[tr], weights=a[tr], minlength=NB) / np.maximum(cnt, 1)
        ok = cnt >= cfg["nmin"]
        dc = np.full(NB, -1)
        r = pd.Series(mu[ok]).rank(pct=True).values
        dc[np.flatnonzero(ok)] = np.minimum((r * 10 - 1e-9).astype(int), 9)
        ranks.append(dc)
    for lag in DEC:
        for k in range(len(MS)):
            if k - (lag - 1) >= 0:
                w = em == k
                DEC[lag][w] = ranks[k - (lag - 1)][b[w]]
    print(f"\n######## h={h}")
    for lag in (1, 3, 6):
        D = DEC[lag]
        line = []
        for pn, (p0, p1) in S.PERS.items():
            ks = [k for k, (m, s0, s1) in enumerate(MS) if str(m) >= p0[:7] and str(m) <= p1[:7]]
            sp = []
            for k in ks:
                w = (em == k) & fin
                hi, lo = a[w & (D == 9)], a[w & (D == 0)]
                if len(hi) and len(lo):
                    sp.append(hi.mean() - lo.mean())
            sp = np.array(sp)
            line.append(f"{pn[:2]} D10−D1 {sp.mean()*100:+.2f}%（t {FT.tstat(sp, 0):+.2f}，{len(sp)}月）")
        print(f"  排名用 {lag} 個月前（再往前 {lag-1} 個月空窗）：" + "｜".join(line))
    D = DEC[1]
    # 同一檔同一月內比較：去掉「挑股」，只剩「時點」
    w = fin & (D >= 0)
    g = pd.DataFrame(dict(jm=j[w].astype(np.int64) * 1000 + em[w], a=a[w], D=D[w]))
    g["dm"] = g.a - g.groupby("jm").a.transform("mean")
    sp = []
    for k in range(len(MS)):
        x = g[(g.jm % 1000) == k]
        hi, lo = x.dm[x.D == 9], x.dm[x.D == 0]
        if len(hi) and len(lo):
            sp.append(hi.mean() - lo.mean())
    sp = np.array(sp)
    print(f"  同檔同月去均值後 D10−D1：{sp.mean()*100:+.3f}%（t {FT.tstat(sp, 0):+.2f}）← 只剩買點時機")
    # 誰在 D10：出現頻率
    R = np.array(ranks)
    top = (R == 9).sum(axis=0)
    bot = (R == 0).sum(axis=0)
    o = np.argsort(-top)[:20]
    print("  最常在 D10 的分點（58 個月中幾次）：" + "、".join(f"{names.get(ubs[i], ubs[i])}{top[i]}" for i in o))
    o = np.argsort(-bot)[:12]
    print("  最常在 D1 的分點：" + "、".join(f"{names.get(ubs[i], ubs[i])}{bot[i]}" for i in o))
    pr = [(R[k] == 9) & (R[k - 1] == 9) for k in range(1, len(R))]
    print(f"  D10 月對月留存率：{np.mean([p.sum() / max((R[k] == 9).sum(), 1) for k, p in enumerate(pr)])*100:.0f}%（隨機約 10%）")
    EV[f"D{h}"] = D

# ───── 個股訊號：昨天有「高能力分點」大買 ─────
print("\n######## 個股訊號（成立 − 不成立；原／動能調整）")
Dd = len(idx)
def grid(mask):
    Z = np.zeros((Dd, len(cols)))
    np.add.at(Z, (d[mask], j[mask]), 1)
    return pd.DataFrame(Z, index=idx, columns=cols)
for h in S.HS:
    D = EV[f"D{h}"].values
    hi, lo = grid(D == 9), grid(D == 0)
    win = np.asarray(idx >= pd.Timestamp("2022-01-01"))[:, None]
    for L in (1, 5):
        H1, L1 = hi.rolling(L, min_periods=1).sum(), lo.rolling(L, min_periods=1).sum()
        sigs = {f"排名h={h}｜近{L}日 D10 分點大買": (H1 > 0),
                f"排名h={h}｜近{L}日 D10 大買且無 D1 大買": (H1 > 0) & (L1 == 0),
                f"排名h={h}｜近{L}日 D10 大買家數 ≥ 3": (H1 >= 3)}
        for n, x in sigs.items():
            S.show(n, x.where(has & win), EXS, idx, cols)

# ───── 更嚴的控制：R20 × R120 × H2 × 20日成交值 各三分（81 格）─────
import hypotheses as Hy  # noqa: E402
print("\n######## 更嚴控制（R20×R120×H2×成交值 各三分，81 格）")
Cl = P["close"].ffill()
E0 = (el & has).values
def q3(Z):
    return np.floor(np.clip(pd.DataFrame(np.where(E0, Z, np.nan)).rank(axis=1, pct=True).values * 3 - 1e-9, 0, 2))
bk = (q3((Cl / Cl.shift(20) - 1).values) * 27 + q3((Cl / Cl.shift(120) - 1).values) * 9
      + q3(Hy.h2_hi52(P).values) * 3 + q3(P["money"].rolling(20, min_periods=15).mean().values))
EX2 = {}
for h in S.HS:
    raw, adj, E = EXS[h]
    R = raw.astype(float)
    a2 = np.full(R.shape, np.nan)
    for g in range(81):
        M = E & (bk == g)
        mu = np.where(M, R, 0).sum(axis=1) / np.maximum(M.sum(axis=1), 1)
        a2 = np.where(M, R - mu[:, None], a2)
    EX2[h] = (raw, a2.astype(np.float32), E & np.isfinite(a2))
for h in S.HS:
    D = EV[f"D{h}"].values
    hi, lo = grid(D == 9), grid(D == 0)
    S.show(f"排名h={h}｜近1日 D10 分點大買", (hi > 0).where(has & win), EX2, idx, cols)
    S.show(f"排名h={h}｜近1日 D10 大買且無 D1", ((hi > 0) & (lo == 0)).where(has & win), EX2, idx, cols)
    S.show(f"排名h={h}｜近1日 D1 分點大買（反向）", (lo > 0).where(has & win), EX2, idx, cols)
