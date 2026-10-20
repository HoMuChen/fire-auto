"""S1 產業動能輪動 / S2c 龍頭突破跟漲 的參數穩健性：各變體每筆超額報酬（vs 大盤 EW）− BASE 超額，分 A/B 期。"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sector_rotation as SR

P, ind = SR.load_panel()
members, idx = SR.industry_frames(P, ind)
c = P["close"]
mkt = (1 + c.pct_change(fill_method=None).clip(-0.105, 0.105).mean(axis=1).fillna(0)).cumprod().values
pos = {d: i for i, d in enumerate(c.index)}
ma = {n: c.rolling(n).mean() for n in (20, 60)}
money60 = P["money"].rolling(60).mean()


def bcast(frame):  # 產業層 DataFrame → 成員層
    out = pd.DataFrame(False, index=c.index, columns=c.columns)
    for k, m in members.items():
        m = [s for s in m if s in c.columns]
        out[m] = np.repeat(frame[k].values[:, None], len(m), axis=1)
    return out


def excess(tr):
    bi = tr.entry.map(pos).values; ei = np.minimum(bi + tr.hold.values, len(c) - 1)
    ex = tr.ret.values - (mkt[ei - 1] / mkt[bi - 1] - 1)
    per = np.where(tr.entry.dt.year <= 2022, "A", "B")
    return pd.Series(ex).groupby(per).mean() * 100, ex.mean() * 100, len(tr)


b0 = pd.DataFrame(False, index=c.index, columns=c.columns)
b0.iloc[::20] = True
base = SR.simulate(P, b0, None, 40)
bA, ball, _ = excess(base)
print(f"BASE 超額 A={bA['A']:+.2f} B={bA['B']:+.2f}")
rows = []
for lb in (20, 60, 120):
    r = (idx / idx.shift(lb) - 1).rank(axis=1, pct=True)
    for th in (0.7, 0.8, 0.9):
        for trail_exit in (True, False):
            ent = bcast((r >= th) & (r.shift(1) < th)) & (c > ma[20])
            ex = bcast(r < 0.5) if trail_exit else None
            e, a, n = excess(SR.simulate(P, ent, ex, 60))
            rows.append(dict(strat="S1", lb=lb, th=th, exit="rank<0.5" if trail_exit else "trail only",
                             n=n, edgeA=e["A"] - bA["A"], edgeB=e["B"] - bA["B"]))
hh = {n: c.rolling(n).max() for n in (60, 120)}
r20 = c / c.shift(20) - 1
ir20 = idx / idx.shift(20) - 1
for win in (60, 120):
    brk = {}
    lead = pd.DataFrame(False, index=c.index, columns=c.columns)
    for k, m in members.items():
        m = [s for s in m if s in c.columns]
        mm = money60[m]; lid = mm.fillna(-1).idxmax(axis=1).where(mm.notna().any(axis=1))
        ba = (c[m] >= hh[win][m]).values; col = {s: i for i, s in enumerate(m)}
        b = pd.Series([bool(ba[i, col[l]]) if isinstance(l, str) else False for i, l in enumerate(lid.values)], index=c.index)
        brk[k] = b & ~b.shift(1).rolling(20, min_periods=1).max().fillna(0).astype(bool)
        for s in m:
            lead[s] = lid == s
    ev = bcast(pd.DataFrame(brk))
    follow = pd.DataFrame(False, index=c.index, columns=c.columns)
    for k, m in members.items():
        m = [s for s in m if s in c.columns]
        follow[m] = r20[m].ge(ir20[k], axis=0)
    for mh in (20, 40, 60):
        for flt in ("MA60", "MA20"):
            ent = ev & ~lead & (c > ma[60 if flt == "MA60" else 20]) & follow
            e, a, n = excess(SR.simulate(P, ent, None, mh))
            rows.append(dict(strat="S2c", lb=win, th=flt, exit=f"hold{mh}", n=n, edgeA=e["A"] - bA["A"], edgeB=e["B"] - bA["B"]))
out = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(out.round(2).to_string(index=False))
print("\n兩期 edge 皆 >0 的比例:", out.groupby("strat").apply(lambda x: ((x.edgeA > 0) & (x.edgeB > 0)).mean()).round(2).to_dict())
