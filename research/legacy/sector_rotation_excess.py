"""sector_rotation.py 交易的超額報酬分析：每筆 − 同期間大盤等權（EW）報酬；S2/S3 另算 − 同期間自身產業。"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sector_rotation as SR

P, ind = SR.load_panel()
members, idx = SR.industry_frames(P, ind)
c = P["close"]
mkt = (1 + c.pct_change(fill_method=None).clip(-0.105, 0.105).mean(axis=1).fillna(0)).cumprod()
dates = c.index
pos = {d: i for i, d in enumerate(dates)}
feats = {s: SR.compute_stock_features([{"close": x} for x in c[s].dropna().values]) for s in c.columns}
groups, _, _ = SR.assign_groups({k: v for k, v in feats.items() if v})
rows = []
for name in ["S1_產業動能輪動", "S2_龍頭突破_老二補漲", "S2c_對照_已跟漲成員", "S3_產業內相對超跌", "S4_輪動到落後產業", "BASE_每20日無條件"]:
    tr = pd.read_csv(SR.BASE / f"data/sector_rotation/{name}.csv", parse_dates=["entry"])
    bi = tr.entry.map(pos).values
    ei = np.minimum(bi + tr.hold.values, len(dates) - 1)
    m = mkt.values
    tr["mkt"] = m[ei - 1] / m[bi - 1] - 1          # 進場前一日收→出場前一日收（近似開盤成交）
    iv = []
    for s, b, e in zip(tr.sid, bi, ei):
        k = ind.get(s)
        iv.append(idx[k].values[e - 1] / idx[k].values[b - 1] - 1 if k in idx else np.nan)
    tr["ind"] = iv
    tr["ex_mkt"] = tr.ret - tr.mkt
    tr["ex_ind"] = tr.ret - tr.ind
    tr["grp"] = tr.sid.map(groups)
    tr["per"] = np.where(tr.entry.dt.year <= 2022, "A", "B")
    for (g, p), x in tr.groupby(["grp", "per"]):
        rows.append(dict(strategy=name, grp=g, per=p, n=len(x), ex_mkt=x.ex_mkt.mean() * 100))
    t = tr.ex_mkt.mean() / tr.ex_mkt.std() * np.sqrt(len(tr))
    yr = tr.groupby(tr.entry.dt.year).ex_mkt.mean() * 100
    print(f"{name:14s} n={len(tr):6d} 淨報酬{tr.ret.mean()*100:+.2f}% 超額vs大盤{tr.ex_mkt.mean()*100:+.2f}% (t={t:.1f}) "
          f"超額vs自身產業{tr.ex_ind.mean()*100:+.2f}%  逐年超額 " + " ".join(f"{y}:{v:+.1f}" for y, v in yr.items()))
r = pd.DataFrame(rows).pivot_table(index="strategy", columns=["grp", "per"], values="ex_mkt")
pd.set_option("display.width", 250)
print("\n各群組×期間 超額報酬 vs 大盤(%)")
print(r.round(2).to_string())
