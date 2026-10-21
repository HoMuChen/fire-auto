"""事件層級檢定：同一產業同日進場的成員視為一個事件（避免成員高度相關灌大 t 值），並拆產業/年份看是否集中。"""
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
ma20 = c.rolling(20).mean()


def bcast(frame):
    out = pd.DataFrame(False, index=c.index, columns=c.columns)
    for k, m in members.items():
        m = [s for s in m if s in c.columns]
        out[m] = np.repeat(frame[k].values[:, None], len(m), axis=1)
    return out


def ev_stats(tr, label):
    bi = tr.entry.map(pos).values; ei = np.minimum(bi + tr.hold.values, len(c) - 1)
    tr = tr.assign(ex=tr.ret.values - (mkt[ei - 1] / mkt[bi - 1] - 1), ind=tr.sid.map(ind))
    ev = tr.groupby(["ind", "entry"]).agg(ex=("ex", "mean"), ret=("ret", "mean"), n=("ex", "size"), hold=("hold", "mean")).reset_index()
    t = ev.ex.mean() / ev.ex.std() * np.sqrt(len(ev))
    print(f"\n### {label}: 交易 {len(tr)} 筆 / 事件 {len(ev)} 個（每年 {len(ev)/6.8:.0f}），事件平均成員 {ev.n.mean():.1f}，持有 {ev.hold.mean():.0f} 日")
    print(f"事件層級：淨報酬 {ev.ret.mean()*100:+.2f}%  超額 {ev.ex.mean()*100:+.2f}%  t={t:.2f}  超額>0 比例 {(ev.ex>0).mean():.2f}")
    print("逐年事件超額:", " ".join(f"{y}:{v*100:+.1f}({n})" for y, (v, n) in ev.groupby(ev.entry.dt.year).ex.agg(["mean", "size"]).iterrows()))
    bi_ = ev.groupby("ind").agg(ex=("ex", "mean"), n=("ex", "size")).sort_values("ex")
    print("各產業事件超額(%)：", " ".join(f"{k}:{r.ex*100:+.1f}({int(r.n)})" for k, r in bi_.iterrows()))
    # 去掉貢獻最大的 2 個產業
    top2 = ev.groupby("ind").ex.sum().nlargest(2).index
    rest = ev[~ev.ind.isin(top2)]
    print(f"去掉貢獻最大 {list(top2)}：超額 {rest.ex.mean()*100:+.2f}%  t={rest.ex.mean()/rest.ex.std()*np.sqrt(len(rest)):.2f}  事件 {len(rest)}")
    return ev


for lb, th in ((20, 0.8), (120, 0.9)):
    r = (idx / idx.shift(lb) - 1).rank(axis=1, pct=True)
    ent = bcast((r >= th) & (r.shift(1) < th)) & (c > ma20)
    ev_stats(SR.simulate(P, ent, bcast(r < 0.5), 60), f"S1 產業動能 lookback={lb} top{int((1-th)*100)}%")
tr = pd.read_csv(SR.BASE / "data/sector_rotation/S2c_對照_已跟漲成員.csv", parse_dates=["entry"])
ev_stats(tr, "S2c 龍頭突破→買已跟漲成員")
b0 = pd.DataFrame(False, index=c.index, columns=c.columns); b0.iloc[::20] = True
ev_stats(SR.simulate(P, b0, None, 40), "BASE")
