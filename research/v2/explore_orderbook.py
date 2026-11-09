"""探索（未登記）：永豐錄製的台指期五檔＋逐筆（data/orderbook，ai-dev 的 orderbook_recorder.py；本機用 rsync 同步 parquet）。

1. tick rule 正確率（對照 Shioaji tick_type 真實內外盤）＋每日價差、一檔量
2. 五檔失衡（一檔、五檔）五分位 → 未來 1／10／60／300 秒 mid 變化（點）
3. 成交流（真 Delta）與委託流 OFI（Cont–Kukanov–Stoikov 2014）的同期與下一段相關（10 秒／1 分／5 分）

用法：.venv/bin/python research/v2/explore_orderbook.py [代號前綴，預設 TXF]
"""
import glob
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
OB = Path(__file__).resolve().parents[2] / "data/orderbook"


def load(d, sym):
    t = pd.read_parquet(sorted(glob.glob(f"{OB}/{d}/tick_{sym}*.parquet"))[0])
    b = pd.read_parquet(sorted(glob.glob(f"{OB}/{d}/bidask_{sym}*.parquet"))[0])
    t = t[~t.simtrade & (t.time >= "08:45:00") & (t.time < "13:45:00")].copy()
    b = b[~b.simtrade & (b.time >= "08:45:00") & (b.time < "13:45:00")].copy()
    t["ts"], b["ts"] = pd.to_datetime(t.datetime), pd.to_datetime(b.datetime)
    return t.reset_index(drop=True), b[(b.bid_price1 > 0) & (b.ask_price1 > b.bid_price1)].sort_values("ts")


def grid(t, b):
    bv = b[[f"bid_volume{i}" for i in range(1, 6)]].values
    av = b[[f"ask_volume{i}" for i in range(1, 6)]].values
    g = pd.DataFrame(dict(mid=((b.bid_price1 + b.ask_price1) / 2).values, spr=(b.ask_price1 - b.bid_price1).values,
                          i1=(bv[:, 0] - av[:, 0]) / (bv[:, 0] + av[:, 0]), i5=(bv.sum(1) - av.sum(1)) / (bv.sum(1) + av.sum(1))),
                     index=b.ts.values)
    g = g.groupby(level=0).last().resample("1s").last().ffill()
    s = t.set_index("ts")
    g["flow"] = (s.volume * s.tick_type.map({1: 1, 2: -1}).fillna(0)).resample("1s").sum().reindex(g.index).fillna(0)
    return g


def ofi(b):
    pb, qb, pa, qa = (b[c].values for c in ("bid_price1", "bid_volume1", "ask_price1", "ask_volume1"))
    e = np.zeros(len(b))
    e[1:] = (pb[1:] >= pb[:-1]) * qb[1:] - (pb[1:] <= pb[:-1]) * qb[:-1] - (pa[1:] <= pa[:-1]) * qa[1:] + (pa[1:] >= pa[:-1]) * qa[:-1]
    return pd.Series(e, index=b.ts.values).groupby(level=0).sum()


def main(sym):
    days = sorted(Path(d).name for d in glob.glob(f"{OB}/20*") if glob.glob(f"{d}/bidask_{sym}*.parquet"))
    G, F = [], {f: [] for f in ("10s", "1min", "5min")}
    print(f"== 1. 基本（{sym}）")
    for d in days:
        t, b = load(d, sym)
        if len(t) < 1000:
            continue
        s = np.sign(t.close.diff()).replace(0, np.nan).ffill()
        tru = t.tick_type.map({1: 1, 2: -1}).fillna(0)
        k = s.notna() & (tru != 0)
        sp = b.ask_price1 - b.bid_price1
        print(f"  {d} 逐筆 {len(t):>6}｜收 {t.close.iloc[-1]:.0f}｜tick rule 正確率 {(s[k] == tru[k]).mean():.1%}"
              f"｜價差中位 {sp.median():.0f}｜一檔量中位 {b.bid_volume1.median():.0f}/{b.ask_volume1.median():.0f}")
        g = grid(t, b)
        for h in (1, 10, 60, 300):
            g[f"f{h}"] = g.mid.shift(-h) - g.mid
        G.append(g.iloc[60:-300].assign(day=d))
        o = ofi(b)
        for f in F:
            x = pd.DataFrame(dict(dm=g.mid.resample(f).last().diff(), flow=g.flow.resample(f).sum(), ofi=o.resample(f).sum())).dropna()
            F[f].append(x.assign(nxt=x.dm.shift(-1)).dropna().assign(day=d))
    G = pd.concat(G)
    print(f"\n== 2. 五檔失衡五分位 → 未來 mid 變化（點；{len(G)} 秒）")
    for v in ("i1", "i5"):
        q = pd.qcut(G[v].rank(method="first"), 5, labels=False)
        print(f"  {v}\n" + G.groupby(q)[["f1", "f10", "f60", "f300"]].mean().round(2).to_string())
    print("\n== 3. 成交流／OFI 與價格變化的相關")
    for f, xs in F.items():
        X = pd.concat(xs)
        print(f"  {f}（{len(X)} 段）" + "｜".join(f" {c} 同期 {X[c].corr(X.dm):+.2f} 下一段 {X[c].corr(X.nxt):+.3f}" for c in ("flow", "ofi"))
              + f"｜價格自身 {X.dm.corr(X.nxt):+.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "TXF")
