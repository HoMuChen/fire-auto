"""
共線檢測：med_vol/trending 內，「高買方集中度」的報酬優勢是否獨立於「規模」？

方法：雙重排序(double sort)。
  規模代理 = 該股全期「每日成交值(trading_money)中位數」（越大=越大型/越流動）。
  在 med_vol/trending 內，按規模中位數分 大/小 兩半；
  每一半內再比較 高集中 vs 低集中(以信號日 T-1 集中度 vs 當日全市場中位數) 的每筆淨報酬。
  若高集中在「大、小」兩半都贏低集中 → 集中度有獨立於規模的訊號（非小股本代理）。

另報：股票層級「平均集中度 vs 規模」的 Spearman 等級相關（越接近0越無共線）。
"""
import sys
sys.path.insert(0, "/Users/mu/fire-auto")
sys.path.insert(0, "/Users/mu/fire-auto/research")
import csv

import pandas as pd

import validate_strategy_groups as vsg
import broker_filter_groupval as g


def size_proxy(sid):
    """全期每日成交值中位數（元）。"""
    path = vsg.DATA_DIR / f"{sid}.csv"
    vals = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                m = float(r["trading_money"])
                if m > 0:
                    vals.append(m)
            except Exception:
                continue
    if not vals:
        return None
    vals.sort()
    return vals[len(vals) // 2]


def main():
    groups = g.build_groups()
    trades = g.collect_all_trades(groups)
    asof = g.attach_conc(trades)
    # 只看 med_vol/trending
    mt = asof[asof["group"] == "med_vol/trending"].copy()
    print(f"med_vol/trending 有效交易: {len(mt):,} 筆\n")

    # 規模代理
    sids = mt["stock_id"].unique()
    size = {s: size_proxy(s) for s in sids}
    mt["size"] = mt["stock_id"].map(size)
    mt = mt[mt["size"].notna()].copy()
    size_med = mt["size"].median()
    mt["big"] = mt["size"] >= size_med

    def stat(d):
        return f"{d['net_return'].mean()*100:+.2f}% (n={len(d)}, 勝率{(d['net_return']>0).mean()*100:.0f}%)"

    print(f"規模切分（每日成交值中位數）門檻 = {size_med/1e8:.2f} 億元\n")
    print("雙重排序：規模 × 集中度（每筆淨報酬，全策略合併）")
    print("-" * 72)
    print(f"{'':<10} {'低集中':<30} {'高集中':<30}")
    for label, big in [("大型股", True), ("小型股", False)]:
        sub = mt[mt["big"] == big]
        lo = sub[~sub["high"]]
        hi = sub[sub["high"]]
        print(f"{label:<10} {stat(lo):<30} {stat(hi):<30}")
    print("-" * 72)

    # 各策略分別（高 vs 低，不分規模，作對照）
    print("\n各策略 high vs low（med_vol/trending，全規模）：")
    for strat in g.STRATS:
        sub = mt[mt["strategy"] == strat]
        lo, hi = sub[~sub["high"]], sub[sub["high"]]
        print(f"  {strat:<10} 低 {stat(lo):<32} 高 {stat(hi)}")

    # 股票層級相關：平均集中度 vs 規模(log)
    import math
    grp = mt.groupby("stock_id").agg(mean_conc=("buy_conc", "mean"), size=("size", "first")).dropna()
    grp = grp[grp["size"] > 0]
    grp["logsize"] = grp["size"].map(math.log)
    rho = grp["mean_conc"].rank().corr(grp["logsize"].rank())
    print(f"\n股票層級 Spearman 相關(平均集中度 vs log規模): {rho:+.3f}  "
          f"（負=大型股集中度較低；|rho|越小越無共線，n={len(grp)}）")


if __name__ == "__main__":
    main()
