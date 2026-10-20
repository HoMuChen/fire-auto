"""
選項 C：用「個股自身歷史中位數」當高集中度門檻（對比全市場橫斷面中位數版）。

高集中 = 信號日 T-1 買方集中度 > 該股自己「過去 W 個交易日（不含當日）」集中度中位數。
  -> 問的是「這檔比它自己平常更集中嗎（籌碼正在收斂）」，而非「比市場其他股票集中嗎」。
無前視：用 T-1 集中度 + 截至 T-1 之前的歷史中位數。
窗口穩健性：W = 120 / 250。

沿用 broker_filter_groupval 的分群、交易收集、validate_pair。
"""
import sys
sys.path.insert(0, "/Users/mu/fire-auto")
sys.path.insert(0, "/Users/mu/fire-auto/research")

import pandas as pd

import broker_concentration as bc
import broker_filter_groupval as g

TARGET_GROUPS = ["med_vol/trending", "low_vol/trending"]


def attach_own_median(trades, window):
    conc = bc.load()
    conc["date"] = pd.to_datetime(conc["date"])
    conc = conc.sort_values(["stock_id", "date"]).reset_index(drop=True)
    # 自身過去 W 天中位數（不含當日，shift(1) 排除前視）
    conc["own_med"] = (
        conc.groupby("stock_id")["buy_conc"]
        .transform(lambda s: s.shift(1).rolling(window, min_periods=60).median())
    )
    conc = conc.sort_values("date").reset_index(drop=True)
    t = trades.copy()
    t["sd"] = pd.to_datetime(t["signal_date"])
    t = t.sort_values("sd").reset_index(drop=True)
    asof = pd.merge_asof(
        t, conc.rename(columns={"date": "cdate"}),
        left_on="sd", right_on="cdate", by="stock_id",
        direction="backward", allow_exact_matches=False,
    )
    asof = asof[asof["buy_conc"].notna() & asof["own_med"].notna()].copy()
    asof["high"] = asof["buy_conc"] > asof["own_med"]
    return asof


def cell(res):
    if res is None:
        return "—"
    v = "✓" if res["verdict"] == "VALID" else ("✗" if res["verdict"] == "INVALID" else "·")
    a, b = res["period_a"], res["period_b"]
    astr = f"{a['avg_return_pct']:+.1f}({a['trade_count']})" if a else "-"
    bstr = f"{b['avg_return_pct']:+.1f}({b['trade_count']})" if b else "-"
    return f"{v} {astr} / {bstr}"


def main():
    print("建立分群 + 收集交易...")
    groups = g.build_groups()
    trades = g.collect_all_trades(groups)
    print(f"  總交易 {len(trades):,} 筆\n")

    for window in (250, 120):
        asof = attach_own_median(trades, window)
        base = g.validate_set(asof)
        high = g.validate_set(asof[asof["high"]])
        bv = sum(1 for r in base.values() if r["verdict"] == "VALID")
        hv = sum(1 for r in high.values() if r["verdict"] == "VALID")

        print("=" * 92)
        print(f"自身歷史中位數版  W={window} 天   (有效交易 {len(asof):,} 筆, 高集中 {asof['high'].sum():,} 筆)")
        print(f"VALID 配對數：基準 {bv} → 高集中(自身) {hv}")
        print("-" * 92)
        print(f"{'策略':<10} {'群組':<20} | {'基準 V A%/B%(n)':<26} | {'高集中(自身) V A%/B%(n)':<28}")
        for strat in g.STRATS:
            for grp in TARGET_GROUPS:
                k = (strat, grp)
                print(f"{strat:<10} {grp:<20} | {cell(base.get(k)):<26} | {cell(high.get(k)):<28}")
        # 各策略 trending 群組合併 高 vs 全體 每筆
        print(f"  {'-'*40}")
        for strat in g.STRATS:
            sub = asof[(asof["strategy"] == strat) & (asof["group"].isin(TARGET_GROUPS))]
            hi = sub[sub["high"]]
            if len(sub) and len(hi):
                print(f"  {strat:<10} trending合併: 全體 {sub['net_return'].mean()*100:+.2f}%(n={len(sub)})"
                      f" → 高集中 {hi['net_return'].mean()*100:+.2f}%(n={len(hi)})")
        print()


if __name__ == "__main__":
    main()
