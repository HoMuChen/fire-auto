"""
全 6 群組乾淨驗證：「高買方集中度過濾」是否提升三策略（基準 vs 過濾）。

- 群組：沿用 validate_strategy_groups 的特徵/分群（波動率三分位 × 趨勢二分 = 6 組）。
- 公平比較：只納入「有 T-1 集中度」的交易（同一筆交易池）。
    baseline = 該池全部信號；high = 池中 T-1 集中度 > 當日全市場中位數者。
- 無前視：用信號日「前一交易日」集中度（merge_asof backward, 嚴格早於）。
- 判定：沿用 validate_pair（A/B 兩期都正且各 ≥30 筆 = VALID）。
- 限制：分點從 2021-06-30 起，A 期實際為 2021H2~2022。
"""
import sys
sys.path.insert(0, "/Users/mu/fire-auto")
import csv

import pandas as pd

import validate_strategy_groups as vsg
import backtest as bt
import broker_concentration as bc

STRATS = {
    "波動率擠壓": (bt.strategy_squeeze, {"type": "trailing_pct", "pct": 0.04}),
    "超跌反彈": (bt.strategy_oversold_reversal, {"type": "trailing_pct", "pct": 0.08}),
    "AD背離": (bt.strategy_ad_divergence, {"type": "trailing_pct", "pct": 0.08}),
}


def build_groups():
    import json
    stocks = json.load(open(vsg.STOCKS_PATH, encoding="utf-8"))
    liquid = [s for s in stocks if s.get("low_liquidity") is False]
    candidates = []
    for s in liquid:
        path = vsg.DATA_DIR / f"{s['stock_id']}.csv"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            first = next(csv.DictReader(f), None)
        if first and first["date"] <= "2020-01-31":
            candidates.append(s)
    feats = {}
    for s in candidates:
        try:
            prices = vsg.read_prices(s["stock_id"])
        except Exception:
            continue
        if len(prices) < 200 or prices[0]["close"] == 0:
            continue
        f = vsg.compute_stock_features(prices)
        if f:
            feats[s["stock_id"]] = f
    groups, _, _ = vsg.assign_groups(feats)
    return groups


def collect_all_trades(groups):
    rows = []
    for sid in groups:
        try:
            prices = vsg.read_prices(sid)
        except Exception:
            continue
        if len(prices) < 200 or prices[0]["close"] == 0:
            continue
        for sname, (fn, stop) in STRATS.items():
            try:
                r = vsg.simulate(prices, fn(prices), stop, {"type": "signal"})
            except Exception:
                continue
            for t in r["trade_details"]:
                if t["buy_price"] <= 0:
                    continue
                net = t["sell_price"] * vsg.SELL_COST_FACTOR / (t["buy_price"] * vsg.BUY_COST_FACTOR) - 1
                rows.append({"stock_id": sid, "strategy": sname, "group": groups[sid],
                             "signal_date": t["buy_date"], "net_return": net,
                             "period": "A" if t["buy_date"] <= vsg.PERIOD_A_END else "B"})
    return pd.DataFrame(rows)


def attach_conc(trades):
    conc = bc.load()
    conc["date"] = pd.to_datetime(conc["date"])
    conc = conc.sort_values("date").reset_index(drop=True)
    median = conc.groupby("date")["buy_conc"].median()
    t = trades.copy()
    t["sd"] = pd.to_datetime(t["signal_date"])
    t = t.sort_values("sd").reset_index(drop=True)
    asof = pd.merge_asof(t, conc.rename(columns={"date": "cdate"}),
                         left_on="sd", right_on="cdate", by="stock_id",
                         direction="backward", allow_exact_matches=False)
    asof = asof[asof["buy_conc"].notna()].copy()
    asof["thr"] = asof["cdate"].map(median)
    asof["high"] = asof["buy_conc"] > asof["thr"]
    return asof


def validate_set(df):
    pair = {}
    for r in df.itertuples(index=False):
        pair.setdefault((r.strategy, r.group), []).append({"net_return": r.net_return, "period": r.period})
    return {k: vsg.validate_pair(v) for k, v in pair.items()}


def main():
    print("建立分群...")
    groups = build_groups()
    print(f"  分群完成: {len(groups)} 檔, {len(set(groups.values()))} 群")
    print("回測三策略收集交易...")
    trades = collect_all_trades(groups)
    print(f"  總交易: {len(trades):,} 筆")
    asof = attach_conc(trades)
    print(f"  有 T-1 集中度(納入比較): {len(asof):,} 筆 ({len(asof)/len(trades)*100:.0f}%)\n")

    base = validate_set(asof)                       # 同池全部
    high = validate_set(asof[asof["high"]])          # 高集中度

    print("=" * 104)
    print("各 (策略×群組)：基準(同池全部) vs 高集中度過濾。括號=VALID標記，A/B為每筆均報酬%")
    print("=" * 104)
    hdr = f"{'策略':<10} {'群組':<20} | {'基準 V A% / B% (n)':<32} | {'高集中 V A% / B% (n)':<32}"
    print(hdr); print("-" * 104)

    def cell(res):
        if res is None:
            return "—"
        v = "✓" if res["verdict"] == "VALID" else ("✗" if res["verdict"] == "INVALID" else "·")
        a, b = res["period_a"], res["period_b"]
        astr = f"{a['avg_return_pct']:+.1f}({a['trade_count']})" if a else "-"
        bstr = f"{b['avg_return_pct']:+.1f}({b['trade_count']})" if b else "-"
        return f"{v} {astr} / {bstr}"

    keys = sorted(set(list(base.keys()) + list(high.keys())))
    cur = None
    for k in keys:
        strat, grp = k
        if strat != cur:
            print(); cur = strat
        print(f"{strat:<10} {grp:<20} | {cell(base.get(k)):<32} | {cell(high.get(k)):<32}")

    print("\n" + "=" * 104)
    bv = sum(1 for r in base.values() if r["verdict"] == "VALID")
    hv = sum(1 for r in high.values() if r["verdict"] == "VALID")
    print(f"VALID 配對數：基準 {bv}　→　高集中度過濾 {hv}")
    # 各策略平均每筆（高集中 vs 基準，全群組合併）
    for strat in STRATS:
        bb = asof[asof["strategy"] == strat]
        hh = bb[bb["high"]]
        print(f"  {strat:<10} 全體均報酬 {bb['net_return'].mean()*100:+.2f}% (n={len(bb)})"
              f"　→　高集中 {hh['net_return'].mean()*100:+.2f}% (n={len(hh)})")


if __name__ == "__main__":
    main()
