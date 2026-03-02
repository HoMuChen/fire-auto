"""
策略 × 股票群組驗證框架
用事前可觀察特徵（波動率、趨勢強度）將股票分群，
驗證每個「策略×群組」在全群組的平均每筆交易期望值是否為正，
並要求 2020-2022 / 2023-2026 兩個時段都成立。
"""
import csv
import json
from pathlib import Path
from math import sqrt

from backtest import (
    STRATEGIES, INITIAL_CAPITAL, BUY_FEE, SELL_FEE, SELL_TAX,
    read_prices, simulate,
)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "stock_prices"
STOCKS_PATH = BASE_DIR / "individual_stocks.json"

# ─── 設定 ───
TREND_R2_THRESHOLD = 0.3
MIN_TRADES_PER_PERIOD = 30
PERIOD_A_END = "2022-12-31"

# 交易成本
BUY_COST_FACTOR = 1 + 0.001425
SELL_COST_FACTOR = 1 - 0.001425 - 0.003


def pearson_corr(xs, ys):
    n = len(xs)
    if n < 10:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = sqrt(sum((x - mx) ** 2 for x in xs))
    sy = sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return 0.0
    return cov / (sx * sy)


def compute_stock_features(prices):
    """計算單檔股票的特徵：年化波動率、趨勢 R²"""
    closes = [p["close"] for p in prices]
    if len(closes) < 60:
        return None

    # 年化波動率
    daily_rets = []
    for i in range(1, len(closes)):
        if closes[i - 1] > 0:
            daily_rets.append(closes[i] / closes[i - 1] - 1)
    if len(daily_rets) < 2:
        return None
    avg_r = sum(daily_rets) / len(daily_rets)
    std_r = sqrt(sum((r - avg_r) ** 2 for r in daily_rets) / (len(daily_rets) - 1))
    annual_vol = std_r * sqrt(252)

    # 趨勢 R²
    x = list(range(len(closes)))
    r = pearson_corr(x, closes)
    trend_r2 = r ** 2

    return {
        "annual_vol": annual_vol,
        "trend_r2": trend_r2,
    }


def assign_groups(all_features):
    """用波動率三分位 × 趨勢二分 => 6 個群組"""
    vols = sorted(f["annual_vol"] for f in all_features.values())
    n = len(vols)
    vol_p33 = vols[n // 3]
    vol_p67 = vols[2 * n // 3]

    groups = {}
    for sid, feat in all_features.items():
        if feat["annual_vol"] <= vol_p33:
            vol_label = "low_vol"
        elif feat["annual_vol"] <= vol_p67:
            vol_label = "med_vol"
        else:
            vol_label = "high_vol"

        trend_label = "trending" if feat["trend_r2"] > TREND_R2_THRESHOLD else "sideways"
        groups[sid] = f"{vol_label}/{trend_label}"

    return groups, vol_p33, vol_p67


def compute_pair_stats(trades):
    """計算一組交易的統計指標"""
    if not trades:
        return None
    returns = [t["net_return"] for t in trades]
    n = len(returns)
    avg = sum(returns) / n
    sorted_rets = sorted(returns)
    median = sorted_rets[n // 2]
    std = sqrt(sum((r - avg) ** 2 for r in returns) / max(n - 1, 1))
    wins = sum(1 for r in returns if r > 0)
    t_stat = (avg / (std / sqrt(n))) if std > 0 and n > 1 else 0

    gross_wins = sum(r for r in returns if r > 0)
    gross_losses = abs(sum(r for r in returns if r < 0))
    pf = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    return {
        "trade_count": n,
        "avg_return_pct": round(avg * 100, 3),
        "median_return_pct": round(median * 100, 3),
        "std_return_pct": round(std * 100, 3),
        "win_rate_pct": round(wins / n * 100, 1),
        "profit_factor": round(min(pf, 99.99), 2),
        "t_statistic": round(t_stat, 2),
    }


def validate_pair(trades):
    """驗證一個 (策略, 群組) 配對"""
    period_a = [t for t in trades if t["period"] == "A"]
    period_b = [t for t in trades if t["period"] == "B"]

    overall = compute_pair_stats(trades)
    stats_a = compute_pair_stats(period_a)
    stats_b = compute_pair_stats(period_b)

    if (stats_a is None or stats_a["trade_count"] < MIN_TRADES_PER_PERIOD or
            stats_b is None or stats_b["trade_count"] < MIN_TRADES_PER_PERIOD):
        verdict = "INSUFFICIENT_DATA"
    elif stats_a["avg_return_pct"] > 0 and stats_b["avg_return_pct"] > 0:
        verdict = "VALID"
    else:
        verdict = "INVALID"

    return {
        "overall": overall,
        "period_a": stats_a,
        "period_b": stats_b,
        "verdict": verdict,
    }


def main():
    # ═══ Step 1: 載入股票 ═══
    print("=" * 100)
    print(" Step 1: 載入股票")
    print("=" * 100)

    stocks = json.load(open(STOCKS_PATH, encoding="utf-8"))
    stock_info = {s["stock_id"]: s for s in stocks}
    liquid = [s for s in stocks if s.get("low_liquidity") is False]

    candidates = []
    for s in liquid:
        sid = s["stock_id"]
        path = DATA_DIR / f"{sid}.csv"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            first = next(reader, None)
        if first and first["date"] <= "2020-01-31":
            candidates.append(s)

    print(f"  有流動性個股: {len(liquid)} 檔")
    print(f"  有 6 年資料: {len(candidates)} 檔")

    # ═══ Step 2: 計算股票特徵 ═══
    print(f"\n{'='*100}")
    print(" Step 2: 計算股票特徵（波動率、趨勢強度）")
    print("=" * 100)

    all_features = {}
    for s in candidates:
        sid = s["stock_id"]
        try:
            prices = read_prices(sid)
        except Exception:
            continue
        if len(prices) < 200 or prices[0]["close"] == 0:
            continue
        feat = compute_stock_features(prices)
        if feat:
            all_features[sid] = feat

    print(f"  成功計算特徵: {len(all_features)} 檔")

    # 特徵分佈
    vols = [f["annual_vol"] for f in all_features.values()]
    r2s = [f["trend_r2"] for f in all_features.values()]
    print(f"\n  年化波動率分佈:")
    print(f"    最小: {min(vols):.2%}  P25: {sorted(vols)[len(vols)//4]:.2%}  "
          f"中位: {sorted(vols)[len(vols)//2]:.2%}  P75: {sorted(vols)[3*len(vols)//4]:.2%}  "
          f"最大: {max(vols):.2%}")
    print(f"\n  趨勢 R² 分佈:")
    print(f"    最小: {min(r2s):.3f}  P25: {sorted(r2s)[len(r2s)//4]:.3f}  "
          f"中位: {sorted(r2s)[len(r2s)//2]:.3f}  P75: {sorted(r2s)[3*len(r2s)//4]:.3f}  "
          f"最大: {max(r2s):.3f}")
    trending_count = sum(1 for r in r2s if r > TREND_R2_THRESHOLD)
    print(f"    趨勢股 (R²>{TREND_R2_THRESHOLD}): {trending_count} 檔 ({trending_count/len(r2s)*100:.1f}%)")

    # ═══ Step 3: 分群 ═══
    print(f"\n{'='*100}")
    print(" Step 3: 分群（波動率三分位 × 趨勢二分 = 6 組）")
    print("=" * 100)

    stock_groups, vol_p33, vol_p67 = assign_groups(all_features)

    group_counts = {}
    for g in stock_groups.values():
        group_counts[g] = group_counts.get(g, 0) + 1

    print(f"  波動率切分: low ≤ {vol_p33:.2%} < med ≤ {vol_p67:.2%} < high")
    print(f"  趨勢切分: R² > {TREND_R2_THRESHOLD} = trending, 否則 sideways")
    print(f"\n  {'群組':<25} {'股票數':>6} {'佔比':>6}")
    print(f"  {'-'*40}")
    for g in sorted(group_counts.keys()):
        cnt = group_counts[g]
        print(f"  {g:<25} {cnt:>6} {cnt/len(stock_groups)*100:>5.1f}%")

    # ═══ Step 4: 跑全市場回測 ═══
    print(f"\n{'='*100}")
    print(f" Step 4: 全市場回測（{len(stock_groups)} 檔 × {len(STRATEGIES)} 策略）")
    print("=" * 100)

    all_trade_data = []
    stock_count = 0
    for s in candidates:
        sid = s["stock_id"]
        if sid not in stock_groups:
            continue

        try:
            prices = read_prices(sid)
        except Exception:
            continue
        if len(prices) < 200 or prices[0]["close"] == 0:
            continue

        stock_count += 1
        for sname, cfg in STRATEGIES.items():
            try:
                signals = cfg["fn"](prices)
                r = simulate(prices, signals, cfg["stop"], cfg.get("exit"))
            except Exception:
                continue
            if r["trades"] == 0:
                continue

            all_trade_data.append({
                "stock_id": sid,
                "strategy": sname,
                "trade_details": r.get("trade_details", []),
            })

        if stock_count % 200 == 0:
            print(f"  進度: {stock_count}/{len(stock_groups)}")

    print(f"  完成: {stock_count} 檔, {len(all_trade_data)} 個有交易的組合")

    # ═══ Step 5: 按 (策略, 群組) 整理交易 ═══
    print(f"\n{'='*100}")
    print(" Step 5: 整理交易（計算淨報酬、標記時段）")
    print("=" * 100)

    pair_trades = {}
    total_trades = 0
    for record in all_trade_data:
        sid = record["stock_id"]
        strategy = record["strategy"]
        group = stock_groups[sid]
        key = (strategy, group)

        if key not in pair_trades:
            pair_trades[key] = []

        for t in record["trade_details"]:
            if t["buy_price"] <= 0:
                continue
            net_ret = (t["sell_price"] * SELL_COST_FACTOR) / (t["buy_price"] * BUY_COST_FACTOR) - 1
            period = "A" if t["buy_date"] <= PERIOD_A_END else "B"
            pair_trades[key].append({
                "net_return": net_ret,
                "period": period,
            })
            total_trades += 1

    print(f"  策略×群組配對: {len(pair_trades)}")
    print(f"  總交易筆數: {total_trades:,}")

    # ═══ Step 6: 驗證每個配對 ═══
    print(f"\n{'='*100}")
    print(" Step 6: 驗證每個配對（前後三年都要正期望值）")
    print("=" * 100)

    results = {}
    for key, trades in pair_trades.items():
        results[key] = validate_pair(trades)

    valid = [(k, v) for k, v in results.items() if v["verdict"] == "VALID"]
    invalid = [(k, v) for k, v in results.items() if v["verdict"] == "INVALID"]
    insufficient = [(k, v) for k, v in results.items() if v["verdict"] == "INSUFFICIENT_DATA"]

    print(f"\n  結果分佈:")
    print(f"    VALID:             {len(valid):>4} 配對（兩期都有正期望值）")
    print(f"    INVALID:           {len(invalid):>4} 配對（至少一期為負）")
    print(f"    INSUFFICIENT_DATA: {len(insufficient):>4} 配對（交易筆數不足）")

    # ═══ Step 7: 輸出驗證矩陣 ═══
    print(f"\n{'='*100}")
    print(" Step 7: 驗證矩陣")
    print("=" * 100)

    # 7a. VALID 配對詳細表
    valid.sort(key=lambda x: x[1]["overall"]["avg_return_pct"], reverse=True)

    if valid:
        print(f"\n  ✓ VALID 配對（{len(valid)} 個）— 按平均每筆報酬排序:")
        print(f"  {'策略':<24} {'群組':<20} {'交易':>6} {'均報酬%':>8} {'中位%':>8} "
              f"{'勝率%':>7} {'PF':>6} {'t值':>6} | "
              f"{'A筆數':>5} {'A均%':>7} | {'B筆數':>5} {'B均%':>7}")
        print(f"  {'-'*130}")
        for (strategy, group), result in valid:
            o = result["overall"]
            a = result["period_a"]
            b = result["period_b"]
            print(f"  {strategy:<24} {group:<20} {o['trade_count']:>6} {o['avg_return_pct']:>7.2f}% "
                  f"{o['median_return_pct']:>7.2f}% {o['win_rate_pct']:>6.1f}% {o['profit_factor']:>5.2f} "
                  f"{o['t_statistic']:>6.2f} | {a['trade_count']:>5} {a['avg_return_pct']:>6.2f}% | "
                  f"{b['trade_count']:>5} {b['avg_return_pct']:>6.2f}%")
    else:
        print("\n  沒有通過驗證的配對。")

    # 7b. 完整矩陣（簡化版）
    groups_sorted = sorted(set(g for _, g in results.keys()))
    strategies_sorted = sorted(set(s for s, _ in results.keys()),
                                key=lambda s: STRATEGIES[s].get("cat", ""))

    print(f"\n  完整矩陣（均報酬% / 判定）:")
    header = f"  {'策略':<24}"
    for g in groups_sorted:
        header += f" {g:>18}"
    print(header)
    print(f"  {'-'*(24 + 19 * len(groups_sorted))}")

    for sname in strategies_sorted:
        row = f"  {sname:<24}"
        for g in groups_sorted:
            key = (sname, g)
            if key in results:
                r = results[key]
                v = r["verdict"]
                avg = r["overall"]["avg_return_pct"] if r["overall"] else 0
                if v == "VALID":
                    row += f"  \033[32m{avg:>6.2f}% ✓\033[0m    "
                elif v == "INVALID":
                    row += f"  {avg:>6.2f}% ✗     "
                else:
                    row += f"       ---        "
            else:
                row += f"       ---        "
        print(row)

    # 7c. 每個群組的推薦策略
    print(f"\n{'='*100}")
    print(" 每個群組的推薦策略")
    print("=" * 100)

    recommended = {}
    for g in groups_sorted:
        group_valid = [(s, r) for (s, grp), r in results.items()
                       if grp == g and r["verdict"] == "VALID"]
        group_valid.sort(key=lambda x: x[1]["overall"]["avg_return_pct"], reverse=True)

        recommended[g] = group_valid
        print(f"\n  【{g}】（{group_counts.get(g, 0)} 檔）")
        if group_valid:
            for rank, (sname, r) in enumerate(group_valid, 1):
                o = r["overall"]
                cat = STRATEGIES[sname].get("cat", "")
                print(f"    {rank}. {sname:<24} {cat:<4} 均報酬={o['avg_return_pct']:>6.2f}% "
                      f"勝率={o['win_rate_pct']:.0f}% PF={o['profit_factor']:.2f} "
                      f"交易={o['trade_count']}筆")
        else:
            print(f"    （無有效策略）")

    # 7d. 策略跨群組表現摘要
    print(f"\n{'='*100}")
    print(" 策略跨群組表現摘要")
    print("=" * 100)
    print(f"\n  {'策略':<24} {'類別':<4} {'VALID':>6} {'INVALID':>8} {'有效群組'}")
    print(f"  {'-'*80}")
    for sname in strategies_sorted:
        cat = STRATEGIES[sname].get("cat", "")
        v_count = sum(1 for (s, g), r in results.items() if s == sname and r["verdict"] == "VALID")
        i_count = sum(1 for (s, g), r in results.items() if s == sname and r["verdict"] == "INVALID")
        valid_groups = [g for (s, g), r in results.items() if s == sname and r["verdict"] == "VALID"]
        print(f"  {sname:<24} {cat:<4} {v_count:>6} {i_count:>8}  {', '.join(valid_groups) if valid_groups else '—'}")

    # ═══ 存 JSON ═══
    output = {
        "description": "策略 × 股票群組驗證框架",
        "methodology": {
            "stock_classification": "6 groups: {low_vol, med_vol, high_vol} x {trending, sideways}",
            "volatility_split": f"Terciles: low ≤ {vol_p33:.4f} < med ≤ {vol_p67:.4f} < high",
            "trend_split": f"R² of close vs time > {TREND_R2_THRESHOLD} = trending",
            "validation": "Positive avg per-trade return in BOTH Period A (2020-2022) and B (2023-2026)",
            "min_trades_per_period": MIN_TRADES_PER_PERIOD,
            "transaction_costs": f"Buy {BUY_FEE*100}% + Sell {SELL_FEE*100}% + Tax {SELL_TAX*100}%",
        },
        "group_distribution": {g: group_counts.get(g, 0) for g in groups_sorted},
        "summary": {
            "total_stocks": len(stock_groups),
            "total_strategies": len(STRATEGIES),
            "total_pairs_tested": len(results),
            "valid": len(valid),
            "invalid": len(invalid),
            "insufficient_data": len(insufficient),
            "total_trades_analyzed": total_trades,
        },
        "valid_pairs": [{
            "strategy": s,
            "category": STRATEGIES[s].get("cat", ""),
            "group": g,
            "overall": r["overall"],
            "period_a": r["period_a"],
            "period_b": r["period_b"],
        } for (s, g), r in valid],
        "recommended_per_group": {
            g: [{
                "strategy": s,
                "category": STRATEGIES[s].get("cat", ""),
                "avg_return_pct": r["overall"]["avg_return_pct"],
                "win_rate_pct": r["overall"]["win_rate_pct"],
                "profit_factor": r["overall"]["profit_factor"],
                "trade_count": r["overall"]["trade_count"],
            } for s, r in gv]
            for g, gv in recommended.items()
        },
    }

    out_path = BASE_DIR / "strategies" / "strategy_group_validation.json"
    json.dump(output, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n結果已存入: {out_path}")


if __name__ == "__main__":
    main()
