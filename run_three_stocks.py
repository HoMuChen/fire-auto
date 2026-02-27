"""
對藍天(2362)、志超(8213)、安可(3615) 跑全部 24 策略
使用本地六年資料（2020-01-02 ~ 2026-02-26）
以風險調整後（Sharpe ratio）排名，找出前十名組合
"""

import sys
sys.path.insert(0, "/Users/mu/fire-auto")

from backtest import read_prices, simulate, STRATEGIES, format_stop_config
from pathlib import Path

DATA_DIR = Path("/Users/mu/fire-auto/data/stock_prices")

TARGETS = [
    {"stock_id": "2362", "stock_name": "藍天"},
    {"stock_id": "8213", "stock_name": "志超"},
    {"stock_id": "3615", "stock_name": "安可"},
]

def run():
    all_results = []

    for t in TARGETS:
        sid = t["stock_id"]
        name = t["stock_name"]
        prices = read_prices(sid)
        print(f"\n{sid} {name}: {prices[0]['date']} ~ {prices[-1]['date']} ({len(prices)} 筆)")

        for sname, cfg in STRATEGIES.items():
            try:
                signals = cfg["fn"](prices)
                r = simulate(prices, signals, cfg["stop"], cfg.get("exit"))
            except Exception as e:
                continue
            if r["trades"] == 0:
                continue
            r["stock_id"] = sid
            r["stock_name"] = name
            r["strategy"] = sname
            r["cat"] = cfg.get("cat", "")
            r["desc"] = cfg.get("desc", "")
            r["stop_desc"] = cfg.get("stop_desc", "")
            r["stop_type"] = format_stop_config(cfg["stop"])
            all_results.append(r)

    # 買入持有基準
    bnh = {}
    for t in TARGETS:
        sid = t["stock_id"]
        prices = read_prices(sid)
        bnh_ret = (prices[-1]["close"] - prices[0]["close"]) / prices[0]["close"] * 100
        bnh[sid] = round(bnh_ret, 1)

    print(f"\n買入持有基準（六年）:")
    for t in TARGETS:
        sid = t["stock_id"]
        print(f"  {sid} {t['stock_name']}: {bnh[sid]:+.1f}%")

    # 風險調整分數：Sharpe 主排，再考慮回撤、勝率
    def risk_score(r):
        sharpe = r["sharpe_ratio"]
        dd = r["max_drawdown_pct"]
        wr = r["win_rate"]
        trades = r["trades"]
        # 加權分數：Sharpe 為主，回撤扣分，勝率加分
        # 另外要求至少 5 筆交易才有統計意義
        if trades < 5:
            return -999
        score = sharpe - (dd / 100) * 0.5 + (wr / 100) * 0.3
        return score

    all_results.sort(key=lambda x: risk_score(x), reverse=True)

    print(f"\n全部結果: {len(all_results)} 筆")
    print()

    # 印前 20 名（含不同組合）
    header = (f"{'排名':>3} {'代號':>5} {'名稱':>4} {'策略':<22} {'類型':<4} "
              f"{'報酬率':>7} {'交易':>4} {'勝率':>6} {'持有':>5} "
              f"{'回撤':>6} {'Sharpe':>7} {'停損次':>5} {'風險分':>7}")
    print(header)
    print("-" * 125)

    shown = 0
    for r in all_results:
        sc = risk_score(r)
        if sc < -900:
            continue
        shown += 1
        if shown > 20:
            break
        print(f"{shown:>3} {r['stock_id']:>5} {r['stock_name']:>4} {r['strategy']:<22} {r['cat']:<4} "
              f"{r['total_return_pct']:>6.1f}% {r['trades']:>4} {r['win_rate']:>5.1f}% "
              f"{r['avg_holding_days']:>4.1f}d {r['max_drawdown_pct']:>5.1f}% "
              f"{r['sharpe_ratio']:>7.2f} {r['stop_loss_count']:>5} {sc:>7.3f}")

    # ── 前十名詳細 ──
    print(f"\n{'='*80}")
    print("TOP 10 風險調整後最佳組合（六年回測）")
    print(f"{'='*80}")

    top10 = [r for r in all_results if risk_score(r) > -900][:10]
    for rank, r in enumerate(top10, 1):
        sc = risk_score(r)
        print(f"\n#{rank}  {r['stock_id']} {r['stock_name']} × {r['strategy']}  [{r['cat']}]")
        print(f"    風險分: {sc:.3f}  |  報酬率: {r['total_return_pct']:+.1f}%  |  Sharpe: {r['sharpe_ratio']:.2f}")
        print(f"    交易次: {r['trades']}  |  勝率: {r['win_rate']:.1f}%  |  平均持有: {r['avg_holding_days']:.1f}天")
        print(f"    最大回撤: {r['max_drawdown_pct']:.1f}%  |  停損出場: {r['stop_loss_count']} 次")
        print(f"    策略描述: {r['desc']}")
        print(f"    停損機制: {r['stop_desc']}")

    # ── 按股票整理 ──
    print(f"\n{'='*80}")
    print("各股票最佳策略（風險調整後）")
    print(f"{'='*80}")
    for t in TARGETS:
        sid = t["stock_id"]
        name = t["stock_name"]
        stock_res = [r for r in all_results if r["stock_id"] == sid and risk_score(r) > -900]
        print(f"\n{sid} {name}（買入持有: {bnh[sid]:+.1f}%）")
        print(f"  {'排名':<3} {'策略':<22} {'類型':<4} {'報酬率':>7} {'Sharpe':>7} {'回撤':>6} {'勝率':>6} {'交易':>4}")
        for i, r in enumerate(stock_res[:5], 1):
            print(f"  {i:<3} {r['strategy']:<22} {r['cat']:<4} "
                  f"{r['total_return_pct']:>6.1f}% {r['sharpe_ratio']:>7.2f} "
                  f"{r['max_drawdown_pct']:>5.1f}% {r['win_rate']:>5.1f}% {r['trades']:>4}")


if __name__ == "__main__":
    run()
