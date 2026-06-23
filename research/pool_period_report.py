"""
三策略各自 + 分池組合，在 A期(~2022) / B期(2023~) / 全期(2020~) 的報酬與最大回撤。

分池規則（依 CLAUDE.md「方案 A」配置）：
  擠壓 : 125檔, max=7, alloc=1/7, 不限每日新倉, 不擁擠過濾, trail=4%
  超跌 : 100檔, max=7, alloc=1/7, 每日限1新倉, 同日>3信號全跳過, trail=8%
  AD   : 116檔, max=5, alloc=1/5, 每日限1新倉, 同日>3信號全跳過, trail=8%

交易來源：backtest.simulate() 逐檔（已含移動停損出場）。
投組 overlay：事件式，按交易日推進，受 max 持倉 / 每日限倉 / 擁擠跳過 限制。
資金模型：每倉位投入「進場時權益 × alloc」，複利；組合=三池各 1/3 資金獨立運行後加總。
"""
import sys
sys.path.insert(0, "/Users/mu/fire-auto")
import json
from datetime import date

import backtest as bt

A_END = "2022-12-31"

CFG = {
    "波動率擠壓": dict(key="squeeze", fn=bt.strategy_squeeze, stop={"type": "trailing_pct", "pct": 0.04},
                   maxpos=7, alloc=1/7, daily_limit=None, skip_gt=None),
    "超跌反彈":  dict(key="oversold", fn=bt.strategy_oversold_reversal, stop={"type": "trailing_pct", "pct": 0.08},
                   maxpos=7, alloc=1/7, daily_limit=1, skip_gt=3),
    "AD背離":   dict(key="ad_divergence", fn=bt.strategy_ad_divergence, stop={"type": "trailing_pct", "pct": 0.08},
                   maxpos=5, alloc=1/5, daily_limit=1, skip_gt=3),
}


def load_universe():
    d = json.load(open("/Users/mu/fire-auto/strategies/filtered_stock_lists.json"))
    return {v["label"]: [x["stock_id"] for x in v["kept"]] for v in d["strategies"].values()}


def all_trading_days():
    prices = bt.read_prices("2330")
    return [p["date"] for p in prices]


def collect_trades(label, cfg, universe):
    """回傳 list of dict: entry, exit, ret(淨)。"""
    trades = []
    for sid in universe[label]:
        try:
            prices = bt.read_prices(sid)
        except Exception:
            continue
        if len(prices) < 200 or prices[0]["close"] == 0:
            continue
        r = bt.simulate(prices, cfg["fn"](prices), cfg["stop"], {"type": "signal"})
        for t in r["trade_details"]:
            if t["buy_price"] <= 0:
                continue
            ret = t["sell_price"] * (1 - bt.SELL_FEE - bt.SELL_TAX) / (t["buy_price"] * (1 + bt.BUY_FEE)) - 1
            trades.append({"entry": t["buy_date"], "exit": t["sell_date"], "ret": ret})
    return trades


def simulate_pool(trades, cfg, days):
    """事件式分池，回傳每日權益 dict{date: equity}。權益起始 1.0。"""
    maxpos, alloc = cfg["maxpos"], cfg["alloc"]
    daily_limit, skip_gt = cfg["daily_limit"], cfg["skip_gt"]
    # 依進場日分組
    by_entry = {}
    for t in trades:
        by_entry.setdefault(t["entry"], []).append(t)

    equity = 1.0
    open_positions = []  # list of dict{exit, invested, ret}
    realized = 0.0       # 已實現累積（加回 equity 由已平倉計入）
    curve = {}
    day_index = {d: i for i, d in enumerate(days)}

    for d in days:
        # 1. 出場：當日到期的倉位結算
        still = []
        for pos in open_positions:
            if pos["exit"] == d:
                equity += pos["invested"] * pos["ret"]
            elif pos["exit"] < d:
                # 已過期（非交易日對齊問題）也結算
                equity += pos["invested"] * pos["ret"]
            else:
                still.append(pos)
        open_positions = still

        # 2. 進場
        todays = by_entry.get(d, [])
        if skip_gt is not None and len(todays) > skip_gt:
            todays = []  # 系統性下殺，全跳過
        if daily_limit is not None:
            todays = todays[:daily_limit]
        for t in todays:
            if len(open_positions) >= maxpos:
                break
            invested = equity * alloc
            open_positions.append({"exit": t["exit"], "invested": invested, "ret": t["ret"]})

        curve[d] = equity
    return curve


def period_stats(curve, days):
    """回傳 (retA, ddA, retB, ddB, retFull, ddFull) — 報酬與最大回撤(%)。"""
    series = [(d, curve[d]) for d in days]

    def maxdd(seq):
        peak = -1e9
        dd = 0.0
        for _, e in seq:
            peak = max(peak, e)
            if peak > 0:
                dd = min(dd, e / peak - 1)
        return dd * 100

    a = [(d, e) for d, e in series if d <= A_END]
    b = [(d, e) for d, e in series if d > A_END]
    eA0, eA1 = a[0][1], a[-1][1]
    eB0, eB1 = b[0][1], b[-1][1]
    e0, e1 = series[0][1], series[-1][1]
    retA = (eA1 / eA0 - 1) * 100
    retB = (eB1 / eB0 - 1) * 100
    retF = (e1 / e0 - 1) * 100
    return retA, maxdd(a), retB, maxdd(b), retF, maxdd(series)


def cagr(total_ret_pct, n_days):
    yrs = n_days / 252
    return ((1 + total_ret_pct / 100) ** (1 / yrs) - 1) * 100


def main():
    universe = load_universe()
    days = all_trading_days()
    nA = len([d for d in days if d <= A_END])
    nB = len([d for d in days if d > A_END])

    print(f"交易日：全期 {len(days)}（{days[0]}~{days[-1]}），A {nA}，B {nB}\n")
    print(f"{'策略/組合':<12} | {'A期 報酬/DD':<18} | {'B期 報酬/DD':<18} | {'全期 報酬/DD':<20} | 全期CAGR")
    print("-" * 92)

    combo_curve = {d: 0.0 for d in days}
    for label, cfg in CFG.items():
        trades = collect_trades(label, cfg, universe)
        curve = simulate_pool(trades, cfg, days)
        rA, dA, rB, dB, rF, dF = period_stats(curve, days)
        print(f"{label:<12} | {rA:>+7.1f}% / {dA:>5.1f}% | {rB:>+7.1f}% / {dB:>5.1f}% | "
              f"{rF:>+8.1f}% / {dF:>5.1f}% | {cagr(rF,len(days)):>+6.1f}%  ({len(trades)}筆)")
        # 組合：各 1/3 資金
        for d in days:
            combo_curve[d] += curve[d] / 3.0

    rA, dA, rB, dB, rF, dF = period_stats(combo_curve, days)
    print("-" * 92)
    print(f"{'分池組合':<12} | {rA:>+7.1f}% / {dA:>5.1f}% | {rB:>+7.1f}% / {dB:>5.1f}% | "
          f"{rF:>+8.1f}% / {dF:>5.1f}% | {cagr(rF,len(days)):>+6.1f}%")


if __name__ == "__main__":
    main()
