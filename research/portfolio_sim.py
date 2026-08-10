"""
三策略分池投組模擬器 — 重建遺失的 /tmp/random_50_test.py 的 triple_combo()

背景：原始腳本（combo_2022.py / stock_filter_per_strategy.py / random_50_test.py）
放在 scratchpad /tmp，已隨環境清除。本檔依 AGENTS.md「最佳策略組合／方案 A」
與 memory portfolio-simulation.md 的規格重建，復用 backtest.py 的策略函數與成本常數，
確保與存活的回測引擎一致。

核心原則（memory 反覆強調，不可違反）：
  ▸ 分池管理：三策略各自獨立資金池、獨立持倉、獨立參數。
  ▸ 絕對不可用「共享池」（所有信號丟同一池搶位置）——長持有策略會堵住短持有策略，
    嚴重低估績效（分池 +30.8% vs 共享池 +18.7%）。

方案 A 配置（med_vol/trending，各策略用各自篩選名單）：
  ┌──────────┬────┬──────┬────────┬──────────┬──────────┬────────┐
  │ 策略     │檔數│ max  │ alloc  │ 每日限倉 │系統性跳過│ 移動停損│
  ├──────────┼────┼──────┼────────┼──────────┼──────────┼────────┤
  │ 波動率擠壓│125 │  7   │ 1/7    │  不限    │   無     │  4%    │
  │ 超跌反彈  │100 │  7   │ 1/7    │  1/日    │ 同日>3全跳│  8%    │
  │ AD背離   │116 │  5   │ 1/5    │  1/日    │ 同日>3全跳│  8%    │
  └──────────┴────┴──────┴────────┴──────────┴──────────┴────────┘

  ▸ 個股篩選：Period A（2020-2022）每策略每檔平均交易報酬 > 0，
    名單存 strategies/filtered_stock_lists.json。
  ▸ 限 1 檔取哪檔：取濾網名單第一檔（對齊 scan.py / 回測）。
  ▸ 三池各 1/3 總資金，獨立管理。

參考基準（AGENTS.md 方案 A，含成本）：年化 +34.0%、DD 9.1%、Sharpe 2.58、114 筆/年。

用法：
    python3 research/portfolio_sim.py                 # 完整期間，同日收盤成交（對齊 backtest.py）
    python3 research/portfolio_sim.py --exec next_open # next-bar 成交（較保守，memory 建議）
    python3 research/portfolio_sim.py --start 2023-01-01 --end 2026-02-25  # 指定期間
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import backtest as bt  # noqa: E402  復用策略函數、指標、成本常數

BASE_DIR = Path(__file__).parent.parent
FILTERED_PATH = BASE_DIR / "strategies" / "filtered_stock_lists.json"

INITIAL_CAPITAL = 3_000_000          # 三池各 1/3 = 各 1,000,000
BUY_FEE = bt.BUY_FEE                  # 0.001425
SELL_FEE = bt.SELL_FEE               # 0.001425
SELL_TAX = bt.SELL_TAX               # 0.003

# ── 方案 A 三池配置 ──
POOLS = {
    "squeeze": {
        "label": "波動率擠壓",
        "fn": bt.strategy_squeeze,
        "list_key": "squeeze",
        "max_pos": 7,
        "trail": 0.04,
        "daily_new": None,   # 不限每日新倉
        "skip_gt": None,     # 無系統性跳過
    },
    "oversold": {
        "label": "超跌反彈",
        "fn": bt.strategy_oversold_reversal,
        "list_key": "oversold",
        "max_pos": 7,
        "trail": 0.08,
        "daily_new": 1,
        "skip_gt": 3,
    },
    "ad": {
        "label": "AD背離",
        "fn": bt.strategy_ad_divergence,
        "list_key": "ad_divergence",
        "max_pos": 5,
        "trail": 0.08,
        "daily_new": 1,
        "skip_gt": 3,
    },
}


def load_filtered_lists():
    d = json.load(open(FILTERED_PATH, encoding="utf-8"))
    out = {}
    for pool_key, cfg in POOLS.items():
        kept = d["strategies"][cfg["list_key"]]["kept"]
        out[pool_key] = [s["stock_id"] for s in kept]   # 保持名單順序（= 優先序）
    return out


def load_stock_bars(stock_id, start, end):
    """讀單檔，過濾日期範圍，回傳 [{date, open, high, low, close, volume}, ...]（升冪）。"""
    try:
        rows = bt.read_prices(stock_id)
    except FileNotFoundError:
        return None
    bars = [r for r in rows if (start is None or r["date"] >= start)
            and (end is None or r["date"] <= end) and r["close"] > 0]
    return bars if len(bars) >= 60 else None


def prep_pool(pool_key, stock_ids, start, end):
    """為一個池預先算好每檔的 bars、buy 訊號、date→idx 索引。

    注意：策略指標（MA60、squeeze 天數等）需要足夠前置資料，因此
    訊號用「完整歷史」計算後，再對齊到回測期間，避免期初 look-back 不足。
    """
    fn = POOLS[pool_key]["fn"]
    stocks = {}
    for sid in stock_ids:
        full = load_stock_bars(sid, None, end)   # 用完整歷史算指標
        if full is None:
            continue
        signals = fn(full)                        # 對齊 full 的 buy 訊號
        # 只保留回測期間的 bar，並帶上該 bar 是否為 buy 訊號
        bars = []
        for i, b in enumerate(full):
            if start is not None and b["date"] < start:
                continue
            bars.append({
                "date": b["date"], "open": b["open"], "high": b["high"],
                "low": b["low"], "close": b["close"], "volume": b["volume"],
                "buy": signals[i] == "buy",
            })
        if len(bars) < 2:
            continue
        date_idx = {b["date"]: j for j, b in enumerate(bars)}
        stocks[sid] = {"bars": bars, "date_idx": date_idx}
    return stocks


def all_dates(pools_data):
    s = set()
    for stocks in pools_data.values():
        for st in stocks.values():
            for b in st["bars"]:
                s.add(b["date"])
    return sorted(s)


def simulate_pool(pool_key, priority_ids, stocks, dates, exec_mode):
    """單一資金池的分池模擬。回傳 (equity_by_date: dict, trades: list)。

    exec_mode: 'close'    — 訊號日收盤成交（對齊 backtest.py 單檔引擎）
               'next_open'— 訊號日的次一交易日開盤成交（next-bar，較保守）
    """
    cfg = POOLS[pool_key]
    max_pos = cfg["max_pos"]
    trail = cfg["trail"]
    daily_new = cfg["daily_new"]
    skip_gt = cfg["skip_gt"]

    cash = INITIAL_CAPITAL / 3.0
    positions = {}   # sid -> {shares, avg_price, peak, entry_date}
    trades = []
    equity_by_date = {}

    # 次日成交佇列：{sid: 'buy'|'sell'}，在下一個交易日以開盤執行
    pending_buys = {}   # sid -> True（等次日開盤買）
    pending_sells = {}  # sid -> True（等次日開盤賣）

    prio_rank = {sid: r for r, sid in enumerate(priority_ids)}

    def mark_value(date):
        val = cash
        for sid, pos in positions.items():
            st = stocks.get(sid)
            if not st:
                continue
            j = st["date_idx"].get(date)
            px = st["bars"][j]["close"] if j is not None else pos["avg_price"]
            val += pos["shares"] * px
        return val

    def do_buy(sid, price, date):
        nonlocal cash
        if price <= 0:
            return
        open_slots = max_pos - len(positions)
        if open_slots <= 0:
            return
        spend = cash / open_slots           # 平均分配現金到剩餘空位
        cost_per = price * (1 + BUY_FEE)
        shares = int(spend / cost_per / 1000) * 1000
        if shares <= 0:
            shares = int(spend / cost_per)
        if shares <= 0:
            return
        buy_cost = shares * price
        fee = int(buy_cost * BUY_FEE)
        cash -= (buy_cost + fee)
        positions[sid] = {"shares": shares, "avg_price": price,
                          "peak": price, "entry_date": date}

    def do_sell(sid, price, date, reason):
        nonlocal cash
        pos = positions.pop(sid)
        rev = pos["shares"] * price
        fee = int(rev * SELL_FEE)
        tax = int(rev * SELL_TAX)
        cash += (rev - fee - tax)
        ret = (price - pos["avg_price"]) / pos["avg_price"]
        trades.append({
            "stock_id": sid, "entry_date": pos["entry_date"], "exit_date": date,
            "entry_price": round(pos["avg_price"], 2), "exit_price": round(price, 2),
            "return": round(ret, 4), "reason": reason,
        })

    for date in dates:
        # ── 0. 執行昨日掛單（next_open 模式）──
        if exec_mode == "next_open":
            for sid in list(pending_sells):
                st = stocks.get(sid)
                j = st["date_idx"].get(date) if st else None
                if j is not None and sid in positions:
                    do_sell(sid, st["bars"][j]["open"], date, "stop")
                pending_sells.pop(sid, None)
            for sid in list(pending_buys):
                st = stocks.get(sid)
                j = st["date_idx"].get(date) if st else None
                if j is not None and sid not in positions and len(positions) < max_pos:
                    do_buy(sid, st["bars"][j]["open"], date)
                pending_buys.pop(sid, None)

        # ── 1. 出場檢查（移動停損）──
        for sid in list(positions):
            st = stocks.get(sid)
            j = st["date_idx"].get(date) if st else None
            if j is None:
                continue
            close = st["bars"][j]["close"]
            pos = positions[sid]
            if close > pos["peak"]:
                pos["peak"] = close
            stop_price = pos["peak"] * (1 - trail)
            if close <= stop_price:
                if exec_mode == "close":
                    do_sell(sid, close, date, "stop")
                else:
                    pending_sells[sid] = True

        # ── 2. 進場訊號蒐集 ──
        raw_signals = []
        for sid in priority_ids:
            st = stocks.get(sid)
            if not st:
                continue
            j = st["date_idx"].get(date)
            if j is None:
                continue
            if st["bars"][j]["buy"]:
                raw_signals.append(sid)

        # 系統性下殺過濾：同日原始訊號 > skip_gt → 全跳過
        if skip_gt is not None and len(raw_signals) > skip_gt:
            raw_signals = []

        # 依優先序（濾網名單順序）取，排除已持有 / 已掛買
        candidates = [sid for sid in sorted(raw_signals, key=lambda s: prio_rank[s])
                      if sid not in positions and sid not in pending_buys]

        new_count = 0
        for sid in candidates:
            if len(positions) + len(pending_buys) >= max_pos:
                break
            if daily_new is not None and new_count >= daily_new:
                break
            st = stocks[sid]
            j = st["date_idx"][date]
            if exec_mode == "close":
                do_buy(sid, st["bars"][j]["close"], date)
            else:
                pending_buys[sid] = True
            new_count += 1

        equity_by_date[date] = mark_value(date)

    # 期末強制平倉（用最後一根收盤，標記為 artifact，計績效但可辨識）
    last_date = dates[-1]
    for sid in list(positions):
        st = stocks.get(sid)
        j = st["date_idx"].get(last_date)
        px = st["bars"][j]["close"] if j is not None else positions[sid]["avg_price"]
        do_sell(sid, px, last_date, "forced_close")
    equity_by_date[last_date] = cash

    return equity_by_date, trades


def combine_and_report(pool_results, dates, exec_mode, label_start, label_end):
    """把三池的每日淨值加總為投組淨值，算整體績效。"""
    portfolio = []
    for date in dates:
        total = sum(res["equity"].get(date, None) or res["last_known"](date)
                    for res in pool_results.values())
        portfolio.append(total)

    # 指標
    start_val = INITIAL_CAPITAL
    end_val = portfolio[-1]
    total_return = (end_val - start_val) / start_val * 100

    # 年化
    d0 = datetime.strptime(dates[0], "%Y-%m-%d")
    d1 = datetime.strptime(dates[-1], "%Y-%m-%d")
    years = (d1 - d0).days / 365.25
    cagr = ((end_val / start_val) ** (1 / years) - 1) * 100 if years > 0 else 0

    # 最大回撤
    peak = 0
    max_dd = 0
    for v in portfolio:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

    # Sharpe（日報酬）
    rets = [portfolio[i] / portfolio[i - 1] - 1 for i in range(1, len(portfolio))
            if portfolio[i - 1] > 0]
    if len(rets) > 1:
        avg = sum(rets) / len(rets)
        std = (sum((r - avg) ** 2 for r in rets) / (len(rets) - 1)) ** 0.5
        sharpe = (avg / std) * (252 ** 0.5) if std > 0 else 0
    else:
        sharpe = 0

    total_trades = sum(len(res["trades"]) for res in pool_results.values())
    trades_per_year = total_trades / years if years > 0 else 0

    print("=" * 66)
    print(f"  三策略分池投組模擬  |  {label_start} ~ {label_end}  |  成交={exec_mode}")
    print("=" * 66)
    print(f"  總報酬     : {total_return:+.1f}%")
    print(f"  年化(CAGR) : {cagr:+.1f}%")
    print(f"  最大回撤   : {max_dd * 100:.1f}%")
    print(f"  Sharpe     : {sharpe:.2f}")
    print(f"  交易/年    : {trades_per_year:.0f}  (總 {total_trades} 筆 / {years:.1f} 年)")
    print("-" * 66)
    for pk, res in pool_results.items():
        tr = res["trades"]
        wins = [t for t in tr if t["return"] > 0 and t["reason"] != "forced_close"]
        closed = [t for t in tr if t["reason"] != "forced_close"]
        wr = len(wins) / len(closed) * 100 if closed else 0
        avg_ret = (sum(t["return"] for t in closed) / len(closed) * 100) if closed else 0
        pool_end = res["equity"].get(dates[-1])
        pool_ret = (pool_end - INITIAL_CAPITAL / 3) / (INITIAL_CAPITAL / 3) * 100
        print(f"  {POOLS[pk]['label']:<8} 池報酬 {pool_ret:+7.1f}%  "
              f"交易 {len(tr):>3}  勝率 {wr:4.0f}%  均報酬/筆 {avg_ret:+.2f}%")
    print("=" * 66)

    return {
        "total_return_pct": round(total_return, 1),
        "cagr_pct": round(cagr, 1),
        "max_drawdown_pct": round(max_dd * 100, 1),
        "sharpe": round(sharpe, 2),
        "trades_per_year": round(trades_per_year),
        "total_trades": total_trades,
    }


def run(start, end, exec_mode):
    lists = load_filtered_lists()
    print(f"載入濾網名單：擠壓 {len(lists['squeeze'])} / "
          f"超跌 {len(lists['oversold'])} / AD {len(lists['ad'])} 檔")
    print("預備各池股價與訊號中...")

    pools_data = {}
    for pk in POOLS:
        pools_data[pk] = prep_pool(pk, lists[pk], start, end)
        print(f"  {POOLS[pk]['label']}：{len(pools_data[pk])} 檔有資料")

    dates = all_dates(pools_data)
    if not dates:
        print("無可用交易日，結束。")
        return
    label_start, label_end = dates[0], dates[-1]

    pool_results = {}
    for pk in POOLS:
        eq, trades = simulate_pool(pk, lists[pk], pools_data[pk], dates, exec_mode)

        # 補齊缺漏日期（用最近一次已知淨值），供加總
        last_known_val = [INITIAL_CAPITAL / 3]

        def make_last_known(eqmap):
            state = {"v": INITIAL_CAPITAL / 3}

            def f(date):
                if date in eqmap:
                    state["v"] = eqmap[date]
                return state["v"]
            return f

        pool_results[pk] = {"equity": eq, "trades": trades,
                            "last_known": make_last_known(eq)}

    metrics = combine_and_report(pool_results, dates, exec_mode, label_start, label_end)
    return metrics


def combined_daily_equity(start=None, end=None, exec_mode="close"):
    """回傳三池合併的每日淨值 (dates, values)，供相關性/混合分析用。"""
    lists = load_filtered_lists()
    pools_data = {pk: prep_pool(pk, lists[pk], start, end) for pk in POOLS}
    dates = all_dates(pools_data)
    if not dates:
        return [], []
    pool_eq = {}
    for pk in POOLS:
        eq, _ = simulate_pool(pk, lists[pk], pools_data[pk], dates, exec_mode)
        pool_eq[pk] = eq
    values = []
    last = {pk: INITIAL_CAPITAL / 3 for pk in POOLS}
    for d in dates:
        tot = 0.0
        for pk in POOLS:
            if d in pool_eq[pk]:
                last[pk] = pool_eq[pk][d]
            tot += last[pk]
        values.append(tot)
    return dates, values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None, help="回測起日 YYYY-MM-DD")
    ap.add_argument("--end", default=None, help="回測迄日 YYYY-MM-DD")
    ap.add_argument("--exec", dest="exec_mode", default="close",
                    choices=["close", "next_open"],
                    help="成交時點：close=訊號日收盤（對齊 backtest.py）；"
                         "next_open=次日開盤（next-bar，較保守）")
    args = ap.parse_args()
    run(args.start, args.end, args.exec_mode)


if __name__ == "__main__":
    main()
