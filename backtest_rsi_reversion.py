"""
逆勢策略回測：RSI(14) Mean Reversion
- 買進：RSI(14) < 30（超賣）
- 賣出：RSI(14) > 70（超買）
- 初始資金：1,000,000
- 交易成本：買入手續費 0.1425%，賣出手續費 0.1425% + 交易稅 0.3%
- 全部個股回測
"""

import csv
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "stock_prices"
STOCKS_PATH = BASE_DIR / "individual_stocks.json"

INITIAL_CAPITAL = 1_000_000
BUY_FEE_RATE = 0.001425       # 買入手續費 0.1425%
SELL_FEE_RATE = 0.001425       # 賣出手續費 0.1425%
SELL_TAX_RATE = 0.003          # 交易稅 0.3%

RSI_PERIOD = 14
RSI_BUY = 30
RSI_SELL = 70


def read_prices(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["close"] = float(r["close"])
    return rows


def calc_rsi(prices: list[dict], period: int = 14) -> list[float | None]:
    """計算 RSI，前 period 筆回傳 None"""
    closes = [p["close"] for p in prices]
    rsi = [None] * len(closes)
    if len(closes) <= period:
        return rsi

    gains = []
    losses = []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))

    # 初始平均
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(closes)):
        if i > period:
            avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period

        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100 - (100 / (1 + rs))

    return rsi


def backtest_one(stock_id: str) -> dict | None:
    csv_path = DATA_DIR / f"{stock_id}.csv"
    if not csv_path.exists():
        return None

    prices = read_prices(csv_path)
    if len(prices) < RSI_PERIOD + 5:
        return None

    rsi = calc_rsi(prices, RSI_PERIOD)

    capital = INITIAL_CAPITAL
    shares = 0
    buy_price = 0
    trades = []
    position_open = False

    for i in range(RSI_PERIOD, len(prices)):
        close = prices[i]["close"]
        date = prices[i]["date"]
        r = rsi[i]
        if r is None:
            continue

        if not position_open and r < RSI_BUY:
            # 買進：用全部資金買
            cost_per_share = close * (1 + BUY_FEE_RATE)
            if cost_per_share <= 0:
                continue
            shares = int(capital / cost_per_share / 1000) * 1000  # 整張（1000股）
            if shares <= 0:
                # 嘗試零股
                shares = int(capital / cost_per_share)
            if shares <= 0:
                continue
            buy_cost = shares * close
            fee = int(buy_cost * BUY_FEE_RATE)
            capital -= (buy_cost + fee)
            buy_price = close
            position_open = True

        elif position_open and r > RSI_SELL:
            # 賣出
            sell_revenue = shares * close
            fee = int(sell_revenue * SELL_FEE_RATE)
            tax = int(sell_revenue * SELL_TAX_RATE)
            capital += (sell_revenue - fee - tax)
            pnl = close / buy_price - 1
            trades.append({
                "buy_date": None,
                "sell_date": date,
                "buy_price": buy_price,
                "sell_price": close,
                "pnl_pct": pnl,
            })
            shares = 0
            position_open = False

    # 如果還有持倉，用最後一天收盤價結算
    if position_open and shares > 0:
        last_close = prices[-1]["close"]
        sell_revenue = shares * last_close
        fee = int(sell_revenue * SELL_FEE_RATE)
        tax = int(sell_revenue * SELL_TAX_RATE)
        capital += (sell_revenue - fee - tax)
        pnl = last_close / buy_price - 1
        trades.append({
            "sell_date": prices[-1]["date"],
            "buy_price": buy_price,
            "sell_price": last_close,
            "pnl_pct": pnl,
        })
        shares = 0

    if not trades:
        return None

    total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    wins = [t for t in trades if t["pnl_pct"] > 0]
    win_rate = len(wins) / len(trades) * 100 if trades else 0

    return {
        "stock_id": stock_id,
        "final_capital": round(capital),
        "total_return_pct": round(total_return, 2),
        "trades": len(trades),
        "win_rate": round(win_rate, 1),
    }


def main():
    stocks = json.loads(STOCKS_PATH.read_text())
    stock_ids = [s["stock_id"] for s in stocks]
    stock_names = {s["stock_id"]: s["stock_name"] for s in stocks}

    print(f"RSI(14) 逆勢策略回測")
    print(f"買進: RSI < {RSI_BUY} | 賣出: RSI > {RSI_SELL}")
    print(f"初始資金: {INITIAL_CAPITAL:,}")
    print(f"回測標的: {len(stock_ids)} 檔")
    print("-" * 60)

    results = []
    no_data = 0
    no_trade = 0

    for i, sid in enumerate(stock_ids):
        r = backtest_one(sid)
        if r is None:
            csv_path = DATA_DIR / f"{sid}.csv"
            if not csv_path.exists():
                no_data += 1
            else:
                no_trade += 1
        else:
            results.append(r)

        if (i + 1) % 500 == 0:
            print(f"  進度: {i+1}/{len(stock_ids)}")

    results.sort(key=lambda x: x["total_return_pct"], reverse=True)

    print(f"\n回測完成！")
    print(f"  有交易紀錄: {len(results)} 檔")
    print(f"  無股價資料: {no_data} 檔")
    print(f"  無觸發交易: {no_trade} 檔")

    # 全體統計
    if results:
        avg_ret = sum(r["total_return_pct"] for r in results) / len(results)
        positive = len([r for r in results if r["total_return_pct"] > 0])
        print(f"\n全體平均報酬: {avg_ret:.2f}%")
        print(f"正報酬檔數: {positive}/{len(results)} ({positive/len(results)*100:.1f}%)")

    # Top 10
    print(f"\n{'='*60}")
    print(f"報酬最好的 10 檔：")
    print(f"{'='*60}")
    print(f"{'排名':>4}  {'代號':>6}  {'名稱':<8}  {'報酬率':>10}  {'最終資金':>12}  {'交易次數':>6}  {'勝率':>6}")
    print(f"{'-'*60}")
    for rank, r in enumerate(results[:10], 1):
        name = stock_names.get(r["stock_id"], "")
        print(f"{rank:>4}  {r['stock_id']:>6}  {name:<8}  {r['total_return_pct']:>9.2f}%  {r['final_capital']:>11,}  {r['trades']:>6}  {r['win_rate']:>5.1f}%")


if __name__ == "__main__":
    main()
