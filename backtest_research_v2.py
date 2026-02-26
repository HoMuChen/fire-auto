"""
第二輪策略研究 — 針對志超(8213) 設計更多短波段策略
"""

import csv
import math
from pathlib import Path
from backtest_research import (
    read_prices, simulate, calc_rsi, calc_sma, calc_ema,
    calc_bollinger, calc_atr, ALL_STRATEGIES, run_all
)

DATA_DIR = Path(__file__).parent / "data" / "stock_prices"


# ─── 新策略 ───

def strategy_consecutive_red(prices, n_red=3, hold_days=5):
    """連續 N 天收黑買入，持有固定天數賣出"""
    signals = [None] * len(prices)
    pos = False
    entry_i = 0
    for i in range(n_red, len(prices)):
        if not pos:
            all_red = all(prices[i-j]["close"] < prices[i-j]["open"] for j in range(n_red))
            if all_red:
                signals[i] = "buy"
                pos = True
                entry_i = i
        elif pos and (i - entry_i) >= hold_days:
            signals[i] = "sell"
            pos = False
    return signals


def strategy_nday_low_trail(prices, lookback=10, trail_pct=0.05):
    """N日新低買入 + 移動停利(從最高點回落 trail_pct 賣出)"""
    signals = [None] * len(prices)
    pos = False
    peak = 0
    for i in range(lookback, len(prices)):
        close = prices[i]["close"]
        if not pos:
            low_n = min(prices[j]["low"] for j in range(i - lookback, i))
            if close <= low_n:
                signals[i] = "buy"
                pos = True
                peak = close
        else:
            if close > peak:
                peak = close
            if close <= peak * (1 - trail_pct):
                signals[i] = "sell"
                pos = False
    return signals


def strategy_kd_oscillator(prices, period=9, buy_th=20, sell_th=80):
    """KD隨機指標：K<buy_th且K上穿D買入，K>sell_th且K下穿D賣出"""
    closes = [p["close"] for p in prices]
    highs = [p["high"] for p in prices]
    lows = [p["low"] for p in prices]
    n = len(prices)
    rsv = [None] * n
    k_vals = [50.0] * n
    d_vals = [50.0] * n

    for i in range(period - 1, n):
        hh = max(highs[i-period+1:i+1])
        ll = min(lows[i-period+1:i+1])
        if hh == ll:
            rsv[i] = 50
        else:
            rsv[i] = (closes[i] - ll) / (hh - ll) * 100

    for i in range(period - 1, n):
        if rsv[i] is not None:
            k_vals[i] = k_vals[i-1] * 2/3 + rsv[i] * 1/3
            d_vals[i] = d_vals[i-1] * 2/3 + k_vals[i] * 1/3

    signals = [None] * n
    pos = False
    for i in range(period, n):
        if not pos and k_vals[i] < buy_th and k_vals[i] > d_vals[i] and k_vals[i-1] <= d_vals[i-1]:
            signals[i] = "buy"
            pos = True
        elif pos and k_vals[i] > sell_th and k_vals[i] < d_vals[i] and k_vals[i-1] >= d_vals[i-1]:
            signals[i] = "sell"
            pos = False
    return signals


def strategy_pct_swing(prices, buy_dip=0.04, sell_gain=0.05, stop_loss=0.03):
    """百分比波段：從近5日高點跌 buy_dip 買入，漲 sell_gain 或跌 stop_loss 賣出"""
    signals = [None] * len(prices)
    pos = False
    buy_price = 0
    for i in range(5, len(prices)):
        close = prices[i]["close"]
        if not pos:
            recent_high = max(prices[j]["high"] for j in range(i-5, i))
            if close <= recent_high * (1 - buy_dip):
                signals[i] = "buy"
                pos = True
                buy_price = close
        else:
            pnl = close / buy_price - 1
            if pnl >= sell_gain or pnl <= -stop_loss:
                signals[i] = "sell"
                pos = False
    return signals


def strategy_ema_reversion(prices, ema_period=10, dev_pct=0.03):
    """EMA均值回歸：偏離EMA超過 dev_pct 買入，回到EMA賣出"""
    closes = [p["close"] for p in prices]
    ema = calc_ema(closes, ema_period)
    signals = [None] * len(prices)
    pos = False
    for i in range(len(prices)):
        if ema[i] is None:
            continue
        deviation = (closes[i] - ema[i]) / ema[i]
        if not pos and deviation < -dev_pct:
            signals[i] = "buy"
            pos = True
        elif pos and closes[i] >= ema[i]:
            signals[i] = "sell"
            pos = False
    return signals


def strategy_rsi_short_swing(prices, rsi_period=5, buy_th=20, sell_th=60):
    """短周期RSI波段：RSI(5) < 20 買，> 60 賣"""
    closes = [p["close"] for p in prices]
    rsi = calc_rsi(closes, rsi_period)
    signals = [None] * len(prices)
    pos = False
    for i in range(len(prices)):
        if rsi[i] is None:
            continue
        if not pos and rsi[i] < buy_th:
            signals[i] = "buy"
            pos = True
        elif pos and rsi[i] > sell_th:
            signals[i] = "sell"
            pos = False
    return signals


def strategy_williams_r(prices, period=14, buy_th=-80, sell_th=-20):
    """Williams %R: < buy_th 買入，> sell_th 賣出"""
    n = len(prices)
    signals = [None] * n
    pos = False
    for i in range(period - 1, n):
        hh = max(prices[j]["high"] for j in range(i-period+1, i+1))
        ll = min(prices[j]["low"] for j in range(i-period+1, i+1))
        if hh == ll:
            wr = -50
        else:
            wr = (hh - prices[i]["close"]) / (hh - ll) * -100
        if not pos and wr < buy_th:
            signals[i] = "buy"
            pos = True
        elif pos and wr > sell_th:
            signals[i] = "sell"
            pos = False
    return signals


def strategy_double_bottom(prices, lookback=20, tolerance=0.02, confirm_pct=0.03):
    """雙底型態：在 lookback 內出現兩個接近的低點，突破頸線買入"""
    signals = [None] * len(prices)
    pos = False
    buy_price = 0
    for i in range(lookback + 5, len(prices)):
        close = prices[i]["close"]
        if not pos:
            window = [prices[j]["low"] for j in range(i-lookback, i)]
            min1_idx = window.index(min(window))
            # find second lowest not adjacent
            min2_val = float('inf')
            min2_idx = -1
            for j in range(len(window)):
                if abs(j - min1_idx) > 3 and window[j] < min2_val:
                    min2_val = window[j]
                    min2_idx = j
            if min2_idx < 0:
                continue
            low1, low2 = window[min1_idx], min2_val
            # double bottom: two lows within tolerance
            if abs(low1 - low2) / low1 < tolerance:
                neckline = max(prices[j]["high"] for j in range(i-lookback+min(min1_idx, min2_idx), i-lookback+max(min1_idx, min2_idx)+1))
                if close > neckline * (1 + confirm_pct * 0.5):
                    signals[i] = "buy"
                    pos = True
                    buy_price = close
        else:
            pnl = close / buy_price - 1
            if pnl >= 0.08 or pnl <= -0.04:
                signals[i] = "sell"
                pos = False
    return signals


def strategy_ma_bounce(prices, ma_period=20, touch_pct=0.01, target_pct=0.04, stop_pct=0.025):
    """均線支撐反彈：碰到MA附近(±touch_pct)且收紅時買入"""
    closes = [p["close"] for p in prices]
    ma = calc_sma(closes, ma_period)
    signals = [None] * len(prices)
    pos = False
    buy_price = 0
    for i in range(1, len(prices)):
        if ma[i] is None:
            continue
        close = prices[i]["close"]
        opn = prices[i]["open"]
        low = prices[i]["low"]
        if not pos:
            near_ma = abs(low - ma[i]) / ma[i] < touch_pct
            red_candle = close > opn  # 收紅
            above_ma = close > ma[i]
            if near_ma and red_candle and above_ma:
                signals[i] = "buy"
                pos = True
                buy_price = close
        else:
            pnl = close / buy_price - 1
            if pnl >= target_pct or pnl <= -stop_pct:
                signals[i] = "sell"
                pos = False
    return signals


def strategy_rsi_divergence_simple(prices, rsi_period=14, lookback=20):
    """簡易RSI背離：價格創新低但RSI未創新低時買入"""
    closes = [p["close"] for p in prices]
    rsi = calc_rsi(closes, rsi_period)
    signals = [None] * len(prices)
    pos = False
    buy_price = 0
    for i in range(lookback + rsi_period, len(prices)):
        if rsi[i] is None:
            continue
        close = prices[i]["close"]
        if not pos:
            price_low = min(closes[i-lookback:i])
            rsi_at_price_low_idx = closes[i-lookback:i].index(price_low) + (i - lookback)
            if rsi[rsi_at_price_low_idx] is None:
                continue
            # 價格創新低但 RSI 沒有更低
            if close <= price_low and rsi[i] > rsi[rsi_at_price_low_idx]:
                signals[i] = "buy"
                pos = True
                buy_price = close
        else:
            pnl = close / buy_price - 1
            if pnl >= 0.06 or pnl <= -0.04:
                signals[i] = "sell"
                pos = False
    return signals


NEW_STRATEGIES = {
    "連3紅買/持5天":            lambda p: strategy_consecutive_red(p, 3, 5),
    "連3紅買/持10天":           lambda p: strategy_consecutive_red(p, 3, 10),
    "連4紅買/持7天":            lambda p: strategy_consecutive_red(p, 4, 7),
    "10日新低+5%停利":          lambda p: strategy_nday_low_trail(p, 10, 0.05),
    "20日新低+8%停利":          lambda p: strategy_nday_low_trail(p, 20, 0.08),
    "10日新低+3%停利":          lambda p: strategy_nday_low_trail(p, 10, 0.03),
    "KD(9) 20/80":             lambda p: strategy_kd_oscillator(p, 9, 20, 80),
    "KD(9) 30/70":             lambda p: strategy_kd_oscillator(p, 9, 30, 70),
    "跌4%買/漲5%賣/停3%":       lambda p: strategy_pct_swing(p, 0.04, 0.05, 0.03),
    "跌3%買/漲4%賣/停2%":       lambda p: strategy_pct_swing(p, 0.03, 0.04, 0.02),
    "跌5%買/漲6%賣/停3%":       lambda p: strategy_pct_swing(p, 0.05, 0.06, 0.03),
    "EMA(10) 偏離3%":          lambda p: strategy_ema_reversion(p, 10, 0.03),
    "EMA(10) 偏離5%":          lambda p: strategy_ema_reversion(p, 10, 0.05),
    "EMA(20) 偏離4%":          lambda p: strategy_ema_reversion(p, 20, 0.04),
    "RSI(5) <20/>60":          lambda p: strategy_rsi_short_swing(p, 5, 20, 60),
    "RSI(5) <25/>55":          lambda p: strategy_rsi_short_swing(p, 5, 25, 55),
    "RSI(3) <15/>60":          lambda p: strategy_rsi_short_swing(p, 3, 15, 60),
    "Williams %R(14)":         lambda p: strategy_williams_r(p, 14, -80, -20),
    "Williams %R(10)":         lambda p: strategy_williams_r(p, 10, -85, -15),
    "雙底型態":                 lambda p: strategy_double_bottom(p),
    "MA(20)支撐反彈":          lambda p: strategy_ma_bounce(p, 20, 0.01, 0.04, 0.025),
    "MA(60)支撐反彈":          lambda p: strategy_ma_bounce(p, 60, 0.015, 0.05, 0.03),
    "RSI背離(14/20)":          lambda p: strategy_rsi_divergence_simple(p, 14, 20),
}


def run_stock(stock_id, stock_name, strategies):
    prices = read_prices(stock_id)
    print(f"\n{'='*75}")
    print(f"  {stock_id} {stock_name}  ({prices[0]['date']} ~ {prices[-1]['date']})")
    print(f"{'='*75}")
    print(f"{'策略':<24} {'報酬率':>8} {'交易次數':>6} {'勝率':>6} {'平均持有':>7} {'最大回撤':>8} {'Sharpe':>7}")
    print("-" * 75)

    results = []
    for name, fn in strategies.items():
        signals = fn(prices)
        r = simulate(prices, signals)
        r["name"] = name
        results.append(r)
        mark = " <<< 達標" if r["total_return_pct"] >= 50 else ""
        print(f"{name:<24} {r['total_return_pct']:>7.2f}% {r['trades']:>6} {r['win_rate']:>5.1f}% {r['avg_holding_days']:>6.1f}天 {r['max_drawdown_pct']:>7.2f}% {r['sharpe_ratio']:>7.2f}{mark}")

    results.sort(key=lambda x: x["total_return_pct"], reverse=True)
    return results


if __name__ == "__main__":
    all_strategies = {**ALL_STRATEGIES, **NEW_STRATEGIES}

    for sid, name in [("8213", "志超"), ("3615", "安可")]:
        results = run_stock(sid, name, all_strategies)
        print(f"\n  TOP 5:")
        for i, r in enumerate(results[:5], 1):
            print(f"    {i}. {r['name']}: {r['total_return_pct']:.2f}% (交易{r['trades']}次, Sharpe {r['sharpe_ratio']})")
        winners = [r for r in results if r["total_return_pct"] >= 50]
        if winners:
            print(f"\n  達標策略({len(winners)}個) ✓")
        else:
            print(f"\n  尚未達標，最佳: {results[0]['name']} = {results[0]['total_return_pct']:.2f}%")
