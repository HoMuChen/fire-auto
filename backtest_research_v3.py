"""
第三輪 — 專攻志超(8213)
核心問題：長下跌段吃掉利潤，需要趨勢濾網 + 更精準的進出場
"""

import csv
from pathlib import Path
from backtest_research import read_prices, simulate, calc_rsi, calc_sma, calc_ema, calc_bollinger, calc_atr

DATA_DIR = Path(__file__).parent / "data" / "stock_prices"


def strategy_kd_trend_filter(prices, kd_period=9, kd_buy=30, kd_sell=70, ma_period=60):
    """KD + 均線趨勢濾網：只在價格 > MA(60) 時才買"""
    closes = [p["close"] for p in prices]
    highs = [p["high"] for p in prices]
    lows = [p["low"] for p in prices]
    ma = calc_sma(closes, ma_period)
    n = len(prices)

    k_vals = [50.0] * n
    d_vals = [50.0] * n
    for i in range(kd_period - 1, n):
        hh = max(highs[i-kd_period+1:i+1])
        ll = min(lows[i-kd_period+1:i+1])
        rsv = (closes[i] - ll) / (hh - ll) * 100 if hh != ll else 50
        k_vals[i] = k_vals[i-1] * 2/3 + rsv * 1/3
        d_vals[i] = d_vals[i-1] * 2/3 + k_vals[i] * 1/3

    signals = [None] * n
    pos = False
    for i in range(1, n):
        if ma[i] is None:
            continue
        if not pos and closes[i] > ma[i] and k_vals[i] < kd_buy and k_vals[i] > d_vals[i] and k_vals[i-1] <= d_vals[i-1]:
            signals[i] = "buy"
            pos = True
        elif pos and (k_vals[i] > kd_sell or closes[i] < ma[i] * 0.97):
            signals[i] = "sell"
            pos = False
    return signals


def strategy_rsi_trend_filter(prices, rsi_period=14, rsi_buy=35, rsi_sell=65, ma_period=60):
    """RSI + 趨勢濾網：只在 MA 之上買"""
    closes = [p["close"] for p in prices]
    rsi = calc_rsi(closes, rsi_period)
    ma = calc_sma(closes, ma_period)
    signals = [None] * len(prices)
    pos = False
    for i in range(len(prices)):
        if rsi[i] is None or ma[i] is None:
            continue
        if not pos and closes[i] > ma[i] and rsi[i] < rsi_buy:
            signals[i] = "buy"
            pos = True
        elif pos and rsi[i] > rsi_sell:
            signals[i] = "sell"
            pos = False
    return signals


def strategy_ema_cross_with_atr_stop(prices, fast=5, slow=20, atr_period=14, atr_mult=2.0):
    """EMA交叉 + ATR停損：金叉買入，跌破買入價-ATR*mult 停損，死叉賣出"""
    closes = [p["close"] for p in prices]
    ema_f = calc_ema(closes, fast)
    ema_s = calc_ema(closes, slow)
    atr = calc_atr(prices, atr_period)
    signals = [None] * len(prices)
    pos = False
    buy_price = 0
    buy_atr = 0
    for i in range(1, len(prices)):
        if ema_f[i] is None or ema_s[i] is None or ema_f[i-1] is None or ema_s[i-1] is None:
            continue
        close = closes[i]
        if not pos and ema_f[i-1] <= ema_s[i-1] and ema_f[i] > ema_s[i]:
            signals[i] = "buy"
            pos = True
            buy_price = close
            buy_atr = atr[i] if atr[i] else close * 0.02
        elif pos:
            stop = buy_price - buy_atr * atr_mult
            if close < stop or (ema_f[i-1] >= ema_s[i-1] and ema_f[i] < ema_s[i]):
                signals[i] = "sell"
                pos = False
    return signals


def strategy_breakout_consolidation(prices, lookback=20, vol_shrink=0.6, breakout_pct=0.02):
    """盤整突破：波動率收縮後向上突破"""
    closes = [p["close"] for p in prices]
    n = len(prices)
    signals = [None] * n
    pos = False
    buy_price = 0

    for i in range(lookback * 2, n):
        window = closes[i-lookback:i]
        prev_window = closes[i-lookback*2:i-lookback]

        range_now = (max(window) - min(window)) / min(window)
        range_prev = (max(prev_window) - min(prev_window)) / min(prev_window)

        if not pos:
            # 波動收縮 + 向上突破
            if range_now < range_prev * vol_shrink:
                upper = max(window)
                if closes[i] > upper * (1 + breakout_pct):
                    signals[i] = "buy"
                    pos = True
                    buy_price = closes[i]
        else:
            pnl = closes[i] / buy_price - 1
            if pnl >= 0.08 or pnl <= -0.04:
                signals[i] = "sell"
                pos = False
    return signals


def strategy_multi_signal(prices):
    """多指標共振：RSI超賣 + 收紅K + 量增，三者同時出現才買"""
    closes = [p["close"] for p in prices]
    vols = [float(p["volume"]) for p in prices]
    rsi = calc_rsi(closes, 14)
    vol_ma = calc_sma(vols, 20)
    signals = [None] * len(prices)
    pos = False
    buy_price = 0

    for i in range(20, len(prices)):
        close = prices[i]["close"]
        opn = prices[i]["open"]
        if rsi[i] is None or vol_ma[i] is None:
            continue
        if not pos:
            oversold = rsi[i] < 40
            bullish_candle = close > opn
            vol_up = vols[i] > vol_ma[i] * 1.3
            if oversold and bullish_candle and vol_up:
                signals[i] = "buy"
                pos = True
                buy_price = close
        else:
            pnl = close / buy_price - 1
            if pnl >= 0.06 or pnl <= -0.03 or rsi[i] > 70:
                signals[i] = "sell"
                pos = False
    return signals


def strategy_pullback_buy(prices, ma_period=20, pullback_days=3, target=0.05, stop=0.03):
    """回檔買入：在上升趨勢中，連續回檔N天後買入"""
    closes = [p["close"] for p in prices]
    ma = calc_sma(closes, ma_period)
    signals = [None] * len(prices)
    pos = False
    buy_price = 0

    for i in range(ma_period + pullback_days, len(prices)):
        close = prices[i]["close"]
        if ma[i] is None:
            continue
        if not pos:
            uptrend = ma[i] > ma[i-5] if ma[i-5] else False  # MA 向上
            pullback = all(closes[i-j] < closes[i-j-1] for j in range(pullback_days))
            near_ma = close > ma[i] * 0.98 and close < ma[i] * 1.03
            if uptrend and pullback and near_ma:
                signals[i] = "buy"
                pos = True
                buy_price = close
        else:
            pnl = close / buy_price - 1
            if pnl >= target or pnl <= -stop:
                signals[i] = "sell"
                pos = False
    return signals


def strategy_range_trade(prices, lookback=40, buy_pct=0.15, sell_pct=0.85):
    """區間交易：在N日區間的低檔買入，高檔賣出"""
    closes = [p["close"] for p in prices]
    signals = [None] * len(prices)
    pos = False

    for i in range(lookback, len(prices)):
        window = closes[i-lookback:i]
        hi = max(window)
        lo = min(window)
        if hi == lo:
            continue
        position_in_range = (closes[i] - lo) / (hi - lo)
        if not pos and position_in_range < buy_pct:
            signals[i] = "buy"
            pos = True
        elif pos and position_in_range > sell_pct:
            signals[i] = "sell"
            pos = False
    return signals


def strategy_chandelier_exit(prices, entry_period=20, atr_period=14, atr_mult=3.0):
    """Chandelier Exit：突破N日高點買入，用ATR做移動停損"""
    closes = [p["close"] for p in prices]
    atr = calc_atr(prices, atr_period)
    n = len(prices)
    signals = [None] * n
    pos = False
    highest_since_buy = 0

    for i in range(entry_period, n):
        close = closes[i]
        if atr[i] is None:
            continue
        if not pos:
            high_n = max(prices[j]["high"] for j in range(i-entry_period, i))
            if close > high_n:
                signals[i] = "buy"
                pos = True
                highest_since_buy = close
        else:
            if close > highest_since_buy:
                highest_since_buy = close
            stop = highest_since_buy - atr[i] * atr_mult
            if close < stop:
                signals[i] = "sell"
                pos = False
    return signals


def strategy_turtle_variant(prices, entry_period=20, exit_period=10):
    """海龜變體：突破20日高買入，跌破10日低賣出"""
    closes = [p["close"] for p in prices]
    n = len(prices)
    signals = [None] * n
    pos = False

    for i in range(max(entry_period, exit_period), n):
        close = closes[i]
        if not pos:
            high_n = max(prices[j]["high"] for j in range(i-entry_period, i))
            if close > high_n:
                signals[i] = "buy"
                pos = True
        else:
            low_n = min(prices[j]["low"] for j in range(i-exit_period, i))
            if close < low_n:
                signals[i] = "sell"
                pos = False
    return signals


def strategy_weekly_momentum(prices, up_weeks=2, hold_days=10):
    """週動能：連續N週收漲後買入，持有固定天數"""
    closes = [p["close"] for p in prices]
    n = len(prices)
    signals = [None] * n
    pos = False
    entry_i = 0

    # 計算每週報酬（每5個交易日）
    weekly_ret = [None] * n
    for i in range(5, n):
        weekly_ret[i] = closes[i] / closes[i-5] - 1

    for i in range(5 * up_weeks, n):
        if not pos:
            # 檢查連續 N 週都是正報酬
            all_up = True
            for w in range(up_weeks):
                idx = i - w * 5
                if idx < 0 or weekly_ret[idx] is None or weekly_ret[idx] <= 0:
                    all_up = False
                    break
            if all_up:
                signals[i] = "buy"
                pos = True
                entry_i = i
        elif pos and (i - entry_i) >= hold_days:
            signals[i] = "sell"
            pos = False
    return signals


def strategy_inside_bar_breakout(prices):
    """內包線突破：今日高低被昨日包住(inside bar)，隔天突破買入"""
    closes = [p["close"] for p in prices]
    n = len(prices)
    signals = [None] * n
    pos = False
    buy_price = 0

    for i in range(2, n):
        if not pos:
            # 前一天是 inside bar
            prev_inside = (prices[i-1]["high"] <= prices[i-2]["high"] and
                          prices[i-1]["low"] >= prices[i-2]["low"])
            if prev_inside and closes[i] > prices[i-2]["high"]:
                signals[i] = "buy"
                pos = True
                buy_price = closes[i]
        else:
            pnl = closes[i] / buy_price - 1
            if pnl >= 0.05 or pnl <= -0.025:
                signals[i] = "sell"
                pos = False
    return signals


V3_STRATEGIES = {
    "KD+MA60濾網 30/70":           lambda p: strategy_kd_trend_filter(p, 9, 30, 70, 60),
    "KD+MA20濾網 30/70":           lambda p: strategy_kd_trend_filter(p, 9, 30, 70, 20),
    "KD+MA60濾網 20/80":           lambda p: strategy_kd_trend_filter(p, 9, 20, 80, 60),
    "RSI+MA60濾網 35/65":          lambda p: strategy_rsi_trend_filter(p, 14, 35, 65, 60),
    "RSI+MA20濾網 40/60":          lambda p: strategy_rsi_trend_filter(p, 14, 40, 60, 20),
    "EMA(5/20)+ATR停損":           lambda p: strategy_ema_cross_with_atr_stop(p, 5, 20, 14, 2.0),
    "EMA(5/20)+ATR停損(1.5)":      lambda p: strategy_ema_cross_with_atr_stop(p, 5, 20, 14, 1.5),
    "盤整突破(20日)":               lambda p: strategy_breakout_consolidation(p, 20, 0.6, 0.02),
    "盤整突破(15日)":               lambda p: strategy_breakout_consolidation(p, 15, 0.5, 0.015),
    "多指標共振":                    lambda p: strategy_multi_signal(p),
    "回檔買入(MA20/3日)":           lambda p: strategy_pullback_buy(p, 20, 3, 0.05, 0.03),
    "回檔買入(MA20/2日)":           lambda p: strategy_pullback_buy(p, 20, 2, 0.04, 0.025),
    "區間交易(40日)":               lambda p: strategy_range_trade(p, 40, 0.15, 0.85),
    "區間交易(20日)":               lambda p: strategy_range_trade(p, 20, 0.15, 0.85),
    "區間交易(60日)":               lambda p: strategy_range_trade(p, 60, 0.10, 0.90),
    "Chandelier(20/3ATR)":        lambda p: strategy_chandelier_exit(p, 20, 14, 3.0),
    "Chandelier(20/2ATR)":        lambda p: strategy_chandelier_exit(p, 20, 14, 2.0),
    "海龜變體(20/10)":              lambda p: strategy_turtle_variant(p, 20, 10),
    "海龜變體(15/7)":               lambda p: strategy_turtle_variant(p, 15, 7),
    "週動能(2週/10天)":             lambda p: strategy_weekly_momentum(p, 2, 10),
    "週動能(2週/15天)":             lambda p: strategy_weekly_momentum(p, 2, 15),
    "內包線突破":                    lambda p: strategy_inside_bar_breakout(p),
}


if __name__ == "__main__":
    for sid, name in [("8213", "志超"), ("3615", "安可")]:
        prices = read_prices(sid)
        print(f"\n{'='*75}")
        print(f"  {sid} {name}  第三輪策略")
        print(f"{'='*75}")
        print(f"{'策略':<26} {'報酬率':>8} {'交易次數':>6} {'勝率':>6} {'平均持有':>7} {'最大回撤':>8} {'Sharpe':>7}")
        print("-" * 75)

        results = []
        for sname, fn in V3_STRATEGIES.items():
            signals = fn(prices)
            r = simulate(prices, signals)
            r["name"] = sname
            results.append(r)
            mark = " <<< 達標" if r["total_return_pct"] >= 50 else ""
            print(f"{sname:<26} {r['total_return_pct']:>7.2f}% {r['trades']:>6} {r['win_rate']:>5.1f}% {r['avg_holding_days']:>6.1f}天 {r['max_drawdown_pct']:>7.2f}% {r['sharpe_ratio']:>7.2f}{mark}")

        results.sort(key=lambda x: x["total_return_pct"], reverse=True)
        print(f"\n  TOP 5:")
        for i, r in enumerate(results[:5], 1):
            print(f"    {i}. {r['name']}: {r['total_return_pct']:.2f}% (交易{r['trades']}次, Sharpe {r['sharpe_ratio']})")
        winners = [r for r in results if r["total_return_pct"] >= 50]
        if winners:
            print(f"\n  達標策略({len(winners)}個) ✓")
        else:
            print(f"\n  尚未達標，最佳: {results[0]['name']} = {results[0]['total_return_pct']:.2f}%")
