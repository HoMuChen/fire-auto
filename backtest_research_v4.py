"""
第四輪 — 高頻短波段 + 網格交易，專攻志超(8213)
志超特性：兩年從41跌到26再漲回38，月波幅3-24%，有大量短線波段機會
"""

from pathlib import Path
from backtest_research import read_prices, simulate, calc_rsi, calc_sma, calc_ema, calc_bollinger, calc_atr


def strategy_grid_trade(prices, grid_pct=0.04, take_profit=0.05):
    """網格交易：每跌 grid_pct 買入一次，每漲 take_profit 賣出
    模擬方式：全倉進出，但用網格概念觸發"""
    closes = [p["close"] for p in prices]
    signals = [None] * len(prices)
    pos = False
    last_ref = closes[0]
    buy_price = 0

    for i in range(1, len(prices)):
        close = closes[i]
        if not pos:
            if close <= last_ref * (1 - grid_pct):
                signals[i] = "buy"
                pos = True
                buy_price = close
                last_ref = close
        else:
            if close >= buy_price * (1 + take_profit):
                signals[i] = "sell"
                pos = False
                last_ref = close
            elif close < buy_price * (1 - grid_pct):
                # 更低了，更新 ref 但不動作（已持倉）
                last_ref = close
    return signals


def strategy_quick_rsi_bounce(prices, rsi_period=3, buy_th=10, target=0.03, stop=0.02, max_hold=5):
    """極短RSI反彈：RSI(3)<10 買，目標3%或停損2%或最多持5天"""
    closes = [p["close"] for p in prices]
    rsi = calc_rsi(closes, rsi_period)
    signals = [None] * len(prices)
    pos = False
    buy_price = 0
    entry_i = 0

    for i in range(len(prices)):
        if rsi[i] is None:
            continue
        close = closes[i]
        if not pos and rsi[i] < buy_th:
            signals[i] = "buy"
            pos = True
            buy_price = close
            entry_i = i
        elif pos:
            pnl = close / buy_price - 1
            days = i - entry_i
            if pnl >= target or pnl <= -stop or days >= max_hold:
                signals[i] = "sell"
                pos = False
    return signals


def strategy_open_range_breakout(prices, n_bars=5, target=0.04, stop=0.02):
    """開盤區間突破：前N根K的高低點作為區間，突破上緣買入"""
    closes = [p["close"] for p in prices]
    signals = [None] * len(prices)
    pos = False
    buy_price = 0

    i = 0
    while i < len(prices):
        if i + n_bars >= len(prices):
            break
        # 前 N 根的區間
        range_high = max(prices[j]["high"] for j in range(i, i + n_bars))
        range_low = min(prices[j]["low"] for j in range(i, i + n_bars))

        # 之後的 K 棒看突破
        for j in range(i + n_bars, min(i + n_bars + 20, len(prices))):
            close = closes[j]
            if not pos and close > range_high:
                signals[j] = "buy"
                pos = True
                buy_price = close
            elif pos:
                pnl = close / buy_price - 1
                if pnl >= target or pnl <= -stop:
                    signals[j] = "sell"
                    pos = False
                    i = j
                    break
        else:
            if pos:
                # 超過20根沒觸發，強制賣
                signals[min(i + n_bars + 19, len(prices)-1)] = "sell"
                pos = False
            i = i + n_bars + 20
            continue
        i += 1
    return signals


def strategy_rsi_ladder(prices, levels=None, sell_rsi=55):
    """RSI 階梯：RSI 越低越買，回到 sell_rsi 以上賣出。模擬全倉但以最佳價位進場"""
    if levels is None:
        levels = [40, 30, 20]  # RSI 到這些水平觸發買入
    closes = [p["close"] for p in prices]
    rsi = calc_rsi(closes, 14)
    signals = [None] * len(prices)
    pos = False
    current_level = 0

    for i in range(len(prices)):
        if rsi[i] is None:
            continue
        if not pos and current_level < len(levels) and rsi[i] < levels[current_level]:
            signals[i] = "buy"
            pos = True
            current_level = min(current_level + 1, len(levels))
        elif pos and rsi[i] > sell_rsi:
            signals[i] = "sell"
            pos = False
    return signals


def strategy_dual_ma_rsi(prices, fast=5, slow=20, rsi_period=14, rsi_confirm=45):
    """雙均線+RSI確認：MA金叉且RSI<45確認超賣反轉時買入"""
    closes = [p["close"] for p in prices]
    ma_f = calc_sma(closes, fast)
    ma_s = calc_sma(closes, slow)
    rsi = calc_rsi(closes, rsi_period)
    signals = [None] * len(prices)
    pos = False

    for i in range(1, len(prices)):
        if ma_f[i] is None or ma_s[i] is None or ma_f[i-1] is None or ma_s[i-1] is None or rsi[i] is None:
            continue
        if not pos and ma_f[i-1] <= ma_s[i-1] and ma_f[i] > ma_s[i] and rsi[i] < rsi_confirm:
            signals[i] = "buy"
            pos = True
        elif pos and ma_f[i-1] >= ma_s[i-1] and ma_f[i] < ma_s[i]:
            signals[i] = "sell"
            pos = False
    return signals


def strategy_percent_from_low(prices, lookback=60, buy_from_low=0.05, sell_gain=0.08, stop=0.04):
    """底部反彈：從 N 日最低點反彈 X% 時買入"""
    closes = [p["close"] for p in prices]
    signals = [None] * len(prices)
    pos = False
    buy_price = 0

    for i in range(lookback, len(prices)):
        close = closes[i]
        low_n = min(closes[i-lookback:i])
        pct_from_low = (close - low_n) / low_n

        if not pos:
            if 0.03 < pct_from_low < buy_from_low:
                # 剛開始反彈
                signals[i] = "buy"
                pos = True
                buy_price = close
        else:
            pnl = close / buy_price - 1
            if pnl >= sell_gain or pnl <= -stop:
                signals[i] = "sell"
                pos = False
    return signals


def strategy_volume_reversal(prices, vol_mult=1.8, max_hold=8):
    """量能反轉：大量長下影線（潛在反轉信號）買入"""
    closes = [p["close"] for p in prices]
    vols = [float(p["volume"]) for p in prices]
    vol_ma = calc_sma(vols, 20)
    signals = [None] * len(prices)
    pos = False
    buy_price = 0
    entry_i = 0

    for i in range(20, len(prices)):
        close = prices[i]["close"]
        opn = prices[i]["open"]
        low = prices[i]["low"]
        high = prices[i]["high"]
        if vol_ma[i] is None:
            continue

        if not pos:
            body = abs(close - opn)
            lower_shadow = min(close, opn) - low
            total_range = high - low
            big_vol = vols[i] > vol_ma[i] * vol_mult
            long_lower = total_range > 0 and lower_shadow / total_range > 0.6
            bullish = close > opn

            if big_vol and long_lower and bullish:
                signals[i] = "buy"
                pos = True
                buy_price = close
                entry_i = i
        else:
            pnl = close / buy_price - 1
            days = i - entry_i
            if pnl >= 0.05 or pnl <= -0.03 or days >= max_hold:
                signals[i] = "sell"
                pos = False
    return signals


def strategy_bollinger_squeeze(prices, bb_period=20, bb_std=2, squeeze_lookback=10, target=0.06, stop=0.03):
    """布林擠壓突破：布林帶寬度達N日最窄後向上突破"""
    closes = [p["close"] for p in prices]
    upper, lower, mid = calc_bollinger(closes, bb_period, bb_std)
    signals = [None] * len(prices)
    pos = False
    buy_price = 0

    widths = [None] * len(prices)
    for i in range(len(prices)):
        if upper[i] is not None and lower[i] is not None and mid[i] and mid[i] > 0:
            widths[i] = (upper[i] - lower[i]) / mid[i]

    for i in range(bb_period + squeeze_lookback, len(prices)):
        close = closes[i]
        if widths[i] is None:
            continue
        if not pos:
            past_widths = [w for w in widths[i-squeeze_lookback:i] if w is not None]
            if not past_widths:
                continue
            if widths[i] <= min(past_widths) and close > upper[i]:
                signals[i] = "buy"
                pos = True
                buy_price = close
        else:
            pnl = close / buy_price - 1
            if pnl >= target or pnl <= -stop:
                signals[i] = "sell"
                pos = False
    return signals


def strategy_adaptive_ma(prices, fast_range=(3,10), slow_range=(15,30), eval_period=60):
    """自適應均線：根據最近表現動態選均線參數"""
    closes = [p["close"] for p in prices]
    n = len(prices)
    signals = [None] * n
    pos = False

    best_fast, best_slow = 5, 20  # defaults
    for i in range(max(slow_range[1], eval_period), n):
        # 每 eval_period 重新評估
        if (i - max(slow_range[1], eval_period)) % eval_period == 0 and i > eval_period:
            best_score = -999
            for f in range(fast_range[0], fast_range[1]+1):
                for s in range(slow_range[0], slow_range[1]+1):
                    if f >= s:
                        continue
                    ma_f = sum(closes[i-eval_period+j] for j in range(eval_period-f, eval_period)) / f
                    ma_s = sum(closes[i-eval_period+j] for j in range(eval_period-s, eval_period)) / s
                    # Simple score: if MA crossover would have been profitable
                    score = (closes[i-1] - closes[i-eval_period]) / closes[i-eval_period]
                    if ma_f > ma_s:
                        if score > best_score:
                            best_score = score
                            best_fast, best_slow = f, s

        ma_f = calc_sma(closes[:i+1], best_fast)
        ma_s = calc_sma(closes[:i+1], best_slow)
        if ma_f[i] is None or ma_s[i] is None or ma_f[i-1] is None or ma_s[i-1] is None:
            continue

        if not pos and ma_f[i-1] <= ma_s[i-1] and ma_f[i] > ma_s[i]:
            signals[i] = "buy"
            pos = True
        elif pos and ma_f[i-1] >= ma_s[i-1] and ma_f[i] < ma_s[i]:
            signals[i] = "sell"
            pos = False
    return signals


V4_STRATEGIES = {
    "網格(4%/5%)":               lambda p: strategy_grid_trade(p, 0.04, 0.05),
    "網格(3%/4%)":               lambda p: strategy_grid_trade(p, 0.03, 0.04),
    "網格(5%/6%)":               lambda p: strategy_grid_trade(p, 0.05, 0.06),
    "網格(3%/3%)":               lambda p: strategy_grid_trade(p, 0.03, 0.03),
    "網格(2%/3%)":               lambda p: strategy_grid_trade(p, 0.02, 0.03),
    "極短RSI(3)<10/3%/2%":       lambda p: strategy_quick_rsi_bounce(p, 3, 10, 0.03, 0.02, 5),
    "極短RSI(3)<15/3%/2%":       lambda p: strategy_quick_rsi_bounce(p, 3, 15, 0.03, 0.02, 5),
    "極短RSI(3)<20/4%/2%":       lambda p: strategy_quick_rsi_bounce(p, 3, 20, 0.04, 0.02, 7),
    "極短RSI(2)<10/3%/2%":       lambda p: strategy_quick_rsi_bounce(p, 2, 10, 0.03, 0.02, 5),
    "RSI階梯[40,30,20]→55":      lambda p: strategy_rsi_ladder(p, [40, 30, 20], 55),
    "RSI階梯[35,25]→50":         lambda p: strategy_rsi_ladder(p, [35, 25], 50),
    "RSI階梯[45,35,25]→55":      lambda p: strategy_rsi_ladder(p, [45, 35, 25], 55),
    "雙MA+RSI確認(5/20/45)":     lambda p: strategy_dual_ma_rsi(p, 5, 20, 14, 45),
    "雙MA+RSI確認(5/20/50)":     lambda p: strategy_dual_ma_rsi(p, 5, 20, 14, 50),
    "底部反彈(60日/5%/8%)":       lambda p: strategy_percent_from_low(p, 60, 0.05, 0.08, 0.04),
    "底部反彈(40日/5%/6%)":       lambda p: strategy_percent_from_low(p, 40, 0.05, 0.06, 0.03),
    "底部反彈(20日/4%/5%)":       lambda p: strategy_percent_from_low(p, 20, 0.04, 0.05, 0.03),
    "量能反轉(1.8x)":            lambda p: strategy_volume_reversal(p, 1.8, 8),
    "量能反轉(1.5x)":            lambda p: strategy_volume_reversal(p, 1.5, 10),
    "布林擠壓突破":                lambda p: strategy_bollinger_squeeze(p),
    "布林擠壓突破(寬)":            lambda p: strategy_bollinger_squeeze(p, 20, 2, 10, 0.08, 0.04),
}


if __name__ == "__main__":
    for sid, name in [("8213", "志超"), ("3615", "安可")]:
        prices = read_prices(sid)
        print(f"\n{'='*75}")
        print(f"  {sid} {name}  第四輪（高頻短波段 + 網格）")
        print(f"{'='*75}")
        print(f"{'策略':<28} {'報酬率':>8} {'交易次數':>6} {'勝率':>6} {'平均持有':>7} {'最大回撤':>8} {'Sharpe':>7}")
        print("-" * 78)

        results = []
        for sname, fn in V4_STRATEGIES.items():
            signals = fn(prices)
            r = simulate(prices, signals)
            r["name"] = sname
            results.append(r)
            mark = " <<< 達標" if r["total_return_pct"] >= 50 else ""
            print(f"{sname:<28} {r['total_return_pct']:>7.2f}% {r['trades']:>6} {r['win_rate']:>5.1f}% {r['avg_holding_days']:>6.1f}天 {r['max_drawdown_pct']:>7.2f}% {r['sharpe_ratio']:>7.2f}{mark}")

        results.sort(key=lambda x: x["total_return_pct"], reverse=True)
        print(f"\n  TOP 5:")
        for i, r in enumerate(results[:5], 1):
            print(f"    {i}. {r['name']}: {r['total_return_pct']:.2f}% (交易{r['trades']}次, 勝率{r['win_rate']}%, Sharpe {r['sharpe_ratio']})")
        winners = [r for r in results if r["total_return_pct"] >= 50]
        if winners:
            print(f"\n  達標策略({len(winners)}個) ✓")
        else:
            print(f"\n  尚未達標，最佳: {results[0]['name']} = {results[0]['total_return_pct']:.2f}%")
