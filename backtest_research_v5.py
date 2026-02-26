"""
第五輪 — 專攻志超：V型反轉 + 跌深反彈 + 動態停利
"""

from pathlib import Path
from backtest_research import read_prices, simulate, calc_rsi, calc_sma, calc_ema, calc_bollinger, calc_atr


def strategy_deep_value_rebound(prices, drop_from_high=0.15, rebound_pct=0.03, trail_stop=0.05):
    """跌深反彈+移動停利：從高點跌 X% 且反彈 Y% 確認後買入，移動停利"""
    closes = [p["close"] for p in prices]
    n = len(prices)
    signals = [None] * n
    pos = False
    peak_since_buy = 0
    running_high = closes[0]

    for i in range(1, n):
        close = closes[i]
        if close > running_high:
            running_high = close
        if not pos:
            drawdown = (running_high - close) / running_high
            if drawdown >= drop_from_high:
                # 找近期最低點看是否反彈
                recent_low = min(closes[max(0, i-10):i+1])
                bounce = (close - recent_low) / recent_low if recent_low > 0 else 0
                if bounce >= rebound_pct:
                    signals[i] = "buy"
                    pos = True
                    peak_since_buy = close
        else:
            if close > peak_since_buy:
                peak_since_buy = close
            if close <= peak_since_buy * (1 - trail_stop):
                signals[i] = "sell"
                pos = False
                running_high = close
    return signals


def strategy_ma_turn(prices, ma_period=20, turn_days=3, target=0.08, stop=0.04):
    """均線轉向：MA連續下降後開始轉升時買入"""
    closes = [p["close"] for p in prices]
    ma = calc_sma(closes, ma_period)
    signals = [None] * len(prices)
    pos = False
    buy_price = 0

    for i in range(ma_period + turn_days + 5, len(prices)):
        if ma[i] is None or any(ma[i-j] is None for j in range(turn_days + 3)):
            continue
        close = closes[i]
        if not pos:
            was_falling = all(ma[i-turn_days-j-1] > ma[i-turn_days-j] for j in range(3))
            now_rising = all(ma[i-j] >= ma[i-j-1] for j in range(turn_days))
            price_above = close > ma[i]
            if was_falling and now_rising and price_above:
                signals[i] = "buy"
                pos = True
                buy_price = close
        else:
            pnl = close / buy_price - 1
            if pnl >= target or pnl <= -stop or close < ma[i] * 0.96:
                signals[i] = "sell"
                pos = False
    return signals


def strategy_higher_low(prices, lookback=30, confirm_pct=0.02, target=0.08, stop=0.04):
    """更高低點：第二個低點高於第一個低點，突破中間高點時買入"""
    closes = [p["close"] for p in prices]
    n = len(prices)
    signals = [None] * n
    pos = False
    buy_price = 0

    for i in range(lookback + 5, n):
        close = closes[i]
        if not pos:
            window = closes[i-lookback:i]
            # find two local minimums
            min_idx1 = window.index(min(window))
            # second min: search away from first
            min_val2 = float('inf')
            min_idx2 = -1
            for j in range(len(window)):
                if abs(j - min_idx1) > 5 and window[j] < min_val2:
                    min_val2 = window[j]
                    min_idx2 = j
            if min_idx2 < 0:
                continue
            # ensure chronological order
            if min_idx1 > min_idx2:
                min_idx1, min_idx2 = min_idx2, min_idx1
                min_val2 = window[min_idx2]
            low1, low2 = window[min_idx1], window[min_idx2]
            # higher low pattern
            if low2 > low1 * (1 + confirm_pct):
                mid_high = max(window[min_idx1:min_idx2+1])
                if close > mid_high:
                    signals[i] = "buy"
                    pos = True
                    buy_price = close
        else:
            pnl = close / buy_price - 1
            if pnl >= target or pnl <= -stop:
                signals[i] = "sell"
                pos = False
    return signals


def strategy_ema_recross(prices, fast=5, slow=20, pullback=True):
    """EMA再次金叉：第二次金叉更可靠，搭配移動停利"""
    closes = [p["close"] for p in prices]
    ema_f = calc_ema(closes, fast)
    ema_s = calc_ema(closes, slow)
    n = len(prices)
    signals = [None] * n
    pos = False
    cross_count = 0
    peak_price = 0

    for i in range(1, n):
        if ema_f[i] is None or ema_s[i] is None or ema_f[i-1] is None or ema_s[i-1] is None:
            continue
        # detect golden cross
        if ema_f[i-1] <= ema_s[i-1] and ema_f[i] > ema_s[i]:
            cross_count += 1
        # reset count on death cross
        if ema_f[i-1] >= ema_s[i-1] and ema_f[i] < ema_s[i]:
            if pos:
                signals[i] = "sell"
                pos = False
            cross_count = 0

        if not pos and cross_count >= (2 if pullback else 1):
            signals[i] = "buy"
            pos = True
            peak_price = closes[i]
            cross_count = 0
        elif pos:
            if closes[i] > peak_price:
                peak_price = closes[i]
            # trailing stop 6%
            if closes[i] < peak_price * 0.94:
                signals[i] = "sell"
                pos = False
    return signals


def strategy_volume_dry_up_reversal(prices, vol_shrink_days=5, vol_ratio=0.5, rsi_th=40):
    """量縮反轉：連續N天量縮至均量的X以下 + RSI低檔 → 買入"""
    closes = [p["close"] for p in prices]
    vols = [float(p["volume"]) for p in prices]
    vol_ma = calc_sma(vols, 20)
    rsi = calc_rsi(closes, 14)
    signals = [None] * len(prices)
    pos = False
    buy_price = 0

    for i in range(20 + vol_shrink_days, len(prices)):
        close = closes[i]
        if rsi[i] is None or vol_ma[i] is None:
            continue
        if not pos:
            low_vol = all(
                vol_ma[i-j] is not None and vols[i-j] < vol_ma[i-j] * vol_ratio
                for j in range(vol_shrink_days)
            )
            if low_vol and rsi[i] < rsi_th and close > prices[i-1]["close"]:
                signals[i] = "buy"
                pos = True
                buy_price = close
        else:
            pnl = close / buy_price - 1
            if pnl >= 0.06 or pnl <= -0.03 or rsi[i] > 65:
                signals[i] = "sell"
                pos = False
    return signals


def strategy_mean_revert_tight(prices, ma_period=10, entry_dev=-0.02, exit_dev=0.01, stop=0.025):
    """緊密均值回歸：偏離短均線2%買入，回到均線+1%賣出"""
    closes = [p["close"] for p in prices]
    ma = calc_sma(closes, ma_period)
    signals = [None] * len(prices)
    pos = False
    buy_price = 0

    for i in range(len(prices)):
        if ma[i] is None:
            continue
        close = closes[i]
        dev = (close - ma[i]) / ma[i]
        if not pos and dev < entry_dev:
            signals[i] = "buy"
            pos = True
            buy_price = close
        elif pos:
            pnl = close / buy_price - 1
            if dev > exit_dev or pnl <= -stop:
                signals[i] = "sell"
                pos = False
    return signals


def strategy_combined_best(prices):
    """組合策略：KD超賣 + 量縮 + 價格在MA20下方 → 買入；KD超買或獲利6% → 賣出"""
    closes = [p["close"] for p in prices]
    highs = [p["high"] for p in prices]
    lows = [p["low"] for p in prices]
    vols = [float(p["volume"]) for p in prices]
    vol_ma = calc_sma(vols, 20)
    ma20 = calc_sma(closes, 20)
    n = len(prices)

    k_vals = [50.0] * n
    d_vals = [50.0] * n
    for i in range(8, n):
        hh = max(highs[i-8:i+1])
        ll = min(lows[i-8:i+1])
        rsv = (closes[i] - ll) / (hh - ll) * 100 if hh != ll else 50
        k_vals[i] = k_vals[i-1] * 2/3 + rsv * 1/3
        d_vals[i] = d_vals[i-1] * 2/3 + k_vals[i] * 1/3

    signals = [None] * n
    pos = False
    buy_price = 0

    for i in range(20, n):
        close = closes[i]
        if vol_ma[i] is None or ma20[i] is None:
            continue
        if not pos:
            kd_low = k_vals[i] < 25
            vol_low = vols[i] < vol_ma[i] * 0.8
            below_ma = close < ma20[i]
            bullish = close > prices[i-1]["close"]  # 今天收漲
            if kd_low and (vol_low or below_ma) and bullish:
                signals[i] = "buy"
                pos = True
                buy_price = close
        else:
            pnl = close / buy_price - 1
            if k_vals[i] > 75 or pnl >= 0.06 or pnl <= -0.03:
                signals[i] = "sell"
                pos = False
    return signals


V5_STRATEGIES = {
    "跌深反彈(15%/3%/5%停利)":      lambda p: strategy_deep_value_rebound(p, 0.15, 0.03, 0.05),
    "跌深反彈(10%/2%/4%停利)":      lambda p: strategy_deep_value_rebound(p, 0.10, 0.02, 0.04),
    "跌深反彈(20%/3%/6%停利)":      lambda p: strategy_deep_value_rebound(p, 0.20, 0.03, 0.06),
    "跌深反彈(10%/2%/8%停利)":      lambda p: strategy_deep_value_rebound(p, 0.10, 0.02, 0.08),
    "跌深反彈(8%/2%/3%停利)":       lambda p: strategy_deep_value_rebound(p, 0.08, 0.02, 0.03),
    "均線轉向(MA20/3天)":           lambda p: strategy_ma_turn(p, 20, 3, 0.08, 0.04),
    "均線轉向(MA10/2天)":           lambda p: strategy_ma_turn(p, 10, 2, 0.06, 0.03),
    "均線轉向(MA20/5天)":           lambda p: strategy_ma_turn(p, 20, 5, 0.10, 0.05),
    "更高低點(30日)":               lambda p: strategy_higher_low(p, 30, 0.02, 0.08, 0.04),
    "更高低點(20日)":               lambda p: strategy_higher_low(p, 20, 0.01, 0.06, 0.03),
    "EMA再次金叉(5/20)":           lambda p: strategy_ema_recross(p, 5, 20, True),
    "EMA首次金叉+停利(5/20)":       lambda p: strategy_ema_recross(p, 5, 20, False),
    "EMA再次金叉(10/30)":          lambda p: strategy_ema_recross(p, 10, 30, True),
    "量縮反轉(5日/0.5)":           lambda p: strategy_volume_dry_up_reversal(p, 5, 0.5, 40),
    "量縮反轉(3日/0.6)":           lambda p: strategy_volume_dry_up_reversal(p, 3, 0.6, 45),
    "緊密均值回歸(MA10/-2%/+1%)":  lambda p: strategy_mean_revert_tight(p, 10, -0.02, 0.01, 0.025),
    "緊密均值回歸(MA5/-2%/+1%)":   lambda p: strategy_mean_revert_tight(p, 5, -0.02, 0.01, 0.02),
    "緊密均值回歸(MA10/-3%/+1.5%)": lambda p: strategy_mean_revert_tight(p, 10, -0.03, 0.015, 0.03),
    "KD+量縮組合策略":              lambda p: strategy_combined_best(p),
}


if __name__ == "__main__":
    for sid, name in [("8213", "志超"), ("3615", "安可")]:
        prices = read_prices(sid)
        print(f"\n{'='*78}")
        print(f"  {sid} {name}  第五輪（V型反轉 + 跌深反彈）")
        print(f"{'='*78}")
        print(f"{'策略':<30} {'報酬率':>8} {'交易次數':>6} {'勝率':>6} {'平均持有':>7} {'最大回撤':>8} {'Sharpe':>7}")
        print("-" * 78)

        results = []
        for sname, fn in V5_STRATEGIES.items():
            signals = fn(prices)
            r = simulate(prices, signals)
            r["name"] = sname
            results.append(r)
            mark = " <<< 達標" if r["total_return_pct"] >= 50 else ""
            print(f"{sname:<30} {r['total_return_pct']:>7.2f}% {r['trades']:>6} {r['win_rate']:>5.1f}% {r['avg_holding_days']:>6.1f}天 {r['max_drawdown_pct']:>7.2f}% {r['sharpe_ratio']:>7.2f}{mark}")

        results.sort(key=lambda x: x["total_return_pct"], reverse=True)
        print(f"\n  TOP 5:")
        for i, r in enumerate(results[:5], 1):
            print(f"    {i}. {r['name']}: {r['total_return_pct']:.2f}% (交易{r['trades']}次, 勝率{r['win_rate']}%, Sharpe {r['sharpe_ratio']})")
        winners = [r for r in results if r["total_return_pct"] >= 50]
        if winners:
            print(f"\n  達標策略({len(winners)}個) ✓")
        else:
            print(f"\n  尚未達標，最佳: {results[0]['name']} = {results[0]['total_return_pct']:.2f}%")
