"""
第六輪 — 最後一搏志超
思路：
1. 高頻小波段複利（每次賺2-3%，靠交易次數堆疊）
2. 趨勢過濾+波段（避開下跌期，只在反彈期操作）
3. 多條件嚴格過濾進場
"""

from pathlib import Path
from backtest_research import read_prices, simulate, calc_rsi, calc_sma, calc_ema, calc_bollinger, calc_atr


def strategy_micro_swing(prices, entry_drop=0.02, take_profit=0.025, stop_loss=0.015, cooldown=2):
    """微波段：日跌幅>X%買入，賺Y%或虧Z%出場，冷卻N天"""
    closes = [p["close"] for p in prices]
    signals = [None] * len(prices)
    pos = False
    buy_price = 0
    last_sell_i = -100

    for i in range(1, len(prices)):
        close = closes[i]
        prev_close = closes[i-1]
        daily_ret = (close - prev_close) / prev_close

        if not pos and i - last_sell_i > cooldown:
            if daily_ret <= -entry_drop:
                signals[i] = "buy"
                pos = True
                buy_price = close
        elif pos:
            pnl = close / buy_price - 1
            if pnl >= take_profit or pnl <= -stop_loss:
                signals[i] = "sell"
                pos = False
                last_sell_i = i
    return signals


def strategy_oversold_bounce_strict(prices):
    """嚴格超賣反彈：RSI(5)<15 + 收紅 + 成交量>昨天 → 買入，RSI>50 或 5天後賣出"""
    closes = [p["close"] for p in prices]
    vols = [float(p["volume"]) for p in prices]
    rsi = calc_rsi(closes, 5)
    signals = [None] * len(prices)
    pos = False
    entry_i = 0

    for i in range(5, len(prices)):
        close = closes[i]
        opn = prices[i]["open"]
        if rsi[i] is None:
            continue
        if not pos:
            bullish = close > opn
            vol_up = vols[i] > vols[i-1]
            rsi_low = rsi[i] < 15
            if rsi_low and bullish and vol_up:
                signals[i] = "buy"
                pos = True
                entry_i = i
        elif pos:
            days = i - entry_i
            if rsi[i] > 50 or days >= 5:
                signals[i] = "sell"
                pos = False
    return signals


def strategy_trend_only(prices, trend_ma=60, entry_ma=10, entry_rsi=40, exit_rsi=65):
    """只在上升趨勢中交易：MA60上升 + 回檔到RSI<40買入 + RSI>65賣出"""
    closes = [p["close"] for p in prices]
    rsi = calc_rsi(closes, 14)
    ma_long = calc_sma(closes, trend_ma)
    ma_short = calc_sma(closes, entry_ma)
    signals = [None] * len(prices)
    pos = False

    for i in range(trend_ma + 5, len(prices)):
        close = closes[i]
        if rsi[i] is None or ma_long[i] is None or ma_long[i-5] is None:
            continue
        uptrend = ma_long[i] > ma_long[i-5]  # MA60 往上
        if not pos:
            if uptrend and rsi[i] < entry_rsi and close > ma_long[i]:
                signals[i] = "buy"
                pos = True
        elif pos:
            if rsi[i] > exit_rsi or close < ma_long[i] * 0.96:
                signals[i] = "sell"
                pos = False
    return signals


def strategy_two_phase(prices):
    """兩階段策略：
    階段1(下跌期): 完全不交易
    階段2(上漲期): 積極波段操作
    判斷依據：MA20 > MA60 時進入上漲期"""
    closes = [p["close"] for p in prices]
    ma20 = calc_sma(closes, 20)
    ma60 = calc_sma(closes, 60)
    rsi = calc_rsi(closes, 14)
    signals = [None] * len(prices)
    pos = False
    buy_price = 0

    for i in range(60, len(prices)):
        close = closes[i]
        if ma20[i] is None or ma60[i] is None or rsi[i] is None:
            continue
        bullish_phase = ma20[i] > ma60[i]
        if not pos:
            if bullish_phase and rsi[i] < 40:
                signals[i] = "buy"
                pos = True
                buy_price = close
        elif pos:
            pnl = close / buy_price - 1
            if rsi[i] > 65 or pnl >= 0.06 or pnl <= -0.03 or not bullish_phase:
                signals[i] = "sell"
                pos = False
    return signals


def strategy_golden_cross_hold(prices, fast=20, slow=60, trail=0.06):
    """黃金交叉長持：MA20上穿MA60買入，移動停利6%出場"""
    closes = [p["close"] for p in prices]
    ma_f = calc_sma(closes, fast)
    ma_s = calc_sma(closes, slow)
    signals = [None] * len(prices)
    pos = False
    peak = 0

    for i in range(1, len(prices)):
        if ma_f[i] is None or ma_s[i] is None or ma_f[i-1] is None or ma_s[i-1] is None:
            continue
        close = closes[i]
        if not pos and ma_f[i-1] <= ma_s[i-1] and ma_f[i] > ma_s[i]:
            signals[i] = "buy"
            pos = True
            peak = close
        elif pos:
            if close > peak:
                peak = close
            if close < peak * (1 - trail):
                signals[i] = "sell"
                pos = False
    return signals


def strategy_buy_low_rsi_hold_long(prices, rsi_period=14, buy_th=25, min_hold=20, trail=0.08):
    """RSI低點買入+長持移動停利：RSI<25買入，至少持20天，之後8%移動停利"""
    closes = [p["close"] for p in prices]
    rsi = calc_rsi(closes, rsi_period)
    signals = [None] * len(prices)
    pos = False
    entry_i = 0
    peak = 0

    for i in range(len(prices)):
        if rsi[i] is None:
            continue
        close = closes[i]
        if not pos and rsi[i] < buy_th:
            signals[i] = "buy"
            pos = True
            entry_i = i
            peak = close
        elif pos:
            if close > peak:
                peak = close
            days = i - entry_i
            if days >= min_hold and close < peak * (1 - trail):
                signals[i] = "sell"
                pos = False
    return signals


def strategy_adaptive_oscillator(prices):
    """自適應震盪：根據近期波動自動調整 RSI 門檻"""
    closes = [p["close"] for p in prices]
    rsi = calc_rsi(closes, 14)
    signals = [None] * len(prices)
    pos = False
    buy_price = 0

    for i in range(40, len(prices)):
        if rsi[i] is None:
            continue
        close = closes[i]
        # 動態門檻：根據最近 20 日 RSI 的範圍
        recent_rsi = [r for r in rsi[i-20:i] if r is not None]
        if len(recent_rsi) < 10:
            continue
        rsi_min = min(recent_rsi)
        rsi_max = max(recent_rsi)
        rsi_range = rsi_max - rsi_min
        buy_th = rsi_min + rsi_range * 0.2
        sell_th = rsi_max - rsi_range * 0.2

        if not pos and rsi[i] < buy_th:
            signals[i] = "buy"
            pos = True
            buy_price = close
        elif pos:
            pnl = close / buy_price - 1
            if rsi[i] > sell_th or pnl <= -0.04:
                signals[i] = "sell"
                pos = False
    return signals


def strategy_price_channel_tight(prices, period=10, target=0.04, stop=0.02):
    """緊密通道突破：突破10日高點買入，目標4%停損2%"""
    closes = [p["close"] for p in prices]
    signals = [None] * len(prices)
    pos = False
    buy_price = 0

    for i in range(period, len(prices)):
        close = closes[i]
        ch_high = max(prices[j]["high"] for j in range(i-period, i))
        if not pos and close > ch_high:
            signals[i] = "buy"
            pos = True
            buy_price = close
        elif pos:
            pnl = close / buy_price - 1
            if pnl >= target or pnl <= -stop:
                signals[i] = "sell"
                pos = False
    return signals


V6_STRATEGIES = {
    "微波段(跌2%/賺2.5%/虧1.5%)":      lambda p: strategy_micro_swing(p, 0.02, 0.025, 0.015, 2),
    "微波段(跌1.5%/賺2%/虧1%)":        lambda p: strategy_micro_swing(p, 0.015, 0.02, 0.01, 1),
    "微波段(跌2.5%/賺3%/虧2%)":        lambda p: strategy_micro_swing(p, 0.025, 0.03, 0.02, 2),
    "微波段(跌1%/賺1.5%/虧1%)":        lambda p: strategy_micro_swing(p, 0.01, 0.015, 0.01, 1),
    "嚴格超賣反彈":                      lambda p: strategy_oversold_bounce_strict(p),
    "趨勢限定(MA60↑/RSI40/65)":       lambda p: strategy_trend_only(p, 60, 10, 40, 65),
    "趨勢限定(MA60↑/RSI35/60)":       lambda p: strategy_trend_only(p, 60, 10, 35, 60),
    "趨勢限定(MA20↑/RSI40/65)":       lambda p: strategy_trend_only(p, 20, 5, 40, 65),
    "兩階段(MA20>MA60)":              lambda p: strategy_two_phase(p),
    "黃金交叉長持(20/60/6%停利)":       lambda p: strategy_golden_cross_hold(p, 20, 60, 0.06),
    "黃金交叉長持(20/60/8%停利)":       lambda p: strategy_golden_cross_hold(p, 20, 60, 0.08),
    "黃金交叉長持(10/40/6%停利)":       lambda p: strategy_golden_cross_hold(p, 10, 40, 0.06),
    "RSI低點長持(25/20天/8%停利)":      lambda p: strategy_buy_low_rsi_hold_long(p, 14, 25, 20, 0.08),
    "RSI低點長持(30/15天/6%停利)":      lambda p: strategy_buy_low_rsi_hold_long(p, 14, 30, 15, 0.06),
    "RSI低點長持(25/30天/10%停利)":     lambda p: strategy_buy_low_rsi_hold_long(p, 14, 25, 30, 0.10),
    "自適應震盪":                       lambda p: strategy_adaptive_oscillator(p),
    "緊密通道(10日/4%/2%)":            lambda p: strategy_price_channel_tight(p, 10, 0.04, 0.02),
    "緊密通道(5日/3%/1.5%)":           lambda p: strategy_price_channel_tight(p, 5, 0.03, 0.015),
    "緊密通道(10日/3%/1.5%)":          lambda p: strategy_price_channel_tight(p, 10, 0.03, 0.015),
}


if __name__ == "__main__":
    for sid, name in [("8213", "志超"), ("3615", "安可")]:
        prices = read_prices(sid)
        print(f"\n{'='*78}")
        print(f"  {sid} {name}  第六輪（微波段 + 趨勢過濾 + 長持）")
        print(f"{'='*78}")
        print(f"{'策略':<32} {'報酬率':>8} {'交易次數':>6} {'勝率':>6} {'平均持有':>7} {'最大回撤':>8} {'Sharpe':>7}")
        print("-" * 80)

        results = []
        for sname, fn in V6_STRATEGIES.items():
            signals = fn(prices)
            r = simulate(prices, signals)
            r["name"] = sname
            results.append(r)
            mark = " <<< 達標" if r["total_return_pct"] >= 50 else ""
            print(f"{sname:<32} {r['total_return_pct']:>7.2f}% {r['trades']:>6} {r['win_rate']:>5.1f}% {r['avg_holding_days']:>6.1f}天 {r['max_drawdown_pct']:>7.2f}% {r['sharpe_ratio']:>7.2f}{mark}")

        results.sort(key=lambda x: x["total_return_pct"], reverse=True)
        print(f"\n  TOP 5:")
        for i, r in enumerate(results[:5], 1):
            print(f"    {i}. {r['name']}: {r['total_return_pct']:.2f}% (交易{r['trades']}次, 勝率{r['win_rate']}%, Sharpe {r['sharpe_ratio']})")
        winners = [r for r in results if r["total_return_pct"] >= 50]
        if winners:
            print(f"\n  達標策略({len(winners)}個) ✓")
        else:
            print(f"\n  尚未達標，最佳: {results[0]['name']} = {results[0]['total_return_pct']:.2f}%")
