"""
剝頭皮式策略回測 v2 — 修復策略/停損同步 bug
策略函數改為無狀態（不追蹤持倉），由 simulate() 統一管理持倉與停損。
"""

import csv
import json
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "stock_prices"
STOCKS_PATH = BASE_DIR / "individual_stocks.json"

INITIAL_CAPITAL = 1_000_000
BUY_FEE = 0.001425
SELL_FEE = 0.001425
SELL_TAX = 0.003


def read_prices(stock_id: str) -> list[dict]:
    path = DATA_DIR / f"{stock_id}.csv"
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ["open", "high", "low", "close", "spread"]:
            r[k] = float(r[k])
        r["volume"] = int(r["volume"])
    return rows


# ─── 技術指標 ───

def calc_rsi(closes, period=14):
    rsi = [None] * len(closes)
    if len(closes) <= period:
        return rsi
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i - 1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))
    avg_g = sum(gains[:period]) / period
    avg_l = sum(losses[:period]) / period
    for i in range(period, len(closes)):
        if i > period:
            avg_g = (avg_g * (period - 1) + gains[i - 1]) / period
            avg_l = (avg_l * (period - 1) + losses[i - 1]) / period
        rsi[i] = 100 if avg_l == 0 else 100 - 100 / (1 + avg_g / avg_l)
    return rsi


def calc_sma(closes, period):
    sma = [None] * len(closes)
    for i in range(period - 1, len(closes)):
        sma[i] = sum(closes[i - period + 1:i + 1]) / period
    return sma


def calc_bollinger(closes, period=20, num_std=2):
    upper, lower, mid = [None]*len(closes), [None]*len(closes), [None]*len(closes)
    for i in range(period - 1, len(closes)):
        window = closes[i - period + 1:i + 1]
        m = sum(window) / period
        std = (sum((x - m)**2 for x in window) / period) ** 0.5
        mid[i] = m
        upper[i] = m + num_std * std
        lower[i] = m - num_std * std
    return upper, lower, mid


def calc_atr(prices, period=14):
    n = len(prices)
    atr = [None] * n
    if n < 2:
        return atr
    trs = []
    for i in range(1, n):
        h = prices[i]["high"]
        l = prices[i]["low"]
        pc = prices[i - 1]["close"]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)
    if len(trs) < period:
        return atr
    atr_val = sum(trs[:period]) / period
    atr[period] = atr_val
    for i in range(period + 1, n):
        atr_val = (atr_val * (period - 1) + trs[i - 1]) / period
        atr[i] = atr_val
    return atr


def calc_williams_r(prices, period=14):
    n = len(prices)
    wr = [None] * n
    closes = [p["close"] for p in prices]
    highs = [p["high"] for p in prices]
    lows = [p["low"] for p in prices]
    for i in range(period - 1, n):
        hh = max(highs[i - period + 1:i + 1])
        ll = min(lows[i - period + 1:i + 1])
        wr[i] = (hh - closes[i]) / (hh - ll) * -100 if hh != ll else -50
    return wr


def calc_macd(closes, fast=12, slow=26, signal=9):
    n = len(closes)
    ema_fast = [None] * n
    ema_slow = [None] * n
    histogram = [None] * n
    ema_fast[fast - 1] = sum(closes[:fast]) / fast
    k_f = 2 / (fast + 1)
    for i in range(fast, n):
        ema_fast[i] = closes[i] * k_f + ema_fast[i - 1] * (1 - k_f)
    ema_slow[slow - 1] = sum(closes[:slow]) / slow
    k_s = 2 / (slow + 1)
    for i in range(slow, n):
        ema_slow[i] = closes[i] * k_s + ema_slow[i - 1] * (1 - k_s)
    macd_line = [None] * n
    for i in range(slow - 1, n):
        if ema_fast[i] is not None and ema_slow[i] is not None:
            macd_line[i] = ema_fast[i] - ema_slow[i]
    macd_vals = [(i, macd_line[i]) for i in range(n) if macd_line[i] is not None]
    signal_line = [None] * n
    if len(macd_vals) >= signal:
        first_idx = macd_vals[signal - 1][0]
        signal_line[first_idx] = sum(v for _, v in macd_vals[:signal]) / signal
        k_sig = 2 / (signal + 1)
        for j in range(signal, len(macd_vals)):
            idx = macd_vals[j][0]
            prev_idx = macd_vals[j - 1][0]
            signal_line[idx] = macd_vals[j][1] * k_sig + signal_line[prev_idx] * (1 - k_sig)
    for i in range(n):
        if macd_line[i] is not None and signal_line[i] is not None:
            histogram[i] = macd_line[i] - signal_line[i]
    return macd_line, signal_line, histogram


# ─── 回測引擎 v2（統一持倉管理）───

def simulate(prices, signals, stop_config, exit_rule=None):
    """
    signals: list — None / "buy" / ("buy", fraction) / "sell"
      策略函數不追蹤持倉，自由輸出 buy/sell 條件。
      simulate() 負責：只在無持倉時執行 buy，只在有持倉時執行 sell。

    exit_rule:
      None 或 {"type": "signal"} — 用策略的 sell 訊號出場（預設）
      {"type": "hold_days", "days": N} — 持有 N 個交易日後出場
      {"type": "profit_or_hold", "profit_pct": P, "days": N} — 獲利 P% 或持 N 天
      {"type": "gap_fill", "days": N} — 缺口回補或持 N 天
    """
    if exit_rule is None:
        exit_rule = {"type": "signal"}

    capital = INITIAL_CAPITAL
    shares = 0
    lots = []
    position_open = False
    trades = []
    equity_curve = []

    atr = None
    if stop_config["type"] == "atr":
        atr = calc_atr(prices, stop_config.get("period", 14))

    stop_price = 0
    peak_price = 0
    entry_idx = 0

    def calc_avg_buy():
        if shares == 0:
            return 0
        return sum(l["shares"] * l["buy_price"] for l in lots) / shares

    def update_stop(bar_idx, avg_price):
        nonlocal stop_price, peak_price
        if stop_config["type"] == "fixed_pct":
            stop_price = avg_price * (1 - stop_config["pct"])
        elif stop_config["type"] == "atr":
            entry_atr = atr[bar_idx] if (atr and atr[bar_idx] is not None) else avg_price * 0.05
            stop_price = avg_price - stop_config["multiplier"] * entry_atr
            stop_price = max(stop_price, 0)
        elif stop_config["type"] == "trailing_pct":
            if peak_price == 0:
                peak_price = avg_price
            stop_price = peak_price * (1 - stop_config["pct"])

    def close_position(close_price, date, reason):
        nonlocal capital, shares, lots, position_open, stop_price, peak_price
        avg_buy = calc_avg_buy()
        rev = shares * close_price
        fee = int(rev * SELL_FEE)
        tax = int(rev * SELL_TAX)
        capital += (rev - fee - tax)
        trades.append({
            "buy_date": lots[0]["buy_date"],
            "buy_price": round(avg_buy, 2),
            "sell_date": date,
            "sell_price": close_price,
            "exit_reason": reason,
        })
        shares = 0
        lots = []
        position_open = False
        stop_price = 0
        peak_price = 0

    for i, p in enumerate(prices):
        close = p["close"]
        sig = signals[i]
        if isinstance(sig, tuple):
            action, fraction = sig[0], sig[1]
        elif sig is not None:
            action = sig
            fraction = 1.0
        else:
            action = None
            fraction = 0

        # 移動停損更新
        if position_open and stop_config["type"] == "trailing_pct":
            if close > peak_price:
                peak_price = close
                stop_price = peak_price * (1 - stop_config["pct"])

        # 1. 停損檢查
        if position_open and close <= stop_price:
            close_position(close, p["date"], "stop_loss")
            equity_curve.append(capital)
            continue

        # 2. 出場規則檢查（持倉中）
        if position_open:
            should_exit = False
            etype = exit_rule["type"]

            if etype == "signal":
                if action == "sell":
                    should_exit = True
            elif etype == "hold_days":
                if (i - entry_idx) >= exit_rule["days"]:
                    should_exit = True
            elif etype == "profit_or_hold":
                avg_buy = calc_avg_buy()
                if close >= avg_buy * (1 + exit_rule["profit_pct"]):
                    should_exit = True
                elif (i - entry_idx) >= exit_rule["days"]:
                    should_exit = True
            elif etype == "gap_fill":
                if entry_idx > 0:
                    gap_top = prices[entry_idx - 1]["close"]
                    if close >= gap_top:
                        should_exit = True
                if (i - entry_idx) >= exit_rule["days"]:
                    should_exit = True

            if should_exit:
                close_position(close, p["date"], "signal")

        # 3. 進場（只在無持倉時）
        if not position_open and action == "buy" and close > 0 and capital > 0:
            spend = capital * fraction
            cost_per = close * (1 + BUY_FEE)
            new_shares = int(spend / cost_per / 1000) * 1000
            if new_shares <= 0:
                new_shares = int(spend / cost_per)
            if new_shares > 0:
                buy_cost = new_shares * close
                fee = int(buy_cost * BUY_FEE)
                capital -= (buy_cost + fee)
                shares += new_shares
                lots.append({"shares": new_shares, "buy_price": close, "buy_date": p["date"]})
                position_open = True
                entry_idx = i
                update_stop(i, calc_avg_buy())

        # 4. 加碼（RSI 階梯：已持倉時可加碼）
        elif position_open and isinstance(sig, tuple) and sig[0] == "buy" and close > 0 and capital > 0:
            fraction = sig[1]
            spend = capital * fraction
            cost_per = close * (1 + BUY_FEE)
            new_shares = int(spend / cost_per / 1000) * 1000
            if new_shares <= 0:
                new_shares = int(spend / cost_per)
            if new_shares > 0:
                buy_cost = new_shares * close
                fee = int(buy_cost * BUY_FEE)
                capital -= (buy_cost + fee)
                shares += new_shares
                lots.append({"shares": new_shares, "buy_price": close, "buy_date": p["date"]})
                update_stop(i, calc_avg_buy())

        if position_open:
            equity_curve.append(capital + shares * close)
        else:
            equity_curve.append(capital)

    # 強制平倉
    if position_open and shares > 0:
        close_position(prices[-1]["close"], prices[-1]["date"], "forced_close")
        equity_curve[-1] = capital

    final_capital = equity_curve[-1] if equity_curve else INITIAL_CAPITAL
    total_return = (final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    peak = 0
    max_dd = 0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    daily_returns = []
    for i in range(1, len(equity_curve)):
        if equity_curve[i - 1] > 0:
            daily_returns.append(equity_curve[i] / equity_curve[i - 1] - 1)
    if len(daily_returns) > 1:
        avg_r = sum(daily_returns) / len(daily_returns)
        std_r = (sum((r - avg_r) ** 2 for r in daily_returns) / (len(daily_returns) - 1)) ** 0.5
        sharpe = (avg_r / std_r) * (252 ** 0.5) if std_r > 0 else 0
    else:
        sharpe = 0

    hold_days = []
    for t in trades:
        d1 = datetime.strptime(t["buy_date"], "%Y-%m-%d")
        d2 = datetime.strptime(t["sell_date"], "%Y-%m-%d")
        hold_days.append((d2 - d1).days)
    avg_hold = sum(hold_days) / len(hold_days) if hold_days else 0

    wins = [t for t in trades if t["sell_price"] > t["buy_price"]]
    win_rate = len(wins) / len(trades) * 100 if trades else 0
    stop_losses = len([t for t in trades if t["exit_reason"] == "stop_loss"])

    return {
        "total_return_pct": round(total_return, 2),
        "final_capital": round(final_capital),
        "trades": len(trades),
        "win_rate": round(win_rate, 1),
        "avg_holding_days": round(avg_hold, 1),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "sharpe_ratio": round(sharpe, 2),
        "stop_loss_count": stop_losses,
    }


# ════════════════════════════════════════════════
# 策略函數（無狀態：不追蹤 pos，自由輸出條件）
# ════════════════════════════════════════════════

def strategy_rsi_ladder(prices, levels=None, sell_rsi=55):
    """RSI 階梯分批建倉（可加碼）"""
    if levels is None:
        levels = [40, 30, 20]
    closes = [p["close"] for p in prices]
    rsi = calc_rsi(closes, 14)
    signals = [None] * len(prices)
    n_levels = len(levels)
    triggered = [False] * n_levels

    for i in range(len(prices)):
        if rsi[i] is None:
            continue
        # 進場：RSI 跌破各關卡
        for lvl_idx in range(n_levels):
            if not triggered[lvl_idx] and rsi[i] < levels[lvl_idx]:
                remaining = sum(1 for j in range(lvl_idx, n_levels) if not triggered[j])
                fraction = 1.0 / remaining
                signals[i] = ("buy", fraction)
                triggered[lvl_idx] = True
                break
        # 出場 + 重置：RSI 回升
        if rsi[i] > sell_rsi:
            if any(triggered):
                signals[i] = "sell"
            triggered = [False] * n_levels
    return signals


def strategy_grid(prices, grid_pct=0.05):
    """網格進場（無狀態）：跌 grid_pct 買入，出場由 exit_rule 控制"""
    closes = [p["close"] for p in prices]
    signals = [None] * len(prices)
    ref = closes[0]
    for i in range(1, len(prices)):
        c = closes[i]
        if c <= ref * (1 - grid_pct):
            signals[i] = "buy"
            ref = c
        elif c > ref:
            ref = c
    return signals


def strategy_consecutive_red(prices, n_red=3):
    """連 N 黑進場"""
    signals = [None] * len(prices)
    for i in range(n_red, len(prices)):
        all_red = all(prices[i - j]["close"] < prices[i - j]["open"] for j in range(n_red))
        if all_red:
            signals[i] = "buy"
    return signals


def strategy_rsi_bollinger(prices, rsi_buy=35, rsi_sell=65, bb_period=20, bb_std=2):
    """RSI + Bollinger 雙重確認"""
    closes = [p["close"] for p in prices]
    rsi = calc_rsi(closes, 14)
    upper, lower, mid = calc_bollinger(closes, bb_period, bb_std)
    signals = [None] * len(prices)
    for i in range(len(prices)):
        if rsi[i] is None or lower[i] is None:
            continue
        if rsi[i] < rsi_buy and closes[i] < lower[i]:
            signals[i] = "buy"
        elif rsi[i] > rsi_sell or closes[i] > upper[i]:
            signals[i] = "sell"
    return signals


def strategy_kd(prices, period=9, buy_th=30, sell_th=70):
    """KD 交叉"""
    closes = [p["close"] for p in prices]
    highs = [p["high"] for p in prices]
    lows = [p["low"] for p in prices]
    n = len(prices)
    k_vals = [50.0] * n
    d_vals = [50.0] * n
    for i in range(period - 1, n):
        hh = max(highs[i - period + 1:i + 1])
        ll = min(lows[i - period + 1:i + 1])
        rsv = (closes[i] - ll) / (hh - ll) * 100 if hh != ll else 50
        k_vals[i] = k_vals[i - 1] * 2 / 3 + rsv * 1 / 3
        d_vals[i] = d_vals[i - 1] * 2 / 3 + k_vals[i] * 1 / 3
    signals = [None] * n
    for i in range(period, n):
        if k_vals[i] < buy_th and k_vals[i] > d_vals[i] and k_vals[i - 1] <= d_vals[i - 1]:
            signals[i] = "buy"
        elif k_vals[i] > sell_th and k_vals[i] < d_vals[i] and k_vals[i - 1] >= d_vals[i - 1]:
            signals[i] = "sell"
    return signals


def strategy_rsi7_flash(prices):
    closes = [p["close"] for p in prices]
    rsi = calc_rsi(closes, 7)
    signals = [None] * len(prices)
    for i in range(len(prices)):
        if rsi[i] is None:
            continue
        if rsi[i] < 25:
            signals[i] = "buy"
        elif rsi[i] > 50:
            signals[i] = "sell"
    return signals


def strategy_bollinger_touch(prices):
    closes = [p["close"] for p in prices]
    upper, lower, mid = calc_bollinger(closes, 20, 2)
    signals = [None] * len(prices)
    for i in range(len(prices)):
        if lower[i] is None:
            continue
        if closes[i] < lower[i]:
            signals[i] = "buy"
        elif closes[i] >= mid[i]:
            signals[i] = "sell"
    return signals


def strategy_deviation(prices, period=10, threshold=-0.04):
    closes = [p["close"] for p in prices]
    sma = calc_sma(closes, period)
    signals = [None] * len(prices)
    for i in range(len(prices)):
        if sma[i] is None or sma[i] == 0:
            continue
        bias = (closes[i] - sma[i]) / sma[i]
        if bias < threshold:
            signals[i] = "buy"
        elif bias >= 0:
            signals[i] = "sell"
    return signals


def strategy_williams_r(prices):
    wr = calc_williams_r(prices, 14)
    signals = [None] * len(prices)
    for i in range(len(prices)):
        if wr[i] is None:
            continue
        if wr[i] < -80:
            signals[i] = "buy"
        elif wr[i] > -20:
            signals[i] = "sell"
    return signals


def strategy_cumulative_drop(prices):
    """連5跌進場"""
    closes = [p["close"] for p in prices]
    signals = [None] * len(prices)
    for i in range(5, len(prices)):
        cum_ret = (closes[i] - closes[i - 5]) / closes[i - 5]
        if cum_ret < -0.08:
            signals[i] = "buy"
    return signals


def strategy_volume_spike(prices):
    """量縮後爆量反彈進場"""
    closes = [p["close"] for p in prices]
    vols = [p["volume"] for p in prices]
    vol_ma = calc_sma([float(v) for v in vols], 20)
    signals = [None] * len(prices)
    for i in range(23, len(prices)):
        if vol_ma[i] is None or vol_ma[i] == 0:
            continue
        shrunk = all(
            vol_ma[i - j - 1] is not None and vols[i - j - 1] < vol_ma[i - j - 1] * 0.5
            for j in range(3)
        )
        spike = vols[i] > vol_ma[i] * 2 and closes[i] > closes[i - 1]
        if shrunk and spike:
            signals[i] = "buy"
    return signals


def strategy_dual_ma(prices):
    closes = [p["close"] for p in prices]
    ma5 = calc_sma(closes, 5)
    ma20 = calc_sma(closes, 20)
    signals = [None] * len(prices)
    for i in range(21, len(prices)):
        if ma5[i] is None or ma20[i] is None or ma5[i - 1] is None or ma20[i - 1] is None:
            continue
        if ma5[i] > ma20[i] and ma5[i - 1] <= ma20[i - 1]:
            signals[i] = "buy"
        elif ma5[i] < ma20[i] and ma5[i - 1] >= ma20[i - 1]:
            signals[i] = "sell"
    return signals


def strategy_atr_channel(prices):
    closes = [p["close"] for p in prices]
    sma = calc_sma(closes, 20)
    atr_vals = calc_atr(prices, 14)
    signals = [None] * len(prices)
    for i in range(len(prices)):
        if sma[i] is None or atr_vals[i] is None:
            continue
        lower = sma[i] - 1.5 * atr_vals[i]
        if closes[i] < lower:
            signals[i] = "buy"
        elif closes[i] >= sma[i]:
            signals[i] = "sell"
    return signals


def strategy_macd_fast(prices):
    closes = [p["close"] for p in prices]
    _, _, hist = calc_macd(closes, 8, 17, 9)
    signals = [None] * len(prices)
    for i in range(1, len(prices)):
        if hist[i] is None or hist[i - 1] is None:
            continue
        if hist[i - 1] < 0 and hist[i] >= 0:
            signals[i] = "buy"
        elif hist[i - 1] > 0 and hist[i] <= 0:
            signals[i] = "sell"
    return signals


def strategy_new_low_bounce(prices):
    """10日新低進場"""
    closes = [p["close"] for p in prices]
    signals = [None] * len(prices)
    for i in range(10, len(prices)):
        low_10 = min(closes[i - 10:i])
        if closes[i] < low_10:
            signals[i] = "buy"
    return signals


def strategy_rsi7_sma10(prices):
    closes = [p["close"] for p in prices]
    rsi = calc_rsi(closes, 7)
    sma = calc_sma(closes, 10)
    signals = [None] * len(prices)
    for i in range(len(prices)):
        if rsi[i] is None or sma[i] is None:
            continue
        if rsi[i] < 30 and closes[i] < sma[i]:
            signals[i] = "buy"
        elif rsi[i] > 50:
            signals[i] = "sell"
    return signals


def strategy_gap_fill(prices):
    """缺口下跌進場"""
    closes = [p["close"] for p in prices]
    signals = [None] * len(prices)
    for i in range(1, len(prices)):
        gap = (closes[i] - closes[i - 1]) / closes[i - 1]
        if gap < -0.03:
            signals[i] = "buy"
    return signals


def strategy_fast_grid(prices, grid_pct=0.03):
    """快速網格進場"""
    closes = [p["close"] for p in prices]
    signals = [None] * len(prices)
    ref = closes[0]
    for i in range(1, len(prices)):
        c = closes[i]
        if c <= ref * (1 - grid_pct):
            signals[i] = "buy"
            ref = c
        elif c > ref:
            ref = c
    return signals


def strategy_kd5_fast(prices):
    return strategy_kd(prices, period=5, buy_th=25, sell_th=65)


def strategy_red2_hold5(prices):
    """連2黑進場"""
    signals = [None] * len(prices)
    for i in range(2, len(prices)):
        r1 = prices[i - 1]["close"] < prices[i - 1]["open"]
        r2 = prices[i]["close"] < prices[i]["open"]
        if r1 and r2:
            signals[i] = "buy"
    return signals


def strategy_rsi_divergence(prices):
    closes = [p["close"] for p in prices]
    rsi = calc_rsi(closes, 14)
    signals = [None] * len(prices)
    lookback = 20
    for i in range(lookback, len(prices)):
        if rsi[i] is None:
            continue
        price_new_low = closes[i] <= min(closes[i - lookback:i])
        rsi_window = [r for r in rsi[i - lookback:i] if r is not None]
        if rsi_window and price_new_low and rsi[i] > min(rsi_window):
            signals[i] = "buy"
        elif rsi[i] > 55:
            signals[i] = "sell"
    return signals


def strategy_inside_bar(prices):
    """內包線突破進場"""
    signals = [None] * len(prices)
    for i in range(2, len(prices)):
        prev_inside = (prices[i - 1]["high"] <= prices[i - 2]["high"] and
                       prices[i - 1]["low"] >= prices[i - 2]["low"])
        if prev_inside and prices[i]["close"] < prices[i - 1]["low"]:
            signals[i] = "buy"
    return signals


def strategy_double_bottom(prices):
    """雙底進場"""
    closes = [p["close"] for p in prices]
    signals = [None] * len(prices)
    for i in range(30, len(prices)):
        window = closes[i - 30:i]
        min_price = min(window)
        min_idx = window.index(min_price)
        near_low = abs(closes[i] - min_price) / min_price < 0.02
        has_gap = min_idx < 25
        if near_low and has_gap:
            signals[i] = "buy"
    return signals


# ─── 策略配置 ───

STRATEGIES = {
    # ── 原始策略 ──
    "RSI階梯[40,30,20]→55": {
        "fn": lambda p: strategy_rsi_ladder(p),
        "stop": {"type": "atr", "multiplier": 2, "period": 14},
        "exit": {"type": "signal"},
        "cat": "逆勢", "desc": "RSI(14)分批建倉：跌破40買1/3，跌破30再買1/3，跌破20買最後1/3。RSI>55全部賣出",
        "stop_desc": "ATR停損：均價-2×ATR(14)",
    },
    "網格(5%/6%)": {
        "fn": lambda p: strategy_grid(p, 0.05),
        "stop": {"type": "fixed_pct", "pct": 0.10},
        "exit": {"type": "profit_or_hold", "profit_pct": 0.06, "days": 99999},
        "cat": "逆勢", "desc": "每跌5%買入，漲6%賣出",
        "stop_desc": "固定停損-10%（2倍網格）",
    },
    "連3黑/持10天": {
        "fn": lambda p: strategy_consecutive_red(p, 3),
        "stop": {"type": "fixed_pct", "pct": 0.05},
        "exit": {"type": "hold_days", "days": 10},
        "cat": "逆勢", "desc": "連續3天收黑買入，持10天賣出",
        "stop_desc": "固定停損-5%",
    },
    "RSI+Bollinger": {
        "fn": lambda p: strategy_rsi_bollinger(p),
        "stop": {"type": "atr", "multiplier": 2.5, "period": 14},
        "exit": {"type": "signal"},
        "cat": "逆勢", "desc": "RSI(14)<35且低於布林下軌買入，RSI>65或碰上軌賣出",
        "stop_desc": "ATR停損：買入價-2.5×ATR(14)",
    },
    "KD(9) 30/70": {
        "fn": lambda p: strategy_kd(p),
        "stop": {"type": "trailing_pct", "pct": 0.08},
        "exit": {"type": "signal"},
        "cat": "逆勢", "desc": "K(9)<30且K上穿D買入，K>70且K下穿D賣出",
        "stop_desc": "移動停損-8%",
    },
    # ── 剝頭皮策略 ──
    "RSI(7)快閃": {
        "fn": lambda p: strategy_rsi7_flash(p),
        "stop": {"type": "atr", "multiplier": 1.5, "period": 7},
        "exit": {"type": "signal"},
        "cat": "逆勢", "desc": "RSI(7)<25買入，>50賣出",
        "stop_desc": "ATR停損：買入價-1.5×ATR(7)",
    },
    "布林觸底→中線": {
        "fn": lambda p: strategy_bollinger_touch(p),
        "stop": {"type": "fixed_pct", "pct": 0.05},
        "exit": {"type": "signal"},
        "cat": "逆勢", "desc": "碰布林下軌買入，到中線賣出",
        "stop_desc": "固定停損-5%",
    },
    "乖離率(10)-4%": {
        "fn": lambda p: strategy_deviation(p, 10, -0.04),
        "stop": {"type": "fixed_pct", "pct": 0.06},
        "exit": {"type": "signal"},
        "cat": "逆勢", "desc": "偏離10日均線-4%買入，回到均線賣出",
        "stop_desc": "固定停損-6%",
    },
    "威廉%R(14)": {
        "fn": lambda p: strategy_williams_r(p),
        "stop": {"type": "atr", "multiplier": 2, "period": 14},
        "exit": {"type": "signal"},
        "cat": "逆勢", "desc": "Williams %R<-80買入，>-20賣出",
        "stop_desc": "ATR停損：買入價-2×ATR(14)",
    },
    "連5跌回彈": {
        "fn": lambda p: strategy_cumulative_drop(p),
        "stop": {"type": "fixed_pct", "pct": 0.04},
        "exit": {"type": "profit_or_hold", "profit_pct": 0.03, "days": 5},
        "cat": "逆勢", "desc": "5日累計跌>8%買入，反彈3%或持5天賣出",
        "stop_desc": "固定停損-4%",
    },
    "量縮爆量": {
        "fn": lambda p: strategy_volume_spike(p),
        "stop": {"type": "trailing_pct", "pct": 0.06},
        "exit": {"type": "hold_days", "days": 5},
        "cat": "量價", "desc": "3天量縮至均量50%以下，爆量2倍+價漲買入，持5天賣出",
        "stop_desc": "移動停損-6%",
    },
    "雙均線(5/20)": {
        "fn": lambda p: strategy_dual_ma(p),
        "stop": {"type": "trailing_pct", "pct": 0.05},
        "exit": {"type": "signal"},
        "cat": "順勢", "desc": "5日線上穿20日線買入，下穿賣出",
        "stop_desc": "移動停損-5%",
    },
    "ATR通道回歸": {
        "fn": lambda p: strategy_atr_channel(p),
        "stop": {"type": "fixed_pct", "pct": 0.08},
        "exit": {"type": "signal"},
        "cat": "逆勢", "desc": "價格低於20日均線-1.5×ATR買入，回到均線賣出",
        "stop_desc": "固定停損-8%",
    },
    "MACD快速(8,17,9)": {
        "fn": lambda p: strategy_macd_fast(p),
        "stop": {"type": "trailing_pct", "pct": 0.07},
        "exit": {"type": "signal"},
        "cat": "順勢", "desc": "MACD(8,17,9)柱狀圖負轉正買入，正轉負賣出",
        "stop_desc": "移動停損-7%",
    },
    "10日新低/持3天": {
        "fn": lambda p: strategy_new_low_bounce(p),
        "stop": {"type": "fixed_pct", "pct": 0.04},
        "exit": {"type": "hold_days", "days": 3},
        "cat": "逆勢", "desc": "創10日新低買入，持3天賣出",
        "stop_desc": "固定停損-4%",
    },
    "RSI(7)+SMA(10)": {
        "fn": lambda p: strategy_rsi7_sma10(p),
        "stop": {"type": "atr", "multiplier": 1.5, "period": 7},
        "exit": {"type": "signal"},
        "cat": "逆勢", "desc": "RSI(7)<30且低於10日均線買入，RSI>50賣出",
        "stop_desc": "ATR停損：買入價-1.5×ATR(7)",
    },
    "缺口回補": {
        "fn": lambda p: strategy_gap_fill(p),
        "stop": {"type": "fixed_pct", "pct": 0.05},
        "exit": {"type": "gap_fill", "days": 5},
        "cat": "逆勢", "desc": "跳空下跌>3%買入，缺口補回或持5天賣出",
        "stop_desc": "固定停損-5%",
    },
    "快速網格(3%/4%)": {
        "fn": lambda p: strategy_fast_grid(p, 0.03),
        "stop": {"type": "fixed_pct", "pct": 0.07},
        "exit": {"type": "profit_or_hold", "profit_pct": 0.04, "days": 99999},
        "cat": "逆勢", "desc": "每跌3%買入，漲4%賣出",
        "stop_desc": "固定停損-7%",
    },
    "KD(5)快速": {
        "fn": lambda p: strategy_kd5_fast(p),
        "stop": {"type": "trailing_pct", "pct": 0.06},
        "exit": {"type": "signal"},
        "cat": "逆勢", "desc": "KD(5) K<25且K上穿D買入，K>65且K下穿D賣出",
        "stop_desc": "移動停損-6%",
    },
    "連2黑/持5天": {
        "fn": lambda p: strategy_red2_hold5(p),
        "stop": {"type": "fixed_pct", "pct": 0.03},
        "exit": {"type": "hold_days", "days": 5},
        "cat": "逆勢", "desc": "連2天收黑買入，持5天賣出",
        "stop_desc": "固定停損-3%",
    },
    "乖離率(5)-3%": {
        "fn": lambda p: strategy_deviation(p, 5, -0.03),
        "stop": {"type": "fixed_pct", "pct": 0.04},
        "exit": {"type": "signal"},
        "cat": "逆勢", "desc": "偏離5日均線-3%買入，回到均線賣出",
        "stop_desc": "固定停損-4%",
    },
    "RSI背離反彈": {
        "fn": lambda p: strategy_rsi_divergence(p),
        "stop": {"type": "atr", "multiplier": 2, "period": 14},
        "exit": {"type": "signal"},
        "cat": "逆勢", "desc": "價格創新低但RSI(14)未創新低（背離）買入，RSI>55賣出",
        "stop_desc": "ATR停損：買入價-2×ATR(14)",
    },
    "內包線突破": {
        "fn": lambda p: strategy_inside_bar(p),
        "stop": {"type": "fixed_pct", "pct": 0.04},
        "exit": {"type": "hold_days", "days": 3},
        "cat": "型態", "desc": "內包線後跌破低點買入，持3天賣出",
        "stop_desc": "固定停損-4%",
    },
    "雙底反彈": {
        "fn": lambda p: strategy_double_bottom(p),
        "stop": {"type": "fixed_pct", "pct": 0.05},
        "exit": {"type": "profit_or_hold", "profit_pct": 0.05, "days": 10},
        "cat": "型態", "desc": "30日內兩次觸底買入，漲5%或持10天賣出",
        "stop_desc": "固定停損-5%",
    },
}


def format_stop_config(stop_config):
    if stop_config["type"] == "fixed_pct":
        return f"固定-{stop_config['pct']*100:.0f}%"
    elif stop_config["type"] == "atr":
        return f"ATR({stop_config.get('period',14)})×{stop_config['multiplier']}"
    elif stop_config["type"] == "trailing_pct":
        return f"移動-{stop_config['pct']*100:.0f}%"
    return str(stop_config)


def main():
    candidates = json.load(open("/tmp/sideways_volatile.json"))
    stock_info = {s["stock_id"]: s for s in json.load(open(STOCKS_PATH))}

    print(f"策略數: {len(STRATEGIES)}")
    print(f"候選股: {len(candidates)}")
    print(f"v2: 修復策略/停損同步 bug\n")

    all_results = []
    for idx, c in enumerate(candidates):
        sid = c["stock_id"]
        path = DATA_DIR / f"{sid}.csv"
        if not path.exists():
            continue
        prices = read_prices(sid)
        if len(prices) < 200:
            continue

        for sname, cfg in STRATEGIES.items():
            try:
                signals = cfg["fn"](prices)
                r = simulate(prices, signals, cfg["stop"], cfg.get("exit"))
            except Exception:
                continue
            if r["trades"] == 0:
                continue
            r["stock_id"] = sid
            r["stock_name"] = c["name"]
            r["strategy"] = sname
            r["daily_vol"] = c["daily_vol_pct"]
            r["bnh"] = c["bnh_pct"]
            r["stop_type"] = format_stop_config(cfg["stop"])
            all_results.append(r)

        if (idx + 1) % 50 == 0:
            print(f"  進度: {idx+1}/{len(candidates)}")

    # 篩選 >50%
    winners = [r for r in all_results if r["total_return_pct"] >= 50]
    winners.sort(key=lambda x: x["total_return_pct"], reverse=True)

    stock_best = {}
    for w in winners:
        sid = w["stock_id"]
        if sid not in stock_best or w["total_return_pct"] > stock_best[sid]["total_return_pct"]:
            stock_best[sid] = w
    unique_stocks = sorted(stock_best.values(), key=lambda x: x["total_return_pct"], reverse=True)

    print(f"\n{'='*120}")
    print(f"全部回測: {len(all_results)} 筆")
    print(f"報酬 > 50%: {len(winners)} 筆, 不重複股票: {len(unique_stocks)} 檔")
    print(f"{'='*120}")

    print(f"\n{'排名':>3} {'代號':>6} {'名稱':<8} {'最佳策略':<18} {'停損':<12} {'報酬率':>8} {'交易':>4} {'勝率':>6} {'持有':>6} {'回撤':>7} {'Sharpe':>7} {'停損次':>5}")
    print("-" * 110)
    for rank, r in enumerate(unique_stocks[:40], 1):
        print(f"{rank:>3} {r['stock_id']:>6} {r['stock_name']:<8} {r['strategy']:<18} {r['stop_type']:<12} {r['total_return_pct']:>7.1f}% {r['trades']:>4} {r['win_rate']:>5.1f}% {r['avg_holding_days']:>5.1f}天 {r['max_drawdown_pct']:>6.1f}% {r['sharpe_ratio']:>7.2f} {r['stop_loss_count']:>5}")

    print(f"\n--- 各策略達標數 ---")
    strat_count = {}
    for w in winners:
        s = w["strategy"]
        strat_count[s] = strat_count.get(s, 0) + 1
    for s, cnt in sorted(strat_count.items(), key=lambda x: -x[1]):
        print(f"  {s:<22} {cnt:>3} 筆")

    # 存策略檔案
    by_stock = {}
    for w in winners:
        sid = w["stock_id"]
        if sid not in by_stock:
            by_stock[sid] = []
        by_stock[sid].append(w)

    strategies_dir = BASE_DIR / "strategies"
    for f in strategies_dir.glob("*.json"):
        f.unlink()

    created = 0
    for sid, strats in by_stock.items():
        info = stock_info.get(sid, {})
        strats.sort(key=lambda x: x["total_return_pct"], reverse=True)
        viable = []
        for rank, s in enumerate(strats, 1):
            cfg = STRATEGIES.get(s["strategy"], {})
            viable.append({
                "rank": rank,
                "name": s["strategy"],
                "category": cfg.get("cat", ""),
                "description": cfg.get("desc", ""),
                "stop_loss": cfg.get("stop_desc", ""),
                "total_return_pct": s["total_return_pct"],
                "final_capital": s["final_capital"],
                "trades": s["trades"],
                "win_rate_pct": s["win_rate"],
                "avg_holding_days": s["avg_holding_days"],
                "max_drawdown_pct": s["max_drawdown_pct"],
                "sharpe_ratio": s["sharpe_ratio"],
                "stop_loss_count": s["stop_loss_count"],
            })
        data = {
            "stock_id": sid,
            "stock_name": strats[0]["stock_name"],
            "industry_category": info.get("industry_category", ""),
            "type": info.get("type", ""),
            "backtest_period": "2024-02-15 ~ 2026-02-25",
            "initial_capital": 1000000,
            "buy_and_hold_return_pct": strats[0]["bnh"],
            "daily_volatility_pct": strats[0]["daily_vol"],
            "viable_strategies": viable,
        }
        name = strats[0]["stock_name"]
        out_path = strategies_dir / f"{sid}_{name}.json"
        json.dump(data, open(out_path, "w"), ensure_ascii=False, indent=2)
        created += 1

    print(f"\n已建立 {created} 個策略檔案")

    if len(unique_stocks) >= 20:
        print(f"\n✓ 達標！找到 {len(unique_stocks)} 檔 >50% 報酬的股票（目標 20 檔）")
    else:
        print(f"\n✗ 尚未達標：{len(unique_stocks)} / 20 檔")


if __name__ == "__main__":
    main()
