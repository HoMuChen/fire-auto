"""
針對志超(8213)、安可(3615) 的多策略回測研究
目標：找到兩年報酬 > 50% 的策略
"""

import csv
import math
from pathlib import Path
from dataclasses import dataclass, field

DATA_DIR = Path(__file__).parent / "data" / "stock_prices"

INITIAL_CAPITAL = 1_000_000
BUY_FEE = 0.001425
SELL_FEE = 0.001425
SELL_TAX = 0.003


@dataclass
class Trade:
    buy_date: str
    buy_price: float
    sell_date: str = ""
    sell_price: float = 0.0
    shares: int = 0

    @property
    def pnl_pct(self):
        if self.buy_price == 0:
            return 0
        return self.sell_price / self.buy_price - 1

    @property
    def holding_days(self):
        if not self.sell_date or not self.buy_date:
            return 0
        from datetime import datetime
        d1 = datetime.strptime(self.buy_date, "%Y-%m-%d")
        d2 = datetime.strptime(self.sell_date, "%Y-%m-%d")
        return (d2 - d1).days


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

def calc_sma(closes, period):
    sma = [None] * len(closes)
    for i in range(period - 1, len(closes)):
        sma[i] = sum(closes[i - period + 1:i + 1]) / period
    return sma


def calc_ema(closes, period):
    ema = [None] * len(closes)
    k = 2 / (period + 1)
    ema[period - 1] = sum(closes[:period]) / period
    for i in range(period, len(closes)):
        ema[i] = closes[i] * k + ema[i - 1] * (1 - k)
    return ema


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
    atr = [None] * len(prices)
    trs = []
    for i in range(1, len(prices)):
        h, l, pc = prices[i]["high"], prices[i]["low"], prices[i-1]["close"]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)
    if len(trs) < period:
        return atr
    atr[period] = sum(trs[:period]) / period
    for i in range(period + 1, len(prices)):
        atr[i] = (atr[i-1] * (period - 1) + trs[i-1]) / period
    return atr


def calc_macd(closes, fast=12, slow=26, signal=9):
    ema_f = calc_ema(closes, fast)
    ema_s = calc_ema(closes, slow)
    macd_line = [None] * len(closes)
    for i in range(len(closes)):
        if ema_f[i] is not None and ema_s[i] is not None:
            macd_line[i] = ema_f[i] - ema_s[i]
    # signal line
    macd_vals = [v for v in macd_line if v is not None]
    sig = [None] * len(closes)
    if len(macd_vals) < signal:
        return macd_line, sig
    start = next(i for i, v in enumerate(macd_line) if v is not None)
    k = 2 / (signal + 1)
    sig[start + signal - 1] = sum(macd_line[start:start+signal]) / signal
    for i in range(start + signal, len(closes)):
        if macd_line[i] is not None and sig[i-1] is not None:
            sig[i] = macd_line[i] * k + sig[i-1] * (1 - k)
    return macd_line, sig


# ─── 回測引擎 ───

def simulate(prices, signals) -> dict:
    """
    signals: list of ("buy"|"sell"|None) 對應每根 bar
    回傳績效統計
    """
    capital = INITIAL_CAPITAL
    shares = 0
    position_open = False
    trades = []
    current_trade = None
    equity_curve = []

    for i, p in enumerate(prices):
        close = p["close"]
        sig = signals[i]

        if not position_open and sig == "buy" and close > 0:
            cost_per = close * (1 + BUY_FEE)
            shares = int(capital / cost_per / 1000) * 1000
            if shares <= 0:
                shares = int(capital / cost_per)
            if shares <= 0:
                equity_curve.append(capital)
                continue
            buy_cost = shares * close
            fee = int(buy_cost * BUY_FEE)
            capital -= (buy_cost + fee)
            current_trade = Trade(buy_date=p["date"], buy_price=close, shares=shares)
            position_open = True

        elif position_open and sig == "sell":
            rev = shares * close
            fee = int(rev * SELL_FEE)
            tax = int(rev * SELL_TAX)
            capital += (rev - fee - tax)
            current_trade.sell_date = p["date"]
            current_trade.sell_price = close
            trades.append(current_trade)
            shares = 0
            position_open = False
            current_trade = None

        # equity
        if position_open:
            equity_curve.append(capital + shares * close)
        else:
            equity_curve.append(capital)

    # 強制平倉
    if position_open and shares > 0:
        last = prices[-1]["close"]
        rev = shares * last
        fee = int(rev * SELL_FEE)
        tax = int(rev * SELL_TAX)
        capital += (rev - fee - tax)
        current_trade.sell_date = prices[-1]["date"]
        current_trade.sell_price = last
        trades.append(current_trade)
        equity_curve[-1] = capital

    final_capital = equity_curve[-1] if equity_curve else INITIAL_CAPITAL
    total_return = (final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    # max drawdown
    peak = 0
    max_dd = 0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    # sharpe ratio (annualized, daily returns)
    daily_returns = []
    for i in range(1, len(equity_curve)):
        if equity_curve[i-1] > 0:
            daily_returns.append(equity_curve[i] / equity_curve[i-1] - 1)
    if len(daily_returns) > 1:
        avg_r = sum(daily_returns) / len(daily_returns)
        std_r = (sum((r - avg_r)**2 for r in daily_returns) / (len(daily_returns)-1)) ** 0.5
        sharpe = (avg_r / std_r) * (252 ** 0.5) if std_r > 0 else 0
    else:
        sharpe = 0

    # avg holding days
    holding_days = [t.holding_days for t in trades if t.holding_days > 0]
    avg_hold = sum(holding_days) / len(holding_days) if holding_days else 0

    wins = [t for t in trades if t.pnl_pct > 0]
    win_rate = len(wins) / len(trades) * 100 if trades else 0

    return {
        "total_return_pct": round(total_return, 2),
        "final_capital": round(final_capital),
        "trades": len(trades),
        "win_rate": round(win_rate, 1),
        "avg_holding_days": round(avg_hold, 1),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "sharpe_ratio": round(sharpe, 2),
    }


# ─── 策略定義 ───

def strategy_rsi_reversion(prices, buy_th=30, sell_th=70, period=14):
    closes = [p["close"] for p in prices]
    rsi = calc_rsi(closes, period)
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


def strategy_bollinger_reversion(prices, period=20, num_std=2):
    closes = [p["close"] for p in prices]
    upper, lower, mid = calc_bollinger(closes, period, num_std)
    signals = [None] * len(prices)
    pos = False
    for i in range(len(prices)):
        if lower[i] is None:
            continue
        if not pos and closes[i] < lower[i]:
            signals[i] = "buy"
            pos = True
        elif pos and closes[i] > mid[i]:
            signals[i] = "sell"
            pos = False
    return signals


def strategy_ma_crossover(prices, fast=5, slow=20):
    closes = [p["close"] for p in prices]
    ma_f = calc_sma(closes, fast)
    ma_s = calc_sma(closes, slow)
    signals = [None] * len(prices)
    pos = False
    for i in range(1, len(prices)):
        if ma_f[i] is None or ma_s[i] is None or ma_f[i-1] is None or ma_s[i-1] is None:
            continue
        if not pos and ma_f[i-1] <= ma_s[i-1] and ma_f[i] > ma_s[i]:
            signals[i] = "buy"
            pos = True
        elif pos and ma_f[i-1] >= ma_s[i-1] and ma_f[i] < ma_s[i]:
            signals[i] = "sell"
            pos = False
    return signals


def strategy_macd_crossover(prices):
    closes = [p["close"] for p in prices]
    macd_line, sig_line = calc_macd(closes)
    signals = [None] * len(prices)
    pos = False
    for i in range(1, len(prices)):
        if macd_line[i] is None or sig_line[i] is None or macd_line[i-1] is None or sig_line[i-1] is None:
            continue
        if not pos and macd_line[i-1] <= sig_line[i-1] and macd_line[i] > sig_line[i]:
            signals[i] = "buy"
            pos = True
        elif pos and macd_line[i-1] >= sig_line[i-1] and macd_line[i] < sig_line[i]:
            signals[i] = "sell"
            pos = False
    return signals


def strategy_donchian_breakout(prices, period=20):
    signals = [None] * len(prices)
    pos = False
    for i in range(period, len(prices)):
        highs = [prices[j]["high"] for j in range(i - period, i)]
        lows = [prices[j]["low"] for j in range(i - period, i)]
        ch_high = max(highs)
        ch_low = min(lows)
        close = prices[i]["close"]
        if not pos and close > ch_high:
            signals[i] = "buy"
            pos = True
        elif pos and close < ch_low:
            signals[i] = "sell"
            pos = False
    return signals


def strategy_dip_buy(prices, dip_pct=0.08, recover_pct=0.10):
    """從近期高點跌 dip_pct 買入，漲 recover_pct 賣出"""
    signals = [None] * len(prices)
    pos = False
    recent_high = prices[0]["close"]
    buy_price = 0
    for i in range(1, len(prices)):
        close = prices[i]["close"]
        if not pos:
            if close > recent_high:
                recent_high = close
            if close <= recent_high * (1 - dip_pct):
                signals[i] = "buy"
                pos = True
                buy_price = close
        else:
            if close >= buy_price * (1 + recover_pct):
                signals[i] = "sell"
                pos = False
                recent_high = close
    return signals


def strategy_rsi_bollinger(prices, rsi_buy=35, rsi_sell=65, bb_period=20, bb_std=2):
    """RSI + Bollinger 結合：RSI 超賣且價格低於下軌買入，RSI 超買或碰上軌賣出"""
    closes = [p["close"] for p in prices]
    rsi = calc_rsi(closes, 14)
    upper, lower, mid = calc_bollinger(closes, bb_period, bb_std)
    signals = [None] * len(prices)
    pos = False
    for i in range(len(prices)):
        if rsi[i] is None or lower[i] is None:
            continue
        if not pos and rsi[i] < rsi_buy and closes[i] < lower[i]:
            signals[i] = "buy"
            pos = True
        elif pos and (rsi[i] > rsi_sell or closes[i] > upper[i]):
            signals[i] = "sell"
            pos = False
    return signals


def strategy_mean_reversion_atr(prices, lookback=20, atr_mult=1.5, atr_period=14):
    """價格低於 SMA - ATR*mult 買入，回到 SMA 以上賣出"""
    closes = [p["close"] for p in prices]
    sma = calc_sma(closes, lookback)
    atr = calc_atr(prices, atr_period)
    signals = [None] * len(prices)
    pos = False
    for i in range(len(prices)):
        if sma[i] is None or atr[i] is None:
            continue
        if not pos and closes[i] < sma[i] - atr[i] * atr_mult:
            signals[i] = "buy"
            pos = True
        elif pos and closes[i] > sma[i]:
            signals[i] = "sell"
            pos = False
    return signals


def strategy_gap_reversal(prices, gap_pct=0.03):
    """跳空下跌反轉：今日開盤跳空低開 > gap_pct，收盤價站回昨收以上時買入"""
    signals = [None] * len(prices)
    pos = False
    buy_price = 0
    for i in range(1, len(prices)):
        prev_close = prices[i-1]["close"]
        opn = prices[i]["open"]
        close = prices[i]["close"]
        if not pos:
            gap = (prev_close - opn) / prev_close
            if gap >= gap_pct and close > prev_close:
                signals[i] = "buy"
                pos = True
                buy_price = close
        else:
            # 持有 5 天或獲利 5% 或虧損 3% 離場
            if hasattr(strategy_gap_reversal, '_entry_idx'):
                days_held = i - strategy_gap_reversal._entry_idx
            else:
                days_held = 0
            pnl = close / buy_price - 1
            if pnl >= 0.05 or pnl <= -0.03 or days_held >= 5:
                signals[i] = "sell"
                pos = False
    return signals


def strategy_volume_breakout(prices, vol_mult=2.0, ma_period=20):
    """量增突破：成交量 > MA(20)*mult 且收紅時買入，跌破 MA20 賣出"""
    closes = [p["close"] for p in prices]
    vols = [p["volume"] for p in prices]
    vol_ma = calc_sma([float(v) for v in vols], ma_period)
    price_ma = calc_sma(closes, ma_period)
    signals = [None] * len(prices)
    pos = False
    for i in range(1, len(prices)):
        if vol_ma[i] is None or price_ma[i] is None:
            continue
        close = prices[i]["close"]
        opn = prices[i]["open"]
        if not pos and vols[i] > vol_ma[i] * vol_mult and close > opn and close > price_ma[i]:
            signals[i] = "buy"
            pos = True
        elif pos and close < price_ma[i]:
            signals[i] = "sell"
            pos = False
    return signals


def strategy_swing_rsi(prices, rsi_buy=40, hold_days=10):
    """RSI 波段：RSI < rsi_buy 買入，固定持有 N 天賣出"""
    closes = [p["close"] for p in prices]
    rsi = calc_rsi(closes, 14)
    signals = [None] * len(prices)
    pos = False
    entry_i = 0
    for i in range(len(prices)):
        if rsi[i] is None:
            continue
        if not pos and rsi[i] < rsi_buy:
            signals[i] = "buy"
            pos = True
            entry_i = i
        elif pos and (i - entry_i) >= hold_days:
            signals[i] = "sell"
            pos = False
    return signals


def strategy_ma_envelope(prices, period=20, pct=0.05):
    """均線通道：跌破 MA*(1-pct) 買入，漲到 MA*(1+pct) 賣出"""
    closes = [p["close"] for p in prices]
    ma = calc_sma(closes, period)
    signals = [None] * len(prices)
    pos = False
    for i in range(len(prices)):
        if ma[i] is None:
            continue
        if not pos and closes[i] < ma[i] * (1 - pct):
            signals[i] = "buy"
            pos = True
        elif pos and closes[i] > ma[i] * (1 + pct):
            signals[i] = "sell"
            pos = False
    return signals


# ─── 主程式 ───

ALL_STRATEGIES = {
    "RSI(14) <30/>70":           lambda p: strategy_rsi_reversion(p, 30, 70),
    "RSI(14) <25/>65":           lambda p: strategy_rsi_reversion(p, 25, 65),
    "RSI(7) <30/>70":            lambda p: strategy_rsi_reversion(p, 30, 70, 7),
    "Bollinger(20,2) 反轉":       lambda p: strategy_bollinger_reversion(p),
    "Bollinger(20,1.5) 反轉":     lambda p: strategy_bollinger_reversion(p, 20, 1.5),
    "MA 交叉(5/20)":              lambda p: strategy_ma_crossover(p, 5, 20),
    "MA 交叉(10/40)":             lambda p: strategy_ma_crossover(p, 10, 40),
    "MACD 交叉":                  lambda p: strategy_macd_crossover(p),
    "Donchian(20) 突破":          lambda p: strategy_donchian_breakout(p, 20),
    "Donchian(10) 突破":          lambda p: strategy_donchian_breakout(p, 10),
    "跌8%買/漲10%賣":             lambda p: strategy_dip_buy(p, 0.08, 0.10),
    "跌5%買/漲8%賣":              lambda p: strategy_dip_buy(p, 0.05, 0.08),
    "跌10%買/漲15%賣":            lambda p: strategy_dip_buy(p, 0.10, 0.15),
    "RSI+Bollinger 結合":         lambda p: strategy_rsi_bollinger(p),
    "均值回歸 ATR(1.5)":          lambda p: strategy_mean_reversion_atr(p),
    "均值回歸 ATR(2.0)":          lambda p: strategy_mean_reversion_atr(p, 20, 2.0),
    "量增突破(2x)":               lambda p: strategy_volume_breakout(p, 2.0),
    "量增突破(1.5x)":             lambda p: strategy_volume_breakout(p, 1.5),
    "RSI波段(40/10天)":           lambda p: strategy_swing_rsi(p, 40, 10),
    "RSI波段(35/15天)":           lambda p: strategy_swing_rsi(p, 35, 15),
    "均線通道(20/5%)":            lambda p: strategy_ma_envelope(p, 20, 0.05),
    "均線通道(20/3%)":            lambda p: strategy_ma_envelope(p, 20, 0.03),
}


def run_all(stock_id, stock_name):
    prices = read_prices(stock_id)
    print(f"\n{'='*70}")
    print(f"  {stock_id} {stock_name}  ({prices[0]['date']} ~ {prices[-1]['date']}, {len(prices)} 筆)")
    print(f"{'='*70}")
    print(f"{'策略':<24} {'報酬率':>8} {'交易次數':>6} {'勝率':>6} {'平均持有':>6} {'最大回撤':>8} {'Sharpe':>7}")
    print("-" * 70)

    results = []
    for name, fn in ALL_STRATEGIES.items():
        signals = fn(prices)
        r = simulate(prices, signals)
        r["name"] = name
        results.append(r)
        mark = " ✓" if r["total_return_pct"] >= 50 else ""
        print(f"{name:<24} {r['total_return_pct']:>7.2f}% {r['trades']:>6} {r['win_rate']:>5.1f}% {r['avg_holding_days']:>5.1f}天 {r['max_drawdown_pct']:>7.2f}% {r['sharpe_ratio']:>7.2f}{mark}")

    results.sort(key=lambda x: x["total_return_pct"], reverse=True)
    best = results[0]
    print(f"\n最佳策略: {best['name']}  報酬: {best['total_return_pct']:.2f}%")
    return results


if __name__ == "__main__":
    for sid, name in [("8213", "志超"), ("3615", "安可")]:
        run_all(sid, name)
