"""
策略專屬停損機制的五大策略回測
每個策略有自己的停損邏輯：
  - RSI階梯 → 分批建倉（1/3, 1/3, 1/3）+ ATR(14)×2 停損
  - 網格 → 固定 10%（2倍網格間距）
  - 連3黑 → 固定 5%（短持快停損）
  - RSI+Bollinger → ATR(14)×2.5 停損
  - KD → 移動停損 8%（追蹤高點）
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

CSV_FIELDS = ["date", "open", "high", "low", "close", "volume", "trading_money", "trading_turnover", "spread"]


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
    """Average True Range"""
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


# ─── 回測引擎（支援分批建倉 + 策略專屬停損）───

def simulate(prices, signals, stop_config) -> dict:
    """
    signals: list of None / "buy" / "sell" / ("buy", fraction)
      - "buy" = 用全部可用現金買入
      - ("buy", 0.33) = 用可用現金的 33% 買入（分批建倉）
      - "sell" = 賣出全部持股

    stop_config:
      {"type": "fixed_pct", "pct": 0.10}
      {"type": "atr", "multiplier": 2, "period": 14}
      {"type": "trailing_pct", "pct": 0.08}
    """
    capital = INITIAL_CAPITAL
    shares = 0
    lots = []  # [{"shares": N, "buy_price": P, "buy_date": D}, ...]
    position_open = False
    trades = []
    equity_curve = []

    # 預計算 ATR
    atr = None
    if stop_config["type"] == "atr":
        atr = calc_atr(prices, stop_config.get("period", 14))

    stop_price = 0
    peak_price = 0

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

        # 解析信號
        sig = signals[i]
        if isinstance(sig, tuple):
            action, fraction = sig
        elif sig is not None:
            action = sig
            fraction = 1.0
        else:
            action = None
            fraction = 0

        # 更新移動停損高點
        if position_open and stop_config["type"] == "trailing_pct":
            if close > peak_price:
                peak_price = close
                stop_price = peak_price * (1 - stop_config["pct"])

        # 停損檢查（優先於策略訊號）
        if position_open and close <= stop_price:
            close_position(close, p["date"], "stop_loss")
            equity_curve.append(capital)
            continue

        # 買入（支援分批）
        if action == "buy" and close > 0 and capital > 0:
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
                update_stop(i, calc_avg_buy())

        # 賣出（全部）
        elif action == "sell" and position_open:
            close_position(close, p["date"], "signal")

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

    # max drawdown
    peak = 0
    max_dd = 0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    # sharpe
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

    # avg holding days
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


# ─── 五大策略 ───

def strategy_rsi_ladder(prices, levels=None, sell_rsi=55):
    """RSI 階梯分批建倉：每個 RSI 關卡用剩餘資金的等比例買入，全部回升後賣出"""
    if levels is None:
        levels = [40, 30, 20]
    closes = [p["close"] for p in prices]
    rsi = calc_rsi(closes, 14)
    signals = [None] * len(prices)
    current_level = 0
    in_position = False
    n_levels = len(levels)

    for i in range(len(prices)):
        if rsi[i] is None:
            continue

        # 還有階梯可用，且 RSI 跌破當前關卡 → 分批買入
        if current_level < n_levels and rsi[i] < levels[current_level]:
            remaining = n_levels - current_level
            fraction = 1.0 / remaining  # 剩餘現金的等比例
            signals[i] = ("buy", fraction)
            current_level += 1
            in_position = True

        # RSI 回升 → 全部賣出，重置階梯
        elif in_position and rsi[i] > sell_rsi:
            signals[i] = "sell"
            in_position = False
            current_level = 0

    return signals


def strategy_grid(prices, grid_pct=0.05, take_profit=0.06):
    closes = [p["close"] for p in prices]
    signals = [None] * len(prices)
    pos = False
    last_ref = closes[0]
    bp = 0
    for i in range(1, len(prices)):
        c = closes[i]
        if not pos:
            if c <= last_ref * (1 - grid_pct):
                signals[i] = "buy"
                pos = True
                bp = c
                last_ref = c
        else:
            if c >= bp * (1 + take_profit):
                signals[i] = "sell"
                pos = False
                last_ref = c
            elif c < bp * (1 - grid_pct):
                last_ref = c
    return signals


def strategy_consecutive_red(prices, n_red=3, hold_days=10):
    closes = [p["close"] for p in prices]
    signals = [None] * len(prices)
    pos = False
    entry_i = 0
    for i in range(n_red, len(prices)):
        if not pos:
            all_red = all(prices[i - j]["close"] < prices[i - j]["open"] for j in range(n_red))
            if all_red:
                signals[i] = "buy"
                pos = True
                entry_i = i
        elif pos and (i - entry_i) >= hold_days:
            signals[i] = "sell"
            pos = False
    return signals


def strategy_rsi_bollinger(prices, rsi_buy=35, rsi_sell=65, bb_period=20, bb_std=2):
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


def strategy_kd(prices, period=9, buy_th=30, sell_th=70):
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
    pos = False
    for i in range(period, n):
        if not pos and k_vals[i] < buy_th and k_vals[i] > d_vals[i] and k_vals[i - 1] <= d_vals[i - 1]:
            signals[i] = "buy"
            pos = True
        elif pos and k_vals[i] > sell_th and k_vals[i] < d_vals[i] and k_vals[i - 1] >= d_vals[i - 1]:
            signals[i] = "sell"
            pos = False
    return signals


# ─── 策略配置（含專屬停損）───

STRATEGIES = {
    "RSI階梯[40,30,20]→55": {
        "fn": lambda p: strategy_rsi_ladder(p),
        "stop": {"type": "atr", "multiplier": 2, "period": 14},
    },
    "網格(5%/6%)": {
        "fn": lambda p: strategy_grid(p),
        "stop": {"type": "fixed_pct", "pct": 0.10},
    },
    "連3黑/持10天": {
        "fn": lambda p: strategy_consecutive_red(p),
        "stop": {"type": "fixed_pct", "pct": 0.05},
    },
    "RSI+Bollinger": {
        "fn": lambda p: strategy_rsi_bollinger(p),
        "stop": {"type": "atr", "multiplier": 2.5, "period": 14},
    },
    "KD(9) 30/70": {
        "fn": lambda p: strategy_kd(p),
        "stop": {"type": "trailing_pct", "pct": 0.08},
    },
}

STRATEGY_DESC = {
    "RSI階梯[40,30,20]→55": {
        "category": "逆勢",
        "description": "RSI(14)分批建倉：跌破40買1/3，跌破30再買1/3，跌破20買最後1/3。RSI回升>55全部賣出。停損：均價 - 2×ATR(14)",
        "entry": "RSI(14) < 40/30/20 分批買入（各1/3資金）",
        "exit": "RSI(14) > 55 全部賣出 或 停損（均價 - 2×ATR）",
        "stop_desc": "ATR停損：均價 - 2×ATR(14)，依個股波動自動調整",
    },
    "網格(5%/6%)": {
        "category": "逆勢",
        "description": "每跌5%買入，每漲6%賣出。停損：買入價 -10%（2倍網格間距）",
        "entry": "價格從參考點下跌5%",
        "exit": "從買入價上漲6% 或 停損 -10%",
        "stop_desc": "固定停損 -10%（2倍網格間距）",
    },
    "連3黑/持10天": {
        "category": "順勢",
        "description": "連續3天收黑後買入，固定持有10個交易日賣出。停損：買入價 -5%（短持緊停損）",
        "entry": "連續3天收黑K線",
        "exit": "持有10個交易日後賣出 或 停損 -5%",
        "stop_desc": "固定停損 -5%（短期持有，快速止損）",
    },
    "RSI+Bollinger": {
        "category": "逆勢",
        "description": "RSI(14)<35且價格低於布林下軌(20,2)時買入，RSI>65或碰布林上軌時賣出。停損：買入價 - 2.5×ATR(14)",
        "entry": "RSI(14) < 35 且 收盤價 < Bollinger下軌(20,2)",
        "exit": "RSI(14) > 65 或 收盤價 > Bollinger上軌(20,2) 或 停損（買入價 - 2.5×ATR）",
        "stop_desc": "ATR停損：買入價 - 2.5×ATR(14)，雙重確認進場給予較大空間",
    },
    "KD(9) 30/70": {
        "category": "逆勢",
        "description": "KD隨機指標(9日)，K值<30且K上穿D時買入，K值>70且K下穿D時賣出。停損：從最高點回撤8%",
        "entry": "K(9) < 30 且 K 上穿 D",
        "exit": "K(9) > 70 且 K 下穿 D 或 停損（從高點回撤8%）",
        "stop_desc": "移動停損 -8%（追蹤持倉期間最高價）",
    },
}


def format_stop_config(stop_config):
    if stop_config["type"] == "fixed_pct":
        return f"固定停損 -{stop_config['pct']*100:.0f}%"
    elif stop_config["type"] == "atr":
        return f"ATR({stop_config.get('period',14)})×{stop_config['multiplier']}停損"
    elif stop_config["type"] == "trailing_pct":
        return f"移動停損 -{stop_config['pct']*100:.0f}%"
    return str(stop_config)


def main():
    candidates = json.load(open("/tmp/sideways_volatile.json"))
    stock_info = {s["stock_id"]: s for s in json.load(open(STOCKS_PATH))}

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
                r = simulate(prices, signals, cfg["stop"])
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

        if (idx + 1) % 100 == 0:
            print(f"  進度: {idx+1}/{len(candidates)}")

    # 篩選 >75%
    winners = [r for r in all_results if r["total_return_pct"] >= 75]
    winners.sort(key=lambda x: x["total_return_pct"], reverse=True)

    # 每檔股票最佳
    stock_best = {}
    for w in winners:
        sid = w["stock_id"]
        if sid not in stock_best or w["total_return_pct"] > stock_best[sid]["total_return_pct"]:
            stock_best[sid] = w
    unique_stocks = sorted(stock_best.values(), key=lambda x: x["total_return_pct"], reverse=True)

    print(f"\n全部回測: {len(all_results)} 筆")
    print(f"報酬 > 75%: {len(winners)} 筆, 不重複股票: {len(unique_stocks)} 檔")
    print(f"\n{'排名':>3} {'代號':>6} {'名稱':<8} {'最佳策略':<20} {'停損類型':<16} {'報酬率':>8} {'交易':>4} {'勝率':>6} {'持有':>6} {'回撤':>7} {'Sharpe':>7} {'停損次':>5}")
    print("-" * 115)
    for rank, r in enumerate(unique_stocks[:40], 1):
        print(f"{rank:>3} {r['stock_id']:>6} {r['stock_name']:<8} {r['strategy']:<20} {r['stop_type']:<16} {r['total_return_pct']:>7.1f}% {r['trades']:>4} {r['win_rate']:>5.1f}% {r['avg_holding_days']:>5.1f}天 {r['max_drawdown_pct']:>6.1f}% {r['sharpe_ratio']:>7.2f} {r['stop_loss_count']:>5}")

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
            desc = STRATEGY_DESC.get(s["strategy"], {})
            viable.append({
                "rank": rank,
                "name": s["strategy"],
                "category": desc.get("category", ""),
                "description": desc.get("description", ""),
                "entry": desc.get("entry", ""),
                "exit": desc.get("exit", ""),
                "stop_loss": desc.get("stop_desc", ""),
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

    print(f"\n已建立 {created} 個策略檔案（策略專屬停損 + RSI分批建倉）")
    json.dump(winners, open("/tmp/winners_75.json", "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
