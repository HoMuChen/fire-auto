"""
針對性策略第三輪
核心洞察：
  - RSI+Bollinger 在安可已 +101.9%（六年），只差 Sharpe（0.70）
  - 藍天 × RSI+Bollinger 每年都正向，但交易少（3/年），Sharpe 被稀釋
  - ATR通道回歸在 2020-2023 系統性虧損，需要 MA 斜率過濾

新策略方向：
  A. 加 MA20 斜率過濾 → 阻擋下跌趨勢中的買單
  B. 多信號疊加 → 提高每筆交易的確定性
  C. 高頻小利 → 多筆交易平滑 equity curve
  D. 順勢騎乘 → 對安可這種有大趨勢的股票用 trend-follow
"""

import sys
sys.path.insert(0, "/Users/mu/fire-auto")

from backtest import (
    read_prices, simulate, STRATEGIES, calc_sma, calc_rsi,
    calc_bollinger, calc_atr, calc_macd
)

TARGETS = [
    {"stock_id": "2362", "stock_name": "藍天"},
    {"stock_id": "8213", "stock_name": "志超"},
    {"stock_id": "3615", "stock_name": "安可"},
]


# ── 工具 ──────────────────────────────────────────────────────
def ma_slope_up(sma, i, lookback=5):
    """MA 的斜率是否向上（lookback 天前到今天）"""
    if sma[i] is None or i < lookback:
        return False
    prev = sma[i - lookback]
    return prev is not None and sma[i] > prev


def calc_kd(prices, period=9):
    closes = [p["close"] for p in prices]
    highs  = [p["high"]  for p in prices]
    lows   = [p["low"]   for p in prices]
    n = len(prices)
    k = [50.0] * n
    d = [50.0] * n
    for i in range(period - 1, n):
        hh = max(highs[i-period+1:i+1])
        ll = min(lows[i-period+1:i+1])
        rsv = (closes[i]-ll)/(hh-ll)*100 if hh!=ll else 50
        k[i] = k[i-1]*2/3 + rsv*1/3
        d[i] = d[i-1]*2/3 + k[i]*1/3
    return k, d


# ── 策略 A：MA20 斜率過濾 ───────────────────────────────────────

def strategy_rsi_bb_ma_slope(prices, rsi_buy=35, rsi_sell=65, bb_period=20, bb_std=2, slope_lb=5):
    """RSI+Bollinger + MA20 斜率向上才進場"""
    closes = [p["close"] for p in prices]
    rsi = calc_rsi(closes, 14)
    upper, lower, mid = calc_bollinger(closes, bb_period, bb_std)
    ma20 = calc_sma(closes, 20)
    signals = [None] * len(prices)
    for i in range(slope_lb, len(prices)):
        if rsi[i] is None or lower[i] is None:
            continue
        slope_ok = ma_slope_up(ma20, i, slope_lb)
        if slope_ok and rsi[i] < rsi_buy and closes[i] < lower[i]:
            signals[i] = "buy"
        elif rsi[i] > rsi_sell or closes[i] > upper[i]:
            signals[i] = "sell"
    return signals


def strategy_atr_ma_slope(prices, atr_mult=1.5, slope_lb=5):
    """ATR通道回歸 + MA20 斜率向上才進場"""
    closes = [p["close"] for p in prices]
    ma20 = calc_sma(closes, 20)
    atr  = calc_atr(prices, 14)
    signals = [None] * len(prices)
    for i in range(slope_lb, len(prices)):
        if ma20[i] is None or atr[i] is None:
            continue
        slope_ok = ma_slope_up(ma20, i, slope_lb)
        lower = ma20[i] - atr_mult * atr[i]
        if slope_ok and closes[i] < lower:
            signals[i] = "buy"
        elif closes[i] >= ma20[i]:
            signals[i] = "sell"
    return signals


def strategy_bollinger_ma_slope(prices, slope_lb=5):
    """布林觸底 + MA20 斜率向上"""
    closes = [p["close"] for p in prices]
    upper, lower, mid = calc_bollinger(closes, 20, 2)
    ma20 = calc_sma(closes, 20)
    signals = [None] * len(prices)
    for i in range(slope_lb, len(prices)):
        if lower[i] is None:
            continue
        if ma_slope_up(ma20, i, slope_lb) and closes[i] < lower[i]:
            signals[i] = "buy"
        elif closes[i] >= mid[i]:
            signals[i] = "sell"
    return signals


# ── 策略 B：多信號疊加 ──────────────────────────────────────────

def strategy_rsi_atr_combo(prices):
    """同時滿足 RSI<35 + 低於ATR通道下緣 才進場（雙重確認）"""
    closes = [p["close"] for p in prices]
    rsi  = calc_rsi(closes, 14)
    ma20 = calc_sma(closes, 20)
    atr  = calc_atr(prices, 14)
    upper, lower, mid = calc_bollinger(closes, 20, 2)
    signals = [None] * len(prices)
    for i in range(len(prices)):
        if rsi[i] is None or ma20[i] is None or atr[i] is None or lower[i] is None:
            continue
        atr_lower = ma20[i] - 1.5 * atr[i]
        rsi_ok = rsi[i] < 35
        atr_ok = closes[i] < atr_lower
        bb_ok  = closes[i] < lower[i]
        # 三條件至少兩個觸發
        cond_count = sum([rsi_ok, atr_ok, bb_ok])
        if cond_count >= 2:
            signals[i] = "buy"
        elif rsi[i] > 65 or closes[i] > ma20[i] * 1.05:
            signals[i] = "sell"
    return signals


def strategy_kd_rsi_double(prices):
    """KD 低位金叉 + RSI<40 + MA20 斜率向上"""
    closes = [p["close"] for p in prices]
    rsi  = calc_rsi(closes, 14)
    ma20 = calc_sma(closes, 20)
    k, d = calc_kd(prices, 9)
    signals = [None] * len(prices)
    for i in range(10, len(prices)):
        if rsi[i] is None or ma20[i] is None:
            continue
        kd_cross = k[i] > d[i] and k[i-1] <= d[i-1] and d[i] < 35
        rsi_ok = rsi[i] < 40
        slope_ok = ma_slope_up(ma20, i, 5)
        if kd_cross and rsi_ok and slope_ok:
            signals[i] = "buy"
        elif k[i] > 75 and k[i] < d[i] and k[i-1] >= d[i-1]:
            signals[i] = "sell"
    return signals


# ── 策略 C：高頻小利（profit target + MA 斜率）──────────────────

def strategy_rsi7_slope_profit(prices):
    """RSI(7)<25 + MA20斜率向上 → 進場；賺3%或持8天出場"""
    closes = [p["close"] for p in prices]
    rsi7 = calc_rsi(closes, 7)
    ma20 = calc_sma(closes, 20)
    signals = [None] * len(prices)
    for i in range(5, len(prices)):
        if rsi7[i] is None:
            continue
        if ma_slope_up(ma20, i, 5) and rsi7[i] < 25:
            signals[i] = "buy"
    return signals


def strategy_dev_slope_profit(prices, dev=-0.04, ma_period=20):
    """偏離MA20 -4% + MA斜率向上 → 回到MA出場"""
    closes = [p["close"] for p in prices]
    ma20 = calc_sma(closes, ma_period)
    signals = [None] * len(prices)
    for i in range(5, len(prices)):
        if ma20[i] is None or ma20[i] == 0:
            continue
        bias = (closes[i] - ma20[i]) / ma20[i]
        if ma_slope_up(ma20, i, 5) and bias < dev:
            signals[i] = "buy"
        elif bias >= 0:
            signals[i] = "sell"
    return signals


def strategy_consec_red_slope(prices, n_red=3):
    """連N黑 + MA20斜率向上 → 持10天"""
    ma20 = calc_sma([p["close"] for p in prices], 20)
    signals = [None] * len(prices)
    for i in range(n_red + 4, len(prices)):
        if not ma_slope_up(ma20, i, 5):
            continue
        all_red = all(prices[i-j]["close"] < prices[i-j]["open"] for j in range(n_red))
        if all_red:
            signals[i] = "buy"
    return signals


# ── 策略 D：順勢騎乘 ─────────────────────────────────────────────

def strategy_trend_ride_ma(prices, fast=20, slow=60):
    """MA20 上穿 MA60 進場，下穿出場（搭配 trailing stop）"""
    closes = [p["close"] for p in prices]
    maf = calc_sma(closes, fast)
    mas = calc_sma(closes, slow)
    signals = [None] * len(prices)
    for i in range(slow, len(prices)):
        if maf[i] is None or mas[i] is None or maf[i-1] is None or mas[i-1] is None:
            continue
        if maf[i] > mas[i] and maf[i-1] <= mas[i-1]:
            signals[i] = "buy"
        elif maf[i] < mas[i] and maf[i-1] >= mas[i-1]:
            signals[i] = "sell"
    return signals


def strategy_trend_ride_ma_10_30(prices):
    return strategy_trend_ride_ma(prices, fast=10, slow=30)


def strategy_rsi_trend_ride(prices):
    """RSI 持續 > 55 表示中期趨勢向上時持倉，跌破 45 出場"""
    closes = [p["close"] for p in prices]
    rsi = calc_rsi(closes, 14)
    ma20 = calc_sma(closes, 20)
    signals = [None] * len(prices)
    for i in range(1, len(prices)):
        if rsi[i] is None or rsi[i-1] is None or ma20[i] is None:
            continue
        # 進場：RSI 由 <55 升到 >55（趨勢啟動）
        if rsi[i] > 55 and rsi[i-1] <= 55 and closes[i] > ma20[i]:
            signals[i] = "buy"
        # 出場：RSI 跌破 45
        elif rsi[i] < 45:
            signals[i] = "sell"
    return signals


def strategy_macd_slope(prices):
    """MACD 轉正 + MA20 斜率向上"""
    closes = [p["close"] for p in prices]
    _, _, hist = calc_macd(closes, 12, 26, 9)
    ma20 = calc_sma(closes, 20)
    signals = [None] * len(prices)
    for i in range(1, len(prices)):
        if hist[i] is None or hist[i-1] is None:
            continue
        if hist[i-1] < 0 and hist[i] >= 0 and ma_slope_up(ma20, i, 5):
            signals[i] = "buy"
        elif hist[i-1] > 0 and hist[i] <= 0:
            signals[i] = "sell"
    return signals


def strategy_williams_slope(prices):
    """Williams %R < -80 + MA20 斜率向上"""
    from backtest import calc_williams_r
    wr = calc_williams_r(prices, 14)
    ma20 = calc_sma([p["close"] for p in prices], 20)
    signals = [None] * len(prices)
    for i in range(5, len(prices)):
        if wr[i] is None:
            continue
        if wr[i] < -80 and ma_slope_up(ma20, i, 5):
            signals[i] = "buy"
        elif wr[i] > -20:
            signals[i] = "sell"
    return signals


# ── 策略配置 ──────────────────────────────────────────────────────

NEW_STRATS = {
    "RSI+BB+MA斜率": {
        "fn": strategy_rsi_bb_ma_slope,
        "stop": {"type": "atr", "multiplier": 2.5, "period": 14},
        "exit": {"type": "signal"},
        "cat": "逆勢+MA斜率",
    },
    "ATR通道+MA斜率": {
        "fn": strategy_atr_ma_slope,
        "stop": {"type": "fixed_pct", "pct": 0.08},
        "exit": {"type": "signal"},
        "cat": "逆勢+MA斜率",
    },
    "布林+MA斜率": {
        "fn": strategy_bollinger_ma_slope,
        "stop": {"type": "fixed_pct", "pct": 0.05},
        "exit": {"type": "signal"},
        "cat": "逆勢+MA斜率",
    },
    "RSI+ATR雙重確認": {
        "fn": strategy_rsi_atr_combo,
        "stop": {"type": "atr", "multiplier": 2.0, "period": 14},
        "exit": {"type": "signal"},
        "cat": "逆勢+多信號",
    },
    "KD+RSI雙重確認": {
        "fn": strategy_kd_rsi_double,
        "stop": {"type": "atr", "multiplier": 2.0, "period": 14},
        "exit": {"type": "signal"},
        "cat": "逆勢+多信號",
    },
    "RSI7+MA斜率/8天": {
        "fn": strategy_rsi7_slope_profit,
        "stop": {"type": "atr", "multiplier": 1.5, "period": 7},
        "exit": {"type": "profit_or_hold", "profit_pct": 0.03, "days": 8},
        "cat": "逆勢高頻",
    },
    "乖離+MA斜率": {
        "fn": strategy_dev_slope_profit,
        "stop": {"type": "fixed_pct", "pct": 0.06},
        "exit": {"type": "signal"},
        "cat": "逆勢+MA斜率",
    },
    "連3黑+MA斜率/10天": {
        "fn": strategy_consec_red_slope,
        "stop": {"type": "fixed_pct", "pct": 0.05},
        "exit": {"type": "hold_days", "days": 10},
        "cat": "逆勢+MA斜率",
    },
    "均線騎乘(20/60)": {
        "fn": strategy_trend_ride_ma,
        "stop": {"type": "trailing_pct", "pct": 0.10},
        "exit": {"type": "signal"},
        "cat": "順勢",
    },
    "均線騎乘(10/30)": {
        "fn": strategy_trend_ride_ma_10_30,
        "stop": {"type": "trailing_pct", "pct": 0.08},
        "exit": {"type": "signal"},
        "cat": "順勢",
    },
    "RSI趨勢騎乘": {
        "fn": strategy_rsi_trend_ride,
        "stop": {"type": "trailing_pct", "pct": 0.08},
        "exit": {"type": "signal"},
        "cat": "順勢+RSI",
    },
    "MACD+MA斜率": {
        "fn": strategy_macd_slope,
        "stop": {"type": "trailing_pct", "pct": 0.07},
        "exit": {"type": "signal"},
        "cat": "順勢+MA斜率",
    },
    "威廉%R+MA斜率": {
        "fn": strategy_williams_slope,
        "stop": {"type": "atr", "multiplier": 2.0, "period": 14},
        "exit": {"type": "signal"},
        "cat": "逆勢+MA斜率",
    },
}

TARGET_SHARPE = 1.5
TARGET_RETURN = 100.0
TARGET_DD     = 20.0


def run():
    all_results = []

    for t in TARGETS:
        sid  = t["stock_id"]
        name = t["stock_name"]
        prices = read_prices(sid)

        for sname, cfg in NEW_STRATS.items():
            try:
                signals = cfg["fn"](prices)
                r = simulate(prices, signals, cfg["stop"], cfg.get("exit"))
            except Exception:
                continue
            if r["trades"] == 0:
                continue
            r["stock_id"]   = sid
            r["stock_name"] = name
            r["strategy"]   = sname
            r["cat"]        = cfg.get("cat", "")
            all_results.append(r)

    # 達標組合
    hits = [r for r in all_results
            if r["sharpe_ratio"] >= TARGET_SHARPE
            and r["total_return_pct"] >= TARGET_RETURN
            and r["max_drawdown_pct"] <= TARGET_DD]
    hits.sort(key=lambda x: x["sharpe_ratio"], reverse=True)

    print(f"{'='*90}")
    print(f"達標組合（Sharpe≥{TARGET_SHARPE}, 報酬≥{TARGET_RETURN}%, 回撤≤{TARGET_DD}%）：{len(hits)} 個")
    print(f"{'='*90}")
    for rank, r in enumerate(hits, 1):
        print(f"\n#{rank}  {r['stock_id']} {r['stock_name']} × {r['strategy']}  [{r['cat']}]")
        print(f"    報酬: {r['total_return_pct']:+.1f}%  Sharpe: {r['sharpe_ratio']:.2f}  "
              f"回撤: {r['max_drawdown_pct']:.1f}%  勝率: {r['win_rate']:.1f}%  "
              f"交易: {r['trades']}次  停損: {r['stop_loss_count']}次")

    # 全部結果按 Sharpe 排序
    all_results.sort(key=lambda x: x["sharpe_ratio"], reverse=True)
    print(f"\n{'─'*90}")
    print(f"全部新策略結果（前30，依 Sharpe）")
    print(f"{'─'*90}")
    print(f"{'策略':<24} {'股票':<8} {'報酬率':>8} {'Sharpe':>7} {'回撤':>7} {'勝率':>7} {'交易':>5} {'達標'}")
    print("-" * 80)
    for r in all_results[:30]:
        ok = "✓" if (r["sharpe_ratio"] >= TARGET_SHARPE
                     and r["total_return_pct"] >= TARGET_RETURN
                     and r["max_drawdown_pct"] <= TARGET_DD) else ""
        print(f"{r['strategy']:<24} {r['stock_id']} {r['stock_name']:<5} "
              f"{r['total_return_pct']:>7.1f}% {r['sharpe_ratio']:>7.2f} "
              f"{r['max_drawdown_pct']:>6.1f}% {r['win_rate']:>6.1f}% "
              f"{r['trades']:>5} {ok}")

    # 年度分解 top 5
    print(f"\n{'─'*90}")
    print("Top 5 年度分解")
    print(f"{'─'*90}")
    for r in all_results[:5]:
        sid  = r["stock_id"]
        name = r["stock_name"]
        sname = r["strategy"]
        cfg = NEW_STRATS[sname]
        prices = read_prices(sid)
        years = sorted(set(p["date"][:4] for p in prices))
        print(f"\n{sid} {name} × {sname}")
        cum = 1.0
        for y in years:
            yp = [p for p in prices if p["date"].startswith(y)]
            if len(yp) < 20:
                continue
            sig = cfg["fn"](yp)
            res = simulate(yp, sig, cfg["stop"], cfg.get("exit"))
            cum *= (1 + res["total_return_pct"]/100)
            print(f"  {y}: {res['total_return_pct']:+6.1f}%  "
                  f"交易{res['trades']:3}次  勝率{res['win_rate']:5.1f}%  "
                  f"Sharpe{res['sharpe_ratio']:6.2f}  累積={(cum-1)*100:+.1f}%")

    if len(hits) >= 3:
        print(f"\n✓ 目標達成！")
    else:
        print(f"\n✗ 尚未達標：{len(hits)}/3")
        # 最接近的
        candidates = [r for r in all_results if r["sharpe_ratio"] >= 1.0]
        if candidates:
            print("最接近達標的組合（Sharpe≥1.0）：")
            for r in candidates[:5]:
                issues = []
                if r["sharpe_ratio"] < TARGET_SHARPE: issues.append(f"Sharpe={r['sharpe_ratio']:.2f}<1.5")
                if r["total_return_pct"] < TARGET_RETURN: issues.append(f"報酬={r['total_return_pct']:.1f}%<100%")
                if r["max_drawdown_pct"] > TARGET_DD: issues.append(f"回撤={r['max_drawdown_pct']:.1f}%>20%")
                print(f"  {r['stock_id']} {r['stock_name']} × {r['strategy']}: {', '.join(issues)}")


if __name__ == "__main__":
    run()
