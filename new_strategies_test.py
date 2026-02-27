"""
新策略回測：加入趨勢過濾器，針對藍天(2362)、志超(8213)、安可(3615)
目標：Sharpe ≥ 1.5、六年報酬 ≥ 100%、最大回撤 ≤ 20%
"""

import sys
sys.path.insert(0, "/Users/mu/fire-auto")

from backtest import (
    read_prices, simulate, STRATEGIES, format_stop_config,
    calc_rsi, calc_sma, calc_bollinger, calc_atr, calc_macd, calc_williams_r
)
from pathlib import Path

DATA_DIR = Path("/Users/mu/fire-auto/data/stock_prices")

TARGETS = [
    {"stock_id": "2362", "stock_name": "藍天"},
    {"stock_id": "8213", "stock_name": "志超"},
    {"stock_id": "3615", "stock_name": "安可"},
]

# ─── 新指標 ───

def calc_kd(prices, period=9):
    closes = [p["close"] for p in prices]
    highs  = [p["high"] for p in prices]
    lows   = [p["low"] for p in prices]
    n = len(prices)
    k_vals = [50.0] * n
    d_vals = [50.0] * n
    for i in range(period - 1, n):
        hh = max(highs[i - period + 1:i + 1])
        ll = min(lows[i - period + 1:i + 1])
        rsv = (closes[i] - ll) / (hh - ll) * 100 if hh != ll else 50
        k_vals[i] = k_vals[i-1] * 2/3 + rsv * 1/3
        d_vals[i] = d_vals[i-1] * 2/3 + k_vals[i] * 1/3
    return k_vals, d_vals


# ═══════════════════════════════════════════════════════
# 新策略（都帶趨勢過濾器）
# ═══════════════════════════════════════════════════════

def strategy_trend_rsi(prices, ma_period=60, rsi_buy=35, rsi_sell=65):
    """趨勢確認RSI反彈：收盤>MA60 + RSI<35 + 今日止跌"""
    closes = [p["close"] for p in prices]
    ma = calc_sma(closes, ma_period)
    rsi = calc_rsi(closes, 14)
    signals = [None] * len(prices)
    for i in range(1, len(prices)):
        if ma[i] is None or rsi[i] is None:
            continue
        if closes[i] > ma[i] and rsi[i] < rsi_buy and closes[i] > closes[i-1]:
            signals[i] = "buy"
        elif rsi[i] > rsi_sell or closes[i] < ma[i]:
            signals[i] = "sell"
    return signals


def strategy_regime_rsi(prices, ma_period=200, rsi_oversold=30, rsi_sell=60):
    """市場狀態切換：MA200多頭區才進場，RSI超賣後回升+縮量"""
    closes = [p["close"] for p in prices]
    vols   = [p["volume"] for p in prices]
    ma = calc_sma(closes, ma_period)
    rsi = calc_rsi(closes, 14)
    vol_ma = calc_sma([float(v) for v in vols], 20)
    signals = [None] * len(prices)
    for i in range(1, len(prices)):
        if ma[i] is None or rsi[i] is None or vol_ma[i] is None:
            continue
        in_bull = closes[i] > ma[i]
        rsi_recovery = rsi[i] > rsi_oversold and rsi[i-1] <= rsi_oversold
        vol_shrink = vols[i] < vol_ma[i]
        if in_bull and rsi_recovery and vol_shrink:
            signals[i] = "buy"
        elif rsi[i] > rsi_sell or not in_bull:
            signals[i] = "sell"
    return signals


def strategy_52w_high(prices):
    """52週高點突破：收盤突破過去252日高點+放量+收紅"""
    closes = [p["close"] for p in prices]
    vols   = [p["volume"] for p in prices]
    vol_ma = calc_sma([float(v) for v in vols], 60)
    ma20   = calc_sma(closes, 20)
    signals = [None] * len(prices)
    for i in range(252, len(prices)):
        if vol_ma[i] is None or ma20[i] is None:
            continue
        prev_high = max(closes[i-252:i])
        today_red_to_green = closes[i] > prices[i]["open"]
        vol_surge = vols[i] > vol_ma[i] * 1.3
        if closes[i] > prev_high and vol_surge and today_red_to_green:
            signals[i] = "buy"
        elif closes[i] < ma20[i]:
            signals[i] = "sell"
    return signals


def strategy_triple_ma(prices):
    """三線多頭排列：MA5>MA20>MA60新形成+放量+突破前高"""
    closes = [p["close"] for p in prices]
    vols   = [p["volume"] for p in prices]
    ma5  = calc_sma(closes, 5)
    ma20 = calc_sma(closes, 20)
    ma60 = calc_sma(closes, 60)
    vol_ma = calc_sma([float(v) for v in vols], 20)
    signals = [None] * len(prices)
    for i in range(61, len(prices)):
        if any(x is None for x in [ma5[i], ma20[i], ma60[i], ma5[i-1], ma20[i-1], vol_ma[i]]):
            continue
        aligned_now  = ma5[i]   > ma20[i]   > ma60[i]
        aligned_prev = ma5[i-1] > ma20[i-1] > ma60[i-1]
        just_aligned = aligned_now and not aligned_prev
        vol_ok = vols[i] > vol_ma[i] * 1.2
        above_prev_high = closes[i] > prices[i-1]["high"]
        if just_aligned and vol_ok and above_prev_high:
            signals[i] = "buy"
        # 排列破壞出場
        if not aligned_now and ma5[i] < ma20[i] and ma5[i-1] >= ma20[i-1]:
            signals[i] = "sell"
    return signals


def strategy_vcp_pullback(prices):
    """VCP拉回進場：MA60上升+從近高回調10-15%+RSI中性+縮量"""
    closes = [p["close"] for p in prices]
    vols   = [p["volume"] for p in prices]
    ma60 = calc_sma(closes, 60)
    rsi  = calc_rsi(closes, 14)
    vol_ma = calc_sma([float(v) for v in vols], 10)
    signals = [None] * len(prices)
    for i in range(65, len(prices)):
        if any(x is None for x in [ma60[i], ma60[i-20], rsi[i], vol_ma[i]]):
            continue
        ma60_rising = ma60[i] > ma60[i-20]
        recent_high = max(closes[i-20:i+1])
        ratio = closes[i] / recent_high
        in_pullback = 0.85 <= ratio <= 0.97
        rsi_neutral = 38 <= rsi[i] <= 58
        vol_shrink = vols[i] < vol_ma[i]
        if ma60_rising and in_pullback and rsi_neutral and vol_shrink:
            signals[i] = "buy"
        if closes[i] > recent_high:
            signals[i] = "sell"
    return signals


def strategy_nr7(prices):
    """NR7波動收縮：今日振幅為7日最小+MA20上方，隔日突破"""
    closes = [p["close"] for p in prices]
    ma20 = calc_sma(closes, 20)
    signals = [None] * len(prices)
    for i in range(8, len(prices)):
        if ma20[i] is None:
            continue
        today_range = prices[i]["high"] - prices[i]["low"]
        min7 = min(prices[j]["high"] - prices[j]["low"] for j in range(i-6, i+1))
        is_nr7 = today_range <= min7
        if is_nr7 and closes[i] > ma20[i]:
            signals[i] = "buy"
    return signals


def strategy_volume_anomaly(prices):
    """量異常偵測：MA120多頭+昨日爆量暴跌+今日止跌縮量收紅"""
    closes = [p["close"] for p in prices]
    vols   = [p["volume"] for p in prices]
    ma120  = calc_sma(closes, 120)
    vol60  = calc_sma([float(v) for v in vols], 60)
    signals = [None] * len(prices)
    for i in range(2, len(prices)):
        if ma120[i] is None or vol60[i-1] is None:
            continue
        in_bull = closes[i] > ma120[i]
        prev_drop = closes[i-1] < closes[i-2] * 0.97
        prev_bigvol = vols[i-1] > vol60[i-1] * 3.0
        today_green = closes[i] > prices[i]["open"]
        vol_calm = vols[i] < vols[i-1]
        if in_bull and prev_drop and prev_bigvol and today_green and vol_calm:
            signals[i] = "buy"
    return signals


def strategy_filtered_kd(prices):
    """過濾型KD交叉：MA20上升+KD<30超賣區金叉+RSI>30"""
    closes = [p["close"] for p in prices]
    ma20 = calc_sma(closes, 20)
    rsi  = calc_rsi(closes, 14)
    k, d = calc_kd(prices, 9)
    n = len(prices)
    signals = [None] * n
    for i in range(25, n):
        if ma20[i] is None or ma20[i-5] is None or rsi[i] is None:
            continue
        ma20_up = ma20[i] > ma20[i-5]
        kd_cross = k[i] > d[i] and k[i-1] <= d[i-1]
        oversold  = d[i] < 30 and k[i] < 30
        rsi_ok = rsi[i] > 30
        if ma20_up and kd_cross and oversold and rsi_ok:
            signals[i] = "buy"
        elif k[i] > 80 and k[i] < d[i] and k[i-1] >= d[i-1]:
            signals[i] = "sell"
    return signals


def strategy_bollinger_squeeze(prices):
    """布林壓縮突破：BB寬度壓縮後向上突破+放量"""
    closes = [p["close"] for p in prices]
    vols   = [p["volume"] for p in prices]
    vol_ma = calc_sma([float(v) for v in vols], 20)
    period = 20
    signals = [None] * len(prices)
    for i in range(period + 20, len(prices)):
        if vol_ma[i] is None:
            continue
        window = closes[i-period+1:i+1]
        m = sum(window) / period
        std = (sum((x-m)**2 for x in window) / period) ** 0.5
        if m == 0:
            continue
        curr_width = 4 * std / m

        widths = []
        for j in range(i-20, i):
            w = closes[j-period+1:j+1]
            m2 = sum(w) / period
            s2 = (sum((x-m2)**2 for x in w) / period) ** 0.5
            widths.append(4 * s2 / m2 if m2 > 0 else 0)
        avg_width = sum(widths) / len(widths) if widths else 0

        upper = m + 2 * std
        is_squeeze   = curr_width < avg_width * 0.8
        above_upper  = closes[i] > upper
        vol_surge    = vols[i] > vol_ma[i] * 1.5
        if is_squeeze and above_upper and vol_surge:
            signals[i] = "buy"
        elif closes[i] < m:
            signals[i] = "sell"
    return signals


def strategy_dual_rsi(prices):
    """雙RSI：RSI28>50趨勢向上+RSI5<25超短線超賣"""
    closes = [p["close"] for p in prices]
    rsi5  = calc_rsi(closes, 5)
    rsi28 = calc_rsi(closes, 28)
    signals = [None] * len(prices)
    for i in range(len(prices)):
        if rsi5[i] is None or rsi28[i] is None:
            continue
        if rsi28[i] > 50 and rsi5[i] < 25:
            signals[i] = "buy"
        elif rsi5[i] > 70:
            signals[i] = "sell"
    return signals


def strategy_ma_cross_vol(prices):
    """量能確認均線突破：MA10上穿MA30+放量（順勢改良版）"""
    closes = [p["close"] for p in prices]
    vols   = [p["volume"] for p in prices]
    ma10 = calc_sma(closes, 10)
    ma30 = calc_sma(closes, 30)
    ma60 = calc_sma(closes, 60)
    vol_ma = calc_sma([float(v) for v in vols], 20)
    signals = [None] * len(prices)
    for i in range(31, len(prices)):
        if any(x is None for x in [ma10[i], ma30[i], ma60[i], ma10[i-1], ma30[i-1], vol_ma[i]]):
            continue
        cross_up   = ma10[i] > ma30[i] and ma10[i-1] <= ma30[i-1]
        trend_ok   = closes[i] > ma60[i]
        vol_ok     = vols[i] > vol_ma[i] * 1.1
        if cross_up and trend_ok and vol_ok:
            signals[i] = "buy"
        elif ma10[i] < ma30[i] and ma10[i-1] >= ma30[i-1]:
            signals[i] = "sell"
    return signals


def strategy_high_base_breakout(prices, lookback=60):
    """高基期突破：股價在近60日高點附近整理後再突破"""
    closes = [p["close"] for p in prices]
    vols   = [p["volume"] for p in prices]
    vol_ma = calc_sma([float(v) for v in vols], 20)
    ma20   = calc_sma(closes, 20)
    signals = [None] * len(prices)
    for i in range(lookback, len(prices)):
        if vol_ma[i] is None or ma20[i] is None:
            continue
        window = closes[i-lookback:i]
        w_high = max(window)
        w_low  = min(window)
        # 高基期：近期振幅壓縮（整理幅度 < 20%），且在高位
        base_range = (w_high - w_low) / w_high
        in_high_base = base_range < 0.20 and closes[i-1] > w_high * 0.90
        breakout = closes[i] > w_high and vols[i] > vol_ma[i] * 1.5
        if in_high_base and breakout:
            signals[i] = "buy"
        elif closes[i] < ma20[i]:
            signals[i] = "sell"
    return signals


def strategy_rsi_ma_combo(prices):
    """複合：RSI回升+均線多頭+縮量，更嚴格的買入條件"""
    closes = [p["close"] for p in prices]
    vols   = [p["volume"] for p in prices]
    rsi  = calc_rsi(closes, 14)
    ma20 = calc_sma(closes, 20)
    ma60 = calc_sma(closes, 60)
    vol_ma = calc_sma([float(v) for v in vols], 20)
    signals = [None] * len(prices)
    for i in range(1, len(prices)):
        if any(x is None for x in [rsi[i], ma20[i], ma60[i], vol_ma[i]]):
            continue
        trend = ma20[i] > ma60[i]           # 短中期均線多頭
        rsi_ok = 30 < rsi[i] < 50           # RSI從超賣回升至中性
        rsi_rising = rsi[i] > rsi[i-1]      # RSI向上
        vol_shrink = vols[i] < vol_ma[i]    # 縮量
        price_above = closes[i] > ma20[i]   # 不低於均線
        if trend and rsi_ok and rsi_rising and vol_shrink and price_above:
            signals[i] = "buy"
        elif rsi[i] > 65 or closes[i] < ma60[i]:
            signals[i] = "sell"
    return signals


# ─── 新策略配置 ───

NEW_STRATEGIES = {
    "趨勢RSI反彈(MA60)": {
        "fn": strategy_trend_rsi,
        "stop": {"type": "atr", "multiplier": 2.0, "period": 14},
        "exit": {"type": "signal"},
        "cat": "逆勢+過濾",
    },
    "市場狀態切換(MA200)": {
        "fn": strategy_regime_rsi,
        "stop": {"type": "atr", "multiplier": 2.5, "period": 14},
        "exit": {"type": "signal"},
        "cat": "逆勢+過濾",
    },
    "52週高點突破": {
        "fn": strategy_52w_high,
        "stop": {"type": "trailing_pct", "pct": 0.10},
        "exit": {"type": "signal"},
        "cat": "順勢動量",
    },
    "三線多頭排列": {
        "fn": strategy_triple_ma,
        "stop": {"type": "trailing_pct", "pct": 0.08},
        "exit": {"type": "signal"},
        "cat": "順勢",
    },
    "VCP拉回進場": {
        "fn": strategy_vcp_pullback,
        "stop": {"type": "trailing_pct", "pct": 0.07},
        "exit": {"type": "signal"},
        "cat": "順勢+拉回",
    },
    "NR7收縮/持5天": {
        "fn": strategy_nr7,
        "stop": {"type": "fixed_pct", "pct": 0.04},
        "exit": {"type": "hold_days", "days": 5},
        "cat": "波動率",
    },
    "量異常偵測/持5天": {
        "fn": strategy_volume_anomaly,
        "stop": {"type": "fixed_pct", "pct": 0.05},
        "exit": {"type": "hold_days", "days": 5},
        "cat": "量價+過濾",
    },
    "過濾型KD交叉": {
        "fn": strategy_filtered_kd,
        "stop": {"type": "atr", "multiplier": 2.0, "period": 14},
        "exit": {"type": "signal"},
        "cat": "逆勢+過濾",
    },
    "布林壓縮突破": {
        "fn": strategy_bollinger_squeeze,
        "stop": {"type": "atr", "multiplier": 1.5, "period": 14},
        "exit": {"type": "signal"},
        "cat": "順勢突破",
    },
    "雙RSI(5/28)": {
        "fn": strategy_dual_rsi,
        "stop": {"type": "atr", "multiplier": 2.0, "period": 14},
        "exit": {"type": "profit_or_hold", "profit_pct": 0.05, "days": 10},
        "cat": "逆勢+過濾",
    },
    "均線量能突破(10/30)": {
        "fn": strategy_ma_cross_vol,
        "stop": {"type": "trailing_pct", "pct": 0.07},
        "exit": {"type": "signal"},
        "cat": "順勢",
    },
    "高基期突破": {
        "fn": strategy_high_base_breakout,
        "stop": {"type": "trailing_pct", "pct": 0.09},
        "exit": {"type": "signal"},
        "cat": "型態突破",
    },
    "RSI均線複合": {
        "fn": strategy_rsi_ma_combo,
        "stop": {"type": "atr", "multiplier": 2.0, "period": 14},
        "exit": {"type": "signal"},
        "cat": "複合",
    },
}


TARGET_SHARPE = 1.5
TARGET_RETURN = 100.0
TARGET_DD     = 20.0


def run():
    all_results = []

    print("策略數（新）:", len(NEW_STRATEGIES))
    print("同時跑舊24策略做對比")
    all_strats = {**STRATEGIES, **NEW_STRATEGIES}

    for t in TARGETS:
        sid  = t["stock_id"]
        name = t["stock_name"]
        prices = read_prices(sid)
        print(f"\n{sid} {name}: {prices[0]['date']} ~ {prices[-1]['date']} ({len(prices)} 筆)")

        for sname, cfg in all_strats.items():
            try:
                signals = cfg["fn"](prices)
                r = simulate(prices, signals, cfg["stop"], cfg.get("exit"))
            except Exception as e:
                continue
            if r["trades"] == 0:
                continue
            r["stock_id"]   = sid
            r["stock_name"] = name
            r["strategy"]   = sname
            r["cat"]        = cfg.get("cat", "")
            r["is_new"]     = sname in NEW_STRATEGIES
            all_results.append(r)

    # 達標組合（三個條件同時滿足）
    hits = [r for r in all_results
            if r["sharpe_ratio"] >= TARGET_SHARPE
            and r["total_return_pct"] >= TARGET_RETURN
            and r["max_drawdown_pct"] <= TARGET_DD]

    hits.sort(key=lambda x: x["sharpe_ratio"], reverse=True)

    print(f"\n{'='*90}")
    print(f"達標（Sharpe≥{TARGET_SHARPE}, 報酬≥{TARGET_RETURN}%, 回撤≤{TARGET_DD}%）：{len(hits)} 個")
    print(f"{'='*90}")

    for rank, r in enumerate(hits, 1):
        tag = "★新" if r["is_new"] else "  舊"
        print(f"\n#{rank} {tag}  {r['stock_id']} {r['stock_name']} × {r['strategy']}  [{r['cat']}]")
        print(f"    報酬率: {r['total_return_pct']:+.1f}%  |  Sharpe: {r['sharpe_ratio']:.2f}  "
              f"|  回撤: {r['max_drawdown_pct']:.1f}%  |  勝率: {r['win_rate']:.1f}%  "
              f"|  交易: {r['trades']}  |  停損: {r['stop_loss_count']}次")

    # 各策略最佳 Sharpe 一覽（新策略）
    print(f"\n{'─'*80}")
    print("各新策略最佳結果（依 Sharpe 排序）")
    print(f"{'─'*80}")
    new_results = [r for r in all_results if r["is_new"]]
    best_per_strat = {}
    for r in new_results:
        key = r["strategy"]
        if key not in best_per_strat or r["sharpe_ratio"] > best_per_strat[key]["sharpe_ratio"]:
            best_per_strat[key] = r
    ranked = sorted(best_per_strat.values(), key=lambda x: x["sharpe_ratio"], reverse=True)
    print(f"{'策略':<22} {'股票':<6} {'報酬率':>8} {'Sharpe':>7} {'回撤':>7} {'勝率':>7} {'交易':>5} {'達標':>4}")
    print("-" * 75)
    for r in ranked:
        ok = "✓" if (r["sharpe_ratio"] >= TARGET_SHARPE
                     and r["total_return_pct"] >= TARGET_RETURN
                     and r["max_drawdown_pct"] <= TARGET_DD) else ""
        print(f"{r['strategy']:<22} {r['stock_id']:<6} {r['total_return_pct']:>7.1f}% "
              f"{r['sharpe_ratio']:>7.2f} {r['max_drawdown_pct']:>6.1f}% "
              f"{r['win_rate']:>6.1f}% {r['trades']:>5} {ok:>4}")

    # 最接近達標但未達標（激勵方向）
    print(f"\n{'─'*80}")
    print("最接近達標（Sharpe≥1.2，全策略）")
    print(f"{'─'*80}")
    close = [r for r in all_results if r["sharpe_ratio"] >= 1.2]
    close.sort(key=lambda x: x["sharpe_ratio"], reverse=True)
    print(f"{'策略':<22} {'股票':<8} {'報酬率':>8} {'Sharpe':>7} {'回撤':>7} {'勝率':>7} {'交易':>5} {'新/舊':>4}")
    print("-" * 75)
    for r in close[:20]:
        tag = "新" if r["is_new"] else "舊"
        ok = " ✓" if (r["sharpe_ratio"] >= TARGET_SHARPE
                      and r["total_return_pct"] >= TARGET_RETURN
                      and r["max_drawdown_pct"] <= TARGET_DD) else ""
        print(f"{r['strategy']:<22} {r['stock_id']} {r['stock_name']:<5} "
              f"{r['total_return_pct']:>7.1f}% {r['sharpe_ratio']:>7.2f} "
              f"{r['max_drawdown_pct']:>6.1f}% {r['win_rate']:>6.1f}% "
              f"{r['trades']:>5} {tag:>4}{ok}")

    if len(hits) >= 3:
        print(f"\n✓ 目標達成！找到 {len(hits)} 個達標組合")
    else:
        print(f"\n✗ 尚未達標：{len(hits)} / 3 個")
        print("  → 可考慮微調參數或組合現有接近達標的策略")


if __name__ == "__main__":
    run()
