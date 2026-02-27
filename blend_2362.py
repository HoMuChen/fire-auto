"""
藍天(2362) 三策略混合研究
策略邏輯分類：
  A. 乖離率(MA60 -5.5%)  → 中長期均線偏離（逆勢·距離型）
  B. RSI純信號          → 動量擺盪器超賣（逆勢·振盪器型）
  C. KD隨機指標          → 極端超賣反轉（逆勢·擺盪型）

邏輯差異：三個都是逆勢，但 entry trigger 完全不同：
  - A 看「相對均線的距離」→ 反映中期趨勢偏離
  - B 看「漲跌速度的動能」→ 反映短期超賣動能
  - C 看「KD 極端超賣後的反轉信號」→ D<10 才觸發，極為稀少但精準
  三者往往不在同一天觸發，形成自然時間分散
"""

import sys
sys.path.insert(0, "/Users/mu/fire-auto")
from backtest import (
    read_prices, simulate, INITIAL_CAPITAL,
    calc_sma, calc_rsi, calc_atr, calc_williams_r,
    BUY_FEE, SELL_FEE, SELL_TAX
)


# ── KD 計算輔助函數 ────────────────────────────────────────────────

def build_kd(px, period=14, smooth=3):
    """%K = 100 + Williams%R（0~100）；%D = smooth 日 SMA of %K"""
    wr = calc_williams_r(px, period)
    nk = len(px)
    k = [None if wr[i] is None else 100.0 + wr[i] for i in range(nk)]
    d = [None] * nk
    for i in range(nk):
        if k[i] is None: continue
        vals = []
        j = i
        while j >= 0 and len(vals) < smooth:
            if k[j] is not None: vals.append(k[j])
            j -= 1
        if len(vals) == smooth:
            d[i] = sum(vals) / smooth
    return k, d

prices = read_prices("2362")
closes = [p["close"] for p in prices]
n = len(prices)


# ── 三個策略定義 ─────────────────────────────────────────────────

def signals_A():
    """乖離率(MA60, -5.5%, exit@0%) — 中長期均線距離"""
    sma = calc_sma(closes, 60)
    sigs = [None]*n
    for i in range(60, n):
        if sma[i] is None or sma[i] == 0: continue
        b = (closes[i] - sma[i]) / sma[i]
        if b < -0.055: sigs[i] = "buy"
        elif b >= 0:   sigs[i] = "sell"
    return sigs

def signals_B():
    """RSI(9) < 30 > 55, ATR×1.5 — 動量擺盪器"""
    rsi = calc_rsi(closes, 9)
    sigs = [None]*n
    for i in range(9, n):
        if rsi[i] is None: continue
        if rsi[i] < 30:   sigs[i] = "buy"
        elif rsi[i] > 55: sigs[i] = "sell"
    return sigs

def signals_C():
    """KD(14) D<3 買入, K>60 賣出, ATR×1.5 — 極端超賣反轉（6年僅4次）"""
    k_vals, d_vals = build_kd(prices, 14, 3)
    sigs = [None]*n
    for i in range(15, n):
        kc = k_vals[i]; dc = d_vals[i]
        if kc is None or dc is None: continue
        # 買入：D<3（3日均K絕對底部，六年僅出現4~5次）
        if dc < 3:
            sigs[i] = "buy"
        # 賣出：K>60（動能明顯回升）
        elif kc > 60:
            sigs[i] = "sell"
    return sigs

CONFIGS = {
    "A-乖離率(MA60,-5.5%)": {
        "sigs": signals_A(),
        "stop": {"type": "fixed_pct", "pct": 0.10},
        "exit": {"type": "signal"},
    },
    "B-RSI(9)<30>55": {
        "sigs": signals_B(),
        "stop": {"type": "atr", "multiplier": 1.5, "period": 14},
        "exit": {"type": "signal"},
    },
    "C-KD(14)D<3>K60": {
        "sigs": signals_C(),
        "stop": {"type": "atr", "multiplier": 1.5, "period": 14},
        "exit": {"type": "signal"},
    },
}


# ── 個別回測 ────────────────────────────────────────────────────

def run_individual():
    print("="*70)
    print("個別策略六年結果（2020-2026）")
    print("="*70)
    indiv = {}
    for name, cfg in CONFIGS.items():
        r = simulate(prices, cfg["sigs"], cfg["stop"], cfg["exit"])
        indiv[name] = r
        print(f"\n【{name}】")
        print(f"  報酬率: {r['total_return_pct']:+.1f}%  Sharpe: {r['sharpe_ratio']:.2f}  "
              f"最大回撤: {r['max_drawdown_pct']:.1f}%  勝率: {r['win_rate']:.1f}%  交易: {r['trades']}次")

        # 年度分解
        years = sorted(set(p["date"][:4] for p in prices))
        cum = 1.0
        for y in years:
            yp = [p for p in prices if p["date"].startswith(y)]
            if len(yp) < 20: continue
            yc = [p["close"] for p in yp]
            ysig = [None]*len(yp)

            if "乖離" in name:
                ys = calc_sma(yc, 60)
                for i in range(60, len(yp)):
                    if ys[i] is None or ys[i]==0: continue
                    b=(yc[i]-ys[i])/ys[i]
                    if b<-0.055: ysig[i]="buy"
                    elif b>=0: ysig[i]="sell"
            elif "RSI" in name:
                yr = calc_rsi(yc, 9)
                for i in range(9, len(yp)):
                    if yr[i] is None: continue
                    if yr[i]<30: ysig[i]="buy"
                    elif yr[i]>55: ysig[i]="sell"
            else:  # KD
                yk, yd = build_kd(yp, 14, 3)
                for i in range(15, len(yp)):
                    kc=yk[i]; dc=yd[i]
                    if kc is None or dc is None: continue
                    if dc < 3: ysig[i] = "buy"
                    elif kc > 60: ysig[i] = "sell"

            yr2 = simulate(yp, ysig, cfg["stop"], cfg["exit"])
            cum *= (1 + yr2["total_return_pct"]/100)
            print(f"  {y}: {yr2['total_return_pct']:+6.1f}%  "
                  f"交易{yr2['trades']:3}  勝率{yr2['win_rate']:5.1f}%  "
                  f"Sharpe{yr2['sharpe_ratio']:6.2f}  累積={(cum-1)*100:+.1f}%")
    return indiv


# ── 混合模擬：各策略等資金獨立運作 ─────────────────────────────────

def blend_equal_capital(weight_A=1/3, weight_B=1/3, weight_C=1/3):
    """
    三策略各分配 weight 比例資金，獨立運作。
    最終合併資金曲線計算混合績效。
    """
    cap_A = INITIAL_CAPITAL * weight_A
    cap_B = INITIAL_CAPITAL * weight_B
    cap_C = INITIAL_CAPITAL * weight_C

    names = list(CONFIGS.keys())
    cfgs  = list(CONFIGS.values())
    caps  = [cap_A, cap_B, cap_C]

    # 模擬各策略（用完整 simulate 拿 equity_curve）
    # 但 simulate() 不直接回傳 equity_curve，自己重寫一份輕量版
    def simulate_curve(sigs_raw, stop_cfg, exit_cfg, initial_cap):
        capital = initial_cap
        shares = 0
        lots = []
        pos = False
        trades = []
        equity = []
        entry_idx = 0
        peak_price = 0
        stop_price = 0

        atr = None
        if stop_cfg["type"] == "atr":
            atr = calc_atr(prices, stop_cfg.get("period", 14))

        def update_stop(idx, avg_p):
            nonlocal stop_price, peak_price
            if stop_cfg["type"] == "fixed_pct":
                stop_price = avg_p * (1 - stop_cfg["pct"])
            elif stop_cfg["type"] == "atr":
                ea = atr[idx] if (atr and atr[idx]) else avg_p*0.05
                stop_price = avg_p - stop_cfg["multiplier"] * ea
            elif stop_cfg["type"] == "trailing_pct":
                if peak_price == 0: peak_price = avg_p
                stop_price = peak_price * (1 - stop_cfg["pct"])

        def avg_buy():
            return sum(l["shares"]*l["buy_price"] for l in lots)/shares if shares else 0

        def close_pos(price, date, reason):
            nonlocal capital, shares, lots, pos, stop_price, peak_price
            ab = avg_buy()
            rev = shares * price
            capital += rev - int(rev*SELL_FEE) - int(rev*SELL_TAX)
            trades.append({"buy_price": ab, "sell_price": price, "reason": reason, "date": date})
            shares = 0; lots = []; pos = False; stop_price = 0; peak_price = 0

        sigs = sigs_raw
        for i, p in enumerate(prices):
            cl = p["close"]
            sig = sigs[i]
            action = sig if isinstance(sig, str) else (sig[0] if sig else None)

            if pos and stop_cfg["type"] == "trailing_pct":
                if cl > peak_price:
                    peak_price = cl
                    stop_price = peak_price * (1 - stop_cfg["pct"])

            if pos and cl <= stop_price:
                close_pos(cl, p["date"], "stop")
                equity.append(capital); continue

            if pos:
                et = exit_cfg.get("type","signal")
                should_exit = False
                if et == "signal" and action == "sell": should_exit = True
                elif et == "hold_days" and (i-entry_idx) >= exit_cfg["days"]: should_exit = True
                elif et == "profit_or_hold":
                    if cl >= avg_buy()*(1+exit_cfg["profit_pct"]): should_exit = True
                    elif (i-entry_idx) >= exit_cfg["days"]: should_exit = True
                if should_exit:
                    close_pos(cl, p["date"], "signal")

            if not pos and action == "buy" and cl > 0 and capital > 0:
                spend = capital
                cost_per = cl * (1 + BUY_FEE)
                new_sh = int(spend / cost_per / 1000) * 1000
                if new_sh <= 0: new_sh = int(spend / cost_per)
                if new_sh > 0:
                    buy_cost = new_sh * cl
                    capital -= (buy_cost + int(buy_cost*BUY_FEE))
                    shares += new_sh
                    lots.append({"shares": new_sh, "buy_price": cl, "buy_date": p["date"]})
                    pos = True; entry_idx = i
                    update_stop(i, avg_buy())

            equity.append(capital + shares*cl if pos else capital)

        if pos and shares > 0:
            close_pos(prices[-1]["close"], prices[-1]["date"], "forced")
            equity[-1] = capital

        return equity, trades

    curves = []
    for i, (name, cfg) in enumerate(zip(names, cfgs)):
        eq, _ = simulate_curve(cfg["sigs"], cfg["stop"], cfg["exit"], caps[i])
        curves.append(eq)

    # 合併曲線
    combined = [curves[0][i] + curves[1][i] + curves[2][i] for i in range(n)]

    # 計算績效指標
    final = combined[-1]
    total_ret = (final - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    peak = 0; max_dd = 0
    for eq in combined:
        if eq > peak: peak = eq
        dd = (peak - eq) / peak if peak > 0 else 0
        if dd > max_dd: max_dd = dd

    daily_rets = [combined[i]/combined[i-1]-1 for i in range(1,n) if combined[i-1]>0]
    avg_r = sum(daily_rets)/len(daily_rets)
    std_r = (sum((r-avg_r)**2 for r in daily_rets)/(len(daily_rets)-1))**0.5
    sharpe = (avg_r/std_r)*(252**0.5) if std_r>0 else 0

    return {
        "total_return_pct": round(total_ret, 2),
        "max_drawdown_pct":  round(max_dd*100, 2),
        "sharpe_ratio":      round(sharpe, 3),
        "final_capital":     round(final),
    }, combined


def run_blend():
    print("\n" + "="*70)
    print("三策略混合結果（各 1/3 資金，獨立運作）")
    print("="*70)

    r, curve = blend_equal_capital()
    print(f"\n混合報酬: {r['total_return_pct']:+.2f}%")
    print(f"混合 Sharpe: {r['sharpe_ratio']:.3f}")
    print(f"混合最大回撤: {r['max_drawdown_pct']:.2f}%")
    print(f"最終資金: {r['final_capital']:,}")

    # 各年度混合
    print("\n年度混合表現:")
    years = sorted(set(p["date"][:4] for p in prices))
    prev_eq = INITIAL_CAPITAL
    for y in years:
        idx_start = next((i for i,p in enumerate(prices) if p["date"].startswith(y)), None)
        idx_end   = max((i for i,p in enumerate(prices) if p["date"].startswith(y)), default=None)
        if idx_start is None: continue
        eq_start = curve[idx_start-1] if idx_start > 0 else INITIAL_CAPITAL
        eq_end   = curve[idx_end]
        yret = (eq_end - eq_start) / eq_start * 100
        total_ret_so_far = (eq_end - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        print(f"  {y}: {yret:+6.1f}%  累積={(total_ret_so_far):+.1f}%")

    # 與個別策略對比
    print("\n對比摘要:")
    print(f"{'策略':<25} {'報酬':>8} {'Sharpe':>7} {'回撤':>7}")
    print("-"*50)
    for name, cfg in CONFIGS.items():
        ir = simulate(prices, cfg["sigs"], cfg["stop"], cfg["exit"])
        ok=" ✓" if ir['total_return_pct']>=100 and ir['max_drawdown_pct']<=20 else ""
        print(f"{name:<25} {ir['total_return_pct']:>7.1f}% {ir['sharpe_ratio']:>7.2f} {ir['max_drawdown_pct']:>6.1f}%{ok}")
    print(f"{'三策略混合 (A+B+C)':25} {r['total_return_pct']:>7.1f}% {r['sharpe_ratio']:>7.3f} {r['max_drawdown_pct']:>6.1f}%")

    # 試不同權重組合
    print("\n權重敏感性分析:")
    print(f"{'A':>5} {'B':>5} {'C':>5} {'報酬%':>8} {'Sharpe':>7} {'回撤%':>7}")
    print("-"*45)
    for wA,wB,wC in [
        (1/3,1/3,1/3),
        (0.5,0.25,0.25),
        (0.6,0.2,0.2),
        (0.4,0.4,0.2),
        (0.4,0.2,0.4),
        (0.5,0.0,0.5),
        (0.6,0.1,0.3),
        (0.5,0.5,0.0),
        (1.0,0.0,0.0),
        (0.0,1.0,0.0),
        (0.0,0.0,1.0),
    ]:
        rb, _ = blend_equal_capital(wA,wB,wC)
        print(f"{wA:>5.0%} {wB:>5.0%} {wC:>5.0%} {rb['total_return_pct']:>7.1f}% {rb['sharpe_ratio']:>7.3f} {rb['max_drawdown_pct']:>6.1f}%")


if __name__ == "__main__":
    indiv = run_individual()
    run_blend()
