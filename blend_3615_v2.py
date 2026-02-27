"""
安可(3615) 策略精選與穩健性驗證 v2

初步掃描結果摘要：
  最佳A類: MA30 乖離<-6% fixed10% → ret=+92.5%  sh=0.59  dd=40.0%
  最佳振盪器: WR(21)<-80>-10 ATR×2.0 → ret=+190.3%  sh=0.76  dd=37.2%
  最佳順勢: MA20>60交叉 trail8% → ret=+150.8%  sh=0.77  dd=21.5%

策略邏輯分析：
  - A（乖離率）：逆勢，看「價格相對中期均線的距離」
  - B（WR振盪器）：逆勢，看「21日內%R動量超賣後回升」
  - C（MA均線交叉）：順勢，看「趨勢形成後追跡」
  → 三個邏輯確實不同：距離型逆勢 / 動量型逆勢 / 趨勢型順勢

本檔案做：
1. 進一步微調A類確保最佳（MA60 vs MA30）
2. 驗證RSI(14)也作為振盪器備選
3. 穩健性驗證：微調參數，確認結果不劇烈變動
4. 選出最終三策略，執行詳細混合模擬
"""

import sys
sys.path.insert(0, "/Users/mu/fire-auto")
from backtest import (
    read_prices, simulate, INITIAL_CAPITAL,
    calc_sma, calc_rsi, calc_atr, calc_williams_r, calc_bollinger,
    BUY_FEE, SELL_FEE, SELL_TAX
)


def build_kd(px, period=14, smooth=3):
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


prices = read_prices("3615")
closes = [p["close"] for p in prices]
n = len(prices)

print(f"安可(3615) {prices[0]['date']} ~ {prices[-1]['date']}, {n} 筆\n")


# ══════════════════════════════════════════════════════════════════
# 策略穩健性驗證（參數微調）
# ══════════════════════════════════════════════════════════════════

print("="*70)
print("策略A — 均線距離型：MA60 vs MA30 穩健性對比")
print("="*70)

print("\n[MA60 乖離率系列]")
for th in [-0.04, -0.045, -0.05, -0.055, -0.06, -0.065, -0.07, -0.08]:
    sma = calc_sma(closes, 60)
    sigs = [None]*n
    for i in range(60, n):
        if sma[i] is None or sma[i] == 0: continue
        b = (closes[i] - sma[i]) / sma[i]
        if b < th:   sigs[i] = "buy"
        elif b >= 0: sigs[i] = "sell"
    r = simulate(prices, sigs, {"type":"fixed_pct","pct":0.08}, {"type":"signal"})
    print(f"  MA60 b<{th*100:+.1f}% fixed8%  → ret={r['total_return_pct']:+6.1f}%  "
          f"sh={r['sharpe_ratio']:.2f}  dd={r['max_drawdown_pct']:.1f}%  T={r['trades']}")

print()
for th in [-0.04, -0.045, -0.05, -0.055, -0.06, -0.065, -0.07, -0.08]:
    sma = calc_sma(closes, 60)
    sigs = [None]*n
    for i in range(60, n):
        if sma[i] is None or sma[i] == 0: continue
        b = (closes[i] - sma[i]) / sma[i]
        if b < th:   sigs[i] = "buy"
        elif b >= 0: sigs[i] = "sell"
    r = simulate(prices, sigs, {"type":"trailing_pct","pct":0.08}, {"type":"signal"})
    print(f"  MA60 b<{th*100:+.1f}% trail8%  → ret={r['total_return_pct']:+6.1f}%  "
          f"sh={r['sharpe_ratio']:.2f}  dd={r['max_drawdown_pct']:.1f}%  T={r['trades']}")

print("\n[MA30 乖離率系列]")
for th in [-0.04, -0.05, -0.055, -0.06, -0.065, -0.07, -0.08]:
    sma = calc_sma(closes, 30)
    sigs = [None]*n
    for i in range(30, n):
        if sma[i] is None or sma[i] == 0: continue
        b = (closes[i] - sma[i]) / sma[i]
        if b < th:   sigs[i] = "buy"
        elif b >= 0: sigs[i] = "sell"
    r = simulate(prices, sigs, {"type":"fixed_pct","pct":0.10}, {"type":"signal"})
    print(f"  MA30 b<{th*100:+.1f}% fixed10% → ret={r['total_return_pct']:+6.1f}%  "
          f"sh={r['sharpe_ratio']:.2f}  dd={r['max_drawdown_pct']:.1f}%  T={r['trades']}")


print("\n" + "="*70)
print("策略B — Williams %R(21) 穩健性對比")
print("="*70)

print("\n[WR(21) 買入閾值 vs 賣出閾值 矩陣 — ATR2.0]")
print(f"{'':10}", end="")
for sell_th in [-30, -20, -15, -10, -5]:
    print(f"  賣>{sell_th:4d}", end="")
print()
for buy_th in [-95, -90, -87, -85, -82, -80, -75]:
    print(f"  買<{buy_th:4d} ", end="")
    wr_vals = calc_williams_r(prices, 21)
    for sell_th in [-30, -20, -15, -10, -5]:
        sigs = [None]*n
        for i in range(21, n):
            if wr_vals[i] is None: continue
            if wr_vals[i] < buy_th:    sigs[i] = "buy"
            elif wr_vals[i] > sell_th: sigs[i] = "sell"
        r = simulate(prices, sigs, {"type":"atr","multiplier":2.0,"period":14}, {"type":"signal"})
        print(f"  {r['total_return_pct']:+6.1f}%", end="")
    print()

print("\n[WR(14) vs WR(21) ATR倍數對比]")
for wr_p in [14, 21]:
    wr_vals = calc_williams_r(prices, wr_p)
    sigs_std = [None]*n
    for i in range(wr_p, n):
        if wr_vals[i] is None: continue
        if wr_vals[i] < -80:   sigs_std[i] = "buy"
        elif wr_vals[i] > -10: sigs_std[i] = "sell"
    for m in [1.0, 1.5, 2.0, 2.5, 3.0]:
        r = simulate(prices, sigs_std,
                     {"type":"atr","multiplier":m,"period":14}, {"type":"signal"})
        print(f"  WR({wr_p})<-80>-10 ATR×{m:.1f}  → ret={r['total_return_pct']:+6.1f}%  "
              f"sh={r['sharpe_ratio']:.2f}  dd={r['max_drawdown_pct']:.1f}%  T={r['trades']}")


print("\n" + "="*70)
print("策略B備選 — RSI(14) 穩健性")
print("="*70)

print("\n[RSI(14) 參數矩陣 — ATR2.5]")
print(f"{'':16}", end="")
for sell_th in [50, 55, 60, 65]:
    print(f"  賣>{sell_th:3d}", end="")
print()
for buy_th in [20, 25, 28, 30, 33, 35]:
    print(f"  買<{buy_th:3d}      ", end="")
    rsi_vals = calc_rsi(closes, 14)
    for sell_th in [50, 55, 60, 65]:
        sigs = [None]*n
        for i in range(14, n):
            if rsi_vals[i] is None: continue
            if rsi_vals[i] < buy_th:    sigs[i] = "buy"
            elif rsi_vals[i] > sell_th: sigs[i] = "sell"
        r = simulate(prices, sigs, {"type":"atr","multiplier":2.5,"period":14}, {"type":"signal"})
        print(f"  {r['total_return_pct']:+6.1f}%", end="")
    print()


print("\n" + "="*70)
print("策略C — 均線交叉 穩健性")
print("="*70)

print("\n[MA快慢線組合 trail8% 對比]")
for fast_p, slow_p in [(5,20),(10,30),(15,40),(20,60),(20,120),(30,90)]:
    fast_sma = calc_sma(closes, fast_p)
    slow_sma = calc_sma(closes, slow_p)
    for trail in [0.06, 0.08, 0.10, 0.12]:
        sigs = [None]*n
        for i in range(slow_p+1, n):
            if fast_sma[i] is None or slow_sma[i] is None: continue
            if fast_sma[i-1] < slow_sma[i-1] and fast_sma[i] >= slow_sma[i]:
                sigs[i] = "buy"
            elif fast_sma[i] < slow_sma[i]:
                sigs[i] = "sell"
        r = simulate(prices, sigs, {"type":"trailing_pct","pct":trail}, {"type":"signal"})
        if r["trades"] >= 2:
            print(f"  MA{fast_p:2d}>MA{slow_p:3d} trail{trail*100:.0f}%  → ret={r['total_return_pct']:+6.1f}%  "
                  f"sh={r['sharpe_ratio']:.2f}  dd={r['max_drawdown_pct']:.1f}%  T={r['trades']}  win={r['win_rate']:.0f}%")


# ══════════════════════════════════════════════════════════════════
# 最終策略確定
# ══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("最終策略選定（考慮穩健性後）")
print("="*70)

# --- 策略 A：MA60 乖離 -5% fixed8%（穩健性最佳的A類）---
# MA60 系列：從 -4% 到 -8% 報酬平穩遞減，不存在「唯一最優」
# 選 -5%/fixed8% 因為交易次數夠多(31次)、勝率61%、Sharpe0.71

sma60 = calc_sma(closes, 60)
sigs_A = [None]*n
for i in range(60, n):
    if sma60[i] is None or sma60[i] == 0: continue
    b = (closes[i] - sma60[i]) / sma60[i]
    if b < -0.05:  sigs_A[i] = "buy"
    elif b >= 0:   sigs_A[i] = "sell"
stop_A = {"type": "fixed_pct", "pct": 0.08}
exit_A = {"type": "signal"}

# --- 策略 B：RSI(14) <30 >55 ATR×2.5（穩健性最佳振盪器）---
# RSI(14)<30>55 ATR2.5: ret=+118.3%, sh=0.84, dd=23.1%, T=13
# 選RSI而非WR，因為WR雖然報酬更高但參數敏感度高（-80 vs -85 結果差異大）
# RSI在買入門檻從25~35、賣出從50~60的矩陣中表現相對穩定
rsi14 = calc_rsi(closes, 14)
sigs_B = [None]*n
for i in range(14, n):
    if rsi14[i] is None: continue
    if rsi14[i] < 30:   sigs_B[i] = "buy"
    elif rsi14[i] > 55: sigs_B[i] = "sell"
stop_B = {"type": "atr", "multiplier": 2.5, "period": 14}
exit_B = {"type": "signal"}

# --- 策略 C：MA20>MA60 交叉 trail8%（唯一順勢策略）---
# MA20>60交叉: ret=+150.8%, sh=0.77, dd=21.5%, T=15, win=67%
# trail 從6%~12% 報酬都在+67%~+150%，方向正確，只是幅度有差
# trail8% 平衡報酬與回撤，trail10% 亦可
fast20 = calc_sma(closes, 20)
slow60 = calc_sma(closes, 60)
sigs_C = [None]*n
for i in range(61, n):
    if fast20[i] is None or slow60[i] is None: continue
    if fast20[i-1] < slow60[i-1] and fast20[i] >= slow60[i]:
        sigs_C[i] = "buy"
    elif fast20[i] < slow60[i]:
        sigs_C[i] = "sell"
stop_C = {"type": "trailing_pct", "pct": 0.08}
exit_C = {"type": "signal"}

rA = simulate(prices, sigs_A, stop_A, exit_A)
rB = simulate(prices, sigs_B, stop_B, exit_B)
rC = simulate(prices, sigs_C, stop_C, exit_C)

print(f"\nA: MA60 乖離<-5% fixed8%")
print(f"   ret={rA['total_return_pct']:+.1f}%  sh={rA['sharpe_ratio']:.3f}  "
      f"dd={rA['max_drawdown_pct']:.1f}%  T={rA['trades']}  win={rA['win_rate']:.0f}%")
print(f"\nB: RSI(14)<30>55 ATR×2.5")
print(f"   ret={rB['total_return_pct']:+.1f}%  sh={rB['sharpe_ratio']:.3f}  "
      f"dd={rB['max_drawdown_pct']:.1f}%  T={rB['trades']}  win={rB['win_rate']:.0f}%")
print(f"\nC: MA20>MA60 交叉 trail8%")
print(f"   ret={rC['total_return_pct']:+.1f}%  sh={rC['sharpe_ratio']:.3f}  "
      f"dd={rC['max_drawdown_pct']:.1f}%  T={rC['trades']}  win={rC['win_rate']:.0f}%")

CONFIGS = {
    "A-MA60 乖離<-5% fixed8%": {"sigs": sigs_A, "stop": stop_A, "exit": exit_A},
    "B-RSI(14)<30>55 ATR×2.5": {"sigs": sigs_B, "stop": stop_B, "exit": exit_B},
    "C-MA20>MA60交叉 trail8%": {"sigs": sigs_C, "stop": stop_C, "exit": exit_C},
}

# ══════════════════════════════════════════════════════════════════
# 年度分解詳細分析
# ══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("年度分解（含各策略觸發次數）")
print("="*70)

years = sorted(set(p["date"][:4] for p in prices))

print(f"\n{'年份':<6} {'A報酬':>8} {'A次':>5} {'B報酬':>8} {'B次':>5} {'C報酬':>8} {'C次':>5}")
print("-"*55)

year_rets = {name: {} for name in CONFIGS}

for y in years:
    yp = [p for p in prices if p["date"].startswith(y)]
    if len(yp) < 20: continue
    yc = [p["close"] for p in yp]

    # 年度A
    ys60 = calc_sma(yc, 60)
    ysigA = [None]*len(yp)
    for i in range(60, len(yp)):
        if ys60[i] is None or ys60[i] == 0: continue
        b = (yc[i] - ys60[i]) / ys60[i]
        if b < -0.05:  ysigA[i] = "buy"
        elif b >= 0:   ysigA[i] = "sell"
    rA_y = simulate(yp, ysigA, stop_A, exit_A)

    # 年度B
    yr14 = calc_rsi(yc, 14)
    ysigB = [None]*len(yp)
    for i in range(14, len(yp)):
        if yr14[i] is None: continue
        if yr14[i] < 30:    ysigB[i] = "buy"
        elif yr14[i] > 55:  ysigB[i] = "sell"
    rB_y = simulate(yp, ysigB, stop_B, exit_B)

    # 年度C
    yf20 = calc_sma(yc, 20)
    ys60_2 = calc_sma(yc, 60)
    ysigC = [None]*len(yp)
    for i in range(61, len(yp)):
        if yf20[i] is None or ys60_2[i] is None: continue
        if yf20[i-1] < ys60_2[i-1] and yf20[i] >= ys60_2[i]: ysigC[i] = "buy"
        elif yf20[i] < ys60_2[i]: ysigC[i] = "sell"
    rC_y = simulate(yp, ysigC, stop_C, exit_C)

    year_rets["A-MA60 乖離<-5% fixed8%"][y] = rA_y["total_return_pct"]
    year_rets["B-RSI(14)<30>55 ATR×2.5"][y] = rB_y["total_return_pct"]
    year_rets["C-MA20>MA60交叉 trail8%"][y] = rC_y["total_return_pct"]

    print(f"{y:<6} {rA_y['total_return_pct']:>+7.1f}% {rA_y['trades']:>5}  "
          f"{rB_y['total_return_pct']:>+7.1f}% {rB_y['trades']:>5}  "
          f"{rC_y['total_return_pct']:>+7.1f}% {rC_y['trades']:>5}")

# 年度相關性
print("\n年度報酬相關性分析:")
names = list(CONFIGS.keys())
common_years = [y for y in years if all(y in year_rets[nm] for nm in names)]
for i, n1 in enumerate(names):
    for j, n2 in enumerate(names):
        if j <= i: continue
        vals1 = [year_rets[n1][y] for y in common_years]
        vals2 = [year_rets[n2][y] for y in common_years]
        avg1 = sum(vals1)/len(vals1)
        avg2 = sum(vals2)/len(vals2)
        cov = sum((vals1[k]-avg1)*(vals2[k]-avg2) for k in range(len(common_years)))
        std1 = (sum((v-avg1)**2 for v in vals1))**0.5
        std2 = (sum((v-avg2)**2 for v in vals2))**0.5
        corr = cov / (std1*std2) if std1*std2 > 0 else 0
        print(f"  corr(A vs B) = {corr:.3f}" if "A" in n1 and "B" in n2 else
              f"  corr(A vs C) = {corr:.3f}" if "A" in n1 and "C" in n2 else
              f"  corr(B vs C) = {corr:.3f}")


# ══════════════════════════════════════════════════════════════════
# 混合模擬（同 blend_2362.py 方法）
# ══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("混合模擬：各策略等資金獨立運作")
print("="*70)

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

    atr_vals = None
    if stop_cfg["type"] == "atr":
        atr_vals = calc_atr(prices, stop_cfg.get("period", 14))

    def update_stop(idx, avg_p):
        nonlocal stop_price, peak_price
        if stop_cfg["type"] == "fixed_pct":
            stop_price = avg_p * (1 - stop_cfg["pct"])
        elif stop_cfg["type"] == "atr":
            ea = atr_vals[idx] if (atr_vals and atr_vals[idx]) else avg_p*0.05
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

    for i, p in enumerate(prices):
        cl = p["close"]
        sig = sigs_raw[i]
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


def blend_equal_capital(wA=1/3, wB=1/3, wC=1/3):
    caps = [INITIAL_CAPITAL*wA, INITIAL_CAPITAL*wB, INITIAL_CAPITAL*wC]
    cfgs = list(CONFIGS.values())
    nm = len(prices)

    curves = []
    for i, cfg in enumerate(cfgs):
        eq, _ = simulate_curve(cfg["sigs"], cfg["stop"], cfg["exit"], caps[i])
        curves.append(eq)

    combined = [curves[0][i] + curves[1][i] + curves[2][i] for i in range(nm)]

    final = combined[-1]
    total_ret = (final - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    peak = 0; max_dd = 0
    for eq in combined:
        if eq > peak: peak = eq
        dd = (peak - eq) / peak if peak > 0 else 0
        if dd > max_dd: max_dd = dd

    daily_rets = [combined[i]/combined[i-1]-1 for i in range(1,nm) if combined[i-1]>0]
    avg_r = sum(daily_rets)/len(daily_rets)
    std_r = (sum((r-avg_r)**2 for r in daily_rets)/(len(daily_rets)-1))**0.5
    sharpe = (avg_r/std_r)*(252**0.5) if std_r>0 else 0

    return {
        "total_return_pct": round(total_ret, 2),
        "max_drawdown_pct":  round(max_dd*100, 2),
        "sharpe_ratio":      round(sharpe, 3),
        "final_capital":     round(final),
    }, combined


print("\n>>> 等權混合 (1/3 : 1/3 : 1/3)")
r, curve = blend_equal_capital()
print(f"混合報酬: {r['total_return_pct']:+.2f}%")
print(f"混合 Sharpe: {r['sharpe_ratio']:.3f}")
print(f"混合最大回撤: {r['max_drawdown_pct']:.2f}%")
print(f"最終資金: {r['final_capital']:,}")

print("\n年度混合表現:")
for y in years:
    idx_s = next((i for i,p in enumerate(prices) if p["date"].startswith(y)), None)
    idx_e = max((i for i,p in enumerate(prices) if p["date"].startswith(y)), default=None)
    if idx_s is None: continue
    eq_s = curve[idx_s-1] if idx_s > 0 else INITIAL_CAPITAL
    eq_e = curve[idx_e]
    yret = (eq_e - eq_s) / eq_s * 100
    total_so_far = (eq_e - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    print(f"  {y}: {yret:+6.1f}%  累積={(total_so_far):+.1f}%")

print("\n對比摘要:")
print(f"{'策略':<35} {'報酬':>8} {'Sharpe':>7} {'回撤':>7} {'T':>4} {'勝率':>6}")
print("-"*70)
for name, cfg in CONFIGS.items():
    ir = simulate(prices, cfg["sigs"], cfg["stop"], cfg["exit"])
    mark = " ✓" if ir['total_return_pct']>=80 and ir['max_drawdown_pct']<=30 else ""
    print(f"{name:<35} {ir['total_return_pct']:>7.1f}% {ir['sharpe_ratio']:>7.3f} "
          f"{ir['max_drawdown_pct']:>6.1f}% {ir['trades']:>4} {ir['win_rate']:>5.0f}%{mark}")
print(f"{'三策略混合 (A+B+C)':<35} {r['total_return_pct']:>7.1f}% {r['sharpe_ratio']:>7.3f} "
      f"{r['max_drawdown_pct']:>6.1f}%")

print("\n權重敏感性分析:")
print(f"{'A':>5} {'B':>5} {'C':>5} {'報酬%':>8} {'Sharpe':>7} {'回撤%':>7}")
print("-"*50)
weight_combos = [
    (1/3,1/3,1/3), (0.5,0.25,0.25), (0.6,0.2,0.2),
    (0.4,0.4,0.2), (0.4,0.2,0.4),  (0.2,0.4,0.4),
    (0.5,0.0,0.5), (0.5,0.5,0.0),  (0.0,0.5,0.5),
    (0.6,0.1,0.3), (0.3,0.4,0.3),  (0.2,0.3,0.5),
    (1.0,0.0,0.0), (0.0,1.0,0.0),  (0.0,0.0,1.0),
]
best_sharpe = 0; best_w = None
for wA,wB,wC in weight_combos:
    rb, _ = blend_equal_capital(wA,wB,wC)
    mark = ""
    if rb["sharpe_ratio"] > best_sharpe:
        best_sharpe = rb["sharpe_ratio"]
        best_w = (wA,wB,wC)
        mark = " ◀ 目前最佳Sharpe"
    print(f"{wA:>5.0%} {wB:>5.0%} {wC:>5.0%} "
          f"{rb['total_return_pct']:>7.1f}% {rb['sharpe_ratio']:>7.3f} "
          f"{rb['max_drawdown_pct']:>6.1f}%{mark}")

print(f"\n最佳Sharpe權重: A={best_w[0]:.0%} B={best_w[1]:.0%} C={best_w[2]:.0%}")
rb_best, _ = blend_equal_capital(*best_w)
print(f"  報酬={rb_best['total_return_pct']:+.2f}%  Sharpe={rb_best['sharpe_ratio']:.3f}  "
      f"回撤={rb_best['max_drawdown_pct']:.2f}%")

print("\n" + "="*70)
print("研究結論摘要")
print("="*70)
print("""
安可(3615) — 三策略最終選定

策略 A：均線距離型（逆勢·均線偏離）
  進場：收盤低於 MA60 超過 5%
  出場：收盤回到 MA60 以上（信號出場）
  停損：固定 8%
  邏輯：中長期趨勢偏離後均值回歸

策略 B：RSI超賣型（逆勢·動量振盪器）
  進場：RSI(14) < 30
  出場：RSI(14) > 55
  停損：ATR(14) × 2.5 自適應停損
  邏輯：短期動能過度賣出後反彈

策略 C：均線交叉型（順勢·趨勢追蹤）
  進場：MA20 上穿 MA60（黃金交叉）
  出場：MA20 跌破 MA60（死亡交叉）
  停損：移動停損 8%（追蹤高點）
  邏輯：中期趨勢形成後跟進

邏輯多元性：
  A → 逆勢，看「相對均線的距離」（絕對超跌）
  B → 逆勢，看「RSI動能衰竭程度」（超賣動能）
  C → 順勢，看「均線排列形成趨勢」（趨勢跟蹤）
  A/B 都是逆勢但觸發條件完全不同；C 方向相反，提供自然對沖
""")
