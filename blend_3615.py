"""
安可(3615) 三策略混合研究
策略邏輯分類：
  A. 均線距離型（乖離率）→ 中長期均線偏離（逆勢·距離型）
  B. 振盪器型（RSI/KD/WR）→ 動量超賣（逆勢·振盪器型）
  C. 第三類（布林通道/順勢/量價）→ 邏輯不同的第三策略

"""

import sys
sys.path.insert(0, "/Users/mu/fire-auto")
from backtest import (
    read_prices, simulate, INITIAL_CAPITAL,
    calc_sma, calc_rsi, calc_atr, calc_williams_r, calc_bollinger,
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


prices = read_prices("3615")
closes = [p["close"] for p in prices]
n = len(prices)

print(f"安可(3615) 資料期間: {prices[0]['date']} ~ {prices[-1]['date']}, {n} 筆")
print(f"收盤價範圍: {min(closes):.2f} ~ {max(closes):.2f}")
print()


# ══════════════════════════════════════════════════════════════════
# 第一步：廣泛掃描
# ══════════════════════════════════════════════════════════════════

print("="*70)
print("第一步：廣泛策略掃描")
print("="*70)

scan_results = []

# ─── A類 均線距離型（乖離率）─────────────────────────────────────

print("\n【A類 均線距離型】")

for ma_period in [20, 30, 60]:
    sma = calc_sma(closes, ma_period)
    for threshold in [-0.05, -0.06, -0.07, -0.08]:
        for stop_type, stop_cfg in [
            ("fixed8%", {"type": "fixed_pct", "pct": 0.08}),
            ("fixed10%", {"type": "fixed_pct", "pct": 0.10}),
            ("trail8%", {"type": "trailing_pct", "pct": 0.08}),
        ]:
            sigs = [None]*n
            for i in range(ma_period, n):
                if sma[i] is None or sma[i] == 0: continue
                b = (closes[i] - sma[i]) / sma[i]
                if b < threshold: sigs[i] = "buy"
                elif b >= 0:      sigs[i] = "sell"
            r = simulate(prices, sigs, stop_cfg, {"type": "signal"})
            name = f"A-MA{ma_period} b<{threshold*100:.0f}% {stop_type}"
            scan_results.append({"name": name, "cat": "A", **r})
            if r["trades"] >= 3:
                print(f"  {name:<40} ret={r['total_return_pct']:+6.1f}%  "
                      f"sharpe={r['sharpe_ratio']:.2f}  dd={r['max_drawdown_pct']:.1f}%  "
                      f"T={r['trades']}  win={r['win_rate']:.0f}%")

# ─── B類 RSI振盪器 ─────────────────────────────────────────────

print("\n【B類 RSI振盪器】")

for rsi_p in [7, 9, 14]:
    rsi_vals = calc_rsi(closes, rsi_p)
    for buy_th, sell_th in [(25,50),(25,55),(30,55),(30,60),(35,60)]:
        for stop_cfg, stop_label in [
            ({"type":"atr","multiplier":1.5,"period":14}, "ATR1.5"),
            ({"type":"atr","multiplier":2.0,"period":14}, "ATR2.0"),
            ({"type":"atr","multiplier":2.5,"period":14}, "ATR2.5"),
        ]:
            sigs = [None]*n
            for i in range(rsi_p, n):
                if rsi_vals[i] is None: continue
                if rsi_vals[i] < buy_th:   sigs[i] = "buy"
                elif rsi_vals[i] > sell_th: sigs[i] = "sell"
            r = simulate(prices, sigs, stop_cfg, {"type": "signal"})
            name = f"B-RSI({rsi_p})<{buy_th}>{sell_th} {stop_label}"
            scan_results.append({"name": name, "cat": "B", **r})
            if r["trades"] >= 3:
                print(f"  {name:<45} ret={r['total_return_pct']:+6.1f}%  "
                      f"sharpe={r['sharpe_ratio']:.2f}  dd={r['max_drawdown_pct']:.1f}%  "
                      f"T={r['trades']}  win={r['win_rate']:.0f}%")

# ─── C類 布林通道 ─────────────────────────────────────────────

print("\n【C類 布林通道】")

for bb_std in [1.5, 2.0, 2.5]:
    upper, lower, mid = calc_bollinger(closes, 20, bb_std)
    for exit_type, exit_label in [
        ({"type":"signal"}, "sig"),
        ({"type":"hold_days","days":10}, "hold10"),
    ]:
        for stop_cfg, stop_label in [
            ({"type":"atr","multiplier":2.0,"period":14}, "ATR2.0"),
            ({"type":"atr","multiplier":2.5,"period":14}, "ATR2.5"),
        ]:
            sigs = [None]*n
            for i in range(20, n):
                if lower[i] is None or mid[i] is None: continue
                if closes[i] < lower[i]: sigs[i] = "buy"
                elif closes[i] > mid[i]: sigs[i] = "sell"
            r = simulate(prices, sigs, stop_cfg, exit_type)
            name = f"C-BB(20,{bb_std}σ) {exit_label} {stop_label}"
            scan_results.append({"name": name, "cat": "C", **r})
            if r["trades"] >= 3:
                print(f"  {name:<45} ret={r['total_return_pct']:+6.1f}%  "
                      f"sharpe={r['sharpe_ratio']:.2f}  dd={r['max_drawdown_pct']:.1f}%  "
                      f"T={r['trades']}  win={r['win_rate']:.0f}%")

# RSI + Bollinger 組合
print("\n  [布林+RSI組合]")
for bb_std in [1.5, 2.0]:
    upper, lower, mid = calc_bollinger(closes, 20, bb_std)
    for rsi_th in [30, 35, 40]:
        rsi_vals = calc_rsi(closes, 14)
        for stop_cfg, stop_label in [
            ({"type":"atr","multiplier":2.0,"period":14}, "ATR2.0"),
            ({"type":"atr","multiplier":2.5,"period":14}, "ATR2.5"),
        ]:
            sigs = [None]*n
            for i in range(20, n):
                if lower[i] is None or rsi_vals[i] is None: continue
                if closes[i] < lower[i] and rsi_vals[i] < rsi_th: sigs[i] = "buy"
                elif closes[i] > mid[i]: sigs[i] = "sell"
            r = simulate(prices, sigs, stop_cfg, {"type":"signal"})
            name = f"C-BB(20,{bb_std}σ)+RSI<{rsi_th} {stop_label}"
            scan_results.append({"name": name, "cat": "C", **r})
            if r["trades"] >= 3:
                print(f"  {name:<45} ret={r['total_return_pct']:+6.1f}%  "
                      f"sharpe={r['sharpe_ratio']:.2f}  dd={r['max_drawdown_pct']:.1f}%  "
                      f"T={r['trades']}  win={r['win_rate']:.0f}%")

# ─── D類 KD極端超賣 ─────────────────────────────────────────────

print("\n【D類 KD極端超賣】")

for kd_period in [9, 14]:
    for smooth in [3, 5]:
        k_vals, d_vals = build_kd(prices, kd_period, smooth)
        for d_buy in [3, 5, 10]:
            for k_sell in [50, 60, 70]:
                for stop_cfg, stop_label in [
                    ({"type":"atr","multiplier":1.5,"period":14}, "ATR1.5"),
                    ({"type":"atr","multiplier":2.0,"period":14}, "ATR2.0"),
                ]:
                    sigs = [None]*n
                    for i in range(kd_period+2, n):
                        kc = k_vals[i]; dc = d_vals[i]
                        if kc is None or dc is None: continue
                        if dc < d_buy:   sigs[i] = "buy"
                        elif kc > k_sell: sigs[i] = "sell"
                    r = simulate(prices, sigs, stop_cfg, {"type":"signal"})
                    name = f"D-KD({kd_period},{smooth}) D<{d_buy} K>{k_sell} {stop_label}"
                    scan_results.append({"name": name, "cat": "D", **r})
                    if r["trades"] >= 3:
                        print(f"  {name:<50} ret={r['total_return_pct']:+6.1f}%  "
                              f"sharpe={r['sharpe_ratio']:.2f}  dd={r['max_drawdown_pct']:.1f}%  "
                              f"T={r['trades']}  win={r['win_rate']:.0f}%")

# ─── E類 Williams %R ─────────────────────────────────────────────

print("\n【E類 Williams %R】")

for wr_p in [14, 21]:
    wr_vals = calc_williams_r(prices, wr_p)
    for buy_th, sell_th in [(-90,-20),(-85,-15),(-80,-10),(-90,-30),(-85,-20)]:
        for stop_cfg, stop_label in [
            ({"type":"atr","multiplier":1.5,"period":14}, "ATR1.5"),
            ({"type":"atr","multiplier":2.0,"period":14}, "ATR2.0"),
        ]:
            sigs = [None]*n
            for i in range(wr_p, n):
                if wr_vals[i] is None: continue
                if wr_vals[i] < buy_th:   sigs[i] = "buy"
                elif wr_vals[i] > sell_th: sigs[i] = "sell"
            r = simulate(prices, sigs, stop_cfg, {"type":"signal"})
            name = f"E-WR({wr_p})<{buy_th}>{sell_th} {stop_label}"
            scan_results.append({"name": name, "cat": "E", **r})
            if r["trades"] >= 3:
                print(f"  {name:<45} ret={r['total_return_pct']:+6.1f}%  "
                      f"sharpe={r['sharpe_ratio']:.2f}  dd={r['max_drawdown_pct']:.1f}%  "
                      f"T={r['trades']}  win={r['win_rate']:.0f}%")

# ─── F類 連跌型態 ─────────────────────────────────────────────

print("\n【F類 連跌型態】")

for consec_days in [3, 4, 5]:
    for cum_drop in [-0.05, -0.08, -0.10, -0.12]:
        for stop_cfg, stop_label in [
            ({"type":"fixed_pct","pct":0.08}, "fixed8%"),
            ({"type":"atr","multiplier":2.0,"period":14}, "ATR2.0"),
        ]:
            for hold_days in [5, 10, 15]:
                sigs = [None]*n
                for i in range(consec_days, n):
                    # 連跌N天
                    consec_down = all(closes[i-k] < closes[i-k-1] for k in range(consec_days))
                    # 累計跌幅
                    cum_ret = (closes[i] - closes[i-consec_days]) / closes[i-consec_days]
                    if consec_down and cum_ret < cum_drop:
                        sigs[i] = "buy"
                r = simulate(prices, sigs, stop_cfg, {"type":"hold_days","days":hold_days})
                name = f"F-連跌{consec_days}天累跌{cum_drop*100:.0f}% hold{hold_days} {stop_label}"
                scan_results.append({"name": name, "cat": "F", **r})
                if r["trades"] >= 3 and r["total_return_pct"] > 0:
                    print(f"  {name:<55} ret={r['total_return_pct']:+6.1f}%  "
                          f"sharpe={r['sharpe_ratio']:.2f}  dd={r['max_drawdown_pct']:.1f}%  "
                          f"T={r['trades']}  win={r['win_rate']:.0f}%")

# ─── G類 順勢策略 ─────────────────────────────────────────────

print("\n【G類 順勢策略】")

# G1: 均線黃金交叉
for fast_p, slow_p in [(20,60),(10,30),(20,120)]:
    fast_sma = calc_sma(closes, fast_p)
    slow_sma = calc_sma(closes, slow_p)
    for stop_cfg, stop_label in [
        ({"type":"trailing_pct","pct":0.08}, "trail8%"),
        ({"type":"trailing_pct","pct":0.10}, "trail10%"),
        ({"type":"atr","multiplier":2.0,"period":14}, "ATR2.0"),
    ]:
        sigs = [None]*n
        for i in range(slow_p+1, n):
            if fast_sma[i] is None or slow_sma[i] is None: continue
            if fast_sma[i-1] < slow_sma[i-1] and fast_sma[i] >= slow_sma[i]:
                sigs[i] = "buy"
            elif fast_sma[i] < slow_sma[i]:
                sigs[i] = "sell"
        r = simulate(prices, sigs, stop_cfg, {"type":"signal"})
        name = f"G-MA{fast_p}>{slow_p}交叉 {stop_label}"
        scan_results.append({"name": name, "cat": "G", **r})
        if r["trades"] >= 2:
            print(f"  {name:<45} ret={r['total_return_pct']:+6.1f}%  "
                  f"sharpe={r['sharpe_ratio']:.2f}  dd={r['max_drawdown_pct']:.1f}%  "
                  f"T={r['trades']}  win={r['win_rate']:.0f}%")

# G2: 突破N日高點
for lookback in [20, 30, 60]:
    for stop_cfg, stop_label in [
        ({"type":"trailing_pct","pct":0.10}, "trail10%"),
        ({"type":"atr","multiplier":2.0,"period":14}, "ATR2.0"),
    ]:
        sigs = [None]*n
        for i in range(lookback+1, n):
            prev_high = max(closes[i-lookback:i])
            if closes[i] > prev_high: sigs[i] = "buy"
            elif closes[i] < calc_sma(closes, lookback)[i] * 0.95: sigs[i] = "sell"
        r = simulate(prices, sigs, stop_cfg, {"type":"signal"})
        name = f"G-突破{lookback}日高 {stop_label}"
        scan_results.append({"name": name, "cat": "G", **r})
        if r["trades"] >= 2:
            print(f"  {name:<45} ret={r['total_return_pct']:+6.1f}%  "
                  f"sharpe={r['sharpe_ratio']:.2f}  dd={r['max_drawdown_pct']:.1f}%  "
                  f"T={r['trades']}  win={r['win_rate']:.0f}%")


# ══════════════════════════════════════════════════════════════════
# 第二步：篩選並排名
# ══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("第二步：篩選結果（trades>=3, ret>40%）")
print("="*70)

candidates = [r for r in scan_results if r["trades"] >= 3 and r["total_return_pct"] >= 40]
candidates.sort(key=lambda x: x["total_return_pct"], reverse=True)

print(f"\n符合條件的策略數: {len(candidates)}")
print(f"\n{'名稱':<50} {'報酬':>8} {'Sharpe':>7} {'回撤':>7} {'次':>4} {'勝率':>6}")
print("-"*90)
for r in candidates[:30]:
    print(f"{r['name']:<50} {r['total_return_pct']:>7.1f}% {r['sharpe_ratio']:>7.2f} "
          f"{r['max_drawdown_pct']:>6.1f}% {r['trades']:>4} {r['win_rate']:>5.0f}%")


# ══════════════════════════════════════════════════════════════════
# 第三步：年度分解分析（前三名各類）
# ══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("第三步：各類最佳策略年度分解")
print("="*70)

years = sorted(set(p["date"][:4] for p in prices))

def yearly_returns(sigs_func, stop_cfg, exit_cfg):
    """計算年度報酬"""
    results = {}
    for y in years:
        yp = [p for p in prices if p["date"].startswith(y)]
        if len(yp) < 20:
            continue
        yc = [p["close"] for p in yp]
        ysig = sigs_func(yp, yc)
        yr = simulate(yp, ysig, stop_cfg, exit_cfg)
        results[y] = yr["total_return_pct"]
    return results


# 最佳A類
best_A = [r for r in candidates if r["cat"] == "A"]
if best_A:
    print(f"\n最佳A類: {best_A[0]['name']}")

best_B = [r for r in candidates if r["cat"] in ["B","D","E"]]
if best_B:
    print(f"最佳振盪器: {best_B[0]['name']}")

best_C = [r for r in candidates if r["cat"] in ["C","F","G"]]
if best_C:
    print(f"最佳第三類: {best_C[0]['name']}")


# ══════════════════════════════════════════════════════════════════
# 第四步：選定三個策略（詳細參數）
# ══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("第四步：選定策略定義")
print("="*70)

# 根據掃描結果選出最優的三個不同邏輯類型

# ===== 策略 A：均線距離型 =====
# 選 MA60 乖離 -6%（或掃描中最佳的A類）

def signals_A_final():
    """均線距離型：MA60 乖離 < threshold 買入，回到均線賣出"""
    # 試不同threshold找最佳A
    best_ret = -999
    best_th = -0.06
    best_ma = 60
    for ma_p in [20, 30, 60]:
        sma = calc_sma(closes, ma_p)
        for th in [-0.05, -0.055, -0.06, -0.07, -0.08]:
            sigs = [None]*n
            for i in range(ma_p, n):
                if sma[i] is None or sma[i] == 0: continue
                b = (closes[i] - sma[i]) / sma[i]
                if b < th: sigs[i] = "buy"
                elif b >= 0: sigs[i] = "sell"
            r = simulate(prices, sigs, {"type":"fixed_pct","pct":0.10}, {"type":"signal"})
            if r["trades"] >= 3 and r["total_return_pct"] > best_ret:
                best_ret = r["total_return_pct"]
                best_th = th
                best_ma = ma_p
    print(f"  A類最佳: MA{best_ma} 乖離<{best_th*100:.1f}% fixed10% → ret={best_ret:.1f}%")

    sma = calc_sma(closes, best_ma)
    sigs = [None]*n
    for i in range(best_ma, n):
        if sma[i] is None or sma[i] == 0: continue
        b = (closes[i] - sma[i]) / sma[i]
        if b < best_th: sigs[i] = "buy"
        elif b >= 0:    sigs[i] = "sell"
    return sigs, best_ma, best_th

def signals_B_final():
    """振盪器型：找最佳RSI/KD/WR組合"""
    best_ret = -999
    best_cfg = None
    best_sigs = None

    # RSI
    for rsi_p in [7, 9, 14]:
        rsi_vals = calc_rsi(closes, rsi_p)
        for buy_th, sell_th in [(25,50),(25,55),(30,55),(30,60),(35,60)]:
            for m in [1.5, 2.0, 2.5]:
                sigs = [None]*n
                for i in range(rsi_p, n):
                    if rsi_vals[i] is None: continue
                    if rsi_vals[i] < buy_th:   sigs[i] = "buy"
                    elif rsi_vals[i] > sell_th: sigs[i] = "sell"
                r = simulate(prices, sigs, {"type":"atr","multiplier":m,"period":14}, {"type":"signal"})
                if r["trades"] >= 3 and r["total_return_pct"] > best_ret:
                    best_ret = r["total_return_pct"]
                    best_cfg = f"RSI({rsi_p})<{buy_th}>{sell_th} ATR×{m}"
                    best_sigs = sigs[:]
                    best_stop = {"type":"atr","multiplier":m,"period":14}

    # KD
    for kd_p in [9, 14]:
        for sm in [3, 5]:
            kv, dv = build_kd(prices, kd_p, sm)
            for d_buy in [3, 5, 10]:
                for k_sell in [50, 60, 70]:
                    for m in [1.5, 2.0]:
                        sigs = [None]*n
                        for i in range(kd_p+2, n):
                            kc = kv[i]; dc = dv[i]
                            if kc is None or dc is None: continue
                            if dc < d_buy:    sigs[i] = "buy"
                            elif kc > k_sell: sigs[i] = "sell"
                        r = simulate(prices, sigs, {"type":"atr","multiplier":m,"period":14}, {"type":"signal"})
                        if r["trades"] >= 3 and r["total_return_pct"] > best_ret:
                            best_ret = r["total_return_pct"]
                            best_cfg = f"KD({kd_p},{sm}) D<{d_buy} K>{k_sell} ATR×{m}"
                            best_sigs = sigs[:]
                            best_stop = {"type":"atr","multiplier":m,"period":14}

    # WR
    for wr_p in [14, 21]:
        wr_vals = calc_williams_r(prices, wr_p)
        for buy_th, sell_th in [(-90,-20),(-85,-15),(-80,-10),(-90,-30)]:
            for m in [1.5, 2.0]:
                sigs = [None]*n
                for i in range(wr_p, n):
                    if wr_vals[i] is None: continue
                    if wr_vals[i] < buy_th:   sigs[i] = "buy"
                    elif wr_vals[i] > sell_th: sigs[i] = "sell"
                r = simulate(prices, sigs, {"type":"atr","multiplier":m,"period":14}, {"type":"signal"})
                if r["trades"] >= 3 and r["total_return_pct"] > best_ret:
                    best_ret = r["total_return_pct"]
                    best_cfg = f"WR({wr_p})<{buy_th}>{sell_th} ATR×{m}"
                    best_sigs = sigs[:]
                    best_stop = {"type":"atr","multiplier":m,"period":14}

    print(f"  B類最佳: {best_cfg} → ret={best_ret:.1f}%")
    return best_sigs, best_cfg, best_stop

def signals_C_final():
    """第三類：找最佳非A/B類策略"""
    best_ret = -999
    best_cfg = None
    best_sigs = None
    best_stop = None
    best_exit = None

    # 布林通道
    for bb_std in [1.5, 2.0, 2.5]:
        upper, lower, mid = calc_bollinger(closes, 20, bb_std)
        for exit_t, exit_l in [
            ({"type":"signal"}, "sig"),
            ({"type":"hold_days","days":10}, "hold10"),
            ({"type":"hold_days","days":15}, "hold15"),
        ]:
            for m in [2.0, 2.5]:
                sigs = [None]*n
                for i in range(20, n):
                    if lower[i] is None or mid[i] is None: continue
                    if closes[i] < lower[i]: sigs[i] = "buy"
                    elif closes[i] > mid[i]: sigs[i] = "sell"
                stop_c = {"type":"atr","multiplier":m,"period":14}
                r = simulate(prices, sigs, stop_c, exit_t)
                if r["trades"] >= 3 and r["total_return_pct"] > best_ret:
                    best_ret = r["total_return_pct"]
                    best_cfg = f"BB(20,{bb_std}σ) {exit_l} ATR×{m}"
                    best_sigs = sigs[:]
                    best_stop = stop_c
                    best_exit = exit_t

    # 連跌型態
    for cd in [3, 4, 5]:
        for cum in [-0.05, -0.08, -0.10, -0.12]:
            for hd in [5, 10, 15]:
                for stop_c, stop_l in [
                    ({"type":"fixed_pct","pct":0.08}, "f8"),
                    ({"type":"atr","multiplier":2.0,"period":14}, "ATR2"),
                ]:
                    sigs = [None]*n
                    for i in range(cd, n):
                        down = all(closes[i-k] < closes[i-k-1] for k in range(cd))
                        cr = (closes[i] - closes[i-cd]) / closes[i-cd]
                        if down and cr < cum: sigs[i] = "buy"
                    exit_c = {"type":"hold_days","days":hd}
                    r = simulate(prices, sigs, stop_c, exit_c)
                    if r["trades"] >= 3 and r["total_return_pct"] > best_ret:
                        best_ret = r["total_return_pct"]
                        best_cfg = f"連跌{cd}天累跌{cum*100:.0f}% hold{hd} {stop_l}"
                        best_sigs = sigs[:]
                        best_stop = stop_c
                        best_exit = exit_c

    # 順勢：均線交叉
    for fast_p, slow_p in [(20,60),(10,30)]:
        fast_sma = calc_sma(closes, fast_p)
        slow_sma = calc_sma(closes, slow_p)
        for trail in [0.08, 0.10]:
            stop_c = {"type":"trailing_pct","pct":trail}
            sigs = [None]*n
            for i in range(slow_p+1, n):
                if fast_sma[i] is None or slow_sma[i] is None: continue
                if fast_sma[i-1] < slow_sma[i-1] and fast_sma[i] >= slow_sma[i]:
                    sigs[i] = "buy"
                elif fast_sma[i] < slow_sma[i]:
                    sigs[i] = "sell"
            r = simulate(prices, sigs, stop_c, {"type":"signal"})
            if r["trades"] >= 2 and r["total_return_pct"] > best_ret:
                best_ret = r["total_return_pct"]
                best_cfg = f"MA{fast_p}>{slow_p}交叉 trail{trail*100:.0f}%"
                best_sigs = sigs[:]
                best_stop = stop_c
                best_exit = {"type":"signal"}

    print(f"  C類最佳: {best_cfg} → ret={best_ret:.1f}%")
    return best_sigs, best_cfg, best_stop, best_exit


print("\n尋找各類最佳策略:")
sigs_A, ma_period_A, th_A = signals_A_final()
sigs_B, cfg_B, stop_B = signals_B_final()
sigs_C, cfg_C, stop_C, exit_C = signals_C_final()

stop_A = {"type": "fixed_pct", "pct": 0.10}
exit_A = {"type": "signal"}
exit_B = {"type": "signal"}

rA = simulate(prices, sigs_A, stop_A, exit_A)
rB = simulate(prices, sigs_B, stop_B, exit_B)
rC = simulate(prices, sigs_C, stop_C, exit_C)

print(f"\n最終選定策略：")
print(f"  A: MA{ma_period_A} 乖離<{th_A*100:.1f}% fixed10% → "
      f"ret={rA['total_return_pct']:+.1f}%  sh={rA['sharpe_ratio']:.2f}  "
      f"dd={rA['max_drawdown_pct']:.1f}%  T={rA['trades']}  win={rA['win_rate']:.0f}%")
print(f"  B: {cfg_B} → "
      f"ret={rB['total_return_pct']:+.1f}%  sh={rB['sharpe_ratio']:.2f}  "
      f"dd={rB['max_drawdown_pct']:.1f}%  T={rB['trades']}  win={rB['win_rate']:.0f}%")
print(f"  C: {cfg_C} → "
      f"ret={rC['total_return_pct']:+.1f}%  sh={rC['sharpe_ratio']:.2f}  "
      f"dd={rC['max_drawdown_pct']:.1f}%  T={rC['trades']}  win={rC['win_rate']:.0f}%")


# ══════════════════════════════════════════════════════════════════
# 第五步：年度分解
# ══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("第五步：三策略年度分解")
print("="*70)

CONFIGS = {
    f"A-MA{ma_period_A} 乖離<{th_A*100:.1f}%": {
        "sigs": sigs_A,
        "stop": stop_A,
        "exit": exit_A,
    },
    f"B-{cfg_B}": {
        "sigs": sigs_B,
        "stop": stop_B,
        "exit": exit_B,
    },
    f"C-{cfg_C}": {
        "sigs": sigs_C,
        "stop": stop_C,
        "exit": exit_C,
    },
}

print(f"\n{'年份':<6}", end="")
for name in CONFIGS:
    short = name[:20]
    print(f"  {short:>20}", end="")
print()
print("-"*80)

year_data = {}
for name in CONFIGS:
    year_data[name] = {}

for y in years:
    yp = [p for p in prices if p["date"].startswith(y)]
    if len(yp) < 20: continue
    print(f"{y:<6}", end="")
    for name, cfg in CONFIGS.items():
        yc = [p["close"] for p in yp]
        ysig = [None]*len(yp)

        # 重新計算年度信號
        if "乖離" in name or "MA" in name and "乖" in name:
            ys = calc_sma(yc, ma_period_A)
            for i in range(ma_period_A, len(yp)):
                if ys[i] is None or ys[i]==0: continue
                b=(yc[i]-ys[i])/ys[i]
                if b<th_A: ysig[i]="buy"
                elif b>=0: ysig[i]="sell"
        elif "RSI" in name:
            parts = name.split("RSI(")[1].split(")")[0]
            rp = int(parts)
            bth = int(name.split("<")[1].split(">")[0])
            sth = int(name.split(">")[1].split(" ")[0])
            yr_ = calc_rsi(yc, rp)
            for i in range(rp, len(yp)):
                if yr_[i] is None: continue
                if yr_[i] < bth:   ysig[i] = "buy"
                elif yr_[i] > sth: ysig[i] = "sell"
        elif "KD" in name:
            # parse KD params
            try:
                kd_part = name.split("KD(")[1].split(")")[0]
                kp_, sm_ = [int(x) for x in kd_part.split(",")]
                db_ = int(name.split("D<")[1].split(" ")[0])
                ks_ = int(name.split("K>")[1].split(" ")[0])
            except:
                kp_, sm_, db_, ks_ = 14, 3, 5, 60
            yk, yd = build_kd(yp, kp_, sm_)
            for i in range(kp_+2, len(yp)):
                kc=yk[i]; dc=yd[i]
                if kc is None or dc is None: continue
                if dc < db_: ysig[i] = "buy"
                elif kc > ks_: ysig[i] = "sell"
        elif "WR" in name:
            try:
                wr_p_ = int(name.split("WR(")[1].split(")")[0])
                bth_ = int(name.split("WR(")[1].split("<")[1].split(">")[0])
                sth_ = int(name.split(">")[1].split(" ")[0])
            except:
                wr_p_, bth_, sth_ = 14, -90, -20
            yw = calc_williams_r(yp, wr_p_)
            for i in range(wr_p_, len(yp)):
                if yw[i] is None: continue
                if yw[i] < bth_:  ysig[i] = "buy"
                elif yw[i] > sth_: ysig[i] = "sell"
        elif "BB" in name:
            try:
                bb_s = float(name.split("BB(20,")[1].split("σ")[0])
            except:
                bb_s = 2.0
            _, yl, ym = calc_bollinger(yc, 20, bb_s)
            for i in range(20, len(yp)):
                if yl[i] is None or ym[i] is None: continue
                if yc[i] < yl[i]: ysig[i] = "buy"
                elif yc[i] > ym[i]: ysig[i] = "sell"
        elif "連跌" in name:
            try:
                cd_ = int(name.split("連跌")[1].split("天")[0])
                cm_ = float(name.split("跌")[2].split("%")[0]) / 100
                hd_ = int(name.split("hold")[1].split(" ")[0])
            except:
                cd_, cm_, hd_ = 3, -0.08, 10
            for i in range(cd_, len(yp)):
                down_ = all(yc[i-k] < yc[i-k-1] for k in range(cd_))
                cr_ = (yc[i] - yc[i-cd_]) / yc[i-cd_]
                if down_ and cr_ < cm_: ysig[i] = "buy"
            exit_cfg_y = {"type":"hold_days","days":hd_}
        elif "交叉" in name:
            try:
                fp_ = int(name.split("MA")[1].split(">")[0])
                sp_ = int(name.split(">")[1].split("交")[0])
            except:
                fp_, sp_ = 20, 60
            yf = calc_sma(yc, fp_)
            ys2 = calc_sma(yc, sp_)
            for i in range(sp_+1, len(yp)):
                if yf[i] is None or ys2[i] is None: continue
                if yf[i-1] < ys2[i-1] and yf[i] >= ys2[i]: ysig[i] = "buy"
                elif yf[i] < ys2[i]: ysig[i] = "sell"

        yr2 = simulate(yp, ysig, cfg["stop"], cfg["exit"])
        year_data[name][y] = yr2["total_return_pct"]
        print(f"  {yr2['total_return_pct']:>+8.1f}%           ", end="")
    print()


# ══════════════════════════════════════════════════════════════════
# 第六步：混合模擬
# ══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("第六步：三策略混合模擬")
print("="*70)

def simulate_curve(sigs_raw, stop_cfg, exit_cfg, initial_cap):
    """輕量版模擬，回傳每日資金曲線"""
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


def blend_equal_capital(wA=1/3, wB=1/3, wC=1/3):
    cap_A = INITIAL_CAPITAL * wA
    cap_B = INITIAL_CAPITAL * wB
    cap_C = INITIAL_CAPITAL * wC

    names = list(CONFIGS.keys())
    cfgs  = list(CONFIGS.values())
    caps  = [cap_A, cap_B, cap_C]

    curves = []
    for i, (name, cfg) in enumerate(zip(names, cfgs)):
        eq, _ = simulate_curve(cfg["sigs"], cfg["stop"], cfg["exit"], caps[i])
        curves.append(eq)

    combined = [curves[0][i] + curves[1][i] + curves[2][i] for i in range(n)]

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
    r, curve = blend_equal_capital()
    print(f"\n混合報酬: {r['total_return_pct']:+.2f}%")
    print(f"混合 Sharpe: {r['sharpe_ratio']:.3f}")
    print(f"混合最大回撤: {r['max_drawdown_pct']:.2f}%")
    print(f"最終資金: {r['final_capital']:,}")

    print("\n年度混合表現:")
    years_list = sorted(set(p["date"][:4] for p in prices))
    for y in years_list:
        idx_start = next((i for i,p in enumerate(prices) if p["date"].startswith(y)), None)
        idx_end   = max((i for i,p in enumerate(prices) if p["date"].startswith(y)), default=None)
        if idx_start is None: continue
        eq_start = curve[idx_start-1] if idx_start > 0 else INITIAL_CAPITAL
        eq_end   = curve[idx_end]
        yret = (eq_end - eq_start) / eq_start * 100
        total_ret_so_far = (eq_end - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        print(f"  {y}: {yret:+6.1f}%  累積={(total_ret_so_far):+.1f}%")

    print("\n對比摘要:")
    print(f"{'策略':<40} {'報酬':>8} {'Sharpe':>7} {'回撤':>7}")
    print("-"*65)
    for name, cfg in CONFIGS.items():
        ir = simulate(prices, cfg["sigs"], cfg["stop"], cfg["exit"])
        ok = " ✓" if ir['total_return_pct']>=50 and ir['max_drawdown_pct']<=25 else ""
        print(f"{name:<40} {ir['total_return_pct']:>7.1f}% {ir['sharpe_ratio']:>7.2f} {ir['max_drawdown_pct']:>6.1f}%{ok}")
    print(f"{'三策略混合 (A+B+C)':40} {r['total_return_pct']:>7.1f}% {r['sharpe_ratio']:>7.3f} {r['max_drawdown_pct']:>6.1f}%")

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

    return r


print()
blend_r = run_blend()

print("\n" + "="*70)
print("研究完成")
print("="*70)
