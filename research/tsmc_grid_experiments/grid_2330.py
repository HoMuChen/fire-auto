"""
台積電(2330) 網格策略回測。

規則（使用者指定）：
- 資金分 7 批（等金額，每批 = 初始資金/7）
- 網格買：每跌 1%（相對「上一次買價」）買 1 批，最多 7 批
- 網格賣：每批漲到自己買價 +1% 就賣（鎖 1% 價差），釋放該批資金
- 沒錢了（7 批用完）：只賣不買，放著等回升
- 賣飛了（持倉歸零）+ 收盤 > 5MA → 立刻買 3 批（底倉/起手/回補）
- 底倉也當一般網格批（漲 +1% 照賣）
- 不停損（網格往下抱著等）
成本：買賣手續費各 0.1425%，賣方證交稅 0.3%。
對照：buy & hold 2330。
"""
import sys
sys.path.insert(0, "/Users/mu/fire-auto")
import backtest as bt

BUY_FEE, SELL_FEE, SELL_TAX = bt.BUY_FEE, bt.SELL_FEE, bt.SELL_TAX
INITIAL = 1_000_000
N_TRANCHES = 7
GRID = 0.01      # 1%
BASE_LOTS = 3    # 5MA 底倉批數


def run():
    prices = bt.read_prices("2330")
    closes = [p["close"] for p in prices]
    ma5 = bt.calc_sma(closes, 5)
    tranche = INITIAL / N_TRANCHES

    cash = INITIAL
    lots = []                 # [{"price":買價, "shares":股數}]
    last_buy = None           # 上一次買價（網格買的參考）
    equity_curve = []
    round_trips = []          # 每筆賣出的淨報酬%
    buys = sells = base_buys = 0

    def buy_one(price, n=1):
        nonlocal cash, last_buy, buys, base_buys
        for _ in range(n):
            if len(lots) >= N_TRANCHES:
                break
            shares = tranche * (1 - BUY_FEE) / price
            cash -= tranche
            lots.append({"price": price, "shares": shares})
            buys += 1
        last_buy = price

    for i, p in enumerate(prices):
        c = p["close"]
        m = ma5[i]

        # 1) 賣：每批漲到買價+1% 就賣
        remain = []
        for lot in lots:
            if c >= lot["price"] * (1 + GRID):
                proceeds = lot["shares"] * c * (1 - SELL_FEE - SELL_TAX)
                cost = tranche
                cash += proceeds
                round_trips.append((proceeds / cost - 1) * 100)
            else:
                remain.append(lot)
        lots[:] = remain

        # 2) 買
        if len(lots) == 0:
            # 空手：站上 5MA 才建 3 批底倉（起手/回補）
            if m is not None and c > m and cash >= BASE_LOTS * tranche * 0.999:
                buy_one(c, BASE_LOTS)
                base_buys += 1
        else:
            # 有倉：每跌 1% 加 1 批，最多 7 批
            if last_buy is not None and c <= last_buy * (1 - GRID) and len(lots) < N_TRANCHES:
                buy_one(c, 1)

        equity = cash + sum(l["shares"] * c for l in lots)
        equity_curve.append(equity)

    return prices, closes, equity_curve, round_trips, dict(buys=buys, sells=len(round_trips), base_buys=base_buys, final_lots=len(lots), final_cash=cash)


def maxdd(curve):
    peak = curve[0]; dd = 0.0
    for e in curve:
        peak = max(peak, e)
        dd = min(dd, e / peak - 1)
    return dd * 100


def stats(label, start_eq, end_eq, curve, n_days):
    ret = (end_eq / start_eq - 1) * 100
    yrs = n_days / 252
    cagr = ((end_eq / start_eq) ** (1 / yrs) - 1) * 100
    return f"{label:<14} 總報酬 {ret:>+8.1f}%  CAGR {cagr:>+6.1f}%  最大回撤 {maxdd(curve):>6.1f}%"


def main():
    prices, closes, curve, rts, info = run()
    n = len(prices)
    print(f"標的 2330  期間 {prices[0]['date']} ~ {prices[-1]['date']}  ({n} 日)\n")

    # 網格策略
    print(stats("網格策略", INITIAL, curve[-1], curve, n))
    print(f"{'':14} 期末權益 {curve[-1]:,.0f}（現金 {info['final_cash']:,.0f} + 持倉 {info['final_lots']} 批）")
    print(f"{'':14} 買進 {info['buys']} 次（其中 5MA 底倉建倉 {info['base_buys']} 回）, 賣出 {info['sells']} 批")
    if rts:
        wins = [r for r in rts if r > 0]
        print(f"{'':14} 每批賣出淨報酬：均 {sum(rts)/len(rts):+.2f}%  勝率 {len(wins)/len(rts)*100:.0f}%（{len(rts)} 筆）")
    print()

    # buy & hold 對照
    bh_curve = [INITIAL * (c / closes[0]) for c in closes]
    # B&H 計一次買進成本與最後賣出成本
    bh_end = INITIAL * (closes[-1] / closes[0]) * (1 - BUY_FEE) * (1 - SELL_FEE - SELL_TAX)
    print(stats("Buy & Hold", INITIAL, bh_curve[-1], bh_curve, n))
    print(f"{'':14} （含一次買賣成本後期末約 {bh_end:,.0f}）")
    print()

    # 分年報酬（網格 vs B&H）
    print("分年報酬（網格 / B&H）：")
    from collections import OrderedDict
    year_idx = OrderedDict()
    for i, p in enumerate(prices):
        year_idx.setdefault(p["date"][:4], []).append(i)
    for y, idxs in year_idx.items():
        g0, g1 = curve[idxs[0]], curve[idxs[-1]]
        b0, b1 = bh_curve[idxs[0]], bh_curve[idxs[-1]]
        print(f"  {y}: 網格 {(g1/g0-1)*100:>+7.1f}%   B&H {(b1/b0-1)*100:>+7.1f}%")


if __name__ == "__main__":
    main()
