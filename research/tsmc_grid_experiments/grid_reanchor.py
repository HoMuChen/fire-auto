"""
再掛單網格（re-anchor）：買進參考=近 H 日高點，跌破 -step 就在「當前區間」掛單買，
同價位不重複；每批漲 +take 賣。舊的高檔套牢批擺著不動(不認賠)，不擋新掛單(靠 n 夠大 + 現金)。
→ 目標：在 2023 那種「低區間震盪」也能收租，填掉乾旱。無槓桿。
"""
import sys, statistics
from collections import defaultdict
sys.path.insert(0, "/Users/mu/fire-auto")
import backtest as bt

# 券商手續費打 28 折（0.1425% × 0.28 ≈ 0.04%/邊），證交稅 0.3% 不打折 → 來回約 0.38%
DISCOUNT = 0.28
BF = 0.001425 * DISCOUNT
SF = 0.001425 * DISCOUNT
ST = 0.003
OWN = 1_000_000


def run(n, step, take, H, lot_frac):
    prices = bt.read_prices("2330")
    closes = [p["close"] for p in prices]
    cash = OWN; lots = []
    curve = []; monthly = defaultdict(float)
    lot_cash = OWN * lot_frac          # 每批固定金額

    for i, p in enumerate(prices):
        c = p["close"]; ym = p["date"][:7]
        ref = max(closes[max(0, i-H):i+1])       # 近 H 日高點
        # 賣：每批 +take
        rem = []
        for lot in lots:
            if c >= lot["price"] * (1 + take):
                proceeds = lot["shares"] * c * (1 - SF - ST)
                cash += proceeds; monthly[ym] += proceeds - lot["invested"]
            else:
                rem.append(lot)
        lots[:] = rem
        # 買：價格跌破近高 -step，且當前價位附近無持倉(間距>=step)，且倉數/現金夠
        if c <= ref * (1 - step) and len(lots) < n and cash >= lot_cash * 0.999:
            near = any(abs(l["price"]/c - 1) < step for l in lots)
            if not near:
                cash -= lot_cash
                lots.append({"price": c, "shares": lot_cash*(1-BF)/c, "invested": lot_cash})
        curve.append(cash + sum(l["shares"]*c for l in lots))
    return prices, curve, monthly


def maxdd(cv):
    peak = cv[0]; dd = 0.0
    for e in cv:
        peak = max(peak, e); dd = min(dd, e/peak-1)
    return dd*100


def months_of(prices):
    s, seen = set(), []
    for p in prices:
        ym = p["date"][:7]
        if ym not in s: s.add(ym); seen.append(ym)
    return seen


def report(name, n, step, take, H, lot_frac):
    prices, cv, mo = run(n, step, take, H, lot_frac)
    ndays = len(prices); ms = months_of(prices)
    vals = [mo.get(mm, 0.0)/OWN*100 for mm in ms]
    pos = sum(1 for v in vals if v > 0.05)
    dry = mx = 0
    for v in vals:
        dry = dry+1 if v < 0.3 else 0; mx = max(mx, dry)
    tot = (cv[-1]/OWN-1)*100
    by_year = defaultdict(float)
    for mm in ms: by_year[mm[:4]] += mo.get(mm, 0.0)/OWN*100
    print(f"{name:<26}{tot:>+7.0f}%{maxdd(cv):>+7.1f}%{statistics.median(vals):>+6.2f}%"
          f"{pos/len(ms)*100:>7.0f}%{mx:>6}月{by_year['2023']:>+8.1f}%{by_year['2021']:>+8.1f}%")
    return by_year


def main():
    print("再掛單網格(跟隨近高) — 重點:2023/2021 乾旱有沒有填（無槓桿，78月）\n")
    print(f"{'版本':<26}{'總報酬':>8}{'最大DD':>8}{'月中位':>7}{'有CF月%':>8}{'乾旱':>6}{'2023':>8}{'2021':>8}")
    print("-"*80)
    report("再掛單 12批/1%/H15/8%",  12, 0.01, 0.01, 15, 0.08)
    report("再掛單 15批/1%/H10/6%",  15, 0.01, 0.01, 10, 0.06)
    report("再掛單 15批/1.5%/H15/6%",15, 0.015,0.01, 15, 0.06)
    by = report("再掛單 20批/1%/H10/5%",  20, 0.01, 0.01, 10, 0.05)
    print("\n【再掛單 20批/1%/H10/5%】逐年現金流：")
    for y in sorted(by): print(f"  {y}: {by[y]:+6.1f}%")


if __name__ == "__main__":
    main()
