"""
台積電(2330) 再掛單網格收租策略 — 定案版回測。

目標：穩定月現金流（非打敗大盤）。無槓桿，跌深抱著（績優股）。
規則見 strategies/tsmc_grid_income.md。

用法：
  python3 research/tsmc_grid_income.py            # 摘要 + 逐年
  python3 research/tsmc_grid_income.py monthly     # 逐月明細（買/賣/持倉/現金流）
"""
import sys
from collections import defaultdict
sys.path.insert(0, "/Users/mu/fire-auto")
import backtest as bt

# ── 參數 ──
STOCK = "2330"
OWN = 1_000_000
N = 20                       # 最多批數
LOT = 0.05 * OWN             # 每批金額（5 萬）
STEP = 0.01                  # 格距 1%（買進門檻 = 近H日高 × (1-STEP)）
TAKE = 0.01                  # 每批 +1% 賣出
H = 10                       # 近期高點視窗（交易日）
# 成本：手續費 0.1425% 打 28 折，證交稅 0.3%（賣）→ 來回約 0.38%
FEE_DISCOUNT = 0.28
BF = 0.001425 * FEE_DISCOUNT
SF = 0.001425 * FEE_DISCOUNT
ST = 0.003


def run():
    prices = bt.read_prices(STOCK)
    closes = [p["close"] for p in prices]
    cash = OWN
    lots = []                # {price, shares, invested}
    curve = []
    mbuy = defaultdict(int); msell = defaultdict(int); mcf = defaultdict(float)
    end_lots = {}

    for i, p in enumerate(prices):
        c = p["close"]; ym = p["date"][:7]
        ref = max(closes[max(0, i - H):i + 1])          # 近 H 日最高收盤
        # 賣：逐批漲到買價 +TAKE 就賣
        rem = []
        for lot in lots:
            if c >= lot["price"] * (1 + TAKE):
                proceeds = lot["shares"] * c * (1 - SF - ST)
                cash += proceeds
                mcf[ym] += proceeds - lot["invested"]; msell[ym] += 1
            else:
                rem.append(lot)
        lots[:] = rem
        # 買：跌破近高 -STEP、當前價位±STEP 內無持倉、批數/現金夠
        if c <= ref * (1 - STEP) and len(lots) < N and cash >= LOT * 0.999:
            if not any(abs(l["price"] / c - 1) < STEP for l in lots):
                cash -= LOT
                lots.append({"price": c, "shares": LOT * (1 - BF) / c, "invested": LOT})
                mbuy[ym] += 1
        curve.append(cash + sum(l["shares"] * c for l in lots))
        end_lots[ym] = len(lots)
    return prices, curve, mbuy, msell, mcf, end_lots


def months_of(prices):
    seen, s = [], set()
    for p in prices:
        ym = p["date"][:7]
        if ym not in s:
            s.add(ym); seen.append(ym)
    return seen


def maxdd(curve):
    peak = curve[0]; dd = 0.0
    for e in curve:
        peak = max(peak, e); dd = min(dd, e / peak - 1)
    return dd * 100


def summary():
    prices, curve, mbuy, msell, mcf, end_lots = run()
    ms = months_of(prices); nd = len(prices)
    vals = [mcf.get(m, 0.0) / OWN * 100 for m in ms]
    import statistics
    pos = sum(1 for v in vals if v > 0.05)
    dry = mx = 0
    for v in vals:
        dry = dry + 1 if v < 0.3 else 0; mx = max(mx, dry)
    tot = (curve[-1] / OWN - 1) * 100
    cagr = ((curve[-1] / OWN) ** (252 / nd) - 1) * 100
    print(f"台積電 再掛單網格收租（{prices[0]['date']} ~ {prices[-1]['date']}，{len(ms)}月，無槓桿）")
    print(f"  參數：{N}批 × {LOT:,.0f}／格{STEP:.0%}／賣+{TAKE:.0%}／近{H}日高／來回成本{ (BF+SF+ST)*100:.2f}%")
    print(f"  總報酬 {tot:+.0f}%   CAGR {cagr:+.1f}%   最大回撤 {maxdd(curve):.1f}%")
    print(f"  月現金流：中位 {statistics.median(vals):+.2f}%  平均 {statistics.mean(vals):+.2f}%  "
          f"有現金流月 {pos}/{len(ms)}={pos/len(ms)*100:.0f}%  最長乾旱 {mx}月")
    by_year = defaultdict(float)
    for m in ms:
        by_year[m[:4]] += mcf.get(m, 0.0) / OWN * 100
    print("  逐年現金流：" + "  ".join(f"{y} {by_year[y]:+.1f}%" for y in sorted(by_year)))
    tb, ts = sum(mbuy.values()), sum(msell.values())
    print(f"  交易：買 {tb} 筆 / 賣 {ts} 筆（月均 買{tb/len(ms):.1f}/賣{ts/len(ms):.1f}）")


def monthly():
    prices, curve, mbuy, msell, mcf, end_lots = run()
    print(f"{'月份':<9}{'買':>4}{'賣':>4}{'月底持倉':>8}{'現金流%':>10}")
    print("-" * 38)
    cur = None; ytot = 0.0
    for ym in months_of(prices):
        if cur and ym[:4] != cur:
            print(f"  {cur} 全年{'':>16}{ytot:>+9.1f}%"); print("-" * 38); ytot = 0.0
        cur = ym[:4]
        cf = mcf.get(ym, 0.0) / OWN * 100; ytot += cf
        print(f"{ym:<9}{mbuy.get(ym,0):>4}{msell.get(ym,0):>4}{end_lots[ym]:>8}{cf:>+9.2f}%")
    print(f"  {cur} 全年{'':>16}{ytot:>+9.1f}%")


if __name__ == "__main__":
    (monthly if len(sys.argv) > 1 and sys.argv[1] == "monthly" else summary)()
