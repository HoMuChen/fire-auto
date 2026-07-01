"""
台積電網格 — 多變體比較，目標：最大化「現金流穩定度」（無槓桿）。

可配置：
  n        批數（網格深度）
  step     每跌多少 % 加一批（買進間距，決定覆蓋範圍 = n*step）
  take     每批漲多少 % 賣出（鎖利，現金流來源；固定 1%）
  sizing   'equal' 或 ('mart', m) 馬丁格爾：第 k 批資金 ∝ m^k（往下加碼放大）
  base_k   5MA 空手回補時一次買幾批底倉
規則同前：賣出鎖利、跌破 last_buy*(1-step) 加批、滿批放著等、空手站上5MA買 base_k 批。
無槓桿。成本含手續費+稅。
"""
import sys, statistics
from collections import defaultdict
sys.path.insert(0, "/Users/mu/fire-auto")
import backtest as bt

BF, SF, ST = bt.BUY_FEE, bt.SELL_FEE, bt.SELL_TAX
INITIAL = 1_000_000


def weights(n, sizing):
    if sizing == "equal":
        raw = [1.0] * n
    else:
        m = sizing[1]
        raw = [m ** i for i in range(n)]
    s = sum(raw)
    return [INITIAL * r / s for r in raw]


def run(cfg, sid="2330"):
    prices = bt.read_prices(sid)
    closes = [p["close"] for p in prices]
    ma5 = bt.calc_sma(closes, 5)
    n, step, take, base_k = cfg["n"], cfg["step"], cfg["take"], cfg["base_k"]
    w = weights(n, cfg["sizing"])
    cash = INITIAL
    lots = []                 # {price, shares, invested}
    last_buy = None
    curve = []
    monthly = defaultdict(float)

    def buy(price):
        nonlocal cash, last_buy
        if len(lots) >= n:
            return
        inv = w[len(lots)]
        if cash < inv * 0.999:
            return
        shares = inv * (1 - BF) / price
        cash -= inv
        lots.append({"price": price, "shares": shares, "invested": inv})
        last_buy = price

    for i, p in enumerate(prices):
        c = p["close"]; m = ma5[i]; ym = p["date"][:7]
        remain = []
        for lot in lots:
            if c >= lot["price"] * (1 + take):
                proceeds = lot["shares"] * c * (1 - SF - ST)
                cash += proceeds
                monthly[ym] += proceeds - lot["invested"]
            else:
                remain.append(lot)
        lots[:] = remain
        if len(lots) == 0:
            if m is not None and c > m:
                for _ in range(base_k):
                    buy(c)
        elif last_buy is not None and c <= last_buy * (1 - step) and len(lots) < n:
            buy(c)
        curve.append(cash + sum(l["shares"] * c for l in lots))
    return prices, curve, monthly


def maxdd(curve):
    peak = curve[0]; dd = 0.0
    for e in curve:
        peak = max(peak, e); dd = min(dd, e/peak - 1)
    return dd * 100


def months_in(prices):
    seen, s = [], set()
    for p in prices:
        ym = p["date"][:7]
        if ym not in s:
            s.add(ym); seen.append(ym)
    return seen


def evaluate(name, cfg):
    prices, curve, monthly = run(cfg)
    n = len(prices)
    months = months_in(prices)
    vals = [monthly.get(mm, 0.0) / INITIAL * 100 for mm in months]
    pos = sum(1 for v in vals if v > 0.05)
    # 最長乾旱
    dry = mx = 0
    for v in vals:
        dry = dry + 1 if v < 0.3 else 0
        mx = max(mx, dry)
    tot = (curve[-1]/INITIAL - 1) * 100
    cagr = ((curve[-1]/INITIAL) ** (252/n) - 1) * 100
    cover = cfg["n"] * cfg["step"] * 100
    return dict(name=name, tot=tot, cagr=cagr, dd=maxdd(curve),
                avg=statistics.mean(vals), med=statistics.median(vals),
                pospct=pos/len(months)*100, drought=mx, cover=cover)


def main():
    configs = {
        "①基準 7批/1%":      dict(n=7,  step=0.01, take=0.01, sizing="equal",        base_k=3),
        "②密鋪 14批/1%":     dict(n=14, step=0.01, take=0.01, sizing="equal",        base_k=3),
        "③寬鋪 10批/2%":     dict(n=10, step=0.02, take=0.01, sizing="equal",        base_k=3),
        "④馬丁 7批/1%x1.6":  dict(n=7,  step=0.01, take=0.01, sizing=("mart", 1.6),  base_k=3),
        "⑤馬丁寬 12批/2%x1.4": dict(n=12, step=0.02, take=0.01, sizing=("mart", 1.4), base_k=3),
        "⑥深鋪 20批/2%":     dict(n=20, step=0.02, take=0.01, sizing="equal",        base_k=3),
    }
    print("台積電網格變體 — 現金流比較（無槓桿，2020-2026.6，共78個月）\n")
    print(f"{'變體':<20}{'覆蓋%':>6}{'總報酬':>8}{'CAGR':>7}{'最大DD':>8}{'月均CF':>7}{'月中位':>7}{'有CF月%':>8}{'最長乾旱':>8}")
    print("-" * 86)
    for name, cfg in configs.items():
        r = evaluate(name, cfg)
        print(f"{r['name']:<20}{r['cover']:>5.0f}%{r['tot']:>+7.0f}%{r['cagr']:>+6.1f}%{r['dd']:>+7.1f}%"
              f"{r['avg']:>+6.2f}%{r['med']:>+6.2f}%{r['pospct']:>7.0f}%{r['drought']:>7}月")


if __name__ == "__main__":
    main()
