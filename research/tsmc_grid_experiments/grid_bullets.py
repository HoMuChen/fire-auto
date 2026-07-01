"""
台積電：收租淺網格 + 深度回調子彈加碼（深層可開槓桿）。

雙 sleeve（共用一個現金帳，cash 可為負=借款，上限 max_borrow）：
  A 收租網格：用 income_alloc 資金跑 1% 網格（買-1%/賣+1%），負責日常小現金流。
  B 子彈：保留現金，當「距一年高點」跌破 [-15,-22,-29,-36]% 分批加碼；
          子彈倉漲 take_B 賣出（吃反彈、填乾旱）。深層可動用借款(槓桿)。

風控檢查：權益 equity=cash+持倉市值；equity<=0 視為爆倉；記錄最高總曝險/權益(槓桿)、最低權益。
成本含手續費+稅。peak=過去252日最高收盤。
"""
import sys, statistics
from collections import defaultdict
sys.path.insert(0, "/Users/mu/fire-auto")
import backtest as bt

BF, SF, ST = bt.BUY_FEE, bt.SELL_FEE, bt.SELL_TAX
OWN = 1_000_000


def run(cfg):
    prices = bt.read_prices("2330")
    closes = [p["close"] for p in prices]
    ma5 = bt.calc_sma(closes, 5)
    n = len(prices)

    income_alloc = cfg["income_alloc"]
    inc_n, step, take = cfg["inc_n"], 0.01, 0.01
    bullet_levels = cfg["bullet_levels"]      # [(dd門檻, 資金)]
    take_B = cfg["take_B"]
    max_borrow = cfg["lev_extra"] * OWN       # 可借上限
    recycle = cfg.get("bullet_recycle", False)  # True=子彈純網格(賣出即解鎖、可再買回)

    cash = OWN
    inc_lots = []          # {price, shares, invested}
    inc_invested = 0.0
    bull_lots = []
    bullets_fired = set()
    last_buy = None
    curve = []
    monthly = defaultdict(float)
    peak_lev = 0.0
    peak_lev_info = None
    min_equity = OWN
    min_eq_info = None
    blown = False
    min_cash = OWN

    def can_spend(amt):
        return cash - amt >= -max_borrow - 1e-6

    for i, p in enumerate(prices):
        c = p["close"]; m = ma5[i]; ym = p["date"][:7]
        peak = max(closes[max(0, i-252):i+1])
        dd = c / peak - 1

        # 賣出（收租網格 +1%）
        rem = []
        for lot in inc_lots:
            if c >= lot["price"] * (1 + take):
                proceeds = lot["shares"] * c * (1 - SF - ST)
                cash += proceeds; inc_invested -= lot["invested"]
                monthly[ym] += proceeds - lot["invested"]
            else:
                rem.append(lot)
        inc_lots[:] = rem
        # 賣出（子彈 +take_B，吃反彈）
        rem = []
        for lot in bull_lots:
            if c >= lot["price"] * (1 + take_B):
                proceeds = lot["shares"] * c * (1 - SF - ST)
                cash += proceeds
                monthly[ym] += proceeds - lot["invested"]
                if recycle:
                    bullets_fired.discard(lot["level"])   # 解鎖該格 → 再跌回可再買
            else:
                rem.append(lot)
        bull_lots[:] = rem

        # 收租網格買進
        tr = income_alloc / inc_n
        if len(inc_lots) == 0:
            if m is not None and c > m and inc_invested + 3*tr <= income_alloc + 1 and can_spend(3*tr):
                for _ in range(3):
                    if inc_invested + tr <= income_alloc + 1 and can_spend(tr):
                        sh = tr*(1-BF)/c; cash -= tr; inc_invested += tr
                        inc_lots.append({"price": c, "shares": sh, "invested": tr})
                last_buy = c
        elif last_buy is not None and c <= last_buy*(1-step) and len(inc_lots) < inc_n:
            if inc_invested + tr <= income_alloc + 1 and can_spend(tr):
                sh = tr*(1-BF)/c; cash -= tr; inc_invested += tr
                inc_lots.append({"price": c, "shares": sh, "invested": tr}); last_buy = c

        # 子彈：距一年高點跌破門檻分批加碼
        for k, (thr, size) in enumerate(bullet_levels):
            if k not in bullets_fired and dd <= thr and can_spend(size):
                sh = size*(1-BF)/c; cash -= size
                bull_lots.append({"price": c, "shares": sh, "invested": size, "level": k})
                bullets_fired.add(k)
        # 回到高點附近重置子彈（可再次使用）
        if dd > -0.05:
            bullets_fired.clear()

        holdings = sum(l["shares"]*c for l in inc_lots) + sum(l["shares"]*c for l in bull_lots)
        equity = cash + holdings
        if equity > 0 and holdings > 0 and holdings/equity > peak_lev:
            peak_lev = holdings/equity
            peak_lev_info = (p["date"], dd*100)
        if equity < min_equity:
            min_equity = equity; min_eq_info = (p["date"], dd*100)
        min_cash = min(min_cash, cash)
        if equity <= 0:
            blown = True
        curve.append(equity)

    return prices, closes, curve, monthly, dict(peak_lev=peak_lev, peak_lev_info=peak_lev_info,
                                                min_equity=min_equity, min_eq_info=min_eq_info,
                                                min_cash=min_cash, blown=blown)


def maxdd(curve):
    peak = curve[0]; dd = 0.0
    for e in curve:
        peak = max(peak, e); dd = min(dd, e/peak - 1)
    return dd*100


def months_in(prices):
    seen, s = [], set()
    for p in prices:
        ym = p["date"][:7]
        if ym not in s: s.add(ym); seen.append(ym)
    return seen


def evaluate(name, cfg):
    prices, closes, curve, monthly, risk = run(cfg)
    n = len(prices); months = months_in(prices)
    vals = [monthly.get(mm, 0.0)/OWN*100 for mm in months]
    pos = sum(1 for v in vals if v > 0.05)
    dry = mx = 0
    for v in vals:
        dry = dry+1 if v < 0.3 else 0; mx = max(mx, dry)
    tot = (curve[-1]/OWN-1)*100
    cagr = ((curve[-1]/OWN)**(252/n)-1)*100
    return dict(name=name, tot=tot, cagr=cagr, dd=maxdd(curve), avg=statistics.mean(vals),
                med=statistics.median(vals), pospct=pos/len(months)*100, drought=mx,
                lev=risk["peak_lev"], mineq=risk["min_equity"], blown=risk["blown"],
                lev_info=risk["peak_lev_info"], min_cash=risk["min_cash"])


def main():
    cfgs = {
        "②純密鋪14批(對照)": dict(income_alloc=1_000_000, inc_n=14, take_B=0.01,
                              bullet_levels=[], lev_extra=0.0),
        "子彈·無槓桿":       dict(income_alloc=450_000, inc_n=6, take_B=0.08,
                              bullet_levels=[(-0.15,140_000),(-0.22,150_000),(-0.26,150_000),(-0.32,110_000)],
                              lev_extra=0.0),
        "子彈·一次性(選定)":   dict(income_alloc=450_000, inc_n=6, take_B=0.08,
                              bullet_levels=[(-0.15,140_000),(-0.22,160_000),(-0.26,250_000),(-0.32,300_000),(-0.40,300_000)],
                              lev_extra=0.6, bullet_recycle=False),
        "子彈·純網格 take8%":  dict(income_alloc=450_000, inc_n=6, take_B=0.08,
                              bullet_levels=[(-0.15,140_000),(-0.22,160_000),(-0.26,250_000),(-0.32,300_000),(-0.40,300_000)],
                              lev_extra=0.6, bullet_recycle=True),
        "子彈·純網格 take3%":  dict(income_alloc=450_000, inc_n=6, take_B=0.03,
                              bullet_levels=[(-0.15,140_000),(-0.22,160_000),(-0.26,250_000),(-0.32,300_000),(-0.40,300_000)],
                              lev_extra=0.6, bullet_recycle=True),
    }
    print("台積電 收租網格+子彈加碼 比較（自有資金100萬，2020-2026.6，78月）\n")
    print(f"{'策略':<18}{'總報酬':>8}{'CAGR':>7}{'最大DD':>8}{'月均CF':>7}{'月中位':>7}{'有CF月%':>8}{'最長乾旱':>8}{'峰值槓桿':>8}{'最低權益':>10}")
    print("-"*100)
    for name, cfg in cfgs.items():
        r = evaluate(name, cfg)
        warn = "  ⚠爆倉" if r["blown"] else ""
        print(f"{r['name']:<18}{r['tot']:>+7.0f}%{r['cagr']:>+6.1f}%{r['dd']:>+7.1f}%{r['avg']:>+6.2f}%"
              f"{r['med']:>+6.2f}%{r['pospct']:>7.0f}%{r['drought']:>7}月{r['lev']:>7.2f}x{r['mineq']:>9,.0f}{warn}")
    print("\n槓桿診斷（峰值槓桿發生時點 / 最大借款）：")
    for name, cfg in cfgs.items():
        r = evaluate(name, cfg)
        li = r["lev_info"]; borrowed = max(0, -r["min_cash"])
        when = f"{li[0]}(距高{li[1]:.0f}%)" if li else "—"
        print(f"  {r['name']:<18} 峰值槓桿 {r['lev']:.2f}x @ {when}  最大借款 {borrowed:,.0f}")


if __name__ == "__main__":
    main()
