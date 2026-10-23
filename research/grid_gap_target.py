"""
QFF 網格「跳空買深 → 停利等到上一批買價」回測對照。

規則（--gap-target）：每批停利價 = max(買價×(1+take), 買進當下上方最近一批的買價)。
  一般一格買進時兩者≈相同；跳空/急跌買到比一格更深時，改等回到上一批買價才賣（多吃反彈）。
設定對齊 live（格1.2%/停利1.2%/5x/12口/intraday_once/固定停利），每年以年初價重設本金、追繳即補錢；
另跑一次 2021-04 起完整期（本金以期初價定義 depth、補錢模式）。

用法：python3 research/grid_gap_target.py [--depth 1.25]
"""
import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import tsmc_futures_grid_30min as g  # noqa: E402
from grid_vix_filter import base_params  # noqa: E402

WINDOWS = [("2021", "2021-04-01", "2021-12-31"), ("2022", "2022-01-01", "2022-12-31"),
           ("2023", "2023-01-01", "2023-12-31"), ("2024", "2024-01-01", "2024-12-31"),
           ("2025", "2025-01-01", "2025-12-31"), ("2026", "2026-01-01", "2026-12-31")]


def one(a, start, end, gap, gap_min=0.0):
    p = base_params(a)
    p.start, p.end, p.topup, p.gap_target, p.gap_min = start, end, True, gap, gap_min
    r = g.run(p)
    cap, curve = p.capital, r["curve"]
    ms = g.months_of(r["dt"])
    vals = [r["mcf"].get(m, 0.0) / cap * 100 for m in ms]
    buys = [t for t in r["trade_log"] if t["action"] == "買"]
    deep = [t for t in buys if t.get("target") and t["target"] > t["adj_price"] * (1 + p.take)]
    sells = [t for t in r["trade_log"] if t["action"] == "賣"]
    return dict(ret=(curve[-1] - cap - r["total_deposit"]) / cap * 100, cf=r["realized"] / cap * 100,
                dep=r["total_deposit"] / cap * 100, pos=sum(1 for x in vals if x > 0.05), nm=len(ms),
                sells=len(sells), buys=len(buys), deep=len(deep),
                deep_gap=[t["target"] / t["adj_price"] - 1 for t in deep],
                avg_pnl=statistics.mean([t["pnl"] for t in sells]) if sells else 0,
                open=r["lots_open"], mdd=g.maxdd(curve))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=float, default=1.25)
    ap.add_argument("--thresholds", action="store_true")
    a = ap.parse_args()
    a.start, a.topup = None, True
    if a.thresholds:
        return thresholds(a)
    print(f"QFF 網格 live 設定，每年以年初價重設本金 depth={a.depth}、追繳即補錢")
    print(f"{'年':<6}{'':<6}{'年報酬%':>8}{'已實現%':>8}{'補錢%':>7}{'正月':>6}{'買':>5}{'賣':>5}"
          f"{'每筆賺':>8}{'期末口':>6}{'跳深批':>7}")
    tot = {False: [0, 0, 0], True: [0, 0, 0]}
    gaps = []
    for y, s0, e0 in WINDOWS:
        for gap in (False, True):
            st = one(a, s0, e0, gap)
            t = tot[gap]; t[0] += st["ret"]; t[1] += st["cf"]; t[2] += st["dep"]
            if gap:
                gaps += st["deep_gap"]
            print(f"{y if not gap else '':<6}{'新規則' if gap else '現行':<6}{st['ret']:>+8.1f}{st['cf']:>+8.1f}"
                  f"{st['dep']:>7.0f}{st['pos']:>3}/{st['nm']:<2}{st['buys']:>5}{st['sells']:>5}"
                  f"{st['avg_pnl']:>8,.0f}{st['open']:>6}{st['deep'] if gap else '':>7}")
    print("-" * 78)
    for gap in (False, True):
        t = tot[gap]
        print(f"{'合計':<6}{'新規則' if gap else '現行':<6}{t[0]:>+8.1f}{t[1]:>+8.1f}{t[2]:>7.0f}")
    if gaps:
        print(f"\n跳深批：{len(gaps)} 次，停利價比原本(+1.2%)多要求 平均 +{statistics.mean(gaps)*100 - 1.2:.2f}pp、"
              f"中位 +{statistics.median(gaps)*100 - 1.2:.2f}pp、最大 +{max(gaps)*100 - 1.2:.1f}pp")
    print("\n完整期 2021-04 起（不重設本金，補錢模式）")
    for gap in (False, True):
        st = one(a, "2021-04-01", None, gap)
        print(f"  {'新規則' if gap else '現行':<6} 總報酬 {st['ret']:+.0f}% 已實現 {st['cf']:+.0f}% 補錢 {st['dep']:.0f}% "
              f"正月 {st['pos']}/{st['nm']} 賣 {st['sells']} 每筆 {st['avg_pnl']:,.0f} 期末 {st['open']}口")


def thresholds(a):
    print("\n門檻比較（跳得比上一批低超過 X 才改停利價）")
    print(f"{'門檻':<14}" + "".join(f"{w[0]:>8}" for w in WINDOWS) + f"{'合計':>9}{'補錢':>7}{'跳深批':>7}{'完整期':>8}")
    for lab, gap, gm in [("現行", False, 0), ("任何更深", True, 0), (">1.5格(1.8%)", True, 0.018),
                         (">2格(2.4%)", True, 0.024), (">3格(3.6%)", True, 0.036)]:
        rs = [one(a, s0, e0, gap, gm) for _, s0, e0 in WINDOWS]
        full = one(a, "2021-04-01", None, gap, gm)
        print(f"{lab:<14}" + "".join(f"{r['ret']:>+8.0f}" for r in rs)
              + f"{sum(r['ret'] for r in rs):>+9.0f}{sum(r['dep'] for r in rs):>7.0f}"
              + f"{sum(r['deep'] for r in rs):>7}{full['ret']:>+8.0f}")


if __name__ == "__main__":
    main()
