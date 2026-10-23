"""
QFF 網格 × 台指 VIX 濾網回測：VIX 處於「恐慌/升溫」狀態時暫停網格開新倉，停利照常。

VIX 來源：research/txo_vix.py 自建（TXO 結算價照 CBOE 公式，與官方比對 r=0.999）。
防前視：當日能否開倉只看「前一交易日收盤」的 VIX。
網格設定對齊 live（固定停利、5x、12口上限、intraday_once），本金以 --depth 定義（期初價×口數）。

規則（市場慣用預設值，不做搜索）：
  abs30   : VIX > 30（常見恐慌線）
  ma20    : VIX > 自身 20 日均線（波動升溫中）
  ma60x   : VIX > 自身 60 日均線 × 1.2（顯著高於常態）
  spike   : VIX 5 日漲幅 > 20%，之後 10 個交易日內都暫停（急升 + 冷卻）

用法：python3 research/grid_vix_filter.py [--depth 2] [--start 2021-04-01]
"""
import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import tsmc_futures_grid_30min as g  # noqa: E402

VIX = Path(__file__).parent.parent / "data" / "futures" / "TXO_vix_daily.csv"


def load_vix():
    rows = list(csv.DictReader(open(VIX)))
    return [r["date"] for r in rows], [float(r["vix"]) for r in rows]


def ma(xs, n, i):
    return sum(xs[i - n + 1:i + 1]) / n if i >= n - 1 else None


def rule_flags(dates, v):
    """回傳 {rule: set(被禁止開倉的『隔一交易日』)}——第 i 天狀態套用到第 i+1 天"""
    flags = {k: set() for k in ("abs30", "ma20", "ma60x", "spike")}
    cool = 0
    for i in range(len(v) - 1):
        nxt = dates[i + 1]
        if v[i] > 30:
            flags["abs30"].add(nxt)
        m20 = ma(v, 20, i)
        if m20 and v[i] > m20:
            flags["ma20"].add(nxt)
        m60 = ma(v, 60, i)
        if m60 and v[i] > m60 * 1.2:
            flags["ma60x"].add(nxt)
        if i >= 5 and v[i] / v[i - 5] - 1 > 0.20:
            cool = 10
        if cool > 0:
            flags["spike"].add(nxt); cool -= 1
    return flags


def base_params(a):
    ns = argparse.Namespace(
        capital=0, depth=a.depth, max_lots=12, contracts_per_lot=1, step=0.012, take=0.012,
        h_days=10, fee_per_side=g.DEFAULT_FEE_PER_SIDE, roll_cost_points=g.DEFAULT_ROLL_COST_POINTS,
        max_leverage=5.0, regime_ma=0, intraday_once=True, topup=a.topup, initial_margin=0.135,
        maintenance_margin=0.1035, start=a.start, back_adjust=False, atr_mult=0.0, trail_tp=0.0,
        widen=0.0, rsi_gate=0.0, rsi_period=70, depth_scale=0.0, depth_ref_bars=600,
        shallow_boost=0.0, shallow_taper=0.15, derisk_lev=0.0, buy_block=None)
    return ns


def stats(p):
    r = g.run(p)
    dt, curve, cap = r["dt"], r["curve"], p.capital
    ms = g.months_of(dt)
    vals = [r["mcf"].get(m, 0.0) / cap * 100 for m in ms]
    dry = mx = 0
    for x in vals:
        dry = dry + 1 if x < 0.3 else 0; mx = max(mx, dry)
    years = len(dt) / (g.BARS_PER_DAY * 252)
    yr = defaultdict(float)
    for m in ms:
        yr[m[:4]] += r["mcf"].get(m, 0.0) / cap * 100
    # 年度 MDD
    ydd = {}
    for y in sorted(yr):
        seg = [e for d, e in zip(dt, curve) if d[:4] == y]
        ydd[y] = g.maxdd(seg) if seg else 0
    return dict(cagr=((curve[-1] / cap) ** (1 / years) - 1) * 100, mdd=g.maxdd(curve),
                pos=sum(1 for x in vals if x > 0.05) / len(vals) * 100, dry=mx,
                med=statistics.median(vals), yr=dict(yr), ydd=ydd, buys=sum(r["mbuy"].values()),
                lev=r["max_lev"], dep=r["total_deposit"], dep_n=r["deposit_events"])


def year_run(a, start, end, block):
    p = base_params(a)
    p.start, p.end, p.buy_block = start, end, block
    p.topup = True
    r = g.run(p)
    cap, curve = p.capital, r["curve"]
    ms = g.months_of(r["dt"])
    vals = [r["mcf"].get(m, 0.0) / cap * 100 for m in ms]
    # 「水下深度」：權益相對（本金+累計補錢）的最差位置
    return dict(ret=(curve[-1] - cap - r["total_deposit"]) / cap * 100,
                cf=r["realized"] / cap * 100, dep=r["total_deposit"] / cap * 100,
                pos=sum(1 for x in vals if x > 0.05), nm=len(ms), buys=sum(r["mbuy"].values()),
                open=r["lots_open"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=float, default=1.25, help="本金=depth×年初價×100（live 30萬/2400≈1.25）")
    a = ap.parse_args()
    a.start, a.topup = None, True
    vd, vv = load_vix()
    flags = rule_flags(vd, vv)
    windows = [("2021", "2021-04-01", "2021-12-31"), ("2022", "2022-01-01", "2022-12-31"),
               ("2023", "2023-01-01", "2023-12-31"), ("2024", "2024-01-01", "2024-12-31"),
               ("2025", "2025-01-01", "2025-12-31"), ("2026", "2026-01-01", "2026-12-31")]
    rules = ["無濾網", "abs30", "ma20", "ma60x", "spike"]
    print(f"QFF 網格 live 設定（格1.2%/停利1.2%/5x/12口/intraday_once），每年以年初價重設本金"
          f" depth={a.depth}，追繳即補錢")
    print("每格：年報酬%(含未實現,扣補錢) | 補錢%本金 | 期末未平倉口")
    print(f"{'規則':<7}" + "".join(f"{w[0]:>20}" for w in windows) + f"{'合計報酬':>9}{'合計補錢':>9}")
    for rn in rules:
        line, tr, td = f"{rn:<7}", 0.0, 0.0
        for y, s0, e0 in windows:
            st = year_run(a, s0, e0, None if rn == "無濾網" else flags[rn])
            tr += st["ret"]; td += st["dep"]
            line += f"{st['ret']:>+8.0f} |{st['dep']:>4.0f} |{st['open']:>3}  "
        print(line + f"{tr:>+8.0f}%{td:>8.0f}%")
    # 各規則每年停買天數比例
    print("\n停買日比例（%）")
    for rn in rules[1:]:
        line = f"{rn:<7}"
        for y, s0, e0 in windows:
            ds = {r["datetime"][:10] for r in csv.DictReader(open(g.CSV)) if s0 <= r["datetime"][:10] <= e0}
            line += f"{sum(d in flags[rn] for d in ds)/len(ds)*100:>20.0f}"
        print(line)


if __name__ == "__main__":
    main()
