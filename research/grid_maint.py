"""
H103：QFF 網格 × 大盤融資維持率「斷頭區加深」回測。
規則（事前登記，第十九批）：前一交易日收盤維持率 < 145% 的日子，格距由 1.2% 改 0.8%（停利仍 +1.2%），其餘照 live 設定。
維持率：research/v2/maint.py → data/v2/futures/mkt_maint.parquet（當日盤後才知 → 套用到下一交易日）。
對照：無濾網 live 設定，每年以年初價重設本金、追繳即補錢（同 grid_vix_filter.py）。

用法：python3 research/grid_maint.py [--depth 1.25] [--thr 145] [--alt 0.008]
"""
import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import grid_vix_filter as gv  # noqa: E402
import tsmc_futures_grid_30min as g  # noqa: E402

MAINT = Path(__file__).parent.parent / "data" / "v2" / "futures" / "mkt_maint.parquet"


def flag_days(thr):
    """回傳 set(維持率 < thr 的『下一交易日』日期字串)。交易日用 QFF 30 分 K 的日期。"""
    mt = pd.read_parquet(MAINT).maint
    mt.index = pd.to_datetime(mt.index)
    days = sorted({r["datetime"][:10] for r in csv.DictReader(open(g.CSV))})
    out = set()
    for i in range(len(days) - 1):
        d = pd.Timestamp(days[i])
        v = mt.get(d)
        if v is not None and 50 < v < thr:
            out.add(days[i + 1])
    return out


def year_run(a, start, end, sdays, alt):
    p = gv.base_params(a)
    p.start, p.end, p.topup = start, end, True
    p.step_days, p.step_alt = sdays, alt
    r = g.run(p)
    cap, curve = p.capital, r["curve"]
    ms = g.months_of(r["dt"])
    vals = [r["mcf"].get(m, 0.0) / cap * 100 for m in ms]
    return dict(ret=(curve[-1] - cap - r["total_deposit"]) / cap * 100, dep=r["total_deposit"] / cap * 100,
                pos=sum(1 for x in vals if x > 0.05), nm=len(ms), buys=sum(r["mbuy"].values()), open=r["lots_open"],
                lev=r["max_lev"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=float, default=1.25)
    ap.add_argument("--thr", type=float, default=145)
    ap.add_argument("--alt", type=float, default=0.008)
    a = ap.parse_args()
    a.start, a.topup = None, True
    sdays = flag_days(a.thr)
    windows = [("2021", "2021-04-01", "2021-12-31"), ("2022", "2022-01-01", "2022-12-31"),
               ("2023", "2023-01-01", "2023-12-31"), ("2024", "2024-01-01", "2024-12-31"),
               ("2025", "2025-01-01", "2025-12-31"), ("2026", "2026-01-01", "2026-12-31")]
    print(f"H103 QFF 網格 live 設定（1.2%/停利1.2%/5x/12口/intraday_once），每年重設本金 depth={a.depth}，追繳即補錢")
    print(f"斷頭區 = 前日收盤維持率 < {a.thr:.0f}% → 格距 {a.alt*100:.1f}%；每格：年報酬% | 補錢% | 期末口 | 正月/月 | 買次")
    hdr = f"{'規則':<10}" + "".join(f"{w[0]:>26}" for w in windows) + f"{'合計報酬':>9}{'合計補錢':>9}{'正月':>8}"
    print(hdr)
    for rn, sd in (("無濾網", None), (f"H103 <{a.thr:.0f}%", sdays)):
        line, tr, td, tp, tn = f"{rn:<10}", 0.0, 0.0, 0, 0
        for y, s0, e0 in windows:
            st = year_run(a, s0, e0, sd, a.alt)
            tr += st["ret"]; td += st["dep"]; tp += st["pos"]; tn += st["nm"]
            line += f"{st['ret']:>+8.0f} |{st['dep']:>4.0f} |{st['open']:>3} |{st['pos']:>2}/{st['nm']:<2}|{st['buys']:>3}"
        print(line + f"{tr:>+8.0f}%{td:>8.0f}%{tp:>5}/{tn}")
    print("\n斷頭區交易日數")
    line = f"{'':<10}"
    for y, s0, e0 in windows:
        line += f"{sum(1 for d in sdays if s0 <= d <= e0):>26}"
    print(line)
    if sdays:
        print("  日期：", ", ".join(sorted(sdays)))


if __name__ == "__main__":
    main()
