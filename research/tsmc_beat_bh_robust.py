"""
穩健性檢驗：「預設持有，破MA且外資賣超才退場」的參數鄰域網格 + 分期驗證
"""
import pandas as pd
import numpy as np
from tsmc_beat_bh import build_dataset, simulate, bench_bh, START, END


def make_signal(df, ma_n, fi_win):
    c = df["close"]
    ma = c.rolling(ma_n).mean()
    fi = (df["net_Foreign_Investor"] / df["volume"]).rolling(fi_win).sum()
    return ~((c < ma) & (fi < 0))


def main():
    df = build_dataset()
    bh = bench_bh(df)
    print(f"B&H: 總報酬 {bh['total']*100:+.1f}%  CAGR {bh['cagr']*100:+.1f}%  MDD {bh['mdd']*100:.1f}%  Sharpe {bh['sharpe']:.2f}\n")

    print("=== 參數網格（總報酬%，✓=贏過B&H） ===")
    ma_list = [100, 120, 150, 200, 240]
    fi_list = [5, 10, 15, 20]
    header = "MA\\FI  " + "".join(f"{w:>12}" for w in fi_list)
    print(header)
    grid = {}
    for ma_n in ma_list:
        row = f"{ma_n:<7}"
        for w in fi_list:
            r = simulate(df, make_signal(df, ma_n, w))
            grid[(ma_n, w)] = r
            mark = "✓" if r["total"] > bh["total"] else " "
            row += f"  {r['total']*100:>+8.1f}%{mark} "
        print(row)

    print("\n=== 同網格 Sharpe ===")
    print(header)
    for ma_n in ma_list:
        row = f"{ma_n:<7}"
        for w in fi_list:
            r = grid[(ma_n, w)]
            mark = "✓" if r["sharpe"] > bh["sharpe"] else " "
            row += f"  {r['sharpe']:>8.2f}{mark}  "
        print(row)

    print("\n=== 同網格 MDD ===")
    print(header)
    for ma_n in ma_list:
        row = f"{ma_n:<7}"
        for w in fi_list:
            r = grid[(ma_n, w)]
            row += f"  {r['mdd']*100:>8.1f}%  "
        print(row)

    # 分期驗證：前半 / 後半
    print("\n=== 分期驗證（MA200 / FI10） ===")
    periods = [("2021-06-30", "2023-12-31"), ("2024-01-01", "2026-07-02")]
    for st, en in periods:
        b = bench_bh(df, st, en)
        r = simulate(df, make_signal(df, 200, 10), st, en)
        beat = "✓贏" if r["total"] > b["total"] else "✗輸"
        print(f"  {st}~{en}: 策略 {r['total']*100:+.1f}% vs B&H {b['total']*100:+.1f}% {beat}"
              f"  (MDD {r['mdd']*100:.1f}% vs {b['mdd']*100:.1f}%)")

    # 全期 2020 起（無券商資料也不影響此策略）
    print("\n=== 更長窗口 2020-06-30 ~ 2026-07-02（僅價量+法人） ===")
    b = bench_bh(df, "2020-06-30", END)
    for ma_n in [120, 200]:
        for w in [10, 20]:
            r = simulate(df, make_signal(df, ma_n, w), "2020-06-30", END)
            beat = "✓" if r["total"] > b["total"] else " "
            print(f"  {beat} MA{ma_n}/FI{w}: {r['total']*100:+.1f}% vs B&H {b['total']*100:+.1f}%"
                  f"  Sharpe {r['sharpe']:.2f} vs {b['sharpe']:.2f}  MDD {r['mdd']*100:.1f}% vs {b['mdd']*100:.1f}%")

    # 勝者的出場區段
    print("\n=== MA200/FI10 的退場區段（在全窗口） ===")
    sig = make_signal(df, 200, 10)
    d = df[(df["date"] >= START) & (df["date"] <= END)].copy()
    d["sig"] = sig.reindex(d.index).fillna(False)
    d = d.reset_index(drop=True)
    out_start = None
    for i in range(len(d)):
        if not d.loc[i, "sig"] and out_start is None:
            out_start = i
        elif d.loc[i, "sig"] and out_start is not None:
            days = i - out_start
            if days >= 3:
                p0, p1 = d.loc[out_start, "close"], d.loc[i, "close"]
                print(f"  {d.loc[out_start,'date'].date()} ~ {d.loc[i,'date'].date()}"
                      f"  ({days:>3}天)  價格 {p0:.0f}→{p1:.0f} ({(p1/p0-1)*100:+.1f}%)")
            out_start = None


if __name__ == "__main__":
    main()
