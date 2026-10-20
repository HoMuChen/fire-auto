"""
三策略各自 + 分池組合，A/B/全期 報酬與DD —— 比較「無過濾」vs「全市場集中度過濾」。

過濾（全市場中位數版）：只保留 entry 信號日 T-1 買方集中度 > 當日全市場中位數的交易。
  - 分點資料 2021-06-30 起；entry < 2021-06-30 或無 T-1 集中度的交易 -> 無法過濾，視為保留（直通）。
  - 故 A 期僅部分過濾（2021H2~2022），B 期完整過濾。
資金/分池規則同 pool_period_report.py。
DD = 平倉結算階梯式（非逐日盯市），僅供相對比較，不等於文件逐日 DD。
"""
import sys
sys.path.insert(0, "/Users/mu/fire-auto")
sys.path.insert(0, "/Users/mu/fire-auto/research")
import json

import pandas as pd

import backtest as bt
import broker_concentration as bc
import pool_period_report as base

CONC_START = "2021-06-30"


def collect_trades_with_sid(label, cfg, universe):
    trades = []
    for sid in universe[label]:
        try:
            prices = bt.read_prices(sid)
        except Exception:
            continue
        if len(prices) < 200 or prices[0]["close"] == 0:
            continue
        r = bt.simulate(prices, cfg["fn"](prices), cfg["stop"], {"type": "signal"})
        for t in r["trade_details"]:
            if t["buy_price"] <= 0:
                continue
            ret = t["sell_price"] * (1 - bt.SELL_FEE - bt.SELL_TAX) / (t["buy_price"] * (1 + bt.BUY_FEE)) - 1
            trades.append({"stock_id": sid, "entry": t["buy_date"], "exit": t["sell_date"], "ret": ret})
    return trades


def mark_high(trades):
    """為每筆交易標記 entry 是否通過高集中度（T-1 > 當日全市場中位數）。"""
    conc = bc.load()
    conc["date"] = pd.to_datetime(conc["date"])
    conc = conc.sort_values("date").reset_index(drop=True)
    median = conc.groupby("date")["buy_conc"].median()
    df = pd.DataFrame(trades)
    df["sd"] = pd.to_datetime(df["entry"])
    df = df.sort_values("sd").reset_index(drop=True)
    asof = pd.merge_asof(df, conc.rename(columns={"date": "cdate"}),
                         left_on="sd", right_on="cdate", by="stock_id",
                         direction="backward", allow_exact_matches=False)
    asof["thr"] = asof["cdate"].map(median)
    # 通過條件：有 T-1 集中度且 > 中位數；無資料(早期)視為直通保留
    has = asof["buy_conc"].notna() & asof["thr"].notna()
    asof["pass"] = (~has) | (asof["buy_conc"] > asof["thr"])
    return asof


def run(scenario, universe, days):
    """scenario: 'none' or 'filter'. 回傳 combo_curve, 各策略 stats。"""
    combo = {d: 0.0 for d in days}
    rows = []
    for label, cfg in base.CFG.items():
        trades = collect_trades_with_sid(label, cfg, universe)
        marked = mark_high(trades)
        if scenario == "filter":
            marked = marked[marked["pass"]]
        kept = [{"entry": r.entry, "exit": r.exit, "ret": r.ret} for r in marked.itertuples(index=False)]
        curve = base.simulate_pool(kept, cfg, days)
        rA, dA, rB, dB, rF, dF = base.period_stats(curve, days)
        rows.append((label, rA, dA, rB, dB, rF, dF, base.cagr(rF, len(days)), len(kept)))
        for d in days:
            combo[d] += curve[d] / 3.0
    cA, ddA, cB, ddB, cF, ddF = base.period_stats(combo, days)
    rows.append(("分池組合", cA, ddA, cB, ddB, cF, ddF, base.cagr(cF, len(days)), None))
    return rows


def main():
    universe = base.load_universe()
    days = base.all_trading_days()
    print(f"交易日 {len(days)}（{days[0]}~{days[-1]}）。DD=階梯式粗估，僅供相對比較。\n")
    print("過濾僅自 2021-06-30 起生效（A 期部分過濾、B 期完整過濾）。\n")

    res = {s: run(s, universe, days) for s in ("none", "filter")}

    hdr = f"{'策略/組合':<12} | {'情境':<6} | {'A期 報酬/DD':<16} | {'B期 報酬/DD':<16} | {'全期 報酬/DD':<18} | CAGR / 筆數"
    print(hdr); print("-" * 104)
    for i in range(len(res["none"])):
        for s in ("none", "filter"):
            label, rA, dA, rB, dB, rF, dF, cg, n = res[s][i]
            tag = "無過濾" if s == "none" else "過濾"
            nstr = f"{n}筆" if n is not None else ""
            print(f"{label:<12} | {tag:<6} | {rA:>+7.1f}% /{dA:>5.1f}% | {rB:>+7.1f}% /{dB:>5.1f}% | "
                  f"{rF:>+8.1f}% /{dF:>5.1f}% | {cg:>+5.1f}% {nstr}")
        print("-" * 104)


if __name__ == "__main__":
    main()
