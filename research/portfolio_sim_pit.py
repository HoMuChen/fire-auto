"""三策略分池組合 — point-in-time（去後見之明）重驗（2026-10-20）

現行方案 A 的股票池有三層可能的未來資訊：
  ① 流動性名單：individual_stocks.json low_liquidity 用 2024-26 量能判定
  ② 群組 med_vol/trending：波動率/趨勢 R² 用 2020-2026 全期計算
  ③ 個股篩選：Period A 每檔平均報酬 > 0（對 A 期是樣本內；對 B 期是樣本外）

變體（皆分池、next_open 成交、復用 portfolio_sim 的模擬器）：
  V0 現行名單
  V1 現行名單 + 進場當日須 point-in-time 有流動性（近 60 日均量 ≥ 200 張）
  V2 Walk-forward（只測 B 期 2023-）：股票池=2022 年底 pit 有流動性的全部個股；
     群組用 2020-22 資料分；個股篩選用 2020-22 回測；進場也套 pit 流動性
  V2n 同 V2 但不做個股篩選（群組內全收），看篩選本身的貢獻
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import portfolio_sim as PS  # noqa: E402
import backtest as bt  # noqa: E402
from validate_strategy_groups import compute_stock_features, assign_groups  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
LIQ_SHARES = 200_000          # 200 張
A_END, B_START = "2022-12-31", "2023-01-01"
STOP = {"squeeze": 0.04, "oversold": 0.08, "ad": 0.08}

_liq = {}


def liquid_dates(sid):
    if sid not in _liq:
        d = pd.read_csv(BASE / "data/stock_prices" / f"{sid}.csv", usecols=["date", "volume", "close"])
        d = d[d.close > 0]
        ok = d.volume.rolling(60).mean() >= LIQ_SHARES
        _liq[sid] = set(d.date[ok])
    return _liq[sid]


def run_variant(lists, start, end, pit_mask):
    pools = {pk: PS.prep_pool(pk, lists[pk], start, end) for pk in PS.POOLS}
    if pit_mask:
        for stocks in pools.values():
            for sid, st in stocks.items():
                ld = liquid_dates(sid)
                for b in st["bars"]:
                    if b["buy"] and b["date"] not in ld:
                        b["buy"] = False
    dates = PS.all_dates(pools)
    total, ntr = None, 0
    for pk in PS.POOLS:
        eq, tr = PS.simulate_pool(pk, lists[pk], pools[pk], dates, "next_open")
        s = pd.Series(eq).reindex(dates).ffill().fillna(PS.INITIAL_CAPITAL / 3)
        total = s if total is None else total + s
        ntr += len(tr)
    total.index = pd.to_datetime(total.index)
    r = total.pct_change().dropna()
    yrs = (total.index[-1] - total.index[0]).days / 365.25
    return dict(CAGR=(total.iloc[-1] / total.iloc[0]) ** (1 / yrs) - 1, MDD=(total / total.cummax() - 1).min(),
                Sharpe=r.mean() / r.std() * np.sqrt(252), 筆每年=ntr / yrs,
                檔數=f"{len(lists['squeeze'])}/{len(lists['oversold'])}/{len(lists['ad'])}")


def walk_forward_lists():
    """只用 2020-2022 資料建 V2 名單。"""
    stocks = json.load(open(BASE / "individual_stocks.json"))
    univ, feats = [], {}
    for s in stocks:
        sid = s["stock_id"]
        f = BASE / "data/stock_prices" / f"{sid}.csv"
        if not f.exists():
            continue
        bars = [b for b in bt.read_prices(sid) if b["date"] <= A_END and b["close"] > 0]
        if len(bars) < 250:
            continue
        ld = liquid_dates(sid)
        if bars[-1]["date"] not in ld:           # 2022 年底當時要有流動性
            continue
        ft = compute_stock_features(bars)
        if ft:
            univ.append((sid, bars)); feats[sid] = ft
    groups, _, _ = assign_groups(feats)
    grp = [(sid, bars) for sid, bars in univ if groups[sid] == "med_vol/trending"]
    print(f"V2 股票池：2022 年底 pit 有流動性 {len(univ)} 檔 → med_vol/trending {len(grp)} 檔")
    fns = {pk: PS.POOLS[pk]["fn"] for pk in PS.POOLS}
    keep = {pk: [] for pk in PS.POOLS}
    for sid, bars in grp:
        for pk, fn in fns.items():
            res = bt.simulate(bars, fn(bars), {"type": "trailing_pct", "pct": STOP[pk]})
            trs = res["trade_details"]
            rets = [t["sell_price"] * (1 - bt.SELL_FEE - bt.SELL_TAX) / (t["buy_price"] * (1 + bt.BUY_FEE)) - 1
                    for t in trs if t["buy_date"] <= A_END]
            if not rets or np.mean(rets) > 0:      # 與原規則一致：無交易者保留
                keep[pk].append(sid)
    allg = sorted(sid for sid, _ in grp)
    return {pk: sorted(v) for pk, v in keep.items()}, {pk: allg for pk in PS.POOLS}


def main():
    cur = PS.load_filtered_lists()
    rows = []
    for lab, s, e in (("全期", None, None), ("A 2020-22", None, A_END), ("B 2023-", B_START, None)):
        rows.append(dict(變體="V0 現行名單", 期間=lab, **run_variant(cur, s, e, False)))
        rows.append(dict(變體="V1 現行名單+pit流動性", 期間=lab, **run_variant(cur, s, e, True)))
    wf, wf_all = walk_forward_lists()
    rows.append(dict(變體="V2 walk-forward(群組+篩選+pit)", 期間="B 2023-", **run_variant(wf, B_START, None, True)))
    rows.append(dict(變體="V2n walk-forward(群組+pit,不篩選)", 期間="B 2023-", **run_variant(wf_all, B_START, None, True)))
    ov = {pk: len(set(wf[pk]) & set(cur[pk])) for pk in PS.POOLS}
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("V2 名單與現行名單重疊檔數:", ov)


if __name__ == "__main__":
    main()
