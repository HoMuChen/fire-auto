"""
v2：驗證「高買方集中度 = 正向過濾」假設（使用者事前假設：籌碼集中易拉抬）。

修正 v1 兩問題：
  (1) 無前視：用信號日「前一個已公布交易日(T-1)」的集中度，而非信號當日盤後資料。
  (2) 穩健性：門檻掃 40/50/60 分位，看「保留高集中度」結論是否一致。

保留規則：T-1 集中度 > 當日該分位門檻(全市場橫斷面) -> 保留(高集中)。
比較 A 期(21H2-22) / B 期(23-26) 每筆淨報酬與勝率。
"""
import json
import sys
sys.path.insert(0, "/Users/mu/fire-auto")

import pandas as pd

import backtest as bt
import broker_concentration as bc

BUY_FEE, SELL_FEE, SELL_TAX = bt.BUY_FEE, bt.SELL_FEE, bt.SELL_TAX
STRATS = {
    "squeeze": ("波動率擠壓", bt.strategy_squeeze, {"type": "trailing_pct", "pct": 0.04}),
    "oversold": ("超跌反彈", bt.strategy_oversold_reversal, {"type": "trailing_pct", "pct": 0.08}),
    "ad_divergence": ("AD背離", bt.strategy_ad_divergence, {"type": "trailing_pct", "pct": 0.08}),
}
A_START, A_END = pd.Timestamp("2021-07-01"), pd.Timestamp("2022-12-31")
B_START, B_END = pd.Timestamp("2023-01-01"), pd.Timestamp("2026-12-31")


def net_return(b, s):
    return s * (1 - SELL_FEE - SELL_TAX) / (b * (1 + BUY_FEE)) - 1


def load_filters():
    lists = json.load(open("/Users/mu/fire-auto/strategies/filtered_stock_lists.json"))
    return {k: [x["stock_id"] for x in v["kept"]] for k, v in lists["strategies"].items()}


def collect_trades(key):
    _, fn, stop = STRATS[key]
    rows = []
    for sid in load_filters()[key]:
        try:
            prices = bt.read_prices(sid)
        except Exception:
            continue
        if len(prices) < 120:
            continue
        res = bt.simulate(prices, fn(prices), stop, {"type": "signal"})
        for t in res["trade_details"]:
            rows.append({"stock_id": sid, "signal_date": t["buy_date"],
                         "net_ret": net_return(t["buy_price"], t["sell_price"])})
    return pd.DataFrame(rows)


def summarize(df, lo, hi):
    d = df[(df["signal_date"] >= lo) & (df["signal_date"] <= hi)]
    if len(d) == 0:
        return (0, None, None)
    return (len(d), d["net_ret"].mean() * 100, (d["net_ret"] > 0).mean() * 100)


def fmt(n, r, w):
    rs = f"{r:+.2f}%" if r is not None else "   -  "
    ws = f"{w:.0f}%" if w is not None else " - "
    return f"{n:>5} {rs:>8} {ws:>6}"


def main():
    conc = bc.load()
    conc["date"] = pd.to_datetime(conc["date"])
    conc = conc.sort_values("date").reset_index(drop=True)
    # 每日各分位門檻（index 為 datetime）
    thresholds = {p: conc.groupby("date")["buy_conc"].quantile(p) for p in (0.4, 0.5, 0.6)}

    print("無前視(T-1集中度)。'高集中'=T-1集中度 > 當日分位門檻。\n")
    for key, (label, _, _) in STRATS.items():
        trades = collect_trades(key)
        trades["signal_date"] = pd.to_datetime(trades["signal_date"])
        trades = trades.sort_values("signal_date").reset_index(drop=True)
        # 無前視：merge_asof 取嚴格早於信號日的最近一筆集中度
        asof = pd.merge_asof(
            trades, conc.rename(columns={"date": "conc_date"}),
            left_on="signal_date", right_on="conc_date", by="stock_id",
            direction="backward", allow_exact_matches=False,
        )
        asof = asof[asof["buy_conc"].notna()].copy()
        print(f"=== {label}  (有效交易 {len(asof)} 筆, 無前視) ===")
        print(f"{'門檻':<8} {'集合':<8} | {'A期 筆/報酬/勝':<22} | {'B期 筆/報酬/勝':<22}")
        na, ra, wa = summarize(asof, A_START, A_END)
        nb, rb, wb = summarize(asof, B_START, B_END)
        print(f"{'(全部)':<8} {'ALL':<8} | {fmt(na,ra,wa)} | {fmt(nb,rb,wb)}")
        for p in (0.4, 0.5, 0.6):
            med = thresholds[p]
            asof["thr"] = asof["conc_date"].map(med)
            high = asof[asof["buy_conc"] > asof["thr"]]
            na, ra, wa = summarize(high, A_START, A_END)
            nb, rb, wb = summarize(high, B_START, B_END)
            print(f"{'>'+str(int(p*100))+'分位':<8} {'高集中':<8} | {fmt(na,ra,wa)} | {fmt(nb,rb,wb)}")
        print()


if __name__ == "__main__":
    main()
