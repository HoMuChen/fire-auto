"""14 策略滾動 walk-forward 群組驗證（2026-10-20，去後見之明版）

與 validate_strategy_groups.py 的差異：
  ① 股票池 = individual_stocks.json 全部個股；進場日須 point-in-time 有流動性（近 60 日均量 ≥ 200 張）
  ② 分群每年重做：測試年 Y 的群組只用 Y 年以前（2020-01 ~ Y-1 年底）的波動率/趨勢 R²，
     且只納入 Y-1 年底有流動性的個股
  ③ 成交改 next-bar：訊號日的次一交易日開盤買/賣（引擎原為同日收盤）
  ④ 對照組：每 20 個交易日無條件進場、同一移動停損比例；並算每筆「超額 vs 同期 pit 大盤等權」

樣本外期間：OOS1 = 2022-2023（含空頭年），OOS2 = 2024-2026
判定：
  VALID     原規則 — 兩期平均每筆淨報酬 > 0 且每期 ≥ 30 筆
  BEAT_BASE 兩期「超額 − 對照組超額」皆 > 0（訊號本身有價值，不只是多頭 beta）
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE))
import backtest as bt  # noqa: E402
from validate_strategy_groups import compute_stock_features, assign_groups  # noqa: E402

LIQ = 200_000
TEST_YEARS = range(2022, 2027)
OOS = {"OOS1 22-23": (2022, 2023), "OOS2 24-26": (2024, 2026)}
STOPS = sorted({v["stop"]["pct"] for v in bt.STRATEGIES.values()})
OUT = BASE / "data/walkforward"


def stock_trades(sid):
    """回傳該股所有策略 + 對照組的 next-open 交易：(name, buy_date, sell_date, ret_close, ret_next)。"""
    try:
        bars = [b for b in bt.read_prices(sid) if b["close"] > 0 and b["open"] > 0]
    except FileNotFoundError:
        return sid, []
    if len(bars) < 120:
        return sid, []
    idx = {b["date"]: i for i, b in enumerate(bars)}
    out = []
    runs = [(k, v["fn"], v["stop"]) for k, v in bt.STRATEGIES.items()]
    base_sig = [("buy" if i % 20 == 0 else None) for i in range(len(bars))]
    runs += [(f"BASE_{int(p*100)}%", None, {"type": "trailing_pct", "pct": p}) for p in STOPS]
    for name, fn, stop in runs:
        try:
            sig = fn(bars) if fn else base_sig
            res = bt.simulate(bars, sig, stop, None)
        except Exception:  # noqa: BLE001
            continue
        for t in res["trade_details"]:
            i, j = idx.get(t["buy_date"]), idx.get(t["sell_date"])
            if i is None or j is None or j + 1 >= len(bars) or i + 1 >= len(bars):
                continue
            bo, so = bars[i + 1]["open"], bars[j + 1]["open"]
            rn = so * (1 - bt.SELL_FEE - bt.SELL_TAX) / (bo * (1 + bt.BUY_FEE)) - 1
            rc = t["sell_price"] * (1 - bt.SELL_FEE - bt.SELL_TAX) / (t["buy_price"] * (1 + bt.BUY_FEE)) - 1
            out.append((name, bars[i + 1]["date"], bars[j + 1]["date"], rc, rn))
    return sid, out


def load_frames(sids):
    c, v = {}, {}
    for s in sids:
        f = BASE / "data/stock_prices" / f"{s}.csv"
        if f.exists():
            d = pd.read_csv(f, usecols=["date", "close", "volume"], parse_dates=["date"]).set_index("date")
            d = d[d.close > 0]
            c[s], v[s] = d.close, d.volume
    return pd.DataFrame(c).sort_index(), pd.DataFrame(v).sort_index()


def yearly_groups(C, liq):
    G = {}
    for Y in TEST_YEARS:
        end = pd.Timestamp(f"{Y-1}-12-31")
        last = C.index[C.index <= end][-1]
        feats = {}
        for s in C.columns:
            if not liq.at[last, s]:
                continue
            cl = C[s].loc[:end].dropna()
            if len(cl) < 250:
                continue
            f = compute_stock_features([{"close": x} for x in cl.values])
            if f:
                feats[s] = f
        G[Y], _, _ = assign_groups(feats)
        print(f"  {Y} 年群組：用 2020~{Y-1} 資料，{len(feats)} 檔有流動性", flush=True)
    return G


def main():
    OUT.mkdir(exist_ok=True)
    sids = [s["stock_id"] for s in json.load(open(BASE / "individual_stocks.json"))]
    C, V = load_frames(sids)
    liq = (V.rolling(60).mean() >= LIQ).reindex_like(C).fillna(False)
    mkt = (1 + C.where(liq).pct_change(fill_method=None).clip(-0.105, 0.105).mean(axis=1).fillna(0)).cumprod()
    print(f"個股 {C.shape[1]} 檔，{C.index[0].date()}~{C.index[-1].date()}")
    G = yearly_groups(C, liq)

    cache = OUT / "trades.parquet"
    if cache.exists():
        T = pd.read_parquet(cache)
    else:
        with Pool(8) as p:
            res = p.map(stock_trades, list(C.columns), chunksize=8)
        T = pd.DataFrame([(s, *t) for s, ts in res for t in ts],
                         columns=["sid", "name", "buy", "sell", "ret_close", "ret"])
        T["buy"], T["sell"] = pd.to_datetime(T.buy), pd.to_datetime(T.sell)
        T.to_parquet(cache)
    T["year"] = T.buy.dt.year
    T = T[T.year.isin(TEST_YEARS)]
    # 進場日 pit 流動性 + 當年群組
    li = liq.stack()
    T = T[[bool(li.get((b, s), False)) for b, s in zip(T.buy, T.sid)]]
    T["group"] = [G[y].get(s) for y, s in zip(T.year, T.sid)]
    T = T.dropna(subset=["group"])
    T["ex"] = T.ret - (mkt.reindex(T.sell).values / mkt.reindex(T.buy).values - 1)
    T["oos"] = np.where(T.year <= 2023, "OOS1 22-23", "OOS2 24-26")
    T.to_parquet(OUT / "trades_oos.parquet")

    agg = T.groupby(["name", "group", "oos"]).agg(n=("ret", "size"), avg=("ret", "mean"), ex=("ex", "mean")).reset_index()
    base = agg[agg.name.str.startswith("BASE_")].rename(columns={"ex": "bex", "avg": "bavg"})
    rows = []
    for k, v in bt.STRATEGIES.items():
        bn = f"BASE_{int(v['stop']['pct']*100)}%"
        a = agg[agg.name == k].merge(base[base.name == bn][["group", "oos", "bex", "bavg"]], on=["group", "oos"])
        for g, x in a.groupby("group"):
            x = x.set_index("oos")
            if len(x) < 2:
                continue
            o1, o2 = x.loc["OOS1 22-23"], x.loc["OOS2 24-26"]
            enough = o1.n >= 30 and o2.n >= 30
            rows.append(dict(strategy=k, group=g, n1=o1.n, avg1=o1.avg * 100, edge1=(o1.ex - o1.bex) * 100,
                             n2=o2.n, avg2=o2.avg * 100, edge2=(o2.ex - o2.bex) * 100,
                             VALID=enough and o1.avg > 0 and o2.avg > 0,
                             BEAT_BASE=enough and o1.ex > o1.bex and o2.ex > o2.bex))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "group_results.csv", index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
    print("\n=== 策略 × 群組（avg=每筆淨報酬%，edge=超額 − 對照組超額 pp；next-open 成交）")
    print(R.round(2).to_string(index=False))

    # 策略層總表（全群組合併，日期 cluster t 值）
    S = []
    for k, v in bt.STRATEGIES.items():
        bn = f"BASE_{int(v['stop']['pct']*100)}%"
        x, b = T[T.name == k], T[T.name == bn]
        d = x.groupby("buy").ex.mean() - b.ex.mean()
        r = R[R.strategy == k]
        S.append(dict(strategy=k, n=len(x), avg=x.ret.mean() * 100, avg_close=x.ret_close.mean() * 100,
                      edge=(x.ex.mean() - b.ex.mean()) * 100, t_date=d.mean() / d.std() * np.sqrt(len(d)),
                      edge1=(x[x.oos == "OOS1 22-23"].ex.mean() - b[b.oos == "OOS1 22-23"].ex.mean()) * 100,
                      edge2=(x[x.oos == "OOS2 24-26"].ex.mean() - b[b.oos == "OOS2 24-26"].ex.mean()) * 100,
                      VALID群組=int(r.VALID.sum()), BEAT_BASE群組=int(r.BEAT_BASE.sum())))
    S = pd.DataFrame(S).sort_values("edge", ascending=False)
    S.to_csv(OUT / "strategy_summary.csv", index=False)
    print("\n=== 策略總表（avg_close=原引擎收盤成交；edge=每筆超額 − 對照組，pp；t_date=以進場日 cluster）")
    print(S.round(2).to_string(index=False))
    print("\n對照組：", T[T.name.str.startswith("BASE_")].groupby("name").agg(n=("ret", "size"), avg=("ret", "mean"), ex=("ex", "mean")).mul([1, 100, 100]).round(2).to_dict("index"))


if __name__ == "__main__":
    main()
