"""
投信認養（Investment-Trust Accumulation）— 不同訊號源（本土法人資金流，非價量技術）

台股特有現象：投信「認養」中小型股後持續買進拉抬。訊號源與現有三策略（純技術）
完全不同，最有機會低相關。用量正規化淨買 + 1 日 lag（盤後才公布）。

進場（選擇性、低週轉）：
  1. 投信近 5 日中 ≥4 日淨買（持續認養）
  2. 投信近 20 日累計淨買(量正規化) > 0 且為近期高檔
  3. 價格 > MA20（趨勢確認，跟著資金走）
出場：移動停損 10%（認養行情可延續）
next-bar 次日開盤成交。母體：趨勢股（R²>0.3）。
"""
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import backtest as bt  # noqa: E402
from momentum_pullback import metrics  # noqa: E402  復用績效輸出

BASE_DIR = Path(__file__).parent.parent
INST_DIR = BASE_DIR / "data" / "institutional"
STOCKS_PATH = BASE_DIR / "individual_stocks.json"
INITIAL_CAPITAL = 1_000_000
BUY_FEE, SELL_FEE, SELL_TAX = bt.BUY_FEE, bt.SELL_FEE, bt.SELL_TAX


def load_trust_net(sid):
    """回傳 {date: 投信淨買股數}。"""
    p = INST_DIR / f"{sid}.csv"
    if not p.exists():
        return {}
    import csv
    out = {}
    for r in csv.DictReader(open(p, encoding="utf-8")):
        if r["investor_name"] == "Investment_Trust":
            try:
                out[r["date"]] = int(r["buy"]) - int(r["sell"])
            except (ValueError, KeyError):
                pass
    return out


def chip_signal(prices, trust_net):
    """per-bar buy 訊號：投信持續認養 + 趨勢確認（用 t-1 籌碼，1 日 lag）。"""
    closes = [p["close"] for p in prices]
    vols = [p["volume"] for p in prices]
    ma20 = bt.calc_sma(closes, 20)
    n = len(prices)
    # 對齊投信淨買到 bar
    tnet = [trust_net.get(p["date"], 0) for p in prices]
    tratio = [tnet[i] / vols[i] if vols[i] > 0 else 0 for i in range(n)]
    signals = [None] * n
    for i in range(25, n):
        if ma20[i] is None:
            continue
        # 用 t-1 及更早（1 日 lag）
        last5 = tnet[i - 5:i]              # 近5日（不含今日）
        buy_days = sum(1 for x in last5 if x > 0)
        if buy_days < 4:
            continue
        accum20 = sum(tratio[i - 20:i])
        if accum20 <= 0:
            continue
        # 近期高檔：今日累計 ≥ 過去 20 日累計序列的相對高位（簡化：accum20 為正且近5日仍在買）
        if closes[i] <= ma20[i]:
            continue
        signals[i] = "buy"
    return signals


def load_trending_universe():
    from validate_strategy_groups import compute_stock_features
    stocks = json.load(open(STOCKS_PATH, encoding="utf-8"))
    liquid = [s["stock_id"] for s in stocks if s.get("low_liquidity") is False]
    uni = []
    for sid in liquid:
        try:
            prices = bt.read_prices(sid)
        except Exception:
            continue
        if len(prices) < 200 or prices[0]["close"] == 0 or prices[0]["date"] > "2020-01-31":
            continue
        feat = compute_stock_features(prices)
        if feat and feat["trend_r2"] > 0.3:
            uni.append(sid)
    return uni


def portfolio_sim(universe, start, end, max_pos=7, trail=0.10, daily_new=2, skip_gt=4):
    stocks = {}
    for sid in universe:
        try:
            full = bt.read_prices(sid)
        except Exception:
            continue
        tnet = load_trust_net(sid)
        if not tnet:
            continue
        sig = chip_signal(full, tnet)
        bars = []
        for i, b in enumerate(full):
            if start and b["date"] < start:
                continue
            if end and b["date"] > end:
                continue
            if b["close"] <= 0:
                continue
            bars.append({"date": b["date"], "open": b["open"], "close": b["close"],
                         "buy": sig[i] == "buy"})
        if len(bars) >= 2:
            stocks[sid] = {"bars": bars, "idx": {b["date"]: j for j, b in enumerate(bars)}}

    dates = sorted({b["date"] for st in stocks.values() for b in st["bars"]})
    cash = INITIAL_CAPITAL
    positions = {}
    trades = []
    equity_by_date = {}
    pending = []
    for d in dates:
        for sid in pending:
            st = stocks.get(sid)
            j = st["idx"].get(d) if st else None
            if j is not None and sid not in positions and len(positions) < max_pos:
                px = st["bars"][j]["open"]
                if px > 0:
                    spend = cash / (max_pos - len(positions))
                    shares = int(spend / (px * (1 + BUY_FEE)) / 1000) * 1000
                    if shares <= 0:
                        shares = int(spend / (px * (1 + BUY_FEE)))
                    if shares > 0:
                        cost = shares * px
                        cash -= cost + int(cost * BUY_FEE)
                        positions[sid] = {"sh": shares, "avg": px, "peak": px}
        pending = []
        for sid in list(positions):
            st = stocks.get(sid)
            j = st["idx"].get(d) if st else None
            if j is None:
                continue
            c = st["bars"][j]["close"]
            pos = positions[sid]
            pos["peak"] = max(pos["peak"], c)
            if c <= pos["peak"] * (1 - trail):
                rev = pos["sh"] * c
                cash += rev - int(rev * SELL_FEE) - int(rev * SELL_TAX)
                trades.append({"ret": (c - pos["avg"]) / pos["avg"], "reason": "stop"})
                del positions[sid]
        raw = [sid for sid in universe
               if (st := stocks.get(sid)) and (j := st["idx"].get(d)) is not None
               and st["bars"][j]["buy"] and sid not in positions and sid not in pending]
        if skip_gt is not None and len(raw) > skip_gt:
            raw = []
        cnt = 0
        for sid in raw:
            if len(positions) + len(pending) >= max_pos or (daily_new and cnt >= daily_new):
                break
            pending.append(sid)
            cnt += 1
        val = cash + sum(p["sh"] * (stocks[s]["bars"][stocks[s]["idx"][d]]["close"]
                         if d in stocks[s]["idx"] else p["avg"])
                         for s, p in positions.items())
        equity_by_date[d] = val
    last = dates[-1]
    for sid in list(positions):
        st = stocks[sid]
        j = st["idx"].get(last)
        c = st["bars"][j]["close"] if j is not None else positions[sid]["avg"]
        rev = positions[sid]["sh"] * c
        cash += rev - int(rev * SELL_FEE) - int(rev * SELL_TAX)
        trades.append({"ret": (c - positions[sid]["avg"]) / positions[sid]["avg"], "reason": "forced"})
    equity_by_date[last] = cash
    return equity_by_date, trades, dates


def main():
    print("篩選趨勢母體...")
    uni = load_trending_universe()
    print(f"  母體 {len(uni)} 檔\n")
    print("=" * 90)
    print("  投信認養（本土法人資金流，不同訊號源）— 分池模擬")
    print("=" * 90)
    for label, (s, e) in [("完整期 2020-2026", (None, None)),
                          ("Period A 2020-2022", ("2020-01-01", "2022-12-31")),
                          ("Period B 2023-2026", ("2023-01-01", None))]:
        eqd, tr, dts = portfolio_sim(uni, s, e)
        metrics(eqd, tr, dts, label)
    print("=" * 90)
    print("  對照：現有三策略分池 CAGR +33% / MDD -13% / Sharpe 2.32")
    print("=" * 90)


if __name__ == "__main__":
    main()
