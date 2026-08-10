"""
追漲回檔續勢（Relative-Strength / Momentum Pullback）— 與現有三策略「買弱勢」相反：買強勢回檔

現有三策略共同性質＝上升趨勢中「買弱」（超跌/新低/背離）。本策略反過來「買強」：
只在強勢上升股（多頭排列、動能為正）回檔到 20MA 附近再站上時進場，緊移動停損。
→ 交易時機與現有三策略互補（追漲 vs 抄底），可望低相關。

進場（全部成立）：
  1. 多頭排列：close > MA20 > MA60，且 MA20、MA60 皆上升
  2. 中期強勢：60 日報酬 > 0（絕對動能為正）
  3. 回檔到位：最近 3 日內最低價曾觸及/跌破 MA20（洗過浮額）
  4. 續勢確認：今日收紅（close > open）且 close 站回 MA20 之上
出場：移動停損（預設 8%），純停損不設指標賣訊（讓贏家跑，對齊 AD/超跌的出場哲學）

參數皆用市場慣用值（MA20/60、動能 60 日、停損 8%），不做最佳化。
回測用單股引擎 backtest.py.simulate，投組用分池（max 持倉/等權/每日限倉/系統性跳過）。
"""
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import backtest as bt  # noqa: E402

BASE_DIR = Path(__file__).parent.parent
STOCKS_PATH = BASE_DIR / "individual_stocks.json"
INITIAL_CAPITAL = 1_000_000
BUY_FEE, SELL_FEE, SELL_TAX = bt.BUY_FEE, bt.SELL_FEE, bt.SELL_TAX


def strategy_momentum_pullback(prices, mom_days=60, pullback_lookback=3):
    """回傳 per-bar signals（只有 'buy'；出場靠移動停損）。"""
    closes = [p["close"] for p in prices]
    opens = [p["open"] for p in prices]
    lows = [p["low"] for p in prices]
    ma20 = bt.calc_sma(closes, 20)
    ma60 = bt.calc_sma(closes, 60)
    n = len(prices)
    signals = [None] * n
    for i in range(65, n):
        if ma20[i] is None or ma60[i] is None or ma20[i - 5] is None or ma60[i - 5] is None:
            continue
        # 1. 多頭排列 + 均線上升
        if not (closes[i] > ma20[i] > ma60[i]):
            continue
        if ma20[i] <= ma20[i - 5] or ma60[i] <= ma60[i - 5]:
            continue
        # 2. 中期絕對動能為正
        if closes[i - mom_days] <= 0 or (closes[i] / closes[i - mom_days] - 1) <= 0:
            continue
        # 3. 最近 pullback_lookback 日曾回檔觸及 MA20
        touched = False
        for j in range(i - pullback_lookback + 1, i + 1):
            if ma20[j] is not None and lows[j] <= ma20[j]:
                touched = True
                break
        if not touched:
            continue
        # 4. 今日收紅且站回 MA20
        if closes[i] <= opens[i]:
            continue
        if closes[i] <= ma20[i]:
            continue
        signals[i] = "buy"
    return signals


def load_trending_universe():
    """篩出『上升趨勢』個股：用 validate 框架同款特徵（趨勢 R²>0.3）。"""
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


def portfolio_sim(universe, start, end, max_pos=7, trail=0.08,
                  daily_new=2, skip_gt=4):
    """單池分池模擬（追漲回檔）：next-bar 次日開盤成交，緊移動停損。"""
    # 預備每檔 bars + 訊號
    stocks = {}
    for sid in universe:
        try:
            full = bt.read_prices(sid)
        except Exception:
            continue
        sig = strategy_momentum_pullback(full)
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
    pending = []   # 次日開盤買進

    for d in dates:
        # 執行昨掛單
        for sid in pending:
            st = stocks.get(sid)
            j = st["idx"].get(d) if st else None
            if j is not None and sid not in positions and len(positions) < max_pos:
                px = st["bars"][j]["open"]
                if px > 0:
                    open_slots = max_pos - len(positions)
                    spend = cash / open_slots
                    shares = int(spend / (px * (1 + BUY_FEE)) / 1000) * 1000
                    if shares <= 0:
                        shares = int(spend / (px * (1 + BUY_FEE)))
                    if shares > 0:
                        cost = shares * px
                        cash -= cost + int(cost * BUY_FEE)
                        positions[sid] = {"sh": shares, "avg": px, "peak": px, "ed": d}
        pending = []

        # 停損檢查
        for sid in list(positions):
            st = stocks.get(sid)
            j = st["idx"].get(d) if st else None
            if j is None:
                continue
            c = st["bars"][j]["close"]
            pos = positions[sid]
            if c > pos["peak"]:
                pos["peak"] = c
            if c <= pos["peak"] * (1 - trail):
                rev = pos["sh"] * c
                cash += rev - int(rev * SELL_FEE) - int(rev * SELL_TAX)
                trades.append({"ret": (c - pos["avg"]) / pos["avg"], "reason": "stop"})
                del positions[sid]

        # 訊號蒐集
        raw = [sid for sid in universe
               if (st := stocks.get(sid)) and (j := st["idx"].get(d)) is not None
               and st["bars"][j]["buy"] and sid not in positions and sid not in pending]
        if skip_gt is not None and len(raw) > skip_gt:
            raw = []
        cnt = 0
        for sid in raw:
            if len(positions) + len(pending) >= max_pos:
                break
            if daily_new is not None and cnt >= daily_new:
                break
            pending.append(sid)
            cnt += 1

        val = cash + sum(p["sh"] * (stocks[s]["bars"][stocks[s]["idx"][d]]["close"]
                         if d in stocks[s]["idx"] else p["avg"])
                         for s, p in positions.items())
        equity_by_date[d] = val

    # 期末平倉
    last = dates[-1]
    for sid in list(positions):
        st = stocks[sid]
        j = st["idx"].get(last)
        c = st["bars"][j]["close"] if j is not None else positions[sid]["avg"]
        rev = positions[sid]["sh"] * c
        cash += rev - int(rev * SELL_FEE) - int(rev * SELL_TAX)
        trades.append({"ret": (c - positions[sid]["avg"]) / positions[sid]["avg"],
                       "reason": "forced"})
        del positions[sid]
    equity_by_date[last] = cash

    return equity_by_date, trades, dates


def metrics(equity_by_date, trades, dates, label):
    eq = [equity_by_date[d] for d in dates]
    tot = (eq[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    years = (datetime.strptime(dates[-1], "%Y-%m-%d") -
             datetime.strptime(dates[0], "%Y-%m-%d")).days / 365.25
    cagr = ((eq[-1] / INITIAL_CAPITAL) ** (1 / years) - 1) * 100 if years > 0 else 0
    peak = 0
    mdd = 0
    for v in eq:
        peak = max(peak, v)
        mdd = max(mdd, (peak - v) / peak if peak else 0)
    rets = [eq[i] / eq[i - 1] - 1 for i in range(1, len(eq)) if eq[i - 1] > 0]
    if len(rets) > 1:
        avg = sum(rets) / len(rets)
        std = (sum((r - avg) ** 2 for r in rets) / (len(rets) - 1)) ** 0.5
        sharpe = avg / std * (252 ** 0.5) if std > 0 else 0
    else:
        sharpe = 0
    closed = [t for t in trades if t["reason"] != "forced"]
    wins = [t for t in closed if t["ret"] > 0]
    wr = len(wins) / len(closed) * 100 if closed else 0
    avg_ret = sum(t["ret"] for t in closed) / len(closed) * 100 if closed else 0
    print(f"  {label}")
    print(f"    總報酬 {tot:+.0f}%  CAGR {cagr:+.1f}%  MDD -{mdd*100:.1f}%  Sharpe {sharpe:.2f}  "
          f"交易 {len(trades)} ({len(trades)/years:.0f}/年)  勝率 {wr:.0f}%  均報酬/筆 {avg_ret:+.2f}%")
    return {"cagr": cagr, "mdd": mdd * 100, "sharpe": sharpe, "trades": len(trades)}


def main():
    print("篩選上升趨勢母體（R²>0.3）...")
    uni = load_trending_universe()
    print(f"  母體 {len(uni)} 檔\n")
    print("=" * 90)
    print("  追漲回檔續勢（買強勢回檔）— 分池模擬")
    print("=" * 90)
    for label, (s, e) in [("完整期 2020-2026", (None, None)),
                          ("Period A 2020-2022", ("2020-01-01", "2022-12-31")),
                          ("Period B 2023-2026", ("2023-01-01", None))]:
        eqd, tr, dts = portfolio_sim(uni, s, e)
        metrics(eqd, tr, dts, label)
    print("=" * 90)
    print("  對照：現有三策略分池 CAGR +33% / MDD -13% / Sharpe 2.32；單策略約 +22~36%")
    print("=" * 90)


if __name__ == "__main__":
    main()
