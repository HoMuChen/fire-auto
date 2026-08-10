"""
橫斷面策略研究台 — 排名型策略回測（與現有三策略「每檔信號」不同性質）

用途：對全體流動性個股做橫斷面排名（動能、低波動、相對強度…），
每 N 交易日換股持有 top-K，含市場趨勢濾網、交易成本，
輸出 CAGR / MDD / Sharpe，並可與現有三策略分池組合比相關性。

score 函數可插拔：score_fn(close_df) -> 同形狀 DataFrame（值越大越優先，NaN=不合格）。

用法：
    python3 research/strat_lab.py            # 跑內建幾個橫斷面策略對照
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "stock_prices"
STOCKS_PATH = BASE_DIR / "individual_stocks.json"

# 交易成本：買 0.1425%，賣 0.1425%+0.3%；一次完整換手（賣舊買新）≈ 0.585%
ROUND_TRIP = 0.001425 + 0.001425 + 0.003


def load_close_matrix(min_start="2020-01-31", start=None, end=None):
    """載入所有流動性且有 2020+ 資料的個股收盤，回傳對齊的 DataFrame(date × stock)。"""
    stocks = json.load(open(STOCKS_PATH, encoding="utf-8"))
    liquid = [s["stock_id"] for s in stocks if s.get("low_liquidity") is False]
    series = {}
    for sid in liquid:
        p = DATA_DIR / f"{sid}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p, usecols=["date", "close"])
        if df.empty or df["date"].iloc[0] > min_start:
            continue
        df = df[df["close"] > 0]
        series[sid] = df.set_index("date")["close"]
    close = pd.DataFrame(series).sort_index()
    if start:
        close = close[close.index >= start]
    if end:
        close = close[close.index <= end]
    return close


def equal_weight_index(close):
    """全市場等權指數（日報酬平均後累乘），當市場趨勢濾網用。"""
    rets = close.pct_change()
    idx = (1 + rets.mean(axis=1)).cumprod()
    return idx


# ─────────────────────────────────────────
# score 函數（橫斷面排名，值越大越優先）
# ─────────────────────────────────────────

def score_momentum(close, lookback=126, skip=21):
    """相對強度動能：過去 lookback 天報酬，跳過最近 skip 天（避開短期反轉）。"""
    return close.shift(skip) / close.shift(lookback) - 1


def score_low_vol(close, window=126):
    """低波動因子：過去 window 天日報酬標準差的「負值」（波動越低分數越高）。"""
    vol = close.pct_change().rolling(window).std()
    return -vol


def score_st_reversal(close, lookback=5):
    """短期反轉：過去 lookback 天報酬的「負值」（跌越多分數越高）→ 買最弱、賭反彈。"""
    return -(close.shift(1) / close.shift(lookback + 1) - 1)


def score_momentum_lowvol(close, lookback=126, skip=21, vol_window=126):
    """動能 / 波動（風險調整動能，類 Sharpe 排名）。"""
    mom = close.shift(skip) / close.shift(lookback) - 1
    vol = close.pct_change().rolling(vol_window).std()
    return mom / vol.replace(0, np.nan)


# ─────────────────────────────────────────
# 橫斷面回測引擎
# ─────────────────────────────────────────

def backtest_xs(close, score_fn, top_n=25, rebal=20, regime_ma=200,
                min_price=5, name="strategy", **score_kw):
    """
    close     : DataFrame(date × stock)
    score_fn  : 排名函數
    top_n     : 每期持有檔數（等權）
    rebal     : 每 rebal 交易日換股一次
    regime_ma : 市場等權指數 > 其 MA 才進場，否則空手（None=不濾）
    回傳 dict：equity(Series), metrics, trades_per_year, turnover
    """
    scores = score_fn(close, **score_kw)
    dates = close.index.tolist()
    n = len(dates)

    # 市場趨勢濾網
    regime_ok = pd.Series(True, index=close.index)
    if regime_ma:
        idx = equal_weight_index(close)
        regime_ok = idx > idx.rolling(regime_ma).mean()

    equity = [1.0]
    eq_dates = [dates[0]]
    prev_basket = set()
    total_rebalances = 0
    total_names_entered = 0
    turnover_sum = 0.0

    i = regime_ma or 126   # 暖身（等指標/MA 成形）
    # 對齊到第一個可換股日
    while i < n:
        d = dates[i]
        # 選股：分數前 top_n 且當日有價、價格 ≥ min_price
        row_score = scores.loc[d]
        row_price = close.loc[d]
        eligible = row_score.dropna()
        eligible = eligible[(row_price.reindex(eligible.index) >= min_price)]
        if regime_ma and not bool(regime_ok.loc[d]):
            basket = set()          # 空手
        else:
            basket = set(eligible.sort_values(ascending=False).head(top_n).index)

        # 換手成本
        if prev_basket or basket:
            union = prev_basket | basket
            changed = len(prev_basket ^ basket)
            turnover = changed / (2 * max(len(union), 1))
            turnover_sum += turnover
        else:
            turnover = 0.0
        total_rebalances += 1
        total_names_entered += len(basket - prev_basket)

        # 持有到下一次換股：等權組合報酬
        j = min(i + rebal, n - 1)
        if basket and j > i:
            held = list(basket)
            p0 = close.loc[dates[i], held]
            p1 = close.loc[dates[j], held]
            valid = p0.notna() & p1.notna() & (p0 > 0)
            if valid.any():
                per = (p1[valid] / p0[valid] - 1).mean()
            else:
                per = 0.0
        else:
            per = 0.0

        gross = equity[-1] * (1 + per)
        gross *= (1 - turnover * ROUND_TRIP)   # 換手成本
        equity.append(gross)
        eq_dates.append(dates[j])

        prev_basket = basket
        i = j
        if i >= n - 1:
            break

    eq = pd.Series(equity, index=pd.to_datetime(eq_dates))
    return _metrics(eq, name, total_rebalances, total_names_entered, dates)


def backtest_xs_stops(close, score_fn, top_n=25, rebal=20, regime_ma=200,
                      trail=0.15, min_price=5, name="strategy", **score_kw):
    """橫斷面 + 每檔移動停損：月換股選 top_n，但每日監控，
    持股自峰值回落 trail 即賣出轉現金，下次換股才補；市場濾網空手時全出。

    這把現有三策略「緊停損砍輸家」的紀律套到動能選股上（風險管理，非調參擬合）。
    """
    scores = score_fn(close, **score_kw)
    dates = close.index.tolist()
    n = len(dates)
    regime_ok = pd.Series(True, index=close.index)
    if regime_ma:
        idx = equal_weight_index(close)
        regime_ok = idx > idx.rolling(regime_ma).mean()

    warm = regime_ma or 126
    cash_w = 1.0                       # 現金權重
    holdings = {}                      # sid -> {"w": weight, "peak": price, "last": price}
    equity = 1.0
    eq_series = {dates[warm]: 1.0}
    names_entered = 0
    rebalances = 0

    for i in range(warm, n):
        d = dates[i]
        row_price = close.loc[d]

        # ── 每日：更新持股淨值、移動停損 ──
        new_holdings = {}
        for sid, h in holdings.items():
            px = row_price.get(sid)
            if px is None or np.isnan(px) or px <= 0:
                new_holdings[sid] = h            # 當日無報價，保留
                continue
            ret = px / h["last"] - 1
            equity_contrib = h["w"] * (1 + ret)
            # 移動停損
            peak = max(h["peak"], px)
            if px <= peak * (1 - trail):
                cash_w += equity_contrib          # 出場轉現金（成本在下方一次算）
                equity *= 1  # 權重轉移不影響總淨值
                # 賣出成本
                cash_w -= h["w"] * (1 + ret) * (0.001425 + 0.003)
            else:
                new_holdings[sid] = {"w": equity_contrib, "peak": peak, "last": px}
        # 重算總權重基準：把每日報酬反映到 equity
        holdings = new_holdings

        # 市場濾網關閉 → 全部轉現金
        if regime_ma and not bool(regime_ok.loc[d]):
            for sid, h in holdings.items():
                cash_w += h["w"] * (1 - (0.001425 + 0.003))
            holdings = {}

        # ── 換股日：補進 top_n ──
        is_rebal = (i - warm) % rebal == 0
        if is_rebal and not (regime_ma and not bool(regime_ok.loc[d])):
            eligible = scores.loc[d].dropna()
            eligible = eligible[row_price.reindex(eligible.index) >= min_price]
            target = list(eligible.sort_values(ascending=False).head(top_n).index)
            rebalances += 1
            slots = top_n - len(holdings)
            if slots > 0:
                total_w = cash_w + sum(h["w"] for h in holdings.values())
                per_w = total_w / top_n
                for sid in target:
                    if sid in holdings or slots <= 0:
                        continue
                    px = row_price.get(sid)
                    if px is None or np.isnan(px) or px <= 0:
                        continue
                    buy_w = min(per_w, cash_w)
                    if buy_w <= 0:
                        break
                    cash_w -= buy_w * (1 + 0.001425)
                    holdings[sid] = {"w": buy_w, "peak": px, "last": px}
                    names_entered += 1
                    slots -= 1

        equity = cash_w + sum(h["w"] for h in holdings.values())
        eq_series[d] = equity

    eq = pd.Series(eq_series)
    eq.index = pd.to_datetime(eq.index)
    return _metrics(eq, name, rebalances, names_entered, dates)


def _metrics(eq, name, rebalances, names_entered, all_dates):
    total_ret = (eq.iloc[-1] - 1) * 100
    days = (eq.index[-1] - eq.index[0]).days
    years = days / 365.25
    cagr = ((eq.iloc[-1]) ** (1 / years) - 1) * 100 if years > 0 else 0

    peak = eq.cummax()
    dd = (eq - peak) / peak
    mdd = dd.min() * 100

    rets = eq.pct_change().dropna()
    # 換股週期報酬 → 年化 Sharpe（每年約 252/rebal 期）
    periods_per_year = len(rets) / years if years > 0 else 0
    sharpe = (rets.mean() / rets.std() * np.sqrt(periods_per_year)) if rets.std() > 0 else 0

    return {
        "name": name,
        "equity": eq,
        "total_return_pct": round(total_ret, 1),
        "cagr_pct": round(cagr, 1),
        "mdd_pct": round(mdd, 1),
        "sharpe": round(sharpe, 2),
        "years": round(years, 1),
        "rebalances": rebalances,
        "names_entered": names_entered,
        "trades_per_year": round(names_entered / years) if years > 0 else 0,
    }


def report(res):
    print(f"  {res['name']:<32} 總報酬 {res['total_return_pct']:+7.0f}%  "
          f"CAGR {res['cagr_pct']:+6.1f}%  MDD {res['mdd_pct']:6.1f}%  "
          f"Sharpe {res['sharpe']:5.2f}  進場/年 {res['trades_per_year']:>4}")


def main():
    print("載入全市場收盤（流動性個股，2020+）...")
    close = load_close_matrix()
    print(f"  {close.shape[1]} 檔 × {close.shape[0]} 交易日"
          f"（{close.index[0]} ~ {close.index[-1]}）\n")

    print("=" * 96)
    print("  橫斷面策略對照（標準參數，含成本，市場趨勢濾網）")
    print("=" * 96)

    configs = [
        ("動能 6-1 (top25, 月換)", score_momentum, dict(top_n=25, rebal=20, regime_ma=200)),
        ("動能 6-1 (top15, 月換)", score_momentum, dict(top_n=15, rebal=20, regime_ma=200)),
        ("動能 12-1 (top25, 月換)", score_momentum, dict(top_n=25, rebal=20, regime_ma=200, lookback=252)),
        ("風險調整動能 (top25)", score_momentum_lowvol, dict(top_n=25, rebal=20, regime_ma=200)),
        ("低波動 (top25, 月換)", score_low_vol, dict(top_n=25, rebal=20, regime_ma=200)),
        ("動能 6-1 (無濾網)", score_momentum, dict(top_n=25, rebal=20, regime_ma=None)),
    ]
    for name, fn, kw in configs:
        score_kw = {k: v for k, v in kw.items()
                    if k in ("lookback", "skip", "window", "vol_window")}
        bt_kw = {k: v for k, v in kw.items() if k not in score_kw}
        res = backtest_xs(close, fn, name=name, **bt_kw, **score_kw)
        report(res)

    print("-" * 96)
    print("  短期反轉（買最弱、賭反彈；橫斷面、週換）— 靠反轉邊際但機制不同")
    print("-" * 96)
    rev_configs = [
        ("反轉 5日 (top25, 週換)", score_st_reversal, dict(top_n=25, rebal=5, regime_ma=200, lookback=5)),
        ("反轉 5日 (top50, 週換)", score_st_reversal, dict(top_n=50, rebal=5, regime_ma=200, lookback=5)),
        ("反轉 10日 (top25, 週換)", score_st_reversal, dict(top_n=25, rebal=5, regime_ma=200, lookback=10)),
        ("反轉 5日 (top25, 週換, 無濾網)", score_st_reversal, dict(top_n=25, rebal=5, regime_ma=None, lookback=5)),
        ("反轉 5日 (top25, 日換)", score_st_reversal, dict(top_n=25, rebal=1, regime_ma=200, lookback=5)),
    ]
    for name, fn, kw in rev_configs:
        score_kw = {k: v for k, v in kw.items() if k in ("lookback",)}
        bt_kw = {k: v for k, v in kw.items() if k not in score_kw}
        res = backtest_xs(close, fn, name=name, **bt_kw, **score_kw)
        report(res)

    print("-" * 96)
    print("  加每檔移動停損（砍輸家紀律套到動能選股）")
    print("-" * 96)
    stop_configs = [
        ("動能12-1 +停損20%", score_momentum_lowvol, dict(top_n=25, rebal=20, regime_ma=200, trail=0.20)),
        ("動能12-1 +停損15%", score_momentum_lowvol, dict(top_n=25, rebal=20, regime_ma=200, trail=0.15)),
        ("動能12-1 +停損10%", score_momentum_lowvol, dict(top_n=25, rebal=20, regime_ma=200, trail=0.10)),
        ("風險動能+停損15%(top15)", score_momentum_lowvol, dict(top_n=15, rebal=20, regime_ma=200, trail=0.15)),
    ]
    for name, fn, kw in stop_configs:
        score_kw = {k: v for k, v in kw.items()
                    if k in ("lookback", "skip", "window", "vol_window")}
        bt_kw = {k: v for k, v in kw.items() if k not in score_kw}
        res = backtest_xs_stops(close, fn, name=name, **bt_kw, **score_kw)
        report(res)

    print("=" * 96)
    print("  對照基準：現有三策略分池組合 = CAGR +33.0% / MDD −13% / Sharpe 2.32（完整期）")
    print("=" * 96)


if __name__ == "__main__":
    main()
