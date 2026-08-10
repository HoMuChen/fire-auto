"""
月營收動能（Revenue Momentum）— 台股特有基本面訊號，與價量/籌碼正交

文獻（FinLab、TEJ）：月營收 YoY 成長 + 價格動能「雙動能」是台股最受驗證的因子，
年化 ~26-29%、回撤低於 0050。訊號源為基本面營收，與現有三策略（純技術）不同性質。

避免前視：某月營收用 create_time（實際公布日，約次月 10 號）之後才可交易。
YoY 成長（vs 去年同月）避開台股營收的強季節性。

策略（橫斷面、月換股）：
  - 每月換股日，取「可得的最新營收 YoY」橫斷面排名前 top_n
  - 可選價格濾網（price > MA60，雙動能）與市場趨勢濾網
  - 等權持有、移動到下次換股、含成本
"""
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import strat_lab as lab  # noqa: E402

BASE = Path(__file__).parent.parent
REV_DIR = BASE / "data" / "revenue"


def load_revenue_yoy_series(sid):
    """回傳 pd.Series：index=公布日(create_time)，value=該月營收 YoY 成長。"""
    p = REV_DIR / f"{sid}.csv"
    if not p.exists():
        return None
    rev = {}
    pub = {}
    for r in csv.DictReader(open(p, encoding="utf-8")):
        try:
            y, m = int(r["revenue_year"]), int(r["revenue_month"])
            val = float(r["revenue"])
        except (ValueError, KeyError):
            continue
        if val <= 0:
            continue
        ct = (r.get("create_time") or "")[:10]
        # 公布日：有 create_time 用它；否則用法規截止日「次月 10 號」（保守防前視）
        pm_y, pm_m = (y + 1, 1) if m == 12 else (y, m + 1)
        synth = f"{pm_y:04d}-{pm_m:02d}-10"
        rev[(y, m)] = val
        pub[(y, m)] = ct if (ct and len(ct) == 10) else synth
    recs = []
    for (y, m), val in rev.items():
        prev = rev.get((y - 1, m))
        if prev and prev > 0:
            recs.append((pub[(y, m)], val / prev - 1))
    if not recs:
        return None
    recs.sort()
    s = pd.Series({d: v for d, v in recs})
    return s[~s.index.duplicated(keep="last")]


def build_revenue_panel(close):
    """把每檔 YoY 序列 forward-fill 對齊到 close 的日期軸 → panel(date × stock)。"""
    cols = {}
    for sid in close.columns:
        s = load_revenue_yoy_series(sid)
        if s is None:
            continue
        s = s.sort_index()   # 字串日期 YYYY-MM-DD，字典序=時序，與 close.index 同型
        union = close.index.union(s.index)
        aligned = s.reindex(union).ffill().reindex(close.index)
        cols[sid] = aligned
    return pd.DataFrame(cols).reindex(columns=close.columns)


def backtest_revenue(close, panel, top_n=25, rebal=20, regime_ma=200,
                     price_filter=True, ma_price=60, min_price=5, name="營收動能"):
    dates = close.index.tolist()
    n = len(dates)
    regime_ok = pd.Series(True, index=close.index)
    if regime_ma:
        idx = lab.equal_weight_index(close)
        regime_ok = idx > idx.rolling(regime_ma).mean()
    ma = close.rolling(ma_price).mean() if price_filter else None

    equity = [1.0]
    eq_dates = [dates[max(regime_ma or 60, ma_price)]]
    prev = set()
    entered = 0
    turnover_sum = 0.0
    i = max(regime_ma or 60, ma_price)
    while i < n:
        d = dates[i]
        if regime_ma and not bool(regime_ok.loc[d]):
            basket = set()
        else:
            sc = panel.loc[d].dropna()
            price_row = close.loc[d]
            sc = sc[price_row.reindex(sc.index) >= min_price]
            if price_filter:
                above = price_row > ma.loc[d]
                sc = sc[above.reindex(sc.index).fillna(False)]
            basket = set(sc.sort_values(ascending=False).head(top_n).index)
        changed = len(prev ^ basket)
        turnover = changed / (2 * max(len(prev | basket), 1))
        turnover_sum += turnover
        entered += len(basket - prev)
        j = min(i + rebal, n - 1)
        if basket and j > i:
            p0 = close.loc[dates[i], list(basket)]
            p1 = close.loc[dates[j], list(basket)]
            valid = p0.notna() & p1.notna() & (p0 > 0)
            per = (p1[valid] / p0[valid] - 1).mean() if valid.any() else 0.0
        else:
            per = 0.0
        val = equity[-1] * (1 + per) * (1 - turnover * lab.ROUND_TRIP)
        equity.append(val)
        eq_dates.append(dates[j])
        prev = basket
        i = j
        if i >= n - 1:
            break
    eq = pd.Series(equity, index=pd.to_datetime(eq_dates))
    return lab._metrics(eq, name, 0, entered, dates)


def segment_metrics(eq, start, end, label):
    """從『完整期』淨值曲線切出子區間算 CAGR/MDD/Sharpe（避免重算濾網暖身的假象）。"""
    seg = eq[(eq.index >= pd.Timestamp(start)) & (eq.index <= pd.Timestamp(end or "2100"))]
    if len(seg) < 3:
        print(f"  {label}: 資料不足")
        return
    seg = seg / seg.iloc[0]
    years = (seg.index[-1] - seg.index[0]).days / 365.25
    cagr = ((seg.iloc[-1]) ** (1 / years) - 1) * 100 if years > 0 else 0
    dd = ((seg - seg.cummax()) / seg.cummax()).min() * 100
    r = seg.pct_change().dropna()
    sharpe = r.mean() / r.std() * (len(r) / years) ** 0.5 if r.std() > 0 and years > 0 else 0
    print(f"  {label:<26} CAGR {cagr:+6.1f}%  MDD {dd:6.1f}%  Sharpe {sharpe:5.2f}")


def main():
    print("載入收盤矩陣 + 建營收 YoY panel...")
    close = lab.load_close_matrix()
    panel = build_revenue_panel(close)
    have = panel.notna().any().sum()
    print(f"  {have} 檔有營收資料\n")

    print("=" * 96)
    print("  月營收動能（不同訊號源：基本面營收；含成本、市場濾網）")
    print("=" * 96)
    configs = [
        ("純營收YoY top25 月換", dict(top_n=25, rebal=20, price_filter=False)),
        ("營收YoY+價格>MA60 雙動能 top25", dict(top_n=25, rebal=20, price_filter=True)),
        ("營收YoY+價格 雙動能 top15", dict(top_n=15, rebal=20, price_filter=True)),
        ("營收YoY+價格 雙動能 top50", dict(top_n=50, rebal=20, price_filter=True)),
    ]
    for label, kw in configs:
        res = backtest_revenue(close, panel, name=label, **kw)
        lab.report(res)

    print("-" * 96)
    print("  分期驗證（切完整期曲線，避免濾網暖身假象）top50 雙動能：")
    res = backtest_revenue(close, panel, name="full", top_n=50, rebal=20, price_filter=True)
    eq = res["equity"]
    segment_metrics(eq, "2020-01-01", "2022-12-31", "Period A 2020-2022")
    segment_metrics(eq, "2023-01-01", "2026-12-31", "Period B 2023-2026")
    print("-" * 96)
    print("  參數穩健性（top_n × 換股週期，雙動能）：")
    for tn in [25, 40, 60]:
        for rb in [10, 20, 40]:
            r = backtest_revenue(close, panel, name=f"top{tn}/rebal{rb}",
                                 top_n=tn, rebal=rb, price_filter=True)
            lab.report(r)
    print("=" * 96)
    print("  對照：現有三策略分池 CAGR +33% / MDD -13% / Sharpe 2.32")
    print("=" * 96)


if __name__ == "__main__":
    main()
