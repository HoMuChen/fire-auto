"""
營收動能 × 現有三策略 分散化檢定 + 四池混合組合

營收動能是基本面訊號源，與現有三策略（純技術）正交，預期低相關。
檢驗：混合『三策略(技術) + 營收動能(基本面)』能否同時提升 Sharpe / 降 MDD。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import strat_lab as lab            # noqa: E402
import portfolio_sim as psim       # noqa: E402
import revenue_strategy as rev      # noqa: E402


def revenue_daily_equity(close, panel, top_n=50, rebal=20, regime_ma=200,
                         ma_price=60, min_price=5):
    """營收+價格雙動能的『每日』淨值曲線。"""
    dates = close.index.tolist()
    n = len(dates)
    idx = lab.equal_weight_index(close)
    regime_ok = idx > idx.rolling(regime_ma).mean()
    ma = close.rolling(ma_price).mean()
    equity = 1.0
    daily = {}
    i = max(regime_ma, ma_price)
    basket = []
    while i < n:
        d = dates[i]
        if not bool(regime_ok.loc[d]):
            basket = []
        else:
            sc = panel.loc[d].dropna()
            pr = close.loc[d]
            sc = sc[pr.reindex(sc.index) >= min_price]
            above = pr > ma.loc[d]
            sc = sc[above.reindex(sc.index).fillna(False)]
            basket = list(sc.sort_values(ascending=False).head(top_n).index)
        j = min(i + rebal, n - 1)
        equity *= (1 - lab.ROUND_TRIP * 0.5)
        if basket and j > i:
            sub = close[basket].iloc[i:j + 1]
            port = sub.div(sub.iloc[0]).mean(axis=1)
            for k in range(1, len(port)):
                daily[dates[i + k]] = equity * port.iloc[k]
            equity *= port.iloc[-1]
        else:
            for k in range(i + 1, j + 1):
                daily[dates[k]] = equity
        i = j
        if i >= n - 1:
            break
    s = pd.Series(daily)
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def metrics(eq):
    eq = eq.dropna()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = ((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1) * 100
    dd = ((eq - eq.cummax()) / eq.cummax()).min() * 100
    r = eq.pct_change().dropna()
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
    return cagr, dd, sharpe


def main():
    print("載入資料與淨值曲線...")
    close = lab.load_close_matrix()
    panel = rev.build_revenue_panel(close)
    rv = revenue_daily_equity(close, panel)

    dates, vals = psim.combined_daily_equity()
    base = pd.Series(vals, index=pd.to_datetime(dates)).sort_index()
    base = base[~base.index.duplicated(keep="last")]

    common = rv.index.intersection(base.index)
    rv = (rv.reindex(common).ffill())
    base = (base.reindex(common).ffill())
    rv = rv / rv.iloc[0]
    base = base / base.iloc[0]

    rr = rv.pct_change().dropna()
    rb = base.pct_change().dropna()
    ix = rr.index.intersection(rb.index)
    corr = np.corrcoef(rr.reindex(ix), rb.reindex(ix))[0, 1]

    print("=" * 88)
    print(f"  共同期間 {common[0].date()} ~ {common[-1].date()}  |  日報酬相關係數 = {corr:.2f}")
    print("=" * 88)
    cb = metrics(base)
    cr = metrics(rv)
    print(f"  現有三策略分池(技術)   CAGR {cb[0]:+.1f}%  MDD {cb[1]:.1f}%  Sharpe {cb[2]:.2f}")
    print(f"  營收動能(基本面)      CAGR {cr[0]:+.1f}%  MDD {cr[1]:.1f}%  Sharpe {cr[2]:.2f}")
    print("-" * 88)
    print("  混合組合（每日再平衡權重）：")
    best = None
    for w in [0.3, 0.4, 0.5, 0.6]:
        blend = (1 - w) * base + w * rv
        cg, dd, sh = metrics(blend)
        tag = ""
        if best is None or sh > best[0]:
            best = (sh, w, cg, dd)
        print(f"    {int((1-w)*100)}% 技術 + {int(w*100)}% 營收   "
              f"CAGR {cg:+.1f}%  MDD {dd:.1f}%  Sharpe {sh:.2f}{tag}")
    print("-" * 88)
    print(f"  ► 最佳混合：{int((1-best[1])*100)}%技術+{int(best[1]*100)}%營收 → "
          f"CAGR {best[2]:+.1f}% / MDD {best[3]:.1f}% / Sharpe {best[0]:.2f}")
    if corr < 0.5:
        print(f"  ► 低相關（{corr:.2f}）：基本面訊號源與技術正交，分散有效")
    print("=" * 88)


if __name__ == "__main__":
    main()
