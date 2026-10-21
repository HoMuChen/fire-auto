"""
分散化檢定：橫斷面動能（追漲/月換）能否分散現有三策略分池組合（抄底/短持）？

即使動能標準單獨報酬較低（+17% vs +33%），只要與現有組合低相關，
當成小 sleeve 加入就可能提升混合組合的 Sharpe / 降低 MDD。這是「不同性質」
策略的正當價值（非硬湊報酬、非過擬合）。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import strat_lab as lab          # noqa: E402
import portfolio_sim as psim     # noqa: E402


def momentum_daily_equity(close, lookback=252, skip=21, top_n=25, rebal=20,
                          regime_ma=200, min_price=5):
    """橫斷面動能的『每日』淨值（換股間逐日按持股報酬 mark）。"""
    scores = lab.score_momentum(close, lookback=lookback, skip=skip)
    idx = lab.equal_weight_index(close)
    regime_ok = idx > idx.rolling(regime_ma).mean()
    dates = close.index.tolist()
    n = len(dates)
    warm = regime_ma
    equity = 1.0
    daily = {}
    basket = []
    i = warm
    while i < n:
        d = dates[i]
        if not bool(regime_ok.loc[d]) if regime_ma else False:
            basket = []
        else:
            row = scores.loc[d].dropna()
            row = row[close.loc[d].reindex(row.index) >= min_price]
            basket = list(row.sort_values(ascending=False).head(top_n).index)
        j = min(i + rebal, n - 1)
        # 換手成本（近似：整籃替換一次）
        equity *= (1 - lab.ROUND_TRIP * 0.5)
        # 逐日 mark
        if basket and j > i:
            sub = close[basket].iloc[i:j + 1]
            norm = sub.div(sub.iloc[0])
            port = norm.mean(axis=1)      # 等權每日淨值（相對換股日）
            for k in range(1, len(port)):
                daily[dates[i + k]] = equity * port.iloc[k]
            equity = equity * port.iloc[-1]
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
    total = (eq.iloc[-1] / eq.iloc[0] - 1) * 100
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = ((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1) * 100
    dd = ((eq - eq.cummax()) / eq.cummax()).min() * 100
    r = eq.pct_change().dropna()
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
    return cagr, dd, sharpe


def main():
    print("載入資料與兩組淨值曲線...")
    close = lab.load_close_matrix()
    mom = momentum_daily_equity(close)

    dates, vals = psim.combined_daily_equity()
    base = pd.Series(vals, index=pd.to_datetime(dates)).sort_index()
    base = base[~base.index.duplicated(keep="last")]

    # 對齊共同日期
    common = mom.index.intersection(base.index)
    mom = mom.reindex(common).ffill()
    base = base.reindex(common).ffill()

    # 正規化到 1.0 起點
    mom = mom / mom.iloc[0]
    base = base / base.iloc[0]

    rm = mom.pct_change().dropna()
    rb = base.pct_change().dropna()
    idx = rm.index.intersection(rb.index)
    corr = np.corrcoef(rm.reindex(idx), rb.reindex(idx))[0, 1]

    print("=" * 84)
    print(f"  共同期間 {common[0].date()} ~ {common[-1].date()}  |  日報酬相關係數 = {corr:.2f}")
    print("=" * 84)
    c1 = metrics(base)
    c2 = metrics(mom)
    print(f"  現有三策略分池      CAGR {c1[0]:+.1f}%  MDD {c1[1]:.1f}%  Sharpe {c1[2]:.2f}")
    print(f"  橫斷面動能(單獨)    CAGR {c2[0]:+.1f}%  MDD {c2[1]:.1f}%  Sharpe {c2[2]:.2f}")
    print("-" * 84)
    print("  混合組合（每日再平衡權重）：")
    for w in [0.15, 0.25, 0.35]:
        blend = (1 - w) * base + w * mom     # 兩條正規化淨值加權
        cg, dd, sh = metrics(blend)
        print(f"    {int((1-w)*100)}% 三策略 + {int(w*100)}% 動能   "
              f"CAGR {cg:+.1f}%  MDD {dd:.1f}%  Sharpe {sh:.2f}")
    print("=" * 84)
    if corr < 0.5:
        print(f"  ► 低相關（{corr:.2f}）：動能雖單獨報酬低，作分散 sleeve 可望改善混合 Sharpe/MDD")
    else:
        print(f"  ► 相關偏高（{corr:.2f}）：分散效益有限")


if __name__ == "__main__":
    main()
