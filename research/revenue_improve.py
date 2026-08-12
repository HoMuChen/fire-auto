"""
改進月營收動能的最大回撤（不過擬合前提）— 真實日 mark 比較

基準（top50/rebal20 雙動能）：CAGR +32.7% / 日mark MDD -24% / Sharpe 1.50
嘗試（皆用標準門檻、可查穩健性）：
  A. 營收加速度篩選：只留 YoY 較上月加速的名單（品質）
  B. 更快市場濾網：MA100 取代 MA200（更早離場）
  C. 每檔移動停損：寬停損（15/20%）砍崩跌名
  D. 低波動傾斜：營收領先者中偏好低波動
逐項與組合測 DD 是否下降、報酬/Sharpe 代價多少。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import strat_lab as lab            # noqa: E402
import revenue_strategy as rev      # noqa: E402


def build_panels(close):
    """回傳 (yoy_panel, accel_panel)：YoY 與『YoY 較上月變化』對齊日期軸。"""
    yoy_cols, acc_cols = {}, {}
    for sid in close.columns:
        s = rev.load_revenue_yoy_series(sid)
        if s is None:
            continue
        s = s.sort_index()
        acc = s.diff()                       # YoY 較上月變化（加速度）
        union = close.index.union(s.index)
        yoy_cols[sid] = s.reindex(union).ffill().reindex(close.index)
        acc_cols[sid] = acc.reindex(union).ffill().reindex(close.index)
    yoy = pd.DataFrame(yoy_cols).reindex(columns=close.columns)
    acc = pd.DataFrame(acc_cols).reindex(columns=close.columns)
    return yoy, acc


def daily_bt(close, yoy, accel=None, vol=None, top_n=50, rebal=20, regime_ma=200,
             ma_price=60, min_price=5, require_accel=False, trail=None,
             lowvol_tilt=False, name="rev"):
    """日 mark 引擎，含各種 DD 改進開關。"""
    dates = close.index.tolist()
    n = len(dates)
    idx = lab.equal_weight_index(close)
    regime_ok = idx > idx.rolling(regime_ma).mean()
    ma = close.rolling(ma_price).mean()

    equity = 1.0
    daily = {}
    holdings = {}          # sid -> {"w":weight, "peak":price, "last":price}
    cash = 1.0
    warm = max(regime_ma, ma_price)
    i = warm
    while i < n:
        d = dates[i]
        prow = close.loc[d]

        # 每日更新持股 + 停損
        if holdings:
            newh = {}
            for sid, h in holdings.items():
                px = prow.get(sid)
                if px is None or np.isnan(px) or px <= 0:
                    newh[sid] = h
                    continue
                h["w"] *= px / h["last"]
                h["last"] = px
                h["peak"] = max(h["peak"], px)
                if trail and px <= h["peak"] * (1 - trail):
                    cash += h["w"] * (1 - (0.001425 + 0.003))
                else:
                    newh[sid] = h
            holdings = newh

        # 市場濾網關 → 全部轉現金
        if not bool(regime_ok.loc[d]):
            for h in holdings.values():
                cash += h["w"] * (1 - (0.001425 + 0.003))
            holdings = {}

        # 換股
        if (i - warm) % rebal == 0 and bool(regime_ok.loc[d]):
            sc = yoy.loc[d].dropna()
            sc = sc[prow.reindex(sc.index) >= min_price]
            sc = sc[(prow > ma.loc[d]).reindex(sc.index).fillna(False)]
            if require_accel and accel is not None:
                acc_row = accel.loc[d]
                sc = sc[(acc_row > 0).reindex(sc.index).fillna(False)]
            ranked = sc.sort_values(ascending=False)
            if lowvol_tilt and vol is not None:
                # 取營收前 2*top_n，再挑其中低波動 top_n
                cand = list(ranked.head(top_n * 2).index)
                vv = vol.loc[d].reindex(cand).dropna()
                target = list(vv.sort_values().head(top_n).index)
            else:
                target = list(ranked.head(top_n).index)

            # 完整換股到等權，保留續抱名的 peak（供停損），依實際換手率計成本
            total = cash + sum(h["w"] for h in holdings.values())
            old_set = set(holdings)
            target = [s for s in target
                      if not (np.isnan(prow.get(s, np.nan)) or prow.get(s, 0) <= 0)]
            tgt_set = set(target)
            turnover = len(old_set ^ tgt_set) / (2 * max(len(old_set | tgt_set), 1))
            total_after = total * (1 - turnover * lab.ROUND_TRIP)
            per_w = total_after / max(len(target), 1)
            newh = {}
            for sid in target:
                px = prow[sid]
                peak = holdings[sid]["peak"] if sid in holdings else px
                newh[sid] = {"w": per_w, "peak": max(peak, px), "last": px}
            holdings = newh
            cash = total_after - sum(h["w"] for h in holdings.values())

        equity = cash + sum(h["w"] for h in holdings.values())
        daily[d] = equity
        i += 1

    s = pd.Series(daily)
    s.index = pd.to_datetime(s.index)
    return _m(s, name)


def _m(eq, name):
    eq = eq.dropna()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = ((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1) * 100
    dd = ((eq - eq.cummax()) / eq.cummax()).min() * 100
    r = eq.pct_change().dropna()
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
    return {"name": name, "cagr": cagr, "mdd": dd, "sharpe": sharpe, "eq": eq}


def rep(m):
    print(f"  {m['name']:<34} CAGR {m['cagr']:+6.1f}%  MDD {m['mdd']:6.1f}%  Sharpe {m['sharpe']:5.2f}")


def main():
    print("載入資料 + 建 YoY / 加速度 / 波動 panel...")
    close = lab.load_close_matrix()
    yoy, accel = build_panels(close)
    vol = close.pct_change().rolling(60).std()
    print(f"  {yoy.notna().any().sum()} 檔有營收\n")

    print("=" * 92)
    print("  月營收動能 DD 改進（真實日 mark，top50/rebal20 為底）")
    print("=" * 92)
    rep(daily_bt(close, yoy, name="基準（MA200 濾網）"))
    rep(daily_bt(close, yoy, regime_ma=100, name="A. 快濾網 MA100"))
    rep(daily_bt(close, yoy, accel=accel, require_accel=True, name="B. +營收加速度篩選"))
    rep(daily_bt(close, yoy, trail=0.20, name="C. +移動停損 20%"))
    rep(daily_bt(close, yoy, trail=0.15, name="C. +移動停損 15%"))
    rep(daily_bt(close, yoy, vol=vol, lowvol_tilt=True, name="D. +低波動傾斜"))
    print("-" * 92)
    print("  組合：")
    rep(daily_bt(close, yoy, accel=accel, require_accel=True, vol=vol, lowvol_tilt=True,
                 name="B+D 加速度+低波動"))
    rep(daily_bt(close, yoy, accel=accel, require_accel=True, trail=0.20,
                 name="B+C 加速度+停損20%"))
    rep(daily_bt(close, yoy, regime_ma=100, vol=vol, lowvol_tilt=True,
                 name="A+D 快濾網+低波動"))
    print("=" * 92)
    print("  對照：現有三策略分池 CAGR +32.4% / MDD -13.2% / Sharpe 2.22")
    print("=" * 92)


if __name__ == "__main__":
    main()
