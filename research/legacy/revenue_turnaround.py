"""
營收轉機（Revenue Turnaround）— 買基本面『拐點』而非『領先者』，找更低相關

動能買營收成長最高的（領先者）。轉機反過來買 YoY 加速度最大（剛從谷底翻揚）的，
偏落後/контра股，時機不同 → 可能與現有組合、與營收動能都更低相關。
用同一 daily_bt 引擎，把排名分數換成『YoY 加速度』。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import strat_lab as lab            # noqa: E402
import revenue_improve as ri        # noqa: E402
import portfolio_sim as psim        # noqa: E402


def corr_to_base(eq):
    dates, vals = psim.combined_daily_equity()
    base = pd.Series(vals, index=pd.to_datetime(dates)).sort_index()
    base = base[~base.index.duplicated(keep="last")]
    common = eq.index.intersection(base.index)
    a = eq.reindex(common).ffill().pct_change().dropna()
    b = base.reindex(common).ffill().pct_change().dropna()
    ix = a.index.intersection(b.index)
    return np.corrcoef(a.reindex(ix), b.reindex(ix))[0, 1]


def main():
    print("載入 + 建 panel...")
    close = lab.load_close_matrix()
    yoy, accel = ri.build_panels(close)
    vol = close.pct_change().rolling(60).std()

    print("=" * 90)
    print("  營收轉機（買 YoY 加速度最大）vs 營收動能（買 YoY 最高）")
    print("=" * 90)
    # 轉機：用加速度排名（把 accel 當 yoy 傳入 daily_bt）
    for tn in [40, 60]:
        m = ri.daily_bt(close, accel, top_n=tn, rebal=20, name=f"轉機 top{tn}")
        c = corr_to_base(m["eq"])
        ri.rep(m)
        print(f"       └ 與三策略相關 {c:.2f}")
    # 轉機 + 低波動
    m = ri.daily_bt(close, accel, vol=vol, lowvol_tilt=True, top_n=50, rebal=20,
                    name="轉機+低波動 top50")
    c = corr_to_base(m["eq"])
    ri.rep(m)
    print(f"       └ 與三策略相關 {c:.2f}")
    print("-" * 90)
    # 對照：動能的相關
    m2 = ri.daily_bt(close, yoy, top_n=50, rebal=20, name="動能 top50（對照）")
    c2 = corr_to_base(m2["eq"])
    ri.rep(m2)
    print(f"       └ 與三策略相關 {c2:.2f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
