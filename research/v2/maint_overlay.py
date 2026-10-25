"""H102：H2 top20 100% 照常，大盤融資維持率收盤 < 140% → 次日開盤加 50% 名目小台指多單，收盤 > 160% → 次日開盤平掉。
對照 H2 100%。日線淨值，只跑 dev/val。用法：python3 research/v2/maint_overlay.py dev
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402
import hypotheses as Hy  # noqa: E402
import mtx  # noqa: E402
import short_hedge as SH  # noqa: E402
import threshold as TH  # noqa: E402
import xs  # noqa: E402

N, W = 20, 0.5


def episodes(flag):
    """回傳 [(起日, 迄日)]（flag=1 的連續區段）"""
    out, start = [], None
    for d, v in flag.items():
        if v and start is None:
            start = d
        if not v and start is not None:
            out.append((start, d)); start = None
    if start is not None:
        out.append((start, flag.index[-1]))
    return out


if __name__ == "__main__":
    period = sys.argv[1] if len(sys.argv) > 1 else "dev"
    if period == "holdout":
        raise PermissionError("holdout 不在此跑")
    p0, p1 = H.PERIODS[period]
    p0 = max(p0, "2011-01-01")
    P = xs.load_panel(p1)
    O = P["open"]
    rd = xs.rebalance_dates(O.index, pd.Timestamp(p0), pd.Timestamp(p1))
    elig = P["liq"].shift(1).fillna(False).astype(bool) & (P["nbars"].shift(1) >= 250) & O.notna()
    sc = Hy.h2_hi52(P).shift(1)
    longs = {}
    for d in rd:
        s = sc.loc[d][elig.loc[d]].dropna()
        longs[d] = list(s.sort_values(ascending=False, kind="stable").head(N).index)
    L = SH.leg_daily(P, longs, rd)
    rL = L.pct_change().fillna(0)

    F = mtx.continuous(p1)
    flag = pd.Series(mtx.maint_flag(F.index, 140, 160), index=F.index)
    n, d = mtx.from_signal(W * flag.values)
    rf, tv = mtx.run_pos(F, n, d)
    rf = rf.reindex(rL.index).fillna(0)
    flag_L = flag.reindex(rL.index).fillna(0)

    cases = {"對照 H2 100%": (1 + rL).cumprod(),
             "H102 H2 100% + 斷頭區加 50% 小台指": (1 + rL + rf).cumprod(),
             "（參考）疊加層單獨（50% 小台指，只在斷頭區）": (1 + rf).cumprod()}
    rows = []
    for k, s in cases.items():
        m = TH.metr(s)
        by = s.resample("YE").last()
        by = pd.concat([pd.Series([s.iloc[0]], index=[s.index[0]]), by]).pct_change().dropna()
        rows.append(dict(策略=k, CAGR=m["CAGR"] * 100, MDD=m["MDD"] * 100, Sharpe=m["Sharpe"],
                         **{str(i.year): v * 100 for i, v in by.items()}))
        with open(H.LOG, "a") as f:
            f.write(json.dumps(dict(ts=time.strftime("%Y-%m-%d %H:%M:%S"), kind="overlay", period=period, strategy=k,
                                    note="第十九批 H102 維持率斷頭區加碼（事前登記）", **m), ensure_ascii=False, default=float) + "\n")
    pd.set_option("display.width", 250)
    print(f"[{period}] {rL.index[0].date()}~{rL.index[-1].date()}  斷頭區天數 {int(flag_L.sum())} / {len(flag_L)}")
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    print("\n斷頭區事件（進場日~出場日，期間 H2 報酬 / 小台指報酬）")
    S = F.S.reindex(rL.index).ffill()
    for a, b in episodes(flag_L):
        ia, ib = rL.index.get_loc(a), rL.index.get_loc(b)
        print(f"  {a.date()} ~ {b.date()}  {ib-ia:3d} 天  H2 {L.iloc[ib]/L.iloc[ia]-1:+.1%}  MTX {S.iloc[ib]/S.iloc[ia]-1:+.1%}")
