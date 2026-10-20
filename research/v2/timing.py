"""H20 大盤 200 日線擇時、H21 H2 + 大盤濾網（風險控制型）。月頻：換股日用前一交易日 TR 指數 vs 200MA 判斷。

H20 對照＝流動股等權一直持有；H21 對照＝H2 top20 一直持有。
切到現金：當期報酬 0，並扣一次賣出成本（手續費+稅）；回場的買進成本已含在 topn 換手成本內（略保守）。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402
import xs  # noqa: E402
from hypotheses import h2_hi52  # noqa: E402

SELL = H.SELL_FEE + H.SELL_TAX


def on_signal(dates):
    tr = pd.read_parquet(H.D / "taiex_tr.parquet").assign(date=lambda x: pd.to_datetime(x.date)).set_index("date").price
    ok = (tr > tr.rolling(200).mean()).shift(1)            # 前一交易日收盤
    return ok.reindex(dates, method="ffill").fillna(True).astype(bool)


def apply(r, on, cost_in_cash):
    out, prev = [], True
    for d, x in r.items():
        if on[d]:
            out.append(x)
        else:
            out.append(-(SELL if prev else 0) if cost_in_cash else 0.0)
        prev = on[d]
    return pd.Series(out, index=r.index)


def metr(r):
    eq = (1 + r).cumprod()
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    return dict(CAGR=eq.iloc[-1] ** (1 / yrs) - 1, MDD=(eq / eq.cummax() - 1).min(), Sharpe=r.mean() / r.std() * np.sqrt(12),
                in_mkt=None)


if __name__ == "__main__":
    period = sys.argv[1] if len(sys.argv) > 1 else "dev"
    p0, p1 = H.PERIODS[period]
    P = xs.load_panel(p1)
    R = xs.backtest(h2_hi52(P), P, p0, p1, topn=20).dropna(subset=["topn"])
    on = on_signal(R.index)
    rows = []
    for name, base in (("H20 大盤擇時(流動股等權)", R.bench), ("H21 H2+大盤濾網", R.topn)):
        a, b = metr(apply(base, on, True)), metr(base)
        a["in_mkt"] = on.mean()
        rows += [dict(策略=name, 版本="擇時", **a), dict(策略=name, 版本="對照(一直持有)", **b)]
        with open(H.LOG, "a") as f:
            f.write(json.dumps(dict(ts=time.strftime("%Y-%m-%d %H:%M:%S"), kind="timing", period=period, strategy=name,
                                    timed=a, base=b), ensure_ascii=False, default=float) + "\n")
    df = pd.DataFrame(rows)
    for c in ("CAGR", "MDD", "in_mkt"):
        df[c] = df[c].astype(float) * 100
    print(f"[{period}] {R.index[0].date()}~{R.index[-1].date()}（月線指標）")
    print(df.round(2).to_string(index=False))
    off = on[~on]
    print("空手月份:", [d.strftime("%Y-%m") for d in off.index])
