"""HYPOTHESES.md 登記的分數函數（參數照登記，不調）。用法：python3 research/v2/hypotheses.py dev H1 H2 ..."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import xs  # noqa: E402


def h1_mom12_1(P):
    C = P["close"].ffill()
    return C.shift(21) / C.shift(252) - 1


def h2_hi52(P):
    C = P["close"]
    return C / C.rolling(252, min_periods=200).max()


def h3_lowvol(P):
    return -np.log(P["close"]).diff().rolling(250, min_periods=200).std()


def h4_rev1m(P):
    C = P["close"].ffill()
    return -(C / C.shift(21) - 1)


def h5_lowpbr(P):
    return -P["pbr"].where(P["pbr"] > 0)


def h6_dy(P):
    return P["dy"]


def h7_rev_yoy(P):
    return P["rev_yoy"]


def h8_rev_price(P):
    C = P["close"]
    return P["rev_yoy"].where(C > C.rolling(60, min_periods=60).mean())


def h9_amihud(P):
    r = P["close"].pct_change(fill_method=None).abs()
    return (r / P["money"].where(P["money"] > 0)).rolling(250, min_periods=200).mean()


FNS = {"H1": (h1_mom12_1, "12-1 動能"), "H2": (h2_hi52, "52週高點接近度"), "H3": (h3_lowvol, "低波動"),
       "H4": (h4_rev1m, "短期反轉"), "H5": (h5_lowpbr, "低PBR"), "H6": (h6_dy, "高殖利率"),
       "H7": (h7_rev_yoy, "月營收YoY"), "H8": (h8_rev_price, "營收YoY+價>60MA"), "H9": (h9_amihud, "Amihud非流動性")}

if __name__ == "__main__":
    period, keys = sys.argv[1], sys.argv[2:]
    res, P = xs.run({k: FNS[k] for k in keys}, period, note="事前登記假設")
    rows = []
    for k, (R, st) in res.items():
        for col in ("top_q", "top30"):
            rows.append(dict(H=k, name=FNS[k][1], port=col, **{a: b for a, b in st[col].items()}))
    df = pd.DataFrame(rows)
    for c in ("CAGR", "bench_CAGR", "ex_ann", "MDD", "turnover"):
        df[c] = (df[c] * 100).round(1)
    pd.set_option("display.width", 200)
    print(df.round(2).to_string(index=False))
    print("\n分位年化超額（Q1 最低分 → Q5 最高分，%）")
    for k, (R, st) in res.items():
        print(k, FNS[k][1], {q: round(v * 100, 1) for q, v in st["Q_ex_ann"].items()})
