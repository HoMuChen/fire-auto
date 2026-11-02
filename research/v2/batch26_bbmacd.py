"""第二十六批 H126：布林通道 + MACD 三條件（社群圖卡），規則見 HYPOTHESES.md 第二十六批。

用法：/usr/local/bin/python3 research/v2/batch26_bbmacd.py dev
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402


def _ind(bars):
    c = pd.Series([b["close"] for b in bars], dtype=float)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    hist = macd - macd.ewm(span=9, adjust=False).mean()
    ma = c.rolling(20).mean()
    bw = 4 * c.rolling(20).std(ddof=0) / ma
    ok = np.arange(len(c)) >= 34                      # MACD 暖機（26 + 9 − 1）
    c1 = ok & (hist.shift() <= 0) & (hist > 0)
    c2 = bw > bw.shift()
    c3 = (c > ma) & (ma > ma.shift())
    return c, hist, ma, c1, c2, c3


def _sig(buy, sell=None):
    return ["buy" if b else ("sell" if sell is not None and s else None)
            for b, s in zip(buy, sell if sell is not None else buy)]


def h126(bars):
    _, _, _, c1, c2, c3 = _ind(bars)
    return _sig((c1 & c2 & c3).values)


def h126b(bars):
    c, hist, ma, c1, c2, c3 = _ind(bars)
    return _sig((c1 & c2 & c3).values, ((hist <= 0) | (c < ma)).values)


def h126c(bars):
    _, _, _, c1, _, _ = _ind(bars)
    return _sig(c1.values)


def h126d(bars):
    _, _, _, c1, _, c3 = _ind(bars)
    return _sig((c1 & c3).values)


SPECS = {
    "H126 三條件 持有20日": {"fn": h126, "max_hold": 20, "exit_on_sell": False, "desc": "MACD柱負轉正+布林帶寬擴大+站上上彎MA20，持有20日"},
    "H126b 三條件 訊號出場": {"fn": h126b, "max_hold": 60, "exit_on_sell": True, "desc": "同H126進場；柱狀體≤0或跌破MA20出場，最長60日"},
    "H126c 只有MACD轉正 20日": {"fn": h126c, "max_hold": 20, "exit_on_sell": False, "desc": "MACD(12,26,9)柱負轉正，持有20日"},
    "H126d MACD轉正+MA20 20日": {"fn": h126d, "max_hold": 20, "exit_on_sell": False, "desc": "MACD柱負轉正+站上上彎MA20，持有20日"},
}


def tag_splits(T, end):
    """S1 大盤 > 200MA、S2 H2 ≥ 0.9；訊號日＝進場日前一個交易日（個股自己的前一根 K）。"""
    tx = pd.read_parquet(H.D / "futures/taiex.parquet")
    tx = tx.assign(date=pd.to_datetime(tx.date)).set_index("date").close.sort_index()
    bull = (tx > tx.rolling(200).mean())
    s2 = {}
    for sid in T.sid.unique():
        df = H.load_stock(sid, end)
        h2 = (df.close / df.close.rolling(252, min_periods=252).max()).shift().values   # 前一根 K 的 H2
        s2[sid] = pd.Series(h2, index=df.date)
    T["sigday"] = [s2[s].index[s2[s].index.get_loc(b) - 1] for s, b in zip(T.sid, T.buy)]
    T["S1"] = np.where(bull.reindex(T.sigday, method="ffill").values, "大盤>200MA", "大盤≤200MA")
    h2v = np.array([s2[s].get(b, np.nan) for s, b in zip(T.sid, T.buy)])
    T["S2"] = np.where(np.isnan(h2v), "H2 無值", np.where(h2v >= 0.9, "H2≥0.9", "H2<0.9"))
    return T


def split_table(T, name, base, col):
    rows = []
    for g, x in T[T.name == name].groupby(col):
        b = T[(T.name == base) & (T[col] == g)]
        bex = b.ex.mean()
        m = x.groupby(x.buy.dt.to_period("M")).ex.mean() - bex
        yr = x.groupby("year").ex.mean() - b.groupby("year").ex.mean()
        rows.append(dict(split=col, group=g, n=len(x), win=(x.ret > 0).mean() * 100, avg=x.ret.mean() * 100,
                         ex=x.ex.mean() * 100, rand_ex=bex * 100, edge=(x.ex.mean() - bex) * 100,
                         t_month=m.mean() / m.std() * np.sqrt(len(m)), yrs_pos=f"{int((yr > 0).sum())}/{yr.notna().sum()}"))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    period = sys.argv[1] if len(sys.argv) > 1 else "dev"
    T, S = H.run(SPECS, period, note="第二十六批 H126 布林+MACD 三條件")
    end = min(pd.Timestamp(H.PERIODS[period][1]), pd.Timestamp.today()).strftime("%Y-%m-%d")
    pd.set_option("display.width", 250)
    f = S.copy()
    for c in ("win", "avg", "ex", "edge"):
        f[c] = (f[c] * 100).round(2)
    print(f.round(2).to_string(index=False))
    out = H.D / "results"
    out.mkdir(exist_ok=True)
    T = tag_splits(T, end)
    T.to_parquet(out / f"b26_{period}.parquet")
    for name in ("H126 三條件 持有20日", "H126b 三條件 訊號出場"):
        sp = SPECS[name]
        base = H._base_name(sp.get("stop"), sp.get("max_hold"))
        print(f"\n== {name}（對照 {base}）")
        print(pd.concat([split_table(T, name, base, c) for c in ("S1", "S2", "group")]).round(2).to_string(index=False))
