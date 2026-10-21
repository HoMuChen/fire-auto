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

# ───────── 第二批 ─────────
def h10_rev_high(P):
    return P["rev3_high"]


def h11_ey(P):
    return (1 / P["per"]).where(P["per"] > 0)


def h12_rev_persist(P):
    return P["rev_pos6"] + 0.01 * P["rev_yoy3"]


def h13_max(P):
    r = P["close"].pct_change(fill_method=None)
    return -r.rolling(21, min_periods=15).max()


def h14_h2_h7(P):
    a = h2_hi52(P).rank(axis=1, pct=True)
    b = h7_rev_yoy(P).rank(axis=1, pct=True)
    return (a + b) / 2


FNS.update({"H10": (h10_rev_high, "營收創新高"), "H11": (h11_ey, "盈餘殖利率"), "H12": (h12_rev_persist, "營收YoY持續性"),
            "H13": (h13_max, "彩券效應反向"), "H14": (h14_h2_h7, "H2×H7合成(事後)")})


def _big_in_topq(score, P, n_hint=None):
    """分數前 20% 中，近 60 日平均成交金額越大分數越高（其餘設 NaN）。topn 取前 20 即「前 20% 裡最大的 20 檔」。"""
    elig = P["liq"] & (P["nbars"] >= 250)
    s = score.where(elig)
    q = s.rank(axis=1, pct=True)
    m = P["money"].rolling(60, min_periods=40).mean()
    return m.where(q > 0.8)


def h15_rev_high_big(P):
    return _big_in_topq(h10_rev_high(P), P)


def h16_rev_persist_big(P):
    return _big_in_topq(h12_rev_persist(P), P)


def h17_rev_yoy_big(P):
    return _big_in_topq(h7_rev_yoy(P), P)


FNS.update({"H15": (h15_rev_high_big, "營收創新高・大型"), "H16": (h16_rev_persist_big, "營收持續性・大型"),
            "H17": (h17_rev_yoy_big, "營收YoY・大型")})


# ───────── 第四批（籌碼；dev 自 2012-07）─────────
def _vol_sum(P, n):
    return P["volume"].rolling(n, min_periods=int(n * 0.8)).sum()


def h22_trust(P):
    return P["inst_t"].rolling(20, min_periods=16).sum() / _vol_sum(P, 20)


def h23_foreign(P):
    return P["inst_f"].rolling(20, min_periods=16).sum() / _vol_sum(P, 20)


def h24_margin_cut(P):
    m = P["margin_bal"]
    return -(m / m.shift(20) - 1).where(m >= 100)


def h25_inst60(P):
    tot = P["inst_f"] + P["inst_t"] + P["inst_d"]
    return tot.rolling(60, min_periods=48).sum() / _vol_sum(P, 60)


FNS.update({"H22": (h22_trust, "投信認養20日"), "H23": (h23_foreign, "外資買超20日"),
            "H24": (h24_margin_cut, "融資減肥20日"), "H25": (h25_inst60, "法人合計60日")})
CHIP = {"H22", "H23", "H24", "H25"}


# ───────── 第五批（事後複合）─────────
def _pct(x):
    return x.rank(axis=1, pct=True)


def h29_double_high(P):
    return (_pct(h2_hi52(P)) + _pct(h10_rev_high(P))) / 2


def h30_rev_combo(P):
    return (_pct(h7_rev_yoy(P)) + _pct(h10_rev_high(P)) + _pct(h12_rev_persist(P))) / 3


FNS.update({"H29": (h29_double_high, "雙新高(事後)"), "H30": (h30_rev_combo, "營收複合(事後)")})


# ───────── 第五批（財報；事前）─────────
def h26_roe(P):
    return P["roe_ttm"]


def h27_gpa(P):
    return P["gpa_ttm"]


def h28_sue(P):
    return P["sue"].replace([np.inf, -np.inf], np.nan)


FNS.update({"H26": (h26_roe, "ROE(TTM)"), "H27": (h27_gpa, "毛利資產比"), "H28": (h28_sue, "盈餘驚喜SUE")})
FIN = {"H26": "2013-05-01", "H27": "2013-05-01", "H28": "2011-06-01"}


# ───────── 第七批（新類型）─────────
def h35_overnight(P):
    on = np.log(P["open"] / P["close"].shift(1))
    return on.rolling(60, min_periods=48).sum()


def h36_intraday_rev(P):
    intra = np.log(P["close"] / P["open"])
    return -intra.rolling(60, min_periods=48).sum()


def h38_seasonal(P):
    O = P["open"]
    idx = O.index
    rd = xs.rebalance_dates(idx, idx[0], idx[-1])
    win = pd.DataFrame({d: O.loc[d2] / O.loc[d] - 1 for d, d2 in zip(rd[:-1], rd[1:])}).T   # 每個換股窗口報酬（已實現才用）
    out = pd.DataFrame(np.nan, index=idx, columns=O.columns)
    for i, d in enumerate(rd):
        past = [rd[i - 12 * k] for k in range(1, 6) if i - 12 * k >= 0 and rd[i - 12 * k] in win.index]
        if len(past) < 3:
            continue
        k = idx.get_loc(d)
        if k == 0:
            continue
        w = win.loc[past]
        sc = w.mean().where(w.notna().sum() >= 3)
        out.iloc[k - 1] = sc.values          # 放在換股日前一日（backtest 用 shift(1) 取用）
    return out


def h39_ind_rev(P):
    u = pd.read_csv(xs.H.D / "universe.csv", dtype=str).set_index("stock_id").industry
    ind = u.reindex(P["close"].columns)
    C = P["close"].ffill()
    r = (C / C.shift(21) - 1).where(P["liq"] & (P["nbars"] >= 250))
    out = pd.DataFrame(np.nan, index=r.index, columns=r.columns)
    for g, cols in ind.dropna().groupby(ind.dropna()).groups.items():
        x = r[list(cols)]
        cnt = x.notna().sum(axis=1)
        out[list(cols)] = (-(x.sub(x.mean(axis=1), axis=0))).where(cnt >= 5, axis=0)
    return out


FNS.update({"H35": (h35_overnight, "隔夜動能"), "H36": (h36_intraday_rev, "盤中反向"),
            "H38": (h38_seasonal, "同窗季節性"), "H39": (h39_ind_rev, "產業內相對反轉")})


if __name__ == "__main__":
    period, keys = sys.argv[1], sys.argv[2:]
    n = int(next((a.split("=")[1] for a in keys if a.startswith("--topn=")), 30))
    keys = [k for k in keys if not k.startswith("--")]
    chip = bool(CHIP & set(keys))
    fin = [FIN[k] for k in keys if k in FIN]
    start = "2012-07-01" if chip else (max(fin) if fin else None)
    res, P = xs.run({k: FNS[k] for k in keys}, period, note="事前登記假設", topn=n, start=start, chips=chip, fin=bool(fin))
    rows = []
    for k, (R, st) in res.items():
        for col in ("top_q", "topn"):
            rows.append(dict(H=k, name=FNS[k][1], port=col, **{a: b for a, b in st[col].items()}))
    df = pd.DataFrame(rows)
    for c in ("CAGR", "bench_CAGR", "ex_ann", "MDD", "turnover"):
        df[c] = (df[c] * 100).round(1)
    pd.set_option("display.width", 200)
    print(df.round(2).to_string(index=False))
    print("\n分位年化超額（Q1 最低分 → Q5 最高分，%）")
    for k, (R, st) in res.items():
        print(k, FNS[k][1], {q: round(v * 100, 1) for q, v in st["Q_ex_ann"].items()})
