"""第五十七批 FS11：學術因子第四輪（FACTOR_MAP.md 剩下的 13 個）。規則見 HYPOTHESES.md 第五十七批。

特徵函數 FNS[代號] = (fn(P), 說明)，介面同 hypotheses.FNS：回傳日頻分數（日期 × 股票），評估時取換股日前一交易日的值。
先呼叫 add_fs11(P)（需已跑 load_panel(fundamentals=True)、xs.add_v16（市值）、feature_tu.add_tu（大盤日報酬）、
feature_fs8.add_fs8（淨負債））。COMBO（QMJ 拆半）照 feature_fs9 的格式，由 feature_library.py 在換股日的股票池裡算。

leak check：.venv/bin/python research/v2/feature_fs11.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import feature_fs10 as F10  # noqa: E402
import harness as H  # noqa: E402
import xs  # noqa: E402

R_E, ROE_M, ROE_P, G_M, G_P, DUR_T = 0.12, 0.12, 0.57, 0.06, 0.24, 10   # Dechow 等 2004
DIV_TA, CORR_WIN, CORR_MIN, CORR_K = 0.06, 1260, 750, 3
DEBT, STI, LTI = F10.DEBT, F10.STI, F10.LTI
QFEATS = ("lti_gr", "cowc_gr", "nncoa_gr", "debt_gr3", "capex_abn", "ebit_bev", "sale_bev", "bev",
          "dur_num", "dur_pv", "be", "roe_avg", "ni", "ta")


def _dur(be0, roe0, g0):
    """Dechow 等 2004：10 年 ROE、成長回歸長期平均，回傳（Σ t·CDₜ／(1+r)ᵗ, Σ CDₜ／(1+r)ᵗ）。輸入可為 Series 或純量。"""
    be, roe, g, num, pv = be0, roe0, g0, 0.0, 0.0
    for t in range(1, DUR_T + 1):
        roe = ROE_M + ROE_P * (roe - ROE_M)
        g = G_M + G_P * (g - G_M)
        cd = be * (roe - g)
        be = be * (1 + g)
        num = num + t * cd / (1 + R_E) ** t
        pv = pv + cd / (1 + R_E) ** t
    return num, pv


def _quarterly(s):
    q, b, c = (xs._pivot(H.D / k / f"{s}.parquet") for k in ("fin", "bs", "cf"))
    if q is None or b is None:
        return {}
    have = [x for x in (q, b, c) if x is not None]
    qi = pd.date_range(min(x.index.min() for x in have), max(x.index.max() for x in have), freq="QE")
    nan = pd.Series(np.nan, index=qi)

    def g(df, k):
        return df[k].reindex(qi) if df is not None and k in df else nan

    def single(k):                                     # cf 年初累計 → 單季
        cum = g(c, k)
        return cum.where(qi.quarter == 1, cum - cum.shift(1))

    def ttm(x):
        return x.rolling(4, min_periods=4).sum()

    def pos(x):
        return x.where(x > 0)

    ta = pos(g(b, "TotalAssets"))
    zero = pd.Series(0.0, index=qi).where(ta.notna())

    def z0(k):                                         # 金額為 0 的科目不列 → 有總資產時補 0
        return g(b, k).fillna(zero)

    def d4(x):                                         # 四季變化 ÷ 四季前 TA
        return (x - x.shift(4)) / ta.shift(4)

    ca, cl, ncl, cash, be = g(b, "CurrentAssets"), g(b, "CurrentLiabilities"), g(b, "NoncurrentLiabilities"), g(b, "CashAndCashEquivalents"), pos(g(b, "Equity"))
    debt, sti, lti = (sum(z0(k) for k in ks) for ks in (DEBT, STI, LTI))
    coa, col = ca - cash - sti, cl - z0("ShorttermBorrowings")
    ncoa, ncol = ta - ca - lti, ncl - z0("LongtermBorrowings") - z0("BondsPayable")
    ni = ttm(g(q, "IncomeAfterTaxes").fillna(g(q, "NetIncome")))   # 2013 以前的科目名稱是 NetIncome
    oi, rev = ttm(g(q, "OperatingIncome")), pos(ttm(g(q, "Revenue")))
    ce = -ttm(single("PropertyAndPlantAndEquipment")) / rev
    ce_avg = (ce.shift(4) + ce.shift(8) + ce.shift(12)) / 3
    bev = pos(be + debt - cash)
    num, pv = _dur(be, ni / be.shift(4), rev / rev.shift(4) - 1)
    return {"lti_gr": d4(lti), "cowc_gr": d4(coa - col), "nncoa_gr": d4(ncoa - ncol),
            "debt_gr3": debt / pos(debt.shift(12)) - 1, "capex_abn": ce / pos(ce_avg) - 1,
            "ebit_bev": oi / bev, "sale_bev": rev / bev, "bev": bev, "dur_num": num, "dur_pv": pv,
            "be": be, "roe_avg": ni / ((be + be.shift(4)) / 2), "ni": ni, "ta": ta}


def add_fs11(P):
    idx, cols = P["close"].index, P["close"].columns
    out = {k: {} for k in QFEATS}
    for s in cols:
        for k, v in _quarterly(s).items():
            a = xs._align(pd.Series(v.values, index=xs._q_avail(v.index)), idx, stale=200)
            if a is not None:
                out[k][s] = a
    for k, x in out.items():
        P[f"f11_{k}"] = pd.DataFrame(x).reindex(index=idx, columns=cols)


def _me(P):
    return P["mv"].where(P["mv"] > 0)


def bev_mev(P):
    mev = _me(P) + P["f8_net_debt"]
    return P["f11_bev"] / mev.where(mev > 0)


def eq_dur(P):
    me = _me(P)
    return P["f11_dur_num"] / me + (DUR_T + (1 + R_E) / R_E) * (1 - P["f11_dur_pv"] / me)


def ival_me(P):
    me, B, roe, ni = _me(P), P["f11_be"], P["f11_roe_avg"], P["f11_ni"]
    div = P["dy"] / 100 * me
    k = (div / ni).where(ni > 0, div / (DIV_TA * P["f11_ta"])).clip(0, 1)
    B1 = B * (1 + roe * (1 - k))
    return (B + (roe - R_E) * B / (1 + R_E) + (roe - R_E) * B1 / ((1 + R_E) * R_E)) / me


def _r3(P):
    C = P["close"].where(P["close"] > 0)
    R = np.log(C).diff(CORR_K)
    m = np.log1p(P["mkt_r"]).rolling(CORR_K).sum()
    M = pd.DataFrame(np.broadcast_to(m.values[:, None], R.shape), index=R.index, columns=R.columns)
    k = R.notna() & M.notna()
    return R.where(k), M.where(k)


def corr_1260d(P):
    """3 日重疊對數報酬，近 1260 日（至少 750 天）與大盤的相關係數。"""
    R, M = _r3(P)
    m = lambda x: x.rolling(CORR_WIN, min_periods=CORR_MIN).mean()   # noqa: E731
    mr, mm = m(R), m(M)
    vr, vm = m(R * R) - mr ** 2, m(M * M) - mm ** 2
    return (m(R * M) - mr * mm) / np.sqrt(vr.where(vr > 1e-12) * vm.where(vm > 1e-12))


def _q(k):
    return lambda P: P[f"f11_{k}"]


FNS = {"LTI_GR": (_q("lti_gr"), "長期投資變化"), "COWC_GR": (_q("cowc_gr"), "淨流動營運資金變化"),
       "NNCOA_GR": (_q("nncoa_gr"), "淨非流動營運資產變化"), "DEBT_GR3": (_q("debt_gr3"), "負債3年成長"),
       "CAPEX_ABN": (_q("capex_abn"), "異常資本支出"), "EBIT_BEV": (_q("ebit_bev"), "淨營運資產報酬"),
       "SALE_BEV": (_q("sale_bev"), "淨營運資產週轉"), "BEV_MEV": (bev_mev, "企業帳面市值比"),
       "EQ_DUR": (eq_dur, "權益存續期間"), "IVAL_ME": (ival_me, "內含價值÷市值"), "CORR_1260D": (corr_1260d, "與大盤相關（5年）")}
COMBO = {"QMJ_PROF": ([("H27", 1), ("H26", 1), ("OP_AT", 1), ("OCF_AT", 1), ("GM", 1), ("TACC", -1)], 3, None, "QMJ：獲利"),
         "QMJ_SAFETY": ([("BETA", -1), ("LEV", -1), ("O_SCORE", -1), ("Z_SCORE", 1), ("NI_VOL", -1)], 3, None, "QMJ：安全")}
NAMES = list(FNS) + list(COMBO)
CAT = {**{f: "財報" for f in FNS}, **{f: "估值" for f in ("BEV_MEV", "EQ_DUR", "IVAL_ME")}, "CORR_1260D": "價格",
       **{f: "組合" for f in COMBO}}
DAILY = ("BEV_MEV", "EQ_DUR", "IVAL_ME", "CORR_1260D")


def _check_dur(P, sid="2330", day="2018-06-29"):
    """逐年迴圈（純量）直接算 EQ_DUR，和向量化結果比對。"""
    q = _quarterly(sid)
    av = pd.Series(q["be"].index, index=xs._q_avail(q["be"].index))
    qd = av[av.index <= pd.Timestamp(day)].iloc[-1]
    be, ni, be4 = q["be"][qd], q["ni"][qd], q["be"].shift(4)[qd]
    fin = xs._pivot(H.D / "fin" / f"{sid}.parquet")
    rv = fin["Revenue"].reindex(q["be"].index).rolling(4, min_periods=4).sum()
    g0 = rv[qd] / rv.shift(4)[qd] - 1
    b, roe, g, num, pv = be, ni / be4, g0, 0.0, 0.0
    for t in range(1, DUR_T + 1):
        roe = ROE_M + ROE_P * (roe - ROE_M)
        g = G_M + G_P * (g - G_M)
        cd = b * (roe - g)
        b *= 1 + g
        num += t * cd / 1.12 ** t
        pv += cd / 1.12 ** t
    me = P["mv"][sid].loc[:day].iloc[-1]
    return num / me + (DUR_T + 1.12 / 0.12) * (1 - pv / me), eq_dur(P)[sid].loc[:day].iloc[-1]


def _check_corr(P, sid="2330", day="2018-06-29"):
    R, M = _r3(P)
    r, m = R[sid].loc[:day].iloc[-CORR_WIN:], M[sid].loc[:day].iloc[-CORR_WIN:]
    k = r.notna() & m.notna()
    return np.corrcoef(r[k], m[k])[0, 1], corr_1260d(P)[sid].loc[day]


if __name__ == "__main__":
    import feature_fs8 as F8
    import feature_tu as TU
    P = xs.load_panel("2019-12-31", fundamentals=True)
    for add in (xs.add_v16, TU.add_tu, F8.add_fs8, add_fs11):
        add(P)
    print("EQ_DUR 逐年迴圈 vs 向量化（2330 2018-06-29）：", _check_dur(P), flush=True)
    print("CORR_1260D numpy vs 滾動動差（2330 2018-06-29）：", _check_corr(P), flush=True)
    for f in DAILY:
        print(f, "leak_check_xs:", xs.leak_check_xs(FNS[f][0], P) or "OK", flush=True)
    for f, (fn, _) in FNS.items():
        v = fn(P).replace([np.inf, -np.inf], np.nan)
        y = v.loc["2018"].where(P["liq"].loc["2018"])
        ok = v.notna().sum(axis=1) >= 100
        print(f"{f:<11} 池內有值檔數（2018 中位數）{int(y.notna().sum(axis=1).median())}  首個 ≥100 檔 "
              f"{ok.idxmax().date() if ok.any() else '—'}  2018 分位 p1/p50/p99 "
              f"{np.nanpercentile(y, 1):.4g}/{np.nanpercentile(y, 50):.4g}/{np.nanpercentile(y, 99):.4g}  "
              f"0 的比例 {float((y == 0).sum().sum() / max(y.notna().sum().sum(), 1)):.0%}", flush=True)
    lq = P["liq"].loc["2018"].fillna(False).astype(bool)
    dy = P["dy"].loc["2018"]
    print(f"殖利率池內缺值比例（2018）{float((dy.isna() & lq).sum().sum() / lq.sum().sum()):.1%}、"
          f"為 0 的比例 {float(((dy == 0) & lq).sum().sum() / lq.sum().sum()):.1%}")
