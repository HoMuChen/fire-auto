"""第五十六批 FS10：學術因子第三輪（FACTOR_MAP.md 精選 20 個）。規則見 HYPOTHESES.md 第五十六批。

特徵函數 FNS[代號] = (fn(P), 說明)，介面同 hypotheses.FNS：回傳日頻分數（日期 × 股票），評估時取換股日前一交易日的值。
先呼叫 add_fs10(P)（需已跑 load_panel(fundamentals=True)、feature_tu.add_tu（大盤日報酬））。評估併入 feature_library.py。

leak check：.venv/bin/python research/v2/feature_fs10.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import feature_fs8 as F8  # noqa: E402
import harness as H  # noqa: E402
import xs  # noqa: E402

WIN, MIN, SG_LAG = 21, 15, 36
DEBT, STI = F8.DEBT, F8.STI
LTI = ("InvestmentAccountedForUsingEquityMethod", "FinancialAssetsAtFairvalueThroughOtherComprehensiveIncomeNonCurrent",
       "NonCurrentFinancialAssetsAtFairvalueThroughProfitOrLoss", "FinancialAssetsAtAmortizedCostNonCurrent",
       "NonCurrentAvailableForSaleFinancialAssets")
QFEATS = ("coa_gr", "col_gr", "ncoa_gr", "ncol_gr", "fnl_gr", "nfna_gr", "be_gr", "sti_gr", "ebit_sale", "ocf_at_chg",
          "opex_at", "tax_gr", "pi_nix", "at_be", "aliq_at", "dsale_dsga", "oacc_ni")


def _quarterly(s):
    q, b, c = (xs._pivot(H.D / k / f"{s}.parquet") for k in ("fin", "bs", "cf"))
    if q is None:
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

    def gr(x):
        return x / pos(x.shift(4)) - 1

    ca, cl, ncl, cash, be = g(b, "CurrentAssets"), g(b, "CurrentLiabilities"), g(b, "NoncurrentLiabilities"), g(b, "CashAndCashEquivalents"), pos(g(b, "Equity"))
    debt, sti, lti = (sum(z0(k) for k in ks) for ks in (DEBT, STI, LTI))
    coa, col = ca - cash - sti, cl - z0("ShorttermBorrowings")
    ncoa, ncol = ta - ca - lti, ncl - z0("LongtermBorrowings") - z0("BondsPayable")
    niq = g(q, "IncomeAfterTaxes").fillna(g(q, "NetIncome"))   # 2013 以前的科目名稱是 NetIncome
    ni, oi, rev, opx = ttm(niq), ttm(g(q, "OperatingIncome")), ttm(g(q, "Revenue")), ttm(g(q, "OperatingExpenses"))
    cogs, tax, pti = ttm(g(q, "CostOfGoodsSold")), ttm(g(q, "TAX")), ttm(g(q, "PreTaxIncome"))
    cfo_k = "CashFlowsFromOperatingActivities" if c is not None and "CashFlowsFromOperatingActivities" in c else "NetCashInflowFromOperatingActivities"
    cfo = ttm(single(cfo_k))
    ocf_at = cfo / ta
    return {"coa_gr": d4(coa), "col_gr": d4(col), "ncoa_gr": d4(ncoa), "ncol_gr": d4(ncol), "fnl_gr": d4(debt),
            "nfna_gr": d4(cash + sti + lti - debt), "be_gr": d4(be), "sti_gr": d4(sti),
            "ebit_sale": oi / pos(rev), "ocf_at_chg": ocf_at - ocf_at.shift(4), "opex_at": (cogs + opx) / ta,
            "tax_gr": d4(tax), "pi_nix": pos(pti) / pos(ni), "at_be": ta / be,
            "aliq_at": (cash + 0.75 * (ca - cash) + 0.5 * (ta - ca - z0("IntangibleAssets"))) / ta.shift(4),
            "dsale_dsga": gr(rev) - gr(opx), "oacc_ni": (ni - cfo) / ni.abs().where(ni != 0)}


def _sale_gr3(s):
    r = F8._rev12(s)                                   # 月頻、連續月份，索引是可得日
    if r is None:
        return None
    return r / r.shift(SG_LAG).where(r.shift(SG_LAG) > 0) - 1


def add_fs10(P):
    idx, cols = P["close"].index, P["close"].columns
    out = {k: {} for k in QFEATS + ("sale_gr3",)}
    for s in cols:
        for k, v in _quarterly(s).items():
            a = xs._align(pd.Series(v.values, index=xs._q_avail(v.index)), idx, stale=200)
            if a is not None:
                out[k][s] = a
        r = _sale_gr3(s)
        if r is not None:
            a = xs._align(r, idx)
            if a is not None:
                out["sale_gr3"][s] = a
    for k, x in out.items():
        P[f"f10_{k}"] = pd.DataFrame(x).reindex(index=idx, columns=cols)


def _roll(R, M, M1):
    k = R.notna() & M.notna() & M1.notna()
    R, M, M1 = R.where(k), M.where(k), M1.where(k)
    m = lambda x: x.rolling(WIN, min_periods=MIN).mean()   # noqa: E731
    return R, M, M1, m


def beta_dimson(P):
    """r_t = a + b0 m_t + b1 m_{t-1}：兩個係數相加（不放領先項）。"""
    R, M = F8._rm(P)
    M1 = pd.DataFrame(np.broadcast_to(P["mkt_r"].shift(1).values[:, None], R.shape), index=R.index, columns=R.columns)
    R, M, M1, m = _roll(R, M, M1)
    mr, mm, m1 = m(R), m(M), m(M1)
    s00, s11, s01 = m(M * M) - mm ** 2, m(M1 * M1) - m1 ** 2, m(M * M1) - mm * m1
    c0, c1 = m(R * M) - mr * mm, m(R * M1) - mr * m1
    det = s00 * s11 - s01 ** 2
    det = det.where(det > 1e-14)
    return ((s11 * c0 - s01 * c1) + (s00 * c1 - s01 * c0)) / det


def iskew_capm(P):
    """21 日 CAPM 殘差的偏態（母體動差，同 rolling 迴歸殘差直接算）。"""
    R, M = F8._rm(P)
    R, M, _, m = _roll(R, M, M)
    mx, my = m(R), m(M)
    mxx, mxy, myy = m(R * R), m(R * M), m(M * M)
    cxx, cxy, cyy = mxx - mx ** 2, mxy - mx * my, myy - my ** 2
    cxxx = m(R ** 3) - 3 * mx * mxx + 2 * mx ** 3
    cyyy = m(M ** 3) - 3 * my * myy + 2 * my ** 3
    cxxy = m(R * R * M) - 2 * mx * mxy - my * mxx + 2 * mx ** 2 * my
    cxyy = m(R * M * M) - 2 * my * mxy - mx * myy + 2 * mx * my ** 2
    b = cxy / cyy.where(cyy > 0)
    e2 = cxx - b * cxy
    e3 = cxxx - 3 * b * cxxy + 3 * b ** 2 * cxyy - b ** 3 * cyyy
    return e3 / e2.where(e2 > 1e-10) ** 1.5


def _q(k):
    return lambda P: P[f"f10_{k}"]


FNS = {"COA_GR": (_q("coa_gr"), "流動營運資產變化"), "COL_GR": (_q("col_gr"), "流動營運負債變化"),
       "NCOA_GR": (_q("ncoa_gr"), "非流動營運資產變化"), "NCOL_GR": (_q("ncol_gr"), "非流動營運負債變化"),
       "FNL_GR": (_q("fnl_gr"), "金融負債變化"), "NFNA_GR": (_q("nfna_gr"), "淨金融資產變化"), "BE_GR": (_q("be_gr"), "權益變化"),
       "STI_GR": (_q("sti_gr"), "短期投資變化"), "SALE_GR3": (_q("sale_gr3"), "營收3年成長"), "EBIT_SALE": (_q("ebit_sale"), "營業利益率"),
       "OCF_AT_CHG": (_q("ocf_at_chg"), "營業現金流÷資產變化"), "OPEX_AT": (_q("opex_at"), "營運槓桿"),
       "TAX_GR": (_q("tax_gr"), "所得稅費用驚喜"), "PI_NIX": (_q("pi_nix"), "稅前淨利÷淨利"), "AT_BE": (_q("at_be"), "帳面槓桿"),
       "ALIQ_AT": (_q("aliq_at"), "帳面資產流動性"), "DSALE_DSGA": (_q("dsale_dsga"), "營收成長−費用成長"),
       "OACC_NI": (_q("oacc_ni"), "應計比例"), "BETA_DIMSON": (beta_dimson, "Dimson beta（21日）"),
       "ISKEW_CAPM": (iskew_capm, "CAPM特質偏態（21日）")}
NAMES = list(FNS)
CAT = {**{f: "財報" for f in NAMES}, "SALE_GR3": "營收", "BETA_DIMSON": "價格", "ISKEW_CAPM": "價格"}
DAILY = ("SALE_GR3", "BETA_DIMSON", "ISKEW_CAPM")


def _check_iskew(P, sid="2330", day="2018-06-29"):
    R, M = F8._rm(P)
    r, mk = R[sid].loc[:day].iloc[-WIN:], M[sid].loc[:day].iloc[-WIN:]
    k = r.notna() & mk.notna()
    x, y = r[k].values, mk[k].values
    b, a = np.polyfit(y, x, 1)
    e = x - a - b * y
    direct = (e ** 3).mean() / (e ** 2).mean() ** 1.5
    return direct, iskew_capm(P)[sid].loc[day]


if __name__ == "__main__":
    import feature_tu as TU
    P = xs.load_panel("2019-12-31", fundamentals=True)
    for add in (xs.add_v16, TU.add_tu, add_fs10):
        add(P)
    print("ISKEW_CAPM 直接迴歸 vs 動差公式（2330 2018-06-29）：", _check_iskew(P), flush=True)
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
