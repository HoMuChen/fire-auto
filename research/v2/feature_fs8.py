"""第四十六批 FS8：學術因子對照表（FACTOR_MAP.md）的 20 個候選。規則見 HYPOTHESES.md 第四十六批。

特徵函數 FNS[代號] = (fn(P), 說明)，介面同 hypotheses.FNS：回傳日頻分數（日期 × 股票），評估時取換股日前一交易日的值。
先呼叫 add_fs8(P)（需已跑 load_panel(fundamentals=True)、xs.add_v16（市值）、feature_tu.add_tu（大盤日報酬）、
feature_fs6.add_fs6（NET_ISSUE））載入財報衍生值與近 12 月營收。評估併入 feature_library.py。

leak check：.venv/bin/python research/v2/feature_fs8.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402
import xs  # noqa: E402

MOM_S, MOM_L, HL_WIN, HL_MIN, BD_WIN, BD_MIN, CS_WIN, CS_MIN = 126, 252, 21, 15, 250, 80, 21, 15
STI = ("CurrentFinancialAssetsAtFairvalueThroughProfitOrLoss", "FinancialAssetsAtFairvalueThroughOtherComprehensiveIncome",
       "FinancialAssetsAtAmortizedCost", "CurrentAvailableForSaleFinancialAssets", "CurrentHeldToMaturityFinancialAssets")
DEBT = ("ShorttermBorrowings", "LongtermBorrowings", "BondsPayable")
QFEATS = ("noa", "noa_gr", "tacc", "cop", "ope", "z_part", "tl", "o_score", "cash_at", "ebitda", "net_debt", "fcf",
          "capx_gr2", "ppeinv", "dsale_dinv", "roe_chg")


def _quarterly(s):
    q, b, c = (xs._pivot(H.D / k / f"{s}.parquet") for k in ("fin", "bs", "cf"))
    if b is None or q is None:
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

    tl, ca, cl = g(b, "Liabilities"), g(b, "CurrentAssets"), g(b, "CurrentLiabilities")
    cash, be = g(b, "CashAndCashEquivalents"), pos(g(b, "Equity"))
    debt = sum(z0(k) for k in DEBT)
    sti = sum(z0(k) for k in STI)
    ni, oi, rev = ttm(g(q, "IncomeAfterTaxes")), ttm(g(q, "OperatingIncome")), ttm(g(q, "Revenue"))
    cfo_k = "CashFlowsFromOperatingActivities" if c is not None and "CashFlowsFromOperatingActivities" in c else "NetCashInflowFromOperatingActivities"
    cfo, cfi = ttm(single(cfo_k)), ttm(single("CashProvidedByInvestingActivities"))
    capex, dep, amo = -ttm(single("PropertyAndPlantAndEquipment")), ttm(single("Depreciation")), ttm(single("AmortizationExpense"))
    intr = ttm(single("InterestExpense"))
    ar, inv, ap, ppe = g(b, "AccountsReceivableNet"), g(b, "Inventories"), g(b, "AccountsPayable"), g(b, "PropertyPlantAndEquipment")
    inv0 = z0("Inventories")
    noa = (ta - cash - sti) - (tl - debt)
    ta4 = ta.shift(4)
    o = (-1.32 - 0.407 * np.log(ta) + 6.03 * tl / ta - 1.43 * (ca - cl) / ta + 0.0757 * cl / pos(ca)
         - 1.72 * (tl > ta).astype(float).where(tl.notna() & ta.notna()) - 2.37 * ni / ta
         - 1.83 * (ttm(g(q, "PreTaxIncome")) + dep) / pos(tl)
         + 0.285 * ((ni < 0) & (ni.shift(4) < 0)).astype(float).where(ni.notna() & ni.shift(4).notna())
         - 0.521 * (ni - ni.shift(4)) / pos(ni.abs() + ni.shift(4).abs()))
    roe_q = g(q, "IncomeAfterTaxes") / be.shift(1)
    return {"noa": noa / ta4, "noa_gr": (noa - noa.shift(4)) / ta4,
            "tacc": (ni - cfo - cfi) / ((ta + ta4) / 2),
            "cop": (oi - (ar - ar.shift(4)) - (inv - inv.shift(4)) + (ap - ap.shift(4))) / ta,
            "ope": (oi - intr) / be,
            "z_part": 1.2 * (ca - cl) / ta + 1.4 * g(b, "RetainedEarnings") / ta + 3.3 * oi / ta + 1.0 * rev / ta,
            "tl": pos(tl), "o_score": o, "cash_at": cash / ta,
            "ebitda": oi + dep + amo, "net_debt": debt - cash, "fcf": cfo - capex,
            "capx_gr2": capex / pos(capex.shift(8)) - 1,
            "ppeinv": ((ppe + inv0) - (ppe + inv0).shift(4)) / ta4,
            "dsale_dinv": (rev / pos(rev.shift(4)) - 1) - (inv / pos(inv.shift(4)) - 1),
            "roe_chg": roe_q - roe_q.shift(4)}


def _rev12(s):
    f = H.D / "revenue" / f"{s}.parquet"
    if not f.exists():
        return None
    d = pd.read_parquet(f)
    if not len(d):
        return None
    d["ym"] = pd.PeriodIndex.from_fields(year=d.revenue_year.astype(int), month=d.revenue_month.astype(int), freq="M")
    r = d.drop_duplicates("ym", keep="last").set_index("ym").revenue.astype(float)
    r = r.reindex(pd.period_range(r.index.min(), r.index.max(), freq="M"))
    v = r.rolling(12, min_periods=12).sum()
    return pd.Series(v.values, index=(v.index + 1).to_timestamp() + pd.Timedelta(days=10))   # m 月 → m+1 月 11 日


def add_fs8(P):
    idx, cols = P["close"].index, P["close"].columns
    out = {k: {} for k in QFEATS + ("rev12",)}
    for s in cols:
        for k, v in _quarterly(s).items():
            a = xs._align(pd.Series(v.values, index=xs._q_avail(v.index)), idx, stale=200)
            if a is not None:
                out[k][s] = a
        r = _rev12(s)
        if r is not None:
            a = xs._align(r, idx)
            if a is not None:
                out["rev12"][s] = a
    for k, x in out.items():
        P[f"f8_{k}"] = pd.DataFrame(x).reindex(index=idx, columns=cols)


def _me(P):
    return P["mv"].where(P["mv"] > 0)


def z_score(P):
    return P["f8_z_part"] + 0.6 * _me(P) / P["f8_tl"]


def ebitda_ev(P):
    ev = _me(P) + P["f8_net_debt"]
    return P["f8_ebitda"] / ev.where(ev > 0)


def sale_me(P):
    return P["f8_rev12"] / _me(P)


def fcf_me(P):
    return P["f8_fcf"] / _me(P)


def eqnpo(P):
    return P["dy"] / 100 - P["net_issue"]


def ret_12_7(P):
    C = P["close"]
    return C.shift(MOM_S) / C.shift(MOM_L) - 1


def hl_spread(P):
    """Corwin-Schultz 2012：相鄰兩日的最高、最低價估價差，負值設 0，取近 21 日平均。"""
    Hh, Ll = P["high"].where(P["high"] > 0), P["low"].where(P["low"] > 0)
    k = 3 - 2 * np.sqrt(2)
    b = np.log(Hh / Ll) ** 2 + np.log(Hh.shift(1) / Ll.shift(1)) ** 2
    gm = np.log(np.maximum(Hh, Hh.shift(1)) / np.minimum(Ll, Ll.shift(1))) ** 2
    a = (np.sqrt(2 * b) - np.sqrt(b)) / k - np.sqrt(gm / k)
    sp = (2 * (np.exp(a) - 1) / (1 + np.exp(a))).clip(lower=0)
    return sp.rolling(HL_WIN, min_periods=HL_MIN).mean()


def _rm(P):
    R = P["close"].pct_change(fill_method=None)
    M = pd.DataFrame(np.broadcast_to(P["mkt_r"].values[:, None], R.shape), index=R.index, columns=R.columns).where(R.notna())
    return R, M


def beta_down(P):
    R, M = _rm(P)
    d = M < 0
    R, M = R.where(d), M.where(d)
    s = lambda x: x.rolling(BD_WIN, min_periods=1).sum()   # noqa: E731
    n = R.notna().astype(float).rolling(BD_WIN, min_periods=1).sum()
    sr, sm, srm, smm = s(R), s(M), s(R * M), s(M * M)
    var = smm - sm ** 2 / n
    return ((srm - sr * sm / n) / var.where(var > 0)).where(n >= BD_MIN)


def coskew(P):
    R, M = _rm(P)
    m = lambda x: x.rolling(CS_WIN, min_periods=CS_MIN).mean()   # noqa: E731
    a, b = m(R), m(M)
    varm, varr = m(M * M) - b ** 2, m(R * R) - a ** 2
    num = m(R * M * M) - 2 * b * m(R * M) + b ** 2 * a - a * varm
    return num / (np.sqrt(varr.where(varr > 0)) * varm.where(varm > 0))


def _q(k):
    return lambda P: P[f"f8_{k}"]


FNS = {"NOA": (_q("noa"), "淨營運資產"), "NOA_GR": (_q("noa_gr"), "淨營運資產變化"), "TACC": (_q("tacc"), "總應計"),
       "COP": (_q("cop"), "現金基礎營業利潤（近似）"), "OPE": (_q("ope"), "營業利潤÷權益"), "Z_SCORE": (z_score, "Altman Z 分數"),
       "O_SCORE": (_q("o_score"), "Ohlson O 分數"), "CASH_AT": (_q("cash_at"), "現金÷資產"), "EBITDA_EV": (ebitda_ev, "EBITDA÷企業價值"),
       "SALE_ME": (sale_me, "營收股價比"), "FCF_ME": (fcf_me, "自由現金流÷市值"), "EQNPO": (eqnpo, "淨發放殖利率（近似）"),
       "CAPX_GR2": (_q("capx_gr2"), "資本支出2年成長"), "PPEINV": (_q("ppeinv"), "固定資產＋存貨變化"),
       "DSALE_DINV": (_q("dsale_dinv"), "營收成長−存貨成長"), "ROE_CHG": (_q("roe_chg"), "季ROE變化"),
       "RET_12_7": (ret_12_7, "中期動能（12～7月）"), "HL_SPREAD": (hl_spread, "高低價價差"),
       "BETA_DOWN": (beta_down, "下檔 beta"), "COSKEW": (coskew, "共偏態")}
NAMES = list(FNS)
JKP = {"NOA": ("noa_at", -1), "NOA_GR": ("noa_gr1a", -1), "TACC": ("taccruals_at", -1), "COP": ("cop_at", 1), "OPE": ("ope_be", 1),
       "Z_SCORE": ("z_score", 1), "O_SCORE": ("o_score", -1), "CASH_AT": ("cash_at", 1), "EBITDA_EV": ("ebitda_mev", 1),
       "SALE_ME": ("sale_me", 1), "FCF_ME": ("fcf_me", 1), "EQNPO": ("eqnpo_me", 1), "CAPX_GR2": ("capx_gr2", -1),
       "PPEINV": ("ppeinv_gr1a", -1), "DSALE_DINV": ("dsale_dinv", 1), "ROE_CHG": ("niq_be_chg1", 1), "RET_12_7": ("ret_12_7", 1),
       "HL_SPREAD": ("bidaskhl_21d", 1), "BETA_DOWN": ("betadown_252d", -1), "COSKEW": ("coskew_21d", -1)}
CAT = {**{f: "財報" for f in NAMES[:9] + ["FCF_ME", "EQNPO", "CAPX_GR2", "PPEINV", "DSALE_DINV", "ROE_CHG"]},
       "SALE_ME": "營收", **{f: "價格" for f in ("RET_12_7", "HL_SPREAD", "BETA_DOWN", "COSKEW")}}
DAILY = ("Z_SCORE", "EBITDA_EV", "SALE_ME", "FCF_ME", "EQNPO", "RET_12_7", "HL_SPREAD", "BETA_DOWN", "COSKEW")


if __name__ == "__main__":
    import feature_fs6 as F6
    import feature_tu as TU
    P = xs.load_panel("2019-12-31", fundamentals=True)
    for add in (xs.add_chips, xs.add_fin, xs.add_v16, xs.add_v17, TU.add_tu, F6.add_fs6, add_fs8):
        add(P)
    for f in DAILY:
        print(f, "leak_check_xs:", xs.leak_check_xs(FNS[f][0], P) or "OK", flush=True)
    for f in FNS:
        v = FNS[f][0](P).replace([np.inf, -np.inf], np.nan)
        y = v.loc["2018"].where(P["liq"].loc["2018"])
        ok = v.notna().sum(axis=1) >= 100
        print(f"{f:<10} 池內有值檔數（2018 中位數）{int(y.notna().sum(axis=1).median())}  首個 ≥100 檔 "
              f"{ok.idxmax().date() if ok.any() else '—'}  2018 分位 p1/p50/p99 "
              f"{np.nanpercentile(y, 1):.4g}/{np.nanpercentile(y, 50):.4g}/{np.nanpercentile(y, 99):.4g}  "
              f"0 的比例 {float((y == 0).sum().sum() / max(y.notna().sum().sum(), 1)):.0%}", flush=True)
