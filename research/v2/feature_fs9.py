"""第四十七批 FS9：學術因子第二輪（FACTOR_MAP.md 精選 16 個）＋產業內估值 2 個（事後）。規則見 HYPOTHESES.md 第四十七批。

特徵函數 FNS[代號] = (fn(P), 說明)，介面同 hypotheses.FNS：回傳日頻分數（日期 × 股票），評估時取換股日前一交易日的值。
COMBO（組合）與 PEER（同業內百分位）照 feature_tu 的格式，由 feature_library.py 在換股日的股票池裡算；
AUX 是組合用的成分（不單獨評估）。
先呼叫 add_fs9(P)（需已跑 load_panel(fundamentals=True)、xs.add_v16（市值））。評估併入 feature_library.py。

leak check：.venv/bin/python research/v2/feature_fs9.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402
import xs  # noqa: E402

DV_WIN, DV_MIN, NQ_VOL, NQ_INC = 126, 100, 8, 8
DEBT = ("ShorttermBorrowings", "LongtermBorrowings", "BondsPayable")
QFEATS = ("op_at", "ocf_at", "at_turn", "dgp_dsale", "ni_inc8q", "debt", "cash", "ta", "tangib", "ni_vol", "cf_vol",
          "kz_cf_k", "kz_q_part", "kz_dc", "kz_cash_k", "k", "be", "gm", "lev", "roa")


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

    niq = g(q, "IncomeAfterTaxes").fillna(g(q, "NetIncome"))   # 2013 以前的科目名稱是 NetIncome
    ni, oi, rev, gp = ttm(niq), ttm(g(q, "OperatingIncome")), ttm(g(q, "Revenue")), ttm(g(q, "GrossProfit"))
    cfo_k = "CashFlowsFromOperatingActivities" if c is not None and "CashFlowsFromOperatingActivities" in c else "NetCashInflowFromOperatingActivities"
    cfoq = single(cfo_k)
    cfo, dep = ttm(cfoq), ttm(single("Depreciation"))
    cash, be, tl = g(b, "CashAndCashEquivalents"), pos(g(b, "Equity")), g(b, "Liabilities")
    debt = sum(z0(k) for k in DEBT)
    ta4 = ta.shift(4)
    up = (niq > niq.shift(4)).astype(float).where(niq.notna() & niq.shift(4).notna())
    run = pd.Series(np.nan, index=qi)
    for i in range(NQ_INC - 1, len(qi)):                # 最近幾季連續成長的季數（最多 8，8 次比較都要有值）
        w = up.iloc[i - NQ_INC + 1:i + 1].values[::-1]
        if np.isfinite(w).all():
            run.iloc[i] = float(np.argmin(np.r_[w, 0]) if (w == 0).any() else NQ_INC)
    k = pos(g(b, "PropertyPlantAndEquipment")).shift(4)
    revq = pos(g(q, "Revenue"))
    return {"op_at": oi / ta, "ocf_at": cfo / ta, "at_turn": rev / ((ta + ta4) / 2),
            "dgp_dsale": (gp / pos(gp.shift(4)) - 1) - (rev / pos(rev.shift(4)) - 1),
            "ni_inc8q": run, "debt": debt, "cash": cash, "ta": ta,
            "tangib": (cash + 0.715 * z0("AccountsReceivableNet") + 0.547 * z0("Inventories")
                       + 0.535 * g(b, "PropertyPlantAndEquipment")) / ta,
            "ni_vol": (niq / ta).rolling(NQ_VOL, min_periods=NQ_VOL).std(),
            "cf_vol": (cfoq / revq).rolling(NQ_VOL, min_periods=NQ_VOL).std(),
            "kz_cf_k": (ni + dep) / k, "kz_q_part": (ta - be) / ta, "kz_dc": debt / pos(debt + be), "kz_cash_k": cash / k,
            "k": k, "be": be, "gm": gp / pos(rev), "lev": tl / ta, "roa": ni / ta4}


def add_fs9(P):
    idx, cols = P["close"].index, P["close"].columns
    out = {k: {} for k in QFEATS}
    for s in cols:
        for k, v in _quarterly(s).items():
            a = xs._align(pd.Series(v.values, index=xs._q_avail(v.index)), idx, stale=200)
            if a is not None:
                out[k][s] = a
    for k, x in out.items():
        P[f"f9_{k}"] = pd.DataFrame(x).reindex(index=idx, columns=cols)


def _me(P):
    return P["mv"].where(P["mv"] > 0)


def debt_me(P):
    return P["f9_debt"] / _me(P)


def netdebt_me(P):
    return (P["f9_debt"] - P["f9_cash"]) / _me(P)


def at_me(P):
    return P["f9_ta"] / _me(P)


def kz(P):
    me = _me(P)
    return (-1.001909 * P["f9_kz_cf_k"] + 0.2826389 * (me / P["f9_ta"] + P["f9_kz_q_part"]) + 3.139193 * P["f9_kz_dc"]
            - 39.3678 * (P["dy"] / 100 * me) / P["f9_k"] - 1.314759 * P["f9_kz_cash_k"])


def dolvol_cv(P):
    m = P["money"].astype(float).where(P["close"].notna())
    r = m.rolling(DV_WIN, min_periods=DV_MIN)
    mu = r.mean()
    return r.std() / mu.where(mu > 0)


def _q(k):
    return lambda P: P[f"f9_{k}"]


FNS = {"OP_AT": (_q("op_at"), "營業利潤÷資產"), "OCF_AT": (_q("ocf_at"), "營業現金流÷資產"), "AT_TURN": (_q("at_turn"), "資產週轉率"),
       "DGP_DSALE": (_q("dgp_dsale"), "毛利成長−營收成長"), "NI_INC8Q": (_q("ni_inc8q"), "連續盈餘成長季數"),
       "DEBT_ME": (debt_me, "負債÷市值"), "NETDEBT_ME": (netdebt_me, "淨負債÷市值"), "AT_ME": (at_me, "資產÷市值"),
       "TANGIB": (_q("tangib"), "資產有形性"), "KZ": (kz, "KZ 財務限制指數"), "NI_VOL": (_q("ni_vol"), "盈餘波動"),
       "CF_VOL": (_q("cf_vol"), "現金流波動"), "DOLVOL_CV": (dolvol_cv, "成交金額變異係數")}
AUX = {"GM": (_q("gm"), "毛利率"), "LEV": (_q("lev"), "負債比"), "ROA": (_q("roa"), "ROA")}
COMBO = {"QMJ": ([("H27", 1), ("H26", 1), ("OP_AT", 1), ("OCF_AT", 1), ("GM", 1), ("TACC", -1),
                  ("BETA", -1), ("LEV", -1), ("O_SCORE", -1), ("Z_SCORE", 1), ("NI_VOL", -1)], 6, None, "品質（獲利＋安全）"),
         "MISP_MGMT": ([("NET_ISSUE", -1), ("H83", 1), ("NOA", -1), ("H80", 1), ("PPEINV", -1)], 3, None, "錯價因子：管理面"),
         "MISP_PERF": ([("O_SCORE", -1), ("H1", 1), ("H27", 1), ("ROA", 1)], 3, None, "錯價因子：績效面")}
PEER = {"PEER_EBITDA_EV": ("EBITDA_EV", "同業內 EBITDA÷EV（事後）"), "PEER_FCF_ME": ("FCF_ME", "同業內自由現金流÷市值（事後）")}
NAMES = list(FNS) + list(COMBO) + list(PEER)
CAT = {**{f: "財報" for f in ("OP_AT", "OCF_AT", "AT_TURN", "DGP_DSALE", "NI_INC8Q", "TANGIB", "NI_VOL", "CF_VOL")},
       **{f: "估值" for f in ("DEBT_ME", "NETDEBT_ME", "AT_ME", "KZ", "PEER_EBITDA_EV", "PEER_FCF_ME")},
       "DOLVOL_CV": "價格", "QMJ": "組合", "MISP_MGMT": "組合", "MISP_PERF": "組合"}
NOTES = {"PEER_EBITDA_EV": "事後（看過第四十六批才提出）", "PEER_FCF_ME": "事後（看過第四十六批才提出）"}
DAILY = ("DEBT_ME", "NETDEBT_ME", "AT_ME", "KZ", "DOLVOL_CV")


if __name__ == "__main__":
    P = xs.load_panel("2019-12-31", fundamentals=True)
    for add in (xs.add_v16, add_fs9):
        add(P)
    for f in DAILY:
        print(f, "leak_check_xs:", xs.leak_check_xs(FNS[f][0], P) or "OK", flush=True)
    for f, (fn, _) in {**FNS, **AUX}.items():
        v = fn(P).replace([np.inf, -np.inf], np.nan)
        y = v.loc["2018"].where(P["liq"].loc["2018"])
        ok = v.notna().sum(axis=1) >= 100
        print(f"{f:<10} 池內有值檔數（2018 中位數）{int(y.notna().sum(axis=1).median())}  首個 ≥100 檔 "
              f"{ok.idxmax().date() if ok.any() else '—'}  2018 分位 p1/p50/p99 "
              f"{np.nanpercentile(y, 1):.4g}/{np.nanpercentile(y, 50):.4g}/{np.nanpercentile(y, 99):.4g}  "
              f"0 的比例 {float((y == 0).sum().sum() / max(y.notna().sum().sum(), 1)):.0%}", flush=True)
