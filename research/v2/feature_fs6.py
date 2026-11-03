"""第四十四批 FS6：現有資料可做、特徵庫還沒有的特徵（FEATURE_PLAN 階段 1）。規則見 HYPOTHESES.md 第四十四批。

特徵函數 FNS[代號] = (fn(P), 說明)，介面同 hypotheses.FNS：回傳日頻分數（日期 × 股票），評估時取換股日前一交易日的值。
先呼叫 add_fs6(P)（需已跑 xs.add_chips、xs.add_v16、xs.add_v17、feature_tu.add_tu）載入財報衍生值與融券餘額。
H79、H91、H100 直接用 hypotheses.FNS 的原函數（H100 需 xs.add_v18）。評估併入 feature_library.py。

leak check：.venv/bin/python research/v2/feature_fs6.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402
import xs  # noqa: E402

BETA_WIN, BETA_MIN, SM_MIN, SM_WIN, INT_TOP = 250, 200, 100, 20, 1e6
QFEATS = ("capex_int", "capex_dep", "ar_rev", "int_cov", "net_issue", "cl_lvl", "cl_chg")


def _stock_div_cum(s):
    """除權交易日 → 累計 log(1 + 股票股利（元）÷ 10)。"""
    f = H.D / "div" / f"{s}.parquet"
    if not f.exists():
        return None
    d = pd.read_parquet(f)
    if not len(d) or "StockExDividendTradingDate" not in d:
        return None
    sd = d.StockEarningsDistribution.astype(float).fillna(0) + d.StockStatutorySurplus.astype(float).fillna(0)
    t = pd.to_datetime(d.StockExDividendTradingDate, errors="coerce")
    x = pd.Series(np.log1p(sd.values / 10), index=t)[(sd.values > 0) & t.notna().values]
    return x.groupby(level=0).sum().sort_index().cumsum() if len(x) else None


def _quarterly(s):
    q, b, c = (xs._pivot(H.D / k / f"{s}.parquet") for k in ("fin", "bs", "cf"))
    if b is None:
        return {}
    qi = pd.date_range(min(x.index.min() for x in (q, b, c) if x is not None),
                       max(x.index.max() for x in (q, b, c) if x is not None), freq="QE")
    nan = pd.Series(np.nan, index=qi)

    def g(df, k):
        return df[k].reindex(qi) if df is not None and k in df else nan

    def single(k):                                     # cf 年初累計 → 單季
        cum = g(c, k)
        return cum.where(qi.quarter == 1, cum - cum.shift(1))

    def ttm(x):
        return x.rolling(4, min_periods=4).sum()

    ta = g(b, "TotalAssets").where(lambda x: x > 0)
    capex, dep, intr = -ttm(single("PropertyAndPlantAndEquipment")), ttm(single("Depreciation")), ttm(single("InterestExpense"))
    oi = ttm(g(q, "OperatingIncome"))
    ar, rev = g(b, "AccountsReceivableNet").where(lambda x: x > 0), g(q, "Revenue").where(lambda x: x > 0)
    cov = (oi / intr.where(intr > 0)).mask((intr == 0) & (oi > 0), INT_TOP)
    sh = (g(b, "OrdinaryShare") / 10 - g(b, "NumberOfSharesInEntityHeldByEntityAndByItsSubsidiaries").fillna(0)).where(lambda x: x > 0)
    cum = _stock_div_cum(s)
    L = pd.Series(0.0, index=qi) if cum is None else pd.Series(
        np.r_[0.0, cum.values][np.searchsorted(cum.index.values, qi.values, side="right")], index=qi)
    cl = g(b, "CurrentContractLiabilities")
    return {"capex_int": capex / ta, "capex_dep": capex / dep.where(dep > 0),
            "ar_rev": (ar / ar.shift(4) - 1) - (rev / rev.shift(4) - 1), "int_cov": cov,
            "net_issue": sh / sh.shift(4) / np.exp(L - L.shift(4)) - 1,
            "cl_lvl": cl / ta, "cl_chg": (cl - cl.shift(1)) / ta}


def add_fs6(P):
    idx, cols = P["close"].index, P["close"].columns
    out = {k: {} for k in QFEATS}
    short = {}
    for s in cols:
        for k, v in _quarterly(s).items():
            a = xs._align(pd.Series(v.values, index=xs._q_avail(v.index)), idx, stale=200)
            if a is not None:
                out[k][s] = a
        f = H.D / "margin" / f"{s}.parquet"
        if f.exists():
            d = pd.read_parquet(f)
            if len(d) and "ShortSaleTodayBalance" in d:
                short[s] = d.assign(date=pd.to_datetime(d.date)).drop_duplicates("date").set_index("date").ShortSaleTodayBalance.astype(float)
    for k, x in out.items():
        P[k] = pd.DataFrame(x).reindex(index=idx, columns=cols)
    P["short_bal"] = pd.DataFrame(short).reindex(index=idx, columns=cols).where(P["close"].notna())


# ───────── 價量 ─────────
def _beta_parts(P):
    R = P["close"].pct_change(fill_method=None)
    M = pd.DataFrame(np.broadcast_to(P["mkt_r"].values[:, None], R.shape), index=R.index, columns=R.columns).where(R.notna())
    roll = lambda x: x.rolling(BETA_WIN, min_periods=BETA_MIN).mean()   # noqa: E731
    mr, mm, mrm, mmm, mrr = roll(R), roll(M), roll(R * M), roll(M * M), roll(R * R)
    varm = mmm - mm ** 2
    beta = (mrm - mr * mm) / varm.where(varm > 0)
    return beta, (mrr - mr ** 2) - beta ** 2 * varm


def beta(P):
    return _beta_parts(P)[0]


def ivol(P):
    _, v = _beta_parts(P)
    return np.sqrt(v.clip(lower=0))


def skew(P):
    return np.log(P["close"]).diff().rolling(BETA_WIN, min_periods=BETA_MIN).skew()


def turn(P):
    return P["turn60"]


def sm_ratio(P):
    m = P["margin_bal"]
    return (P["short_bal"] / m).where(m >= SM_MIN)


def sm_dev(P):
    r = sm_ratio(P)
    return r - r.rolling(SM_WIN, min_periods=15).mean()


def _q(k):
    return lambda P: P[k]


FNS = {"CAPEX_INT": (_q("capex_int"), "資本支出強度"), "CAPEX_DEP": (_q("capex_dep"), "資本支出÷折舊"),
       "AR_REV": (_q("ar_rev"), "應收成長−營收成長"), "INT_COV": (_q("int_cov"), "利息保障倍數"),
       "NET_ISSUE": (_q("net_issue"), "淨發行股數"), "CL_LVL": (_q("cl_lvl"), "合約負債比"),
       "CL_CHG": (_q("cl_chg"), "合約負債季增"), "F_SCORE": (_q("fscore"), "F-score（全池）"),
       "BETA": (beta, "市場 beta"), "IVOL": (ivol, "特質波動"), "SKEW": (skew, "報酬偏態"), "TURN": (turn, "週轉率水準"),
       "SM_RATIO": (sm_ratio, "券資比"), "SM_DEV": (sm_dev, "券資比偏離20日均")}
OLD = ("H79", "H91", "H100")                                           # hypotheses.FNS 原函數
NAMES = list(FNS) + list(OLD)
CAT = {**{f: "財報" for f in ("CAPEX_INT", "CAPEX_DEP", "AR_REV", "INT_COV", "NET_ISSUE", "CL_LVL", "CL_CHG", "F_SCORE")},
       **{f: "價格" for f in ("BETA", "IVOL", "SKEW", "TURN", "H79", "H91")},
       "SM_RATIO": "籌碼", "SM_DEV": "籌碼", "H100": "股利"}
DAILY = ("BETA", "IVOL", "SKEW", "TURN", "SM_RATIO", "SM_DEV")


if __name__ == "__main__":
    import feature_tu as TU
    P = xs.load_panel("2019-12-31", fundamentals=True)
    for add in (xs.add_chips, xs.add_fin, xs.add_v16, xs.add_v17, TU.add_tu, add_fs6):
        add(P)
    for f in DAILY:
        print(f, "leak_check_xs:", xs.leak_check_xs(FNS[f][0], P) or "OK", flush=True)
    for f in FNS:
        X = FNS[f][0](P)
        v = X.replace([np.inf, -np.inf], np.nan)
        print(f"{f:<10} 有值檔數（2018 年中位數）{int(v.loc['2018'].notna().sum(axis=1).median())}  首個 ≥100 檔的日期 "
              f"{(v.notna().sum(axis=1) >= 100).idxmax().date()}  0 的比例 {float((v == 0).sum().sum() / max(v.notna().sum().sum(), 1)):.0%}")
