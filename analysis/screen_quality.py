"""追蹤清單的候選來源：「本業體質好、但近期股價弱」篩選。

這不是交易訊號（專案研究顯示價值面在台股不是有效訊號），只是幫質化追蹤找候選。
篩出來的每一檔都要再做綜合分析，回答「股價為什麼跌、是暫時還是永久」。

用法：.venv/bin/python analysis/screen_quality.py [YYYY-MM-DD] [--dd -0.25]
讀本地 data/v2（日期以本地資料為準；最新資料在 ai-dev）。

條件：
  股票池：pit 流動池（60 日均量 ≥ 200 張、≥ 250 根 K）、60 日均量 ≥ 300 張、非金融
  體質（2021~2025 五個完整年度）：
    每年 EPS > 0、ROE 最低 ≥ 8%、平均 ≥ 13%、業外佔稅前平均 < 30%、營益率平均 ≥ 6%
  最近的體質（2026-11-05 加，美時的教訓：五年資料看不到最近的併購和舉債）：
    近 4 季 EPS ≥ 五年平均的 80%、近 4 季 ROE ≥ 10%、
    負債比 ≤ 60% 且一年內增加 ≤ 15 個百分點、無形資產 ≤ 股東權益的 50%
  本業估值（2026-11-05 加，聯華食的教訓：證交所本益比用配股前股數、含證券評價等業外）：
    近 4 季業外佔稅前 < 25%；
    本業 EPS＝近 4 季營業利益 × 0.8（稅）× 歸屬母公司比例 ÷ 最新股數（股本含待分配股票股利）；
    本業本益比 ≤ 自己 5 年的本益比中位數
  股價弱：距一年高點跌 ≥ 25%、本益比在自己 5 年的中位數以下
  已知限制：子公司是上市櫃公司時（例如振曜持有沛亨），本業 EPS 會混進子公司；
    「少數股權%」高（> 20%）的要在綜合分析裡拆開
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "v2"))
import xs  # noqa: E402

YEARS = range(2021, 2026)
EXCLUDE_IND = {"金融保險業", "金融業", "存託憑證"}


def pivot(path):
    q = pd.read_parquet(path).pivot_table(index="date", columns="type", values="value", aggfunc="last").sort_index()
    q.index = pd.to_datetime(q.index)
    return q


def ratio(q, num, den):
    if num not in q or den not in q:
        return pd.Series(np.nan, index=list(YEARS))
    y = q.groupby(q.index.year)
    return (y[num].sum() / y[den].sum()).reindex(list(YEARS))


def tracked():
    wj = ROOT / "analysis" / "watchlist.json"
    if wj.exists():
        d = json.loads(wj.read_text(encoding="utf-8"))
        return {
            item["stock_id"]
            for sec in ("tracking", "passed", "removed")
            for item in d.get(sec, [])
            if isinstance(item, dict) and "stock_id" in item
        }
    w = ROOT / "analysis" / "WATCHLIST.md"
    if not w.exists():
        return set()
    return {line.split()[1] for line in w.read_text().splitlines() if line.startswith("### ")} | \
           {line.split("|")[1].strip() for line in w.read_text().splitlines() if line.startswith("| ") and line.split("|")[1].strip().isdigit()}


def main(asof, dd_max=-0.25):
    P = xs.load_panel(asof, fundamentals=True)
    d = P["close"].index[-1]
    vol60 = P["volume"].iloc[-60:].mean() / 1000
    pool = (P["liq"].fillna(False).astype(bool) & (P["nbars"] >= 250) & (P["raw_close"] > 0)).loc[d] & (vol60 >= 300)
    u = pd.read_csv(ROOT / "data" / "v2" / "universe.csv", dtype=str).set_index("stock_id")
    adj, raw, per, pbr = P["close"], P["raw_close"], P["per"], P["pbr"]
    dd = adj.loc[d] / adj.iloc[-252:].max() - 1
    rows = []
    for s in pool[pool].index:
        if u.loc[s, "industry"] in EXCLUDE_IND:
            continue
        try:
            q = pivot(ROOT / "data" / "v2" / "fin" / f"{s}.parquet")
            b = pivot(ROOT / "data" / "v2" / "bs" / f"{s}.parquet")
        except Exception:
            continue
        if "EPS" not in q:
            continue
        y = q.groupby(q.index.year)
        eps_y, cnt = y.EPS.sum(), y.EPS.count()
        if any(cnt.get(t, 0) != 4 for t in YEARS):
            continue
        e5 = eps_y.loc[list(YEARS)]
        bv = (raw[s] / pbr[s].replace(0, np.nan)).dropna()
        if bv.empty:
            continue
        bvy = bv.groupby(bv.index.year).last()
        roe = np.array([e5[t] / bvy.get(t - 1, np.nan) for t in YEARS], float)
        ttm = q.EPS.dropna().iloc[-4:].sum()
        debt = (b["Liabilities"] / b["TotalAssets"]).dropna() if {"Liabilities", "TotalAssets"} <= set(b) else pd.Series(dtype=float)
        intang = (b.get("IntangibleAssets", pd.Series(0.0, index=b.index)).fillna(0) / b["Equity"]).dropna() if "Equity" in b else pd.Series(dtype=float)
        pe5 = per[s].loc[d - pd.DateOffset(years=5):]
        pe5 = pe5[pe5 > 0]
        last4 = q.iloc[-4:]
        shares = max(b.iloc[-1].get("CapitalStock", 0) or 0, b.iloc[-1].get("OrdinaryShare", 0) or 0) / 10
        tot4 = last4.get("TotalConsolidatedProfitForThePeriod", pd.Series(dtype=float)).sum()
        nci4 = last4.get("NoncontrollingInterests", pd.Series(dtype=float)).sum()
        nci = float(np.clip(nci4 / tot4, 0, 1)) if tot4 > 0 else 0.0
        core_eps = last4.get("OperatingIncome", pd.Series(dtype=float)).sum() * 0.8 * (1 - nci) / shares if shares > 0 else np.nan
        pre4 = last4.get("PreTaxIncome", pd.Series(dtype=float)).sum()
        rows.append(dict(
            id=s, name=u.loc[s, "name"], ind=u.loc[s, "industry"], close=raw.loc[d, s], vol60=round(vol60[s]),
            dd=dd[s], eps5min=e5.min(), eps5avg=e5.mean(), ttm=ttm, ttm_vs=ttm / e5.mean() if e5.mean() > 0 else np.nan,
            roe_min=np.nanmin(roe), roe_avg=np.nanmean(roe), roe_ttm=ttm / bv.iloc[-1],
            nonop=ratio(q, "TotalNonoperatingIncomeAndExpense", "PreTaxIncome").mean(),
            opm=ratio(q, "OperatingIncome", "Revenue").mean(),
            debt=debt.iloc[-1] if len(debt) else np.nan,
            debt_chg=debt.iloc[-1] - debt.iloc[-5] if len(debt) >= 5 else np.nan,
            intang=intang.iloc[-1] if len(intang) else np.nan,
            per=per.loc[d, s], pe_pct=(pe5 < per.loc[d, s]).mean() if len(pe5) else np.nan, pe_med=pe5.median(),
            pbr=pbr.loc[d, s], dy=P["dy"].loc[d, s], yoy3=P["rev_yoy3"].loc[d, s],
            nonop4=last4.get("TotalNonoperatingIncomeAndExpense", pd.Series(dtype=float)).sum() / pre4 if pre4 > 0 else np.nan,
            nci=nci, core_eps=core_eps, core_pe=raw.loc[d, s] / core_eps if core_eps and core_eps > 0 else np.nan))
    R = pd.DataFrame(rows).set_index("id")
    Q = R[(R.eps5min > 0) & (R.roe_min >= 0.08) & (R.roe_avg >= 0.13) & (R.nonop < 0.3) & (R.opm >= 0.06)
          & (R.ttm_vs >= 0.8) & (R.roe_ttm >= 0.10) & (R.debt <= 0.60) & (R.debt_chg.fillna(0) <= 0.15) & (R.intang.fillna(0) <= 0.5)
          & (R.nonop4 < 0.25) & (R.core_pe > 0) & (R.core_pe <= R.pe_med)]
    W = Q[(Q.dd <= dd_max) & (Q.pe_pct <= 0.5)].copy()
    W["core_vs_med"] = W.core_pe / W.pe_med
    W = W.sort_values("core_vs_med")
    W.insert(0, "清單", ["✓" if s in tracked() else "" for s in W.index])
    print(f"{d.date()}：池 {len(R)} 檔 → 體質好、本業估值不高於自己 5 年中位數 {len(Q)} 檔 → 而且股價弱 {len(W)} 檔（依「本業本益比 ÷ 自己 5 年本益比中位數」排序；✓＝已在 WATCHLIST，含不追蹤；nci＝少數股權佔淨利%，> 20 要拆子公司）")
    f = W.copy()
    f = f.drop(columns=["eps5min", "nonop", "debt_chg", "intang"])
    for c in ("dd", "roe_min", "roe_avg", "roe_ttm", "nonop4", "nci", "opm", "debt", "pe_pct", "yoy3"):
        f[c] = (f[c] * 100).round(0)
    for c in ("eps5avg", "ttm", "ttm_vs", "core_eps", "core_pe", "per", "pe_med", "pbr"):
        f[c] = f[c].round(2)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_rows", 300)
    print(f.to_string())


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("asof", nargs="?", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    ap.add_argument("--dd", type=float, default=-0.25, help="距一年高點的跌幅上限（預設 -0.25）")
    a = ap.parse_args()
    main(a.asof, a.dd)
