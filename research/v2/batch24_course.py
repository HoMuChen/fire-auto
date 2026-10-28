"""第二十四批：法人選股課程可量化四項（H121 QMV、H122 GARP、H123 營運槓桿、H124 H2 + 紅旗排除）。
用法：python3 research/v2/batch24_course.py dev [H121 H122 H123 H124]
新特徵在此檔自建（add_b24），不改 xs.py；財報依法定截止 +1 天對齊（xs._q_avail）。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402
import hypotheses as Hy  # noqa: E402
import xs  # noqa: E402

TAX = 0.20
DSRI = 1.465
NA = -999.0   # 「本季已知但不合格」：xs._align 會丟 NaN 而沿用上一季舊值，故先標記、對齊後再轉 NaN
STARTS = {"H121": "2013-04-01", "H122": "2011-04-01", "H123": "2010-06-01", "H124": "2014-04-01"}


def add_b24(P):
    """roic、ebitda / net_debt（→ ev_y）、ni_g、oplev、flag（紅旗 0/1）。"""
    idx, cols = P["close"].index, P["close"].columns
    keys = ("roic", "ebitda", "net_debt", "ni_g", "oplev", "flag")
    out = {k: {} for k in keys}
    for s in cols:
        q, b, c = (xs._pivot(H.D / k / f"{s}.parquet") for k in ("fin", "bs", "cf"))
        if q is None:
            continue
        ix = [x.index for x in (q, b, c) if x is not None]
        qi = pd.date_range(min(i.min() for i in ix), max(i.max() for i in ix), freq="QE")
        nan = pd.Series(np.nan, index=qi)
        g = lambda df, k: df[k].reindex(qi) if df is not None and k in df else nan   # noqa: E731
        ttm = lambda x: x.rolling(4, min_periods=4).sum()                           # noqa: E731
        ytd = lambda x: x.where(qi.quarter == 1, x - x.shift(1))                    # 現金流量表年初累計 → 單季  # noqa: E731
        rev, oi = g(q, "Revenue"), g(q, "OperatingIncome")
        ni = g(q, "IncomeAfterTaxes").combine_first(g(q, "NetIncome"))
        cogs = g(q, "CostOfGoodsSold")
        rev_t, oi_t, ni_t, cogs_t = ttm(rev), ttm(oi), ttm(ni), ttm(cogs)
        feats = {}
        # H122：淨利 TTM 成長
        mark = lambda v, known: v.where(v.notna() | ~known, NA)                    # noqa: E731
        feats["ni_g"] = mark((ni_t / ni_t.shift(4).where(lambda x: x > 0) - 1).clip(upper=1.0), ni_t.notna())
        # H123：單季營收 / 營業利益 YoY
        rg = rev / rev.shift(4).where(lambda x: x > 0) - 1
        og = (oi / oi.shift(4).where(lambda x: x > 0) - 1).where(oi > 0).clip(upper=3.0)
        feats["oplev"] = mark((og - rg).where(rg > 0), rev.notna())
        if b is not None:
            ta = g(b, "TotalAssets")
            has = ta.notna()
            z = lambda k: g(b, k).fillna(0).where(has)                               # noqa: E731
            debt = z("ShorttermBorrowings") + z("LongtermBorrowings") + z("BondsPayable")
            cash = z("CashAndCashEquivalents")
            ic = (g(b, "Equity") + debt - cash).where(lambda x: x > 0)
            feats["roic"] = mark(oi_t * (1 - TAX) / ic, has & oi_t.notna())
            feats["net_debt"] = debt - cash
            ar = z("AccountsReceivableNet") + z("AccountsReceivableDuefromRelatedPartiesNet")
            ari = ar / rev_t.where(rev_t > 0)
            ivi = g(b, "Inventories") / cogs_t.where(cogs_t > 0)
            f2 = (ari / ari.shift(4).where(lambda x: x > 0)) > DSRI
            f3 = (ivi / ivi.shift(4).where(lambda x: x > 0)) > DSRI
        else:
            f2 = f3 = pd.Series(False, index=qi)
        f1 = pd.Series(False, index=qi)
        if c is not None:
            cfo_k = "CashFlowsFromOperatingActivities" if "CashFlowsFromOperatingActivities" in c else "NetCashInflowFromOperatingActivities"
            cfo = ttm(ytd(g(c, cfo_k)))
            capex = ttm(ytd(g(c, "PropertyAndPlantAndEquipment").fillna(0)))
            fcf = cfo + capex                                                        # capex 為負值
            f1 = (fcf < 0) & (fcf.shift(4) < 0)
            da = ttm(ytd(g(c, "Depreciation"))) + ttm(ytd(g(c, "AmortizationExpense").fillna(0)))
            feats["ebitda"] = mark(oi_t + da, cfo.notna() & oi_t.notna())
        feats["flag"] = (f1 | f2 | f3).astype(float)
        av = xs._q_avail(qi)
        for k, v in feats.items():
            a = xs._align(pd.Series(v.values, index=av), idx, stale=200)
            if a is not None:
                out[k][s] = a.where(a != NA)
    for k, x in out.items():
        P[k] = pd.DataFrame(x).reindex(index=idx, columns=cols)
    ev = P["mv"] + P["net_debt"]
    P["ev_y"] = P["ebitda"] / ev.where(ev > 0)
    P["flag"] = P["flag"].fillna(0)


def h121_qmv(P):
    return (Hy._pct(P["roic"]) + Hy._pct(Hy.h1_mom12_1(P)) + Hy._pct(P["ev_y"])) / 3


def h122_garp(P):
    g, pe = P["ni_g"], P["per"]
    return (g * 100 / pe).where((g > 0) & (pe > 0))


def h123_oplev(P):
    return P["oplev"]


def h124_h2_redflag(P):
    return Hy.h2_hi52(P) - 10 * P["flag"]


FNS = {"H121": (h121_qmv, "QMV三因子"), "H122": (h122_garp, "GARP倒PEG"), "H123": (h123_oplev, "營運槓桿"),
       "H124": (h124_h2_redflag, "H2+紅旗排除")}


def flagged_diag(P, p0, p1, topn=20):
    """H2 原始 top20 中有紅旗者 vs 無紅旗者的次月報酬（相對 pit 等權大盤）。"""
    O, C = P["open"], P["close"]
    Cf, idx = C.ffill(), O.index
    rd = xs.rebalance_dates(idx, pd.Timestamp(p0), pd.Timestamp(p1))
    ss, fl = Hy.h2_hi52(P).shift(1), P["flag"].shift(1)
    elig = P["liq"].shift(1).fillna(False).astype(bool) & (P["nbars"].shift(1) >= 250) & O.notna()
    a, b, n = [], [], []
    for d, d2 in zip(rd[:-1], rd[1:]):
        e = elig.loc[d]
        top = ss.loc[d][e].dropna().nlargest(topn).index
        k2 = idx.get_loc(d2)
        r = (O.loc[d2].fillna(Cf.iloc[k2 - 1]) / O.loc[d] - 1)[e].clip(-0.95, 10)
        bench = r.mean()
        f = fl.loc[d, top] > 0
        n.append(int(f.sum()))
        a += list(r[top[f.values]] - bench)
        b += list(r[top[~f.values]] - bench)
    a, b = np.array(a), np.array(b)
    t = (a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) if len(a) > 2 else np.nan
    return (f"H2 top20 紅旗股：平均每月 {np.mean(n):.2f} 檔（{len(a)} 筆），次月超額 {a.mean()*100:+.2f}% "
            f"vs 無紅旗 {b.mean()*100:+.2f}%（Welch t {t:+.2f}）")


if __name__ == "__main__":
    period = sys.argv[1]
    want = [k for k in sys.argv[2:] if not k.startswith("--")] or list(FNS)
    p0, p1 = H.PERIODS[period]
    if period == "holdout":
        raise PermissionError("holdout 封存中")
    end = min(pd.Timestamp(p1), pd.Timestamp.today()).strftime("%Y-%m-%d")
    P = xs.load_panel(end, fundamentals=True)
    xs.add_v16(P)
    add_b24(P)
    rows, cmp_lines = [], []
    for k in want:
        st = STARTS[k]
        res, _ = xs.run({k: FNS[k], "H2": Hy.FNS["H2"], "H29": Hy.FNS["H29"]}, period, P=P,
                        note=f"第二十四批 法人選股課程（事前，起 {st}）", topn=20, start=st)
        for nm, (R, s) in res.items():
            for col in ("top_q", "topn"):
                rows.append(dict(批=k, H=nm, port=col, **s[col]))
            rows[-1]["Q"] = " / ".join(f"{v*100:+.1f}" for v in s["Q_ex_ann"].values())
        from batch19_xs import compare
        cmp_lines += [compare(res[k][0], res["H2"][0], k, "H2"), compare(res[k][0], res["H29"][0], k, "H29")]
        if k == "H124":
            cmp_lines.append(flagged_diag(P, max(p0, st), p1))
    df = pd.DataFrame(rows)
    for c in ("CAGR", "bench_CAGR", "ex_ann", "MDD", "turnover"):
        df[c] = (df[c] * 100).round(1)
    pd.set_option("display.width", 250)
    print(f"[{period}]")
    print(df.round(2).to_string(index=False))
    print("\n對照（top20 月報酬差）")
    print("\n".join("  " + x for x in cmp_lines))
