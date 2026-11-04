"""第六十四批 H173：PORTFOLIO_PLAN P4 跨資產（只做多、固定比例）。規則見 HYPOTHESES.md 第六十四批（事前登記）。
資料：fetch_xasset.py（美國 ETF 代理＋台灣上市對應 ETF）、fetch_adr.py（美元匯率）。

用法：.venv/bin/python research/v2/xasset_lab.py dev|val
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402
import portfolio_lab as PL  # noqa: E402
import strategy_lab as SL  # noqa: E402
import xs  # noqa: E402

RES = H.D / "results"
FEE = 0.005
US = {"X1": "SPY", "X2": "TLT", "X3": "IEF", "X4": "GLD"}
TW = {"X1": "00646", "X2": "00679B", "X3": "00697B", "X4": "00635U"}
LABEL = {"X1": "美股", "X2": "美國長債", "X3": "美國中期公債", "X4": "黃金", "X5": "外部籃子（等權）"}
NAME = "第六十四批 H173 跨資產"
TRIALS = 62832
SEGS = {"dev": ("A", "B"), "val": ("val",)}


def _series(path, col):
    d = pd.read_parquet(path)
    d["date"] = pd.to_datetime(d.date)
    return d.drop_duplicates("date").set_index("date")[col].astype(float).sort_index()


def fx():
    d = pd.read_parquet(H.D / "us" / "usdtwd.parquet")
    d["date"] = pd.to_datetime(d.date)
    s = ((d.spot_buy.astype(float) + d.spot_sell.astype(float)) / 2).where(lambda x: x > 0)
    return pd.Series(s.values, index=d.date).dropna().sort_index()


def windows(period):
    """H136 報酬帳的換股窗 (d, d2) 與 H136 月報酬。"""
    led = pd.read_parquet(RES / "sleeves" / period / "H136.parquet")
    md = led.index
    tr = pd.read_parquet(H.D / "taiex_tr.parquet")
    days = pd.DatetimeIndex(pd.to_datetime(tr.date)).drop_duplicates().sort_values()
    rd = xs.rebalance_dates(days, md[0], pd.Timestamp(H.PERIODS[period][1]))
    assert list(rd[:len(md)]) == list(md), "換股日對不上"
    return md, pd.DatetimeIndex(rd[1:len(md) + 1]), led.ret.values


def proxy_monthly(md, md2):
    F = fx()
    out = {}
    for k, t in US.items():
        px = _series(H.D / "us" / f"{t}.parquet", "Adj_Close")
        r = []
        for d, d2 in zip(md, md2):
            a, b = d - pd.Timedelta(days=1), d2 - pd.Timedelta(days=1)
            p1, p2, f1, f2 = px.asof(a), px.asof(b), F.asof(a), F.asof(b)
            r.append(p2 * f2 / (p1 * f1) - 1 - FEE * (d2 - d).days / 365 if all(np.isfinite([p1, p2, f1, f2])) else np.nan)
        out[k] = np.array(r)
    out["X5"] = np.mean([out[k] for k in US], axis=0)
    return out


def tracking(p0, p1):
    """台灣上市 ETF（還原）和代理（台幣、扣費前）的月底報酬相關與年化差距；近 60 日平均成交金額（百萬元）。"""
    F = fx()
    res = {}
    for k, t in TW.items():
        tw = _series(H.D / "twetf" / f"{t}_adj.parquet", "close")
        raw = pd.read_parquet(H.D / "twetf" / f"{t}_raw.parquet")
        raw["date"] = pd.to_datetime(raw.date)
        us = _series(H.D / "us" / f"{US[k]}.parquet", "Adj_Close")
        me = pd.date_range(p0, p1, freq="ME")
        a = np.array([tw.asof(d) for d in me])
        b = np.array([us.asof(d - pd.Timedelta(days=1)) * F.asof(d) for d in me])   # 台灣月底收盤反映前一晚美國收盤
        ra, rb = a[1:] / a[:-1] - 1, b[1:] / b[:-1] - 1
        m = np.isfinite(ra) & np.isfinite(rb)
        liq = raw[raw.date <= pd.Timestamp(p1)].tail(60).Trading_money.astype(float).mean() / 1e6
        res[k] = dict(etf=t, months=int(m.sum()), corr=float(np.corrcoef(ra[m], rb[m])[0, 1]) if m.sum() > 12 else np.nan,
                      gap_ann=float((ra[m] - rb[m]).mean() * 12) if m.sum() > 12 else np.nan, money60_m=float(liq))
    return res


def report(k, x, h, md, period):
    out = dict(name=k, label=LABEL[k])
    seg = "dev" if period == "dev" else "val"
    s = PL.st(x, md, seg)
    out.update(sharpe=s.get("sharpe"), ann=s.get("ann"), vol=s.get("vol"), mdd=s.get("mdd"), yrs=s.get("yrs"), by_year=s.get("by_year"))
    for g in SEGS[period]:
        out[f"sharpe_{g}"] = PL.st(x, md, g).get("sharpe")
    out["rho"] = PL.corr(x, h, md, seg)
    out["worst25"] = PL.worst25(x, h, md, seg)
    for rf in (0.0, 0.01):
        g3, ok = PL.gate3(x, h, md, SEGS[period], rf)
        out[f"gate3_rf{int(rf*100)}"] = {a: (round(b[0], 3), round(b[1], 3)) for a, b in g3.items()}
        out[f"gate3_ok_rf{int(rf*100)}"] = ok
    out["gate2_ok"] = bool(out["rho"] <= 0.3)
    return out


def show(r):
    w = r["worst25"]
    print(f"  {r['name']} {r['label']}：年化 {r['ann']*100:+.1f}% 波動 {r['vol']*100:.1f}% Sharpe {r['sharpe']:.2f}"
          f"（{'、'.join(f'{g} {r[f'sharpe_{g}']:.2f}' for g in ('A', 'B', 'val') if f'sharpe_{g}' in r)}）MDD {r['mdd']*100:.0f}% 正年 {r['yrs']}"
          f"｜和 H136 相關 {r['rho']:+.2f} {'✓' if r['gate2_ok'] else '✗'}｜H136 最差 25% 月份：它 {w[0]*100:+.2f}%（H136 {w[1]*100:+.2f}%）"
          f"｜關卡 3 {r['gate3_rf0']} {'✓' if r['gate3_ok_rf0'] else '✗'}（現金 1%：{'✓' if r['gate3_ok_rf1'] else '✗'}）")
    print(f"     逐年 {r['by_year']}")


def run(period):
    md, md2, h = windows(period)
    X = proxy_monthly(md, md2)
    seg = "dev" if period == "dev" else "val"
    hs = PL.st(h, md, seg)
    print(f"H136 {period}：Sharpe {hs['sharpe']:.2f}（{'、'.join(f'{g} {PL.st(h, md, g)['sharpe']:.2f}' for g in SEGS[period])}）年化 {hs['ann']*100:+.1f}%")
    if period == "dev":
        sel = list(LABEL)
    else:
        rec = [json.loads(l) for l in open(H.LOG) if f'"{NAME}"' in l and '"period": "dev"' in l][-1]
        sel = rec["selected"]
        if not sel:
            print("dev 沒有成員過關卡 2、3 → 依登記不跑 val")
            return
    rows = []
    for k in sel:
        r = report(k, X[k], h, md, period)
        show(r)
        rows.append(r)
        PL.save(period, f"X_{US.get(k, 'BASKET')}", md, PL.sleeve_df(X[k]), dict(form="跨資產固定做多（美國 ETF 台幣代理，扣 0.5%／年）", batch=64))
    C = np.corrcoef(np.vstack([X[k][PL._mask(md, seg, *[X[j] for j in US])] for k in US]))
    print("  成員之間的相關：" + "；".join(f"{a}-{b} {C[i, j]:+.2f}" for i, a in enumerate(US) for j, b in enumerate(US) if j > i))
    trk = tracking(*(("2018-01-01", "2019-12-31") if period == "dev" else ("2020-01-01", "2023-12-31")))
    print("  關卡 4（台灣 ETF 對代理）：" + "；".join(f"{k} {v['etf']} 相關 {v['corr']:.2f} 年化差 {v['gap_ann']*100:+.1f}% 近 60 日均成交 {v['money60_m']:.0f} 百萬" for k, v in trk.items()))
    if period == "dev":
        selected = [r["name"] for r in rows if r["gate2_ok"] and r["gate3_ok_rf0"]]
        print(f"  dev 過關卡 2、3 → 進 val：{selected}")
        SL.log(dict(kind="strategy", period="dev", strategy=NAME, note="第六十四批 事前登記", result=rows, tracking=trk,
                    selected=selected, trials_cum=TRIALS))
    else:
        for r in rows:
            r["verdict"] = "✓" if r["gate3_ok_rf0"] else "✗"
            print(f"  {r['name']} val 判定（關卡 3）：{r['verdict']}")
        SL.log(dict(kind="strategy", period="val", strategy=NAME, note="第六十四批 dev 選出的", result=rows, tracking=trk))


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "dev")
