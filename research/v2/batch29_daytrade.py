"""第二十九批 H131：短線券商（隔日沖）大買 → 隔天開高走低，規則見 HYPOTHESES.md 第二十九批。

資料：data/broker/{sid}.parquet（broker_cache.py 格式，2021-06-30 起、舊個股清單 1,439 檔）＋ data/v2 價格。
期間：單一期間（使用者 2026-11-02 指示，含原 holdout 區段）；只跑一次、不調參。

隔日沖分點 G(M)：每月 M 用 M−6 ~ M−1 的「t 買超、t' 賣超」比率，取買超金額前 100 大中比率最高的 12 家。
每筆 t 的貢獻歸到 t' 所在的月份（t' 收盤後才知道有沒有賣），所以 G(M) 只用到 M 之前已知的資料。

用法：python research/v2/batch29_daytrade.py            # 主判定（寫紀錄）
      python research/v2/batch29_daytrade.py --posthoc  # 事後描述：漲停股另配對照（讀主判定存的結果，不寫紀錄）
"""
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402

BROKER = H.BASE / "data" / "broker"
WIN, NCAND, NG = 6, 100, 12
M0 = pd.Period("2022-01", "M")
EVENTS = {"H131": ("share", 0.10, "G淨買超≥當日量10%"), "H131b": ("share", 0.20, "G淨買超≥當日量20%"),
          "H131c": ("gnet", 500_000, "G淨買超≥500張")}
CTRL = 0.01
YS = ("oc1", "gap1", "cc1", "oc5", "fade")
DAYTRADE_COST = 0.001425 * 2 + 0.0015
_W = {}


def _cal():
    d = pd.read_parquet(BROKER / "2330.parquet", columns=["date"]).date.astype(str).unique()
    cal = np.array(sorted(d))
    return cal, {s: i for i, s in enumerate(cal)}, pd.PeriodIndex(pd.to_datetime(cal), freq="M")


def _broker(sid, names=False):
    cols = ["date", "securities_trader_id", "buy", "sell"] + (["securities_trader"] if names else [])
    b = pd.read_parquet(BROKER / f"{sid}.parquet", columns=cols)
    b["di"] = b.date.astype(str).map(_W["ix"])
    b = b.dropna(subset=["di"])
    b["di"] = b.di.astype(int)
    b["net"] = b.buy.astype(np.int64) - b.sell.astype(np.int64)
    return b.rename(columns={"securities_trader_id": "bid"})


def _price_di(df):
    return df.date.dt.strftime("%Y-%m-%d").map(_W["ix"])


# ───────── 第 1 輪：每檔每分點每月的買超金額與隔日沖金額 ─────────
def pass1(sid):
    if not (BROKER / f"{sid}.parquet").exists() or not (H.D / "raw" / f"{sid}.parquet").exists():
        return None, {}
    px = pd.read_parquet(H.D / "raw" / f"{sid}.parquet", columns=["date", "close"])
    px = px[px.close > 0]
    px = pd.Series(px.close.values, index=pd.to_datetime(px.date).dt.strftime("%Y-%m-%d").map(_W["ix"]).values)
    px = px[px.index.notna()]
    px.index = px.index.astype(int)
    b0 = _broker(sid, names=True)
    names = b0.drop_duplicates("bid", keep="last").set_index("bid").securities_trader.to_dict()
    b = b0.groupby(["di", "bid"], as_index=False).net.sum()
    b1 = b.assign(di=b.di - 1).rename(columns={"net": "net1"})
    m = b.merge(b1, on=["di", "bid"], how="left")
    m["close"] = m.di.map(px)
    m = m[(m.net > 0) & m.close.notna() & (m.di + 1 < len(_W["cal"]))]
    pos = m.net.astype(float)
    m = m.assign(buyv=pos * m.close, revv=np.minimum(pos, (-m.net1.fillna(0)).clip(lower=0)) * m.close,
                 ym=_W["calm"][m.di.values + 1].astype(str))
    return m.groupby(["ym", "bid"])[["buyv", "revv"]].sum(), names


def select(A):
    """每月 G(M)：M−6 ~ M−1（依 t' 月份）買超金額前 NCAND 大中，隔日沖比率最高的 NG 家（同分依代號）。"""
    A = A.reset_index()
    A["ym"] = pd.PeriodIndex(A.ym, freq="M")
    last = _W["calm"][-1]
    G, R = {}, {}
    for M in pd.period_range(M0, last, freq="M"):
        w = A[(A.ym >= M - WIN) & (A.ym <= M - 1)].groupby("bid")[["buyv", "revv"]].sum()
        c = w.nlargest(NCAND, "buyv").assign(r=lambda x: x.revv / x.buyv).reset_index()
        c = c.sort_values(["r", "bid"], ascending=[False, True])
        G[M] = list(c.bid[:NG])
        R[M] = (c.r.iloc[:NG].mean(), c.r.median())
    return G, R


# ───────── 第 2 輪：每檔每日的 G 淨買超、當日狀態格、隔天結果 ─────────
def pass2(sid):
    if not (BROKER / f"{sid}.parquet").exists():
        return None
    df = H.load_stock(sid, _W["end"])
    if df is None or len(df) < 260:
        return None
    b = _broker(sid)
    b["ym"] = _W["calm"][b.di.values].astype(str)
    g = b.merge(_W["gpairs"], on=["ym", "bid"]).groupby("di").net.sum()
    g.index = g.index.astype(float)                  # di 在價格表是 float（日曆外為 NaN）
    di = _price_di(df)
    c, o, v = df.close, df.open, df.volume.astype(float)
    rc = df.raw_close
    rr = rc / rc.shift() - 1
    lim = rc >= rc.shift() * (1 + np.where(df.date < H.LIMIT_CHANGE, 0.07, 0.10) - 0.005)
    rb = np.select([lim, rr >= 0.05, rr >= 0.02, rr >= -0.02], ["LU", "5~LU", "2~5", "±2"], "<-2")
    vr = v / v.shift(1).rolling(20).mean()
    vb = np.select([vr >= 3, vr >= 1.5], ["V3", "V1.5"], "V<1.5")
    o1, c1 = o.shift(-1), c.shift(-1)
    out = pd.DataFrame({
        "date": df.date, "sid": sid, "di": di,
        "gnet": di.map(g).fillna(0).values,
        "vol": v, "cell": pd.Series(rb, index=df.index) + "|" + pd.Series(vb, index=df.index), "lu": lim,
        "oc1": c1 / o1 - 1, "gap1": o1 / c - 1, "cc1": c1 / c - 1, "oc5": c.shift(-5) / o1 - 1,
        "bias5": c / c.rolling(5).mean() - 1,
        "ok": (v.rolling(H.LIQ_WIN).mean() >= H.LIQ_SHARES) & (np.arange(len(df)) >= 249) & (di.shift(-1) == di + 1),
    })
    out["fade"] = ((out.gap1 > 0) & (out.oc1 < 0)).astype(float).where(out.oc1.notna())
    out = out[out.ok & out.di.notna() & (out.vol > 0)].copy()
    out = out[pd.PeriodIndex(out.date, freq="M") >= M0]
    out["share"] = out.gnet / out.vol
    return out.drop(columns=["ok", "di"])


def _pool(fn, sids):
    with mp.get_context("fork").Pool(8) as pool:
        return pool.map(fn, sids, chunksize=8)


def stats(E, y):
    d = (E[y] - E[f"{y}_c"]).dropna()
    m = d.groupby(E.loc[d.index, "date"]).mean()
    yr = m.groupby(m.index.year).mean()
    return dict(n=len(d), days=len(m), ev=E.loc[d.index, y].mean(), ctrl=E.loc[d.index, f"{y}_c"].mean(),
                diff_ev=d.mean(), diff=m.mean(), t=m.mean() / m.std() * np.sqrt(len(m)) if len(m) > 2 else np.nan,
                yrs_neg=f"{int((yr < 0).sum())}/{len(yr)}", by_year={int(k): v for k, v in yr.items()})


def line(lab, s):
    return (f"  {lab:<24} n={s['n']:>6} 天={s['days']:>4}  事件 {s['ev']*100:+6.2f}%  對照 {s['ctrl']*100:+6.2f}%"
            f"  差（日平均）{s['diff']*100:+6.2f}pp  t {s['t']:+6.2f}  負年 {s['yrs_neg']}")


def main():
    end = pd.Timestamp.today().strftime("%Y-%m-%d")
    cal, ix, calm = _cal()
    _W.update(cal=cal, ix=ix, calm=calm, end=end)
    sids = [s for s in H.universe() if (BROKER / f"{s}.parquet").exists()]
    t0 = time.time()
    A, buf, names = None, [], {}
    with mp.get_context("fork").Pool(8) as pool:
        for a, nm in pool.imap_unordered(pass1, sids, chunksize=8):
            names.update(nm)
            if a is not None:
                buf.append(a)
            if len(buf) >= 100:                      # 邊收邊加總，避免一次握著 ~5 千萬列
                A = pd.concat(([A] if A is not None else []) + buf).groupby(level=[0, 1]).sum()
                buf = []
    A = pd.concat(([A] if A is not None else []) + buf).groupby(level=[0, 1]).sum()
    G, R = select(A)
    print(f"H131 隔日沖：{len(sids)} 檔有分點資料，分點日曆 {cal[0]} ~ {cal[-1]}；第 1 輪 {time.time()-t0:.0f}s")
    _W["gpairs"] = pd.DataFrame([(str(M), b) for M, bs in G.items() for b in bs], columns=["ym", "bid"])
    freq = pd.Series([b for bs in G.values() for b in bs]).value_counts()
    print(f"  G(M) {len(G)} 個月；入選分點的隔日沖比率平均 {np.mean([r[0] for r in R.values()])*100:.0f}%"
          f"（前 100 大中位 {np.mean([r[1] for r in R.values()])*100:.0f}%）")
    print("  最常入選：" + "、".join(f"{names.get(b, b)}({b}) {n}/{len(G)}" for b, n in freq.head(15).items()))
    X = pd.concat([x for x in _pool(pass2, sids) if x is not None], ignore_index=True)
    print(f"  第 2 輪 {time.time()-t0:.0f}s：{len(X):,} 檔日（{X.date.min().date()} ~ {X.date.max().date()}）")
    ctrl = X[X.share.abs() <= CTRL]
    cm = ctrl.groupby(["date", "cell"])[list(YS)].mean()
    cn = ctrl.groupby(["date", "cell"]).size()
    cm = cm[cn >= 3].add_suffix("_c")
    res = {}
    for name, (col, thr, desc) in EVENTS.items():
        E = X[X[col] >= thr].join(cm, on=["date", "cell"]).dropna(subset=["oc1_c"])
        raw_n = int((X[col] >= thr).sum())
        S = {y: stats(E, y) for y in YS}
        s = S["oc1"]
        ok = s["diff"] <= -0.005 and s["t"] <= -2.5 and int(s["yrs_neg"].split("/")[0]) >= 4
        res[name] = dict(desc=desc, events=raw_n, matched=len(E), lu_share=E.lu.mean(), pass_=bool(ok), **S)
        print(f"\n== {name}（{desc}）：{raw_n:,} 個事件，配到對照 {len(E):,}；漲停收盤占 {E.lu.mean()*100:.0f}%")
        for y, lab in (("oc1", "oc1 隔天開→收（主）"), ("gap1", "gap1 隔天跳空"), ("cc1", "cc1 隔天收→收"),
                       ("oc5", "oc5 隔天開盤買抱 5 天"), ("fade", "P（開高走低）")):
            print(line(lab, S[y]))
        print(f"  主判定：oc1 差 {s['diff']*100:+.2f}pp（≤ −0.5）、t {s['t']:+.2f}（≤ −2.5）、負年 {s['yrs_neg']}（≥ 4/5）"
              f" → {'✓' if ok else '✗'}")
        print("  年別 oc1 差（pp）：" + "  ".join(f"{k} {v*100:+.2f}" for k, v in s["by_year"].items()))
        if name == "H131":
            ev_oc1 = E.oc1
            print(f"  可交易性（描述）：事件 oc1 平均 {ev_oc1.mean()*100:+.2f}%、中位 {ev_oc1.median()*100:+.2f}%；"
                  f"現股當沖先賣後買成本約 {DAYTRADE_COST*100:.3f}%")
            for lab, sub in (("漲停收盤", E[E.lu]), ("非漲停", E[~E.lu]), ("乖離 > 13%", E[E.bias5 > 0.13]),
                             ("乖離 ≤ 13%", E[E.bias5 <= 0.13])):
                if len(sub) > 2:
                    print(line(f"子群 {lab} oc1", stats(sub, "oc1")))
    with open(H.LOG, "a") as f:
        for name, r in res.items():
            f.write(json.dumps(dict(ts=time.strftime("%Y-%m-%d %H:%M:%S"), kind="event_diff", period="2022-01~（單一期間，含原 holdout）",
                                    strategy=f"{name} 隔日沖大買", note="第二十九批 事前登記（使用者同意不分 dev/val）",
                                    top_branches={names.get(b, b): int(n) for b, n in freq.head(15).items()}, **r),
                               ensure_ascii=False, default=lambda v: round(float(v), 5)) + "\n")
    out = H.D / "results"
    out.mkdir(exist_ok=True)
    X.to_parquet(out / "b29_daytrade.parquet")


def posthoc_lu():
    """事後描述（不判定，不寫紀錄）：看過 H131 結果後才做。H131 事件多半不是隔日沖（隔天 G 仍淨買 54%），
    只有漲停收盤的事件隔天 G 賣掉約 72%；但 15 格配對要求同格 ≥ 3 檔對照，漲停格常湊不滿。
    這裡只比漲停股：同一天、漲停收盤、G 沒參與（|G 淨買超| ≤ 1%）的對照 ≥ 1 檔。"""
    X = pd.read_parquet(H.D / "results" / "b29_daytrade.parquet")
    L = X[X.lu]
    c = L[L.share.abs() <= CTRL].groupby("date")[list(YS)].mean().add_suffix("_c")
    print(f"\n== H131 事後描述：漲停收盤股（{len(L):,} 檔日；G 沒參與的對照 {int((L.share.abs() <= CTRL).sum()):,}）")
    for lab, m in (("G ≥ 當日量 10%", L.share >= 0.10), ("G ≥ 當日量 20%", L.share >= 0.20), ("G ≥ 500 張", L.gnet >= 500_000)):
        E = L[m].join(c, on="date").dropna(subset=["oc1_c"])
        print(f"  {lab}：{int(m.sum()):,} 筆，配到 {len(E):,}")
        for y, nm in (("oc1", "oc1 隔天開→收"), ("gap1", "gap1 隔天跳空"), ("cc1", "cc1 隔天收→收"),
                      ("oc5", "oc5 隔天開盤買抱 5 天"), ("fade", "P（開高走低）")):
            print(line(nm, stats(E, y)))


if __name__ == "__main__":
    if "--posthoc" in sys.argv:
        posthoc_lu()
    else:
        main()
