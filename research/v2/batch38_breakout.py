"""第三十八批 H140/H141：營收驚喜狀態（SUR ≥ 2）下的 52 週新高突破，訊號觸發就買（不限檔數）。規則見 HYPOTHESES.md 第三十八批（事前登記）。

訊號用面板算（收盤判斷），成交、漲跌停、下市、成本全部交給 harness.simulate（次日開盤）。
對照：RANDOM（harness 隨機進場）、CS（同狀態、月換股時點）、CB（只有突破）；CH（同狀態＋H2 ≥ 0.95、月換股時點）是看過 dev 才加的事後描述。出場規則兩組：持有 20 日／移動停損 8%（最長 120 日）。

用法：.venv/bin/python research/v2/batch38_breakout.py dev
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
import xs  # noqa: E402
from hypotheses import h2_hi52  # noqa: E402

SUR_MIN, FRESH, P0 = 2.0, 20, "2011-01-01"
EXITS = {"20": dict(stop=None, max_hold=20), "T": dict(stop=0.08, max_hold=120)}
MAIN = {"H140": ("E", "20"), "H141": ("E", "T")}
SPECS = {f"{k}{x}": (k, x) for k in ("E", "CS", "CH", "CB", "R") for x in EXITS}   # E20=H140、ET=H141
_G = {}


def sur_panel(P):
    """照 xs.add_v17 的 SUR（次月 11 日可得、超過 75 天沒新值視為缺）。"""
    idx, cols = P["close"].index, P["close"].columns
    out = {}
    for s in cols:
        f = H.D / "revenue" / f"{s}.parquet"
        if not f.exists():
            continue
        d = pd.read_parquet(f)
        if not len(d):
            continue
        d["ym"] = pd.PeriodIndex.from_fields(year=d.revenue_year.astype(int), month=d.revenue_month.astype(int), freq="M")
        d = d.drop_duplicates("ym", keep="last").set_index("ym").revenue.astype(float)
        d = d.reindex(pd.period_range(d.index.min(), d.index.max(), freq="M"))
        dif = d - d.shift(12)
        h = dif.shift(1).rolling(24, min_periods=12)
        avail = (d.index + 1).to_timestamp() + pd.Timedelta(days=10)
        v = xs._align(pd.Series(((dif - h.mean()) / h.std()).values, index=avail), idx)
        if v is not None:
            out[s] = v
    return pd.DataFrame(out).reindex(index=idx, columns=cols)


def signals(P, p0, p1):
    """回傳 {E, CS, CH, CB, elig}：訊號日（收盤判斷）的布林面板。"""
    elig = P["liq"].fillna(False).astype(bool) & (P["nbars"] >= 250)
    h2 = h2_hi52(P)
    hi = h2 >= 1 - 1e-12
    fresh = hi & (hi.astype(float).shift(1).rolling(FRESH, min_periods=1).max() == 0)
    state = sur_panel(P) >= SUR_MIN
    idx = P["close"].index
    cs = pd.DataFrame(False, index=idx, columns=P["close"].columns)
    for d in xs.rebalance_dates(idx, pd.Timestamp(p0), pd.Timestamp(p1) + pd.Timedelta(days=40)):
        k = idx.get_loc(d)
        if k > 0:
            cs.iloc[k - 1] = True
    return {"E": state & fresh & elig, "CS": state & cs & elig, "CH": state & (h2 >= 0.95) & cs & elig, "CB": fresh & elig, "elig": elig}


def leak_check(P, p0, p1, cuts=("2013-06-28", "2016-03-31", "2018-09-28")):
    full = signals(P, p0, p1)
    bad = []
    for c in cuts:
        part = signals({k: v.loc[:c] for k, v in P.items()}, p0, c)
        for k in ("E", "CB"):
            a, b = part[k], full[k].loc[part[k].index]
            n = int((a != b).to_numpy().sum())
            if n:
                bad.append((c, k, n))
    return bad


def _work(sid):
    df = H.load_stock(sid, _G["end"])
    if df is None or len(df) < H.MIN_BARS + 2:
        return []
    dates = df.date
    rnd = np.array(H._random_sig(sid, len(df))) == "buy"
    elig = _G["sig"]["elig"][sid].reindex(dates).fillna(False).values if sid in _G["sig"]["elig"] else np.zeros(len(df), bool)
    out = []
    for name, (k, x) in SPECS.items():
        if k == "R":
            m = rnd & elig
        elif sid in _G["sig"][k]:
            m = _G["sig"][k][sid].reindex(dates).fillna(False).values
        else:
            continue
        if not m.any():
            continue
        sig = np.where(m, "buy", None).tolist()
        for t in H.simulate(df, sig, EXITS[x]["stop"], EXITS[x]["max_hold"], False, _G["period"], _G["end"]):
            out.append((name, sid) + t)
    return out


def nw_t(x, lag=3):
    x = np.asarray(x, float)
    e, n = x - x.mean(), len(x)
    s = e @ e / n
    for k in range(1, min(lag, n - 1) + 1):
        s += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    return x.mean() / np.sqrt(s / n)


def mdiff(a, b):
    """同月進場的每筆超額平均相減（只取兩邊都有交易的月）。"""
    ma, mb = a.groupby(a.buy.dt.to_period("M")).ex.mean(), b.groupby(b.buy.dt.to_period("M")).ex.mean()
    d = (ma - mb).dropna()
    ya = a.groupby(a.buy.dt.year).ex.mean() - b.groupby(b.buy.dt.year).ex.mean()
    ya = ya.dropna()
    return dict(diff=d.mean(), t=d.mean() / d.std(ddof=1) * np.sqrt(len(d)), nw_t=nw_t(d.values), months=len(d),
                yrs_pos=f"{int((ya > 0).sum())}/{len(ya)}", by_year={str(y): v for y, v in ya.items()})


def describe(x, idx, months):
    ex = x.ex.sort_values(ascending=False)
    k5 = max(1, int(len(ex) * 0.05))
    cnt = np.zeros(len(idx) + 1)
    np.add.at(cnt, idx.searchsorted(x.buy.values), 1)
    np.add.at(cnt, idx.searchsorted(x.sell.values), -1)
    conc = np.cumsum(cnt)[:-1]
    conc = conc[(idx >= x.buy.min()) & (idx <= x.buy.max())]
    return dict(n=len(x), per_month=len(x) / months, win=(x.ret > 0).mean(), avg=x.ret.mean(), ex=x.ex.mean(),
                ex_med=x.ex.median(), top5_share=ex.iloc[:k5].sum() / ex.sum() if ex.sum() > 0 else np.nan,
                hold=x.bars.mean(), reasons=x.reason.value_counts().to_dict(),
                conc_mean=conc.mean(), conc_p90=np.percentile(conc, 90), conc_max=conc.max(),
                by_year={str(y): v for y, v in x.groupby(x.buy.dt.year).ex.mean().items()},
                n_year={str(y): int(v) for y, v in x.groupby(x.buy.dt.year).size().items()})


def main():
    period = sys.argv[1] if len(sys.argv) > 1 else "dev"
    if period == "holdout":
        raise PermissionError("holdout 不開（登記：SUR 與 52 週高的 holdout 已在 H136 看過）")
    p0, p1 = max(H.PERIODS[period][0], P0), H.PERIODS[period][1]
    t0 = time.time()
    P = xs.load_panel(p1)
    print(f"載入 {time.time()-t0:.0f}s", flush=True)
    bad = leak_check(P, p0, p1)
    if bad:
        raise RuntimeError(f"前視檢查失敗：{bad}")
    print(f"前視檢查通過 {time.time()-t0:.0f}s", flush=True)
    sig = signals(P, p0, p1)
    _G.update(sig=sig, end=p1, period=(p0, p1))
    with mp.get_context("fork").Pool(8) as pool:
        res = pool.map(_work, list(P["close"].columns), chunksize=16)
    T = pd.DataFrame([t for r in res for t in r], columns=["name", "sid", "buy", "sell", "ret", "reason", "bars"])
    V = P["volume"]
    liq = V.rolling(H.LIQ_WIN, min_periods=H.LIQ_WIN).mean() >= H.LIQ_SHARES
    O = P["open"]
    mkt = (1 + (O / O.shift() - 1).where(liq.shift().fillna(False).astype(bool)).clip(-0.5, 0.5).mean(axis=1).fillna(0)).cumprod()
    T["mret"] = mkt.reindex(T.sell).values / mkt.reindex(T.buy).values - 1
    T["ex"] = T.ret - T.mret
    print(f"模擬完成 {time.time()-t0:.0f}s；{len(T):,} 筆", flush=True)
    idx = P["close"].index
    months = len(pd.period_range(p0, p1, freq="M"))
    D = {n: describe(T[T.name == n], idx, months) for n in SPECS if (T.name == n).any()}

    print(f"\n== {period} {p0} ~ {p1}（每筆 vs pit 等權開盤指數；成本已扣）")
    print(f"  {'組':<5}{'筆數':>7}{'每月':>6}{'勝率':>7}{'平均報酬':>9}{'平均超額':>9}{'中位超額':>9}{'前5%佔':>8}{'持有':>6}{'同時持倉 平均/p90/最大':>22}")
    for n, d in D.items():
        print(f"  {n:<5}{d['n']:>7,}{d['per_month']:>6.1f}{d['win']*100:>6.1f}%{d['avg']*100:>8.2f}%{d['ex']*100:>8.2f}%"
              f"{d['ex_med']*100:>8.2f}%{d['top5_share']*100:>7.0f}%{d['hold']:>6.1f}   {d['conc_mean']:>5.1f} / {d['conc_p90']:>4.0f} / {d['conc_max']:>4.0f}")
    print("  逐年每筆超額（%）：")
    for n, d in D.items():
        print(f"    {n:<5}" + " ".join(f"{y} {v*100:+6.2f}" for y, v in d["by_year"].items()))
    for n in ("E20",):
        print(f"  {n} 逐年筆數：" + " ".join(f"{y} {v}" for y, v in D[n]["n_year"].items()))
    for n in ("E20", "ET"):
        tw = D[n]["ex"] - D["R" + n[1:]]["ex"]
        print(f"  {n} 以筆數加權的 edge（每筆超額 − RANDOM 每筆超額）：{tw*100:+.2f}%")

    out = {}
    print("\n== 判定")
    for h, (k, x) in MAIN.items():
        s, r, cs = T[T.name == k + x], T[T.name == "R" + x], T[T.name == "CS" + x]
        e, c = mdiff(s, r), mdiff(s, cs)
        d = D[k + x]
        pos, tot = map(int, e["yrs_pos"].split("/"))
        ok = [d["ex"] > 0, e["diff"] > 0 and e["t"] >= 2, pos / tot >= 7 / 9, d["per_month"] >= 5]   # 正年比例同 dev（7/9；val 4 年要 4/4）
        v = ("✓ 進 val" if period == "dev" else "✓ val 過") if all(ok) else "✗"
        tv = "時點有價值" if c["diff"] > 0 and c["t"] >= 2 else ("方向對" if c["diff"] > 0 else "時點無價值")
        out[h] = dict(desc=D[k + x], edge=e, vs_cs=c, checks=ok, verdict=v, timing=tv)
        print(f"  {h}（{k}{x}）：每筆超額 {d['ex']*100:+.2f}%｜對 RANDOM edge {e['diff']*100:+.2f}%（月 t {e['t']:+.2f}、NW t {e['nw_t']:+.2f}、正年 {e['yrs_pos']}）"
              f"｜每月 {d['per_month']:.1f} 件 → {v}  {ok}")
        print(f"        對 CS（同狀態、月曆進場）{c['diff']*100:+.2f}%（月 t {c['t']:+.2f}、NW t {c['nw_t']:+.2f}、{c['months']} 個月、正年 {c['yrs_pos']}）→ {tv}")
    for n in ("CB20", "CBT"):
        if n in D:
            x = n[2:]
            e = mdiff(T[T.name == n], T[T.name == "R" + x])
            g = mdiff(T[T.name == "E" + x], T[T.name == n])
            out[n] = dict(edge=e, e_minus_cb=g)
            print(f"  描述 {n}：對 RANDOM edge {e['diff']*100:+.2f}%（t {e['t']:+.2f}）；加營收狀態後（E{x} − {n}）{g['diff']*100:+.2f}%（t {g['t']:+.2f}）")
    for x in EXITS:
        if "CH" + x in D:
            g = mdiff(T[T.name == "E" + x], T[T.name == "CH" + x])
            out["E_minus_CH" + x] = g
            print(f"  事後描述 E{x} − CH{x}（同樣營收驚喜且股價強，突破日 vs 月曆）：{g['diff']*100:+.2f}%（月 t {g['t']:+.2f}、NW t {g['nw_t']:+.2f}、正年 {g['yrs_pos']}）")
    with open(H.LOG, "a") as f:
        f.write(json.dumps(dict(ts=time.strftime("%Y-%m-%d %H:%M:%S"), kind="event_breakout", period=period,
                                strategy="H140/H141 SUR≥2 + 52週新高突破", note="第三十八批 事前登記", desc=D, judge=out),
                           ensure_ascii=False, default=lambda v: round(float(v), 5) if isinstance(v, (float, np.floating)) else str(v)) + "\n")


if __name__ == "__main__":
    main()
