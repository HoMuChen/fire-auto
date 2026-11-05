"""第七十批 H182：事件化 H136（10 格資金）。規則見 HYPOTHESES.md 第七十批（事前登記）。
分數＝H136（H2、H87、H97 在 pit 流動池內的百分位平均，≥ 2 項）；名次 ≤ N_in 進場、> N_out 出場；E1 每月／E2 每日檢查；60 日線停損有／無。
模擬器沿用第六十八批 pool_event_lab.simulate／evaluate。對照 R20（20 格＝H136 top 20）、R10（10 格 top 10）不算試驗。

用法：.venv/bin/python research/v2/h136_event_lab.py cache dev|val
      .venv/bin/python research/v2/h136_event_lab.py check      # R20 對 FS2 引擎 H136 的核對
      .venv/bin/python research/v2/h136_event_lab.py dev
      .venv/bin/python research/v2/h136_event_lab.py val
      .venv/bin/python research/v2/h136_event_lab.py clean      # 事後診斷：FS2 引擎下的緩衝版
"""
import itertools
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402
import hypotheses as Hy  # noqa: E402
import pool_event_lab as PE  # noqa: E402
import strategy_lab as SL  # noqa: E402
import vpool_lab as V  # noqa: E402
import xs  # noqa: E402

RES = H.D / "results"
NAME = "第七十批 H182 事件化 H136"
TRIALS = 1701601
FREQ, NIN, NOUT, STOP = ("E1", "E2"), (10, 20), (20, 50, 100), (False, True)
REFS = {"R20": dict(freq="E1", nin=20, nout=20, stop=False, K=20), "R10": dict(freq="E1", nin=10, nout=10, stop=False, K=10)}
NEVER = 10 ** 9
NVAL = 5


# ═════════ 資料 ═════════
def _key(freq, n, stop, io):
    return f"{freq}{io}{n}{'s' if stop else ''}"


def cache(period):
    end = H.PERIODS[period][1]
    t0 = time.time()
    P = xs.load_panel(end, fundamentals=True)
    xs.add_v17(P)
    cols, days = P["close"].columns, P["close"].index
    fx, trade, _ = V.gap_fix(P, cols)
    LU, LD, _, _ = V.limit_flags(fx)
    print(f"載入 {time.time()-t0:.0f}s；{len(days)} 個交易日、{len(cols)} 檔", flush=True)
    # ── 池與分數（特徵用原始面板，同 FS2 引擎；訊號日 t 的收盤）──
    inliq = P["liq"].fillna(False).astype(bool) & (P["nbars"] >= 250)
    pool = inliq & trade
    rd = xs.rebalance_dates(days, days[0], days[-1])
    sday = np.zeros(len(days), bool)
    sday[[days.get_loc(d) - 1 for d in rd if days.get_loc(d) > 0]] = True
    f = {g: Hy.FNS[g][0](P).replace([np.inf, -np.inf], np.nan) for g in ("H2", "H87", "H97")}
    h97 = f["H97"]
    fixed = h97.where(pd.Series(sday, index=days), axis=0).ffill()          # E2：H97 固定在最近一次換股訊號日
    assert np.allclose(fixed.values[sday], h97.values[sday], equal_nan=True)
    f["H97"] = fixed
    pct = [x.where(pool).rank(axis=1, pct=True) for x in f.values()]
    cnt = sum(x.notna().astype(int) for x in pct)
    score = (sum(x.fillna(0.0) for x in pct) / cnt.where(cnt > 0)).where((cnt >= 2) & pool)
    rank = score.rank(axis=1, ascending=False, method="first")              # 平手依代號（欄位已依代號排序）
    assert list(cols) == sorted(cols)
    U = np.flatnonzero((rank <= max(NOUT)).values.any(axis=0))
    uc = cols[U]
    rk = rank[uc].values
    Cf = fx["close"][uc].ffill()
    ma60 = Cf.rolling(60, min_periods=60).mean()
    above, below = (Cf >= ma60).values, (Cf < ma60).values
    tr_ok, liq_u = trade[uc].values, inliq[uc].values
    sd = sday[:, None]
    ent, ex = {}, {}
    for fr, stop in itertools.product(FREQ, STOP):
        for n in sorted(set(NIN) | {10, 20}):
            e = rk <= n
            if stop:
                e = e & above
            ent[_key(fr, n, stop, "i")] = e & sd if fr == "E1" else e
        for n in sorted(set(NOUT) | {10}):
            x = ~liq_u | (tr_ok & ~(rk <= n))                                  # 名次 > N_out、沒有分數、或離開流動池
            if fr == "E1":
                x = x & sd
            ex[_key(fr, n, stop, "o")] = x | below if stop else x
    tr = pd.read_parquet(H.D / "taiex_tr.parquet")
    tr["date"] = pd.to_datetime(tr.date)
    mkt = tr.drop_duplicates("date").set_index("date").price.astype(float).sort_index().reindex(days).ffill()
    lastv = np.array([np.flatnonzero(tr_ok[:, j]).max() if tr_ok[:, j].any() else -1 for j in range(len(uc))])
    D = dict(days=days, cols=uc, period=period, sday=sday, O=fx["open"][uc].values.astype(np.float64), H=fx["high"][uc].fillna(Cf).values, C=Cf.values,
             trade=tr_ok, LU=LU[uc].fillna(False).values, LD=LD[uc].fillna(False).values, atr=None, lastv=lastv, mkt=mkt.values,
             ent=ent, ex=ex, pool={"L": pool[uc].values}, prio={"S": rk}, pool_n=pool.values.sum(axis=1))
    with open(RES / f"h136_event_{period}.pkl", "wb") as fh:
        pickle.dump(D, fh)
    print(f"快取完成 {time.time()-t0:.0f}s；曾進前 {max(NOUT)} 名的股票 {len(U)} 檔；流動池平均 {pool.values.sum(1)[sday].mean():.0f} 檔；"
          f"換股訊號日 {sday.sum()} 個", flush=True)


def load(period):
    with open(RES / f"h136_event_{period}.pkl", "rb") as fh:
        return pickle.load(fh)


# ═════════ 模型 ═════════
def label(c):
    s = f"{c['freq']} 前{c['nin']}進／{c['nout']}名外出" + ("＋60日線停損" if c["stop"] else "")
    return f"{s}（{c['K']}格）" if c.get("K", PE.K) != PE.K else s


def spec(c, name=None):
    m = dict(pool="L", fam="H", entry=_key(c["freq"], c["nin"], c["stop"], "i"), exit=_key(c["freq"], c["nout"], c["stop"], "o"),
             prio="S", max_hold=NEVER, label=name or label(c))
    if "K" in c:
        m["K"] = c["K"]
    return m


def grid():
    return [dict(freq=fr, nin=a, nout=b, stop=s) for fr, a, b, s in itertools.product(FREQ, NIN, NOUT, STOP)]


def monthly(s, days):
    x = pd.Series(s, index=days)
    return (1 + x).groupby(x.index.to_period("M")).prod() - 1


def nw_t(x, lag=3):
    x = np.asarray(x, float)
    e, n = x - x.mean(), len(x)
    v = e @ e / n
    for k in range(1, min(lag, n - 1) + 1):
        v += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    return x.mean() / np.sqrt(v / n)


_G = {}


def _work(i):
    return PE.evaluate(_G["D"], _G["M"][i], _G["seg"])


def run(D, cs, seg, procs=9):
    """cs 的模型加上對照 R20、R10；回傳 (rows, 日報酬 dict)。每列附上對 R20／R10 的月差。"""
    import multiprocessing as mp
    cs = list(cs) + list(REFS.values())
    M = [spec(c) for c in cs[:-len(REFS)]] + [spec(c, k) for k, c in REFS.items()]
    _G.update(D=D, M=M, seg=seg)
    with mp.get_context("fork").Pool(procs) as pool:
        out = pool.map(_work, range(len(M)), chunksize=1)
    days = D["days"][D["days"].slice_indexer(*PE.SEG[seg])]
    ser = {r["name"]: s for r, s in out}
    mo = {k: monthly(s, days) for k, s in ser.items()}
    rows = []
    for c, (r, s) in zip(cs, out):
        r.update({k: c[k] for k in ("freq", "nin", "nout", "stop")})
        for ref in REFS:
            d = mo[r["name"]] - mo[ref]
            r[f"d_{ref}"], r[f"t_{ref}"] = float(d.mean()), float(nw_t(d.values)) if d.std() > 0 else np.nan
        yrs = (days[-1] - days[0]).days / 365.25
        r["trades_yr"] = r["n_trades"] / yrs
        rows.append(r)
    return rows, ser, days


def show(r, extra=""):
    PE.show(r, f"｜每年 {r['trades_yr']:.0f} 筆｜對 R20 月差 {r['d_R20']*100:+.2f}%（t {r['t_R20']:+.2f}）、對 R10 {r['d_R10']*100:+.2f}%（t {r['t_R10']:+.2f}）{extra}")


def check():
    """R20（模擬器裡的 H136 top 20）對 FS2 引擎 H136 月報酬的核對。"""
    for seg in ("dev", "val"):
        f = RES / f"h136_event_{seg}.pkl"
        if not f.exists():
            continue
        D = load(seg)
        _G.update(D=D, M=[spec(REFS["R20"], "R20")], seg=seg)
        r, s = _work(0)
        rho = PE.monthly_corr_h136(D, s, seg)
        led = pd.read_parquet(RES / "sleeves" / seg / "H136.parquet").ret.dropna()
        days = D["days"][D["days"].slice_indexer(*PE.SEG[seg])]
        ys = (1 + pd.Series(s, index=days)).groupby(days.year).prod() - 1
        yl = (1 + led).groupby(led.index.year).prod() - 1
        print(f"{seg}：R20 年化 {r['cagr']*100:+.1f}%、Sharpe {r['sharpe']:.2f}、alpha t {r['alpha_t']:+.2f}；和 FS2 H136 月報酬相關 {rho:.3f}", flush=True)
        print("  逐年（R20 ／ FS2 H136）：" + "  ".join(f"{y} {ys[y]*100:+.0f}/{yl.get(y, np.nan)*100:+.0f}" for y in ys.index), flush=True)


def table(rows):
    T = pd.DataFrame(rows)
    g = {k: T.groupby(k)[["alpha_t", "sharpe", "d_R20", "t_R20"]].mean().round(3) for k in ("freq", "nin", "nout", "stop")}
    for k, x in g.items():
        x = x.assign(d_R20=(x.d_R20 * 100).round(2))
        print(f"依 {k} 平均（d_R20 單位 %／月）：\n" + x.to_string(), flush=True)
    return {k: x.to_dict() for k, x in g.items()}


def dev():
    t0 = time.time()
    D = load("dev")
    rows, ser, days = run(D, grid(), "dev")
    refs, rows = rows[-len(REFS):], rows[:-len(REFS)]
    print(f"dev {PE.SEG['dev']}；模型 {len(rows)} 個（{time.time()-t0:.0f}s）", flush=True)
    print("對照：", flush=True)
    for r in refs:
        r["rho_h136"] = PE.monthly_corr_h136(D, ser[r["name"]], "dev")
        show(r, f"｜和 FS2 H136 月相關 {r['rho_h136']:.2f}")
    print("\n全部（依 alpha t）：", flush=True)
    for r in sorted(rows, key=lambda r: -r["alpha_t"]):
        show(r, " ✓" if PE.passes(r) else "")
    g = table(rows)
    ok = [r for r in rows if PE.passes(r)]
    cand = sorted([r for r in ok if r["d_R20"] > 0], key=lambda r: -r["t_R20"])[:NVAL]
    beat = [r for r in ok if r["t_R20"] >= 2]
    print(f"\n過「自己夠好」：{len(ok)}／{len(rows)}；dev 勝 H136（月差 t ≥ 2）：{len(beat)}；進 val：{len(cand)} 個", flush=True)
    for r in cand:
        r["rho_h136"] = PE.monthly_corr_h136(D, ser[r["name"]], "dev")
        show(r, f"｜和 H136 月相關 {r['rho_h136']:.2f}｜逐年超額 {r['by_year']}")
    keep = ("name", "freq", "nin", "nout", "stop", "alpha_t", "sharpe", "yrs_win", "cagr", "mdd", "d_R20", "t_R20", "d_R10", "t_R10", "trades_yr", "hold", "exposure")
    SL.log(dict(kind="strategy", period="dev", strategy=NAME, note="第七十批 H182 事前登記", n_pass=len(ok), n_beat=len(beat),
                selected=[{k: r[k] for k in ("freq", "nin", "nout", "stop")} for r in cand], refs=[{k: r[k] for k in keep + ("rho_h136",)} for r in refs],
                extra=[{k: r[k] for k in keep + ("rho_h136", "by_year")} for r in cand], groups=g,
                all=[{k: r[k] for k in keep} for r in rows], trials_cum=TRIALS))


def val():
    rec = [json.loads(l) for l in open(H.LOG) if f'"{NAME}"' in l and '"period": "dev"' in l][-1]
    if not rec["selected"]:
        print("dev 沒有候選，依登記不跑 val")
        return
    D = load("val")
    rows, ser, days = run(D, rec["selected"], "val")
    refs, rows = rows[-len(REFS):], rows[:-len(REFS)]
    print(f"val {PE.SEG['val']}；對照：", flush=True)
    for r in refs:
        r["rho_h136"] = PE.monthly_corr_h136(D, ser[r["name"]], "val")
        show(r, f"｜和 FS2 H136 月相關 {r['rho_h136']:.2f}｜逐年超額 {r['by_year']}")
    print("候選：", flush=True)
    for r in rows:
        r["rho_h136"] = PE.monthly_corr_h136(D, ser[r["name"]], "val")
        r["good"] = bool(r["alpha_t"] >= 2 and r["sharpe"] >= 1.0)
        r["beat"] = bool(r["d_R20"] > 0)
        r["verdict"] = "✓" if r["good"] and r["beat"] else ("自己夠好、但沒贏 H136" if r["good"] else "✗")
        show(r, f"｜和 H136 月相關 {r['rho_h136']:.2f}｜逐年超額 {r['by_year']} → {r['verdict']}")
    print(f"val：自己夠好 {sum(r['good'] for r in rows)}／{len(rows)}、贏 H136 {sum(r['beat'] for r in rows)}／{len(rows)}、兩個都過 {sum(r['verdict'] == '✓' for r in rows)}")
    keep = ("name", "freq", "nin", "nout", "stop", "alpha_t", "sharpe", "yrs_win", "cagr", "mdd", "d_R20", "t_R20", "d_R10", "t_R10", "trades_yr", "hold",
            "exposure", "rho_h136", "by_year")
    SL.log(dict(kind="strategy", period="val", strategy=NAME, note="第七十批 H182 dev 選出的", refs=[{k: r[k] for k in keep} for r in refs],
                result=[{k: r[k] for k in keep + ("verdict",)} for r in rows]))


def clean():
    """事後診斷（不算試驗、不做判定）：FS2 引擎（每月等權再平衡、沒有格子漂移）下的緩衝版。"""
    import feature_combo as FC
    rows = []
    for seg in ("dev", "val"):
        end = H.PERIODS[seg][1]
        P = xs.load_panel(end, fundamentals=True)
        xs.add_v17(P)
        dates, R, E, F = FC.prep(P, *PE.SEG[seg], ["H2", "H87", "H97"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            M = np.nanmean(F, axis=0)
        M[(np.isfinite(F).sum(axis=0) < 2) | ~E] = np.nan
        T = M.shape[0]
        order = np.argsort(np.where(np.isfinite(M), -M, np.inf), axis=1, kind="stable")
        rank = np.empty_like(order)
        rank[np.arange(T)[:, None], order] = np.arange(M.shape[1])[None, :]
        rank = np.where(np.isfinite(M), rank + 1, np.iinfo(np.int64).max)
        bench = np.array([R[t][E[t]].mean() for t in range(T)])
        out = {}
        for n, buf in ((20, 20), (10, 10), (10, 20), (20, 40)):
            net = np.full(T, np.nan)
            prev, turn = set(), []
            for t in range(T):
                if np.isfinite(M[t]).sum() < 50:
                    continue
                keep = [j for j in prev if rank[t, j] <= buf]
                keep = sorted(keep, key=lambda j: rank[t, j])[:n]
                cur = set(keep)
                for j in order[t]:
                    if len(cur) >= n:
                        break
                    cur.add(int(j))
                tv = 1 - len(cur & prev) / n if prev else 1.0
                net[t] = R[t, list(cur)].mean() - tv * xs.COST
                turn.append(tv)
                prev = cur
            ex = net - bench
            k = np.isfinite(ex)
            eq = np.cumprod(1 + net[k])
            out[(n, buf)] = pd.Series(net, index=dates)
            rows.append(dict(seg=seg, n=n, buf=buf, ex_ann=ex[k].mean() * 12, t=ex[k].mean() / ex[k].std(ddof=1) * np.sqrt(k.sum()),
                             sharpe=net[k].mean() / net[k].std(ddof=1) * np.sqrt(12), mdd=float((eq / np.maximum.accumulate(eq) - 1).min()),
                             turn=float(np.mean(turn[1:]))))
        for r in rows[-4:]:
            d = (out[(r["n"], r["buf"])] - out[(20, 20)]).dropna()
            r["d_top20"], r["t_top20"] = float(d.mean()), float(nw_t(d.values)) if d.std() > 0 else np.nan
            print(f"{seg} top{r['n']}／留任前{r['buf']}：年化超額 {r['ex_ann']*100:+.1f}%（t {r['t']:.2f}）Sharpe {r['sharpe']:.2f} MDD {r['mdd']*100:.0f}% "
                  f"月換手 {r['turn']:.0%}｜對 top20 月差 {r['d_top20']*100:+.2f}%（t {r['t_top20']:+.2f}）", flush=True)
    SL.log(dict(kind="describe", period="dev+val", strategy=NAME, note="第七十批事後診斷：FS2 引擎（每月等權）下的緩衝版，不算試驗", rows=rows))


if __name__ == "__main__":
    warnings.simplefilter("ignore", RuntimeWarning)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "cache":
        cache(sys.argv[2])
    else:
        {"check": check, "dev": dev, "val": val, "clean": clean}[cmd]()
