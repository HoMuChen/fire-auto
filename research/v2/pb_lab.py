"""第七十一批 H183：股價淨值比 < 1 池的逆勢策略（10 格資金）。規則見 HYPOTHESES.md 第七十一批（事前登記）。
池：PB1（流動池裡 0 < PBR < 1，每天認定）、L（全流動池，對照）。
模型：短線逆勢事件（第六十八批 R 族 36 個）＋每月池內反轉輪動（V1 20 日最弱、V2 12-1 月最弱、V3 距 252 日高最遠，各 10 檔）。
模擬器沿用第六十八批 pool_event_lab.simulate／evaluate。

用法：.venv/bin/python research/v2/pb_lab.py cache dev|val
      .venv/bin/python research/v2/pb_lab.py dev
      .venv/bin/python research/v2/pb_lab.py val
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
import pool_event_lab as PE  # noqa: E402
import strategy_lab as SL  # noqa: E402
import vpool_lab as V  # noqa: E402
import xs  # noqa: E402

RES = H.D / "results"
NAME = "第七十一批 H183 股價淨值比 < 1 池的逆勢"
TRIALS = 1701679
POOLS = ("PB1", "L")
R_ENTRY, R_EXIT, FILTS, R_MAX = ("R1", "R2", "R3"), ("Y1", "Y2", "Y3", "Y4"), ("F0", "F1", "F2"), 20
ROT = ("V1", "V2", "V3")
NROT, NEVER, NSEL = 10, 10 ** 9, 10
LABEL = {"V1": "20日報酬最低", "V2": "12-1月報酬最低", "V3": "距252日高最遠"}


# ═════════ 資料 ═════════
def cache(period):
    end = H.PERIODS[period][1]
    t0 = time.time()
    P = xs.load_panel(end, fundamentals=True)
    cols, days = P["close"].columns, P["close"].index
    assert list(cols) == sorted(cols)
    fx, trade, _ = V.gap_fix(P, cols)
    LU, LD, _, _ = V.limit_flags(fx)
    print(f"載入 {time.time()-t0:.0f}s；{len(days)} 個交易日、{len(cols)} 檔", flush=True)
    inliq = P["liq"].fillna(False).astype(bool) & (P["nbars"] >= 250)
    Lp = inliq & trade
    pb = P["pbr"]
    PM = {"PB1": Lp & (pb > 0) & (pb < 1), "L": Lp}
    U = np.flatnonzero(Lp.values.any(axis=0))
    uc = cols[U]
    O, Hh, Cc = fx["open"][uc], fx["high"][uc], fx["close"][uc]
    tr_ok = trade[uc]
    Cf = Cc.ffill()
    MA = {n: Cf.rolling(n, min_periods=n).mean() for n in (5, 20, 200)}
    sd20 = Cf.rolling(20, min_periods=20).std(ddof=0)
    rsi2, rsi14 = PE._rsi(Cf, 2), PE._rsi(Cf, 14)
    ent = {"R1": rsi2 < 10, "R2": Cf < MA[20] - 2 * sd20, "R3": rsi14 < 30}
    ex = {"Y1": Cf > MA[5], "Y2": rsi2 > 70, "Y4": Cf > MA[20]}
    ent = {k: (v & tr_ok).fillna(False).values for k, v in ent.items()}
    ex = {k: v.fillna(False).values for k, v in ex.items()}
    # ── 每月池內反轉輪動（換股日前一交易日收盤選）──
    rd = xs.rebalance_dates(days, days[0], days[-1])
    sday = np.zeros(len(days), bool)
    sday[[days.get_loc(d) - 1 for d in rd if days.get_loc(d) > 0]] = True
    feat = {"V1": Cf / Cf.shift(20) - 1, "V2": Cf.shift(21) / Cf.shift(252) - 1, "V3": Cf / Cf.rolling(252, min_periods=200).max()}
    sd, liq_u, trv = sday[:, None], inliq[uc].values, tr_ok.values
    prio = {}
    for v, p in itertools.product(ROT, POOLS):
        rk = feat[v].where(PM[p][uc]).rank(axis=1, method="first")          # 低 → 高；平手依代號
        r = rk.values
        ent[f"{v}{p}i"] = (r <= NROT) & sd
        ex[f"{v}{p}o"] = (~liq_u | (trv & ~(r <= NROT))) & sd
        if p == "PB1":
            ex[f"{v}{p}o"] |= sd & trv & ~PM[p][uc].values                  # 離開 PB1 池（PBR ≥ 1）
        prio[f"{v}{p}"] = r.astype(np.float32)
    # ── 描述：池每月等權買進持有（換股日開盤 → 下個換股日開盤，不扣成本）──
    Of = O.fillna(Cf.shift(1))
    pool_m = {p: [] for p in POOLS}
    for d, d2 in zip(rd[:-1], rd[1:]):
        k = days.get_loc(d)
        r = (Of.loc[d2] / O.loc[d] - 1).clip(-0.95, 10)
        for p in POOLS:
            m = PM[p][uc].values[k - 1] & np.isfinite(r.values)
            pool_m[p].append(r.values[m].mean() if m.sum() else np.nan)
    tr = pd.read_parquet(H.D / "taiex_tr.parquet")
    tr["date"] = pd.to_datetime(tr.date)
    mkt = tr.drop_duplicates("date").set_index("date").price.astype(float).sort_index().reindex(days).ffill()
    pool_m = pd.DataFrame(pool_m, index=pd.DatetimeIndex(rd[:-1]))
    pool_m["TR"] = [mkt.loc[d2] / mkt.loc[d] - 1 for d, d2 in zip(rd[:-1], rd[1:])]
    lastv = np.array([np.flatnonzero(trv[:, j]).max() if trv[:, j].any() else -1 for j in range(len(uc))])
    D = dict(days=days, cols=uc, period=period, O=O.values.astype(np.float64), H=Hh.fillna(Cf).values, C=Cf.values, trade=trv,
             LU=LU[uc].fillna(False).values, LD=LD[uc].fillna(False).values, atr=None, lastv=lastv, ret20=(Cf / Cf.shift(20) - 1).values,
             rsi2=rsi2.values, ent=ent, ex=ex, prio=prio, f2=(Cf > MA[200]).values, mkt=mkt.values,
             f1=(mkt > mkt.rolling(200, min_periods=200).mean()).values, pool={p: M[uc].values for p, M in PM.items()},
             pool_n={p: M.values.sum(axis=1) for p, M in PM.items()}, pool_m=pool_m)
    with open(RES / f"pb_{period}.pkl", "wb") as fh:
        pickle.dump(D, fh)
    print(f"快取完成 {time.time()-t0:.0f}s；股票 {len(U)} 檔；PB1 池每天平均 {PM['PB1'].values.sum(1)[sday].mean():.0f} 檔"
          f"（流動池 {Lp.values.sum(1)[sday].mean():.0f}）", flush=True)


def load(period):
    with open(RES / f"pb_{period}.pkl", "rb") as fh:
        return pickle.load(fh)


# ═════════ 模型 ═════════
def models():
    out = []
    for p in POOLS:
        for e, x, f in itertools.product(R_ENTRY, R_EXIT, FILTS):
            out.append(dict(pool=p, fam="R", entry=e, exit=x, filt=f, max_hold=R_MAX))
        for v in ROT:
            out.append(dict(pool=p, fam="V", entry=f"{v}{p}i", exit=f"{v}{p}o", prio=f"{v}{p}", max_hold=NEVER,
                            label=f"{p}池 每月{LABEL[v]}的 {NROT} 檔"))
    return out


def spec(r):
    return {k: r[k] for k in ("pool", "fam", "entry", "exit", "filt", "prio", "max_hold", "label") if k in r and r[k] == r[k] and r[k] is not None}


def describe(D, seg):
    pm = D["pool_m"].loc[PE.SEG[seg][0]:PE.SEG[seg][1]].dropna()
    out = {}
    for a, b in (("PB1", "TR"), ("PB1", "L"), ("L", "TR")):
        d = pm[a] - pm[b]
        out[f"{a}-{b}"] = dict(ann=float(d.mean() * 12), t=float(d.mean() / d.std(ddof=1) * np.sqrt(len(d))),
                               yrs=f"{int((d.groupby(d.index.year).sum() > 0).sum())}/{d.index.year.nunique()}")
    n = D["pool_n"]["PB1"][D["days"].slice_indexer(*PE.SEG[seg])]
    print(f"描述：PB1 池每天平均 {n.mean():.0f} 檔（最少 {n.min()}）；池每月等權買進持有（不扣成本）："
          + "；".join(f"{k} 年化 {v['ann']*100:+.1f}%（t {v['t']:.2f}，正年 {v['yrs']}）" for k, v in out.items()), flush=True)
    return out


def fam_table(rows):
    T = pd.DataFrame(rows)
    g = T.groupby(["pool", "fam"]).alpha_t.mean().unstack()
    g["全部"] = T.groupby("pool").alpha_t.mean()
    print("各池 × 族平均 alpha t：\n" + g.round(2).to_string(), flush=True)
    return g


def dev():
    t0 = time.time()
    D = load("dev")
    M = models()
    desc = describe(D, "dev")
    rows = PE.run_all(D, M, "dev")
    print(f"dev {PE.SEG['dev']}；模型 {len(M)} 個（{time.time()-t0:.0f}s）", flush=True)
    g = fam_table(rows)
    fit = bool(g.loc["PB1", "全部"] > g.loc["L", "全部"])
    print(f"池的預測（PB1 > L，全部模型平均）dev：{'成立' if fit else '不成立'}", flush=True)
    T = pd.DataFrame(rows)
    print("PB1 池依進場／出場／濾網平均 alpha t：", {k: T[T.pool == "PB1"].groupby(k).alpha_t.mean().round(2).to_dict() for k in ("entry", "exit", "filt")}, flush=True)
    for p in POOLS:
        print(f"\n{p} 池 alpha t 前 8：", flush=True)
        for _, r in T[T.pool == p].sort_values("alpha_t", ascending=False).head(8).iterrows():
            PE.show(r)
    ok = [r for r in sorted(rows, key=lambda r: -r["alpha_t"]) if PE.passes(r)]
    print(f"\n過 dev 三條：{len(ok)} 個（" + "、".join(f"{p} {sum(r['pool'] == p for r in ok)}" for p in POOLS) + "）；L 池不進 val", flush=True)
    sel, ser = [], []
    for r in ok:
        if r["pool"] != "PB1":
            continue
        _, s = PE.evaluate(D, spec(r), "dev")
        if any(np.corrcoef(s, z)[0, 1] > 0.8 for z in ser):
            continue
        r["rho_h136"] = PE.monthly_corr_h136(D, s, "dev")
        sel.append(r)
        ser.append(s)
        PE.show(r, f"｜和 H136 月相關 {r['rho_h136']:+.2f}｜逐年超額 {r['by_year']}")
        if len(sel) == NSEL:
            break
    keep = ("name", "pool", "fam", "alpha_t", "sharpe", "yrs_win", "cagr", "mdd", "n_trades", "tr_mean", "tr_med", "win", "hold", "exposure")
    SL.log(dict(kind="strategy", period="dev", strategy=NAME, note="第七十一批 H183 事前登記", describe=desc, fam=g.round(3).to_dict(), fit_dev=fit,
                n_pass=len(ok), selected=[spec(r) for r in sel], extra=[{k: r[k] for k in keep + ("rho_h136", "by_year")} for r in sel],
                all=[{k: r[k] for k in keep} for r in rows], trials_cum=TRIALS))


def val():
    rec = [json.loads(l) for l in open(H.LOG) if f'"{NAME}"' in l and '"period": "dev"' in l][-1]
    D = load("val")
    desc = describe(D, "val")
    rows = PE.run_all(D, models(), "val")
    g = fam_table(rows)
    fit = bool(g.loc["PB1", "全部"] > g.loc["L", "全部"])
    print(f"池的預測 val：{'成立' if fit else '不成立'} → 整體{'成立' if fit and rec['fit_dev'] else '不成立'}", flush=True)
    out = []
    if not rec["selected"]:
        print("dev 沒有候選，依登記不跑 val 的策略判定", flush=True)
    for m in rec["selected"]:
        r, s = PE.evaluate(D, m, "val")
        r["rho_h136"] = PE.monthly_corr_h136(D, s, "val")
        r["verdict"] = "✓" if r["alpha_t"] >= 2 and r["sharpe"] >= 1.0 else "✗"
        out.append(r)
        PE.show(r, f"｜和 H136 月相關 {r['rho_h136']:+.2f}｜逐年超額 {r['by_year']} → {r['verdict']}")
    if out:
        print(f"val 通過 {sum(r['verdict'] == '✓' for r in out)}／{len(out)}")
    keep = ("name", "pool", "fam", "alpha_t", "sharpe", "yrs_win", "cagr", "mdd", "n_trades", "tr_mean", "tr_med", "win", "hold", "exposure")
    SL.log(dict(kind="strategy", period="val", strategy=NAME, note="第七十一批 H183 dev 選出的＋池的預測", describe=desc, fam=g.round(3).to_dict(),
                fit_val=fit, result=[{k: r[k] for k in keep + ("rho_h136", "by_year", "verdict")} for r in out]))


if __name__ == "__main__":
    warnings.simplefilter("ignore", RuntimeWarning)
    cmd = sys.argv[1]
    if cmd == "cache":
        cache(sys.argv[2])
    else:
        {"dev": dev, "val": val}[cmd]()
