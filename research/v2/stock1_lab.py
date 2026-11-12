"""第七十九批 H192 探索：單一個股規則大搜尋（預設志超 8213），目標 Sharpe ≥ 3。規則見 HYPOTHESES.md 第七十九批（事前登記）。
資料一律截到 2023-12-31（holdout 不讀）。

用法：.venv/bin/python research/v2/stock1_lab.py [stage1|stage2] [股票代號]
"""
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402

END = "2023-12-31"
DEV, VAL = ("2012-01-01", "2019-12-31"), ("2020-01-01", "2023-12-31")
BUY, SELL = 0.001425, 0.001425 + 0.003
HOLDS = (5, 10, 20, 60)
MIN_TR = 20
NPLAC = 20
CHUNK = 4000


def rd(kind, sid):
    d = pd.read_parquet(H.D / kind / f"{sid}.parquet")
    d["date"] = pd.to_datetime(d["date"])
    return d[d.date <= END]


# ───────── 資料與條件 ─────────
def load(sid):
    raw = rd("raw", sid).set_index("date")
    raw = raw[raw.close > 0]
    adj = rd("adj", sid).set_index("date").reindex(raw.index)
    idx = raw.index
    C, Hh, L, V = adj.close, adj.high, adj.low, raw.volume.astype(float)
    cond = {}
    ma = {n: C.rolling(n).mean() for n in (5, 10, 20, 60, 120, 240)}
    for f, s in ((5, 20), (5, 60), (10, 20), (10, 60), (20, 60), (20, 120), (20, 240), (60, 240)):
        cond[f"MA{f}>MA{s}"] = ma[f] > ma[s]
    for n in (5, 10, 20, 60, 120, 240):
        cond[f"C>MA{n}"] = C > ma[n]
    for n in (20, 60, 240):
        cond[f"C<MA{n}"] = C < ma[n]
    for n in (5, 20, 60, 120, 250):
        r = C.pct_change(n)
        cond[f"R{n}>0"] = r > 0
        cond[f"R{n}<0"] = r < 0
    hi = C.rolling(250).max()
    cond["近52週高90%"] = C >= 0.9 * hi
    cond["近52週高95%"] = C >= 0.95 * hi

    def rsi(k):
        dx = C.diff()
        up = dx.clip(lower=0).ewm(alpha=1 / k, adjust=False).mean()
        dn = (-dx.clip(upper=0)).ewm(alpha=1 / k, adjust=False).mean()
        return 100 - 100 / (1 + up / dn)
    r14, r2 = rsi(14), rsi(2)
    cond["RSI14>50"], cond["RSI14<30"], cond["RSI2<10"], cond["RSI2>90"] = r14 > 50, r14 < 30, r2 < 10, r2 > 90
    m20, s20 = C.rolling(20).mean(), C.rolling(20).std()
    cond["布林下軌下"], cond["布林上軌上"] = C < m20 - 2 * s20, C > m20 + 2 * s20
    macd = C.ewm(span=12, adjust=False).mean() - C.ewm(span=26, adjust=False).mean()
    cond["MACD柱>0"], cond["MACD>0"] = macd - macd.ewm(span=9, adjust=False).mean() > 0, macd > 0
    rsv = (C - L.rolling(9).min()) / (Hh.rolling(9).max() - L.rolling(9).min()) * 100
    K = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    D = K.ewm(alpha=1 / 3, adjust=False).mean()
    cond["K>D"], cond["K<20"] = K > D, K < 20
    vr = V.rolling(5).mean() / V.rolling(60).mean()
    cond["量增收紅"], cond["量縮"] = (vr > 1.5) & (C > C.shift(1)), vr < 0.7
    # 營收：m 月營收 m+1 月 11 日可得
    rv = pd.read_parquet(H.D / "revenue" / f"{sid}.parquet")
    rv["ym"] = pd.to_datetime(dict(year=rv.revenue_year, month=rv.revenue_month, day=1))
    rv = rv.drop_duplicates("ym").set_index("ym").sort_index().revenue.astype(float)
    rv = rv.reindex(pd.date_range(rv.index.min(), rv.index.max(), freq="MS"))
    yoy = rv / rv.shift(12) - 1
    s3 = rv.rolling(3).sum()
    newhi = s3 >= s3.rolling(12).max()
    mom = rv / rv.shift(1) - 1
    avail = rv.index + pd.offsets.MonthBegin(1) + pd.Timedelta(days=10)
    mf = pd.DataFrame(dict(yoy=yoy.values, newhi=newhi.values, mom=mom.values), index=avail)
    mf = mf[mf.index <= END].reindex(idx.union(mf.index)).ffill().reindex(idx)
    cond["營收YoY>0"], cond["營收YoY>20%"] = mf.yoy > 0, mf.yoy > 0.2
    cond["營收3月新高"], cond["營收MoM>0"] = mf.newhi.astype(float) > 0.5, mf.mom > 0
    # 法人
    ins = rd("inst", sid)
    ins["net"] = ins.buy - ins.sell
    pv = ins.pivot_table(index="date", columns="name", values="net", aggfunc="sum").reindex(idx).fillna(0)
    g = lambda cols: pv[[c for c in cols if c in pv]].sum(axis=1)  # noqa: E731
    fo, it, de = g(["Foreign_Investor", "Foreign_Dealer_Self"]), g(["Investment_Trust"]), g(["Dealer", "Dealer_self", "Dealer_Hedging"])
    for n in (5, 20, 60):
        cond[f"外資{n}日買超"] = fo.rolling(n).sum() > 0
        cond[f"投信{n}日買超"] = it.rolling(n).sum() > 0
    cond["自營20日買超"] = de.rolling(20).sum() > 0
    mg = rd("margin", sid).set_index("date").MarginPurchaseTodayBalance.reindex(idx).ffill()
    cond["融資20日增"], cond["融資20日減"] = mg.diff(20) > 0, mg.diff(20) < 0
    pb = rd("per", sid).set_index("date").PBR.reindex(idx).ffill()
    pr = pb.rolling(1250, min_periods=500).apply(lambda x: (x[:-1] < x[-1]).mean(), raw=True)
    cond["PBR低20%"], cond["PBR高80%"] = pr < 0.2, pr > 0.8
    tx = pd.read_parquet(H.D / "futures" / "taiex.parquet")
    tx = tx.assign(date=pd.to_datetime(tx.date)).set_index("date").close
    tx = tx[(tx > 0) & (tx.index <= END)]
    cond["大盤>MA200"] = (tx > tx.rolling(200).mean()).reindex(idx).ffill().fillna(False).astype(bool)
    cond["大盤R20>0"] = (tx.pct_change(20) > 0).reindex(idx).ffill().fillna(False).astype(bool)
    names = list(cond)
    B = np.array([np.asarray(cond[k].fillna(False), dtype=bool) for k in names])
    O = adj.open.values
    r = np.r_[O[1:] / O[:-1] - 1, np.nan]                  # 第 j 天開盤 → 第 j+1 天開盤
    ro, rh, rl, rc = (raw[c].values for c in ("open", "high", "low", "close"))
    return dict(idx=idx, names=names, B=B, r=r, raw=(ro, rh, rl, rc))


# ───────── 快速評估（忽略漲跌停） ─────────
def wmask(idx, w):
    return np.asarray((idx >= w[0]) & (idx <= w[1]))


def evaluate(P, r, m):
    """P：規則 × 天的部位（收盤 t 決定）。回傳 Sharpe、交易次數、曝險（只算 m 期間）。"""
    P = P.astype(np.float32)
    p1 = np.concatenate([np.zeros((P.shape[0], 1), np.float32), P[:, :-1]], axis=1)   # 第 j 天開盤後持有的部位
    p2 = np.concatenate([np.zeros((P.shape[0], 1), np.float32), p1[:, :-1]], axis=1)
    rr = np.nan_to_num(r).astype(np.float32)
    ret = p1 * rr - BUY * (p1 > p2) - SELL * (p1 < p2)
    ret = ret[:, m]
    mu, sd = ret.mean(1), ret.std(1)
    sh = np.where(sd > 0, mu / sd * np.sqrt(252), 0)
    tr = ((p1 > p2)[:, m]).sum(1)
    return sh, tr, p1[:, m].mean(1)


def hold(E, h):
    """條件由假變真就買、持有 h 日（再觸發就延長）。"""
    edge = E & ~np.concatenate([np.zeros((E.shape[0], 1), bool), E[:, :-1]], axis=1)
    c = np.cumsum(edge, axis=1)
    lag = np.concatenate([np.zeros((E.shape[0], h), c.dtype), c[:, :-h]], axis=1)
    return (c - lag) > 0


def rules_stage1(B, names):
    n = len(names)
    yield [((names[i],), "持有") for i in range(n)], B
    pairs = list(itertools.combinations(range(n), 2))
    for s in range(0, len(pairs), CHUNK):
        ps = pairs[s:s + CHUNK]
        yield [((names[i], names[j]), "持有") for i, j in ps], B[[i for i, _ in ps]] & B[[j for _, j in ps]]
    tri = list(itertools.combinations(range(n), 3))
    for s in range(0, len(tri), CHUNK):
        ts = tri[s:s + CHUNK]
        a, b, c = zip(*ts)
        yield [((names[i], names[j], names[k]), "持有") for i, j, k in ts], B[list(a)] & B[list(b)] & B[list(c)]
    for h in HOLDS:
        yield [((names[i],), f"買後{h}日") for i in range(n)], hold(B, h)
        for s in range(0, len(pairs), CHUNK):
            ps = pairs[s:s + CHUNK]
            yield [((names[i], names[j]), f"買後{h}日") for i, j in ps], hold(B[[i for i, _ in ps]] & B[[j for _, j in ps]], h)


def rules_stage2(B, names):
    quad = itertools.combinations(range(len(names)), 4)
    while True:
        qs = list(itertools.islice(quad, CHUNK))
        if not qs:
            return
        a, b, c, d = zip(*qs)
        yield [(tuple(names[x] for x in q), "持有") for q in qs], B[list(a)] & B[list(b)] & B[list(c)] & B[list(d)]


# ───────── 完整模擬（一字漲停買不到、一字跌停賣不掉） ─────────
def full_sim(pos, X, w):
    idx, r = X["idx"], X["r"]
    ro, rh, rl, rc = X["raw"]
    lim = np.where(idx < pd.Timestamp("2015-06-01"), 0.07, 0.10)
    prev = np.r_[np.nan, rc[:-1]]
    up_lock = (ro == rh) & (ro == rl) & (ro >= prev * (1 + lim - 0.006))
    dn_lock = (ro == rh) & (ro == rl) & (ro <= prev * (1 - lim + 0.006))
    T = len(idx)
    held, ret, H_, ent = 0, np.zeros(T), np.zeros(T), np.zeros(T, bool)
    for j in range(1, T):
        want = pos[j - 1]
        c = 0.0
        if want and not held and not up_lock[j]:
            held, c, ent[j] = 1, BUY, True
        elif not want and held and not dn_lock[j]:
            held, c = 0, SELL
        H_[j] = held
        ret[j] = held * (0 if np.isnan(r[j]) else r[j]) - c
    m = wmask(idx, w)
    x = ret[m]
    return dict(sharpe=float(x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else 0.0,
                cagr=float(np.prod(1 + x) ** (252 / len(x)) - 1), mdd=float((np.cumprod(1 + x) / np.maximum.accumulate(np.cumprod(1 + x)) - 1).min()),
                trades=int(ent[m].sum()), expo=float(H_[m].mean()))


def build_pos(X, rule):
    conds, mode = rule
    E = np.ones(len(X["idx"]), bool)
    for c in conds:
        E &= X["B"][X["names"].index(c)]
    return hold(E[None], int(mode[2:-1]))[0] if mode.startswith("買後") else E


def main(stage, sid):
    t0 = time.time()
    X = load(sid)
    idx, r = X["idx"], X["r"]
    md, mv = wmask(idx, DEV), wmask(idx, VAL)
    T = len(idx)
    rng = np.random.default_rng(0)
    nd = md.sum()
    offs = rng.integers(250, nd - 250, NPLAC)
    di = np.nonzero(md)[0]
    rP = []
    for k in offs:                                         # 只在 dev 期間內循環錯開報酬
        rr = r.copy()
        rr[di] = np.roll(r[di], k)
        rP.append(rr)
    print(f"{sid}：{T} 天（截到 {END}），條件 {len(X['names'])} 個；dev {nd} 天（{time.time()-t0:.0f}s）", flush=True)
    gen = rules_stage1(X["B"], X["names"]) if stage == "stage1" else rules_stage2(X["B"], X["names"])
    allr, best_pl, best_pl_q = [], np.full(NPLAC, -9.0), np.full(NPLAC, -9.0)
    n_rules = 0
    for desc, P in gen:
        sh, tr, ex = evaluate(P, r, md)
        q = tr >= MIN_TR
        n_rules += len(desc)
        keep = np.nonzero(sh >= 1.0)[0]
        for i in keep:
            allr.append((float(sh[i]), int(tr[i]), float(ex[i]), desc[i]))
        for p, rr in enumerate(rP):
            s2, _, _ = evaluate(P, rr, md)
            best_pl[p] = max(best_pl[p], s2.max())
            if q.any():
                best_pl_q[p] = max(best_pl_q[p], s2[q].max())
        if n_rules % 40000 < CHUNK:
            print(f"  已評 {n_rules:,} 個（{time.time()-t0:.0f}s）", flush=True)
    allr.sort(key=lambda x: -x[0])
    qual = [x for x in allr if x[1] >= MIN_TR]
    thr, thr_q = np.quantile(best_pl, 0.95), np.quantile(best_pl_q, 0.95)
    print(f"\n共評 {n_rules:,} 個規則（{time.time()-t0:.0f}s）")
    print(f"安慰劑（{NPLAC} 次）最高 dev Sharpe：不限次數 中位 {np.median(best_pl):.2f}、95 分位 {thr:.2f}"
          f"｜交易 ≥ {MIN_TR} 次 中位 {np.median(best_pl_q):.2f}、95 分位 {thr_q:.2f}")
    n3 = sum(1 for x in qual if x[0] >= 3)
    print(f"實際：最高 dev Sharpe 不限次數 {allr[0][0]:.2f}（{allr[0][1]} 次）｜交易 ≥ {MIN_TR} 次 {qual[0][0]:.2f}"
          f"｜dev ≥ 3 且有資格 {n3} 個｜dev ≥ 2 {sum(1 for x in qual if x[0] >= 2)} 個")
    bh = np.ones(T, bool)
    b_dev, b_val = full_sim(bh, X, DEV), full_sim(bh, X, VAL)
    print(f"買進持有：dev Sharpe {b_dev['sharpe']:.2f}、val {b_val['sharpe']:.2f}")
    print("\n== dev 前 5 名（有資格），完整模擬（漲跌停）→ val")
    top = []
    for sh, tr, ex, rule in qual[:5]:
        pos = build_pos(X, rule)
        a, b = full_sim(pos, X, DEV), full_sim(pos, X, VAL)
        top.append(dict(rule=" & ".join(rule[0]) + f"｜{rule[1]}", dev=a, val=b))
        print(f"  {' & '.join(rule[0])}｜{rule[1]}\n     dev Sharpe {a['sharpe']:.2f}（{tr} 次、曝險 {ex:.0%}、CAGR {a['cagr']*100:+.1f}%、MDD {a['mdd']*100:.1f}%）"
              f" → val Sharpe {b['sharpe']:.2f}（CAGR {b['cagr']*100:+.1f}%、MDD {b['mdd']*100:.1f}%）")
    vs = []
    for sh, tr, ex, rule in [x for x in qual if x[0] >= 3][:2000]:
        vs.append(evaluate(build_pos(X, rule)[None], r, mv)[0][0])
    if vs:
        vs = np.array(vs)
        print(f"\n== dev ≥ 3 的 {len(vs)} 個規則在 val：中位 {np.median(vs):.2f}、75 分位 {np.quantile(vs, .75):.2f}、"
              f"最高 {vs.max():.2f}、≥ 3 的 {int((vs >= 3).sum())} 個、< 買進持有 {b_val['sharpe']:.2f} 的 {(vs < b_val['sharpe']).mean():.0%}")
    dv = np.array([x[0] for x in qual[:2000]])
    vv = np.array([evaluate(build_pos(X, x[3])[None], r, mv)[0][0] for x in qual[:2000]])
    print(f"== dev 前 2,000 名：dev 中位 {np.median(dv):.2f} → val 中位 {np.median(vv):.2f}；dev 和 val Sharpe 相關 {np.corrcoef(dv, vv)[0,1]:+.2f}")
    win = any(t["val"]["sharpe"] >= 3 and t["dev"]["sharpe"] > thr_q for t in top)
    print(f"\n判定：{'✓' if win else '✗'}（dev 前 5 名有沒有 val Sharpe ≥ 3 且 dev 超過安慰劑門檻 {thr_q:.2f}）")
    with open(H.LOG, "a") as f:
        f.write(json.dumps(dict(ts=time.strftime("%Y-%m-%d %H:%M:%S"), kind="single_stock", period="dev+val", stock=sid,
                                strategy=f"第七十九批 H192 單一個股規則搜尋 {stage}", note="第七十九批 事前登記（探索）", n_rules=n_rules,
                                placebo=dict(all=best_pl.tolist(), qual=best_pl_q.tolist(), thr=thr, thr_q=thr_q),
                                best_all=allr[0][0], best_qual=qual[0][0], n_ge3=n3, top=top, bh=dict(dev=b_dev, val=b_val),
                                val_of_ge3=dict(n=len(vs), med=float(np.median(vs)) if len(vs) else None,
                                                n_ge3=int((np.asarray(vs) >= 3).sum()) if len(vs) else 0),
                                rho_dev_val=float(np.corrcoef(dv, vv)[0, 1]), verdict="✓" if win else "✗"),
                           ensure_ascii=False, default=lambda v: round(float(v), 4) if isinstance(v, (float, np.floating)) else str(v)) + "\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "stage1", sys.argv[2] if len(sys.argv) > 2 else "8213")
