"""第七十二批 H184：自身淨值比分位帶（每檔自己的便宜線）；H185 逐股學習便宜線（先過關卡）。規則見 HYPOTHESES.md 第七十二批（事前登記）。
便宜線＝過去 5 年（1,250 交易日、至少 750 個有值）淨值比的第 q 分位；出場＝回到自身中位數（B1）或 80 分位（B2），最長 250 日。
模擬器沿用第六十八批 pool_event_lab.simulate／evaluate；dev 改成 2012–19（要 3 年歷史）。

用法：.venv/bin/python research/v2/pbband_lab.py cache dev|val
      .venv/bin/python research/v2/pbband_lab.py gate      # H185 關卡（只用 dev）
      .venv/bin/python research/v2/pbband_lab.py dev
      .venv/bin/python research/v2/pbband_lab.py val
      .venv/bin/python research/v2/pbband_lab.py h185 dev|val   # 關卡過了才跑
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
NAME = "第七十二批 H184 自身淨值比分位帶"
TRIALS = 1701703
WIN, MINP, RECENT, MAXH = 1250, 750, 60, 250
QS = (5, 10, 20, 30, 50, 80)
BANDS, TIMING, EXITS, POOLS = (10, 20), ("T0", "T1", "T2"), ("B1", "B2"), ("PB1", "L")
GATE_Q = (5, 10, 20, 30)
SEG = {"dev": ("2012-01-01", "2019-12-31"), "val": ("2020-01-01", "2023-12-31")}
YRS_MIN = {"dev": 6 / 8, "val": 0.0}
HALF = pd.Timestamp("2016-01-01")
NSEL = 10
LABEL = {"T0": "直接", "T1": "站上20日線", "T2": "站上60日線", "B1": "回到中位數", "B2": "到80分位"}
for k, v in SEG.items():                         # 模擬器用 PE.SEG 切期間
    PE.SEG[f"pb_{k}"] = v
assert not set(EXITS) & set(PE.HOLD)               # 出場代號不能和模擬器的固定持有期撞名


# ═════════ 資料 ═════════
def raw_pbr(cols, end):
    out = {}
    for s in cols:
        f = H.D / "per" / f"{s}.parquet"
        if not f.exists():
            continue
        d = pd.read_parquet(f)
        if "date" not in d or not len(d):
            continue
        d = d.assign(date=pd.to_datetime(d.date)).drop_duplicates("date", keep="last").set_index("date").PBR.astype(float)
        out[s] = d[d.index <= end]
    X = pd.DataFrame(out).sort_index()
    return X.where(X > 0)


def cache(period):
    end = pd.Timestamp(H.PERIODS[period][1])
    t0 = time.time()
    P = xs.load_panel(str(end.date()), fundamentals=True)
    cols, days = P["close"].columns, P["close"].index
    assert list(cols) == sorted(cols)
    fx, trade, _ = V.gap_fix(P, cols)
    LU, LD, _, _ = V.limit_flags(fx)
    X = raw_pbr(cols, end)
    print(f"載入 {time.time()-t0:.0f}s；{len(days)} 個交易日；淨值比 {X.shape[1]} 檔、{X.index[0].date()} 起", flush=True)
    roll = X.rolling(WIN, min_periods=MINP)
    band = {q: roll.quantile(q / 100).reindex(index=days, columns=cols) for q in QS}
    pbr = X.reindex(index=days, columns=cols)
    print(f"分位數 {time.time()-t0:.0f}s", flush=True)
    inliq = P["liq"].fillna(False).astype(bool) & (P["nbars"] >= 250)
    Lp = inliq & trade
    PM = {"PB1": Lp & (pbr > 0) & (pbr < 1), "L": Lp & (pbr > 0)}
    U = np.flatnonzero(PM["L"].values.any(axis=0))
    uc = cols[U]
    O, Hh, Cc = fx["open"][uc], fx["high"][uc], fx["close"][uc]
    tr_ok = trade[uc]
    Cf = Cc.ffill()
    ma20, ma60 = Cf.rolling(20, min_periods=20).mean(), Cf.rolling(60, min_periods=60).mean()
    pb = pbr[uc]
    b = {q: band[q][uc] for q in QS}
    mid = b[50]
    ent, prio = {}, {}
    for q in BANDS:
        below = (pb < b[q]).fillna(False)
        recent = below.astype(np.int8).rolling(RECENT, min_periods=1).max().astype(bool)
        under = (pb < mid).fillna(False)
        ent[f"T0q{q}"] = below & tr_ok
        ent[f"T1q{q}"] = recent & (Cf > ma20) & under & tr_ok
        ent[f"T2q{q}"] = recent & (Cf > ma60) & under & tr_ok
        prio[f"q{q}"] = (pb / b[q]).values.astype(np.float32)
    ent = {k: v.fillna(False).values for k, v in ent.items()}
    ex = {"B1": (pb > mid).fillna(False).values, "B2": (pb > b[80]).fillna(False).values}
    tr = pd.read_parquet(H.D / "taiex_tr.parquet")
    tr["date"] = pd.to_datetime(tr.date)
    mkt = tr.drop_duplicates("date").set_index("date").price.astype(float).sort_index().reindex(days).ffill()
    trv = tr_ok.values
    lastv = np.array([np.flatnonzero(trv[:, j]).max() if trv[:, j].any() else -1 for j in range(len(uc))])
    D = dict(days=days, cols=uc, period=period, O=O.values.astype(np.float64), H=Hh.fillna(Cf).values, C=Cf.values, trade=trv,
             LU=LU[uc].fillna(False).values, LD=LD[uc].fillna(False).values, atr=None, lastv=lastv, mkt=mkt.values, ent=ent, ex=ex, prio=prio,
             pool={p: M[uc].values for p, M in PM.items()}, pool_n={p: M.values.sum(axis=1) for p, M in PM.items()},
             pbr=pb.values.astype(np.float32), band={q: b[q].values.astype(np.float32) for q in QS})
    with open(RES / f"pbband_{period}.pkl", "wb") as fh:
        pickle.dump(D, fh)
    k = days.slice_indexer(*SEG[period])
    hit = {q: float((PM["PB1"][uc].values & (pb < b[q]).fillna(False).values)[k].sum(axis=1).mean()) for q in BANDS}
    print(f"快取完成 {time.time()-t0:.0f}s；股票 {len(U)} 檔；PB1 每天平均 {PM['PB1'].values.sum(1)[k].mean():.0f} 檔、L {PM['L'].values.sum(1)[k].mean():.0f} 檔；"
          f"PB1 裡低於便宜線的每天平均：" + "、".join(f"q{q} {v:.0f} 檔" for q, v in hit.items()), flush=True)


def load(period):
    with open(RES / f"pbband_{period}.pkl", "rb") as fh:
        return pickle.load(fh)


# ═════════ H185 關卡 ═════════
def episodes(D, q, pool="PB1", span=SEG["dev"], known=False):
    """不限格數：剛跌破便宜線 q 的隔日開盤買、淨值比 > 自身中位數的隔日開盤賣（最長 250 日），同檔同線不重疊。
    回傳 [(股, 進場日, 報酬)]；known=True 時多回傳賣出日的位置（收盤後才算得知）。"""
    days, O, C, trade, lastv = D["days"], D["O"], D["C"], D["trade"], D["lastv"]
    t0, t1 = (int(np.searchsorted(days.values, np.datetime64(x), side=s)) for x, s in zip(span, ("left", "right")))
    below = np.nan_to_num(D["pbr"] < D["band"][q])
    cross = below & ~np.vstack([np.zeros((1, below.shape[1]), bool), below[:-1]]) & D["pool"][pool]
    exit_ = D["pbr"] > D["band"][50]
    out = []
    for j in range(len(D["cols"])):
        ix = np.flatnonzero(cross[t0:t1 - 1, j]) + t0
        if not len(ix):
            continue
        xs_ = np.flatnonzero(exit_[:, j])
        busy = -1
        for e in ix:
            if e <= busy or e + 1 > lastv[j] or not trade[e + 1, j] or not np.isfinite(O[e + 1, j]):
                continue
            k = np.searchsorted(xs_, e, side="right")
            x = min(xs_[k] if k < len(xs_) else 10 ** 9, e + MAXH, lastv[j])
            sell_open = x + 1 <= lastv[j] and np.isfinite(O[x + 1, j])
            px = O[x + 1, j] if sell_open else C[min(x, lastv[j]), j]
            r = (1 - PE.BUY) * px / O[e + 1, j] * (1 - PE.SELL) - 1
            out.append((j, days[e], r, x + 1 if sell_open else min(x, lastv[j])) if known else (j, days[e], r))
            busy = x + 1
    return out


def gate():
    D = load("dev")
    rows = []
    for q in GATE_Q:
        for j, d, r in episodes(D, q):
            rows.append(dict(j=j, q=q, half="A" if d < HALF else "B", r=r))
    T = pd.DataFrame(rows)
    print("每條線的筆數與平均（扣成本）：\n" + T.groupby(["q", "half"]).r.agg(["count", "mean", "median"]).round(4).to_string(), flush=True)
    g = T.groupby(["j", "q", "half"]).r.mean().unstack(["half"])
    g = g.dropna().unstack("q")
    g = g[g.notna().all(axis=1)]
    A, B = g["A"], g["B"]
    best = A.idxmax(axis=1)
    adv = np.array([B.loc[i, best[i]] for i in g.index]) - B.mean(axis=1).values
    same = (best.values == B.idxmax(axis=1).values).mean()
    t = adv.mean() / adv.std(ddof=1) * np.sqrt(len(adv))
    ok = bool(adv.mean() > 0 and t >= 2)
    print(f"\n四條線前後兩半都有交易的股票：{len(g)} 檔", flush=True)
    print(f"前半最佳線在後半的優勢（減該股四條線平均）：平均 {adv.mean()*100:+.2f}%／筆（t {t:+.2f}）；前後最佳線相同的比例 {same:.0%}（隨機 25%）", flush=True)
    print("前半最佳線的分布：", best.value_counts().sort_index().to_dict(), flush=True)
    print(f"關卡：{'過 → H185 要跑' if ok else '沒過 → H185 不跑'}", flush=True)
    # 事後核對（看過關卡結果才加，不改判定）：把「前半最佳線」在股票之間打亂 2,000 次，保留各線被選中的次數
    rng = np.random.default_rng(2026)
    Bv, bi = B.values, np.searchsorted(np.array(B.columns), best.values)
    plc = np.array([(Bv[np.arange(len(bi)), rng.permutation(bi)] - Bv.mean(axis=1)).mean() for _ in range(2000)])
    print(f"事後核對：打亂後的平均優勢 {plc.mean()*100:+.2f}%（95 分位 {np.quantile(plc, 0.95)*100:+.2f}%）；真實 {adv.mean()*100:+.2f}% "
          f"→ 扣掉共同部分 {(adv.mean() - plc.mean())*100:+.2f}%（單尾 p {(plc >= adv.mean()).mean():.3f}）", flush=True)
    SL.log(dict(kind="describe", period="dev", strategy="第七十二批 H185 逐股學習便宜線", note="第七十二批 H185 關卡（不算試驗）＋事後打亂核對",
                n_stocks=len(g), adv_mean=float(adv.mean()), adv_t=float(t), same_best=float(same), passed=ok,
                best_dist={str(k): int(v) for k, v in best.value_counts().sort_index().items()},
                by_q={f"q{q}_{h}": float(v) for (q, h), v in T.groupby(["q", "half"]).r.mean().items()},
                placebo_mean=float(plc.mean()), placebo_q95=float(np.quantile(plc, 0.95)), placebo_p=float((plc >= adv.mean()).mean())))
    return ok


# ═════════ H184 ═════════
def models():
    return [dict(pool=p, fam="B", entry=f"{t}q{q}", exit=x, prio=f"q{q}", max_hold=MAXH, label=f"{p}池 跌破{q}分位→{LABEL[t]}→{LABEL[x]}")
            for p, q, t, x in itertools.product(POOLS, BANDS, TIMING, EXITS)]


def passes(r, seg):
    return np.isfinite(r["alpha_t"]) and r["alpha_t"] >= 2 and r["sharpe"] >= 1.0 and r["yrs_win"] >= YRS_MIN[seg] - 1e-9


def gross(r):
    return (1 + r["tr_mean"]) / ((1 - PE.BUY) * (1 - PE.SELL)) - 1


def corr_h136(D, s, seg):
    led = pd.read_parquet(RES / "sleeves" / seg / "H136.parquet")
    days = D["days"][D["days"].slice_indexer(*SEG[seg])]
    nav = (1 + pd.Series(s, index=days)).cumprod()
    led = led[led.index >= days[0]]
    px = [nav[nav.index < d].iloc[-1] if (nav.index < d).any() else np.nan for d in led.index]
    mr = pd.Series(px, index=led.index).pct_change().shift(-1)
    k = mr.notna() & led.ret.notna()
    return float(np.corrcoef(mr[k], led.ret[k])[0, 1]) if k.sum() > 12 else np.nan


def show(r, extra=""):
    PE.show(r, f"｜每筆毛 {gross(r)*100:+.2f}%{extra}")


def table(rows):
    T = pd.DataFrame(rows)
    T["band"] = T.entry.str[-2:]
    T["timing"] = T.entry.str[:2]
    g = {k: T.groupby(["pool", k]).alpha_t.mean().unstack().round(2) for k in ("band", "timing", "exit")}
    for k, x in g.items():
        print(f"依 {k} 平均 alpha t：\n" + x.to_string(), flush=True)
    m = T.groupby("pool")[["alpha_t", "sharpe"]].mean()
    print("各池平均：\n" + m.round(2).to_string(), flush=True)
    return T, m


def run(D, seg):
    M = models()
    return M, PE.run_all(D, M, f"pb_{seg}")


def dev():
    t0 = time.time()
    D = load("dev")
    M, rows = run(D, "dev")
    print(f"dev {SEG['dev']}；模型 {len(M)} 個（{time.time()-t0:.0f}s）", flush=True)
    T, m = table(rows)
    fit = bool(m.loc["PB1", "alpha_t"] > m.loc["L", "alpha_t"])
    print(f"池的預測（PB1 > L）dev：{'成立' if fit else '不成立'}", flush=True)
    print("\n全部（依 alpha t）：", flush=True)
    for r in sorted(rows, key=lambda r: -r["alpha_t"]):
        show(r, " ✓" if passes(r, "dev") else "")
    ok = [r for r in sorted(rows, key=lambda r: -r["alpha_t"]) if passes(r, "dev")]
    print(f"\n過 dev 三條：{len(ok)} 個（" + "、".join(f"{p} {sum(r['pool'] == p for r in ok)}" for p in POOLS) + "）；L 池不進 val", flush=True)
    sel, ser = [], []
    for r in ok:
        if r["pool"] != "PB1":
            continue
        mm = {k: r[k] for k in ("pool", "fam", "entry", "exit", "prio", "max_hold", "label")}
        _, s = PE.evaluate(D, mm, "pb_dev")
        if any(np.corrcoef(s, z)[0, 1] > 0.8 for z in ser):
            continue
        r["rho_h136"] = corr_h136(D, s, "dev")
        sel.append(r)
        ser.append(s)
        show(r, f"｜和 H136 月相關 {r['rho_h136']:+.2f}｜逐年超額 {r['by_year']}")
        if len(sel) == NSEL:
            break
    keep = ("name", "pool", "entry", "exit", "alpha_t", "sharpe", "yrs_win", "cagr", "mdd", "n_trades", "tr_mean", "tr_med", "win", "hold", "exposure")
    SL.log(dict(kind="strategy", period="dev", strategy=NAME, note="第七十二批 H184 事前登記", pool_mean=m.round(3).to_dict(), fit_dev=fit, n_pass=len(ok),
                selected=[{k: r[k] for k in ("pool", "fam", "entry", "exit", "prio", "max_hold", "label")} for r in sel],
                extra=[{k: r[k] for k in keep + ("rho_h136", "by_year")} for r in sel], all=[{k: r[k] for k in keep} for r in rows], trials_cum=TRIALS))


def val():
    rec = [json.loads(l) for l in open(H.LOG) if f'"{NAME}"' in l and '"period": "dev"' in l][-1]
    if not rec["selected"] and not rec["fit_dev"]:
        print("dev 沒有候選、池的預測也不成立，依登記不跑 val")
        return
    D = load("val")
    M, rows = run(D, "val")
    _, m = table(rows)
    fit = bool(m.loc["PB1", "alpha_t"] > m.loc["L", "alpha_t"])
    print(f"池的預測 val：{'成立' if fit else '不成立'} → 整體{'成立' if fit and rec['fit_dev'] else '不成立'}", flush=True)
    out = []
    for mm in rec["selected"]:
        r, s = PE.evaluate(D, mm, "pb_val")
        r["rho_h136"] = corr_h136(D, s, "val")
        r["verdict"] = "✓" if r["alpha_t"] >= 2 and r["sharpe"] >= 1.0 else "✗"
        out.append(r)
        show(r, f"｜和 H136 月相關 {r['rho_h136']:+.2f}｜逐年超額 {r['by_year']} → {r['verdict']}")
    if out:
        print(f"val 通過 {sum(r['verdict'] == '✓' for r in out)}／{len(out)}")
    keep = ("name", "pool", "entry", "exit", "alpha_t", "sharpe", "yrs_win", "cagr", "mdd", "n_trades", "tr_mean", "tr_med", "win", "hold", "exposure")
    SL.log(dict(kind="strategy", period="val", strategy=NAME, note="第七十二批 H184 dev 選出的＋池的預測", pool_mean=m.round(3).to_dict(), fit_val=fit,
                result=[{k: r[k] for k in keep + ("rho_h136", "by_year", "verdict")} for r in out]))


# ═════════ H185 逐股學習便宜線（關卡過了才跑）═════════
NAME185 = "第七十二批 H185 逐股學習便宜線"


def learn(D):
    """每檔每天：到今天為止已經出場的歷史交易（同關卡的定義）平均報酬最高的線（該線至少 2 筆，否則 10 分位）。加進 D 的訊號。"""
    T, J = len(D["days"]), len(D["cols"])
    span = (str(D["days"][0].date()), str(D["days"][-1].date()))
    mean = []
    for q in GATE_Q:
        S, N = np.zeros((T, J)), np.zeros((T, J))
        for j, _, r, k in episodes(D, q, span=span, known=True):
            S[k, j] += r
            N[k, j] += 1
        S, N = S.cumsum(axis=0), N.cumsum(axis=0)
        mean.append(np.where(N >= 2, S / np.maximum(N, 1), -np.inf))
    M = np.stack(mean)
    pick = np.where(np.isfinite(M).any(axis=0), M.argmax(axis=0), GATE_Q.index(10))
    band = np.choose(pick, [D["band"][q] for q in GATE_Q])
    pb = D["pbr"]
    below = np.nan_to_num(pb < band).astype(bool)
    recent = pd.DataFrame(below).rolling(RECENT, min_periods=1).max().values.astype(bool)
    under = np.nan_to_num(pb < D["band"][50]).astype(bool)
    Cf = pd.DataFrame(D["C"])
    ma20, ma60 = Cf.rolling(20, min_periods=20).mean().values, Cf.rolling(60, min_periods=60).mean().values
    with np.errstate(invalid="ignore"):
        D["ent"]["T0qL"] = below & D["trade"]
        D["ent"]["T1qL"] = recent & (D["C"] > ma20) & under & D["trade"]
        D["ent"]["T2qL"] = recent & (D["C"] > ma60) & under & D["trade"]
        D["prio"]["qL"] = (pb / band).astype(np.float32)
    return pick


def models185():
    return [dict(pool="PB1", fam="L", entry=f"{t}qL", exit=x, prio="qL", max_hold=MAXH, label=f"PB1池 逐股學習便宜線→{LABEL[t]}→{LABEL[x]}")
            for t, x in itertools.product(TIMING, EXITS)]


def h185(seg):
    D = load(seg)
    pick = learn(D)
    k = D["days"].slice_indexer(*SEG[seg])
    m = D["pool"]["PB1"][k]
    dist = {f"q{q}": float((pick[k][m] == i).mean()) for i, q in enumerate(GATE_Q)}
    print(f"{seg}：PB1 池股票日的學習結果分布：" + "、".join(f"{a} {v:.0%}" for a, v in dist.items()) + "（沒有足夠歷史的用 q10）", flush=True)
    rows = PE.run_all(D, models185(), f"pb_{seg}")
    keep = ("name", "pool", "entry", "exit", "alpha_t", "sharpe", "yrs_win", "cagr", "mdd", "n_trades", "tr_mean", "tr_med", "win", "hold", "exposure")
    if seg == "dev":
        for r in sorted(rows, key=lambda r: -r["alpha_t"]):
            show(r, " ✓" if passes(r, "dev") else "")
        ok = [r for r in rows if passes(r, "dev")]
        print(f"過 dev 三條：{len(ok)}／{len(rows)}", flush=True)
        SL.log(dict(kind="strategy", period="dev", strategy=NAME185, note="第七十二批 H185（關卡依登記通過）", pick=dist, n_pass=len(ok),
                    selected=[{k_: r[k_] for k_ in ("pool", "fam", "entry", "exit", "prio", "max_hold", "label")} for r in ok],
                    all=[{k_: r[k_] for k_ in keep} for r in rows], trials_cum=TRIALS + 6))
    else:
        rec = [json.loads(l) for l in open(H.LOG) if f'"{NAME185}"' in l and '"period": "dev"' in l and '"kind": "strategy"' in l][-1]
        sel = {r["entry"] + r["exit"] for r in rec["selected"]}
        out = [r for r in rows if r["entry"] + r["exit"] in sel]
        for r in out:
            r["verdict"] = "✓" if r["alpha_t"] >= 2 and r["sharpe"] >= 1.0 else "✗"
            show(r, f"｜逐年超額 {r['by_year']} → {r['verdict']}")
        SL.log(dict(kind="strategy", period="val", strategy=NAME185, note="第七十二批 H185 dev 選出的", result=[{k_: r[k_] for k_ in keep + ("verdict",)} for r in out]))


if __name__ == "__main__":
    warnings.simplefilter("ignore", RuntimeWarning)
    cmd = sys.argv[1]
    if cmd == "cache":
        cache(sys.argv[2])
    else:
        {"gate": gate, "dev": dev, "val": val, "h185": lambda: h185(sys.argv[2])}[cmd]()
