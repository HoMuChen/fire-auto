"""第八十批 H193 探索：高波動、有個股期貨的單一股票（預設玉晶光 3406），多空都可以。規則見 HYPOTHESES.md 第八十批（事前登記）。
條件與資料沿用 stock1_lab（截到 2023-12-31）；個股期貨用還原股價近似，每邊成本 0.1%。

用法：.venv/bin/python research/v2/stock2_lab.py [股票代號]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402
import stock1_lab as S1  # noqa: E402

COST = 0.001
TOPK = 100
MIN_TR = S1.MIN_TR
NPLAC = S1.NPLAC


def evaluate(P, r, m):
    """P：規則 × 天的部位（−1／0／+1，收盤 t 決定）。"""
    P = P.astype(np.float32)
    p1 = np.concatenate([np.zeros((P.shape[0], 1), np.float32), P[:, :-1]], axis=1)
    p2 = np.concatenate([np.zeros((P.shape[0], 1), np.float32), p1[:, :-1]], axis=1)
    rr = np.nan_to_num(r).astype(np.float32)
    ret = (p1 * rr - COST * np.abs(p1 - p2))[:, m]
    mu, sd = ret.mean(1), ret.std(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sh = np.where(sd > 0, mu / sd * np.sqrt(252), 0)
    tr = (((p1 != 0) & (p1 != p2))[:, m]).sum(1)
    return sh, tr


def pos_of(X, rule):
    fam, payload = rule
    if fam == "多":
        return S1.build_pos(X, payload).astype(np.int8)
    if fam == "空":
        return -S1.build_pos(X, payload).astype(np.int8)
    if fam == "翻轉":
        return np.where(S1.build_pos(X, payload), 1, -1).astype(np.int8)
    L, Sh = S1.build_pos(X, payload[0]), S1.build_pos(X, payload[1])
    return np.where(L, 1, np.where(Sh, -1, 0)).astype(np.int8)


def label(rule):
    fam, p = rule
    f = lambda q: " & ".join(q[0]) + f"｜{q[1]}"  # noqa: E731
    return f"[{fam}] " + (f"多：{f(p[0])}；空：{f(p[1])}" if fam == "組合" else f(p))


def full_sim(pos, X, w):
    idx, r = X["idx"], X["r"]
    ro, rh, rl, rc = X["raw"]
    lim = np.where(idx < pd.Timestamp("2015-06-01"), 0.07, 0.10)
    prev = np.r_[np.nan, rc[:-1]]
    up = (ro == rh) & (ro == rl) & (ro >= prev * (1 + lim - 0.006))
    dn = (ro == rh) & (ro == rl) & (ro <= prev * (1 - lim + 0.006))
    T = len(idx)
    held, ret, hold_, ent = 0, np.zeros(T), np.zeros(T), np.zeros(T, bool)
    for j in range(1, T):
        want = int(pos[j - 1])
        new = held
        if want != held:
            new = want
            if up[j] and (want > held):                    # 漲停鎖死：買不到（開多、回補空都不行）
                new = held
            if dn[j] and (want < held):                    # 跌停鎖死：賣不掉（平多、開空都不行）
                new = held
        c = COST * abs(new - held)
        ent[j] = new != 0 and new != held
        held = new
        hold_[j] = held
        ret[j] = held * (0 if np.isnan(r[j]) else r[j]) - c
    m = S1.wmask(idx, w)
    x = ret[m]
    eq = np.cumprod(1 + x)
    return dict(sharpe=float(x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else 0.0,
                cagr=float(eq[-1] ** (252 / len(x)) - 1), mdd=float((eq / np.maximum.accumulate(eq) - 1).min()),
                trades=int(ent[m].sum()), long=float((hold_[m] > 0).mean()), short=float((hold_[m] < 0).mean()))


def main(sid):
    t0 = time.time()
    X = S1.load(sid)
    idx, r = X["idx"], X["r"]
    md, mv = S1.wmask(idx, S1.DEV), S1.wmask(idx, S1.VAL)
    di = np.nonzero(md)[0]
    rng = np.random.default_rng(0)
    rs = [r]
    for k in rng.integers(250, md.sum() - 250, NPLAC):
        rr = r.copy()
        rr[di] = np.roll(r[di], k)
        rs.append(rr)
    NS = len(rs)
    print(f"{sid}：{len(idx)} 天（截到 {S1.END}），條件 {len(X['names'])} 個；dev {md.sum()} 天", flush=True)
    # 1、2：只做多／只放空（同一批規則正負號）
    descs, shL, shS, trs = [], [[] for _ in range(NS)], [[] for _ in range(NS)], []
    for desc, P in S1.rules_stage1(X["B"], X["names"]):
        descs += desc
        for s, rr in enumerate(rs):
            a, t = evaluate(P, rr, md)
            b, _ = evaluate(-P.astype(np.int8), rr, md)
            shL[s].append(a)
            shS[s].append(b)
            if s == 0:
                trs.append(t)
    shL = [np.concatenate(x) for x in shL]
    shS = [np.concatenate(x) for x in shS]
    trs = np.concatenate(trs)
    q = trs >= MIN_TR
    print(f"  只做多／只放空 各 {len(descs):,} 個（{time.time()-t0:.0f}s）", flush=True)
    # 3：多空翻轉（單一、兩兩）
    nflip = len(X["names"]) + len(X["names"]) * (len(X["names"]) - 1) // 2
    fdesc = descs[:nflip]
    PF = np.where(np.array([S1.build_pos(X, d) for d in fdesc]), 1, -1).astype(np.int8)
    shF, trF = [], None
    for s, rr in enumerate(rs):
        a, t = evaluate(PF, rr, md)
        shF.append(a)
        trF = t if s == 0 else trF
    print(f"  多空翻轉 {len(fdesc):,} 個（{time.time()-t0:.0f}s）", flush=True)
    # 4：多空組合（每個報酬序列各自重選前 100 名）
    best = np.full(NS, -9.0)
    real_combo = []
    for s, rr in enumerate(rs):
        li = np.argsort(-np.where(q, shL[s], -9))[:TOPK]
        si = np.argsort(-np.where(q, shS[s], -9))[:TOPK]
        L = np.array([S1.build_pos(X, descs[i]) for i in li])
        Sh = np.array([S1.build_pos(X, descs[i]) for i in si])
        for a_, l in enumerate(L):
            PC = np.where(l[None], 1, np.where(Sh, -1, 0)).astype(np.int8)
            sh, tr = evaluate(PC, rr, md)
            ok = tr >= MIN_TR
            if ok.any():
                best[s] = max(best[s], sh[ok].max())
            if s == 0:
                real_combo += [(float(sh[b_]), int(tr[b_]), ("組合", (descs[li[a_]], descs[si[b_]]))) for b_ in range(len(si))]
        best[s] = max(best[s], np.where(q, shL[s], -9).max(), np.where(q, shS[s], -9).max(),
                      np.where(trF >= MIN_TR, shF[s], -9).max())
    print(f"  多空組合 {TOPK*TOPK:,} 個 × {NS} 個報酬序列（{time.time()-t0:.0f}s）", flush=True)
    fam_best = {"只做多": float(np.where(q, shL[0], -9).max()), "只放空": float(np.where(q, shS[0], -9).max()),
                "多空翻轉": float(np.where(trF >= MIN_TR, shF[0], -9).max()),
                "多空組合": max(x[0] for x in real_combo if x[1] >= MIN_TR)}
    thr = float(np.quantile(best[1:], 0.95))
    print(f"\n各族最高 dev Sharpe（交易 ≥ {MIN_TR} 次）：" + "、".join(f"{k} {v:.2f}" for k, v in fam_best.items()))
    print(f"安慰劑（{NPLAC} 次，整套重跑）最高 dev Sharpe：中位 {np.median(best[1:]):.2f}、95 分位 {thr:.2f}｜實際 {best[0]:.2f}")
    allr = [(float(shL[0][i]), int(trs[i]), ("多", descs[i])) for i in np.nonzero(q)[0]]
    allr += [(float(shS[0][i]), int(trs[i]), ("空", descs[i])) for i in np.nonzero(q)[0]]
    allr += [(float(shF[0][i]), int(trF[i]), ("翻轉", fdesc[i])) for i in np.nonzero(trF >= MIN_TR)[0]]
    allr += [x for x in real_combo if x[1] >= MIN_TR]
    allr.sort(key=lambda x: -x[0])
    print(f"dev ≥ 2：{sum(1 for x in allr if x[0] >= 2)} 個｜≥ 3：{sum(1 for x in allr if x[0] >= 3)} 個")
    bh = np.ones(len(idx), np.int8)
    bd, bv = full_sim(bh, X, S1.DEV), full_sim(bh, X, S1.VAL)
    print(f"買進持有：dev Sharpe {bd['sharpe']:.2f}（CAGR {bd['cagr']*100:+.1f}%）、val {bv['sharpe']:.2f}（CAGR {bv['cagr']*100:+.1f}%）")
    print("\n== dev 前 5 名（完整模擬：漲跌停）→ val")
    top = []
    for sh, tr, rule in allr[:5]:
        pos = pos_of(X, rule)
        a, b = full_sim(pos, X, S1.DEV), full_sim(pos, X, S1.VAL)
        top.append(dict(rule=label(rule), dev=a, val=b))
        print(f"  {label(rule)}\n     dev Sharpe {a['sharpe']:.2f}（{a['trades']} 次、多 {a['long']:.0%}／空 {a['short']:.0%}、CAGR {a['cagr']*100:+.1f}%、"
              f"MDD {a['mdd']*100:.1f}%） → val Sharpe {b['sharpe']:.2f}（CAGR {b['cagr']*100:+.1f}%、MDD {b['mdd']*100:.1f}%）")
    dv = np.array([x[0] for x in allr[:2000]])
    vv = np.array([evaluate(pos_of(X, x[2])[None], r, mv)[0][0] for x in allr[:2000]])
    print(f"\n== dev 前 2,000 名：dev 中位 {np.median(dv):.2f} → val 中位 {np.median(vv):.2f}（val ≥ 3：{int((vv >= 3).sum())} 個、"
          f"val > 買進持有 {bv['sharpe']:.2f}：{(vv > bv['sharpe']).mean():.0%}）；dev 和 val 相關 {np.corrcoef(dv, vv)[0,1]:+.2f}")
    fams = {}
    for (sh, tr, rule), v in zip(allr[:2000], vv):
        fams.setdefault(rule[0], []).append(v)
    print("   各族 val 中位：" + "、".join(f"{k} {np.median(v):.2f}（{len(v)} 個）" for k, v in fams.items()))
    win = any(t["val"]["sharpe"] >= 3 and t["dev"]["sharpe"] > thr for t in top)
    print(f"\n判定：{'✓' if win else '✗'}（dev 前 5 名有沒有 val Sharpe ≥ 3 且 dev 超過安慰劑門檻 {thr:.2f}）")
    with open(H.LOG, "a") as f:
        f.write(json.dumps(dict(ts=time.strftime("%Y-%m-%d %H:%M:%S"), kind="single_stock", period="dev+val", stock=sid,
                                strategy="第八十批 H193 單一個股期貨多空搜尋", note="第八十批 事前登記（探索）",
                                n_rules=2 * len(descs) + len(fdesc) + TOPK * TOPK, fam_best=fam_best,
                                placebo=dict(best=best[1:].tolist(), thr=thr), real_best=float(best[0]), top=top,
                                bh=dict(dev=bd, val=bv), top2000=dict(dev_med=float(np.median(dv)), val_med=float(np.median(vv)),
                                                                     rho=float(np.corrcoef(dv, vv)[0, 1]), n_val_ge3=int((vv >= 3).sum())),
                                verdict="✓" if win else "✗"), ensure_ascii=False,
                           default=lambda v: round(float(v), 4) if isinstance(v, (float, np.floating)) else str(v)) + "\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "3406")
