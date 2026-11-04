"""第六十一批：市值前 20 大（B 池）的其他三種做法。規則見 HYPOTHESES.md 第六十一批（事前登記）。
- H166 池內輪動（月換股，特徵庫 163 個特徵：單一 × 兩方向、兩兩組合）
- H167 波動目標化（每檔依波動調整部位，和抱著同一檔比）
- H168 除息前後（事件擇時）

用法：.venv/bin/python research/v2/bpool_lab.py cache dev|val   # 月換股日的特徵矩陣（只存前 20 大）
      .venv/bin/python research/v2/bpool_lab.py rot dev|val
      .venv/bin/python research/v2/bpool_lab.py vol dev|val
      .venv/bin/python research/v2/bpool_lab.py exdiv dev|val
H167／H168 用第五十九批的日資料快取（timing_lab.py cache）。
"""
import itertools
import json
import multiprocessing as mp
import pickle
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402
import strategy_lab as SL  # noqa: E402
import timing_lab as TL  # noqa: E402

RES = H.D / "results"
SIDE, ROLL_M, TOPN, NPL = 0.0005, 0.001, 20, 100
KS = (5, 10)
MIN_DEV_M = 72
SEG = TL.SEG
TRIALS = 31119
NAME = "第六十一批 B 池其他方法"


def mask(dates, seg):
    return TL.mask(dates, *SEG[seg])


def alpha_t(s, h, m, lag):
    y, x = s[m], h[m]
    X = np.column_stack([np.ones(len(x)), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for L in range(1, lag + 1):
        G = Xe[L:].T @ Xe[:-L]
        S += (1 - L / (lag + 1)) * (G + G.T)
    Bi = np.linalg.inv(X.T @ X)
    V = Bi @ S @ Bi
    return float(b[0] / np.sqrt(V[0, 0])), float(b[0]), float(b[1])


# ═════════ H166 池內輪動 ═════════
def cache(period):
    import feature_library as FL
    end = H.PERIODS[period][1]
    p0 = "2010-01-01" if period == "dev" else "2020-01-01"
    t0 = time.time()
    lab = SL.Lab(end)
    P = lab.P
    dates, R, E = lab.monthly(p0, end)
    cols = lab.cols
    mv = P["mv"].shift(1).reindex(index=dates, columns=cols).values
    rk = pd.DataFrame(np.where(E & np.isfinite(mv), mv, np.nan)).rank(axis=1, ascending=False).values
    M = E & (rk <= TOPN)
    U = np.flatnonzero(M.any(axis=0))
    print(f"載入 {time.time()-t0:.0f}s；換股日 {len(dates)} 個；曾在前 20 大 {len(U)} 檔", flush=True)
    X, bad = {}, {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i, f in enumerate(FL.FEATS):
            try:
                A = np.asarray(lab.feat(f, dates, E), float)
            except Exception as e:   # noqa: BLE001
                bad[f] = repr(e)[:200]
                continue
            X[f] = np.where(M[:, U], A[:, U], np.nan)
            if i % 20 == 0:
                print(f"  特徵 {i+1}/{len(FL.FEATS)}（{time.time()-t0:.0f}s）", flush=True)
    import batch55_s8 as B55
    import xs
    days = P["open"].index
    mtx = B55.mtx_chain(days)
    rd = xs.rebalance_dates(days, pd.Timestamp(p0), pd.Timestamp(end))
    assert list(rd[:-1]) == list(dates), "換股日對不上"
    out = dict(dates=dates, cols=cols[U], M=M[:, U], R=R[:, U], X=X, bad=bad, end=end,
               mtx=np.array([mtx.loc[d2] / mtx.loc[d] - 1 for d, d2 in zip(rd[:-1], rd[1:])], float))
    with open(RES / f"bpool_{period}.pkl", "wb") as f:
        pickle.dump(out, f)
    print(f"快取完成 {time.time()-t0:.0f}s；特徵 {len(X)} 個；失敗 {len(bad)} 個 {list(bad)[:10]}")


def load(period):
    with open(RES / f"bpool_{period}.pkl", "rb") as f:
        return pickle.load(f)


def lib_sign():
    """特徵庫總表的方向欄（大＝好端要乘的號）。"""
    out = {}
    txt = (Path(__file__).resolve().parent / "FEATURES.md").read_text()
    sec = txt.split("## 總表", 1)[1]
    for line in sec.splitlines():
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) > 4 and c[3] in ("+", "−", "-"):
            out[c[0]] = 1 if c[3] == "+" else -1
    return out


def pct(X, M):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return pd.DataFrame(np.where(M & np.isfinite(X), X, np.nan)).rank(axis=1, pct=True).values


class Rot:
    def __init__(self, B):
        self.dates, self.M, self.R = B["dates"], B["M"], np.nan_to_num(B["R"])
        self.n = np.maximum(self.M.sum(axis=1), 1)
        w = self.M / self.n[:, None]
        dw = np.abs(np.diff(w, axis=0, prepend=0)).sum(axis=1)
        live = self.M.sum(axis=1) > 0
        self.h = np.where(live, (w * self.R).sum(axis=1) - SIDE * dw - ROLL_M, np.nan)
        self.T, self.U = self.M.shape
        self.colrank = np.arange(self.U)   # 代號小→大（cols 已排序）

    def run(self, S, k):
        """S：分數（大＝好），非成員或缺值為 nan。回傳月報酬（無效月份 nan）與選股矩陣。"""
        ok = np.isfinite(S) & self.M
        nv = ok.sum(axis=1)
        key = np.where(ok, S, -np.inf)
        idx = np.argsort(-key, axis=1, kind="stable")[:, :k]   # 同分依代號小→大（cols 已排序，stable）
        sel = np.zeros_like(ok)
        np.put_along_axis(sel, idx, True, axis=1)
        sel &= ok
        valid = nv >= k + 5
        sel[~valid] = False
        w = sel / k
        dw = np.abs(np.diff(w, axis=0, prepend=0)).sum(axis=1)
        s = (w * self.R).sum(axis=1) - SIDE * dw - ROLL_M
        s = np.where(valid & np.isfinite(self.h), s, np.nan)
        return s, sel


def rot_menu(X, sign):
    """回傳 (菜單 A, 菜單 B)：每個是 [(名稱, [(特徵, 方向)], k)]。"""
    fs = sorted(X, key=lambda f: list(X).index(f))
    A = [(f"{f}{'+' if d > 0 else '−'}", [(f, d)], k) for f in fs for d in (1, -1) for k in KS]
    A += [("H136分數", [("H2", 1), ("H87", 1), ("H97", 1)], k) for k in KS]
    Bm = [(f"{f}{'+' if sign.get(f, 1) > 0 else '−'}＆{g}{'+' if sign.get(g, 1) > 0 else '−'}",
           [(f, sign.get(f, 1)), (g, sign.get(g, 1))], k) for f, g in itertools.combinations(fs, 2) for k in KS]
    return A, Bm


_G = {}


def score(spec, Pp, M):
    parts = spec[1]
    if len(parts) == 1:
        f, d = parts[0]
        return Pp[f] if d > 0 else Pp[f + "−"]
    st = np.stack([Pp[f] if d > 0 else Pp[f + "−"] for f, d in parts])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = np.nanmean(st, axis=0)
    out[np.isfinite(st).sum(axis=0) < 2] = np.nan   # 兩兩組合：兩項都要有；H136 分數：3 項 ≥ 2
    return out


def pct_all(X, M, perm=None):
    Pp = {}
    for f, A in X.items():
        if perm is not None:
            A = np.take_along_axis(A, perm, axis=1)
        Pp[f] = pct(A, M)
        Pp[f + "−"] = pct(-A, M)
    return Pp


def month_perm(M, rng):
    """每個月把成員之間的位置用同一個隨機排列打亂（非成員不動）。"""
    T, U = M.shape
    perm = np.tile(np.arange(U), (T, 1))
    for t in range(T):
        mem = np.flatnonzero(M[t])
        perm[t, mem] = rng.permutation(mem)
    return perm


def scan_rot(menu, Pp, rot, m, ndev_ok):
    out = np.full(len(menu), np.nan)
    for i, sp in enumerate(menu):
        if not ndev_ok[i]:
            continue
        s, _ = rot.run(score(sp, Pp, rot.M), sp[2])
        mm = m & np.isfinite(s)
        if mm.sum() < MIN_DEV_M:
            continue
        out[i] = alpha_t(s, rot.h, mm, 3)[0]
    return out


def _rot_pl(seed):
    rot, X, menuA, menuB, m, okA, okB = (_G[k] for k in ("rot", "X", "A", "B", "m", "okA", "okB"))
    rng = np.random.default_rng(2061_000 + seed)
    Pp = pct_all(X, rot.M, month_perm(rot.M, rng))
    a = scan_rot(menuA, Pp, rot, m, okA)
    b = scan_rot(menuB, Pp, rot, m, okB)
    return float(np.nanmax(a)), float(np.nanmax(b))


def rot_detail(sp, Pp, rot, B, seg_main, segs):
    k = sp[2]
    s, sel = rot.run(score(sp, Pp, rot.M), k)
    d = rot.dates
    m = mask(d, seg_main) & np.isfinite(s)
    t, a, b = alpha_t(s, rot.h, m, 3)
    out = dict(name=sp[0], k=k, alpha_t=t, alpha_ann=a * 12, beta=b, months=int(m.sum()),
               sharpe=TL.sharpe(s[m]) / np.sqrt(21), sharpe_pool=TL.sharpe(rot.h[m]) / np.sqrt(21))
    for g in segs:
        mg = mask(d, g) & np.isfinite(s)
        out[f"alpha_{g}"] = alpha_t(s, rot.h, mg, 3)[1] * 12 if mg.sum() > 12 else np.nan
    ex = pd.Series(s[m] - rot.h[m]).groupby(d[m].year).sum()
    out["yrs_pos"] = f"{int((ex > 0).sum())}/{len(ex)}"
    out["by_year"] = (ex * 100).round(1).to_dict()
    # 後 k 檔（反向分數）
    S = score(sp, Pp, rot.M)
    s2, _ = rot.run(-S, k)
    out["top_minus_bottom_ann"] = float(np.nanmean((s - s2)[m]) * 12)
    w = sel / k
    out["turnover"] = float(np.abs(np.diff(w, axis=0)).sum(axis=1)[m[1:]].mean() / 2)
    # 避險版：s − β×小台指（同段迴歸，只當描述）
    mt = B["mtx"]
    mh = m & np.isfinite(mt)
    bm = np.polyfit(mt[mh], s[mh], 1)[0]
    hd = s[mh] - bm * mt[mh]
    out["hedged_ann"] = float(hd.mean() * 12)
    out["hedged_sharpe"] = float(hd.mean() / hd.std(ddof=1) * np.sqrt(12))
    out["beta_mtx"] = float(bm)
    return out, s


def h136_monthly(s, dates, period, segs):
    import portfolio_lab as PL
    led = pd.read_parquet(RES / "sleeves" / period / "H136.parquet")
    md = led.index
    x = pd.Series(s, index=dates).reindex(md).values
    h = led.ret.values
    seg = "dev" if period == "dev" else "val"
    g3, ok3 = PL.gate3(x, h, md, segs)
    rho = PL.corr(x, h, md, seg)
    sp = PL.st(h, md, seg).get("sharpe", np.nan)
    sb = PL.st(x, md, seg).get("sharpe", np.nan)
    return dict(rho_h136=rho, gate3=g3, gate3_ok=ok3, margin=f"SR_B {sb:.2f} vs ρ·SR_P {rho*sp:.2f}", margin_ok=bool(sb > rho * sp))


def rot_setup(period):
    B = load(period)
    rot = Rot(B)
    X = B["X"]
    sign = lib_sign()
    menuA, menuB = rot_menu(X, sign)
    Pp = pct_all(X, rot.M)
    return B, rot, X, sign, menuA, menuB, Pp


def rot_dev():
    t0 = time.time()
    B, rot, X, sign, menuA, menuB, Pp = rot_setup("dev")
    miss = [f for f in X if f not in sign]
    m = mask(rot.dates, "dev") & np.isfinite(rot.h)
    # dev 有效月份：每個特徵在 ≥ k+5 檔成員有值的月份數
    nval = {f: (np.isfinite(A) & rot.M).sum(axis=1) for f, A in X.items()}

    def ok(sp):
        need = sp[2] + 5
        if sp[0] == "H136分數":
            return True
        return all((m & (nval[f] >= need)).sum() >= MIN_DEV_M for f, _ in sp[1])
    okA = np.array([ok(sp) for sp in menuA])
    okB = np.array([ok(sp) for sp in menuB])
    print(f"載入 {time.time()-t0:.0f}s；特徵 {len(X)}（失敗 {len(B['bad'])}：{list(B['bad'])}；無方向 {miss}）；"
          f"菜單 A {len(menuA)}（可評 {okA.sum()}）、B {len(menuB)}（可評 {okB.sum()}）；對照組 dev Sharpe {TL.sharpe(rot.h[m])/np.sqrt(21):.2f}", flush=True)
    t1 = time.time()
    ra = scan_rot(menuA, Pp, rot, m, okA)
    print(f"菜單 A {time.time()-t1:.0f}s：alpha t ≥ 2 的 {(ra >= 2).sum()} 個，最高 {np.nanmax(ra):.2f}", flush=True)
    t1 = time.time()
    rb = scan_rot(menuB, Pp, rot, m, okB)
    print(f"菜單 B {time.time()-t1:.0f}s：alpha t ≥ 2 的 {(rb >= 2).sum()} 個，最高 {np.nanmax(rb):.2f}", flush=True)
    _G.update(rot=rot, X=X, A=menuA, B=menuB, m=m, okA=okA, okB=okB)
    t1 = time.time()
    with mp.get_context("fork").Pool(8) as pool:
        pl = np.array(pool.map(_rot_pl, range(NPL)))
    pa, pb = np.percentile(pl[:, 0], 95), np.percentile(pl[:, 1], 95)
    print(f"安慰劑 {NPL} 次（{time.time()-t1:.0f}s）：菜單 A 最佳 50／95 分位 {np.percentile(pl[:,0],50):.2f}／{pa:.2f}；"
          f"菜單 B {np.percentile(pl[:,1],50):.2f}／{pb:.2f}", flush=True)
    for lab_, menu, r in (("A", menuA, ra), ("B", menuB, rb)):
        o = np.argsort(-np.nan_to_num(r, nan=-9))[:12]
        print(f"\n  菜單 {lab_} alpha t 前 12：" + "；".join(f"{menu[i][0]}/k{menu[i][2]} {r[i]:.2f}" for i in o))
    cand = [("A", i, ra[i]) for i in np.flatnonzero(ra >= max(2, pa))] + [("B", i, rb[i]) for i in np.flatnonzero(rb >= max(2, pb))]
    cand.sort(key=lambda x: -x[2])
    print(f"\n== 超過門檻 {len(cand)} 個，檢查 A、B 段 alpha > 0、逐年 ≥ 7/9")
    rows = []
    for lab_, i, _ in cand[:400]:
        sp = (menuA if lab_ == "A" else menuB)[i]
        r, s = rot_detail(sp, Pp, rot, B, "dev", ("A", "B"))
        r["menu"] = lab_
        r["feats"] = [f for f, _ in sp[1]]
        r["pass"] = r["alpha_A"] > 0 and r["alpha_B"] > 0 and int(r["yrs_pos"].split("/")[0]) >= 7
        rows.append(r)
    T = pd.DataFrame(rows)
    sel, used = [], set()
    if len(T):
        for _, r in T[T["pass"]].sort_values("alpha_t", ascending=False).iterrows():
            if used & set(r.feats):
                continue
            sel.append((r.menu, r["name"], int(r.k)))
            used |= set(r.feats)
            if len(sel) == 3:
                break
        print(f"  通過 {int(T['pass'].sum())} 個；前 10：")
        for _, r in T[T["pass"]].sort_values("alpha_t", ascending=False).head(10).iterrows():
            print(f"   [{r.menu}] {r['name']}/k{r.k}：alpha t {r.alpha_t:.2f}（年化 {r.alpha_ann*100:+.1f}%）A {r.alpha_A*100:+.1f}% B {r.alpha_B*100:+.1f}% "
                  f"逐年 {r.yrs_pos}｜Sharpe {r.sharpe:.2f}（池 {r.sharpe_pool:.2f}）｜前減後 {r.top_minus_bottom_ann*100:+.1f}%｜換手 {r.turnover:.0%}"
                  f"｜避險版 {r.hedged_ann*100:+.1f}%（Sharpe {r.hedged_sharpe:.2f}）")
    extra = {}
    for mn, nm, k in sel:
        sp = next(x for x in (menuA if mn == "A" else menuB) if x[0] == nm and x[2] == k)
        r, s = rot_detail(sp, Pp, rot, B, "dev", ("A", "B"))
        hc = h136_monthly(s, rot.dates, "dev", ("A", "B"))
        extra[f"{nm}/k{k}"] = hc
        print(f"  選 val：[{mn}] {nm}/k{k}｜和 H136 相關 {hc['rho_h136']:+.2f}｜關卡 3 {hc['gate3']} {'✓' if hc['gate3_ok'] else '✗'}｜{hc['margin']}")
    if len(T):
        T.drop(columns=["by_year"]).to_csv(RES / "batch61_rot_dev.csv", index=False)
    pd.DataFrame(dict(name=[x[0] for x in menuA] + [x[0] for x in menuB], k=[x[2] for x in menuA] + [x[2] for x in menuB],
                      menu=["A"] * len(menuA) + ["B"] * len(menuB), alpha_t=np.concatenate([ra, rb]))).to_csv(RES / "batch61_rot_menu.csv", index=False)
    SL.log(dict(kind="strategy", period="dev", strategy=f"{NAME} H166 池內輪動", note="第六十一批 事前登記",
                menuA=len(menuA), menuB=len(menuB), evalA=int(okA.sum()), evalB=int(okB.sum()),
                n_t2=int((ra >= 2).sum() + (rb >= 2).sum()), bestA=float(np.nanmax(ra)), bestB=float(np.nanmax(rb)),
                placeboA_p95=float(pa), placeboB_p95=float(pb), n_cand=len(cand), n_pass=int(T["pass"].sum()) if len(T) else 0,
                selected=[f"{a}|{b}|{c}" for a, b, c in sel], extra=extra, trials_cum=TRIALS))


def rot_val():
    rec = [json.loads(l) for l in open(H.LOG) if f'"{NAME} H166 池內輪動"' in l and '"period": "dev"' in l][-1]
    sel = [s.split("|") for s in rec["selected"]]
    if not sel:
        print("dev 沒有選出 → 依登記不跑 val")
        return
    B, rot, X, sign, menuA, menuB, Pp = rot_setup("val")
    out = {}
    for mn, nm, k in sel:
        sp = next(x for x in (menuA if mn == "A" else menuB) if x[0] == nm and x[2] == int(k))
        r, s = rot_detail(sp, Pp, rot, B, "val", ())
        ok = r["alpha_t"] >= 2 and r["sharpe"] > r["sharpe_pool"]
        hc = h136_monthly(s, rot.dates, "val", ("val",))
        out[f"{nm}/k{k}"] = dict(**r, **hc, verdict="✓" if ok else "✗")
        print(f"  [{mn}] {nm}/k{k}：alpha t {r['alpha_t']:.2f}（年化 {r['alpha_ann']*100:+.1f}%）Sharpe {r['sharpe']:.2f}（池 {r['sharpe_pool']:.2f}）"
              f"｜逐年 {r['by_year']}｜前減後 {r['top_minus_bottom_ann']*100:+.1f}%｜避險版 {r['hedged_ann']*100:+.1f}%"
              f"｜和 H136 相關 {hc['rho_h136']:+.2f}｜關卡 3 {hc['gate3']} → {'✓' if ok else '✗'}")
    SL.log(dict(kind="strategy", period="val", strategy=f"{NAME} H166 池內輪動", note="第六十一批 dev 選出的", result=out))


# ═════════ H167 波動目標化、H168 除息前後（日頻，和抱著同一檔比） ═════════
def wrun(bk, w, side=SIDE):
    """w：收盤決定的部位（日 × 股，可 > 1）→ 隔天開盤起生效。"""
    eff = np.vstack([np.zeros((1, w.shape[1])), w[:-1]]) * bk.mem
    dw = np.abs(np.diff(eff, axis=0, prepend=0)).sum(axis=1)
    s = ((eff * bk.r).sum(axis=1) - side * dw - ROLL_M / 21 * eff.sum(axis=1)) / bk.n
    return np.where(np.isfinite(bk.h), s, np.nan), eff


def ewvol(r, lam=0.94):
    return np.sqrt((r ** 2).ewm(alpha=1 - lam, adjust=False, min_periods=20).mean())


def vol_weights(D):
    C = D["close"]
    r = C.pct_change(fill_method=None)
    rm = D["mkt"].pct_change(fill_method=None)
    wk = pd.Series(C.index, index=C.index).dt.to_period("W")
    week_end = np.broadcast_to(wk.ne(wk.shift(-1)).values[:, None], C.shape)
    W = {}
    for src in ("個股", "大盤"):
        x = r if src == "個股" else pd.DataFrame(np.broadcast_to(rm.values[:, None], C.shape), index=C.index, columns=C.columns)
        for win in ("20日", "60日", "EWMA"):
            sig = (x.rolling(20, min_periods=15).std() if win == "20日" else x.rolling(60, min_periods=45).std() if win == "60日" else ewvol(x))
            tgt = sig.rolling(TL.WIN, min_periods=200).median()
            raw = (tgt / sig).replace([np.inf, -np.inf], np.nan)
            for cap in (1.0, 1.5, 2.0):
                w = raw.clip(upper=cap)
                w = w.where(week_end).ffill().fillna(0).values
                W[(src, win, cap)] = w
    return W


def ex_dates(D):
    """除息（權）日：還原收盤／原始收盤的比例單日變動 > 0.1%。回傳布林矩陣（日 × 股）。"""
    C = D["close"]
    E = np.zeros(C.shape, bool)
    for j, c in enumerate(C.columns):
        a = pd.read_parquet(H.D / "adj" / f"{c}.parquet", columns=["date", "close"])
        r = pd.read_parquet(H.D / "raw" / f"{c}.parquet", columns=["date", "close"])
        a["date"], r["date"] = pd.to_datetime(a.date), pd.to_datetime(r.date)
        f = (a.set_index("date").close / r.set_index("date").close.replace(0, np.nan)).reindex(C.index)
        ch = (f / f.ffill().shift(1) - 1).abs()
        E[:, j] = (ch > 0.001).fillna(False).values
    return E


EX_SPECS = [(a, b) for a in (-10, -5, -3, -1) for b in (0, 1, 5, 10, 20)] + [(0, b) for b in (1, 5, 10, 20)]


def ex_hold(E, a, b):
    """eff[t]＝t 在 [除息日 + a, 除息日 + b) 之間（持有 t 開盤到 t+1 開盤）。"""
    T = E.shape[0]
    eff = np.zeros(E.shape, bool)
    ti, sj = np.nonzero(E)
    for o in range(a, b):
        tt = ti + o
        ok = (tt >= 0) & (tt < T)
        eff[tt[ok], sj[ok]] = True
    return eff


def daily_detail(eff_or_w, bk, seg_main, segs, is_w):
    w = eff_or_w.astype(float)
    inn = np.vstack([w[1:], np.zeros((1, w.shape[1]))])   # eff → 前一日收盤決定
    s, eff = wrun(bk, inn if not is_w else w)
    s2, _ = wrun(bk, inn if not is_w else w, side=2 * SIDE)
    d = bk.days
    m = TL.mask(d, *SEG[seg_main]) & np.isfinite(s)
    t, a, b = TL.alpha_t(s, bk.h, m)
    out = dict(alpha_t=t, alpha_ann=a * 252, beta=b, alpha_t_cost2=TL.alpha_t(s2, bk.h, m)[0], sharpe=TL.sharpe(s[m]), sharpe_hold=TL.sharpe(bk.h[m]),
               expo=float(eff[m].sum(axis=1).mean() / bk.n[m].mean()))
    for g in segs:
        mg = TL.mask(d, *SEG[g]) & np.isfinite(s)
        out[f"sharpe_{g}"], out[f"hold_{g}"] = TL.sharpe(s[mg]), TL.sharpe(bk.h[mg])
    pos_a = []
    for j in range(eff.shape[1]):
        mj = m & bk.mem[:, j]
        if mj.sum() < 250:
            continue
        y = eff[mj, j] * bk.r[mj, j]
        X = np.column_stack([np.ones(mj.sum()), bk.r[mj, j]])
        pos_a.append(np.linalg.lstsq(X, y, rcond=None)[0][0] > 0)
    out["breadth"] = float(np.mean(pos_a)) if pos_a else np.nan
    out["alpha_by_year"] = (pd.Series(s[m] - b * bk.h[m]).groupby(d[m].year).sum() * 100).round(1).to_dict()
    return out, s


def _daily_scan(seed):
    bk, specs, mk, m, kind = (_G[k] for k in ("bk", "specs", "mk", "m", "kind"))
    T, N = bk.mem.shape
    if seed is None:
        ix = None
    else:
        rng = np.random.default_rng(2061_500 + seed)
        k = rng.integers(TL.WIN, T - TL.WIN, N)
        ix = (np.arange(T)[:, None] - k[None, :]) % T
    out = np.full(len(specs), np.nan)
    for i, sp in enumerate(specs):
        w = mk(sp)
        if ix is not None:
            if kind == "vol" and sp[0] == "大盤":
                k0 = int(np.random.default_rng(2061_900 + seed).integers(TL.WIN, T - TL.WIN))
                w = np.roll(w, k0, axis=0)
            else:
                w = w[ix, np.arange(N)[None, :]]
        if kind == "exdiv":
            w = np.vstack([w[1:], np.zeros((1, N))])
        s, _ = wrun(bk, w.astype(float))
        out[i] = TL.alpha_t(s, bk.h, m & np.isfinite(s))[0]
    return out


def _daily_pl(seed):
    return float(np.nanmax(_daily_scan(seed)))


def daily_setup(period, kind):
    D = TL.load(period)
    bk = TL.Book(D)
    if kind == "vol":
        W = vol_weights(D)
        specs = list(W)
        mk = W.__getitem__
    else:
        E = ex_dates(D) & D["member"].values
        specs = EX_SPECS
        mk = (lambda sp: ex_hold(E, *sp))   # noqa: E731
    return D, bk, specs, mk


def daily_dev(kind):
    t0 = time.time()
    D, bk, specs, mk = daily_setup("dev", kind)
    m = TL.mask(bk.days, *SEG["dev"]) & np.isfinite(bk.h)
    _G.update(bk=bk, specs=specs, mk=mk, m=m, kind=kind)
    real = _daily_scan(None)
    hyp = "H167 波動目標化" if kind == "vol" else "H168 除息前後"
    print(f"{hyp}：載入 {time.time()-t0:.0f}s；菜單 {len(specs)}；alpha t ≥ 2 的 {(real >= 2).sum()} 個，最高 {np.nanmax(real):.2f}", flush=True)
    if kind == "exdiv":
        E = ex_dates(D) & D["member"].values
        print(f"  dev 期間成員的除息（權）事件 {int(E[m].sum())} 個", flush=True)
    with mp.get_context("fork").Pool(8) as pool:
        pl = np.array(pool.map(_daily_pl, range(NPL)))
    p95 = float(np.percentile(pl, 95))
    print(f"  安慰劑菜單最佳 50／95 分位 {np.percentile(pl, 50):.2f}／{p95:.2f}", flush=True)
    for i in np.argsort(-np.nan_to_num(real, nan=-9)):
        print(f"   {specs[i]}：alpha t {real[i]:.2f}")
    rows = []
    for i in np.flatnonzero(real >= max(2, p95)):
        sp = specs[i]
        r, s = daily_detail(mk(sp), bk, "dev", ("A", "B"), kind == "vol")
        r["spec"] = str(sp)
        if kind == "exdiv":
            r["events"] = int((ex_dates(D) & D["member"].values)[m].sum())
        r["pass"] = r["sharpe_A"] > r["hold_A"] and r["sharpe_B"] > r["hold_B"] and r["breadth"] >= 0.5 and (kind == "vol" or r["events"] >= 100)
        rows.append(r)
        print(f"  候選 {sp}：{ {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} }")
    T = pd.DataFrame(rows)
    sel = []
    if len(T) and T["pass"].any():
        sel = [T[T["pass"]].sort_values("alpha_t", ascending=False).iloc[0].spec]
    extra = {}
    for sp in sel:
        spec = next(x for x in specs if str(x) == sp)
        r, s = daily_detail(mk(spec), bk, "dev", ("A", "B"), kind == "vol")
        hc = TL.h136_check(s, bk, D, "dev", ("A", "B"))
        extra[sp] = dict(**r, **hc)
        print(f"  選 val：{sp}｜和 H136 相關 {hc['rho_h136']:+.2f}｜關卡 3 {hc['gate3']}")
    SL.log(dict(kind="strategy", period="dev", strategy=f"{NAME} {hyp}", note="第六十一批 事前登記", menu=len(specs),
                n_t2=int((real >= 2).sum()), best=float(np.nanmax(real)), placebo_p50=float(np.percentile(pl, 50)), placebo_p95=p95,
                menu_t={str(s): float(v) for s, v in zip(specs, real)}, candidates=T.to_dict("records") if len(T) else [],
                selected=sel, extra=extra, trials_cum=TRIALS))


def daily_val(kind):
    hyp = "H167 波動目標化" if kind == "vol" else "H168 除息前後"
    rec = [json.loads(l) for l in open(H.LOG) if f'"{NAME} {hyp}"' in l and '"period": "dev"' in l][-1]
    if not rec["selected"]:
        print(f"{hyp}：dev 沒有選出 → 依登記不跑 val")
        return
    D, bk, specs, mk = daily_setup("val", kind)
    out = {}
    for sp in rec["selected"]:
        spec = next(x for x in specs if str(x) == sp)
        r, s = daily_detail(mk(spec), bk, "val", ("val",), kind == "vol")
        ok = r["alpha_t"] >= 2 and r["sharpe"] > r["sharpe_hold"]
        hc = TL.h136_check(s, bk, D, "val", ("val",))
        out[sp] = dict(**r, **hc, verdict="✓" if ok else "✗")
        print(f"  {hyp} {sp}：alpha t {r['alpha_t']:.2f}（年化 {r['alpha_ann']*100:+.1f}%）Sharpe {r['sharpe']:.2f}（持有 {r['sharpe_hold']:.2f}）"
              f"｜逐年 alpha {r['alpha_by_year']}｜廣度 {r['breadth']:.0%}｜和 H136 相關 {hc['rho_h136']:+.2f} → {'✓' if ok else '✗'}")
    SL.log(dict(kind="strategy", period="val", strategy=f"{NAME} {hyp}", note="第六十一批 dev 選出的", result=out))


if __name__ == "__main__":
    cmd, per = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "dev")
    if cmd == "cache":
        cache(per)
    elif cmd == "rot":
        rot_dev() if per == "dev" else rot_val()
    elif cmd in ("vol", "exdiv"):
        daily_dev(cmd) if per == "dev" else daily_val(cmd)
