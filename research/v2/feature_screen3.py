"""第三十九批 FS3：42 個 FS1 候選特徵重篩 — 條件（H136 候選池內）、右尾（全池前 10%）、衰減速度。規則見 HYPOTHESES.md 第三十九批。

引擎同 FS1（feature_screen.forward：換股日開盤 → 下個換股日開盤；特徵取前一交易日）。只用 dev；只當原料，不判定能不能交易。
收＝A 段 |t| ≥ 1 且 B 段同號；安慰劑＝把特徵的股票代號整個打亂（同 FS1 的亂數種子），每個特徵 5 次。

用法：.venv/bin/python research/v2/feature_screen3.py
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
import feature_screen as FS  # noqa: E402
import harness as H  # noqa: E402
import hypotheses as Hy  # noqa: E402
import xs  # noqa: E402

FEATS, HALVES = FS.FEATS, FS.HALVES
NPL, NC, MIN_C, TOPC, TAIL, LAGS, MIN_M = 5, 100, 50, 20, 0.10, (1, 2, 3, 6, 12), 24
H136 = ("H2", "H87", "H97")


def spear(x, y):
    return np.corrcoef(rankdata(x), rankdata(y))[0, 1]


def candidates(Xs, R, E):
    """每月 H136 分數前 NC 檔的欄位位置（池內百分位平均、≥ 2 項；同 FS2 引擎，同分依代號小者優先）。"""
    M = E & np.isfinite(R)
    pct = [pd.DataFrame(np.where(M, X, np.nan)).rank(axis=1, pct=True).values for X in Xs]
    st = np.stack(pct)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        comp = np.nanmean(st, axis=0)
    comp[(np.isfinite(st).sum(axis=0) < 2) | ~M] = np.nan
    out = []
    for t in range(comp.shape[0]):
        ok = np.flatnonzero(np.isfinite(comp[t]))
        out.append(ok[np.argsort(-comp[t, ok], kind="stable")[:NC]] if len(ok) >= NC else None)
    return out


def monthly(X, R, E, C):
    """每月：條件 IC、條件前 20 超額、右尾前 10% 超額、大贏家命中倍數。"""
    T = X.shape[0]
    out = np.full((T, 4), np.nan)
    for t in range(T):
        m = E[t] & np.isfinite(X[t]) & np.isfinite(R[t])
        if m.sum() >= FS.MIN_N:
            x, r = X[t, m], R[t, m]
            n = len(x)
            top = rankdata(x, method="ordinal") > (1 - TAIL) * n        # 同分依代號（同 FS1 的 rank(method="first")）
            win = rankdata(r, method="ordinal") > (1 - TAIL) * n
            out[t, 2] = r[top].mean() - r.mean()
            out[t, 3] = (top & win).sum() / win.sum() / (top.sum() / n)
        if C[t] is not None:
            c = C[t][np.isfinite(X[t, C[t]])]
            if len(c) >= MIN_C:
                x, r = X[t, c], R[t, c]
                out[t, 0] = spear(x, r)
                top = rankdata(x, method="ordinal") > len(x) - TOPC
                out[t, 1] = r[top].mean() - r.mean()
    return out


def halves(v, dates):
    res = {}
    for h, (a, b) in HALVES.items():
        k = (dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b)) & np.isfinite(v)
        x = v[k]
        n = len(x)
        res[h] = dict(n=n, mean=x.mean() if n else np.nan, t=x.mean() / x.std(ddof=1) * np.sqrt(n) if n > 2 else np.nan)
    return res


def verdict(st):
    a, b = st["A"], st["B"]
    if a["n"] < MIN_M or b["n"] < MIN_M:
        return "資料不足"
    return "收" if abs(a["t"]) >= 1 and np.sign(b["mean"]) == np.sign(a["mean"]) else "—"


def decay(X, R, E):
    """k 月後那一窗的全池 IC 與前 10% 超額（整段 dev）。"""
    T = X.shape[0]
    res = {}
    for k in LAGS:
        ic, tx = [], []
        for t in range(T - k + 1):
            r = R[t + k - 1]
            m = E[t] & np.isfinite(X[t]) & np.isfinite(r)
            if m.sum() < FS.MIN_N:
                continue
            x, y = X[t, m], r[m]
            ic.append(spear(x, y))
            top = rankdata(x, method="ordinal") > (1 - TAIL) * len(x)
            tx.append(y[top].mean() - y.mean())
        ic = np.array(ic)
        res[k] = dict(ic=ic.mean(), t=ic.mean() / ic.std(ddof=1) * np.sqrt(len(ic)), tail=np.mean(tx), n=len(ic))
    s = np.sign(res[1]["ic"])
    v = [s * res[k]["ic"] for k in LAGS]
    hl = np.nan
    for i in range(1, len(LAGS)):
        if v[i] < v[0] / 2:
            k0, k1 = LAGS[i - 1], LAGS[i]
            hl = k0 + (v[i - 1] - v[0] / 2) / (v[i - 1] - v[i]) * (k1 - k0)
            break
    return res, hl


def main():
    t0 = time.time()
    P = xs.load_panel(HALVES["B"][1], fundamentals=True)
    for add in (xs.add_chips, xs.add_fin, xs.add_v16, xs.add_v17):
        add(P)
    dates, Rdf, Edf = FS.forward(P)
    cols = P["close"].columns
    dates = pd.DatetimeIndex(dates)
    R = Rdf.reindex(index=dates, columns=cols).values
    E = Edf.reindex(index=dates, columns=cols).fillna(False).values.astype(bool)

    def mat(f):
        S = Hy.FNS[f][0](P).replace([np.inf, -np.inf], np.nan)
        return S.shift(1).reindex(index=dates, columns=cols).values

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        C = candidates([mat(f) for f in H136], R, E)
        print(f"FS3：{len(FEATS)} 個特徵 × 安慰劑 {NPL}；換股日 {len(dates)} 個（候選池有值 {sum(c is not None for c in C)} 個）；"
              f"載入 {time.time()-t0:.0f}s", flush=True)
        names = ["cond_ic", "cond_top20", "tail", "lift"]
        res, plc = {}, []
        for f in FEATS:
            X = mat(f)
            mo = monthly(X, R, E, C)
            st = {nm: halves(mo[:, i], dates) for i, nm in enumerate(names)}
            dc, hl = decay(X, R, E)
            res[f] = dict(desc=Hy.FNS[f][1], st=st, cond=verdict(st["cond_ic"]), tailv=verdict(st["tail"]), cond20=verdict(st["cond_top20"]),
                          decay=dc, half_life=hl)
            for k in range(NPL):
                perm = np.random.default_rng(1000 + k).permutation(X.shape[1])
                pm = monthly(X[:, perm], R, E, C)
                plc.append(dict(f=f, k=k, cond=verdict(halves(pm[:, 0], dates)), tail=verdict(halves(pm[:, 2], dates)),
                                cond20=verdict(halves(pm[:, 1], dates)),   # 事後加的描述：候選池內前 20
                                tA_cond=halves(pm[:, 0], dates)["A"]["t"], tA_tail=halves(pm[:, 2], dates)["A"]["t"],
                                tA_cond20=halves(pm[:, 1], dates)["A"]["t"]))
            a, b = st["cond_ic"]["A"], st["cond_ic"]["B"]
            ta, tb = st["tail"]["A"], st["tail"]["B"]
            print(f"  {f:<4} {res[f]['desc']:<12} 條件 IC A {a['mean']:+.3f}（t {a['t']:+.2f}）B {b['mean']:+.3f}（t {b['t']:+.2f}）→ {res[f]['cond']:<4}"
                  f" 右尾 A {ta['mean']*1200:+5.1f}（t {ta['t']:+.2f}）B {tb['mean']*1200:+5.1f}（t {tb['t']:+.2f}）→ {res[f]['tailv']:<4}"
                  f" 半衰期 {hl:4.1f} 月  [{time.time()-t0:.0f}s]", flush=True)

    PL = pd.DataFrame(plc)
    fs1 = pd.read_csv(H.D / "results" / "fs1_dev.csv").set_index("H")
    fp = {}
    for k in ("cond", "tail", "cond20"):
        ok = PL[PL[k] != "資料不足"]
        fp[k] = dict(n=len(ok), passed=int((ok[k] == "收").sum()), rate=(ok[k] == "收").mean(),
                     t90=ok[f"tA_{k}"].abs().quantile(0.9))
    rows = []
    for f in FEATS:
        r = res[f]
        st = r["st"]
        row = dict(H=f, name=r["desc"], fs1=fs1.verdict.get(f, ""), cond=r["cond"], cond20=r["cond20"], tail=r["tailv"], half_life=r["half_life"])
        for nm in names:
            for h in HALVES:
                row[f"{nm}_{h}"] = st[nm][h]["mean"]
                row[f"{nm}_{h}_t"] = st[nm][h]["t"]
                row[f"{nm}_{h}_n"] = st[nm][h]["n"]
        for k in LAGS:
            row[f"ic_lag{k}"] = r["decay"][k]["ic"]
            row[f"tail_lag{k}"] = r["decay"][k]["tail"]
        rows.append(row)
    T = pd.DataFrame(rows)
    T.to_csv(H.D / "results" / "fs3_dev.csv", index=False)

    pd.set_option("display.width", 260)
    print("\n== 安慰劑假陽性率")
    LBL = {"cond": "條件 IC", "tail": "右尾", "cond20": "候選池內前 20（事後描述）"}
    for k, v in fp.items():
        print(f"  {LBL[k]}：{v['passed']}/{v['n']} = {v['rate']*100:.1f}%；A 段 |t| 90 分位 {v['t90']:.2f}")
    n_eval = {k: int((T[k] != "資料不足").sum()) for k in fp}
    for k in fp:
        kept = int((T[k] == "收").sum())
        print(f"  {LBL[k]}收 {kept} / 可評估 {n_eval[k]}，預期假的 {fp[k]['rate']*n_eval[k]:.1f} 個")

    def tbl(df, title):
        print(f"\n== {title}")
        s = df[["H", "name", "fs1", "cond", "cond20", "tail", "cond_ic_A", "cond_ic_A_t", "cond_ic_B", "cond_ic_B_t", "cond_top20_A", "cond_top20_A_t", "cond_top20_B", "cond_top20_B_t",
                "tail_A", "tail_A_t", "tail_B", "tail_B_t", "lift_A", "lift_B", "half_life"]].copy()
        for c in ("cond_top20_A", "cond_top20_B", "tail_A", "tail_B"):
            s[c] = (s[c] * 1200).round(1)
        print(s.round(3).to_string(index=False))

    tbl(T.sort_values("cond_ic_A_t", key=lambda x: -x.abs()), "全部（依條件 IC A 段 |t| 排序；條件前 20、右尾為年化 %）")
    tbl(T[(T.fs1 == "收") & (T.cond != "收")], "全池收、條件不收")
    tbl(T[(T.fs1 != "收") & (T.cond == "收")], "全池不收、條件收")
    tbl(T[(T.fs1 == "收") & (T["tail"] != "收")], "全池收、右尾不收")
    print("\n== 衰減（全池 IC，k 月後那一窗；整段 dev）")
    d = T[["H", "name", "fs1"] + [f"ic_lag{k}" for k in LAGS] + ["half_life"]].sort_values("half_life")
    print(d.round(3).to_string(index=False))
    with open(H.LOG, "a") as fh:
        fh.write(json.dumps(dict(ts=time.strftime("%Y-%m-%d %H:%M:%S"), kind="feature_screen", period="dev（A 2010-14 / B 2015-19）",
                                 strategy="FS3 條件／右尾／衰減重篩", note="第三十九批 事前登記；收＝A |t|≥1 且 B 同號",
                                 placebo=fp, kept_cond=T.H[T.cond == "收"].tolist(), kept_tail=T.H[T["tail"] == "收"].tolist(), kept_cond20_posthoc=T.H[T.cond20 == "收"].tolist(),
                                 half_life=dict(zip(T.H, T.half_life))), ensure_ascii=False,
                            default=lambda v: round(float(v), 5) if isinstance(v, (float, np.floating)) else str(v)) + "\n")


if __name__ == "__main__":
    main()
