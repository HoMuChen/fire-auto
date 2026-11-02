"""第四十批 FS4：同業營收驚喜擴散（IND_SUR、REL_SUR）。規則見 HYPOTHESES.md 第四十批（事前登記）。

引擎同 FS3（feature_screen.forward；特徵取換股日前一交易日、pit 池內）。只用 dev，只當原料。
安慰劑＝產業標籤在股票間打亂（保留各產業家數），20 次。

用法：.venv/bin/python research/v2/feature_industry.py
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
import feature_screen3 as F3  # noqa: E402
import harness as H  # noqa: E402
import hypotheses as Hy  # noqa: E402
import xs  # noqa: E402

MERGE = {"其他電子類": "其他電子業", "創新版股票": "創新板股票", "綠能環保類": "綠能環保", "數位雲端類": "數位雲端", "金融業": "金融保險"}
NO_PEER = {"其他", "創新板股票"}
MIN_PEERS, NPL = 5, 20


def labels(cols):
    u = pd.read_csv(H.D / "universe.csv", dtype=str).set_index("stock_id").industry
    g = u.reindex(cols).replace(MERGE)
    g = g.where(~g.isin(NO_PEER))
    codes, uniq = pd.factorize(g)          # 缺值 → −1
    return codes


def ind_feats(SUR, E, codes):
    """IND_SUR（同業不含自己的 SUR 平均，≥ MIN_PEERS 家）與 REL_SUR。"""
    V = E & np.isfinite(SUR)
    S0 = np.where(V, SUR, 0.0)
    IND = np.full(SUR.shape, np.nan)
    for g in np.unique(codes[codes >= 0]):
        cols = np.flatnonzero(codes == g)
        s, n = S0[:, cols].sum(axis=1, keepdims=True), V[:, cols].sum(axis=1, keepdims=True)
        own, ov = S0[:, cols], V[:, cols]
        cnt = n - ov
        val = (s - own) / np.where(cnt > 0, cnt, np.nan)
        IND[:, cols] = np.where((cnt >= MIN_PEERS) & E[:, cols], val, np.nan)
    return IND, SUR - IND


def pct(X, M):
    return pd.DataFrame(np.where(M, X, np.nan)).rank(axis=1, pct=True).values


def extra(X, SUR, H2, R, E):
    """每月：全池 IC、增量 FM 斜率（IND 對自己 SUR、H2）、擴散（自己 SUR 低於中位數者中 X 前 10% 的超額）。"""
    T = X.shape[0]
    out = np.full((T, 3), np.nan)
    M = E & np.isfinite(R)
    Y, PS, PX, PH = pct(R, M), pct(SUR, M), pct(X, M), pct(H2, M)
    for t in range(T):
        m = M[t] & np.isfinite(X[t])
        if m.sum() < FS.MIN_N:
            continue
        out[t, 0] = F3.spear(X[t, m], R[t, m])
        mm = M[t]
        A = np.column_stack([np.ones(mm.sum())] + [np.nan_to_num(Z[t, mm], nan=0.5) for Z in (PS, PX, PH)])
        out[t, 1] = np.linalg.lstsq(A, Y[t, mm], rcond=None)[0][2]
        low = m & np.isfinite(PS[t]) & (PS[t] < 0.5)
        if low.sum() >= FS.MIN_N:
            x, r = X[t, low], R[t, low]
            top = rankdata(x, method="ordinal") > 0.9 * len(x)
            out[t, 2] = r[top].mean() - r.mean()
    return out


METRICS = ["cond_ic", "cond_top20", "tail", "lift", "ic", "fm", "diffuse"]


def evaluate(X, SUR, H2, R, E, C, dates):
    mo = np.column_stack([F3.monthly(X, R, E, C), extra(X, SUR, H2, R, E)])
    return {nm: F3.halves(mo[:, i], dates) for i, nm in enumerate(METRICS)}


def main():
    t0 = time.time()
    P = xs.load_panel(FS.HALVES["B"][1], fundamentals=True)
    for add in (xs.add_chips, xs.add_fin, xs.add_v16, xs.add_v17):
        add(P)
    dates, Rdf, Edf = FS.forward(P)
    cols = P["close"].columns
    dates = pd.DatetimeIndex(dates)
    R = Rdf.reindex(index=dates, columns=cols).values
    E = Edf.reindex(index=dates, columns=cols).fillna(False).values.astype(bool)

    def mat(f):
        return Hy.FNS[f][0](P).replace([np.inf, -np.inf], np.nan).shift(1).reindex(index=dates, columns=cols).values

    codes = labels(cols)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        SUR, H2 = mat("H87"), mat("H2")
        C = F3.candidates([H2, SUR, mat("H97")], R, E)
        IND, REL = ind_feats(SUR, E, codes)
        cov = np.isfinite(IND)[E].mean()
        print(f"FS4：{len(dates)} 個換股日；有產業的股票 {(codes >= 0).sum()}/{len(codes)}；池內 IND_SUR 覆蓋 {cov*100:.0f}%；"
              f"與自己 SUR 的平均橫斷面相關 {np.nanmean([F3.spear(SUR[t, m], IND[t, m]) for t in range(len(dates)) if (m := E[t] & np.isfinite(SUR[t]) & np.isfinite(IND[t])).sum() > 100]):.2f}"
              f"（載入 {time.time()-t0:.0f}s）", flush=True)
        res = {k: evaluate(X, SUR, H2, R, E, C, dates) for k, X in (("IND_SUR", IND), ("REL_SUR", REL), ("SUR", SUR))}
        dec = {k: F3.decay(X, R, E) for k, X in (("IND_SUR", IND), ("REL_SUR", REL))}
        plc = []
        for k in range(NPL):
            pc = np.random.default_rng(2000 + k).permutation(codes)
            pi, pr = ind_feats(SUR, E, pc)
            for nm, X in (("IND_SUR", pi), ("REL_SUR", pr)):
                st = evaluate(X, SUR, H2, R, E, C, dates)
                plc.append(dict(f=nm, k=k, **{m: F3.verdict(st[m]) for m in METRICS}, **{f"{m}_tA": st[m]["A"]["t"] for m in METRICS}))
            print(f"  安慰劑 {k+1}/{NPL} [{time.time()-t0:.0f}s]", flush=True)
    PL = pd.DataFrame(plc)

    LBL = {"ic": "全池 IC", "tail": "右尾前 10% 超額（年化）", "lift": "大贏家命中倍數", "cond_ic": "條件 IC（H136 前 100）",
           "cond_top20": "候選池內前 20（年化）", "fm": "增量 FM 斜率（控制自己 SUR、H2）", "diffuse": "擴散：自己沒驚喜者中前 10%（年化）"}
    scale = {"tail": 1200, "cond_top20": 1200, "diffuse": 1200}
    print("\n== 結果（A 2010-14 / B 2015-19；收＝A |t| ≥ 1 且 B 同號；安慰劑＝產業標籤打亂 20 次的收率）")
    out = {}
    for m in ("ic", "fm", "tail", "lift", "diffuse", "cond_ic", "cond_top20"):
        print(f"  {LBL[m]}")
        for k in ("IND_SUR", "REL_SUR", "SUR"):
            a, b = res[k][m]["A"], res[k][m]["B"]
            sc = scale.get(m, 1)
            v = F3.verdict(res[k][m])
            fp = "" if k == "SUR" else f"  安慰劑收率 {(PL[PL.f == k][m] == '收').mean()*100:4.0f}%、A |t| 90 分位 {PL[PL.f == k][f'{m}_tA'].abs().quantile(0.9):.2f}"
            print(f"    {k:<8} A {a['mean']*sc:+8.3f}（t {a['t']:+5.2f}）  B {b['mean']*sc:+8.3f}（t {b['t']:+5.2f}） → {v}{fp}")
            out.setdefault(k, {})[m] = dict(A=a, B=b, verdict=v)
    print("\n== 衰減（全池 IC，k 月後那一窗）")
    for k, (d, hl) in dec.items():
        print(f"  {k:<8} " + "  ".join(f"k={j} {d[j]['ic']:+.3f}" for j in F3.LAGS) + f"  半衰期 {hl:.1f} 月")
    fm = out["IND_SUR"]["fm"]
    go = fm["A"]["t"] >= 2 and fm["B"]["t"] >= 2
    print(f"\n增量判定（A、B 的 t 都 ≥ 2 才值得登記策略）：IND_SUR FM t A {fm['A']['t']:+.2f} / B {fm['B']['t']:+.2f} → {'值得' if go else '不夠'}")
    with open(H.LOG, "a") as fh:
        fh.write(json.dumps(dict(ts=time.strftime("%Y-%m-%d %H:%M:%S"), kind="feature_screen", period="dev（A 2010-14 / B 2015-19）",
                                 strategy="FS4 同業營收驚喜擴散", note="第四十批 事前登記", results=out,
                                 decay={k: dict(half_life=hl, ic={j: d[j]["ic"] for j in F3.LAGS}) for k, (d, hl) in dec.items()},
                                 placebo={k: {m: float((PL[PL.f == k][m] == "收").mean()) for m in METRICS} for k in ("IND_SUR", "REL_SUR")},
                                 incremental_go=go), ensure_ascii=False,
                            default=lambda v: round(float(v), 5) if isinstance(v, (float, np.floating)) else str(v)) + "\n")


if __name__ == "__main__":
    main()
