"""第三十批 特徵庫篩選 FS1：42 個既有全池分數 → dev 兩段（A 2010-14 / B 2015-19）IC 方向檢驗 + 安慰劑 + 去重。

規則見 HYPOTHESES.md 第三十批。只用 dev；不改特徵程式（各特徵在原批次已通過 leak_check_xs）。
收進特徵庫＝A 段 IC 的 |t| ≥ 1、B 段 IC 同號；安慰劑＝把股票代號整個打亂後同樣篩，過關比例＝假陽性率。

用法：python research/v2/feature_screen.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402
import hypotheses as Hy  # noqa: E402
import xs  # noqa: E402

FEATS = ["H1", "H2", "H3", "H4", "H9", "H13", "H35", "H36", "H38", "H39", "H76", "H77", "H78", "H95", "H96", "H97", "H98", "H99",
         "H7", "H10", "H12", "H87", "H5", "H6", "H11", "H22", "H23", "H24", "H25",
         "H26", "H27", "H28", "H80", "H81", "H83", "H88", "H89", "H90", "H84", "H85", "H86", "H92"]
HALVES = {"A": ("2010-01-01", "2014-12-31"), "B": ("2015-01-01", "2019-12-31")}
NPLACEBO, MIN_N, MIN_M, T_A, RHO = 5, 100, 24, 1.0, 0.6


def forward(P):
    """每個換股日 d：pit 池與 d 開盤 → 下個換股日開盤的報酬（同 xs.backtest）。"""
    O, C = P["open"], P["close"]
    Cf = C.ffill()
    idx = O.index
    rd = xs.rebalance_dates(idx, pd.Timestamp(HALVES["A"][0]), pd.Timestamp(HALVES["B"][1]))
    elig = P["liq"].shift(1).fillna(False).astype(bool) & (P["nbars"].shift(1) >= 250) & O.notna()
    R, E = {}, {}
    for d, d2 in zip(rd[:-1], rd[1:]):
        k2 = idx.get_loc(d2)
        R[d] = (O.loc[d2].fillna(Cf.iloc[k2 - 1]) / O.loc[d] - 1).clip(-0.95, 10)
        E[d] = elig.loc[d]
    return rd[:-1], pd.DataFrame(R).T, pd.DataFrame(E).T.astype(bool)


def series(S, dates, R, E):
    """每月 IC、Q5 超額、Q5 換手。S 為日頻分數；換股日 d 用 d 前一交易日的值。"""
    ss = S.shift(1).reindex(dates)
    rows, prev = [], None
    for d in dates:
        s, r = ss.loc[d], R.loc[d]
        m = E.loc[d] & s.notna() & np.isfinite(s) & r.notna()
        if m.sum() < MIN_N:
            rows.append((d, np.nan, np.nan, np.nan))
            prev = None
            continue
        x, y = s[m], r[m]
        ic = np.corrcoef(x.rank().values, y.rank().values)[0, 1]
        top = x.index[x.rank(method="first", pct=True) > 0.8]     # 同 xs.backtest 的 qcut(rank(method="first"))：同分依代號
        q5 = y[top].mean() - y.mean()
        cur = set(top)
        turn = 1 - len(cur & prev) / len(cur) if prev and cur else 1.0
        rows.append((d, ic, q5, turn))
        prev = cur
    return pd.DataFrame(rows, columns=["d", "ic", "q5", "turn"]).set_index("d")


def halves(df):
    out = {}
    for h, (a, b) in HALVES.items():
        x = df.loc[a:b].dropna(subset=["ic"])
        n = len(x)
        out[h] = dict(n=n, ic=x.ic.mean() if n else np.nan,
                      t=x.ic.mean() / x.ic.std() * np.sqrt(n) if n > 2 else np.nan,
                      q5=x.q5.mean() * 12 if n else np.nan,
                      q5net=(x.q5 - x.turn * xs.COST).mean() * 12 if n else np.nan, turn=x.turn.mean() if n else np.nan)
    return out


def verdict(st):
    a, b = st["A"], st["B"]
    if a["n"] < MIN_M or b["n"] < MIN_M:
        return "資料不足"
    return "收" if abs(a["t"]) >= T_A and np.sign(b["ic"]) == np.sign(a["ic"]) else "—"


def placebo(S, k):
    perm = np.random.default_rng(1000 + k).permutation(S.shape[1])
    return pd.DataFrame(S.values[:, perm], index=S.index, columns=S.columns)


def dedup(keep, snaps, E, sign):
    """依 A 段 |t| 由大到小，與現有代表平均橫斷面 Spearman |ρ| ≥ RHO 就併入。回傳 {代表: [成員]} 與相關矩陣。"""
    names = list(keep)
    dates = snaps[names[0]].index
    rho = pd.DataFrame(np.nan, index=names, columns=names)
    for i, a in enumerate(names):
        for b in names[i:]:
            v = []
            for d in dates:
                x, y = snaps[a].loc[d], snaps[b].loc[d]
                m = E.loc[d] & x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
                if m.sum() >= MIN_N:
                    v.append(np.corrcoef(x[m].rank().values, y[m].rank().values)[0, 1])
            rho.loc[a, b] = rho.loc[b, a] = (np.mean(v) if v else np.nan) * sign[a] * sign[b]
    groups = {}
    for f in names:
        rep = next((g for g in groups if abs(rho.loc[f, g]) >= RHO), None)
        groups.setdefault(rep or f, []).append(f)
    return groups, rho


def main():
    t0 = time.time()
    P = xs.load_panel(HALVES["B"][1], fundamentals=True)
    xs.add_chips(P)
    xs.add_fin(P)
    xs.add_v16(P)
    xs.add_v17(P)
    dates, R, E = forward(P)
    sample = [d for d in dates if d <= pd.Timestamp(HALVES["A"][1])][::3]
    print(f"FS1 特徵篩選：{len(FEATS)} 個特徵 × 安慰劑 {NPLACEBO}；換股日 {len(dates)} 個；載入 {time.time()-t0:.0f}s", flush=True)
    res, plc, snaps = {}, [], {}
    for f in FEATS:
        fn, desc = Hy.FNS[f]
        S = fn(P)
        st = halves(series(S, dates, R, E))
        st["verdict"], st["desc"] = verdict(st), desc
        res[f] = st
        for k in range(NPLACEBO):
            ps = halves(series(placebo(S, k), dates, R, E))
            plc.append(dict(f=f, k=k, verdict=verdict(ps), tA=ps["A"]["t"]))
        if st["verdict"] == "收":
            snaps[f] = S.shift(1).reindex(sample)
        a, b = st["A"], st["B"]
        print(f"  {f:<4} {desc:<14} A: IC {a['ic']:+.4f} t {a['t']:+5.2f} ({a['n']:>2}月)  B: IC {b['ic']:+.4f} t {b['t']:+5.2f} ({b['n']:>2}月)"
              f"  → {st['verdict']}   [{time.time()-t0:.0f}s]", flush=True)
    PL = pd.DataFrame(plc)
    ok_pl = PL[PL.verdict != "資料不足"]
    fp = (ok_pl.verdict == "收").mean()
    ok_real = [f for f in FEATS if res[f]["verdict"] != "資料不足"]
    keep = [f for f in FEATS if res[f]["verdict"] == "收"]
    keep.sort(key=lambda f: -abs(res[f]["A"]["t"]))
    sign = {f: np.sign(res[f]["A"]["ic"]) for f in keep}
    print(f"\n可評估 {len(ok_real)} 個、收進 {len(keep)} 個；安慰劑可評估 {len(ok_pl)} 個、過關 {int((ok_pl.verdict == '收').sum())} 個"
          f"＝假陽性率 {fp*100:.1f}% → 預期假的 {fp*len(ok_real):.1f} 個，扣掉後約 {len(keep) - fp*len(ok_real):.1f} 個是真的")
    print(f"安慰劑 A 段 |t| 分布：中位 {ok_pl.tA.abs().median():.2f}、90 分位 {ok_pl.tA.abs().quantile(0.9):.2f}、最大 {ok_pl.tA.abs().max():.2f}")
    groups, rho = dedup(keep, snaps, E, sign) if keep else ({}, None)
    print(f"\n去重（|ρ| ≥ {RHO}）：{len(groups)} 群")
    for rep, mem in groups.items():
        print(f"  {rep} {res[rep]['desc']}（{'正向' if sign[rep] > 0 else '反向'}）← " + "、".join(f"{m} {res[m]['desc']} ρ {rho.loc[rep, m]:+.2f}" for m in mem if m != rep))
    rows = []
    for f in FEATS:
        st = res[f]
        rep = next((g for g, m in groups.items() if f in m), "")
        rows.append(dict(H=f, name=st["desc"], verdict=st["verdict"], group=rep,
                         **{f"{h}_{k}": st[h][k] for h in HALVES for k in ("n", "ic", "t", "q5", "q5net", "turn")}))
    T = pd.DataFrame(rows)
    out = H.D / "results"
    out.mkdir(exist_ok=True)
    T.to_csv(out / "fs1_dev.csv", index=False)
    if rho is not None:
        rho.to_csv(out / "fs1_rho.csv")
    pd.set_option("display.width", 250)
    show = T[["H", "name", "verdict", "group", "A_n", "A_ic", "A_t", "B_n", "B_ic", "B_t", "A_q5", "B_q5", "B_q5net", "B_turn"]].copy()
    for c in ("A_q5", "B_q5", "B_q5net", "B_turn"):
        show[c] = (show[c] * 100).round(1)
    show = show.assign(o=show.verdict.map({"收": 0, "—": 1, "資料不足": 2}), a=-show.A_t.abs()).sort_values(["o", "a"]).drop(columns=["o", "a"])
    print("\n" + show.round(4).to_string(index=False))
    with open(H.LOG, "a") as fh:
        fh.write(json.dumps(dict(ts=time.strftime("%Y-%m-%d %H:%M:%S"), kind="feature_screen", period="dev（A 2010-14 / B 2015-19）",
                                 strategy="FS1 特徵庫篩選", note="第三十批 事前登記；收＝A |t|≥1 且 B 同號",
                                 n_feats=len(FEATS), n_eval=len(ok_real), kept=keep, placebo_n=len(ok_pl), placebo_fp=fp,
                                 groups=groups, features={f: res[f] for f in FEATS}),
                            ensure_ascii=False, default=lambda v: round(float(v), 5)) + "\n")


if __name__ == "__main__":
    main()
