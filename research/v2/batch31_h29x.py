"""第三十一批 H132：H29 + 「避開爛股」排除層（FS1 的 7 個特徵）。規則見 HYPOTHESES.md 第三十一批（事前登記）。

排除分數 X＝7 個特徵在 pit 流動池內百分位的平均（反向特徵先取負號），有值的特徵 ≥ 4 個才算。
換股日前一交易日 X 最低 20% 的股票，H29 分數減 2 排到最後，top20 由下一名遞補。
主判定在 val：H132 − H29 的 top20 月差 > 0 且 t ≥ 2。dev 是樣本內，只當描述。

用法：.venv/bin/python research/v2/batch31_h29x.py dev|val
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

POS = ["H13", "H3", "H11"]                 # 彩券效應反向、低波動、盈餘殖利率：分數高＝之後比較好
NEG = ["H35", "H96", "H36", "H85"]         # 隔夜動能、漲停次數、盤中反向、千張大戶增持：分數高＝之後比較差
EXCL_Q, TOPN = 0.2, 20


def _log(rec):
    with open(H.LOG, "a") as f:
        f.write(json.dumps(dict(ts=time.strftime("%Y-%m-%d %H:%M:%S"), **rec), ensure_ascii=False,
                           default=lambda v: round(float(v), 5)) + "\n")


def _pool(P):
    return P["liq"].fillna(False).astype(bool) & (P["nbars"] >= 250)


def xscore(P, feats):
    """pit 流動池內各特徵百分位的平均；有值的特徵數 ≥ ceil(4/7 × 特徵數)（7 個要 4、3 個要 2、4 個要 3）。"""
    pool = _pool(P)
    need = -(-4 * len(feats) // 7)
    tot, cnt = 0, 0
    for f in feats:
        S = Hy.FNS[f][0](P).replace([np.inf, -np.inf], np.nan)
        r = (-S if f in NEG else S).where(pool).rank(axis=1, pct=True)
        tot = tot + r.fillna(0)
        cnt = cnt + r.notna().astype(int)
    return (tot / cnt).where(cnt >= need)


def excl_mask(P, feats):
    return xscore(P, feats).rank(axis=1, pct=True) <= EXCL_Q


def _h29x(P, feats):
    s = Hy.h29_double_high(P)
    return s.mask(excl_mask(P, feats), s - 2)


def h132_h29x(P):
    return _h29x(P, POS + NEG)


def h132p_pos(P):
    return _h29x(P, POS)


def h132n_neg(P):
    return _h29x(P, NEG)


def _cmp(Ra, Rb):
    a, b = Ra["topn"].dropna(), Rb["topn"].dropna()
    ix = a.index.intersection(b.index)
    d = a[ix] - b[ix]
    by = d.groupby(d.index.year).sum()
    return dict(diff_m=d.mean(), diff_ann=d.mean() * 12, t=d.mean() / d.std() * np.sqrt(len(d)) if d.std() > 0 else np.nan,
                yrs_won=f"{int((by > 0).sum())}/{len(by)}", months_diff=int((d.abs() > 1e-12).sum()), n=len(d))


def diag(P, p0, p1):
    """H29 原 top20 有幾檔落在 X 最低 20%、它們次月超額 vs 留下的；pit 池全體 X 最低 20% 的次月超額。"""
    O, C = P["open"], P["close"]
    Cf = C.ffill()
    idx = O.index
    rd = xs.rebalance_dates(idx, pd.Timestamp(p0), pd.Timestamp(p1))
    s29 = Hy.h29_double_high(P).shift(1)
    ex = excl_mask(P, POS + NEG).shift(1)
    elig = P["liq"].shift(1).fillna(False).astype(bool) & (P["nbars"].shift(1) >= 250) & O.notna()
    rows, names_ex, names_kept = [], [], []
    for d, d2 in zip(rd[:-1], rd[1:]):
        e = elig.loc[d]
        s = s29.loc[d][e].dropna()
        if len(s) < 50:
            continue
        k2 = idx.get_loc(d2)
        r = ((O.loc[d2].fillna(Cf.iloc[k2 - 1]) / O.loc[d] - 1)[e]).clip(-0.95, 10)
        bench = r.mean()
        x = ex.loc[d][e].fillna(False).astype(bool)
        top = s.nlargest(TOPN).index
        tx, tk = top[x[top].values], top[~x[top].values]
        names_ex += list(r[tx] - bench)
        names_kept += list(r[tk] - bench)
        rows.append(dict(d=d, n_top_ex=len(tx), pool_ex=r[x[x].index].mean() - bench, pool_keep=r[x[~x].index].mean() - bench,
                         frac_ex=x.mean()))
    D = pd.DataFrame(rows).set_index("d")
    ne, nk = np.array(names_ex), np.array(names_kept)
    pe = D.pool_ex.dropna()
    return dict(months=len(D), n_top_ex_mean=D.n_top_ex.mean(), months_any_ex=int((D.n_top_ex > 0).sum()),
                frac_ex=D.frac_ex.mean(), top_ex_n=len(ne), top_ex_m=ne.mean() if len(ne) else np.nan,
                top_kept_m=nk.mean(), pool_ex_ann=pe.mean() * 12, pool_ex_t=pe.mean() / pe.std() * np.sqrt(len(pe)),
                pool_keep_ann=D.pool_keep.mean() * 12)


def main(period):
    if period not in ("dev", "val"):
        raise PermissionError("只跑 dev / val；holdout 封存中")
    p0, p1 = H.PERIODS[period]
    end = min(pd.Timestamp(p1), pd.Timestamp.today()).strftime("%Y-%m-%d")
    t0 = time.time()
    P = xs.load_panel(end, fundamentals=True)
    xs.add_v16(P)                                   # H85 的 big_chg
    fns = {"H29": Hy.FNS["H29"], "H132": (h132_h29x, "H29+排除X最低20%(7特徵)"),
           "H132p": (h132p_pos, "H29+排除X最低20%(正向3個，描述)"), "H132n": (h132n_neg, "H29+排除X最低20%(反向4個，描述)")}
    res, P = xs.run(fns, period, P=P, note="第三十一批 H29+避開爛股排除層（事前；H132p/H132n 只描述）", topn=TOPN)
    print(f"\n== H132 [{period}] top{TOPN}（年化、%）  [{time.time()-t0:.0f}s]")
    for k, (R, st) in res.items():
        s = st["topn"]
        print(f"  {k:<6} CAGR {s['CAGR']*100:5.1f}  超額 {s['ex_ann']*100:5.1f}  t {s['t']:.2f}  正年 {s['yrs_pos']}"
              f"  MDD {s['MDD']*100:5.1f}  換手 {s['turnover']*100:4.1f}")
    print(f"\n== 對 H29 的月差（扣成本後；年化 %）")
    out = {}
    for k in ("H132", "H132p", "H132n"):
        c = _cmp(res[k][0], res["H29"][0])
        out[k] = c
        print(f"  {k:<6} 月差 {c['diff_m']*100:+.3f}%（年化 {c['diff_ann']*100:+.2f}%）  t {c['t']:+.2f}  贏的年 {c['yrs_won']}"
              f"  持股不同的月 {c['months_diff']}/{c['n']}")
    c = out["H132"]
    if period == "dev":
        v = "可跑 val（dev 樣本內，只是基本檢查）" if c["diff_m"] > 0 else "✗（dev 月差 ≤ 0，不跑 val）"
    else:
        v = "✓" if c["diff_m"] > 0 and c["t"] >= 2 else ("方向對、不足以換" if c["diff_m"] > 0 else "✗")
    print(f"  → H132 [{period}]：{v}")
    g = diag(P, p0, p1)
    print(f"\n== 描述 [{period}]")
    print(f"  H29 原 top20 每月平均 {g['n_top_ex_mean']:.2f} 檔落在 X 最低 20%（{g['months_any_ex']}/{g['months']} 個月至少 1 檔）")
    print(f"  這些被剔除的 top20 股：次月超額 {g['top_ex_m']*100:+.2f}%/月（{g['top_ex_n']} 檔次）；留下的 {g['top_kept_m']*100:+.2f}%/月")
    print(f"  pit 池全體 X 最低 20%（實際比例 {g['frac_ex']*100:.1f}%）：次月超額年化 {g['pool_ex_ann']*100:+.2f}%（t {g['pool_ex_t']:+.2f}）；"
          f"其餘 80% {g['pool_keep_ann']*100:+.2f}%")
    for k in out:
        _log(dict(kind="xs_cmp", period=period, strategy=f"{k} vs H29", note="第三十一批", **out[k],
                  **({"verdict": v} if k == "H132" else {"verdict": "描述，不判定"})))
    _log(dict(kind="diag", period=period, strategy="H132 排除層描述", note="第三十一批", **g))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dev")
