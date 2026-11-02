"""第三十二批 H133~H135：H29 風控變體，主判定日線 Sharpe。規則見 HYPOTHESES.md 第三十二批（事前登記）。

日線淨值照 threshold.py（H33/H34）：H29 排序取前 20、換股日開盤進場、月內權重漂移、空出部位留現金（0 息），
成本＝Σ|Δ目標權重| / 2 × 來回成本（等權時與 threshold.py 相同）。H29 的 holdout 已開封 → 只跑 dev / val。
  H133 排除層：pit 池 H13/H3/H11 百分位平均最低 20% 排到最後（＝batch31 的 H132p），等權
  H134 反波動加權：權重 ∝ 1/σ60（換股日前一天為止 60 日日對數報酬標準差）
  H135 波動目標：曝險 min(1, σ*/σ̂)；σ̂＝H29 基本版前 126 日日報酬標準差，σ*＝之前換股日 σ̂ 的擴張中位數（≥24 個）

用法：.venv/bin/python research/v2/batch32_risk.py dev|val   （val 會先重算 dev 關卡，只對過關者算 val）
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch31_h29x as B31  # noqa: E402
import harness as H  # noqa: E402
import hypotheses as Hy  # noqa: E402
import threshold as Th  # noqa: E402
import xs  # noqa: E402

N, VOL_WIN, VOL_MIN, TV_WIN, TV_MINOBS, TV_MINHIST = 20, 60, 40, 126, 100, 24
DEV0, HIST0 = "2011-01-01", "2008-01-01"
NAMES = {"H29": "H29 原版（等權）", "H133": "H133 排除層（H13/H3/H11 最低 20%）", "H134": "H134 反波動加權",
         "H135": "H135 波動目標（曝險 ≤ 1）"}


def _log(rec):
    with open(H.LOG, "a") as f:
        f.write(json.dumps(dict(ts=time.strftime("%Y-%m-%d %H:%M:%S"), **rec), ensure_ascii=False,
                           default=lambda v: round(float(v), 5)) + "\n")


def sim(P, score, p0, p1, weight=None, expo=None):
    """日線淨值。weight(d, top) → 權重（和為 1）；expo(d) → 總曝險 [0, 1]。回傳淨值與每月 (曝險, 最大單檔佔投入比)。"""
    O, Cf = P["open"], P["close"].ffill()
    elig = P["liq"].shift(1).fillna(False).astype(bool) & (P["nbars"].shift(1) >= 250) & O.notna()
    ss = score.shift(1)
    rd = xs.rebalance_dates(O.index, pd.Timestamp(p0), pd.Timestamp(p1))
    v, vals, prev, info = 1.0, [], pd.Series(dtype=float), []
    for d, d2 in zip(rd[:-1], rd[1:]):
        s = ss.loc[d][elig.loc[d]].dropna()
        top = list(s.sort_values(ascending=False, kind="stable").head(N).index)
        seg = Cf.loc[d:d2].iloc[:-1]
        if not top:
            w = pd.Series(dtype=float)
        else:
            w = weight(d, top) if weight else pd.Series(1.0 / N, index=top)
            w = w * (expo(d) if expo else 1.0)
        v *= 1 - w.sub(prev, fill_value=0).abs().sum() / 2 * xs.COST
        path = v * ((1 - w.sum()) + (seg[top].div(O.loc[d, top]).mul(w, axis=1).sum(axis=1) if top else 0))
        if not isinstance(path, pd.Series):
            path = pd.Series(path, index=seg.index)
        vals.append(path)
        v = path.iloc[-1]
        prev = w
        info.append(dict(d=d, expo=w.sum(), wmax=w.max() / w.sum() if w.sum() > 0 else np.nan))
    s = pd.concat(vals)
    return s[~s.index.duplicated()], pd.DataFrame(info).set_index("d")


def inv_vol(P):
    sig = np.log(P["close"]).diff().rolling(VOL_WIN, min_periods=VOL_MIN).std().shift(1)

    def w(d, top):
        x = sig.loc[d, top]
        x = x.fillna(x.median())
        x = 1 / x
        return x / x.sum()
    return w


def vol_target(P, h29, p1):
    """σ̂：H29 基本版自己的日報酬（換股日前一天為止 126 日）；σ*：之前換股日 σ̂ 的擴張中位數。"""
    base, _ = sim(P, h29, HIST0, p1)
    rb = base.pct_change().dropna()
    rdh = xs.rebalance_dates(P["open"].index, pd.Timestamp(HIST0), pd.Timestamp(p1))
    sig = {}
    for d in rdh:
        x = rb[rb.index < d].tail(TV_WIN)
        sig[d] = x.std() if len(x) >= TV_MINOBS else np.nan
    sig = pd.Series(sig)
    tgt = sig.expanding().median().shift(1).where(sig.notna().cumsum().shift(1) >= TV_MINHIST)

    def e(d):
        if pd.isna(tgt.get(d)) or pd.isna(sig.get(d)):
            return 1.0
        return float(min(1.0, tgt[d] / sig[d]))
    return e, sig, tgt


def jk_z(r, b):
    r, b = r.align(b, join="inner")
    a1, a2 = r.mean() / r.std(), b.mean() / b.std()
    rho, T = np.corrcoef(r, b)[0, 1], len(r)
    return (a1 - a2) / np.sqrt((2 - 2 * rho + 0.5 * (a1 * a1 + a2 * a2 - 2 * a1 * a2 * rho * rho)) / T)


def evaluate(P, p0, p1, which):
    h29 = Hy.h29_double_high(P)
    out = {}
    for k in which:
        if k == "H29":
            s, info = sim(P, h29, p0, p1)
        elif k == "H133":
            s, info = sim(P, B31.h132p_pos(P), p0, p1)
        elif k == "H134":
            s, info = sim(P, h29, p0, p1, weight=inv_vol(P))
        elif k == "H135":
            e, _, _ = vol_target(P, h29, p1)
            s, info = sim(P, h29, p0, p1, expo=e)
        m = Th.metr(s)
        by = s.resample("YE").last().pct_change()
        by.iloc[0] = s.resample("YE").last().iloc[0] / s.iloc[0] - 1
        out[k] = dict(s=s, m=dict(**m, Calmar=m["CAGR"] / abs(m["MDD"]), expo=info.expo.mean(), wmax=info.wmax.mean(),
                                  expo_min=info.expo.min()), years={str(i.year): x for i, x in by.items()})
    base = out["H29"]["s"].pct_change().dropna()
    for k in which:
        if k != "H29":
            out[k]["m"]["JK_z"] = jk_z(out[k]["s"].pct_change().dropna(), base)
    return out


def leak135(P, cut="2016-03-31"):
    """截斷資料到 cut，cut 之前換股日的曝險要和全資料一樣。"""
    h29 = Hy.h29_double_high(P)
    e_full, _, _ = vol_target(P, h29, "2019-12-31")
    Pt = {k: v.loc[:cut] for k, v in P.items()}
    e_part, _, _ = vol_target(Pt, Hy.h29_double_high(Pt), cut)
    rd = xs.rebalance_dates(P["open"].index, pd.Timestamp(DEV0), pd.Timestamp(cut))[:-1]
    return [d for d in rd if abs(e_full(d) - e_part(d)) > 1e-9]


def show(period, res):
    b = res["H29"]["m"]
    print(f"\n== [{period}] 日線（CAGR / MDD / Sharpe / Calmar / JK z / 平均曝險 / 平均最大單檔）")
    for k, r in res.items():
        m = r["m"]
        flag = "  ⚠ CAGR 少 >2pp" if k != "H29" and m["CAGR"] < b["CAGR"] - 0.02 else ""
        z = f"{m['JK_z']:+.2f}" if "JK_z" in m else "  — "
        print(f"  {NAMES[k]:<28} {m['CAGR']*100:+6.1f}% / {m['MDD']*100:6.1f}% / {m['Sharpe']:.2f} / {m['Calmar']:.2f}"
              f" / z {z} / {m['expo']*100:5.1f}% / {m['wmax']*100:4.1f}%{flag}")
    print("  逐年（%）：")
    for k, r in res.items():
        print(f"    {k:<5} " + "  ".join(f"{y} {v*100:+6.1f}" for y, v in r["years"].items()))


def gate_dev(res):
    b = res["H29"]["m"]
    return {k: r["m"]["Sharpe"] > b["Sharpe"] and r["m"]["MDD"] >= b["MDD"] for k, r in res.items() if k != "H29"}


def main(period):
    if period not in ("dev", "val"):
        raise PermissionError("H29 holdout 已開封，變體只跑 dev / val")
    t0 = time.time()
    p1 = H.PERIODS[period][1]
    P = xs.load_panel(p1, fundamentals=True)
    print(f"載入 {time.time()-t0:.0f}s", flush=True)
    chk = Th.run(P, Hy.h29_double_high(P), None, DEV0, H.PERIODS["dev"][1])[0]
    dev = evaluate(P, DEV0, H.PERIODS["dev"][1], ["H29", "H133", "H134", "H135"])
    assert np.allclose(chk.values, dev["H29"]["s"].reindex(chk.index).values), "H29 基本版和 threshold.py 不一致"
    if period == "dev":
        bad133 = xs.leak_check_xs(B31.h132p_pos, P)
        bad135 = leak135(P)
        print(f"前視檢查：H133 {bad133 or '通過'}；H135 {bad135 or '通過'}")
        if bad133 or bad135:
            raise RuntimeError("前視檢查失敗")
    show("dev 2011-19", dev)
    g = gate_dev(dev)
    print("  dev 關卡（Sharpe > H29 且 MDD 不比 H29 差）：" + "、".join(f"{k} {'過' if v else '✗'}" for k, v in g.items()))
    for k, r in dev.items():
        _log(dict(kind="risk", period="dev", strategy=NAMES[k], note="第三十二批 H29 風控變體（日線；dev 從 2011）",
                  **r["m"], years=r["years"], **({"dev_gate": g[k]} if k in g else {})))
    if period == "val":
        which = ["H29"] + [k for k, v in g.items() if v]
        if len(which) == 1:
            print("\n沒有變體通過 dev 關卡 → 不跑 val")
            return
        val = evaluate(P, H.PERIODS["val"][0], p1, which)
        show("val 2020-23", val)
        b = val["H29"]["m"]
        for k, r in val.items():
            m, v = r["m"], None
            if k != "H29":
                better = m["Sharpe"] > b["Sharpe"] and m["MDD"] >= b["MDD"]
                v = "✓" if better and m["JK_z"] >= 2 else ("方向對、不足以換" if m["Sharpe"] > b["Sharpe"] else "✗")
                print(f"  → {k} [val]：{v}" + ("（⚠ CAGR 少 >2pp，由使用者決定）" if m["CAGR"] < b["CAGR"] - 0.02 else ""))
            _log(dict(kind="risk", period="val", strategy=NAMES[k], note="第三十二批 H29 風控變體（日線）",
                      **m, years=r["years"], **({"verdict": v} if v else {})))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dev")
