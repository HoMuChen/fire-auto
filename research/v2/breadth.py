"""H107：市場寬度背離 — 跌離 52 週高點 ≥10%/20% 的家數比例 vs 大盤創高。
登記見 HYPOTHESES.md 第二十一批。
用法：python3 research/v2/breadth.py dev          # 建指標（快取 data/v2/futures/breadth.parquet）+ dev 判定
      python3 research/v2/breadth.py val          # dev 過才跑；門檻沿用 dev
      python3 research/v2/breadth.py build        # 只重建指標
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402

OUT = H.D / "futures" / "breadth.parquet"
LOG = Path(__file__).resolve().parent / "experiments.jsonl"
WIN, NEAR, HORIZONS = 252, 0.97, (20, 60, 120, 250)
THRESH_FILE = H.D / "results" / "h107_thresh.json"


def build(end="2026-12-31"):
    closes = {}
    for s in H.universe():
        df = H.load_stock(s, end)
        if df is not None and len(df) >= WIN:
            closes[s] = df.set_index("date").close
    C = pd.DataFrame(closes).sort_index()
    hi = C.rolling(WIN, min_periods=WIN).max()
    dd = C / hi - 1
    valid = dd.notna()
    n = valid.sum(axis=1)
    out = pd.DataFrame({"n": n,
                        "B10": (dd <= -0.10).sum(axis=1) / n,
                        "B20": (dd <= -0.20).sum(axis=1) / n,
                        "B50": (dd <= -0.50).sum(axis=1) / n,
                        "med_dd": dd.median(axis=1)})
    tx = pd.read_parquet(H.D / "futures" / "taiex.parquet")
    tx = tx[tx.close > 0].assign(date=pd.to_datetime(tx.date)).set_index("date").close
    out["taiex"] = tx.reindex(out.index)
    out["idx_dd"] = out.taiex / out.taiex.rolling(WIN, min_periods=WIN).max() - 1
    tr = pd.read_parquet(H.D / "taiex_tr.parquet")
    tr = tr.assign(date=pd.to_datetime(tr.date)).set_index("date").price
    out["tr"] = tr.reindex(out.index)
    out = out[out.n >= 200]
    out.to_parquet(OUT)
    print(f"breadth {out.index[0].date()} ~ {out.index[-1].date()}  {len(out)} 天，家數中位 {int(out.n.median())}")
    return out


def nw_t(y, x=None, lag=60):
    """y 的均值（x=None）或 y ~ a + b·x 的 b，Newey-West t。"""
    y = np.asarray(y, float)
    if x is None:
        X = np.ones((len(y), 1)); j = 0
    else:
        X = np.column_stack([np.ones(len(y)), np.asarray(x, float)]); j = 1
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    XtX_inv = np.linalg.inv(X.T @ X)
    S = (X * e[:, None]).T @ (X * e[:, None])
    for k in range(1, lag + 1):
        w = 1 - k / (lag + 1)
        G = (X[k:] * e[k:, None]).T @ (X[:-k] * e[:-k, None])
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    return beta[j], beta[j] / np.sqrt(V[j, j])


def fwd(tr, h):
    return tr.shift(-h) / tr - 1


def analyse(B, p0, p1, thresh=None, timing_only=False):
    d = B[(B.index >= p0) & (B.index <= p1)].copy()
    full = B  # 未來報酬要跨期末，用全表 shift
    for h in HORIZONS:
        d[f"f{h}"] = fwd(full.tr, h).reindex(d.index)
    strong = d[d.idx_dd >= NEAR - 1]
    newhigh = d[d.idx_dd >= -1e-9]
    if thresh is None:
        thresh = float(strong.B20.median())
    print(f"\n[{p0}~{p1}] 交易日 {len(d)}，強勢日 {len(strong)}（{len(strong)/len(d)*100:.0f}%），創高日 {len(newhigh)}；"
          f"B20 門檻（dev 強勢日中位數）= {thresh*100:.1f}%")
    print(f"  B10 / B20 全期中位 {d.B10.median()*100:.0f}% / {d.B20.median()*100:.0f}%；強勢日 {strong.B10.median()*100:.0f}% / {strong.B20.median()*100:.0f}%")
    res = {}
    rows = []
    if timing_only:
        print("  （主判定 dev 未過，val 不看；只跑次判定擇時）")
    groups = {} if timing_only else {"全部交易日": d, "強勢日（距高 ≤3%）": strong, "創 252 日新高日": newhigh,
              "強勢且 B20 高（背離）": strong[strong.B20 > thresh], "強勢且 B20 低": strong[strong.B20 <= thresh]}
    for nm, g in groups.items():
        r = {"組": nm, "天": len(g)}
        for h in HORIZONS:
            v = g[f"f{h}"].dropna()
            r[f"f{h}"] = v.mean() * 100 if len(v) else np.nan
            r[f"f{h}<0"] = (v < 0).mean() * 100 if len(v) else np.nan
        rows.append(r)
    if not timing_only:
      T = pd.DataFrame(rows).set_index("組")
      pd.set_option("display.width", 220)
      print("  未來報酬均值（%）與為負的比例（%）")
      print(T.round(1).to_string())
      # 主判定：強勢日內 高半 − 低半，NW t
      print("  主判定：強勢日內，B20 高半 − 低半（加權報酬指數未來報酬差）")
      for h in HORIZONS:
        s = strong.dropna(subset=[f"f{h}"])
        b, t = nw_t(s[f"f{h}"].values, (s.B20 > thresh).astype(float).values, lag=h)
        res[h] = (b, t)
        print(f"    {h:>3} 日：差 {b*100:+.2f}%  NW t {t:+.2f}")
      # 三分位
      q = strong.dropna(subset=["f60"])
      q["tier"] = pd.qcut(q.B20.rank(method="first"), 3, labels=["低", "中", "高"])
      print("  強勢日依 B20 三分位的未來 60 / 120 日均報酬（%）：",
            " | ".join(f"{k} {v.f60.mean()*100:+.1f}/{v.f120.mean()*100:+.1f}" for k, v in q.groupby("tier", observed=True)))
      # 連續版：B20 與未來報酬的斜率（強勢日）
      s = strong.dropna(subset=["f60"])
      b, t = nw_t(s.f60.values, s.B20.values, lag=60)
      print(f"  連續版：強勢日 f60 對 B20 斜率 {b:+.2f}（B20 +10pp → {b*0.1*100:+.2f}%）NW t {t:+.2f}")
    # 次判定：擇時（+ 兩個對照：只看大盤位置、反向組）
    strong_f = (B.idx_dd >= NEAR - 1)
    flags = {"擇時（背離→現金）": (strong_f & (B.B20 > thresh)), "對照：強勢日→現金（不看寬度）": strong_f,
             "對照：強勢且 B20 低→現金（反向）": (strong_f & (B.B20 <= thresh))}
    ret = B.tr.pct_change()
    m = (B.index >= p0) & (B.index <= p1)
    out, yr = {}, {}
    series = {"買進持有": ret[m]}
    for nm, fl in flags.items():
        series[nm] = (ret * (1 - fl.astype(float).shift(1).fillna(0)))[m]
    for nm, r in series.items():
        eq = (1 + r.fillna(0)).cumprod()
        yrs = len(r) / 252
        cagr = eq.iloc[-1] ** (1 / yrs) - 1
        sharpe = r.mean() / r.std() * np.sqrt(252)
        mdd = (eq / eq.cummax() - 1).min()
        out[nm] = dict(CAGR=cagr, Sharpe=sharpe, MDD=mdd)
        yr[nm] = r.groupby(r.index.year).apply(lambda x: (1 + x.fillna(0)).prod() - 1) * 100
        print(f"  次判定 {nm:<22} CAGR {cagr*100:+.1f}%  Sharpe {sharpe:.2f}  MDD {mdd*100:.1f}%")
    Y = pd.DataFrame(yr).round(1)
    Y["擇時−B&H"] = (Y.iloc[:, 1] - Y.iloc[:, 0]).round(1)
    print("  逐年（%）"); print(Y.to_string())
    flag = flags["擇時（背離→現金）"].astype(float)
    print(f"  背離日數 {int(flag[m].sum())}（{flag[m].mean()*100:.0f}% 的交易日），空手期間段數 {int(((flag.diff() == 1) & m).sum())}")
    ev = flag[m]
    starts = ev.index[(ev.diff().fillna(ev) == 1)]
    if len(starts):
        print("  背離開始日（前 12 段）：", ", ".join(x.strftime("%Y-%m-%d") for x in starts[:12]))
    return thresh, res, out


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "dev"
    if cmd == "build" or not OUT.exists():
        B = build()
    else:
        B = pd.read_parquet(OUT)
    if cmd == "build":
        return
    if cmd == "holdout":
        raise SystemExit("holdout 需使用者明確同意後再開")
    p0, p1 = H.PERIODS[cmd]
    thresh = None
    if cmd == "val":
        thresh = json.load(open(THRESH_FILE))["B20_thresh"]
    thresh, res, out = analyse(B, p0, p1, thresh, timing_only="--timing-only" in sys.argv)
    if cmd == "dev":
        THRESH_FILE.parent.mkdir(exist_ok=True)
        json.dump({"B20_thresh": thresh}, open(THRESH_FILE, "w"))
    with open(LOG, "a") as f:
        f.write(json.dumps(dict(ts=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"), kind="breadth", period=cmd, strategy="H107",
                                note="第二十一批 事前登記：寬度背離（B20 高 且 大盤距 52 週高 ≤3%）", B20_thresh=thresh,
                                diff={str(h): {"d": float(b), "t": float(t)} for h, (b, t) in res.items()},
                                timing={k: {a: float(b) for a, b in v.items()} for k, v in out.items()}), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
