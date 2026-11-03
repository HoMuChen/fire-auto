"""第四十一批 FL1：可重複使用的特徵庫。規則見 HYPOTHESES.md 第四十一批（事前登記）；第四十二批 FS5 加入 16 個新特徵；
第四十三批 FL2 加新評估軌道（B＋V、週、60 日、風險目標，見 feature_tracks.py），月報酬的結果不變；
第四十四批 FS6 加入 17 個現有資料可做的特徵（feature_fs6.py）；第四十五批 FS7 加入 7 個新資料特徵（feature_fs7.py：融資維持率、外資持股、當沖）；
第四十六批 FS8 加入 20 個學術因子（feature_fs8.py，候選來自 FACTOR_MAP.md）；第四十七批 FS9 再加 18 個（feature_fs9.py，含組合與產業內估值）。

68 個特徵（FS1 的 42 個候選 + FS4 的 IND_SUR、REL_SUR + FS5 的 24 個，見 feature_tu.py），依用途（排序、多空、挑贏家、避雷）評估：
方向用 A 段 IC 決定 → dev（A 2010-14、B 2015-19）給標籤 → val（2020-23）確認。holdout 不碰。
產出 research/v2/FEATURES.md（只放統計量）與 data/v2/results/feature_library.csv（月報酬）、feature_library_tracks.csv（新軌道）。

用法：.venv/bin/python research/v2/feature_library.py
      .venv/bin/python research/v2/feature_library.py --no-sbl   # 預覽：不含借券特徵、不寫 FEATURES.md
"""
import json
import sys
import time
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
import feature_fs6 as F6  # noqa: E402
import feature_fs7 as F7  # noqa: E402
import feature_fs8 as F8  # noqa: E402
import feature_fs9 as F9  # noqa: E402
import feature_industry as FI  # noqa: E402
import feature_screen as FS  # noqa: E402
import feature_screen3 as F3  # noqa: E402
import feature_tracks as FT  # noqa: E402
import feature_tu as TU  # noqa: E402
import harness as H  # noqa: E402
import hypotheses as Hy  # noqa: E402
import xs  # noqa: E402

PER = {"A": ("2010-01-01", "2014-12-31"), "B": ("2015-01-01", "2019-12-31"), "V": ("2020-01-01", "2023-12-31")}
END = PER["V"][1]
PREVIEW = "--no-sbl" in sys.argv
FEATS = [f for f in FS.FEATS + ["IND_SUR", "REL_SUR"] + TU.NAMES + F6.NAMES + F7.NAMES + F8.NAMES + F9.NAMES if not (PREVIEW and f in TU.SBL)]
NPL, MIN_N, MIN_M, RHO = 5, 100, 24, 0.6
PEER, COMBO = {**TU.PEER, **F9.PEER}, {**TU.COMBO, **F9.COMBO}   # 換股日池內算：同業內百分位、組合
USES = {"ic": "排序", "ls": "多空", "good": "挑贏家", "bad": "避雷"}
SIGN = {"ic": 1, "ls": 1, "good": 1, "bad": -1}            # 標籤要的方向（避雷＝壞端為負）
COLS = ["ic", "ls", "good", "bad", "wl_good", "ll_bad", "wl_bad", "ll_good", "turn"]
CAT = {**{f: "價格" for f in ("H1", "H2", "H3", "H4", "H9", "H13", "H35", "H36", "H38", "H39", "H76", "H77", "H78", "H84", "H92",
                               "H95", "H96", "H97", "H98", "H99")},
       **{f: "營收" for f in ("H7", "H10", "H12", "H87", "IND_SUR", "REL_SUR")},
       **{f: "籌碼" for f in ("H22", "H23", "H24", "H25")}, **{f: "集保" for f in ("H85", "H86")},
       **{f: "估值" for f in ("H5", "H6", "H11")},
       **{f: "財報" for f in ("H26", "H27", "H28", "H80", "H81", "H83", "H88", "H89", "H90")}, **TU.CAT, **F6.CAT, **F7.CAT, **F8.CAT, **F9.CAT}
FREQ = {"價格": "每日", "籌碼": "每日", "估值": "每日（分母季／年更新）", "營收": "每月（次月 11 日）", "集保": "每週", "財報": "每季",
        "借券": "每日（21:00 公布，隔日可用）", "組合": "依成分（最慢的成分決定）", "股利": "每年（除息日；月換股日更新）"}
NOTES = {**F9.NOTES, "IND_SUR": "FS4 產業標籤打亂的安慰劑下 IC、增量都在雜訊範圍", "REL_SUR": "≈ H87 減雜訊，見高度相關欄"}
DESC = {"IND_SUR": "同業營收驚喜（同產業其他股 SUR 平均）", "REL_SUR": "相對營收驚喜（自己 SUR − 同業）",
        **{f: v[1] for f, v in TU.FNS.items()}, **{f: v[1] for f, v in TU.PEER.items()}, **{f: v[3] for f, v in TU.COMBO.items()},
        **{f: v[1] for f, v in F6.FNS.items()}, **{f: v[1] for f, v in F7.FNS.items()}, **{f: v[1] for f, v in F8.FNS.items()},
        **{f: v[1] for f, v in F9.FNS.items()}, **{f: v[3] for f, v in F9.COMBO.items()}, **{f: v[1] for f, v in F9.PEER.items()}}


def forward(P, p0, p1):
    """同 feature_screen.forward，期間可指定。"""
    O, C = P["open"], P["close"]
    Cf = C.ffill()
    idx = O.index
    rd = xs.rebalance_dates(idx, pd.Timestamp(p0), pd.Timestamp(p1))
    elig = P["liq"].shift(1).fillna(False).astype(bool) & (P["nbars"].shift(1) >= 250) & O.notna()
    R, E = {}, {}
    for d, d2 in zip(rd[:-1], rd[1:]):
        k2 = idx.get_loc(d2)
        R[d] = (O.loc[d2].fillna(Cf.iloc[k2 - 1]) / O.loc[d] - 1).clip(-0.95, 10)
        E[d] = elig.loc[d]
    return pd.DatetimeIndex(rd[:-1]), pd.DataFrame(R).T, pd.DataFrame(E).T.astype(bool)


def monthly(X, R, E, full=True):
    """X 已定好方向（大＝好端）。每月：IC、多空、好端 10%、壞端 10%、命中倍數、好端 20% 換手。"""
    T = X.shape[0]
    out = np.full((T, len(COLS) if full else 4), np.nan)
    prev = None
    for t in range(T):
        m = E[t] & np.isfinite(X[t]) & np.isfinite(R[t])
        n = int(m.sum())
        if n < MIN_N:
            prev = None
            continue
        x, r = X[t, m], R[t, m]
        rk = rankdata(x, method="ordinal")
        g20, b20, g10, b10 = rk > 0.8 * n, rk <= 0.2 * n, rk > 0.9 * n, rk <= 0.1 * n
        mu = r.mean()
        v = [F3.spear(x, r), r[g20].mean() - r[b20].mean(), r[g10].mean() - mu, r[b10].mean() - mu]
        if full:
            rr = rankdata(r, method="ordinal")
            win, lose = rr > 0.9 * n, rr <= 0.1 * n

            def lift(a, b):
                return (a & b).sum() / b.sum() / (a.sum() / n)

            cur = set(np.flatnonzero(m)[g20])
            turn = 1 - len(cur & prev) / len(cur) if prev else np.nan
            prev = cur
            v += [lift(g10, win), lift(b10, lose), lift(b10, win), lift(g10, lose), turn]
        out[t] = v
    return out


def pstats(v, dates):
    res = {}
    for p, (a, b) in PER.items():
        k = (dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b)) & np.isfinite(v)
        x = v[k]
        n = len(x)
        res[p] = dict(n=n, mean=x.mean() if n else np.nan, t=x.mean() / x.std(ddof=1) * np.sqrt(n) if n > 2 else np.nan)
    return res


def orient(X, R, E, dates):
    ic = monthly(X, R, E, full=False)[:, 0]
    st = pstats(ic, dates)
    base = "A" if st["A"]["n"] >= MIN_M else "B"           # A 段不足的特徵（資料不足）只為描述借 B 段定方向
    return (1.0 if st[base]["mean"] >= 0 else -1.0), base


def tags(st):
    """回傳 {用途: dict(dev, strong, val)}；dev＝A |t|≥1 且 B 同向；val＝確認 / 強確認 / 反向。"""
    out = {}
    for u, s in SIGN.items():
        a, b, v = st[u]["A"], st[u]["B"], st[u]["V"]
        if a["n"] < MIN_M or b["n"] < MIN_M:
            out[u] = dict(dev=False, strong=False, val=None)
            continue
        dev = s * a["t"] >= 1 and s * b["mean"] > 0
        strong = dev and s * a["t"] >= 2 and s * b["t"] >= 2
        val = None
        if dev and v["n"] >= MIN_M:
            val = "強確認" if s * v["t"] >= 2 else ("確認" if s * v["mean"] > 0 else "反向")
        out[u] = dict(dev=dev, strong=strong, val=val)
    return out


def status(st, tg):
    if st["ic"]["A"]["n"] < MIN_M or st["ic"]["B"]["n"] < MIN_M:
        return "資料不足"
    if not any(t["dev"] for t in tg.values()):
        return "未收"
    return "收" if any(t["val"] in ("確認", "強確認") for t in tg.values()) else "待確認"


def corr_pairs(Xo, E, dates):
    """dev 每 3 個月抽一次，兩兩重新排名後的平均橫斷面 Spearman（已定方向）。"""
    dev = np.flatnonzero(dates <= pd.Timestamp(PER["B"][1]))[::3]
    names = list(Xo)
    rho = pd.DataFrame(np.nan, index=names, columns=names)
    for i, a in enumerate(names):
        for b in names[i:]:
            v = []
            for t in dev:
                m = E[t] & np.isfinite(Xo[a][t]) & np.isfinite(Xo[b][t])
                if m.sum() >= MIN_N:
                    v.append(F3.spear(Xo[a][t, m], Xo[b][t, m]))
            rho.loc[a, b] = rho.loc[b, a] = np.mean(v) if v else np.nan
    return rho


def ties(X, E):
    """dev 每月池內「壞端／好端 10% 邊界那個值」的同分比例（平均）。≥ 10% 代表該端有一部分靠代號排序挑出。"""
    lo, hi = [], []
    for t in range(X.shape[0]):
        m = E[t] & np.isfinite(X[t])
        n = int(m.sum())
        if n >= MIN_N:
            x = np.sort(X[t, m])
            k = int(0.1 * n)
            lo.append((x == x[k]).mean())
            hi.append((x == x[n - 1 - k]).mean())
    return (np.mean(lo), np.mean(hi)) if lo else (0.0, 0.0)


def cell(t):
    if not t["dev"]:
        return "·"
    d = "◎" if t["strong"] else "○"
    return d + {"強確認": "✓✓", "確認": "✓", "反向": "✗", None: "？"}[t["val"]]


def write_md(rows, fp, path, TR=None, PLR=None):
    L = ["# 特徵庫（FL1，程式產生，勿手改）", "",
         "由 `research/v2/feature_library.py` 產生（第四十一批建立、第四十二批 FS5 加入涂敏峰清單的特徵與借券、第四十三批 FL2 加週／60 日／風險軌道與 B＋V 評估、第四十四批 FS6 加入現有資料可做的特徵、第四十五批 FS7 加入融資維持率／外資持股／當沖、第四十六、四十七批 FS8、FS9 加入學術因子，規則見 `HYPOTHESES.md`）。"
         "只放統計量，不放 FinMind 原始數字。",
         "原則：特徵只看自己夠不夠好，依用途貼標籤，留著以後重複使用；**不以能否改進 H136 或適不適合每月 20 檔為門檻**（H136 增量只是描述欄位）。", "",
         "## 怎麼用", "",
         "```python",
         "P = xs.load_panel(end, fundamentals=True)",
         "for add in (xs.add_chips, xs.add_fin, xs.add_v16, xs.add_v17):",
         "    add(P)",
         "S = hypotheses.FNS[\"H87\"][0](P)      # 日頻分數（日期 × 股票）；用前一交易日的值避免前視",
         "# IND_SUR / REL_SUR：feature_industry.ind_feats(SUR 矩陣, 池, feature_industry.labels(欄位))",
         "feature_tu.add_tu(P)                    # FS5：借券、千張大戶比例、負債比、預估 EPS、大盤日報酬",
         "S = feature_tu.FNS[\"SBL_MR\"][0](P)",
         "xs.add_v18(P); feature_fs6.add_fs6(P)  # FS6：資本支出、應收、利息保障、淨發行、合約負債、融券（H100 需 add_v18）",
         "S = feature_fs6.FNS[\"BETA\"][0](P)",
         "feature_fs7.add_fs7(P)                  # FS7：融資維持率、外資持股、當沖（data/v2/maint、fhold、dt）",
         "feature_fs8.add_fs8(P)                  # FS8：學術因子（NOA、Z／O 分數、EBITDA／EV…；需先 add_fs6）",
         "feature_fs9.add_fs9(P)                  # FS9：學術因子第二輪；QMJ、MISP_* 是組合（同 THREE_LOW），PEER_EBITDA_EV、PEER_FCF_ME 用 peer_rank",
         "# PEER_EY / PEER_BP：feature_tu.peer_rank(H11 或 H5 的換股日矩陣, 池, feature_industry.labels(欄位))",
         "```", "",
         "方向欄是「大＝好端」要乘的號（−1 代表原始值越小越好）。",
         "**期間**：A 2010-14（實際從 2011 起）、B 2015-19（以上是 dev）、V 2020-23（val）。holdout 2024 年以後沒有用。", "",
         "**用途標籤**（`·` 是 dev 沒有標籤）：",
         "- dev：`○` 是 A |t| ≥ 1 且 B 同號，`◎` 是兩段 |t| 都 ≥ 2。",
         "- val：`✓` 確認，`✓✓` 強確認（|t| ≥ 2），`✗` 方向相反，`？` val 不能評估。", "",
         "**用途的定義**：",
         "- 排序：全池 Spearman IC。",
         "- 多空：好端 20% 減壞端 20%（毛，未扣成本）。",
         "- 挑贏家：好端 10% 減池平均。",
         "- 避雷：壞端 10% 減池平均，要顯著為負。",
         "表中的報酬都是年化百分比。", "",
         "**狀態**：收＝至少一個用途 val 確認；待確認＝有 dev 標籤但 val 沒確認；未收＝dev 沒有標籤；資料不足＝A 或 B 段少於 24 個月。"
         "資料不足的特徵改用 B＋V 評估（見文末「新評估軌道」），狀態欄寫成「資料不足 → B＋V：…」。", "",
         "## 安慰劑（把股票代號打亂，每個特徵 5 次）", "",
         "| 用途 | dev 標籤（○ 以上）假陽性率 | 有 dev 標籤時 val 也同號的比例 | `◎✓✓` 假陽性率 |", "|---|---|---|---|"]
    for u, nm in USES.items():
        L.append(f"| {nm} | {fp[u]['dev']*100:.0f}% | {fp[u]['val']*100:.0f}% | {fp[u]['top']*100:.1f}% |")
    L += ["", "注意：打亂代號後，每檔股票拿到的是一整條固定序列。有些股票長期報酬本來就較高，所以假特徵在 dev 與 val 之間也會延續同一方向，"
          "val 同號的比例會超過一半（排序最明顯）。換句話說，「val 確認」只能擋掉一部分假陽性。",
          f"**`◎✓✓`（dev 兩段 |t| ≥ 2，且 val 強確認）的假陽性率低很多，是最可靠的標籤**。整體「收」的安慰劑比例是 {fp['status']*100:.0f}%，所以「收」裡面仍可能有假的。", ""]
    order = {"收": 0, "待確認": 1, "未收": 2, "資料不足": 3}
    rows = sorted(rows, key=lambda r: (order[r["status"]], -r["n_strong"], r["H"]))
    bv = {} if TR is None else TR[TR.track == "M"].set_index("H").BV_status.to_dict()

    def stt(r):
        return r["status"] + (f" → B＋V：{bv[r['H']]}" if r["status"] == "資料不足" and r["H"] in bv else "")

    L += ["## 總表", "", "| 特徵 | 名稱 | 類別／更新頻率 | 方向 | 排序 | 多空 | 挑贏家 | 避雷 | 狀態 | 註記 | 半衰期（月） | 好端換手 | 起始 | 高度相關（\\|ρ\\| ≥ 0.6） | H136 增量（描述） |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        hl = "> 12" if not np.isfinite(r["half_life"]) else f"{r['half_life']:.1f}"
        L.append(f"| {r['H']} | {r['name']} | {r['cat']}／{FREQ[r['cat']]} | {'+' if r['sign'] > 0 else '−'} | "
                 + " | ".join(r[f"tag_{u}"] for u in USES) + f" | {stt(r)} | {r['note']} | {hl} | {r['turn']*100:.0f}% | {r['start']} | {r['related']} | {r['h136']} |")
    L += ["", "## 指標（A / B / V；括號為 t）", "",
          "| 特徵 | 排序 IC | 多空（年化） | 好端 10%（年化） | 壞端 10%（年化） | 大贏家命中（好端） | 大輸家命中（壞端） | IC 同向年數（dev） |",
          "|---|---|---|---|---|---|---|---|"]

    def f3(r, u, sc, d):
        return " / ".join(f"{r[f'{u}_{p}']*sc:+.{d}f}（{r[f'{u}_{p}_t']:+.1f}）" if np.isfinite(r[f"{u}_{p}"]) else "—" for p in PER)

    for r in rows:
        L.append(f"| {r['H']} | {f3(r, 'ic', 1, 3)} | {f3(r, 'ls', 1200, 1)} | {f3(r, 'good', 1200, 1)} | {f3(r, 'bad', 1200, 1)} | "
                 + " / ".join(f"{r[f'wl_good_{p}']:.2f}" if np.isfinite(r[f'wl_good_{p}']) else "—" for p in PER) + " | "
                 + " / ".join(f"{r[f'll_bad_{p}']:.2f}" if np.isfinite(r[f'll_bad_{p}']) else "—" for p in PER) + f" | {r['yrs_ic']} |")
    if TR is not None:
        L += FT.md(TR, PLR, {r["H"]: r for r in rows})
    path.write_text("\n".join(L) + "\n")


ADDS = (xs.add_chips, xs.add_fin, xs.add_v16, xs.add_v17, xs.add_v18, TU.add_tu, F6.add_fs6, F7.add_fs7, F8.add_fs8, F9.add_fs9)


def load(end=END):
    """載入面板與全部特徵所需的衍生值（特徵庫與 strategy_lab.py 共用）。"""
    P = xs.load_panel(end, fundamentals=True)
    for add in ADDS:
        add(P)
    return P


def feature_fn(f):
    for m in (TU.FNS, F6.FNS, F7.FNS, F8.FNS, F9.FNS, F9.AUX):
        if f in m:
            return m[f][0]
    return Hy.FNS[f][0]


class Source:
    """特徵來源：日頻分數（已 shift 1＝換股日用前一交易日的值）、換股日矩陣，以及 PEER／COMBO／IND 這些要在池內算的特徵。"""

    def __init__(self, P):
        self.P, self.cols = P, P["close"].columns
        self.codes = FI.labels(self.cols)
        self.daily = lru_cache(maxsize=32)(self._daily)

    def _daily(self, f):
        return feature_fn(f)(self.P).replace([np.inf, -np.inf], np.nan).shift(1)

    def mat(self, f, d):
        return self.daily(f).reindex(index=d, columns=self.cols).values

    def build(self, f, d, Eg, ind=None):
        """換股日 d、池 Eg 下的特徵矩陣（日期 × 股）。ind 可傳入已算好的 IND_SUR／REL_SUR。"""
        if f in ("IND_SUR", "REL_SUR"):
            if ind is None:
                ind = dict(zip(("IND_SUR", "REL_SUR"), FI.ind_feats(self.mat("H87", d), Eg, self.codes)))
            return ind[f]
        if f in PEER:
            return TU.peer_rank(self.mat(PEER[f][0], d), Eg, self.codes)
        if f in COMBO:
            sp = COMBO[f]
            return TU.combo({g: self.mat(g, d) for g, _ in sp[0]} | ({sp[2]: self.mat(sp[2], d)} if sp[2] else {}), Eg, sp)
        return self.mat(f, d)


def main():
    t0 = time.time()
    P = load()
    dates, Rdf, Edf = forward(P, PER["A"][0], END)
    cols = P["close"].columns
    R = Rdf.reindex(index=dates, columns=cols).values
    E = Edf.reindex(index=dates, columns=cols).fillna(False).values.astype(bool)
    devm = dates <= pd.Timestamp(PER["B"][1])
    wdates, Rw, Ew = FT.weekly(P, cols)
    Rq, Eq = FT.q60(P, dates, cols)
    TG = FT.risk_targets(P, dates, cols, E)

    src = Source(P)
    build, codes = src.build, src.codes

    def mat(f, d=dates):
        return src.mat(f, d)

    fs3 = pd.read_csv(H.D / "results" / "fs3_dev.csv").set_index("H")
    rows, plc, Xo, trk, ptrk = [], [], {}, [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        IND, REL = FI.ind_feats(mat("H87"), E, codes)
        INDw = dict(zip(("IND_SUR", "REL_SUR"), FI.ind_feats(mat("H87", wdates), Ew, codes)))
        print(f"FL1：{len(FEATS)} 個特徵、{len(dates)} 個換股日（{dates[0].date()} ~ {dates[-1].date()}）；載入 {time.time()-t0:.0f}s", flush=True)
        for f in FEATS:
            if f in ("IND_SUR", "REL_SUR"):
                X = {"IND_SUR": IND, "REL_SUR": REL}[f]
            elif f in PEER:
                X = TU.peer_rank(mat(PEER[f][0]), E, codes)
            elif f in COMBO:
                sp = COMBO[f]
                X = TU.combo({g: mat(g) for g, _ in sp[0]} | ({sp[2]: mat(sp[2])} if sp[2] else {}), E, sp)
            else:
                X = mat(f)
            s, base = orient(X, R, E, dates)
            Xs = s * X
            Xo[f] = Xs
            mo = monthly(Xs, R, E)
            st = {c: pstats(mo[:, i], dates) for i, c in enumerate(COLS)}
            tg = tags(st)
            stt = status(st, tg)
            dc, hl = F3.decay(Xs[devm], R[devm], E[devm])
            valid = (E & np.isfinite(X)).sum(axis=1)
            start = dates[np.argmax(valid >= MIN_N)].strftime("%Y-%m") if (valid >= MIN_N).any() else "—"
            ic = mo[:, 0]
            yrs = pd.Series(ic[devm], index=dates[devm]).dropna().groupby(lambda d: d.year).mean()
            lot = [e for e, (w, lo) in (("好端", ("wl_good", "ll_good")), ("壞端", ("wl_bad", "ll_bad")))
                   if all(st[w][p]["mean"] > 1.3 and st[lo][p]["mean"] > 1.3 for p in "AB")]
            note = []
            if lot:
                note.append("彩券（" + "、".join(lot) + "）")
            if f in NOTES:
                note.append(NOTES[f])
            if f in TU.HIS:
                note.append("涂敏峰說的方向" + ("相同" if s == TU.HIS[f] else "相反"))
            for end, sh in zip(("壞端", "好端"), ties(Xs[devm], E[devm])):
                if sh >= 0.1:
                    note.append(f"{end}同分 {sh*100:.0f}%（10% 邊界落在同分區，部分靠代號排序）")
            if base == "B":
                note.append("方向借 B 段")
            if f in fs3.index:
                h136 = f"{ {'—': '不收'}.get(fs3.cond20[f], fs3.cond20[f]) }（B {fs3.cond_top20_B[f]*1200:+.1f}%）"
            elif f in ("IND_SUR", "REL_SUR"):
                h136 = "FS4：無增量"
            else:
                h136 = "未評"
            row = dict(H=f, name=DESC.get(f, Hy.FNS[f][1] if f in Hy.FNS else f), cat=CAT[f], sign=s, status=stt,
                       n_strong=sum(t["strong"] and t["val"] == "強確認" for t in tg.values()),
                       note="、".join(note), half_life=hl, turn=np.nanmean(mo[devm, COLS.index("turn")]), start=start,
                       yrs_ic=f"{int((yrs > 0).sum())}/{len(yrs)}", h136=h136,
                       **{f"tag_{u}": cell(tg[u]) for u in USES}, **{f"val_{u}": tg[u]["val"] for u in USES},
                       **{f"dev_{u}": tg[u]["dev"] for u in USES})
            for c in COLS[:-1]:
                for p in PER:
                    row[f"{c}_{p}"] = st[c][p]["mean"]
                    row[f"{c}_{p}_t"] = st[c][p]["t"]
                    row[f"{c}_{p}_n"] = st[c][p]["n"]
            rows.append(row)
            # 第四十三批 FL2 新軌道（月的 AB 沿用上面的結果；A 段不足的正式採 BV）
            Xw = build(f, wdates, Ew, INDw)
            a_ok = st["ic"]["A"]["n"] >= MIN_M
            tr = {"M": FT.eval_ret(X, R, E, dates, pre=(s, mo[:, :4])), "W": FT.eval_ret(Xw, Rw, Ew, wdates, turn=True),
                  "Q": FT.eval_ret(X, Rq, Eq, dates, lag=FT.Q_LAG, turn=True), "R": FT.eval_risk(X, TG, dates)}
            wv = (np.isfinite(Xw) & Ew).sum(axis=1) >= MIN_N
            w_ok = wv[np.argmax(wv):].mean() >= 0.5 if wv.any() else True      # 從第一個有值的週起算（晚起始的特徵不算缺）
            if not w_ok:
                tr["W"] = FT.not_applicable(tr["W"])
            assert [FT.cell(tr["M"]["AB"]["tg"][u], "AB") for u in USES] == [row[f"tag_{u}"] for u in USES]
            for k_, v_ in tr.items():
                trk.append(FT.flat(f, k_, v_, FT.RISK if k_ == "R" else FT.RET) | dict(a_ok=a_ok, mo_sign=s))
            for k in range(NPL):
                perm = np.random.default_rng(1000 + k).permutation(X.shape[1])
                Xp = X[:, perm]
                sp, _ = orient(Xp, R, E, dates)
                pm = monthly(sp * Xp, R, E, full=False)
                ptr = {"M": FT.eval_ret(Xp, R, E, dates, pre=(sp, pm)), "W": FT.eval_ret(Xw[:, perm], Rw, Ew, wdates),
                       "Q": FT.eval_ret(Xp, Rq, Eq, dates, lag=FT.Q_LAG), "R": FT.eval_risk(Xp, TG, dates)}
                for k_, v_ in ptr.items():
                    if k_ == "W" and not w_ok:
                        continue
                    ptrk.append(FT.placebo_row(f, k, k_, v_, FT.RISK if k_ == "R" else FT.RET))
                pst = {c: pstats(pm[:, i], dates) for i, c in enumerate(COLS[:4])}
                ptg = tags(pst)
                plc.append(dict(f=f, k=k, status=status(pst, ptg), **{f"dev_{u}": ptg[u]["dev"] for u in USES},
                                **{f"top_{u}": ptg[u]["strong"] and ptg[u]["val"] == "強確認" for u in USES},
                                **{f"val_{u}": ptg[u]["val"] for u in USES}))
            print(f"  {f:<7} {row['name'][:12]:<12} {'+' if s > 0 else '−'}  " + " ".join(f"{USES[u]} {row['tag_' + u]:<3}" for u in USES)
                  + f"  → {stt}" + "".join(f"｜{k_} " + FT.official(v_, a_ok)[0]["status"] for k_, v_ in tr.items()) + f"  [{time.time()-t0:.0f}s]", flush=True)
        rho = corr_pairs(Xo, E, dates)
    for r in rows:
        rel = rho.loc[r["H"]].drop(r["H"]).dropna()
        rel = rel[rel.abs() >= RHO].sort_values(key=lambda x: -x.abs())
        r["related"] = "、".join(f"{k} {v:+.2f}" for k, v in rel.items()) or "—"
    PL = pd.DataFrame(plc)
    ev = PL[PL.status != "資料不足"]
    fp = {u: dict(dev=ev[f"dev_{u}"].mean(), val=ev[ev[f"dev_{u}"]][f"val_{u}"].isin(["確認", "強確認"]).mean(),
                  top=ev[f"top_{u}"].mean()) for u in USES}
    fp["status"] = (ev.status == "收").mean()
    T = pd.DataFrame(rows)
    TR = pd.DataFrame(trk)
    PLR = FT.placebo_rates(pd.DataFrame(ptrk), {"M": FT.RET, "W": FT.RET, "Q": FT.RET, "R": FT.RISK})
    sfx = "_preview" if PREVIEW else ""
    T.to_csv(H.D / "results" / f"feature_library{sfx}.csv", index=False)
    TR.to_csv(H.D / "results" / f"feature_library_tracks{sfx}.csv", index=False)
    rho.to_csv(H.D / "results" / f"feature_library_rho{sfx}.csv")
    if not PREVIEW:
        write_md(rows, fp, Path(__file__).resolve().parent / "FEATURES.md", TR, PLR)
    cnt = T.status.value_counts().to_dict()
    tcnt = {k_: g.apply(lambda r: r["AB_status"] if r["a_ok"] else r["BV_status"], axis=1).value_counts().to_dict() for k_, g in TR.groupby("track")}
    print("新軌道正式狀態：" + "；".join(f"{FT.TRACKS[k_]} {v_}" for k_, v_ in tcnt.items()))
    print("新軌道安慰劑（收）：" + "、".join(f"{k_[0]}-{k_[1]} {v_['status']*100:.0f}%" for k_, v_ in PLR.items()))
    print(f"\n狀態：{cnt}；安慰劑 dev 標籤假陽性 " + "、".join(f"{USES[u]} {fp[u]['dev']*100:.0f}%（val 同號 {fp[u]['val']*100:.0f}%）" for u in USES)
          + f"；安慰劑「收」比例 {fp['status']*100:.0f}%；◎✓✓ 假陽性 " + "、".join(f"{USES[u]} {fp[u]['top']*100:.1f}%" for u in USES))
    with open(H.LOG, "a") as fh:
        fh.write(json.dumps(dict(ts=time.strftime("%Y-%m-%d %H:%M:%S"), kind="feature_library", period="dev（A/B）＋ val（V）",
                                 strategy="FL1 特徵庫（含 FS5、FL2 新軌道、FS6、FS7、FS8、FS9）" + ("預覽（不含借券）" if PREVIEW else ""), note="第四十一～四十七批事前登記", status=cnt, placebo=fp,
                                 tracks_status=tcnt, tracks_placebo={f"{k_[0]}-{k_[1]}": v_ for k_, v_ in PLR.items()},
                                 features={r["H"]: dict(status=r["status"], sign=r["sign"], **{u: r[f"tag_{u}"] for u in USES}) for r in rows}),
                            ensure_ascii=False, default=lambda v: round(float(v), 5) if isinstance(v, (float, np.floating)) else str(v)) + "\n")


if __name__ == "__main__":
    main()
