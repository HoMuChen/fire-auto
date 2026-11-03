"""第四十三批 FL2：特徵庫的新評估軌道。規則見 HYPOTHESES.md 第四十三批（事前登記）。

由 feature_library.py 呼叫，不單獨執行：
- B＋V：每個軌道同時算 A＋B 與 B＋V 兩套標籤；A 段不足 24 個月的特徵正式採 B＋V（方向用 B 段，B t ≥ 2 為 B○、≥ 3 為 B◎）。
- 持有期：週（每個日曆週第一個交易日開盤 → 下週第一個交易日開盤）、60 日（月換股日開盤 → 第 60 個交易日後開盤，Newey-West lag 2）。
- 風險目標：月持有窗的實現波動 RV、最大回落 MDD，與對過去 60 日波動做殘差後的增量。
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata

import feature_screen3 as F3
import xs

PER = {"A": ("2010-01-01", "2014-12-31"), "B": ("2015-01-01", "2019-12-31"), "V": ("2020-01-01", "2023-12-31")}
MIN_N, MIN_M, Q_DAYS, Q_LAG, VOL_WIN = 100, 24, 60, 2, 60
RET = {"ic": 1, "ls": 1, "good": 1, "bad": -1}                    # 標籤要的方向（同 FL1）
RISK = {"rv": 1, "mdd": 1, "rv_x": 1, "mdd_x": 1}                  # 定好方向後「大＝風險高」
RISK_STATUS = ("rv_x", "mdd_x")                                     # 風險狀態只看增量
NAMES = {"ic": "排序", "ls": "多空", "good": "挑贏家", "bad": "避雷",
         "rv": "波動", "mdd": "回落", "rv_x": "波動增量", "mdd_x": "回落增量"}
TRACKS = {"M": "月（≈20 日）", "W": "週（≈5 日）", "Q": "60 日", "R": "風險（月持有窗）"}
PPY = {"M": 12, "W": 52, "Q": 252 / Q_DAYS}                         # 年化倍數
COST = 0.00585


# ───────── 換股日與報酬 ─────────
def _elig(P):
    return P["liq"].shift(1).fillna(False).astype(bool) & (P["nbars"].shift(1) >= 250) & P["open"].notna()


def weekly_dates(idx, p0, p1):
    s = pd.Series(idx, index=idx)
    s = s[(s >= pd.Timestamp(p0)) & (s <= pd.Timestamp(p1))]
    return list(s.groupby(s.dt.to_period("W-SUN")).first())


def grid(P, entries, exits, cols):
    """entries[i] 開盤進場、exits[i]（日索引位置）開盤出場；缺開盤補前一日收盤（同 feature_library.forward）。"""
    O, Cf = P["open"], P["close"].ffill()
    idx = O.index
    el = _elig(P)
    Ov, Cv = O.reindex(columns=cols).values, Cf.reindex(columns=cols).values
    k0 = np.array([idx.get_loc(d) for d in entries])
    k1 = np.asarray(exits)
    out = np.where(np.isnan(Ov[k1]), Cv[k1 - 1], Ov[k1])
    R = np.clip(out / Ov[k0] - 1, -0.95, 10)
    E = el.reindex(index=pd.DatetimeIndex(entries), columns=cols).fillna(False).values.astype(bool)
    return R, E


def weekly(P, cols):
    idx = P["open"].index
    wd = weekly_dates(idx, PER["A"][0], PER["V"][1])
    R, E = grid(P, wd[:-1], [idx.get_loc(d) for d in wd[1:]], cols)
    return pd.DatetimeIndex(wd[:-1]), R, E


def q60(P, dates, cols):
    """dates：月換股日（feature_library 的 dates）。出場超出資料尾（2023-12-31）的期別為 NaN。"""
    idx = P["open"].index
    k = np.array([idx.get_loc(d) for d in dates])
    ok = k + Q_DAYS <= len(idx) - 1
    R = np.full((len(dates), len(cols)), np.nan)
    R[ok], _ = grid(P, list(dates[ok]), k[ok] + Q_DAYS, cols)
    E = _elig(P).reindex(index=dates, columns=cols).fillna(False).values.astype(bool)
    return R, E


def risk_targets(P, dates, cols, E):
    """月持有窗（dates[i] → 下個換股日）的 RV、MDD，及對過去 60 日波動 rank 殘差後的版本。"""
    O, Cf = P["open"], P["close"].ffill()
    idx = O.index
    Ov, Cv = O.reindex(columns=cols).values, Cf.reindex(columns=cols).values
    rd = xs.rebalance_dates(idx, dates[0], pd.Timestamp(PER["V"][1]))
    rd = rd[:len(dates) + 1]
    vol = np.log(P["close"]).diff().rolling(VOL_WIN, min_periods=40).std().shift(1).reindex(index=dates, columns=cols).values
    T, N = len(dates), len(cols)
    out = {u: np.full((T, N), np.nan) for u in RISK}
    for t, (d, d2) in enumerate(zip(rd[:-1], rd[1:])):
        k, k2 = idx.get_loc(d), idx.get_loc(d2)
        o2 = np.where(np.isnan(Ov[k2]), Cv[k2 - 1], Ov[k2])
        path = np.vstack([Ov[k], Cv[k:k2], o2])
        with np.errstate(invalid="ignore", divide="ignore"):
            lr = np.diff(np.log(path), axis=0)
            nstep = np.isfinite(lr).sum(axis=0)
            rv = np.where(nstep >= 10, np.nanstd(lr, axis=0, ddof=1), np.nan)
            mdd = 1 - np.nanmin(path / np.fmax.accumulate(path, axis=0), axis=0)
        rv, mdd = np.where(E[t], rv, np.nan), np.where(E[t], mdd, np.nan)
        out["rv"][t], out["mdd"][t] = rv, mdd
        for u, y in (("rv_x", rv), ("mdd_x", mdd)):
            m = E[t] & np.isfinite(y) & np.isfinite(vol[t])
            if m.sum() >= MIN_N:
                ry, rx = rankdata(y[m]), rankdata(vol[t, m])
                b = np.polyfit(rx, ry, 1)
                out[u][t, m] = ry - np.polyval(b, rx)
    return out


# ───────── 每期指標 ─────────
def ret_metrics(X, R, E, turn=False):
    """X 已定方向（大＝好端）。同 feature_library.monthly 的前四欄；turn=True 另算好端 20% 換手。"""
    T = X.shape[0]
    out = np.full((T, 5), np.nan)
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
        out[t, :4] = [F3.spear(x, r), r[g20].mean() - r[b20].mean(), r[g10].mean() - mu, r[b10].mean() - mu]
        if turn:
            cur = set(np.flatnonzero(m)[g20])
            out[t, 4] = 1 - len(cur & prev) / len(cur) if prev else np.nan
            prev = cur
    return out


def risk_metrics(X, T):
    """未定方向的 IC（Spearman 對號反轉是精確的，方向之後直接乘）。"""
    out = np.full((X.shape[0], len(RISK)), np.nan)
    for t in range(X.shape[0]):
        for j, u in enumerate(RISK):
            y = T[u][t]
            m = np.isfinite(X[t]) & np.isfinite(y)
            if m.sum() >= MIN_N:
                out[t, j] = F3.spear(X[t, m], y[m])
    return out


# ───────── 統計與標籤 ─────────
def tstat(x, lag):
    n = len(x)
    if n <= 2:
        return np.nan
    if lag == 0:
        return x.mean() / x.std(ddof=1) * np.sqrt(n)
    e = x - x.mean()
    s2 = e @ e / n
    for l in range(1, lag + 1):
        s2 += 2 * (1 - l / (lag + 1)) * (e[l:] @ e[:-l]) / n
    return x.mean() / np.sqrt(s2 / n) if s2 > 0 else np.nan


def pstats(v, dates, lag=0):
    res = {}
    for p, (a, b) in PER.items():
        k = (dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b)) & np.isfinite(v)
        x = v[k]
        res[p] = dict(n=len(x), mean=x.mean() if len(x) else np.nan, t=tstat(x, lag))
    return res


def tags(st, sign, scheme):
    """scheme AB＝FL1 規則；BV＝B t ≥ 2 給標籤、≥ 3 為強，V 確認。"""
    out = {}
    for u, s in sign.items():
        a, b, v = st[u]["A"], st[u]["B"], st[u]["V"]
        if b["n"] < MIN_M or (scheme == "AB" and a["n"] < MIN_M):
            out[u] = dict(dev=False, strong=False, val=None)
            continue
        if scheme == "AB":
            dev = s * a["t"] >= 1 and s * b["mean"] > 0
            strong = dev and s * a["t"] >= 2 and s * b["t"] >= 2
        else:
            dev = s * b["t"] >= 2
            strong = s * b["t"] >= 3
        val = None
        if dev and v["n"] >= MIN_M:
            val = "強確認" if s * v["t"] >= 2 else ("確認" if s * v["mean"] > 0 else "反向")
        out[u] = dict(dev=bool(dev), strong=bool(strong), val=val)
    return out


def status(st, tg, scheme, uses):
    ref = st[next(iter(st))]
    if ref["B"]["n"] < MIN_M or (scheme == "AB" and ref["A"]["n"] < MIN_M):
        return "資料不足"
    if not any(tg[u]["dev"] for u in uses):
        return "未收"
    return "收" if any(tg[u]["val"] in ("確認", "強確認") for u in uses) else "待確認"


def cell(t, scheme):
    if not t["dev"]:
        return "·"
    d = ("◎" if t["strong"] else "○") if scheme == "AB" else ("B◎" if t["strong"] else "B○")
    return d + {"強確認": "✓✓", "確認": "✓", "反向": "✗", None: "？"}[t["val"]]


def _ic_sign(v, dates, base, lag):
    return 1.0 if pstats(v, dates, lag)[base]["mean"] >= 0 else -1.0


def eval_ret(X, R, E, dates, lag=0, turn=False, pre=None):
    """報酬軌道：AB、BV 各自定方向（AB 用 A 段 IC、A 不足時借 B 段；BV 用 B 段 IC）並評估。
    pre＝(方向, 已定方向的指標)：已算過的月軌道直接沿用，AB 的方向也照它。"""
    cache = {} if pre is None else {pre[0]: pre[1]}

    def met(s):
        if s not in cache:
            cache[s] = ret_metrics(s * X, R, E, turn=turn)
        return cache[s]

    s0 = next(iter(cache), 1.0)
    st0 = pstats(s0 * met(s0)[:, 0], dates, lag)          # 未定方向的 IC（Spearman 對號反轉是精確的）
    res = {}
    for scheme in ("AB", "BV"):
        if pre is not None and scheme == "AB":
            s = pre[0]
        else:
            base = "B" if scheme == "BV" or st0["A"]["n"] < MIN_M else "A"
            s = 1.0 if st0[base]["mean"] >= 0 else -1.0
        mo = met(s)
        st = {u: pstats(mo[:, i], dates, lag) for i, u in enumerate(RET)}
        tg = tags(st, RET, scheme)
        res[scheme] = dict(sign=s, st=st, tg=tg, status=status(st, tg, scheme, RET),
                           turn=np.nanmean(mo[dates <= pd.Timestamp(PER["B"][1]), 4]) if turn else np.nan)
    return res


def eval_risk(X, T, dates):
    raw = risk_metrics(X, T)
    res = {}
    for scheme in ("AB", "BV"):
        sg, st = {}, {}
        for j, u in enumerate(RISK):
            st0 = pstats(raw[:, j], dates)
            base = "B" if scheme == "BV" or st0["A"]["n"] < MIN_M else "A"
            sg[u] = 1.0 if st0[base]["mean"] >= 0 else -1.0
            st[u] = pstats(sg[u] * raw[:, j], dates)
        tg = tags(st, RISK, scheme)
        res[scheme] = dict(sign=sg, st=st, tg=tg, status=status(st, tg, scheme, RISK_STATUS))
    return res


def not_applicable(res):
    """週軌道：特徵只在月換股日前一日有值（H38、H77、H99），週換股日大多沒值 → 不評。"""
    for r in res.values():
        r["status"] = "不適用"
        r["tg"] = {u: dict(dev=False, strong=False, val=None) for u in r["tg"]}
    return res


def official(res, a_ok):
    """A 段足夠用 AB，否則 BV。"""
    return res["AB" if a_ok else "BV"], ("AB" if a_ok else "BV")


def flat(H, track, res, uses):
    """tracks CSV 的一列（每個 scheme 一組欄位）。"""
    row = dict(H=H, track=track)
    for sc, r in res.items():
        sg = r["sign"]
        row[f"{sc}_sign"] = sg if not isinstance(sg, dict) else ",".join(f"{u}{'+' if v > 0 else '-'}" for u, v in sg.items())
        row[f"{sc}_status"] = r["status"]
        if "turn" in r:
            row[f"{sc}_turn"] = r["turn"]
        for u in uses:
            row[f"{sc}_tag_{u}"] = cell(r["tg"][u], sc)
            for p in PER:
                for k in ("mean", "t", "n"):
                    row[f"{sc}_{u}_{p}_{k}"] = r["st"][u][p][k]
    return row


def placebo_row(f, k, track, res, uses):
    row = dict(f=f, k=k, track=track)
    for sc, r in res.items():
        row[f"{sc}_status"] = r["status"]
        for u in uses:
            t = r["tg"][u]
            row[f"{sc}_dev_{u}"] = t["dev"]
            row[f"{sc}_val_{u}"] = t["val"]
            row[f"{sc}_top_{u}"] = t["strong"] and t["val"] == "強確認"
    return row


def placebo_rates(PL, uses):
    """每個軌道 × scheme：各用途 dev 假陽性、val 同號、頂級假陽性，與「收」比例（排除資料不足）。"""
    out = {}
    for (tr,), g in PL.groupby(["track"]):
        for sc in ("AB", "BV"):
            ev = g[g[f"{sc}_status"] != "資料不足"]
            if not len(ev):
                continue
            us = uses[tr]
            out[(tr, sc)] = dict(n=len(ev), status=(ev[f"{sc}_status"] == "收").mean(),
                                 **{f"dev_{u}": ev[f"{sc}_dev_{u}"].mean() for u in us},
                                 **{f"val_{u}": ev[ev[f"{sc}_dev_{u}"].astype(bool)][f"{sc}_val_{u}"].isin(["確認", "強確認"]).mean() for u in us},
                                 **{f"top_{u}": ev[f"{sc}_top_{u}"].mean() for u in us})
    return out


# ───────── FEATURES.md 段落 ─────────
def _pct(x):
    return f"{x*100:.0f}%" if np.isfinite(x) else "—"


def _ic3(r, sc, u, sc_mul=1, d=3):
    out = []
    for p in PER:
        m, t = r[f"{sc}_{u}_{p}_mean"], r[f"{sc}_{u}_{p}_t"]
        out.append(f"{m*sc_mul:+.{d}f}（{t:+.1f}）" if np.isfinite(m) and np.isfinite(t) else "—")
    return " / ".join(out)


def md(TR, PLR, rows):
    """rows：{H: feature_library 的列}（名稱、起始、月方向）。"""
    TR = TR.set_index(["H", "track"])
    feats = list(rows)
    sc = {f: "AB" if TR.loc[(f, "M"), "a_ok"] else "BV" for f in feats}

    def g(f, tr, col):
        return TR.loc[(f, tr), f"{sc[f]}_{col}"]

    def tags4(f, tr, uses):
        return "／".join(g(f, tr, f"tag_{u}") for u in uses)

    def sg(f, tr):
        return "+" if g(f, tr, "sign") > 0 else "−"

    order = {"收": 0, "待確認": 1, "未收": 2, "資料不足": 3, "不適用": 4}
    L = ["", "## 新評估軌道（第四十三批 FL2）", "",
         "- **正式標籤**：A 段可評估月份 ≥ 24 用 A＋B（FL1 規則）；不足的用 **B＋V**：方向用 B 段 IC，`B○`＝B 段 |t| ≥ 2，`B◎`＝≥ 3，val 確認同 FL1。",
         "- **持有期**：週＝每個日曆週第一個交易日開盤 → 下週第一個交易日開盤；月＝上表（≈ 20 個交易日）；60 日＝月換股日開盤 → 第 60 個交易日後開盤（每月進場、重疊，t 用 Newey-West lag 2）。"
         "每個持有期各自用 A 段（B＋V 用 B 段）IC 定方向。H38、H77、H99 只在月換股日前一日有值，週軌道「不適用」。",
         "- **風險目標**（月持有窗）：實現波動 RV、最大回落 MDD；「增量」＝對過去 60 日波動做池內 rank 迴歸後的殘差。方向定好後「大＝風險高」。風險狀態只看兩個增量用途。",
         "- 標籤符號同上表：用途依序寫成 `排序／多空／挑贏家／避雷`（風險為 `波動／回落／波動增量／回落增量`）。", "",
         "### 安慰劑（同一組打亂，每個特徵 5 次）", "",
         "每格是「dev 標籤假陽性／頂級（`◎✓✓` 或 `B◎✓✓`）假陽性」。", "",
         "| 軌道 | 規則 | 次數 | 排序 | 多空 | 挑贏家 | 避雷 | 「收」比例 |", "|---|---|---|---|---|---|---|---|"]
    for tr in ("M", "W", "Q"):
        for s_ in ("AB", "BV"):
            if (tr, s_) in PLR:
                z = PLR[(tr, s_)]
                L.append(f"| {TRACKS[tr]} | {'A＋B' if s_ == 'AB' else 'B＋V'} | {z['n']} | "
                         + " | ".join(f"{z['dev_' + u]*100:.0f}%／{z['top_' + u]*100:.1f}%" for u in RET) + f" | {z['status']*100:.0f}% |")
    L += ["", "| 軌道 | 規則 | 次數 | 波動 | 回落 | 波動增量 | 回落增量 | 「收」比例（看增量） |", "|---|---|---|---|---|---|---|---|"]
    for s_ in ("AB", "BV"):
        if ("R", s_) in PLR:
            z = PLR[("R", s_)]
            L.append(f"| {TRACKS['R']} | {'A＋B' if s_ == 'AB' else 'B＋V'} | {z['n']} | "
                     + " | ".join(f"{z['dev_' + u]*100:.0f}%／{z['top_' + u]*100:.1f}%" for u in RISK) + f" | {z['status']*100:.0f}% |")

    bv = [f for f in feats if sc[f] == "BV"]
    L += ["", "### B＋V：A 段不足的特徵（月報酬）", "",
          "| 特徵 | 名稱 | 起始 | 方向 | 標籤 | B＋V 狀態 | 排序 IC B / V（t） | 多空 B / V（年化，t） |", "|---|---|---|---|---|---|---|---|"]
    for f in sorted(bv, key=lambda f: (order[g(f, "M", "status")], f)):
        r = TR.loc[(f, "M")]
        ic = " / ".join(f"{r[f'BV_ic_{p}_mean']:+.3f}（{r[f'BV_ic_{p}_t']:+.1f}）" if np.isfinite(r[f"BV_ic_{p}_mean"]) else "—" for p in "BV")
        ls = " / ".join(f"{r[f'BV_ls_{p}_mean']*1200:+.1f}（{r[f'BV_ls_{p}_t']:+.1f}）" if np.isfinite(r[f"BV_ls_{p}_mean"]) else "—" for p in "BV")
        L.append(f"| {f} | {rows[f]['name']} | {rows[f]['start']} | {sg(f, 'M')} | {tags4(f, 'M', RET)} | {g(f, 'M', 'status')} | {ic} | {ls} |")

    L += ["", "### 持有期", "",
          "方向欄依序是 週／月／60 日（和月不同代表短期或長期方向反轉）。換手是好端 20% 每期換掉的比例（dev 平均），括號是換算的年化成本（換手 × 每年期數 × 0.585%）。", "",
          "| 特徵 | 名稱 | 方向 | 週 | 週狀態 | 月 | 月狀態 | 60 日 | 60 日狀態 | 週排序 IC A / B / V（t） | 60 日排序 IC A / B / V（t） | 週挑贏家（年化） | 週換手（成本） |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    ko = sorted(feats, key=lambda f: (order[g(f, "W", "status")], order[g(f, "Q", "status")], f))
    for f in ko:
        rw = TR.loc[(f, "W")]
        tw = g(f, "W", "turn")
        na = g(f, "W", "status") == "不適用"
        L.append(f"| {f} | {rows[f]['name']} | {sg(f, 'W')}／{sg(f, 'M')}／{sg(f, 'Q')} | {tags4(f, 'W', RET)} | {g(f, 'W', 'status')} | "
                 f"{tags4(f, 'M', RET)} | {g(f, 'M', 'status')} | {tags4(f, 'Q', RET)} | {g(f, 'Q', 'status')} | "
                 f"{'—' if na else _ic3(rw, sc[f], 'ic')} | {_ic3(TR.loc[(f, 'Q')], sc[f], 'ic')} | {'—' if na else _ic3(rw, sc[f], 'good', 5200, 1)} | "
                 f"{_pct(tw)}（{_pct(tw * PPY['W'] * COST) if np.isfinite(tw) else '—'}） |")

    L += ["", "### 風險目標", "",
          "方向欄是兩個增量用途的方向（+＝原始值越大、下期風險越高）。", "",
          "| 特徵 | 名稱 | 方向（波動增量／回落增量） | 標籤 | 風險狀態 | 波動 IC A / B / V（t） | 波動增量 IC A / B / V（t） | 回落增量 IC A / B / V（t） |",
          "|---|---|---|---|---|---|---|---|"]
    for f in sorted(feats, key=lambda f: (order[g(f, "R", "status")], f)):
        r = TR.loc[(f, "R")]
        sgn = dict((x[:-1], x[-1]) for x in g(f, "R", "sign").split(","))
        d = "／".join("+" if sgn[u] == "+" else "−" for u in RISK_STATUS)
        L.append(f"| {f} | {rows[f]['name']} | {d} | {tags4(f, 'R', RISK)} | {g(f, 'R', 'status')} | "
                 f"{_ic3(r, sc[f], 'rv')} | {_ic3(r, sc[f], 'rv_x')} | {_ic3(r, sc[f], 'mdd_x')} |")
    return L
