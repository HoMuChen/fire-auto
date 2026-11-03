"""策略引擎（STRATEGY_PLAN.md S0）：用特徵庫的特徵組策略、回測、安慰劑。

- 特徵：feature_library.Source（日頻分數已 shift 1，換股日用前一交易日的值；PEER／COMBO／IND 在池內算）。
- 投組：portfolio()——換股日（月換股：每月第一個 ≥12 日的交易日）開盤進出、檔數 N、篩選遮罩、留任緩衝、
  權重、一字漲跌停（可開關）；成本在買進時一次扣來回 0.585%（同 FS2 的 bt）。基準＝pit 池等權（不扣成本）。
- 事件：event_trades()——訊號日收盤判斷，成交、漲跌停、下市、成本全部交給 harness.simulate（同第三十八批）。
- 安慰劑：placebo()——同一份菜單套到打亂股票代號的特徵上，取「菜單最佳」的分布。

驗收：.venv/bin/python research/v2/strategy_lab.py verify
  ① 投組引擎重算 H136 dev，要和 FS2 引擎（feature_combo.bt）一致（一字漲跌停關閉，FS2 沒處理）。
  ② 事件引擎重算 H140 dev，要和第三十八批一致（933 筆、每筆超額 +1.64%）。
"""
import json
import math
import multiprocessing as mp
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import feature_library as FL  # noqa: E402
import feature_tracks as FT  # noqa: E402
import harness as H  # noqa: E402
import xs  # noqa: E402

COST, MIN_VALID = xs.COST, 50
PER = FL.PER


# ───────── 資料 ─────────
class Lab:
    """面板＋特徵來源＋月換股的報酬與池。end 決定資料截到哪天（dev 研究用 2019-12-31）。"""

    def __init__(self, end):
        self.end = end
        self.P = FL.load(end)
        self.src = FL.Source(self.P)
        self.cols = self.P["close"].columns
        self._lim = None

    def monthly(self, p0, p1):
        """換股日、期間報酬 R（開盤到下次開盤）、池 E（pit 流動池且 R 有值）。"""
        dates, Rdf, Edf = FL.forward(self.P, p0, p1)
        R = Rdf.reindex(index=dates, columns=self.cols).values
        E = Edf.reindex(index=dates, columns=self.cols).fillna(False).values.astype(bool) & np.isfinite(R)
        return dates, R, E

    def feat(self, f, dates, E):
        return self.src.build(f, dates, E)

    def limits(self, dates):
        """換股日開盤是否一字漲停（買不到）／一字跌停（賣不掉）。用還原價判斷（同日比例與原始價相同）。"""
        if self._lim is None:
            O, Hh, Lw, C = (self.P[k] for k in ("open", "high", "low", "close"))
            lim = pd.Series(np.where(O.index < H.LIMIT_CHANGE, 0.07, 0.10) - 0.005, index=O.index)
            flat = Hh == Lw
            chg = O / C.shift(1) - 1
            self._lim = (flat & chg.ge(lim, axis=0), flat & chg.le(-lim, axis=0))
        up, dn = self._lim
        return (up.reindex(index=dates, columns=self.cols).fillna(False).values,
                dn.reindex(index=dates, columns=self.cols).fillna(False).values)


def combo(mats, E, need):
    """池內百分位平均（H29 法、TU.combo）：mats＝[(矩陣, 方向)]，有值成分數 ≥ need。"""
    pct = [pd.DataFrame(np.where(E & np.isfinite(m), s * m, np.nan)).rank(axis=1, pct=True).values for m, s in mats]
    st = np.stack(pct)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = np.nanmean(st, axis=0)
    out[(np.isfinite(st).sum(axis=0) < need) | ~E] = np.nan
    return out


# ───────── 投組 ─────────
def portfolio(S, R, E, n, buffer=None, filt=None, weight=None, limits=None, cost=COST, min_valid=MIN_VALID, group=None, cap=None, rebal=None):
    """S：分數（大＝好）；R、E 同 Lab.monthly；n：檔數。
    buffer＝k：上期持股還在前 k·n 名就留任，空位由排名遞補。filt：布林遮罩（False 的不能新買，持股也會被換掉）。
    weight(t, idx)→權重；預設等權。limits＝(up, dn)：一字漲停的新股買不到（由下一名遞補）、一字跌停的舊股賣不掉（續抱）。
    group（各欄的產業代碼，< 0＝無）＋ cap：每個產業最多 cap 檔（留任的也算，排名較後的換掉；賣不掉的例外）。
    rebal：布林陣列，False 的月份不換股（持股照當月報酬計、不扣成本；還沒有持股就空手）。
    回傳 DataFrame：net（扣成本）、bench（池等權）、ex、turn（買進比例）、n。"""
    T, N = S.shape
    ok = E & np.isfinite(S) & (filt if filt is not None else True)
    key = np.where(ok, -S, np.inf)
    order = np.argsort(key, axis=1, kind="stable")                  # 平手依欄位（代號）順序
    bench = np.array([R[t][E[t]].mean() if E[t].any() else np.nan for t in range(T)])
    rows, prev, W = [], {}, [{} for _ in range(T)]
    for t in range(T):
        if rebal is not None and not rebal[t]:
            if prev:
                idx = np.array(list(prev), int)
                w = np.array(list(prev.values()))
                net = float(w @ np.nan_to_num(R[t, idx], nan=0.0))
                rows.append((net, bench[t], net - bench[t], 0.0, len(idx)))
                W[t] = prev
            else:
                rows.append((np.nan, bench[t], np.nan, np.nan, 0))
            continue
        cnt = int(ok[t].sum())
        if cnt < min_valid:
            rows.append((np.nan, bench[t], np.nan, np.nan, 0))
            continue
        ranked = order[t, :cnt]
        rank = np.full(N, np.inf)
        rank[ranked] = np.arange(cnt)
        forced = [c for c in prev if limits is not None and limits[1][t, c] and E[t, c]]   # 一字跌停賣不掉
        keep = forced + (sorted((c for c in prev if c not in forced and rank[c] < buffer * n), key=lambda c: rank[c]) if buffer else [])
        hold, ng = [], {}

        def room(c):
            return cap is None or group is None or group[c] < 0 or ng.get(group[c], 0) < cap

        def add(c):
            hold.append(c)
            if group is not None:
                ng[group[c]] = ng.get(group[c], 0) + 1
        for c in keep:
            if c in forced or room(c):
                add(c)
        held = set(hold)
        for c in ranked:
            if len(hold) >= n:
                break
            if c in held or (limits is not None and limits[0][t, c] and c not in prev) or not room(c):
                continue
            add(c)
            held.add(c)
        idx = np.array(hold, int)
        w = weight(t, idx) if weight is not None else np.full(len(idx), 1 / len(idx))
        cur = dict(zip(idx.tolist(), w))
        turn = sum(max(v - prev.get(c, 0.0), 0.0) for c, v in cur.items())
        r = np.nan_to_num(R[t, idx], nan=0.0)
        net = float(w @ r) - turn * cost
        rows.append((net, bench[t], net - bench[t], turn, len(idx)))
        prev = cur
        W[t] = cur
    out = pd.DataFrame(rows, columns=["net", "bench", "ex", "turn", "n"])
    out.attrs["w"] = W
    return out


def staggered(S, R, E, n, k=3, **kw):
    """分 k 批：資金分 k 份，第 j 份只在 t ≡ j (mod k) 的換股日換股、各選 n 檔；淨報酬＝已開始的各份平均。"""
    t = np.arange(len(S))
    parts = [portfolio(S, R, E, n, rebal=(t % k == j), **kw) for j in range(k)]
    nets = np.vstack([p.net.values for p in parts])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        net = np.nanmean(nets, axis=0)
    turn = np.nansum(np.vstack([p.turn.values for p in parts]), axis=0) / np.maximum(np.isfinite(nets).sum(axis=0), 1)
    bench = parts[0].bench.values
    W = []
    for i in range(len(S)):
        live = [p.attrs["w"][i] for p in parts if p.attrs["w"][i]]
        u = {}
        for w in live:
            for c, v in w.items():
                u[c] = u.get(c, 0.0) + v / len(live)
        W.append(u)
    out = pd.DataFrame(dict(net=net, bench=bench, ex=net - bench, turn=turn, n=[len(w) for w in W]))
    out.attrs["w"] = W
    return out


def weights_portfolio(Wm, R, E, bench=None, cost=COST):
    """權重矩陣版（寬投組用）：Wm[t] 每列加總 1（全 0＝空手）。成本＝買進比例 × 來回成本；bench 預設池等權。"""
    Wm = np.nan_to_num(Wm)
    prev = np.vstack([np.zeros(Wm.shape[1]), Wm[:-1]])
    turn = np.clip(Wm - prev, 0, None).sum(axis=1)
    net = (Wm * np.nan_to_num(R)).sum(axis=1) - turn * cost
    if bench is None:
        bench = np.array([R[t][E[t]].mean() if E[t].any() else np.nan for t in range(len(R))])
    held = Wm.sum(axis=1) > 0
    net = np.where(held, net, np.nan)
    out = pd.DataFrame(dict(net=net, bench=bench, ex=net - bench, turn=turn, n=(Wm > 0).sum(axis=1)))
    out.attrs["w"] = [dict(zip(np.flatnonzero(w > 0).tolist(), w[w > 0])) for w in Wm]
    return out


def capacity(df, money20, share=0.05):
    """每月持股的「前 20 日平均成交金額 × share ÷ 權重」最小值，取各月中位數（元）＝ share 規則下可容納的資金。"""
    caps = [min(share * money20[t, c] / w for c, w in W.items()) for t, W in enumerate(df.attrs.get("w", [])) if W]
    caps = [c for c in caps if np.isfinite(c)]
    return float(np.median(caps)) if caps else np.nan


def deflated_sharpe(x, srs, n_trials):
    """Bailey & López de Prado（2014）：x＝候選的月超額序列；srs＝所有試驗的月 Sharpe（估變異）；n_trials＝累計試驗數。回傳機率。"""
    from scipy.stats import kurtosis, norm, skew
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    sr = x.mean() / x.std(ddof=1)
    g = 0.5772156649
    v = np.nanvar(srs, ddof=1) if len(srs) > 1 else 0.0
    sr0 = np.sqrt(v) * ((1 - g) * norm.ppf(1 - 1 / n_trials) + g * norm.ppf(1 - 1 / (n_trials * np.e))) if n_trials > 1 else 0.0
    den = np.sqrt(1 - skew(x) * sr + (kurtosis(x, fisher=False) - 1) / 4 * sr ** 2)
    return float(norm.cdf((sr - sr0) * np.sqrt(len(x) - 1) / den))


def stats(df, dates, p0, p1, ppy=12, lag=0):
    """期間內：年化超額、t（lag>0 用 Newey-West）、正年數、Sharpe、MDD、年化報酬、平均換手。"""
    m = (dates >= pd.Timestamp(p0)) & (dates <= pd.Timestamp(p1)) & df.ex.notna().values
    x = df[m]
    if len(x) < 12:
        return dict(n=len(x))
    yr = x.ex.groupby(dates[m].year).sum()
    nav = (1 + x.net).cumprod()
    return dict(n=len(x), ex=x.ex.mean() * ppy, t=FT.tstat(x.ex.values, lag), yrs=f"{int((yr > 0).sum())}/{len(yr)}",
                sharpe=x.net.mean() / x.net.std(ddof=1) * np.sqrt(ppy), mdd=float((nav / nav.cummax() - 1).min()),
                ret=nav.iloc[-1] ** (ppy / len(x)) - 1, turn=x.turn.mean(), hold=x.n.mean())


# ───────── 事件 ─────────
_G = {}


def _ev_work(sid):
    df = H.load_stock(sid, _G["end"])
    if df is None or len(df) < H.MIN_BARS + 2 or sid not in _G["sig"]:
        return []
    m = _G["sig"][sid].reindex(df.date).fillna(False).values
    if not m.any():
        return []
    sig = np.where(m, "buy", None).tolist()
    return [(sid,) + t for t in H.simulate(df, sig, _G["stop"], _G["max_hold"], False, _G["period"], _G["end"])]


def event_trades(P, sig, p0, p1, end, stop=None, max_hold=None, procs=8):
    """sig：訊號日（收盤判斷）的布林面板。回傳每筆交易（含 ex＝相對 pit 等權開盤指數的超額）。"""
    _G.update(sig=sig, end=end, period=(p0, p1), stop=stop, max_hold=max_hold)
    with mp.get_context("fork").Pool(procs) as pool:
        res = pool.map(_ev_work, list(sig.columns), chunksize=16)
    T = pd.DataFrame([t for r in res for t in r], columns=["sid", "buy", "sell", "ret", "reason", "bars"])
    liq = P["volume"].rolling(H.LIQ_WIN, min_periods=H.LIQ_WIN).mean() >= H.LIQ_SHARES
    O = P["open"]
    mkt = (1 + (O / O.shift() - 1).where(liq.shift().fillna(False).astype(bool)).clip(-0.5, 0.5).mean(axis=1).fillna(0)).cumprod()
    T["mret"] = mkt.reindex(T.sell).values / mkt.reindex(T.buy).values - 1
    T["ex"] = T.ret - T.mret
    return T


def event_stats(T, cluster="M"):
    """每筆：n、平均、中位數、勝率；t 依進場的自然群集（M＝月、Q＝季）先平均再算。"""
    g = T.groupby(T.buy.dt.to_period(cluster)).ex.mean()
    return dict(n=len(T), ex=T.ex.mean(), med=T.ex.median(), win=(T.ret > 0).mean(), clusters=len(g),
                t=g.mean() / g.std(ddof=1) * np.sqrt(len(g)) if len(g) > 1 else np.nan)


# ───────── 安慰劑 ─────────
def placebo(run, mats, n=100, seed=2026):
    """run(mats) → 菜單最佳的分數（純數字）。每次把所有特徵矩陣的欄（股票）用同一個排列打亂。"""
    rng = np.random.default_rng(seed)
    N = next(iter(mats.values())).shape[1]
    out = []
    for _ in range(n):
        perm = rng.permutation(N)
        out.append(run({k: v[:, perm] for k, v in mats.items()}))
    return np.array(out)


def log(rec):
    with open(H.LOG, "a") as f:
        f.write(json.dumps(dict(ts=time.strftime("%Y-%m-%d %H:%M:%S"), **rec), ensure_ascii=False,
                           default=lambda v: round(float(v), 5) if isinstance(v, (float, np.floating)) else str(v)) + "\n")


# ───────── 驗收 ─────────
def verify():
    import batch38_breakout as B38
    import feature_combo as FC
    t0 = time.time()
    p0, p1 = H.PERIODS["dev"]
    lab = Lab(p1)
    print(f"載入 {time.time()-t0:.0f}s", flush=True)

    # ① H136 dev：FS2 引擎 vs 本引擎
    feats = ["H2", "H87", "H97"]
    d1, R1, E1, F = FC.prep(lab.P, p0, p1, feats=feats)
    net1, ex1 = FC.bt(F, R1, E1, (0, 1, 2))
    dates, R, E = lab.monthly(p0, p1)
    S = combo([(lab.feat(f, dates, E), 1) for f in feats], E, math.ceil(2 * len(feats) / 3))
    df = portfolio(S, R, E, FC.TOPN)
    same = (d1.equals(dates) and np.allclose(net1, df.net.values, equal_nan=True, atol=1e-12)
            and np.allclose(ex1, df.ex.values, equal_nan=True, atol=1e-12))
    s_all = stats(df, dates, p0, p1)
    print(f"① H136 dev：FS2 年化超額 {np.nanmean(ex1)*12*100:+.2f}%｜本引擎 {s_all['ex']*100:+.2f}%（t {s_all['t']:.2f}，正年 {s_all['yrs']}）"
          f" → 逐月{'完全一致' if same else '不一致'}", flush=True)
    dfl = portfolio(S, R, E, FC.TOPN, limits=lab.limits(dates))
    sl = stats(dfl, dates, p0, p1)
    print(f"   描述：開啟一字漲跌停後 年化超額 {sl['ex']*100:+.2f}%（差 {(sl['ex']-s_all['ex'])*100:+.2f}%）", flush=True)

    # ② H140 dev：事件引擎 vs 第三十八批
    q0 = max(p0, B38.P0)
    Pe = xs.load_panel(p1)
    sig = B38.signals(Pe, q0, p1)["E"]
    T = event_trades(Pe, sig, q0, p1, p1, stop=None, max_hold=20)
    es = event_stats(T)
    ok2 = len(T) == 933 and round(es["ex"] * 100, 2) == 1.64
    print(f"② H140 dev：{len(T)} 筆、每筆超額 {es['ex']*100:+.2f}%、中位數 {es['med']*100:+.2f}%（第三十八批：933 筆、+1.64%、−0.54%）"
          f" → {'一致' if ok2 else '不一致'}", flush=True)
    log(dict(kind="strategy_lab", period="dev", strategy="S0 引擎驗收", note="STRATEGY_PLAN S0",
             h136_same_as_fs2=same, h136_ex=s_all["ex"], h136_ex_limits=sl["ex"], h140_n=len(T), h140_ex=es["ex"], h140_ok=ok2))
    if not (same and ok2):
        raise SystemExit("驗收失敗")


if __name__ == "__main__":
    if sys.argv[1:2] == ["verify"]:
        verify()
