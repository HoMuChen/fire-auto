"""PORTFOLIO_PLAN P0：組合引擎；P1：現有策略報酬帳、載體盤點（H163）、H159／H160 避險版、決策點（第五十八批）。
規則見 HYPOTHESES.md 第五十八批（事前登記）。限制：不放空個股，放空只到微台（回測用小台指當代理）。

用法：.venv/bin/python research/v2/portfolio_lab.py calib     # β 估計法校準（只用已知的事後 β）
      .venv/bin/python research/v2/portfolio_lab.py verify    # 驗收：重算第四十八、五十三～五十五批與大小價差表
      .venv/bin/python research/v2/portfolio_lab.py p1 dev    # 報酬帳、H163 載體、H159／H160 避險版、決策點
      .venv/bin/python research/v2/portfolio_lab.py p1 val    # 只有 val 已經看過的現有策略
"""
import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch48_s1s2 as B48  # noqa: E402
import batch49_concentration as B49  # noqa: E402
import batch53_value as B53  # noqa: E402
import batch54_fwdey as B54  # noqa: E402
import batch55_s8 as B55  # noqa: E402
import feature_tracks as FT  # noqa: E402
import harness as H  # noqa: E402
import strategy_lab as SL  # noqa: E402

OUT = H.D / "results" / "sleeves"
MARGIN, FUT_COST, MICRO, BETA_WIN, BETA_MIN = 0.15, 0.0002, 10, 250, 200
SEG = {"A": ("2011-01-01", "2014-12-31"), "B": ("2015-01-01", "2019-12-31"),
       "dev": ("2011-01-01", "2019-12-31"), "val": ("2020-01-01", "2023-12-31")}
CARRIERS = {"C1": None, "C2": ("m60", 100), "C3": ("mv", 150), "C4": ("mv", 50)}
CAPITAL, ALLOC, CORE_ALLOC = 2_000_000, 0.3, 0.7
TRIALS = 97
BETA_EST = "DIMSON"        # 第五十八批 calib：Dimson 平均絕對誤差 0.050、BETA 0.138


# ───────── 資料 ─────────
def dimson(P):
    """Dimson β：r_t = a + b0 m_t + b1 m_{t-1}（250 日、至少 200 日），回傳 b0 + b1；已 shift 1（換股日用前一交易日）。"""
    R = P["close"].pct_change(fill_method=None)
    m0 = P["mkt_r"]
    M = pd.DataFrame(np.broadcast_to(m0.values[:, None], R.shape), index=R.index, columns=R.columns).where(R.notna())
    M1 = pd.DataFrame(np.broadcast_to(m0.shift(1).values[:, None], R.shape), index=R.index, columns=R.columns).where(R.notna())
    ok = R.notna() & M.notna() & M1.notna()
    R, M, M1 = R.where(ok), M.where(ok), M1.where(ok)
    m = lambda x: x.rolling(BETA_WIN, min_periods=BETA_MIN).mean()   # noqa: E731
    mr, mm, m1 = m(R), m(M), m(M1)
    s00, s11, s01 = m(M * M) - mm ** 2, m(M1 * M1) - m1 ** 2, m(M * M1) - mm * m1
    c0, c1 = m(R * M) - mr * mm, m(R * M1) - mr * m1
    det = (s00 * s11 - s01 ** 2).where(lambda d: d > 1e-14)
    return (((s11 * c0 - s01 * c1) + (s00 * c1 - s01 * c0)) / det).shift(1)


class Ctx:
    """一次載入：月換股的 R、E、特徵、漲跌停、小台指、加權報酬指數、市值、成交金額、原始價、β。"""

    def __init__(self, end):
        t0 = time.time()
        self.end = end
        self.lab, self.dates, self.R, self.E, self.mats, self.lim, self.codes = B49.load(end)
        lab, d = self.lab, self.dates
        for f in list(B53.COMP) + ["FWD_EY", "BETA"] + B54.FILT:
            if f not in self.mats:
                self.mats[f] = lab.feat(f, d, self.E)
        P = lab.P
        rx = lambda X: X.reindex(index=d, columns=lab.cols).values   # noqa: E731
        self.mv = rx(P["mv"].shift(1))
        money = P["money"].astype(float)
        self.m60 = rx(money.rolling(60, min_periods=40).mean().shift(1))
        self.m20 = rx(money.rolling(20, min_periods=15).mean().shift(1))
        self.px = rx(P["raw_close"].shift(1))
        idx = P["open"].index
        k = [idx.get_loc(x) for x in d]
        nxt = k[1:] + [None]
        ch = B55.mtx_chain(idx)
        self.mtx = np.array([ch.iloc[b] / ch.iloc[a] - 1 if b is not None else np.nan for a, b in zip(k, nxt)])
        tr = pd.read_parquet(H.D / "taiex_tr.parquet")
        tr = tr.assign(date=pd.to_datetime(tr.date)).drop_duplicates("date").set_index("date").price.astype(float)
        trs = tr.reindex(idx, method="ffill")
        self.tr = np.array([trs.iloc[b - 1] / trs.iloc[a - 1] - 1 if b is not None else np.nan for a, b in zip(k, nxt)])
        self.mtx_px = self._front_open(d)
        self.mtx_px_last = self._front_open(None)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.ew = np.array([np.nanmean(self.R[t][self.E[t]]) if self.E[t].any() else np.nan for t in range(len(d))])
        self.ss = self.ew - self.tr
        self._beta = {}
        print(f"載入（資料截到 {end}）{time.time()-t0:.0f}s；{len(d)} 個換股日", flush=True)

    def _front_open(self, days):
        m = pd.read_parquet(H.D / "futures" / "MTX_daily.parquet")
        m = m[(m.trading_session == "position") & (m.open > 0)].copy()
        m["date"] = pd.to_datetime(m.date)
        m["c"] = m.contract_date.astype(str)
        last = m.groupby("c").date.max()
        m = m[m.c.map(last) >= m.date]
        front = m.sort_values(["date", "c"]).groupby("date").first().open.astype(float)
        if days is None:                      # 最新價位（只用來算今天的口數，不是報酬）
            return float(front.iloc[-1])
        return front.reindex(days, method="ffill").values

    def beta(self, est=None):
        est = est or BETA_EST
        if est not in self._beta:
            self._beta[est] = self.mats["BETA"] if est == "BETA" else dimson(self.lab.P).reindex(index=self.dates, columns=self.lab.cols).values
        return self._beta[est]

    def carrier(self, name):
        spec = CARRIERS[name]
        if spec is None:
            return self.E.copy()
        X = getattr(self, spec[0])
        rk = pd.DataFrame(np.where(self.E & np.isfinite(X), X, np.nan)).rank(axis=1, ascending=False).values
        return self.E & (rk <= spec[1])


# ───────── 投組與避險 ─────────
def ew_weights(K):
    s = K.sum(axis=1, keepdims=True)
    return np.divide(K.astype(float), s, out=np.zeros(K.shape), where=s > 0)


def frac_port(S, K, ctx, top=0.3, keep=0.4, min_valid=20):
    """K 內分數前 top 等權做多；已持有的掉出前 keep 才賣；一字漲停的新股買不到、一字跌停的舊股賣不掉。"""
    T, N = S.shape
    W = np.zeros((T, N))
    prev = np.zeros(N, bool)
    up, dn = ctx.lim
    for t in range(T):
        ok = K[t] & ctx.E[t] & np.isfinite(S[t])
        if ok.sum() < min_valid:
            prev = np.zeros(N, bool)
            continue
        q = pd.Series(np.where(ok, S[t], np.nan)).rank(pct=True).values
        new = ok & (q > 1 - top) & ~prev & ~up[t]
        hold = prev & ok & (q > 1 - keep)
        forced = prev & dn[t] & ctx.E[t]
        sel = new | hold | forced
        if sel.any():
            W[t, sel] = 1 / sel.sum()
        prev = sel
    return SL.weights_portfolio(W, ctx.R, ctx.E)


def port_beta(W, B):
    """持股 β：權重平均，β 缺值的股票不計、權重重新正規化；全缺時用 1。"""
    out = np.full(len(W), np.nan)
    for t, w in enumerate(W):
        if not w:
            continue
        idx = np.fromiter(w.keys(), int)
        ww = np.fromiter(w.values(), float)
        b = B[t, idx]
        ok = np.isfinite(b)
        out[t] = float((ww[ok] * b[ok]).sum() / ww[ok].sum()) if ok.any() else 1.0
    return out


def hedge(ctx, net, beta, rf=0.0, capital=None):
    """85% 股票 − 85% × β × 小台指；15% 現金（賺 rf）；期貨成本每月 0.02%。capital 給定時用整數口微台。"""
    hr = np.asarray(beta, float).copy()
    err = np.zeros(len(hr))
    if capital is not None:
        stock = capital * (1 - MARGIN)
        lots = np.round(stock * hr / (ctx.mtx_px * MICRO))
        eff = lots * ctx.mtx_px * MICRO / stock
        with np.errstate(divide="ignore", invalid="ignore"):
            err = eff / hr - 1
        hr = eff
    ret = (1 - MARGIN) * np.asarray(net, float) + MARGIN * rf / 12 - (1 - MARGIN) * hr * ctx.mtx - FUT_COST
    return ret, hr, err


def sleeve_df(ret, turn=None, n=None, stock=1.0, fut=0.0, margin=0.0, beta=np.nan):
    T = len(ret)
    f = lambda v: np.broadcast_to(np.asarray(v, float), (T,)) if np.ndim(v) == 0 else np.asarray(v, float)   # noqa: E731
    return pd.DataFrame(dict(ret=f(ret), turn=f(np.nan if turn is None else turn), n=f(np.nan if n is None else n),
                             stock_exp=f(stock), fut_exp=f(fut), margin=f(margin), beta=f(beta)))


def save(tag, name, dates, df, meta):
    d = OUT / tag
    d.mkdir(parents=True, exist_ok=True)
    df.index = pd.DatetimeIndex(dates, name="date")
    df.to_parquet(d / f"{name}.parquet")
    mp = d / "meta.json"
    m = json.loads(mp.read_text()) if mp.exists() else {}
    m[name] = dict(meta, saved=time.strftime("%Y-%m-%d %H:%M:%S"))
    mp.write_text(json.dumps(m, ensure_ascii=False, indent=1))


# ───────── 評估 ─────────
def _mask(dates, seg, *xs):
    p = SEG[seg] if isinstance(seg, str) else seg
    m = (dates >= pd.Timestamp(p[0])) & (dates <= pd.Timestamp(p[1]))
    for x in xs:
        m &= np.isfinite(x)
    return m


def st(x, dates, seg, rf=0.0):
    m = _mask(dates, seg, x)
    if m.sum() < 12:
        return dict(n=int(m.sum()))
    y = x[m] - rf / 12
    yr = pd.Series(y).groupby(dates[m].year).sum()
    nav = np.cumprod(1 + x[m])
    sd = y.std(ddof=1)
    return dict(n=int(m.sum()), ann=y.mean() * 12, t=y.mean() / sd * np.sqrt(len(y)), sharpe=y.mean() / sd * np.sqrt(12),
                vol=sd * np.sqrt(12), mdd=float((nav / np.maximum.accumulate(nav) - 1).min()),
                yrs=f"{int((yr > 0).sum())}/{len(yr)}", by_year=(yr * 100).round(1).to_dict())


def gate1(s):
    if "t" not in s:
        return False
    a, b = map(int, s["yrs"].split("/"))
    return s["t"] >= 2 and a >= math.ceil(7 / 9 * b)


def corr(a, b, dates, seg):
    m = _mask(dates, seg, a, b)
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 12 else np.nan


def worst25(x, p, dates, seg):
    """P（H136）最差 25% 月份裡，x 與 P 的平均（%／月）。"""
    m = _mask(dates, seg, x, p)
    q = np.quantile(p[m], 0.25)
    w = m & (p <= q)
    return float(x[w].mean()), float(p[w].mean())


def mix_fixed(xs, risk, dates, seg, rf=0.0):
    """「權重 × 波動」固定比例（波動用該段樣本內）：回傳組合的月超額序列（只在 seg 內有值）。"""
    m = _mask(dates, seg, *xs)
    out = np.full(len(dates), np.nan)
    out[m] = sum(r * (x[m] - rf / 12) / (x[m] - rf / 12).std(ddof=1) for x, r in zip(xs, risk))
    return out


def gate3(x, p, dates, segs, rf=0.0):
    """70：30（P：B）組合的 Sharpe 要在每一段都高於 P 單獨。回傳 {段: (組合, P)}、是否全過。"""
    res = {}
    for sg in segs:
        m = _mask(dates, sg, x, p)
        if m.sum() < 24:
            continue
        c = mix_fixed([p, x], [CORE_ALLOC, 1 - CORE_ALLOC], dates, sg, rf)
        sp = (p[m] - rf / 12).mean() / (p[m] - rf / 12).std(ddof=1) * np.sqrt(12)
        sc = c[m].mean() / c[m].std(ddof=1) * np.sqrt(12)
        res[sg] = (float(sc), float(sp))
    return res, bool(res) and all(a > b for a, b in res.values())


def size_exposure(x, ctx, seg):
    m = _mask(ctx.dates, seg, x, ctx.mtx, ctx.ss)
    X = np.column_stack([np.ones(m.sum()), ctx.mtx[m], ctx.ss[m]])
    b = np.linalg.lstsq(X, x[m], rcond=None)[0]
    return float(b[1]), float(b[2])


def post_beta(x, ctx, seg):
    m = _mask(ctx.dates, seg, x, ctx.mtx)
    return float(np.polyfit(ctx.mtx[m], x[m], 1)[0])


def rolling_corr(a, b, win=24):
    return pd.Series(a).rolling(win, min_periods=win).corr(pd.Series(b)).values


def risk_rule(xs, dates, budget=None, look=36, min_n=12, every=3):
    """P5 固定規則：資金權重 ∝ 風險預算 ÷ 過去 look 個月波動，每 every 個月調整；budget＝None 是等風險。回傳組合月報酬。"""
    X = np.column_stack(xs)
    T, k = X.shape
    b = np.ones(k) / k if budget is None else np.asarray(budget, float)
    out, w = np.full(T, np.nan), None
    for t in range(T):
        if t % every == 0 or w is None:
            past = X[max(0, t - look):t]
            ok = np.isfinite(past).all(axis=1)
            if ok.sum() >= min_n:
                sd = past[ok].std(axis=0, ddof=1)
                w = b / sd
                w = w / w.sum()
        if w is not None and np.isfinite(X[t]).all():
            out[t] = float(X[t] @ w)
    return out


def mdd_at_vol(x, target_vol, dates, seg):
    m = _mask(dates, seg, x)
    y = x[m] * target_vol / (x[m].std(ddof=1) * np.sqrt(12))
    nav = np.cumprod(1 + y)
    return float((nav / np.maximum.accumulate(nav) - 1).min())


def boot_sr_diff(c, p, dates, seg, block=12, n=2000, seed=2058):
    """block bootstrap（循環區塊）Sharpe 差 c − p 的 2.5／97.5 分位。"""
    m = _mask(dates, seg, c, p)
    a, b = c[m], p[m]
    T = len(a)
    rng = np.random.default_rng(seed)
    nb = math.ceil(T / block)
    out = []
    for _ in range(n):
        idx = (rng.integers(0, T, nb)[:, None] + np.arange(block)[None, :]).ravel()[:T] % T
        x, y = a[idx], b[idx]
        out.append(x.mean() / x.std(ddof=1) * np.sqrt(12) - y.mean() / y.std(ddof=1) * np.sqrt(12))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def tradeable(ctx, W, beta, alloc, seg="dev", hedged=True):
    """NT$200 萬 × alloc：每檔金額、零股比例、微台口數（歷史與最新價位）、避險誤差、容量（單檔 ≤ 前 20 日均額 5%）。"""
    m = _mask(ctx.dates, seg)
    stock = CAPITAL * alloc * ((1 - MARGIN) if hedged else 1)
    amt, odd, caps, lots, err = [], [], [], [], []
    for t in np.flatnonzero(m):
        w = W[t]
        if not w:
            continue
        idx = np.fromiter(w.keys(), int)
        ww = np.fromiter(w.values(), float)
        a = stock * ww
        lot = ctx.px[t, idx] * 1000
        okp = np.isfinite(lot) & (lot > 0)
        full = np.floor(a[okp] / lot[okp]) * lot[okp]
        amt.append(np.median(a))
        odd.append(1 - full.sum() / a[okp].sum())
        c = 0.05 * ctx.m20[t, idx] / ww
        c = c[np.isfinite(c)]
        if len(c):
            caps.append(c.min())
        if hedged and np.isfinite(beta[t]):
            x = stock * beta[t] / (ctx.mtx_px[t] * MICRO)
            lots.append(x)
            err.append(abs(round(x) / x - 1) if x > 0 else np.nan)
    out = dict(stock=stock, amt_med=float(np.median(amt)), odd=float(np.mean(odd)), cap_med=float(np.median(caps)))
    if hedged:
        b = float(np.nanmean(beta[m]))
        now = stock * b / (ctx.mtx_px_last * MICRO)
        out.update(lots=float(np.mean(lots)), err=float(np.nanmean(err)), lots_now=now, err_now=abs(round(now) / now - 1))
    return out


# ───────── 策略 ─────────
def as_net(df):
    return df.net.values


def build_existing(ctx):
    """現有策略：回傳 {名稱: (sleeve_df, 權重清單或 None, meta)}。"""
    R, E, lim, mats = ctx.R, ctx.E, ctx.lim, ctx.mats
    out = {}
    h136 = B49.run136(mats, R, E, lim, 20)
    out["H136"] = (sleeve_df(h136.net.values, h136.turn.values, h136.n.values), h136.attrs["w"],
                   dict(form="只做多 20 檔、每月換", carrier="pit 池", val_seen=True, holdout_seen=True, live=True))
    h146 = B55.h146(mats, R, E, lim)
    out["H146"] = (sleeve_df(h146.net.values, h146.turn.values, h146.n.values), h146.attrs["w"],
                   dict(form="只做多 50 檔、留任 3N", carrier="pit 池", val_seen=True, holdout_seen=False))
    d161, hr = B55.h161(h146, ctx.mtx, "β")
    out["H161"] = (sleeve_df(d161.net.values, h146.turn.values, h146.n.values, 1 - MARGIN, (1 - MARGIN) * hr, MARGIN, hr), h146.attrs["w"],
                   dict(form="H146 ＋ 空小台指（過去 24 個月迴歸 β）", carrier="pit 池", val_seen=True, holdout_seen=False))
    U = E & np.isfinite(ctx.m60) & (pd.DataFrame(np.where(E, ctx.m60, np.nan)).rank(axis=1, ascending=False).values <= B55.TOPV)
    d162 = B55.h162(mats, R, E, U, lim, "動能營收")
    out["H162"] = (sleeve_df(d162.net.values, stock=1.0, fut=1.0), None,
                   dict(form="前 150 大動能營收多空 20＋20（要放空個股：做不到，只當參考）", carrier="成交金額前 150", val_seen=True,
                        holdout_seen=False, tradeable=False))
    d143, _ = B48.run2(mats, R, E, None, ("all", 0.2, False))
    out["H143"] = (sleeve_df(d143.net.values, d143.turn.values, d143.n.values), d143.attrs["w"],
                   dict(form="全池剔除垃圾 20% 等權", carrier="pit 池", val_seen=True, holdout_seen=False))
    d159 = B53.h159(mats, R, E, "前30%")
    out["H159"] = (sleeve_df(d159.net.values, d159.turn.values, d159.n.values), d159.attrs["w"],
                   dict(form="價值紀律前 30% 等權（只做多）", carrier="pit 池", val_seen=True, holdout_seen=False))
    w160, s160 = h160_monthly(ctx)
    out["H160"] = (sleeve_df(w160), None, dict(form="FWD_EY 篩／50 檔／週換（週報酬併成月）", carrier="pit 池", val_seen=True,
                                              holdout_seen=False, weekly_stats=s160))
    out["MTX"] = (sleeve_df(ctx.mtx, stock=0.0, fut=-1.0), None, dict(form="小台指近月連續做多", val_seen=True, holdout_seen=True))
    out["TAIEX_TR"] = (sleeve_df(ctx.tr), None, dict(form="加權報酬指數（收盤代理）", val_seen=True, holdout_seen=True))
    return out


def h160_monthly(ctx):
    rec = [json.loads(l) for l in open(H.LOG) if '"第五十四批 H160 FWD_EY 集中投組"' in l and '"period": "dev"' in l][-1]
    lab = ctx.lab
    wd, Rw, Ew = FT.weekly(lab.P, lab.cols)
    Ew = Ew & np.isfinite(Rw)
    keep = wd <= pd.Timestamp(ctx.end)
    wd, Rw, Ew = wd[keep], Rw[keep], Ew[keep]
    G = {"週": dict(d=wd, R=Rw, E=Ew, lim=lab.limits(wd))}
    mats = {"週": {f: lab.feat(f, wd, Ew) for f in B54.SIGN}}
    df = B54.run(mats, G, rec["filter_comps"], (True, 50, "週"))
    s = SL.stats(df, wd, B48.P1[0], min(B48.P1[1], ctx.end), ppy=52)
    t = np.searchsorted(ctx.dates.values, wd.values, side="right") - 1
    out = np.full(len(ctx.dates), np.nan)
    g = pd.Series(np.log1p(df.net.values)).groupby(t).sum(min_count=1)
    g = g[(g.index >= 0)]
    out[g.index.values] = np.expm1(g.values)
    out[np.flatnonzero(np.isfinite(ctx.mtx)).max() + 1:] = np.nan
    return out, dict(ex=s["ex"], t=s["t"], sharpe=s["sharpe"], yrs=s["yrs"])


def carrier_sleeve(ctx, name, est=None):
    K = ctx.carrier(name)
    df = SL.weights_portfolio(ew_weights(K), ctx.R, ctx.E)
    b = port_beta(df.attrs["w"], ctx.beta(est))
    ret, hr, _ = hedge(ctx, df.net.values, b)
    return K, df, b, ret, hr


# ───────── 指令 ─────────
def calib():
    ctx = Ctx(H.PERIODS["dev"][1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        R, E, lim, mats = ctx.R, ctx.E, ctx.lim, ctx.mats
        cases = {"pit 池等權": SL.weights_portfolio(ew_weights(E), R, E), "H136": B49.run136(mats, R, E, lim, 20),
                 "H146": B55.h146(mats, R, E, lim)}
        print(f"\n== β 校準（dev 2011–19；事後 β＝月報酬對小台指的迴歸斜率）")
        err = {"BETA": [], "DIMSON": []}
        rows = {}
        for nm, df in cases.items():
            x = df.net.values if nm != "pit 池等權" else ctx.ew
            post = post_beta(x, ctx, "dev")
            r = dict(post=post)
            for est in err:
                b = port_beta(df.attrs["w"], ctx.beta(est))
                r[est] = float(np.nanmean(b[_mask(ctx.dates, "dev")]))
                err[est].append(abs(r[est] - post))
            rows[nm] = r
            print(f"  {nm:<10} 事後 β {post:.2f}｜持股 BETA {r['BETA']:.2f}｜持股 Dimson {r['DIMSON']:.2f}")
        mae = {k: float(np.mean(v)) for k, v in err.items()}
        pick = min(mae, key=mae.get)
        print(f"  平均絕對誤差：BETA {mae['BETA']:.3f}｜Dimson {mae['DIMSON']:.3f} → 用 {pick}")
    SL.log(dict(kind="portfolio", period="dev", strategy="第五十八批 P0 β 校準", note="第五十八批 事前登記（只用已知事後 β）",
                rows=rows, mae=mae, pick=pick))


def verify():
    ctx = Ctx(H.PERIODS["dev"][1])
    logs = [json.loads(l) for l in open(H.LOG)]
    rec = lambda s, p="dev": [r for r in logs if r.get("strategy") == s and r.get("period") == p][-1]   # noqa: E731
    ok_all = []

    def chk(nm, got, want, tol):
        ok = all(abs(g - w) <= tol for g, w in zip(got, want))
        ok_all.append(ok)
        print(f"  {nm:<34} 重算 {', '.join(f'{g:+.3f}' for g in got)}｜原紀錄 {', '.join(f'{w:+.3f}' for w in want)} → {'一致' if ok else '不一致'}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ex = build_existing(ctx)
        d = ctx.dates
        print("\n== 驗收（年化、t、Sharpe）")
        r55 = {r["var"]: r for r in rec("第五十五批 H161/H162 S8 避險與多空")["rows"]}
        s = st(ex["H146"][0].ret.values, d, "dev")
        chk("H146（第五十五批 +26.7%、1.64）", (s["ann"], s["sharpe"]), (0.267, 1.64), 0.006)
        for k, v in (("H161", "H161[避險β]"), ("H162", "H162[動能營收]")):
            s = st(ex[k][0].ret.values, d, "dev")
            chk(f"{v}", (s["ann"], s["t"], s["sharpe"]), (r55[v]["dev_ex"], r55[v]["dev_t"], r55[v]["dev_sharpe"]), 1e-6)
        ov = rec("第五十五批 H161/H162 S8 避險與多空")["overlay"]
        o = (1 - MARGIN) * ex["H146"][0].ret.values + 0.5 * ex["H162"][0].ret.values
        so = st(o, d, "dev")
        chk("疊加 85% H146＋0.5 多空", (so["ann"], so["sharpe"]), (ov["ex"], ov["sharpe"]), 1e-6)
        r53 = rec("第五十三批 H159 價值＋紀律／與 H142 組合")["rows"]
        r53 = [r for r in r53 if r["var"] == "H159[前30%]"][0]
        df159 = B53.h159(ctx.mats, ctx.R, ctx.E, "前30%")
        s = SL.stats(df159, d, *SEG["B"])
        chk("H159[前30%] B 段（超額）", (s["ex"], s["t"], s["sharpe"]), (r53["B_ex"], r53["B_t"], r53["B_sharpe"]), 1e-6)
        r54 = [r for r in rec("第五十四批 H160 FWD_EY 集中投組")["rows"] if r["var"] == "H160[篩/50檔/週換]"][0]
        w = ex["H160"][2]["weekly_stats"]
        chk("H160[篩/50檔/週換]（週、超額）", (w["ex"], w["t"], w["sharpe"]), (r54["dev_ex"], r54["dev_t"], r54["dev_sharpe"]), 1e-6)
        r48 = [r for r in rec("第四十八批 H142/H143")["rows"] if r["var"] == "H143[全池/剔20%]"][0]
        d143, _ = B48.run2(ctx.mats, ctx.R, ctx.E, None, ("all", 0.2, False))
        s = SL.stats(d143, d, *SEG["B"])
        chk("H143[全池/剔20%] B 段（超額）", (s["ex"], s["t"]), (r48["dev_ex"], r48["dev_t"]), 1e-6)
        print("\n== 大小價差（dev 2011–19；PORTFOLIO_PLAN 的表）")
        h136 = ex["H136"][0].ret.values
        b = post_beta(ctx.ew, ctx, "dev")
        for nm, x, want in (("pit 池等權（毛）", ctx.ew, (0.060, 0.35)), ("小台指近月連續", ctx.mtx, (0.084, 0.70)),
                            ("加權報酬指數", ctx.tr, (0.073, 0.64)), ("等權 − 1 倍小台指", ctx.ew - ctx.mtx, (-0.027, -0.27)),
                            (f"等權 − β（{b:.2f}）× 小台指", ctx.ew - b * ctx.mtx, (-0.038, -0.38)), ("H136 淨", h136, (0.333, 1.53))):
            s = st(x, d, "dev")
            chk(nm, (s["ann"], s["sharpe"]), want, 0.006)
        rr = corr(ctx.ew - b * ctx.mtx, h136, d, "dev")
        chk("ρ（等權 − β × 小台指, H136）", (rr,), (0.48,), 0.006)
    print(f"\n驗收：{'全部一致' if all(ok_all) else '有不一致 → 先停'}")
    SL.log(dict(kind="portfolio", period="dev", strategy="第五十八批 P0 引擎驗收", note="第五十八批 事前登記", ok=all(ok_all)))
    if not all(ok_all):
        raise SystemExit(1)


def fmt(s):
    if "t" not in s:
        return f"（樣本 {s['n']} 個月）"
    return f"年化 {s['ann']*100:+6.1f}% t {s['t']:5.2f} 正年 {s['yrs']:>4} Sharpe {s['sharpe']:5.2f} MDD {s['mdd']*100:6.1f}%"


def describe(nm, x, h136, ctx, seg, segs3):
    d = ctx.dates
    s = st(x, d, seg)
    s1 = st(x, d, seg, rf=0.01)
    rho = corr(x, h136, d, seg)
    w25 = worst25(x, h136, d, seg)
    g3, ok3 = gate3(x, h136, d, segs3)
    bm, bs = size_exposure(x, ctx, seg)
    return dict(name=nm, **s, sharpe_rf1=s1.get("sharpe"), rho_h136=rho, worst25=w25[0], worst25_h136=w25[1], gate3=g3,
                gate3_ok=ok3, b_mtx=bm, b_ss=bs)


def p1(period):
    end = H.PERIODS[period][1]
    ctx = Ctx(end)
    d = ctx.dates
    seg = "dev" if period == "dev" else "val"
    segs3 = ("A", "B") if period == "dev" else ("val",)
    res = dict(existing={}, carriers={}, checks={}, trade={})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ex = build_existing(ctx)
        h136 = ex["H136"][0].ret.values
        print(f"\n== P1-1 現有策略（{SEG[seg][0][:4]}–{SEG[seg][1][:4]}，對現金；ρ＝和 H136 的月總報酬相關；最差 25%＝H136 最差 25% 月份的平均；"
              f"關卡 3＝70：30 組合 Sharpe 對 H136）")
        for nm, (df, W, meta) in ex.items():
            save(period, nm, d, df.copy(), dict(meta, end=end))
            r = describe(nm, df.ret.values, h136, ctx, seg, segs3)
            res["existing"][nm] = r
            g3 = "、".join(f"{k} {a:.2f}/{b:.2f}" for k, (a, b) in r["gate3"].items())
            print(f"  {nm:<8} {fmt(r)}｜rf 1% Sharpe {r['sharpe_rf1']:.2f}｜ρ {r['rho_h136']:+.2f}｜最差 25% {r['worst25']*100:+5.1f}%"
                  f"（H136 {r['worst25_h136']*100:+.1f}%）｜關卡 3 {g3} {'✓' if r['gate3_ok'] else '✗'}｜b 小台指 {r['b_mtx']:+.2f} 大小價差 {r['b_ss']:+.2f}",
                  flush=True)
        names = [k for k in ex]
        M = pd.DataFrame({k: ex[k][0].ret.values for k in names})[_mask(d, seg)].corr().round(2)
        print(f"\n  總報酬相關矩陣（{seg}）\n{M.to_string()}")
        res["corr"] = M.to_dict()
        rc = rolling_corr(ex["H161"][0].ret.values, h136)
        m = _mask(d, seg, rc)
        print(f"\n  H161 和 H136 滾動 24 個月相關：最小 {np.nanmin(rc[m]):+.2f}、中位 {np.nanmedian(rc[m]):+.2f}、最大 {np.nanmax(rc[m]):+.2f}")
        # P5 規則示範（描述）：H136＋H161
        x161 = ex["H161"][0].ret.values
        for nm, c in (("核心加衛星 70：30", risk_rule([h136, x161], d, budget=[0.7, 0.3])), ("等風險", risk_rule([h136, x161], d))):
            sc, sp = st(c, d, seg), st(h136, d, seg)
            if "t" in sc:
                lo, hi = boot_sr_diff(c, h136, d, seg)
                print(f"  示範 H136＋H161 {nm}：{fmt(sc)}｜H136 {sp['sharpe']:.2f}｜JK z {B53.jk(c[_mask(d, seg, c, h136)], h136[_mask(d, seg, c, h136)]):+.2f}"
                      f"｜Sharpe 差 95% 區間 [{lo:+.2f}, {hi:+.2f}]｜同波動 MDD {mdd_at_vol(c, sp['vol'], d, seg)*100:.1f}%（H136 {sp['mdd']*100:.1f}%）")
        if period == "val":
            SL.log(dict(kind="portfolio", period="val", strategy="第五十八批 P1 現有策略報酬帳", note="第五十八批（都已看過 val）",
                        existing=res["existing"], corr=res["corr"]))
            return

        # H163 載體盤點
        print(f"\n== H163 載體盤點（dev；等權做多 − 持股 β（{BETA_EST}）× 小台指，85／15；對現金）")
        car = {}
        for nm in CARRIERS:
            K, df, b, ret, hr = carrier_sleeve(ctx, nm)
            save("dev", f"{nm}_hedged", d, sleeve_df(ret, df.turn.values, df.n.values, 1 - MARGIN, (1 - MARGIN) * hr, MARGIN, hr),
                 dict(form=f"載體 {nm} 等權做多 − 持股 β × 小台指", carrier=nm, end=end, val_seen=False, holdout_seen=False))
            r = describe(nm, ret, h136, ctx, "dev", ("A", "B"))
            sa, sb = st(ret, d, "A"), st(ret, d, "B")
            un = st(df.net.values - ctx.tr, d, "dev")
            r.update(A=sa["ann"], B=sb["ann"], post_beta=post_beta(ret, ctx, "dev"), beta_mean=float(np.nanmean(b[_mask(d, "dev")])),
                     n_mean=float(np.nanmean(df.n.values[_mask(d, "dev")])), turn=float(np.nanmean(df.turn.values[_mask(d, "dev")])),
                     vs_tr=un["ann"], vs_tr_t=un["t"])
            car[nm] = (K, df, b, ret)
            res["carriers"][nm] = r
            print(f"  {nm} {fmt(r)}｜A {sa['ann']*100:+.1f}% B {sb['ann']*100:+.1f}%｜ρ H136 {r['rho_h136']:+.2f}｜持股 β {r['beta_mean']:.2f}"
                  f" 避險後事後 β {r['post_beta']:+.2f}｜未避險 − 加權報酬指數 {un['ann']*100:+.1f}%（t {un['t']:.2f}）｜檔數 {r['n_mean']:.0f} 換手 {r['turn']*100:.0f}%",
                  flush=True)
            print(f"      逐年 {r['by_year']}")
        best = max(res["carriers"], key=lambda k: res["carriers"][k]["ann"])
        c1 = res["carriers"]["C1"]
        sel = best if best == "C1" or (res["carriers"][best]["A"] > c1["A"] and res["carriers"][best]["B"] > c1["B"]) else "C1"
        print(f"  dev 避險後年化最高：{best}；A、B 兩段都比 C1 好：{sel == best} → 選出 {sel}")
        res["selected"] = sel

        # 走不走得通的檢查
        K, *_ = car[sel]
        sc159 = SL.combo([(ctx.mats[f], s) for f, s in B53.COMP.items()], K, 3)
        sc160 = SL.combo([(ctx.mats["FWD_EY"], 1)], K, 1)
        print(f"\n== 走不走得通（載體 {sel} 內前 30%、掉出前 40% 才賣、β 避險；描述）")
        for nm, S, sg in (("H159h", sc159, "B"), ("H160h", sc160, "dev")):
            df = frac_port(S, K, ctx)
            b = port_beta(df.attrs["w"], ctx.beta())
            ret, hr, _ = hedge(ctx, df.net.values, b)
            save("dev", f"{nm}_{sel}", d, sleeve_df(ret, df.turn.values, df.n.values, 1 - MARGIN, (1 - MARGIN) * hr, MARGIN, hr),
                 dict(form=f"{nm[:4]} 分數在 {sel} 內前 30% − 持股 β × 小台指", carrier=sel, end=end, val_seen=False, holdout_seen=False))
            segs3 = ("B",) if sg == "B" else ("A", "B")
            r = describe(nm, ret, h136, ctx, sg, segs3)
            r["gate1"] = gate1(r)
            Kc = car[sel][3]
            sig = st(ret - Kc, d, sg)
            r.update(signal_ann=sig["ann"], signal_t=sig["t"], post_beta=post_beta(ret, ctx, sg), n_mean=float(np.nanmean(df.n.values[_mask(d, sg)])),
                     turn=float(np.nanmean(df.turn.values[_mask(d, sg)])))
            rret, _, rerr = hedge(ctx, df.net.values, b, capital=CAPITAL * ALLOC)
            sr_ = st(rret, d, sg)
            r.update(rounded_sharpe=sr_["sharpe"], rounded_t=sr_["t"])
            res["checks"][nm] = r
            res["trade"][nm] = tradeable(ctx, df.attrs["w"], b, ALLOC, sg)
            g3 = "、".join(f"{k} {a:.2f}/{b_:.2f}" for k, (a, b_) in r["gate3"].items())
            print(f"  {nm}（{SEG[sg][0][:4]}–{SEG[sg][1][:4]}）{fmt(r)} → 關卡 1 {'過' if r['gate1'] else '✗'}｜訊號部分（對載體避險版）{sig['ann']*100:+.1f}%（t {sig['t']:.2f}）"
                  f"｜ρ H136 {r['rho_h136']:+.2f}｜最差 25% {r['worst25']*100:+.1f}%｜關卡 3 {g3} {'✓' if r['gate3_ok'] else '✗'}"
                  f"｜事後 β {r['post_beta']:+.2f}｜檔數 {r['n_mean']:.0f} 換手 {r['turn']*100:.0f}%｜整數口微台版 Sharpe {sr_['sharpe']:.2f}", flush=True)
            print(f"      逐年 {r['by_year']}")
        print(f"\n== 可交易性（NT$200 萬；策略 {ALLOC:.0%}、H136 {CORE_ALLOC:.0%}；微台最新價位用 {ctx.mtx_px_last:.0f} 點）")
        res["trade"]["H136"] = tradeable(ctx, ex["H136"][1], None, CORE_ALLOC, "dev", hedged=False)
        for nm, t in res["trade"].items():
            s = (f"  {nm:<6} 股票 {t['stock']/1e4:.0f} 萬｜每檔中位 {t['amt_med']/1e4:.1f} 萬｜零股比例 {t['odd']*100:.0f}%｜容量中位 {t['cap_med']/1e8:.1f} 億")
            if "lots" in t:
                s += (f"｜微台（歷史價位）平均 {t['lots']:.1f} 口、誤差 {t['err']*100:.0f}%｜最新價位 {t['lots_now']:.1f} 口、誤差 {t['err_now']*100:.0f}%")
            print(s)

        # 決策點
        car_bad = res["carriers"][sel]["ann"] <= -0.01
        any_ok = any(r["gate1"] for r in res["checks"].values())
        stop = car_bad and not any_ok
        res["decision"] = dict(carrier_ann=res["carriers"][sel]["ann"], carrier_bad=car_bad, any_gate1=any_ok, stop_p3=stop)
        print(f"\n== 決策點：載體 {sel} 避險後 dev 年化 {res['carriers'][sel]['ann']*100:+.1f}%（≤ −1%：{car_bad}）；"
              f"H159h／H160h 有過關卡 1：{any_ok} → {'P3 不做' if stop else '可以往 P2'}（都先回報使用者）")
    SL.log(dict(kind="portfolio", period="dev", strategy="第五十八批 P1 報酬帳與 H163 載體盤點", note="第五十八批 事前登記",
                beta_est=BETA_EST, **res, trials_cum=TRIALS))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if cmd == "calib":
        calib()
    elif cmd == "verify":
        verify()
    elif cmd == "p1":
        p1(sys.argv[2] if len(sys.argv) > 2 else "dev")
