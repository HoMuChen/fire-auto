"""第五十九批 H164：市值前 20 大短線擇時（個股期貨做多／空手），特徵組合 × 規則網格搜尋。
規則見 HYPOTHESES.md 第五十九批（事前登記）。

用法：.venv/bin/python research/v2/timing_lab.py cache dev|val   # 載入面板、存前 20 大的日資料（約 6 分鐘）
      .venv/bin/python research/v2/timing_lab.py dev             # 菜單 3,860 個、安慰劑 100 次、候選、選 val
      .venv/bin/python research/v2/timing_lab.py val             # 只跑 dev 選出的（每族最多 1 個）
"""
import json
import multiprocessing as mp
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402
import strategy_lab as SL  # noqa: E402

RES = H.D / "results"
TOPN, SIDE, ROLL_M, WIN, NW_LAG, NPL, MAXH = 20, 0.0005, 0.001, 250, 5, 100, 20
HOLDS = (1, 3, 5, 10, 20)
SEG = {"A": ("2011-01-01", "2014-12-31"), "B": ("2015-01-01", "2019-12-31"), "dev": ("2011-01-01", "2019-12-31"),
       "val": ("2020-01-01", "2023-12-31")}
CHIP_START = "2013-05-01"
TRIALS = 3957


# ───────── 資料 ─────────
def cache(period):
    import feature_library as FL
    end = H.PERIODS[period][1]
    t0 = time.time()
    lab = SL.Lab(end)
    P = lab.P
    dates, Rdf, Edf = FL.forward(P, "2010-01-01", end)
    cols = lab.cols
    E0 = Edf.reindex(index=dates, columns=cols).fillna(False).values.astype(bool)
    mv = P["mv"].shift(1).reindex(index=dates, columns=cols).values
    rk = pd.DataFrame(np.where(E0 & np.isfinite(mv), mv, np.nan)).rank(axis=1, ascending=False).values
    M = E0 & (rk <= TOPN)
    U = np.flatnonzero(M.any(axis=0))
    ucols = cols[U]
    days = P["open"].index
    # 每日池成員：最近一個換股日（≤ 當天）的名單
    k = np.searchsorted(dates.values, days.values, side="right") - 1
    mem = np.zeros((len(days), len(U)), bool)
    ok = k >= 0
    mem[ok] = M[k[ok]][:, U]
    pick = lambda f: P[f].reindex(columns=ucols).astype(float)   # noqa: E731
    need = ["open", "high", "low", "close", "volume", "inst_f", "inst_t", "margin_bal", "short_bal", "sbl_bal"]
    miss = [f for f in need if f not in P]
    if miss:
        raise SystemExit(f"面板缺欄位：{miss}；有的欄位：{sorted(P)[:60]}")
    D = {f: pick(f) for f in need}
    D["member"] = pd.DataFrame(mem, index=days, columns=ucols)
    tr = pd.read_parquet(H.D / "taiex_tr.parquet")
    D["mkt"] = tr.assign(date=pd.to_datetime(tr.date)).drop_duplicates("date").set_index("date").price.astype(float).reindex(days).ffill()
    import batch55_s8 as B55
    D["mtx"] = B55.mtx_chain(days)
    D["mdates"] = dates
    D["end"] = end
    with open(RES / f"timing_{period}.pkl", "wb") as f:
        pickle.dump(D, f)
    n = D["member"].sum(axis=1)
    print(f"快取 {period}：{time.time()-t0:.0f}s；曾在前 20 大的股票 {len(U)} 檔；每日成員數中位 {n[n > 0].median():.0f}；"
          f"2330 成交量樣本 {D['volume']['2330'].dropna().iloc[-1]:.0f}、外資淨買樣本 {D['inst_f']['2330'].dropna().iloc[-1]:.0f}")


def load(period):
    with open(RES / f"timing_{period}.pkl", "rb") as f:
        return pickle.load(f)


# ───────── 特徵與條件 ─────────
def own_pct(X):
    """今天的值在自己過去 250 個交易日（含今天共 251 個）裡的分位。"""
    return X.rolling(WIN + 1, min_periods=200).rank(pct=True)


def rsi(C, n):
    d = C.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d).clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def streak(X, k, pos=True):
    s = (X > 0) if pos else (X < 0)
    out = s.copy()
    for j in range(1, k):
        out &= s.shift(j, fill_value=False)
    return out


FAMILY = {}


def conditions(D):
    """回傳 (entries, states, filters, exits, chip)：每個都是 {名稱: 布林矩陣（日 × 股）}，值＝「當天收盤時」成立。"""
    O, Hh, L, C, V = (D[k] for k in ("open", "high", "low", "close", "volume"))
    F, T = D["inst_f"], D["inst_t"]
    mkt = pd.DataFrame(np.broadcast_to(D["mkt"].values[:, None], C.shape), index=C.index, columns=C.columns)
    ma = {n: C.rolling(n, min_periods=n).mean() for n in (5, 10, 20, 60, 120, 250)}
    r1 = C.pct_change(fill_method=None)
    rv20 = r1.rolling(20, min_periods=15).std()
    pc = C.shift(1)
    tr = pd.concat([Hh - L, (Hh - pc).abs(), (L - pc).abs()]).groupby(level=0).max().reindex(C.index)
    lo20, hi20 = L.rolling(20).min(), Hh.rolling(20).max()
    cont = {
        "ret1": r1, "ret5": C / C.shift(5) - 1, "ret20": C / C.shift(20) - 1,
        "dma20": C / ma[20] - 1, "dma60": C / ma[60] - 1, "rngpos": (C - lo20) / (hi20 - lo20),
        "gap": O / pc - 1, "rv20": rv20, "atr": tr.rolling(5).mean() / tr.rolling(20).mean(), "vratio": V / V.rolling(20).mean(),
        "f1": F / V, "f5": F.rolling(5).sum() / V.rolling(5).sum(), "t1": T / V, "t5": T.rolling(5).sum() / V.rolling(5).sum(),
        "mg5": D["margin_bal"] / D["margin_bal"].shift(5) - 1, "sh5": D["short_bal"] / D["short_bal"].shift(5) - 1,
        "sbl5": D["sbl_bal"] / D["sbl_bal"].shift(5) - 1, "mkt5": mkt / mkt.shift(5) - 1,
    }
    cont = {k: v.replace([np.inf, -np.inf], np.nan) for k, v in cont.items()}
    chipc = {"f1", "f5", "t1", "t5", "mg5", "sh5", "sbl5"}
    fam_c = {"ret1": ("趨勢", "反轉"), "ret5": ("趨勢", "反轉"), "ret20": ("趨勢", "反轉"), "dma20": ("趨勢", "反轉"),
             "dma60": ("趨勢", "反轉"), "rngpos": ("趨勢", "反轉"), "gap": ("趨勢", "反轉"), "rv20": ("波動", "波動"),
             "atr": ("波動", "波動"), "vratio": ("量", "量"), "mkt5": ("大盤", "大盤")}
    ent, chip = {}, set()
    for k, X in cont.items():
        p = own_pct(X)
        for tail, cond in (("高", p >= 0.9), ("低", p <= 0.1)):
            nm = f"{k}{tail}"
            ent[nm] = cond.values
            FAMILY[nm] = "籌碼" if k in chipc else fam_c[k][0 if tail == "高" else 1]
            if k in chipc:
                chip.add(nm)
    r2, r14 = rsi(C, 2), rsi(C, 14)
    sd20 = C.rolling(20).std(ddof=0)
    rng = Hh - L
    tech = {
        "新高20": (C >= C.rolling(20).max(), "趨勢"), "新高55": (C >= C.rolling(55).max(), "趨勢"),
        "新高120": (C >= C.rolling(120).max(), "趨勢"), "新高250": (C >= C.rolling(250).max(), "趨勢"),
        "RSI2<10": (r2 < 10, "反轉"), "RSI2<5": (r2 < 5, "反轉"), "RSI14<30": (r14 < 30, "反轉"), "RSI14>70": (r14 > 70, "趨勢"),
        "布林下軌": (C < ma[20] - 2 * sd20, "反轉"), "布林上軌": (C > ma[20] + 2 * sd20, "趨勢"),
        "NR7": (rng <= rng.rolling(7).min(), "波動"), "NR4": (rng <= rng.rolling(4).min(), "波動"),
        "連3黑": (streak(r1, 3, False), "反轉"), "連3紅": (streak(r1, 3, True), "趨勢"),
        "外資連3買": (streak(F, 3, True), "籌碼"), "外資連5買": (streak(F, 5, True), "籌碼"),
        "外資連3賣": (streak(F, 3, False), "籌碼"), "投信連3買": (streak(T, 3, True), "籌碼"),
    }
    for k, (cond, fam) in tech.items():
        ent[k] = cond.fillna(False).values.astype(bool)
        FAMILY[k] = fam
        if fam == "籌碼":
            chip.add(k)
    states = {f"站上MA{n}": (C > ma[n]) for n in (5, 10, 20, 60, 120)}
    states.update({"MA5>MA20": ma[5] > ma[20], "MA10>MA60": ma[10] > ma[60], "MA20>MA120": ma[20] > ma[120]})
    states = {k: v.fillna(False).values.astype(bool) for k, v in states.items()}
    for k in states:
        FAMILY[k] = "趨勢"
    mma = D["mkt"].rolling(60).mean()
    mk = (D["mkt"] > mma).values[:, None] & np.isfinite(mma.values)[:, None]
    mk2 = (D["mkt"] < mma).values[:, None]
    p_rv = own_pct(rv20)
    f20 = F.rolling(20, min_periods=20).sum()
    filt = {"無": None, "收>MA60": (C > ma[60]).values, "收<MA60": (C < ma[60]).values, "收>MA250": (C > ma[250]).values,
            "大盤>MA60": np.broadcast_to(mk, C.shape), "大盤<MA60": np.broadcast_to(mk2, C.shape),
            "低波動": (p_rv <= 0.5).values, "高波動": (p_rv > 0.5).values, "外資20日買": (f20 > 0).values, "外資20日賣": (f20 < 0).values}
    exits = {"站上MA5": (C > ma[5]).values, "跌破MA10": (C < ma[10]).values}
    return ent, states, filt, exits, chip | {"外資20日買", "外資20日賣"}


# ───────── 部位與績效 ─────────
def pos_hold(e, n):
    c = np.cumsum(e, axis=0, dtype=np.int32)
    lag = np.vstack([np.zeros((n, e.shape[1]), np.int32), c[:-n]])
    return (c - lag) > 0


def pos_exit(e, x, idx):
    le = np.maximum.accumulate(np.where(e, idx, -1), axis=0)
    lx = np.maximum.accumulate(np.where(x, idx, -1), axis=0)
    return (le >= 0) & (le >= lx) & (idx - le < MAXH)


class Book:
    """固定的報酬、成員與對照組；給「收盤時是否持有」的布林矩陣，回傳策略每日報酬。"""

    def __init__(self, D):
        O = D["open"]
        Of = O.fillna(D["close"].shift(1)).ffill()
        r = (Of.shift(-1) / Of - 1).values
        self.days = O.index
        self.mem = D["member"].values
        self.r = np.nan_to_num(np.where(self.mem, r, 0.0))
        self.n = np.maximum(self.mem.sum(axis=1), 1)
        live = self.mem.sum(axis=1) > 0
        dm = np.abs(np.diff(self.mem.astype(np.int8), axis=0, prepend=0)).sum(axis=1)
        self.h = np.where(live, self.r.sum(axis=1) / self.n - SIDE * dm / self.n - ROLL_M / 21, np.nan)
        self.idx = np.arange(len(self.days))[:, None]
        mtx = D["mtx"]
        self.mtx = (mtx.shift(-1) / mtx - 1).values

    def run(self, inn, side=SIDE):
        eff = np.vstack([np.zeros((1, inn.shape[1]), bool), inn[:-1]]) & self.mem   # 收盤訊號 → 隔天開盤起持有
        tr = np.abs(np.diff(eff.astype(np.int8), axis=0, prepend=0)).sum(axis=1)
        s = ((eff * self.r).sum(axis=1) - side * tr - ROLL_M / 21 * eff.sum(axis=1)) / self.n
        return np.where(np.isfinite(self.h), s, np.nan), eff


def mask(days, p0, p1):
    return np.asarray((days >= pd.Timestamp(p0)) & (days <= pd.Timestamp(p1)))


def alpha_t(s, h, m):
    y, x = s[m], h[m]
    X = np.column_stack([np.ones(len(x)), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for L in range(1, NW_LAG + 1):
        G = Xe[L:].T @ Xe[:-L]
        S += (1 - L / (NW_LAG + 1)) * (G + G.T)
    Bi = np.linalg.inv(X.T @ X)
    V = Bi @ S @ Bi
    return float(b[0] / np.sqrt(V[0, 0])), float(b[0]), float(b[1])


def sharpe(x):
    x = x[np.isfinite(x)]
    return float(x.mean() / x.std(ddof=1) * np.sqrt(252)) if len(x) > 20 else np.nan


# ───────── 菜單 ─────────
def menu(ent, states, filt, exits, chip):
    M = []
    for e in ent:
        for f in filt:
            for x in [f"持有{n}" for n in HOLDS] + list(exits):
                M.append((e, f, x, (e in chip) or (f in chip)))
    for s in states:
        for f in filt:
            M.append((s, f, "狀態", f in chip))
    return M


def position(spec, ent, states, filt, exits, idx):
    e, f, x, _ = spec
    if x == "狀態":
        c = states[e]
        return c if filt[f] is None else (c & filt[f])
    c = ent[e] if filt[f] is None else (ent[e] & filt[f])
    if x.startswith("持有"):
        return pos_hold(c, int(x[2:]))
    return pos_exit(c, exits[x], idx)


_G = {}


def _scan(shift_seed):
    """整個菜單的 alpha t；shift_seed 不是 None 時先把所有條件按股票循環平移（安慰劑）。"""
    ent, states, filt, exits, M, bk, mdev, mchip = (_G[k] for k in ("ent", "states", "filt", "exits", "M", "bk", "mdev", "mchip"))
    if shift_seed is not None:
        rng = np.random.default_rng(shift_seed)
        T, N = bk.mem.shape
        k = rng.integers(WIN, T - WIN, N)
        ix = (np.arange(T)[:, None] - k[None, :]) % T
        cj = np.arange(N)[None, :]
        sh = lambda A: A[ix, cj]   # noqa: E731
        ent = {a: sh(b) for a, b in ent.items()}
        states = {a: sh(b) for a, b in states.items()}
        filt = {a: (None if b is None else sh(np.ascontiguousarray(b))) for a, b in filt.items()}
        exits = {a: sh(b) for a, b in exits.items()}
    out = np.full(len(M), np.nan)
    for i, sp in enumerate(M):
        s, _ = bk.run(position(sp, ent, states, filt, exits, bk.idx))
        out[i] = alpha_t(s, bk.h, mchip if sp[3] else mdev)[0]
    return out


def setup(period):
    D = load(period)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ent, states, filt, exits, chip = conditions(D)
    bk = Book(D)
    M = menu(ent, states, filt, exits, chip)
    return D, bk, ent, states, filt, exits, chip, M


def detail(sp, D, bk, ent, states, filt, exits, seg_main, segs):
    inn = position(sp, ent, states, filt, exits, bk.idx)
    s, eff = bk.run(inn)
    s2, _ = bk.run(inn, side=2 * SIDE)
    days = bk.days
    p0 = max(SEG[seg_main][0], CHIP_START) if sp[3] else SEG[seg_main][0]
    m = mask(days, p0, SEG[seg_main][1]) & np.isfinite(s)
    t, a, b = alpha_t(s, bk.h, m)
    t2 = alpha_t(s2, bk.h, m)[0]
    out = dict(entry=sp[0], filt=sp[1], exit=sp[2], chip=sp[3], family=FAMILY[sp[0]], alpha_t=t, alpha_ann=a * 252, beta=b, alpha_t_cost2=t2,
               sharpe=sharpe(s[m]), sharpe_hold=sharpe(bk.h[m]), expo=float(eff[m].sum(axis=1).mean() / bk.n[m].mean()))
    for g in segs:
        q0 = max(SEG[g][0], CHIP_START) if sp[3] else SEG[g][0]
        mg = mask(days, q0, SEG[g][1]) & np.isfinite(s)
        out[f"sharpe_{g}"], out[f"hold_{g}"] = sharpe(s[mg]), sharpe(bk.h[mg])
    # 廣度：池內待滿 250 天的股票，個股 alpha（持有時的報酬對該股報酬的截距）
    pos_a = []
    for j in range(eff.shape[1]):
        mj = m & bk.mem[:, j]
        if mj.sum() < 250:
            continue
        y = eff[mj, j] * bk.r[mj, j]
        X = np.column_stack([np.ones(mj.sum()), bk.r[mj, j]])
        pos_a.append(np.linalg.lstsq(X, y, rcond=None)[0][0] > 0)
    out["breadth"] = float(np.mean(pos_a)) if pos_a else np.nan
    ent_on = np.vstack([np.zeros((1, eff.shape[1]), bool), eff[1:] & ~eff[:-1]])
    out["entries"] = int(ent_on[m].sum())
    out["hold_days"] = float(eff[m].sum() / max(out["entries"], 1))
    # 訊號開的期間，個股相對小台指的超額（年化）
    ex = bk.r - np.nan_to_num(bk.mtx)[:, None]
    on = eff & m[:, None]
    out["on_ex_mtx"] = float(ex[on].mean() * 252) if on.any() else np.nan
    out["mem_ex_mtx"] = float(ex[bk.mem & m[:, None]].mean() * 252)
    return out, s


def mtx_check(sp, D, bk, seg):
    """同一套規則套在加權報酬指數上（只限只用收盤價的條件與濾網），報酬用小台指近月連續。"""
    only_close = {"ret1", "ret5", "ret20", "dma20", "dma60", "rv20", "mkt5"}
    base = sp[0][:-1] if sp[0][-1] in "高低" else sp[0]
    ok_e = base in only_close or sp[0].startswith(("新高", "RSI", "布林", "連3", "站上MA", "MA"))
    ok_f = sp[1] in ("無", "收>MA60", "收<MA60", "收>MA250", "大盤>MA60", "大盤<MA60", "低波動", "高波動")
    ok_x = sp[2] in ("狀態", "站上MA5", "跌破MA10") or sp[2].startswith("持有")
    if not (ok_e and ok_f and ok_x):
        return None
    c = D["mkt"]
    days = bk.days
    one = pd.DataFrame({"m": c})
    D2 = {k: one.copy() for k in ("open", "high", "low", "close")}
    D2["volume"] = one * np.nan
    for k in ("inst_f", "inst_t", "margin_bal", "short_bal", "sbl_bal"):
        D2[k] = one * np.nan
    D2["mkt"] = c
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ent, states, filt, exits, _ = conditions(D2)
    inn = position(sp, ent, states, filt, exits, np.arange(len(days))[:, None])
    r = np.nan_to_num(bk.mtx)
    eff = np.concatenate([[False], inn[:-1, 0]])
    tr = np.abs(np.diff(eff.astype(np.int8), prepend=0))
    s = eff * r - SIDE * tr - ROLL_M / 21 * eff
    h = r - ROLL_M / 21
    m = mask(days, *SEG[seg]) & np.isfinite(bk.mtx)
    t, a, _ = alpha_t(s, h, m)
    return dict(alpha_t=t, alpha_ann=a * 252, sharpe=sharpe(s[m]), sharpe_hold=sharpe(h[m]))


def h136_check(s, bk, D, period, segs):
    led = pd.read_parquet(RES / "sleeves" / period / "H136.parquet")
    md = led.index
    k = np.searchsorted(md.values, bk.days.values, side="right") - 1
    g = pd.Series(np.log1p(np.nan_to_num(s))).groupby(k).sum()
    g = g[g.index >= 0]
    mret = np.full(len(md), np.nan)
    mret[g.index.values] = np.expm1(g.values)
    import portfolio_lab as PL
    h = led.ret.values
    seg = "dev" if period == "dev" else "val"
    g3, ok3 = PL.gate3(mret, h, md, segs)
    return dict(rho_h136=PL.corr(mret, h, md, seg), gate3=g3, gate3_ok=ok3)


# ───────── 指令 ─────────
def dev():
    t0 = time.time()
    D, bk, ent, states, filt, exits, chip, M = setup("dev")
    days = bk.days
    mdev = mask(days, *SEG["dev"]) & np.isfinite(bk.h)
    mchip = mask(days, CHIP_START, SEG["dev"][1]) & np.isfinite(bk.h)
    _G.update(ent=ent, states=states, filt=filt, exits=exits, M=M, bk=bk, mdev=mdev, mchip=mchip)
    print(f"載入＋條件 {time.time()-t0:.0f}s；菜單 {len(M)} 個；對照組（20 格永遠持有）dev Sharpe {sharpe(bk.h[mdev]):.2f}", flush=True)
    t1 = time.time()
    real = _scan(None)
    print(f"真實菜單 {time.time()-t1:.0f}s；alpha t ≥ 2 的有 {(real >= 2).sum()} 個；最高 {np.nanmax(real):.2f}", flush=True)
    t1 = time.time()
    with mp.get_context("fork").Pool(8) as pool:
        pl = np.array(pool.map(lambda_best, range(NPL)))
    p95 = float(np.percentile(pl, 95))
    print(f"安慰劑 {NPL} 次（{time.time()-t1:.0f}s）：菜單最佳 alpha t 的 50／95 分位 {np.percentile(pl, 50):.2f}／{p95:.2f}", flush=True)
    order = np.argsort(-np.nan_to_num(real, nan=-9))
    cand = [i for i in order if real[i] >= max(2, p95)]
    print(f"\n== 超過安慰劑門檻的 {len(cand)} 個，逐一檢查（Sharpe A、B 段都高於對照組、廣度 ≥ 50%、進場 ≥ 200）")
    rows = []
    for i in cand[:300]:
        r, s = detail(M[i], D, bk, ent, states, filt, exits, "dev", ("A", "B"))
        r["pass"] = (r["sharpe_A"] > r["hold_A"]) and (r["sharpe_B"] > r["hold_B"]) and r["breadth"] >= 0.5 and r["entries"] >= 200
        rows.append(r)
    T = pd.DataFrame(rows)
    top = pd.DataFrame([dict(entry=M[i][0], filt=M[i][1], exit=M[i][2], alpha_t=real[i]) for i in order[:15]])
    print("\n  菜單 alpha t 前 15：")
    print(top.to_string(index=False))
    sel = []
    if len(T):
        P_ = T[T["pass"]]
        print(f"\n  通過全部條件的 {len(P_)} 個（依族）：")
        for fam, g in P_.groupby("family"):
            b = g.sort_values("alpha_t", ascending=False).iloc[0]
            sel.append((b.entry, b.filt, b.exit))
            print(f"  [{fam}] {len(g)} 個；最佳：{b.entry}＋{b.filt}／{b.exit}  alpha t {b.alpha_t:.2f}（年化 {b.alpha_ann*100:+.1f}%）"
                  f" Sharpe {b.sharpe:.2f}（持有 {b.sharpe_hold:.2f}；A {b.sharpe_A:.2f}/{b.hold_A:.2f}、B {b.sharpe_B:.2f}/{b.hold_B:.2f}）"
                  f" 廣度 {b.breadth:.0%} 進場 {b.entries} 平均持有 {b.hold_days:.1f} 天 曝險 {b.expo:.0%}｜成本 0.2% 時 alpha t {b.alpha_t_cost2:.2f}"
                  f"｜開時對小台指超額 {b.on_ex_mtx*100:+.1f}%（成員平均 {b.mem_ex_mtx*100:+.1f}%）")
    extra = {}
    for sp in sel:
        spf = next(x for x in M if x[:3] == sp)
        r, s = detail(spf, D, bk, ent, states, filt, exits, "dev", ("A", "B"))
        mx = mtx_check(spf, D, bk, "dev")
        hc = h136_check(s, bk, D, "dev", ("A", "B"))
        a = s[mdev] - r["beta"] * bk.h[mdev]
        srs = real[np.isfinite(real)] / np.sqrt(mdev.sum())
        extra["／".join(sp)] = dict(mtx=mx, **hc, dsr=SL.deflated_sharpe(a, srs, TRIALS))
        print(f"  {'＋'.join(sp)}：小台指版 {mx if mx is None else {k: round(v, 2) for k, v in mx.items()}}｜和 H136 月相關 {hc['rho_h136']:+.2f}"
              f"｜關卡 3 {hc['gate3']} {'✓' if hc['gate3_ok'] else '✗'}｜DSR（試驗數 {TRIALS}）{extra['／'.join(sp)]['dsr']:.3f}")
    if len(T):
        T.to_csv(RES / "batch59_dev.csv", index=False)
    pd.DataFrame(dict(entry=[m[0] for m in M], filt=[m[1] for m in M], exit=[m[2] for m in M], alpha_t=real)).to_csv(RES / "batch59_menu.csv", index=False)
    SL.log(dict(kind="strategy", period="dev", strategy="第五十九批 H164 前 20 大短線擇時", note="第五十九批 事前登記",
                menu=len(M), n_t2=int((real >= 2).sum()), best=float(np.nanmax(real)), placebo_p50=float(np.percentile(pl, 50)),
                placebo_p95=p95, n_cand=len(cand), candidates=T.to_dict("records") if len(T) else [], selected=["／".join(s) for s in sel],
                extra=extra, trials_cum=TRIALS))


def lambda_best(seed):
    return float(np.nanmax(_scan(2059_000 + seed)))


def val():
    rec = [json.loads(l) for l in open(H.LOG) if '"第五十九批 H164 前 20 大短線擇時"' in l and '"period": "dev"' in l][-1]
    sel = [tuple(s.split("／")) for s in rec["selected"]]
    if not sel:
        print("dev 沒有選出 → 依登記不跑 val")
        return
    D, bk, ent, states, filt, exits, chip, M = setup("val")
    out = {}
    for sp in sel:
        spf = next(x for x in M if x[:3] == sp)
        r, s = detail(spf, D, bk, ent, states, filt, exits, "val", ("val",))
        ok = r["alpha_t"] >= 2 and r["sharpe"] > r["sharpe_hold"]
        mx = mtx_check(spf, D, bk, "val")
        hc = h136_check(s, bk, D, "val", ("val",))
        m = mask(bk.days, *SEG["val"]) & np.isfinite(s)
        yr = pd.Series(s[m] - r["beta"] * bk.h[m]).groupby(bk.days[m].year).sum().mul(100).round(1).to_dict()
        out["／".join(sp)] = dict(**r, mtx=mx, **hc, alpha_by_year=yr, verdict="✓" if ok else "✗")
        print(f"  [{r['family']}] {'＋'.join(sp)}：alpha t {r['alpha_t']:.2f}（年化 {r['alpha_ann']*100:+.1f}%）Sharpe {r['sharpe']:.2f}（持有 {r['sharpe_hold']:.2f}）"
              f"｜逐年 alpha {yr}｜廣度 {r['breadth']:.0%}｜和 H136 相關 {hc['rho_h136']:+.2f}｜小台指版 {mx} → {'✓' if ok else '✗'}")
    SL.log(dict(kind="strategy", period="val", strategy="第五十九批 H164 前 20 大短線擇時", note="第五十九批 dev 選出的", result=out))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "dev"
    if cmd == "cache":
        cache(sys.argv[2])
    else:
        {"dev": dev, "val": val}[cmd]()
