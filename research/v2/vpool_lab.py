"""第六十三批：高波動池（V 池，pit 流動池裡 60 日波動前 100 檔，現股做多）。規則見 HYPOTHESES.md 第六十三批（事前登記）。
- H171 短線擇時（同第五十九批 H164 的菜單，加「收盤漲停／跌停」進場與「停損 2×ATR」出場）
- H172 池內輪動（同第六十一批 H166 的菜單，k ∈ {10, 20}）

用法：.venv/bin/python research/v2/vpool_lab.py cache dev|val   # 日資料（V 池成員）＋月換股日的特徵矩陣
      .venv/bin/python research/v2/vpool_lab.py pool dev|val    # V 池本身的描述（不算試驗）
      .venv/bin/python research/v2/vpool_lab.py tim dev|val     # H171
      .venv/bin/python research/v2/vpool_lab.py rot dev|val     # H172
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
import bpool_lab as BL  # noqa: E402
import harness as H  # noqa: E402
import strategy_lab as SL  # noqa: E402
import timing_lab as TL  # noqa: E402
import xs  # noqa: E402

RES = H.D / "results"
TOPN, VOL_WIN, VOL_MIN = 100, 60, 50
BUY, SELL, COST = H.BUY_FEE, H.SELL_FEE + H.SELL_TAX, xs.COST
FUT_SIDE, ROLL_M = 0.0005, 0.001
GAP = 5   # 停牌 ≥ 5 個交易日後復牌：視為減資等股本變動，跨停牌報酬設為 0（還原價沒有還原減資）
KS, NPL, MIN_DEV_M, BREADTH_MIN = (10, 20), 100, 72, 120
SEG = TL.SEG
TRIALS = 62827
NAME = "第六十三批 V 池"


# ═════════ 資料 ═════════
def gap_fix(P, cols):
    """原始收盤 ≤ 0 的日子視為沒交易；停牌 ≥ GAP 個交易日後復牌的，把之前的價格乘上「復牌開盤 ÷ 停牌前收盤」。
    回傳 (修正後的 open/high/low/close, 交易遮罩, 事件表 [(股, 停牌前最後一天, 復牌日, 因子)])。"""
    trade = (P["raw_close"].reindex(columns=cols) > 0) & P["close"].reindex(columns=cols).notna()
    px = {f: P[f].reindex(columns=cols).where(trade).values.copy() for f in ("open", "high", "low", "close")}
    days = P["close"].index
    ev = []
    tv = trade.values
    for j, c in enumerate(cols):
        ix = np.flatnonzero(tv[:, j])
        for g in np.flatnonzero(np.diff(ix) - 1 >= GAP):
            a, b = ix[g], ix[g + 1]
            o, cl = px["open"][b, j], px["close"][a, j]
            if not (np.isfinite(o) and np.isfinite(cl) and cl > 0):
                continue
            f = o / cl
            for k in px:
                px[k][:b, j] *= f
            ev.append((c, days[a], days[b], f))
    out = {k: pd.DataFrame(v, index=days, columns=cols) for k, v in px.items()}
    return out, trade, ev


def limit_flags(P):
    """一字漲停／一字跌停（開＝高＝低，開盤相對前收接近上限）與收盤漲停／跌停。用還原價（同日比例與原始價相同）。"""
    O, Hh, Lw, C = (P[k].where(P[k] > 0) for k in ("open", "high", "low", "close"))
    lim = pd.Series(np.where(O.index < H.LIMIT_CHANGE, 0.07, 0.10) - 0.005, index=O.index)
    flat = Hh == Lw
    chg = O / C.shift(1) - 1
    cc = C / C.shift(1) - 1
    return (flat & chg.ge(lim, axis=0), flat & chg.le(-lim, axis=0), cc.ge(lim, axis=0), cc.le(-lim, axis=0))


def cache(period):
    import batch55_s8 as B55
    import feature_library as FL
    end = H.PERIODS[period][1]
    p0 = "2010-01-01" if period == "dev" else "2020-01-01"
    t0 = time.time()
    lab = SL.Lab(end)
    P = lab.P
    cols = lab.cols
    dates, R, E = lab.monthly(p0, end)
    R = R.copy()
    days = P["open"].index
    fxa, _, evm = gap_fix(P, cols)
    # 月報酬修正：復牌日落在換股窗 (d_t, d_{t+1}] 裡的，除掉該次跳動（和日資料同一個修正）
    cpos = {c: i for i, c in enumerate(cols)}
    rdx = pd.DatetimeIndex(xs.rebalance_dates(days, pd.Timestamp(p0), pd.Timestamp(end)))
    fixed = []
    for c, a, b, f in evm:
        t = np.searchsorted(rdx.values, b.to_datetime64(), side="left") - 1
        if 0 <= t < len(dates) and dates[t] < b and np.isfinite(R[t, cpos[c]]):
            R[t, cpos[c]] = (1 + R[t, cpos[c]]) / f - 1
            fixed.append((t, cpos[c]))
    C = fxa["close"]
    vol = C.pct_change(fill_method=None).rolling(VOL_WIN, min_periods=VOL_MIN).std().shift(1)
    v = vol.reindex(index=dates, columns=cols).values
    rk = pd.DataFrame(np.where(E & np.isfinite(v), v, np.nan)).rank(axis=1, ascending=False, method="first").values
    M = E & (rk <= TOPN)
    U = np.flatnonzero(M.any(axis=0))
    ucols = cols[U]
    print(f"載入 {time.time()-t0:.0f}s；換股日 {len(dates)} 個；每月成員 {M.sum(axis=1).min()}～{M.sum(axis=1).max()} 檔；曾在池內 {len(U)} 檔；"
          f"停牌 ≥ {GAP} 日後復牌 {len(evm)} 次（全市場），修正月報酬 {len(fixed)} 筆（V 池成員 {sum(bool(M[t, j]) for t, j in fixed)} 筆）", flush=True)
    LU, LD, CU, CD = limit_flags(fxa)
    mtx = B55.mtx_chain(days)

    # ── 日資料（H171）──
    k = np.searchsorted(dates.values, days.values, side="right") - 1
    mem = np.zeros((len(days), len(U)), bool)
    ok = k >= 0
    mem[ok] = M[k[ok]][:, U]
    need = ["open", "high", "low", "close", "volume", "inst_f", "inst_t", "margin_bal", "short_bal", "sbl_bal"]
    D = {f: P[f].reindex(columns=ucols).astype(float) for f in need}
    D.update({k: fxa[k][ucols] for k in fxa})
    ev = [e for e in evm if e[0] in set(ucols)]
    D["gap_events"] = ev
    D["member"] = pd.DataFrame(mem, index=days, columns=ucols)
    for nm, X in (("lu", LU), ("ld", LD), ("cu", CU), ("cd", CD)):
        D[nm] = X.reindex(columns=ucols).fillna(False).astype(bool)
    tr = pd.read_parquet(H.D / "taiex_tr.parquet")
    D["mkt"] = tr.assign(date=pd.to_datetime(tr.date)).drop_duplicates("date").set_index("date").price.astype(float).reindex(days).ffill()
    D["mtx"] = mtx
    D["mdates"] = dates
    D["end"] = end
    D["vol_med"] = (float(np.nanmedian(np.where(M, v, np.nan))), float(np.nanmedian(np.where(E, v, np.nan))))
    with open(RES / f"vpool_daily_{period}.pkl", "wb") as f:
        pickle.dump(D, f)
    print(f"日資料完成 {time.time()-t0:.0f}s；V 池股票的停牌復牌修正 {len(ev)} 個；成員 60 日波動中位 {D['vol_med'][0]*100:.2f}%（pit 池 {D['vol_med'][1]*100:.2f}%）", flush=True)

    # ── 月特徵（H172）──
    X, bad, full = {}, {}, {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i, f in enumerate(FL.FEATS):
            try:
                A = np.asarray(lab.feat(f, dates, E), float)
            except Exception as e:   # noqa: BLE001
                bad[f] = repr(e)[:200]
                continue
            if f in ("H2", "H87", "H97"):
                full[f] = A
            X[f] = np.where(M[:, U], A[:, U], np.nan)
            if i % 20 == 0:
                print(f"  特徵 {i+1}/{len(FL.FEATS)}（{time.time()-t0:.0f}s）", flush=True)
    # H136 每月持股（全池前 20）落在 V 池的比例（描述）
    S136 = SL.combo([(full["H2"], 1), (full["H87"], 1), (full["H97"], 1)], E, 2)
    key = np.where(np.isfinite(S136), S136, -np.inf)
    top = np.argsort(-key, axis=1, kind="stable")[:, :20]
    in_v = np.take_along_axis(M, top, axis=1).mean(axis=1)
    up, dn = lab.limits(dates)
    rd = xs.rebalance_dates(days, pd.Timestamp(p0), pd.Timestamp(end))
    assert list(rd[:-1]) == list(dates), "換股日對不上"
    out = dict(dates=dates, cols=ucols, M=M[:, U], R=R[:, U], X=X, bad=bad, end=end, up=up[:, U], dn=dn[:, U], vol=v[:, U],
               mtx=np.array([mtx.loc[d2] / mtx.loc[d] - 1 for d, d2 in zip(rd[:-1], rd[1:])], float),
               pit_ew=np.array([np.nanmean(R[t][E[t]]) if E[t].any() else np.nan for t in range(len(dates))]),
               h136_in_v=in_v)
    with open(RES / f"vpool_rot_{period}.pkl", "wb") as f:
        pickle.dump(out, f)
    print(f"快取完成 {time.time()-t0:.0f}s；特徵 {len(X)} 個；失敗 {len(bad)} 個 {list(bad)[:10]}；H136 持股在 V 池的比例平均 {in_v.mean():.0%}")


def load(kind, period):
    with open(RES / f"vpool_{kind}_{period}.pkl", "rb") as f:
        return pickle.load(f)


# ═════════ H171 短線擇時 ═════════
class VBook:
    """現股做多、一字漲停買不到（順延）、一字跌停賣不掉（順延）。給「收盤時是否想持有」的布林矩陣，回傳每日報酬。"""

    def __init__(self, D):
        O = D["open"]
        Of = O.fillna(D["close"].shift(1)).ffill()
        self.days = O.index
        self.mem = D["member"].values
        self.r = np.nan_to_num((Of.shift(-1) / Of - 1).values)   # 不遮成員：順延時可能抱到池外
        self.LU, self.LD = D["lu"].values, D["ld"].values
        self.n = np.maximum(self.mem.sum(axis=1), 1)
        self.idx = np.arange(len(self.days))[:, None]
        live = self.mem.sum(axis=1) > 0
        s, _ = self.pnl(self.mem, BUY, SELL, 0.0)
        self.h = np.where(live, s, np.nan)
        sf, _ = self.pnl(self.mem, FUT_SIDE, FUT_SIDE, ROLL_M / 21)
        self.h_fut = np.where(live, sf, np.nan)
        mtx = D["mtx"]
        self.mtx = (mtx.shift(-1) / mtx - 1).values

    def carry(self, d):
        """d[t]：第 t 天開盤想持有。一字漲停想買、一字跌停想賣的那天做不到，沿用前一天的部位。"""
        c = (self.LU & d) | (self.LD & ~d)
        last = np.maximum.accumulate(np.where(c, -1, self.idx), axis=0)
        return np.where(last >= 0, np.take_along_axis(d, np.maximum(last, 0), axis=0), False)

    def pnl(self, d, buy, sell, roll):
        a = self.carry(d)
        prev = np.vstack([np.zeros((1, a.shape[1]), bool), a[:-1]])
        nb = (a & ~prev).sum(axis=1)
        ns = (prev & ~a).sum(axis=1)
        s = ((a * self.r).sum(axis=1) - buy * nb - sell * ns - roll * a.sum(axis=1)) / self.n
        return s, a

    def run(self, inn, fut=False):
        d = np.vstack([np.zeros((1, inn.shape[1]), bool), inn[:-1]]) & self.mem   # 收盤訊號 → 隔天開盤起持有
        s, a = self.pnl(d, FUT_SIDE, FUT_SIDE, ROLL_M / 21) if fut else self.pnl(d, BUY, SELL, 0.0)
        return np.where(np.isfinite(self.h), s, np.nan), a


def tim_setup(period):
    D = load("daily", period)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ent, states, filt, exits, chip = TL.conditions(D)
    ent["收盤漲停"], ent["收盤跌停"] = D["cu"].values, D["cd"].values
    TL.FAMILY["收盤漲停"], TL.FAMILY["收盤跌停"] = "趨勢", "反轉"
    C = D["close"].where(D["close"] > 0)
    pc = C.shift(1)
    tr = pd.concat([D["high"] - D["low"], (D["high"] - pc).abs(), (D["low"] - pc).abs()]).groupby(level=0).max().reindex(C.index)
    exits[STOP] = None
    stop = {"C": C.values, "SL": (C - 2 * tr.rolling(20, min_periods=15).mean()).values}
    bk = VBook(D)
    M = TL.menu(ent, states, filt, exits, chip)
    return D, bk, ent, states, filt, exits, stop, chip, M


STOP = "停損2ATR"


def pos_stop(c, stop, idx):
    """c：進場訊號。收盤低於「最近一次訊號日的停損線（收盤 − 2×ATR20）」就出場；最多持有 20 天。"""
    le = np.maximum.accumulate(np.where(c, idx, -1), axis=0)
    ref = np.take_along_axis(stop["SL"], np.maximum(le, 0), axis=0)
    with np.errstate(invalid="ignore"):
        hit = (stop["C"] < ref) & (le >= 0)
    lx = np.maximum.accumulate(np.where(hit, idx, -1), axis=0)
    return (le >= 0) & (lx < le) & (idx - le < TL.MAXH)


def position(sp, ent, states, filt, exits, stop, idx):
    if sp[2] == STOP:
        c = ent[sp[0]] if filt[sp[1]] is None else (ent[sp[0]] & filt[sp[1]])
        return pos_stop(c, stop, idx)
    return TL.position(sp, ent, states, filt, exits, idx)


_G = {}


def _tim_scan(shift_seed):
    ent, states, filt, exits, stop, M, bk, mdev, mchip = (_G[k] for k in ("ent", "states", "filt", "exits", "stop", "M", "bk", "mdev", "mchip"))
    if shift_seed is not None:
        rng = np.random.default_rng(shift_seed)
        T, N = bk.mem.shape
        k = rng.integers(TL.WIN, T - TL.WIN, N)
        ix = (np.arange(T)[:, None] - k[None, :]) % T
        cj = np.arange(N)[None, :]
        sh = lambda A: A[ix, cj]   # noqa: E731
        ent = {a: sh(b) for a, b in ent.items()}
        states = {a: sh(b) for a, b in states.items()}
        filt = {a: (None if b is None else sh(np.ascontiguousarray(b))) for a, b in filt.items()}
        exits = {a: (None if b is None else sh(b)) for a, b in exits.items()}
        stop = {a: sh(b) for a, b in stop.items()}
    out = np.full(len(M), np.nan)
    for i, sp in enumerate(M):
        s, _ = bk.run(position(sp, ent, states, filt, exits, stop, bk.idx))
        out[i] = TL.alpha_t(s, bk.h, mchip if sp[3] else mdev)[0]
    return out


def _tim_best(seed):
    return float(np.nanmax(_tim_scan(2063_000 + seed)))


def tim_detail(sp, D, bk, ent, states, filt, exits, stop, seg_main, segs):
    inn = position(sp, ent, states, filt, exits, stop, bk.idx)
    s, eff = bk.run(inn)
    sf, _ = bk.run(inn, fut=True)
    days = bk.days
    p0 = max(SEG[seg_main][0], TL.CHIP_START) if sp[3] else SEG[seg_main][0]
    m = TL.mask(days, p0, SEG[seg_main][1]) & np.isfinite(s)
    t, a, b = TL.alpha_t(s, bk.h, m)
    tf = TL.alpha_t(sf, bk.h_fut, m & np.isfinite(bk.h_fut))[0]
    out = dict(entry=sp[0], filt=sp[1], exit=sp[2], chip=sp[3], family=TL.FAMILY[sp[0]], alpha_t=t, alpha_ann=a * 252, beta=b,
               alpha_t_fut=tf, sharpe=TL.sharpe(s[m]), sharpe_hold=TL.sharpe(bk.h[m]), expo=float(eff[m].sum(axis=1).mean() / bk.n[m].mean()))
    for g in segs:
        q0 = max(SEG[g][0], TL.CHIP_START) if sp[3] else SEG[g][0]
        mg = TL.mask(days, q0, SEG[g][1]) & np.isfinite(s)
        out[f"sharpe_{g}"], out[f"hold_{g}"] = TL.sharpe(s[mg]), TL.sharpe(bk.h[mg])
    pos_a = []
    for j in range(eff.shape[1]):
        mj = m & bk.mem[:, j]
        if mj.sum() < BREADTH_MIN:
            continue
        y = eff[mj, j] * bk.r[mj, j]
        X = np.column_stack([np.ones(mj.sum()), bk.r[mj, j]])
        pos_a.append(np.linalg.lstsq(X, y, rcond=None)[0][0] > 0)
    out["breadth"] = float(np.mean(pos_a)) if pos_a else np.nan
    out["n_breadth"] = len(pos_a)
    ent_on = np.vstack([np.zeros((1, eff.shape[1]), bool), eff[1:] & ~eff[:-1]])
    out["entries"] = int(ent_on[m].sum())
    out["hold_days"] = float(eff[m].sum() / max(out["entries"], 1))
    ex = bk.r - np.nan_to_num(bk.mtx)[:, None]
    on = eff & m[:, None]
    out["on_ex_mtx"] = float(ex[on].mean() * 252) if on.any() else np.nan
    out["mem_ex_mtx"] = float(ex[bk.mem & m[:, None]].mean() * 252)
    return out, s


def tim_dev():
    t0 = time.time()
    D, bk, ent, states, filt, exits, stop, chip, M = tim_setup("dev")
    days = bk.days
    mdev = TL.mask(days, *SEG["dev"]) & np.isfinite(bk.h)
    mchip = TL.mask(days, TL.CHIP_START, SEG["dev"][1]) & np.isfinite(bk.h)
    _G.update(ent=ent, states=states, filt=filt, exits=exits, stop=stop, M=M, bk=bk, mdev=mdev, mchip=mchip)
    print(f"載入＋條件 {time.time()-t0:.0f}s；股票 {bk.mem.shape[1]} 檔；菜單 {len(M)} 個；對照組（100 格永遠持有）dev Sharpe {TL.sharpe(bk.h[mdev]):.2f}"
          f"（期貨成本版 {TL.sharpe(bk.h_fut[mdev]):.2f}）", flush=True)
    t1 = time.time()
    real = _tim_scan(None)
    print(f"真實菜單 {time.time()-t1:.0f}s；alpha t ≥ 2 的有 {(real >= 2).sum()} 個；最高 {np.nanmax(real):.2f}", flush=True)
    t1 = time.time()
    with mp.get_context("fork").Pool(9) as pool:
        pl = np.array(pool.map(_tim_best, range(NPL)))
    p95 = float(np.percentile(pl, 95))
    print(f"安慰劑 {NPL} 次（{time.time()-t1:.0f}s）：菜單最佳 alpha t 的 50／95 分位 {np.percentile(pl, 50):.2f}／{p95:.2f}", flush=True)
    order = np.argsort(-np.nan_to_num(real, nan=-9))
    print("\n  菜單 alpha t 前 15：")
    print(pd.DataFrame([dict(entry=M[i][0], filt=M[i][1], exit=M[i][2], alpha_t=round(real[i], 2)) for i in order[:15]]).to_string(index=False))
    print("  菜單 alpha t 後 8：")
    print(pd.DataFrame([dict(entry=M[i][0], filt=M[i][1], exit=M[i][2], alpha_t=round(real[i], 2)) for i in order[::-1][:8]]).to_string(index=False))
    cand = [i for i in order if real[i] >= max(2, p95)]
    print(f"\n== 超過門檻的 {len(cand)} 個，逐一檢查（A、B 段 Sharpe 都高於對照組、廣度 ≥ 50%、進場 ≥ 200）")
    rows = []
    for i in cand[:300]:
        r, _ = tim_detail(M[i], D, bk, ent, states, filt, exits, stop, "dev", ("A", "B"))
        r["pass"] = (r["sharpe_A"] > r["hold_A"]) and (r["sharpe_B"] > r["hold_B"]) and r["breadth"] >= 0.5 and r["entries"] >= 200
        rows.append(r)
    T = pd.DataFrame(rows)
    sel = []
    if len(T):
        P_ = T[T["pass"]]
        print(f"  通過全部條件的 {len(P_)} 個（依族）：")
        for fam, g in P_.groupby("family"):
            b = g.sort_values("alpha_t", ascending=False).iloc[0]
            sel.append((b.entry, b.filt, b.exit))
            print(f"  [{fam}] {len(g)} 個；最佳：{b.entry}＋{b.filt}／{b.exit}  alpha t {b.alpha_t:.2f}（年化 {b.alpha_ann*100:+.1f}%）"
                  f" Sharpe {b.sharpe:.2f}（持有 {b.sharpe_hold:.2f}；A {b.sharpe_A:.2f}/{b.hold_A:.2f}、B {b.sharpe_B:.2f}/{b.hold_B:.2f}）"
                  f" 廣度 {b.breadth:.0%}（{b.n_breadth} 檔） 進場 {b.entries} 平均持有 {b.hold_days:.1f} 天 曝險 {b.expo:.0%}｜期貨成本時 alpha t {b.alpha_t_fut:.2f}"
                  f"｜開時對小台指超額 {b.on_ex_mtx*100:+.1f}%（成員平均 {b.mem_ex_mtx*100:+.1f}%）")
        nf = T[~T["pass"]]
        if len(nf):
            print(f"  沒通過的 {len(nf)} 個（前 8）：")
            for _, b in nf.head(8).iterrows():
                print(f"   {b.entry}＋{b.filt}／{b.exit} t {b.alpha_t:.2f} A {b.sharpe_A:.2f}/{b.hold_A:.2f} B {b.sharpe_B:.2f}/{b.hold_B:.2f} 廣度 {b.breadth:.0%} 進場 {b.entries}")
    extra = {}
    for sp in sel:
        spf = next(x for x in M if x[:3] == sp)
        r, s = tim_detail(spf, D, bk, ent, states, filt, exits, stop, "dev", ("A", "B"))
        mx = TL.mtx_check(spf, D, bk, "dev")
        hc = TL.h136_check(s, bk, D, "dev", ("A", "B"))
        a = s[mdev] - r["beta"] * bk.h[mdev]
        srs = real[np.isfinite(real)] / np.sqrt(mdev.sum())
        extra["／".join(sp)] = dict(mtx=mx, **hc, dsr=SL.deflated_sharpe(a, srs, TRIALS))
        print(f"  {'＋'.join(sp)}：小台指版 {mx if mx is None else {k: round(v, 2) for k, v in mx.items()}}｜和 H136 月相關 {hc['rho_h136']:+.2f}"
              f"｜關卡 3 {hc['gate3']} {'✓' if hc['gate3_ok'] else '✗'}｜DSR（試驗數 {TRIALS}）{extra['／'.join(sp)]['dsr']:.3f}")
    if len(T):
        T.to_csv(RES / "batch63_tim_dev.csv", index=False)
    pd.DataFrame(dict(entry=[m[0] for m in M], filt=[m[1] for m in M], exit=[m[2] for m in M], alpha_t=real)).to_csv(RES / "batch63_tim_menu.csv", index=False)
    SL.log(dict(kind="strategy", period="dev", strategy=f"{NAME} H171 短線擇時", note="第六十三批 事前登記",
                menu=len(M), n_t2=int((real >= 2).sum()), best=float(np.nanmax(real)), placebo_p50=float(np.percentile(pl, 50)),
                placebo_p95=p95, n_cand=len(cand), candidates=T.to_dict("records") if len(T) else [], selected=["／".join(s) for s in sel],
                extra=extra, trials_cum=TRIALS))


def tim_val():
    rec = [json.loads(l) for l in open(H.LOG) if f'"{NAME} H171 短線擇時"' in l and '"period": "dev"' in l][-1]
    sel = [tuple(s.split("／")) for s in rec["selected"]]
    if not sel:
        print("dev 沒有選出 → 依登記不跑 val")
        return
    D, bk, ent, states, filt, exits, stop, chip, M = tim_setup("val")
    mv = TL.mask(bk.days, *SEG["val"]) & np.isfinite(bk.h)
    print(f"對照組 val Sharpe {TL.sharpe(bk.h[mv]):.2f}")
    out = {}
    for sp in sel:
        spf = next(x for x in M if x[:3] == sp)
        r, s = tim_detail(spf, D, bk, ent, states, filt, exits, stop, "val", ("val",))
        ok = r["alpha_t"] >= 2 and r["sharpe"] > r["sharpe_hold"]
        mx = TL.mtx_check(spf, D, bk, "val")
        hc = TL.h136_check(s, bk, D, "val", ("val",))
        m = TL.mask(bk.days, *SEG["val"]) & np.isfinite(s)
        yr = pd.Series(s[m] - r["beta"] * bk.h[m]).groupby(bk.days[m].year).sum().mul(100).round(1).to_dict()
        out["／".join(sp)] = dict(**r, mtx=mx, **hc, alpha_by_year=yr, verdict="✓" if ok else "✗")
        print(f"  [{r['family']}] {'＋'.join(sp)}：alpha t {r['alpha_t']:.2f}（年化 {r['alpha_ann']*100:+.1f}%；期貨成本 {r['alpha_t_fut']:.2f}）"
              f"Sharpe {r['sharpe']:.2f}（持有 {r['sharpe_hold']:.2f}）｜逐年 alpha {yr}｜廣度 {r['breadth']:.0%}｜曝險 {r['expo']:.0%}"
              f"｜和 H136 相關 {hc['rho_h136']:+.2f}｜關卡 3 {hc['gate3']}｜小台指版 {mx} → {'✓' if ok else '✗'}")
    SL.log(dict(kind="strategy", period="val", strategy=f"{NAME} H171 短線擇時", note="第六十三批 dev 選出的", result=out))


# ═════════ H172 池內輪動 ═════════
class VRot:
    """成員壓縮成每月最多 TOPN 欄（依代號排序、不足補 nan），排序只在成員之間。成本：買進比例 × 來回 0.585%。"""

    def __init__(self, B):
        self.dates, self.Mf = B["dates"], B["M"]
        T, U = self.Mf.shape
        K = int(self.Mf.sum(axis=1).max())
        self.ix = np.full((T, K), U)   # U＝虛擬欄（全 nan）
        for t in range(T):
            m = np.flatnonzero(self.Mf[t])
            self.ix[t, :len(m)] = m
        self.M = self.ix < U
        Rp = np.hstack([np.nan_to_num(B["R"]), np.zeros((T, 1))])
        self.R = np.take_along_axis(Rp, self.ix, axis=1)
        self.T, self.U = T, U
        n = np.maximum(self.Mf.sum(axis=1), 1)
        w = self.Mf / n[:, None]
        buy = np.clip(np.diff(w, axis=0, prepend=0), 0, None).sum(axis=1)
        live = self.Mf.sum(axis=1) > 0
        self.h = np.where(live, (w * np.nan_to_num(B["R"])).sum(axis=1) - COST * buy, np.nan)

    def comp(self, A):
        Ap = np.hstack([A, np.full((self.T, 1), np.nan)])
        return np.take_along_axis(Ap, self.ix, axis=1)

    def full(self, Sc):
        S = np.full((self.T, self.U + 1), np.nan)
        np.put_along_axis(S, self.ix, Sc, axis=1)
        return S[:, :self.U]

    def run(self, S, k):
        ok = np.isfinite(S) & self.M
        nv = ok.sum(axis=1)
        key = np.where(ok, S, -np.inf)
        top = np.argsort(-key, axis=1, kind="stable")[:, :k]   # 同分依代號小→大
        valid = nv >= 2 * k
        sel = np.zeros((self.T, self.U + 1), bool)
        np.put_along_axis(sel, np.take_along_axis(self.ix, top, axis=1), True, axis=1)
        sel = sel[:, :self.U]
        sel[~valid] = False
        buy = (sel & ~np.vstack([np.zeros((1, self.U), bool), sel[:-1]])).sum(axis=1) / k
        s = np.take_along_axis(self.R, top, axis=1).mean(axis=1) - COST * buy
        s = np.where(valid & np.isfinite(self.h), s, np.nan)
        return s, sel


def rot_menu(X, sign):
    fs = list(X)
    A = [(f"{f}{'+' if d > 0 else '−'}", [(f, d)], k) for f in fs for d in (1, -1) for k in KS]
    A += [("H136分數", [("H2", 1), ("H87", 1), ("H97", 1)], k) for k in KS]
    import itertools
    Bm = [(f"{f}{'+' if sign.get(f, 1) > 0 else '−'}＆{g}{'+' if sign.get(g, 1) > 0 else '−'}",
           [(f, sign.get(f, 1)), (g, sign.get(g, 1))], k) for f, g in itertools.combinations(fs, 2) for k in KS]
    return A, Bm


def pct_all(Xc, perm=None):
    Pp = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for f, A in Xc.items():
            if perm is not None:
                A = np.take_along_axis(A, perm, axis=1)
            Pp[f] = pd.DataFrame(A).rank(axis=1, pct=True).values
            Pp[f + "−"] = pd.DataFrame(-A).rank(axis=1, pct=True).values
    return Pp


def month_perm(rot, rng):
    perm = np.tile(np.arange(rot.M.shape[1]), (rot.T, 1))
    for t in range(rot.T):
        m = np.flatnonzero(rot.M[t])
        perm[t, m] = rng.permutation(m)
    return perm


def scan_rot(menu, Pp, rot, m, ok):
    out = np.full(len(menu), np.nan)
    for i, sp in enumerate(menu):
        if not ok[i]:
            continue
        s, _ = rot.run(BL.score(sp, Pp, None), sp[2])
        mm = m & np.isfinite(s)
        if mm.sum() < MIN_DEV_M:
            continue
        out[i] = BL.alpha_t(s, rot.h, mm, 3)[0]
    return out


def _rot_pl(seed):
    rot, Xc, menuA, menuB, m, okA, okB = (_G[k] for k in ("rot", "Xc", "A", "B", "m", "okA", "okB"))
    Pp = pct_all(Xc, month_perm(rot, np.random.default_rng(2063_500 + seed)))
    return float(np.nanmax(scan_rot(menuA, Pp, rot, m, okA))), float(np.nanmax(scan_rot(menuB, Pp, rot, m, okB)))


def rot_detail(sp, Pp, rot, B, seg_main, segs):
    k = sp[2]
    Sc = BL.score(sp, Pp, None)
    s, sel = rot.run(Sc, k)
    d = rot.dates
    m = BL.mask(d, seg_main) & np.isfinite(s)
    t, a, b = BL.alpha_t(s, rot.h, m, 3)
    out = dict(name=sp[0], k=k, alpha_t=t, alpha_ann=a * 12, beta=b, months=int(m.sum()),
               sharpe=TL.sharpe(s[m]) / np.sqrt(21), sharpe_pool=TL.sharpe(rot.h[m]) / np.sqrt(21))
    for g in segs:
        mg = BL.mask(d, g) & np.isfinite(s)
        out[f"alpha_{g}"] = BL.alpha_t(s, rot.h, mg, 3)[1] * 12 if mg.sum() > 12 else np.nan
    ex = pd.Series(s[m] - rot.h[m]).groupby(d[m].year).sum()
    out["yrs_pos"] = f"{int((ex > 0).sum())}/{len(ex)}"
    out["by_year"] = (ex * 100).round(1).to_dict()
    s2, _ = rot.run(np.where(np.isfinite(Sc), -Sc, np.nan), k)
    out["top_minus_bottom_ann"] = float(np.nanmean((s - s2)[m]) * 12)
    out["turnover"] = float((sel[1:] & ~sel[:-1]).sum(axis=1)[m[1:]].mean() / k)
    # 一字漲跌停處理後（strategy_lab.portfolio）：val 判定用這個
    lim = SL.portfolio(rot.full(Sc), np.nan_to_num(B["R"]), rot.Mf, k, limits=(B["up"], B["dn"]), cost=COST, min_valid=2 * k)
    sl = np.where(np.isfinite(s), lim.net.values, np.nan)
    ml = m & np.isfinite(sl)
    tl, al, _ = BL.alpha_t(sl, rot.h, ml, 3)
    out.update(alpha_t_lim=tl, alpha_ann_lim=al * 12, sharpe_lim=TL.sharpe(sl[ml]) / np.sqrt(21))
    # 權重 ∝ 1／60 日波動（只當描述，沒有漲跌停處理）
    W = np.where(sel, 1 / np.where(np.isfinite(B["vol"]) & (B["vol"] > 0), B["vol"], np.nan), 0.0)
    W = np.nan_to_num(W / W.sum(axis=1, keepdims=True))
    siv = (W * np.nan_to_num(B["R"])).sum(axis=1) - COST * np.clip(np.diff(W, axis=0, prepend=0), 0, None).sum(axis=1)
    siv = np.where(np.isfinite(s), siv, np.nan)
    tv, av, _ = BL.alpha_t(siv, rot.h, m, 3)
    out.update(alpha_t_iv=tv, alpha_ann_iv=av * 12, sharpe_iv=TL.sharpe(siv[m]) / np.sqrt(21))
    mt = B["mtx"]
    mh = ml & np.isfinite(mt)
    bm = np.polyfit(mt[mh], sl[mh], 1)[0]
    hd = sl[mh] - bm * mt[mh]
    out.update(hedged_ann=float(hd.mean() * 12), hedged_sharpe=float(hd.mean() / hd.std(ddof=1) * np.sqrt(12)), beta_mtx=float(bm))
    return out, s, sl


def rot_setup(period):
    B = load("rot", period)
    rot = VRot(B)
    X = B["X"]
    Xc = {f: rot.comp(A) for f, A in X.items()}
    sign = BL.lib_sign()
    menuA, menuB = rot_menu(X, sign)
    return B, rot, Xc, sign, menuA, menuB, pct_all(Xc)


def rot_dev():
    t0 = time.time()
    B, rot, Xc, sign, menuA, menuB, Pp = rot_setup("dev")
    m = BL.mask(rot.dates, "dev") & np.isfinite(rot.h)
    nval = {f: np.isfinite(A).sum(axis=1) for f, A in Xc.items()}

    def ok(sp):
        if sp[0] == "H136分數":
            return True
        return all((m & (nval[f] >= 2 * sp[2])).sum() >= MIN_DEV_M for f, _ in sp[1])
    okA = np.array([ok(sp) for sp in menuA])
    okB = np.array([ok(sp) for sp in menuB])
    miss = [f for f in Xc if f not in sign]
    print(f"載入 {time.time()-t0:.0f}s；特徵 {len(Xc)}（失敗 {len(B['bad'])}：{list(B['bad'])}；無方向 {miss}）；"
          f"菜單 A {len(menuA)}（可評 {okA.sum()}）、B {len(menuB)}（可評 {okB.sum()}）；對照組 dev Sharpe {TL.sharpe(rot.h[m])/np.sqrt(21):.2f}", flush=True)
    t1 = time.time()
    ra = scan_rot(menuA, Pp, rot, m, okA)
    print(f"菜單 A {time.time()-t1:.0f}s：alpha t ≥ 2 的 {(ra >= 2).sum()} 個，最高 {np.nanmax(ra):.2f}", flush=True)
    t1 = time.time()
    rb = scan_rot(menuB, Pp, rot, m, okB)
    print(f"菜單 B {time.time()-t1:.0f}s：alpha t ≥ 2 的 {(rb >= 2).sum()} 個，最高 {np.nanmax(rb):.2f}", flush=True)
    _G.update(rot=rot, Xc=Xc, A=menuA, B=menuB, m=m, okA=okA, okB=okB)
    t1 = time.time()
    with mp.get_context("fork").Pool(9) as pool:
        pl = np.array(pool.map(_rot_pl, range(NPL)))
    pa, pb = np.percentile(pl[:, 0], 95), np.percentile(pl[:, 1], 95)
    print(f"安慰劑 {NPL} 次（{time.time()-t1:.0f}s）：菜單 A 最佳 50／95 分位 {np.percentile(pl[:,0],50):.2f}／{pa:.2f}；"
          f"菜單 B {np.percentile(pl[:,1],50):.2f}／{pb:.2f}", flush=True)
    for lab_, menu, r in (("A", menuA, ra), ("B", menuB, rb)):
        o = np.argsort(-np.nan_to_num(r, nan=-9))[:12]
        print(f"\n  菜單 {lab_} alpha t 前 12：" + "；".join(f"{menu[i][0]}/k{menu[i][2]} {r[i]:.2f}" for i in o))
    cand = [("A", i, ra[i]) for i in np.flatnonzero(ra >= max(2, pa))] + [("B", i, rb[i]) for i in np.flatnonzero(rb >= max(2, pb))]
    cand.sort(key=lambda x: -x[2])
    print(f"\n== 超過門檻 {len(cand)} 個，檢查 A、B 段 alpha > 0、逐年 ≥ 7/9")
    rows = []
    for lab_, i, _ in cand[:400]:
        sp = (menuA if lab_ == "A" else menuB)[i]
        r, _, _ = rot_detail(sp, Pp, rot, B, "dev", ("A", "B"))
        r["menu"] = lab_
        r["feats"] = [f for f, _ in sp[1]]
        r["pass"] = r["alpha_A"] > 0 and r["alpha_B"] > 0 and int(r["yrs_pos"].split("/")[0]) >= 7
        rows.append(r)
    T = pd.DataFrame(rows)
    sel, used = [], set()
    if len(T):
        for _, r in T[T["pass"]].sort_values("alpha_t", ascending=False).iterrows():
            if used & set(r.feats):
                continue
            sel.append((r.menu, r["name"], int(r.k)))
            used |= set(r.feats)
            if len(sel) == 3:
                break
        print(f"  通過 {int(T['pass'].sum())} 個；前 10：")
        for _, r in T[T["pass"]].sort_values("alpha_t", ascending=False).head(10).iterrows():
            print(f"   [{r.menu}] {r['name']}/k{r.k}：alpha t {r.alpha_t:.2f}（年化 {r.alpha_ann*100:+.1f}%；漲跌停處理後 {r.alpha_t_lim:.2f}）"
                  f"A {r.alpha_A*100:+.1f}% B {r.alpha_B*100:+.1f}% 逐年 {r.yrs_pos}｜Sharpe {r.sharpe:.2f}（池 {r.sharpe_pool:.2f}）"
                  f"｜前減後 {r.top_minus_bottom_ann*100:+.1f}%｜換手 {r.turnover:.0%}｜避險版 {r.hedged_ann*100:+.1f}%（Sharpe {r.hedged_sharpe:.2f}）"
                  f"｜反波動權重 t {r.alpha_t_iv:.2f}（Sharpe {r.sharpe_iv:.2f}）")
    extra = {}
    for mn, nm, k in sel:
        sp = next(x for x in (menuA if mn == "A" else menuB) if x[0] == nm and x[2] == k)
        r, s, sl = rot_detail(sp, Pp, rot, B, "dev", ("A", "B"))
        hc = BL.h136_monthly(sl, rot.dates, "dev", ("A", "B"))
        extra[f"{nm}/k{k}"] = hc
        print(f"  選 val：[{mn}] {nm}/k{k}｜和 H136 相關 {hc['rho_h136']:+.2f}｜關卡 3 {hc['gate3']} {'✓' if hc['gate3_ok'] else '✗'}｜{hc['margin']}")
    if len(T):
        T.drop(columns=["by_year"]).to_csv(RES / "batch63_rot_dev.csv", index=False)
    pd.DataFrame(dict(name=[x[0] for x in menuA] + [x[0] for x in menuB], k=[x[2] for x in menuA] + [x[2] for x in menuB],
                      menu=["A"] * len(menuA) + ["B"] * len(menuB), alpha_t=np.concatenate([ra, rb]))).to_csv(RES / "batch63_rot_menu.csv", index=False)
    SL.log(dict(kind="strategy", period="dev", strategy=f"{NAME} H172 池內輪動", note="第六十三批 事前登記",
                menuA=len(menuA), menuB=len(menuB), evalA=int(okA.sum()), evalB=int(okB.sum()),
                n_t2=int((ra >= 2).sum() + (rb >= 2).sum()), bestA=float(np.nanmax(ra)), bestB=float(np.nanmax(rb)),
                placeboA_p95=float(pa), placeboB_p95=float(pb), n_cand=len(cand), n_pass=int(T["pass"].sum()) if len(T) else 0,
                selected=[f"{a}|{b}|{c}" for a, b, c in sel], extra=extra, trials_cum=TRIALS))


def rot_val():
    rec = [json.loads(l) for l in open(H.LOG) if f'"{NAME} H172 池內輪動"' in l and '"period": "dev"' in l][-1]
    sel = [s.split("|") for s in rec["selected"]]
    if not sel:
        print("dev 沒有選出 → 依登記不跑 val")
        return
    B, rot, Xc, sign, menuA, menuB, Pp = rot_setup("val")
    out = {}
    for mn, nm, k in sel:
        sp = next(x for x in (menuA if mn == "A" else menuB) if x[0] == nm and x[2] == int(k))
        r, s, sl = rot_detail(sp, Pp, rot, B, "val", ())
        ok = r["alpha_t_lim"] >= 2 and r["sharpe_lim"] > r["sharpe_pool"]
        hc = BL.h136_monthly(sl, rot.dates, "val", ("val",))
        out[f"{nm}/k{k}"] = dict(**r, **hc, verdict="✓" if ok else "✗")
        print(f"  [{mn}] {nm}/k{k}：漲跌停處理後 alpha t {r['alpha_t_lim']:.2f}（年化 {r['alpha_ann_lim']*100:+.1f}%；未處理 {r['alpha_t']:.2f}）"
              f"Sharpe {r['sharpe_lim']:.2f}（池 {r['sharpe_pool']:.2f}）｜逐年 {r['by_year']}｜前減後 {r['top_minus_bottom_ann']*100:+.1f}%"
              f"｜避險版 {r['hedged_ann']*100:+.1f}%（Sharpe {r['hedged_sharpe']:.2f}）｜反波動權重 t {r['alpha_t_iv']:.2f}｜和 H136 相關 {hc['rho_h136']:+.2f}｜關卡 3 {hc['gate3']} → {'✓' if ok else '✗'}")
    SL.log(dict(kind="strategy", period="val", strategy=f"{NAME} H172 池內輪動", note="第六十三批 dev 選出的", result=out))


# ═════════ V 池描述 ═════════
def pool(period):
    import portfolio_lab as PL
    B = load("rot", period)
    rot = VRot(B)
    seg = "dev" if period == "dev" else "val"
    m = BL.mask(rot.dates, seg) & np.isfinite(rot.h)
    h, pit, mt = rot.h[m], B["pit_ew"][m], B["mtx"][m]
    ann = lambda x: (x.mean() * 12, x.mean() / x.std(ddof=1) * np.sqrt(12))   # noqa: E731
    bm = np.polyfit(mt, h, 1)[0]
    led = pd.read_parquet(RES / "sleeves" / period / "H136.parquet")
    x = pd.Series(rot.h, index=rot.dates).reindex(led.index).values
    rho = PL.corr(x, led.ret.values, led.index, seg)
    D = load("daily", period)
    buy = (rot.Mf & ~np.vstack([np.zeros((1, rot.U), bool), rot.Mf[:-1]])).sum(axis=1)[m] / TOPN
    print(f"V 池 {period}（{m.sum()} 個月）：60 日波動中位 {D['vol_med'][0]*100:.2f}%（pit 池 {D['vol_med'][1]*100:.2f}%）；每月成員換掉 {buy.mean():.0%}")
    print(f"  池內等權（扣成本）年化 {ann(h)[0]*100:+.1f}% Sharpe {ann(h)[1]:.2f}｜pit 池等權（毛）{ann(pit)[0]*100:+.1f}%／{ann(pit)[1]:.2f}"
          f"｜超額 {ann(h - pit)[0]*100:+.1f}%（t {ann(h - pit)[1]*np.sqrt(m.sum()/12):.2f}）")
    print(f"  − β（{bm:.2f}）× 小台指：{ann(h - bm*mt)[0]*100:+.1f}%／Sharpe {ann(h - bm*mt)[1]:.2f}｜和 H136 月相關 {rho:+.2f}"
          f"｜H136 持股在 V 池的比例平均 {B['h136_in_v'][m].mean():.0%}")
    yr = pd.Series(h - pit, index=rot.dates[m]).groupby(rot.dates[m].year).sum().mul(100).round(1).to_dict()
    print(f"  逐年對 pit 池超額：{yr}")
    SL.log(dict(kind="describe", period=period, strategy=f"{NAME} 池描述", note="第六十三批 描述（不算試驗）",
                ann=float(ann(h)[0]), sharpe=float(ann(h)[1]), ex_pit=float(ann(h - pit)[0]), beta_mtx=float(bm),
                hedged=float(ann(h - bm * mt)[0]), rho_h136=float(rho), h136_in_v=float(B["h136_in_v"][m].mean())))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "tim"
    per = sys.argv[2] if len(sys.argv) > 2 else "dev"
    if cmd == "cache":
        cache(per)
    elif cmd == "pool":
        pool(per)
    elif cmd == "tim":
        {"dev": tim_dev, "val": tim_val}[per]()
    elif cmd == "rot":
        {"dev": rot_dev, "val": rot_val}[per]()
