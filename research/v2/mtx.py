"""第十批：小台指（MTX）單一標的、可多可空（H47~H54）。規則與過關標準見 HYPOTHESES.md 第十批。

部位：n[i] = 收盤 i → 開盤 i+1 持有（跳空/夜盤段），d[i] = 開盤 i → 收盤 i 持有（日盤段），單位為名目/權益。
一般訊號 s[i] 於收盤 i 算出 → d[i+1] = s[i]、n[i] = s[i-1]（次日開盤才換）。
用法：python3 research/v2/mtx.py fetch | dev | val [H47 ...]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402

RAW = H.D / "futures" / "MTX_daily.parquet"
TAX, FEE = 0.00002, 0.00004                     # 每邊期交稅、手續費（另加滑價 1 點）


def fetch():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from build_data_v2 import RateLimiter, api
    lim, out = RateLimiter(80), []
    for m in pd.date_range("2008-06-01", pd.Timestamp.today(), freq="MS"):
        e = m + pd.offsets.MonthEnd(0)
        d = api(lim, dataset="TaiwanFuturesDaily", data_id="MTX", start_date=f"{m:%Y-%m-%d}", end_date=f"{e:%Y-%m-%d}")
        if d is not None and len(d):
            out.append(d)
    df = pd.concat(out, ignore_index=True)
    df = df[df.contract_date.str.fullmatch(r"\d{6}")]
    RAW.parent.mkdir(exist_ok=True)
    df.to_parquet(RAW)
    print(len(df), df.date.min(), df.date.max())


def _last_day(contract, idx):
    """合約最後交易日 = 該月第三個週三（遇休市順延到下一交易日）。用日曆算，不看資料（防前視）。"""
    y, m = int(contract[:4]), int(contract[4:])
    d = pd.Timestamp(y, m, 1)
    wed3 = d + pd.Timedelta(days=(2 - d.weekday()) % 7 + 14)
    pos = idx.searchsorted(wed3)
    if pos == len(idx):                                     # 在資料之後：用日曆天數往後推（只需保持順序）
        pos += (wed3 - idx[-1]).days
    return pos


def continuous(end):
    """回傳 DataFrame：gap（收盤 i-1→開盤 i）、intra（開盤 i→收盤 i）、roll（收盤 i 轉倉）、po/pc（成本用價格）、S（訊號指數）。"""
    df = pd.read_parquet(RAW)
    df = df[(df.trading_session == "position")].assign(date=lambda x: pd.to_datetime(x.date))
    df = df[df.date <= end]
    O = df.pivot_table(index="date", columns="contract_date", values="open")
    C = df.pivot_table(index="date", columns="contract_date", values="close")
    idx = C.index
    last = {k: _last_day(k, idx) for k in C.columns}
    ks = sorted(C.columns)
    front = []
    for i in range(len(idx)):                              # 近月：最後交易日位置 > i+1（到期前一日收盤轉倉）
        front.append(next(k for k in ks if last[k] > i + 1))
    Cv, Ov = C.values, O.values
    col = {k: j for j, k in enumerate(C.columns)}
    n = len(idx)
    gap, intra, po, pc, roll = np.zeros(n), np.zeros(n), np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, bool)
    for i in range(n):
        pc[i] = Cv[i, col[front[i]]]
        if i == 0:
            continue
        j = col[front[i - 1]]
        c0, o1, c1 = Cv[i - 1, j], Ov[i, j], Cv[i, j]
        if not (o1 > 0):
            o1 = c0
        gap[i], intra[i], po[i] = o1 / c0 - 1, c1 / o1 - 1, o1
        roll[i] = front[i] != front[i - 1]
    out = pd.DataFrame(dict(gap=gap, intra=intra, po=po, pc=pc, roll=roll, front=front), index=idx)
    out["S"] = ((1 + out.gap) * (1 + out.intra)).cumprod()
    return out


def run_pos(F, n, d):
    """n, d：與 F 同 index 的部位陣列。回傳日報酬（已扣成本）與單邊交易量。"""
    n, d = np.asarray(n, float), np.asarray(d, float)
    cs_o = TAX + FEE + 1 / F.po.values
    cs_c = TAX + FEE + 1 / F.pc.values
    n_prev = np.r_[0.0, n[:-1]]
    ret = (1 + n_prev * F.gap.values) * (1 + d * F.intra.values) - 1
    t_open = np.abs(d - n_prev)
    t_close = np.where(F.roll.values, np.abs(d) + np.abs(n), np.abs(n - d))
    cost = np.nan_to_num(t_open * cs_o) + t_close * cs_c
    return pd.Series(ret - cost, index=F.index), pd.Series(t_open + t_close, index=F.index)


def from_signal(s):
    """收盤訊號 → (n, d)：d[i+1] = s[i]、n[i] = s[i-1]。"""
    s = pd.Series(s).fillna(0).values
    d = np.r_[0.0, s[:-1]]
    return d.copy(), d                                     # n[i] = d[i]（收盤不換倉，隔夜延續當日部位）


def month_last(idx):
    s = pd.Series(idx, index=idx)
    return s.groupby(s.dt.to_period("M")).transform("max").eq(s).values


def monthly(sig, idx):
    """只在每月最後一個交易日收盤更新訊號，其餘沿用。"""
    m = month_last(idx)
    return pd.Series(np.where(m, sig, np.nan), index=idx).ffill().fillna(0)


def rsi(x, k=2):
    dx = x.diff()
    up, dn = dx.clip(lower=0).ewm(alpha=1 / k, adjust=False).mean(), (-dx.clip(upper=0)).ewm(alpha=1 / k, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def strategy(h, F):
    S, idx = F.S, F.index
    ma200 = S.rolling(200).mean()
    if h == "BH":
        one = np.ones(len(idx))
        return one, one
    if h == "H47":
        return from_signal(np.where(ma200.isna(), 0, np.where(S > ma200, 1, -1)))
    if h == "H48":
        return from_signal(np.where(S > ma200, 1, 0))
    if h == "H49":
        return from_signal(monthly(np.where(S.shift(252).isna(), 0, np.where(S / S.shift(252) > 1, 1, -1)), idx))
    if h == "H50":
        r = S.pct_change()
        w = (r.pow(2).expanding(60).mean() / r.rolling(21).var()).clip(0, 2)
        return from_signal(monthly(w.values, idx))
    if h == "H51":
        return np.ones(len(idx)), np.zeros(len(idx))
    if h == "H52":
        ml = month_last(idx)
        pos = np.arange(len(idx))
        L = pos[ml]
        n, d = np.zeros(len(idx)), np.zeros(len(idx))
        for k in L:
            days = [k, k + 1, k + 2, k + 3]                 # 月底日 + 次月前 3 日
            days = [x for x in days if x < len(idx)]
            d[days] = 1
            n[days[:-1]] = 1                                # 第 3 日收盤平倉
        return n, d
    if h == "H53":
        hi20, lo20 = S.rolling(20).max().shift(1), S.rolling(20).min().shift(1)
        hi10, lo10 = S.rolling(10).max().shift(1), S.rolling(10).min().shift(1)
        s, p = np.zeros(len(idx)), 0
        for i in range(len(idx)):
            x = S.iloc[i]
            if p == 1 and x < lo10.iloc[i]:
                p = 0
            elif p == -1 and x > hi10.iloc[i]:
                p = 0
            if p == 0:
                p = 1 if x > hi20.iloc[i] else (-1 if x < lo20.iloc[i] else 0)
            s[i] = p
        return from_signal(s)
    if h == "H54":
        r2, ma5 = rsi(S), S.rolling(5).mean()
        s, p = np.zeros(len(idx)), 0
        for i in range(len(idx)):
            x = S.iloc[i]
            if p == 1 and x > ma5.iloc[i]:
                p = 0
            elif p == -1 and x < ma5.iloc[i]:
                p = 0
            if p == 0 and not np.isnan(ma200.iloc[i]):
                p = 1 if (x > ma200.iloc[i] and r2.iloc[i] < 10) else (-1 if (x < ma200.iloc[i] and r2.iloc[i] > 90) else 0)
            s[i] = p
        return from_signal(s)
    if h in ("H55", "H56"):
        lt = pd.read_parquet(RAW.parent / "TX_large_traders.parquet").assign(date=lambda x: pd.to_datetime(x.date))
        lt = lt.drop_duplicates("date").set_index("date").reindex(idx)       # 盤後公布 → 收盤 t 可用
        net = (lt.buy_top10_specific_open_interest - lt.sell_top10_specific_open_interest) / lt.market_open_interest
        up = net > net.rolling(20).mean()
        return from_signal(np.where(net.rolling(20).mean().isna(), 0, np.where(up, 1, -1 if h == "H55" else 0)))
    if h == "H57":
        oi = pd.read_parquet(RAW.parent / "TXO_oi.parquet")
        oi.index = pd.to_datetime(oi.index)
        pcr = (oi["put"] / oi["call"]).reindex(idx)
        ma = pcr.rolling(20).mean()
        return from_signal(np.where(ma.isna(), 0, np.where(pcr > ma, 1, -1)))
    if h == "H58":
        mo = idx.month
        d = np.where((mo >= 11) | (mo <= 4), 1.0, 0.0)                      # 日盤段：11~4 月
        n = d.copy()
        last_apr = month_last(idx) & (mo == 4)
        n[last_apr] = 0                                                      # 4 月最後交易日收盤平倉
        return n, d
    if h == "H59":
        r = S.pct_change()
        vol = r.rolling(20).std()
        w = (vol.expanding(60).median() / vol).clip(0, 2)
        k = np.arange(len(idx)) % 5 == 0
        return from_signal(pd.Series(np.where(k, w, np.nan), index=idx).ffill().fillna(0).values)
    raise KeyError(h)


NAMES = {"BH": "(a) MTX 連續做多", "H47": "H47 200日線多空", "H48": "H48 200日線多/空手", "H49": "H49 時間序列動能",
         "H50": "H50 波動管理做多", "H51": "H51 只抱夜間", "H52": "H52 月初效應", "H53": "H53 唐奇安突破多空", "H54": "H54 RSI(2)回歸多空",
         "H55": "H55 特定法人淨部位多空", "H56": "H56 特定法人淨部位多/空手", "H57": "H57 P/C未平倉比多空",
         "H58": "H58 萬聖節效應", "H59": "H59 波動目標做多(事後)"}


def metrics(r, bench=None):
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    eq = (1 + r).cumprod()
    out = dict(CAGR=eq.iloc[-1] ** (1 / yrs) - 1, MDD=(eq / eq.cummax() - 1).min(), Sharpe=r.mean() / r.std() * np.sqrt(252),
               years={str(k): v for k, v in r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1).items()})
    if bench is not None:                                   # Jobson-Korkie（Memmel 修正）Sharpe 差 z
        a, b = r.mean() / r.std(), bench.mean() / bench.std()
        rho, T = np.corrcoef(r, bench)[0, 1], len(r)
        out["JK_z"] = (a - b) / np.sqrt((2 - 2 * rho + 0.5 * (a * a + b * b - 2 * a * b * rho * rho)) / T)
    return out


def leak_check(h, end, cut="2015-06-30"):
    full, part = continuous(end), continuous(cut)
    nf, df_ = strategy(h, full)
    npart, dpart = strategy(h, part)
    k = len(part) - 1                                       # 最後一根的 n 在截斷資料下無法知道月底 → 只比到前一根
    return bool(np.allclose(np.asarray(nf)[:k], np.asarray(npart)[:k]) and np.allclose(np.asarray(df_)[:k + 1], np.asarray(dpart)[:k + 1]))


def fetch_aux():
    """大額交易人（TX all）與 TXO 每日 P/C 未平倉比 → data/v2/futures/。"""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from build_data_v2 import RateLimiter, api
    lim, lt, pcr = RateLimiter(80), [], []
    for m in pd.date_range("2009-01-01", pd.Timestamp.today(), freq="MS"):
        s, e = f"{m:%Y-%m-%d}", f"{m + pd.offsets.MonthEnd(0):%Y-%m-%d}"
        d = api(lim, dataset="TaiwanFuturesOpenInterestLargeTraders", data_id="TX", start_date=s, end_date=e)
        if d is not None and len(d):
            lt.append(d[d.contract_type == "all"])
        o = api(lim, dataset="TaiwanOptionDaily", data_id="TXO", start_date=s, end_date=e)
        if o is not None and len(o):
            o = o[(o.trading_session == "position") & o.contract_date.str.fullmatch(r"\d{6}")]
            g = o.groupby(["date", "call_put"]).open_interest.sum().unstack()
            pcr.append(g)
    pd.concat(lt, ignore_index=True).to_parquet(RAW.parent / "TX_large_traders.parquet")
    p = pd.concat(pcr)
    p.to_parquet(RAW.parent / "TXO_oi.parquet")
    print(len(lt), p.columns.tolist(), p.index.min(), p.index.max())


if __name__ == "__main__":
    if sys.argv[1] == "fetch":
        fetch()
        sys.exit()
    if sys.argv[1] == "fetch_aux":
        fetch_aux()
        sys.exit()
    period = sys.argv[1]
    unseal = "--unseal" in sys.argv
    if period == "holdout" and not unseal:
        raise PermissionError("holdout 未開封（需使用者同意，加 --unseal）")
    want = [a for a in sys.argv[2:] if not a.startswith("--")] or ["H47", "H48", "H49", "H50", "H51", "H52", "H53", "H54"]
    p0, p1 = H.PERIODS[period]
    F = continuous(p1)
    tr = pd.read_parquet(H.D / "taiex_tr.parquet").assign(date=lambda x: pd.to_datetime(x.date)).set_index("date").price
    rtr = tr.pct_change().loc[p0:p1]
    rb, _ = run_pos(F, *strategy("BH", F))
    rb = rb.loc[p0:p1]
    mb, mt = metrics(rb), metrics(rtr)
    bar = max(mb["Sharpe"], mt["Sharpe"])
    print(f"[{period}] 基準 (a) MTX 做多 Sharpe {mb['Sharpe']:.2f} CAGR {mb['CAGR']*100:+.1f}% MDD {mb['MDD']*100:.1f}% ｜ "
          f"(b) 加權報酬指數 Sharpe {mt['Sharpe']:.2f} CAGR {mt['CAGR']*100:+.1f}% MDD {mt['MDD']*100:.1f}% → 門檻 {bar:.2f}")
    for h in want:
        n, d = strategy(h, F)
        r, tv = run_pos(F, n, d)
        r, tv = r.loc[p0:p1], tv.loc[p0:p1]
        m = metrics(r, rb.reindex(r.index))
        yrs = (r.index[-1] - r.index[0]).days / 365.25
        expo = float(np.mean(np.abs(pd.Series(d, index=F.index).loc[p0:p1])))
        ok = m["Sharpe"] > bar
        lk = leak_check(h, p1) if period == "dev" else None
        print(f"{NAMES[h]:20s} Sharpe {m['Sharpe']:5.2f} {'✓' if ok else '✗'}  CAGR {m['CAGR']*100:+6.1f}%  MDD {m['MDD']*100:6.1f}%  "
              f"曝險 {expo:.2f}  年交易 {tv.sum()/yrs:5.1f} 邊  JK z {m['JK_z']:+.2f}" + (f"  前視檢查 {'OK' if lk else '✗✗'}" if lk is not None else ""))
        print("   " + "  ".join(f"{k}:{v*100:+.1f}" for k, v in m["years"].items()))
        with open(H.LOG, "a") as f:
            f.write(json.dumps(dict(ts=time.strftime("%Y-%m-%d %H:%M:%S"), kind="mtx", period=period, strategy=NAMES[h],
                                    note="小台指單一標的 第十/十一批", bar=bar, passed=bool(ok), leak_ok=lk,
                                    **{k: v for k, v in m.items() if k != "years"}, years=m["years"]),
                               ensure_ascii=False, default=float) + "\n")

