"""Polymarket 美股每日 Up/Down 市場 是否為領先指標？

核心問題：同一時點 T，Polymarket 的 P(Up) 扣掉「標的當下價格已隱含的機率」後，
是否還能預測 (1) 收盤漲跌、(2) T→收盤的剩餘報酬。

- 標的價格 S_T：個股用 yfinance 1h（含盤前盤後）；指數盤前用期貨（ES/NQ/RTY/YM）換算，盤中用指數本身
- Polymarket 價格取 T−2 分鐘（保守：不讓 PM 偷看到 S_T 那根 K 之後的資訊），需 30 分鐘內有報價
- 價格隱含 z = ln(S_T/C0) / (σ20·√τ)，τ = 剩餘交易時間比例（盤前=1）
- 檢定：logit(Up) ~ z + logit(p_PM)，按日期 cluster SE；剩餘報酬 ~ PM 殘差

資料：research/polymarket_fetch.py → data/polymarket/
"""
import glob, json, os
from datetime import datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

D = "data/polymarket"
ET = ZoneInfo("America/New_York")
STOCKS = "MSFT AMZN GOOGL META TSLA NFLX AAPL NVDA PLTR OPEN COIN HOOD ABNB RKLB MU SPY EWY QQQ".split()
INDEX = {"SPX": ("^GSPC", "ES=F"), "NYA": ("^NYA", "ES=F"), "NDX": ("^NDX", "NQ=F"),
         "RUT": ("^RUT", "RTY=F"), "DJIA": ("^DJI", "YM=F"), "DJI": ("^DJI", "YM=F")}
# (名稱, 相對 D 的天數, ET 時刻)
SNAPS = [("prev20", -1, dtime(20, 0)), ("pre0900", 0, dtime(9, 0)), ("1030", 0, dtime(10, 30)),
         ("1230", 0, dtime(12, 30)), ("1430", 0, dtime(14, 30)), ("1530", 0, dtime(15, 30))]
OPEN_T, CLOSE_T = dtime(9, 30), dtime(16, 0)


def fn(sym):
    return sym.replace("^", "_").replace("=", "_")


_cache = {}


def hourly(sym):
    """回傳 (bar_end_utc ndarray[int64 秒], close ndarray)。bar_end = min(start+1h, 下一根 start)。"""
    if sym not in _cache:
        h = pd.read_csv(f"{D}/yf/{fn(sym)}_1h.csv", index_col=0, parse_dates=True).dropna()
        st = ((h.index - pd.Timestamp(0, tz="UTC")) // pd.Timedelta("1s")).values.astype("int64")
        end = np.minimum(st + 3600, np.append(st[1:], st[-1] + 3600))
        _cache[sym] = (end, h["Close"].values, st, h["High"].values, h["Low"].values)
    return _cache[sym]


def daily(sym):
    k = sym + "_d"
    if k not in _cache:
        d = pd.read_csv(f"{D}/yf/{fn(sym)}_1d.csv", index_col=0, parse_dates=True)["Close"].dropna()
        d.index = d.index.date
        _cache[k] = d
    return _cache[k]


def px_at(sym, ts):
    end, c = hourly(sym)[:2]
    i = np.searchsorted(end, ts, side="right") - 1
    if i < 0 or ts - end[i] > 4 * 3600:
        return np.nan
    return c[i]


def range_since(sym, t0, t1):
    """t0(開盤)~t1 之間已收完的 1h K 的 ln(最高/最低)。"""
    end, _, st, hi, lo = hourly(sym)
    m = (st >= t0) & (end <= t1)
    return np.log(hi[m].max() / lo[m].min()) if m.any() else np.nan


def pm_at(hist, ts, stale=1800):
    if not hist:
        return np.nan
    t = np.array([x["t"] for x in hist]); p = np.array([x["p"] for x in hist])
    i = np.searchsorted(t, ts - 120, side="right") - 1
    if i < 0 or ts - 120 - t[i] > stale:
        return np.nan
    return p[i]


def build():
    evs = json.load(open(f"{D}/events.json"))
    rows = []
    for e in evs:
        tk = e["ticker"]
        if tk in STOCKS:
            dsym, intra, pre = tk, tk, tk
        elif tk in INDEX:
            dsym, fut = INDEX[tk]; intra, pre = dsym, fut
        else:
            continue
        p = f"{D}/prices/{e['slug']}.json"
        if not os.path.exists(p):
            continue
        hist = json.load(open(p))
        day = datetime.fromisoformat(e["end"].replace("Z", "+00:00")).astimezone(ET).date()
        dc = daily(dsym)
        if day not in dc.index:
            continue
        loc = dc.index.get_loc(day)
        if loc < 21:
            continue
        c0, c1 = dc.iloc[loc - 1], dc.iloc[loc]
        prev_day = dc.index[loc - 1]
        r = np.diff(np.log(dc.iloc[loc - 21:loc].values))
        sig = r.std(ddof=1)
        yf_up = int(c1 > c0)
        for name, off, tt in SNAPS:
            dd = prev_day if off == -1 else day
            T = datetime.combine(dd, tt, ET)
            ts = int(T.timestamp())
            pm = pm_at(hist, ts)
            if np.isnan(pm):
                continue
            if off == 0 and tt >= OPEN_T or tk in STOCKS:
                s = px_at(intra, ts)
            else:  # 指數盤前：期貨相對前日 16:00 的變動 × C0
                f0 = px_at(pre, int(datetime.combine(prev_day, CLOSE_T, ET).timestamp()))
                s = c0 * px_at(pre, ts) / f0
            if not np.isfinite(s):
                continue
            tau = 1.0 if (off == -1 or tt < OPEN_T) else \
                (datetime.combine(day, CLOSE_T) - datetime.combine(day, tt)).seconds / 23400
            # 當日波動調整：Parkinson 區間估計的今日 σ 與 σ20 等權混合
            z2 = np.nan
            if tau < 1:
                t_open = int(datetime.combine(day, OPEN_T, ET).timestamp())
                rg = range_since(intra, t_open, ts)
                if np.isfinite(rg) and rg > 0:
                    sig_today = rg / np.sqrt(4 * np.log(2)) / np.sqrt(1 - tau)
                    sig_mix = np.sqrt(0.5 * sig**2 + 0.5 * sig_today**2)
                    z2 = np.log(s / c0) / (sig_mix * np.sqrt(tau))
            rows.append(dict(ticker=tk, date=day, snap=name, pm=pm, up=e["outcome"], yf_up=yf_up, z2=z2,
                             z=np.log(s / c0) / (sig * np.sqrt(tau)), rem=np.log(c1 / s) / (sig * np.sqrt(tau)),
                             vol=e["volume"], kind="index" if tk in INDEX else "stock"))
    return pd.DataFrame(rows)


def logit(p):
    p = np.clip(p, 0.01, 0.99)
    return np.log(p / (1 - p))


def test(df, label):
    out = []
    for snap, g in df.groupby("snap", sort=False):
        if len(g) < 80 or g.date.nunique() < 30:
            continue
        g = g.copy(); g["L"] = logit(g.pm)
        grp = pd.factorize(g.date)[0]
        # 1) 收盤方向：Up ~ z + L
        X = sm.add_constant(g[["z", "L"]])
        try:
            m = sm.Logit(g.up, X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": grp})
        except Exception:
            continue
        # 1b) 加控當日波動 z2 與非線性 z^3：PM 係數是否還顯著
        b_PM2 = t_PM2 = np.nan
        if g.z2.notna().mean() > 0.9:
            h = g.dropna(subset=["z2"]).copy(); h["z3"] = h.z ** 3
            m2 = sm.Logit(h.up, sm.add_constant(h[["z", "z2", "z3", "L"]])).fit(
                disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(h.date)[0]})
            b_PM2, t_PM2 = m2.params["L"], m2.tvalues["L"]
        # 2) 剩餘報酬 ~ PM 殘差（L 對 z 迴歸後的殘差）
        res = g.L - sm.OLS(g.L, sm.add_constant(g.z)).fit().fittedvalues
        o = sm.OLS(g.rem, sm.add_constant(res)).fit(cov_type="cluster", cov_kwds={"groups": grp})
        hit = (np.sign(res) == np.sign(g.rem))[res.abs() > 0.2].mean()
        # 2b) 方向檢定：1{剩餘報酬>0} ~ PM 殘差（純方向資訊，不受波動校準影響）
        md = sm.Logit((g.rem > 0).astype(int), sm.add_constant(res)).fit(
            disp=0, cov_type="cluster", cov_kwds={"groups": grp})
        dir_t = md.tvalues.iloc[1]
        # 3) 時間切半 OOS：z-only vs z+PM 的 log-loss
        dates = np.sort(g.date.unique()); cut = dates[len(dates) // 2]
        tr, te = g[g.date < cut], g[g.date >= cut]
        def ll(cols):
            mm = sm.Logit(tr.up, sm.add_constant(tr[cols])).fit(disp=0)
            pr = np.clip(mm.predict(sm.add_constant(te[cols])), 1e-6, 1 - 1e-6)
            return -np.mean(te.up * np.log(pr) + (1 - te.up) * np.log(1 - pr))
        try:
            ll_z, ll_zl, ll_l = ll(["z"]), ll(["z", "L"]), ll(["L"])
        except Exception:
            ll_z = ll_zl = ll_l = np.nan
        out.append(dict(group=label, snap=snap, n=len(g), days=g.date.nunique(),
                        b_z=m.params["z"], b_PM=m.params["L"], t_PM=m.tvalues["L"], t_PM_ctrl=t_PM2,
                        rem_b=o.params.iloc[1], rem_t=o.tvalues.iloc[1], dir_t=dir_t, hit=hit,
                        oos_ll_z=ll_z, oos_ll_zPM=ll_zl, oos_ll_PM=ll_l,
                        corr_L_z=np.corrcoef(g.L, g.z)[0, 1]))
    return out


def main():
    df = build()
    df.to_csv(f"{D}/snapshots.csv", index=False)
    agree = (df.drop_duplicates(["ticker", "date"]).eval("up == yf_up")).mean()
    print(f"快照列數 {len(df)}，市場-日 {df[['ticker','date']].drop_duplicates().shape[0]}，PM 結算 vs yfinance 一致率 {agree:.3f}")
    df = df[df.up == df.yf_up]
    res = []
    res += test(df[df.ticker == "SPX"], "SPX")
    res += test(df[df.kind == "index"], "US指數")
    res += test(df[df.kind == "stock"], "個股+ETF")
    res += test(df[(df.kind == "stock") & (df.vol >= df[df.kind == "stock"].vol.median())], "個股(高量半)")
    r = pd.DataFrame(res)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    print(r.round(3).to_string(index=False))
    r.to_csv(f"{D}/lead_results.csv", index=False)


if __name__ == "__main__":
    main()
