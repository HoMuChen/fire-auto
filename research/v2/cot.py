"""第三十七批：外匯課程「美國政府每週公布的答案」= CFTC COT「大戶指標」＋「波動指標」停利（H138 / H139）。
規則與過關標準見 HYPOTHESES.md 第三十七批。

訊號：legacy 期貨報告「非商業（大型投機者）淨部位」週變化的正負。週二持倉、週五 15:30 ET 公布 → 下週一開盤進場；
     公布週有美國聯邦假日（延到下週一才公布）或政府關門期間 → 該週不交易。
停利：上週最後收盤 ± V，V = min(近 2/4/8/10/12 根 K 棒「高 − 低」平均)；停損：進場價 ∓ V（1:1）；都沒碰到 → 該週最後收盤出場。
同一根 K 棒同時碰到停利與停損 → 算停損（保守）；開盤跳空越過 → 以開盤價成交。
H138 用日 K 算 V、H139 用週 K 算 V；對照 M＝同一套進出場、方向改用上週漲跌（價格動能）。

用法：.venv/bin/python research/v2/cot.py fetch [cot|px] | dev | val | holdout --unseal
資料：data/v2/cot/legacy_fut.parquet（CFTC Socrata 6dca-aqww）、px_*.parquet（Yahoo 日 K）。
"""
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402

D = H.D / "cot"
# 名稱, COT 代碼, 價格代號, 價格方向（COT 看多外幣 = 做空 USD/xxx → −1）, 類別
MK = [
    ("EUR", "099741", "EURUSD=X", 1, "外匯"), ("JPY", "097741", "USDJPY=X", -1, "外匯"),
    ("GBP", "096742", "GBPUSD=X", 1, "外匯"), ("AUD", "232741", "AUDUSD=X", 1, "外匯"),
    ("CAD", "090741", "USDCAD=X", -1, "外匯"), ("CHF", "092741", "USDCHF=X", -1, "外匯"),
    ("NZD", "112741", "NZDUSD=X", 1, "外匯"), ("MXN", "095741", "USDMXN=X", -1, "外匯"),
    ("黃金", "088691", "GLD", 1, "金屬"), ("白銀", "084691", "SLV", 1, "金屬"),
    ("原油", "067651", "CL=F", 1, "能源"),
    ("S&P500", "13874A", "ES=F", 1, "指數"), ("Nasdaq100", "209742", "NQ=F", 1, "指數"), ("道瓊", "124603", "YM=F", 1, "指數"),
]
NS = (2, 4, 8, 10, 12)
COST = 0.0005                                      # 每邊（名目），來回 0.1%
SKIP = [("2013-10-07", "2013-11-29"), ("2018-12-31", "2019-03-08"), ("2025-10-06", "2026-01-30")]   # 政府關門、補發期間


# ---------------------------------------------------------------- 資料
def fetch(what=("cot", "px"), force=False):
    D.mkdir(parents=True, exist_ok=True)
    if "cot" in what:
        codes = ",".join(f"'{m[1]}'" for m in MK)
        w = f"cftc_contract_market_code in({codes}) AND report_date_as_yyyy_mm_dd between '2008-01-01' and '2099-12-31'"
        u = "https://publicreporting.cftc.gov/resource/6dca-aqww.json?" + urllib.parse.urlencode(
            {"$where": w, "$limit": 50000, "$order": "report_date_as_yyyy_mm_dd"})
        df = pd.DataFrame(json.load(urllib.request.urlopen(u, timeout=300)))
        df.to_parquet(D / "legacy_fut.parquet")
        print("COT", df.shape, df.report_date_as_yyyy_mm_dd.max()[:10])
    if "px" in what:
        p1, p2 = int(pd.Timestamp("2007-06-01").timestamp()), int(time.time())
        for m in MK:
            f = D / f"px_{m[2].replace('=', '_')}.parquet"
            if f.exists() and not force:
                continue
            u = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(m[2])}?period1={p1}&period2={p2}&interval=1d"
            j = json.load(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}), timeout=60))
            j = j["chart"]["result"][0]
            q, tz = j["indicators"]["quote"][0], j["meta"]["exchangeTimezoneName"]
            df = pd.DataFrame({"ts": j["timestamp"], **{k: q[k] for k in ("open", "high", "low", "close")}})
            df["date"] = pd.to_datetime(df.ts, unit="s", utc=True).dt.tz_convert(tz).dt.tz_localize(None).dt.normalize()
            df = df.dropna(subset=["open", "high", "low", "close"]).drop(columns="ts").drop_duplicates("date", keep="last")
            df.to_parquet(f)
            print(m[2], len(df), df.date.min().date(), df.date.max().date())
            time.sleep(1)


def load_px(sym, end):
    df = pd.read_parquet(D / f"px_{sym.replace('=', '_')}.parquet")
    df = df[df.date <= end].sort_values("date").reset_index(drop=True)
    df["high"] = df[["high", "open", "close"]].max(axis=1)
    df["low"] = df[["low", "open", "close"]].min(axis=1)
    d2 = df.date + pd.Timedelta(days=1)                                   # 週日晚盤歸到下週一
    df["wk"] = (d2 - pd.to_timedelta(d2.dt.weekday, unit="D")).dt.normalize()
    return df


def load_cot(end):
    """回傳 {code: Series(index=可進場的週一, value=淨部位週變化)}；只用 end 之前已公布的報告。"""
    c = pd.read_parquet(D / "legacy_fut.parquet")
    c["T"] = pd.to_datetime(c.report_date_as_yyyy_mm_dd).dt.normalize()
    c["d"] = pd.to_numeric(c.change_in_noncomm_long_all, errors="coerce") - pd.to_numeric(c.change_in_noncomm_short_all, errors="coerce")
    lv = pd.to_numeric(c.noncomm_positions_long_all, errors="coerce") - pd.to_numeric(c.noncomm_positions_short_all, errors="coerce")
    c["d"] = c.d.fillna(lv - lv.groupby(c.cftc_contract_market_code).shift())
    c["rel"] = c["T"] + pd.to_timedelta(4 - c["T"].dt.weekday, unit="D")   # 該週週五公布
    c["M"] = c.rel + pd.Timedelta(days=3)                                  # 下週一進場
    hol = USFederalHolidayCalendar().holidays("2007-01-01", "2030-12-31")
    mon = c["T"] - pd.to_timedelta(c["T"].dt.weekday, unit="D")
    late = np.array([((hol >= a) & (hol <= b)).any() for a, b in zip(mon, c.rel)])
    for a, b in SKIP:
        late |= ((c.M >= a) & (c.M <= b)).to_numpy()
    c = c[~late & (c.rel + pd.Timedelta(hours=15.5) <= pd.Timestamp(end) + pd.Timedelta(hours=23))]
    return {k: g.set_index("M").d.sort_index() for k, g in c.groupby("cftc_contract_market_code")}


# ---------------------------------------------------------------- 計畫（每週方向、基準價、V）
def plans(m, end, P=None, C=None):
    """回傳 DataFrame index=週一，欄：C0（上週收盤）、mom（上週漲跌方向）、cot（COT 方向，已換成價格方向）、Vd、Vw。"""
    name, code, sym, sgn, _ = m
    px = load_px(sym, end) if P is None else P
    rng = px.high - px.low
    vd = pd.concat([rng.rolling(n).mean() for n in NS], axis=1).min(axis=1, skipna=False)
    W = px.groupby("wk").agg(close=("close", "last"), high=("high", "max"), low=("low", "min"), last_i=("close", lambda s: s.index[-1]))
    wr = W.high - W.low
    vw = pd.concat([wr.rolling(n).mean() for n in NS], axis=1).min(axis=1, skipna=False)
    out = pd.DataFrame(index=W.index[1:])
    prev = W.index[:-1]
    out["C0"] = W.close.values[:-1]
    out["mom"] = np.sign(W.close.diff().values[:-1])
    out["Vd"] = vd.reindex(W.last_i.values[:-1]).values
    out["Vw"] = vw.values[:-1]
    out["prev_wk"] = prev
    cot = (load_cot(end) if C is None else C).get(code, pd.Series(dtype=float))
    out["cot"] = np.sign(cot.reindex(out.index)) * sgn
    return out


def simulate(m, end, vcol, sigcol, P=None, C=None):
    px = load_px(m[2], end) if P is None else P
    pl = plans(m, end, px, C)
    g = {k: v for k, v in px.groupby("wk")}
    rows = []
    for wk, r in pl.iterrows():
        s, V, C0 = r[sigcol], r[vcol], r.C0
        if wk not in g or not np.isfinite(s) or s == 0 or not np.isfinite(V) or V <= 0:
            continue
        b = g[wk]
        if C0 <= 0 or (b.low <= 0).any():                                 # 原油負價格週：資料防護
            continue
        e = b.open.iloc[0]
        tp, sl = C0 + s * V, e - s * V
        x, why = None, "time"
        if s * (e - tp) >= 0:
            x, why = e, "tp_gap"
        else:
            for i, (o, h, lo, c) in enumerate(zip(b.open, b.high, b.low, b.close)):
                if i > 0 and s * (o - sl) <= 0:
                    x, why = o, "sl_gap"; break
                if i > 0 and s * (o - tp) >= 0:
                    x, why = o, "tp_gap"; break
                hit_sl = (lo <= sl) if s > 0 else (h >= sl)
                hit_tp = (h >= tp) if s > 0 else (lo <= tp)
                if hit_sl:
                    x, why = sl, "sl"; break
                if hit_tp:
                    x, why = tp, "tp"; break
            if x is None:
                x = b.close.iloc[-1]
        gross = s * (x / e - 1)
        rows.append(dict(mk=m[0], cls=m[4], wk=wk, s=s, e=e, x=x, why=why, V=V / C0, gross=gross,
                         net=gross - 2 * COST, dir_ret=s * (b.close.iloc[-1] / e - 1) - 2 * COST,
                         long_ret=b.close.iloc[-1] / e - 1 - 2 * COST))
    return pd.DataFrame(rows)


def run_all(end, vcol, sigcol):
    C = load_cot(end)
    return pd.concat([simulate(m, end, vcol, sigcol, C=C) for m in MK], ignore_index=True)


# ---------------------------------------------------------------- 統計
def weekly(T, col="net"):
    return T.groupby("wk")[col].mean()


def metrics(r):
    r = r.dropna()
    eq = (1 + r).cumprod()
    yrs = r.groupby(r.index.year).sum()
    return dict(ann=r.mean() * 52, t=r.mean() / r.std() * np.sqrt(len(r)), Sharpe=r.mean() / r.std() * np.sqrt(52),
                MDD=float((eq / eq.cummax() - 1).min()), weeks=len(r), yrs_pos=f"{(yrs > 0).sum()}/{len(yrs)}",
                years={int(k): float(v) for k, v in yrs.items()})


def leak_check(vcol, sigcol, end, cuts=(0.4, 0.7, 0.9)):
    """截斷價格與 COT 到 cut 日重算計畫，cut 之前的週必須完全一致。"""
    p0 = pd.Timestamp(H.PERIODS["dev"][0])
    ok = True
    for m in MK:
        full = plans(m, end)
        for f in cuts:
            cut = p0 + (pd.Timestamp(end) - p0) * f
            part = plans(m, cut)
            idx = part.index[part.index <= cut]
            a, b = full.loc[idx, ["C0", "mom", "cot", vcol]], part.loc[idx, ["C0", "mom", "cot", vcol]]
            if not np.allclose(a.fillna(-9e9).to_numpy(float), b.fillna(-9e9).to_numpy(float)):
                ok = False
                print("  ✗✗ 前視", m[0], cut.date())
    return ok


def report(period, unseal=False):
    if period == "holdout" and not unseal:
        raise PermissionError("holdout 未開封（需使用者同意，加 --unseal）")
    p0, p1 = H.PERIODS[period]
    end = min(pd.Timestamp(p1), pd.Timestamp.today().normalize())
    res = {}
    for hid, vcol in (("H138", "Vd"), ("H139", "Vw")):
        T = run_all(end, vcol, "cot")
        Mo = run_all(end, vcol, "mom")
        T, Mo = T[(T.wk >= p0) & (T.wk <= p1)], Mo[(Mo.wk >= p0) & (Mo.wk <= p1)]
        r, rm = weekly(T), weekly(Mo)
        mt, mm = metrics(r), metrics(rm)
        diff = (r.reindex(r.index.union(rm.index)).fillna(0) - rm.reindex(r.index.union(rm.index)).fillna(0))
        dt = diff.mean() / diff.std() * np.sqrt(len(diff))
        need = int(np.ceil(len(mt["years"]) * 0.7))
        ok = mt["ann"] > 0 and mt["t"] >= 2 and int(mt["yrs_pos"].split("/")[0]) >= need
        verdict = ("✓" if diff.mean() > 0 else "✗（只是動能）") if ok else "✗"
        lk = leak_check(vcol, "cot", end) if period == "dev" else None
        why = T.why.value_counts(normalize=True)
        print(f"\n=== [{period}] {hid}（{'日' if vcol == 'Vd' else '週'} K 算 V）  {p0}~{end.date()} ===")
        print(f"COT 大戶指標：年化 {mt['ann']*100:+.1f}%  t {mt['t']:+.2f}  Sharpe {mt['Sharpe']:+.2f}  MDD {mt['MDD']*100:.1f}%  "
              f"年正 {mt['yrs_pos']}  交易 {len(T)} 筆 / {mt['weeks']} 週 → {verdict}" + (f"  前視檢查 {'OK' if lk else '✗✗'}" if lk is not None else ""))
        print(f"   停利 {why.get('tp', 0)+why.get('tp_gap', 0):.1%}（跳空 {why.get('tp_gap', 0):.1%}）  停損 {why.get('sl', 0)+why.get('sl_gap', 0):.1%}  "
              f"到期 {why.get('time', 0):.1%}  勝率（淨 > 0）{(T.net > 0).mean():.1%}  每筆淨 {T.net.mean()*100:+.3f}%  毛 {T.gross.mean()*100:+.3f}%  "
              f"V 中位數 {T.V.median()*100:.2f}%")
        print(f"對照 M 價格動能：年化 {mm['ann']*100:+.1f}%  t {mm['t']:+.2f}  年正 {mm['yrs_pos']}  勝率 {(Mo.net > 0).mean():.1%}  ｜ COT − M 週差 t {dt:+.2f}")
        print("   逐年 COT：" + "  ".join(f"{k}:{v*100:+.1f}" for k, v in mt["years"].items()))
        if hid == "H138":
            cls = T.groupby(["cls", "wk"]).net.mean().groupby("cls").agg(lambda s: (s.mean() * 52, s.mean() / s.std() * np.sqrt(len(s))))
            print("   分類（描述）：" + "  ".join(f"{k} 年化 {v[0]*100:+.1f}% t {v[1]:+.2f}" for k, v in cls.items()))
            mk = T.groupby("mk").net.agg(["mean", "count"])
            print("   各標的每筆淨（描述）：" + "  ".join(f"{k}:{v['mean']*100:+.2f}%" for k, v in mk.iterrows()))
            agree = (T.merge(Mo[["mk", "wk", "s"]], on=["mk", "wk"], suffixes=("", "_m")).eval("s == s_m")).mean()
            print(f"   COT 方向與上週漲跌方向一致比例 {agree:.1%}")
            for lab, col, X in (("COT 方向", "dir_ret", T), ("上週漲跌方向", "dir_ret", Mo), ("一律做多", "long_ret", T)):
                mw = metrics(weekly(X, col))
                print(f"   純方向（週一開→週五收，扣成本）{lab}：年化 {mw['ann']*100:+.1f}%  t {mw['t']:+.2f}  年正 {mw['yrs_pos']}  勝率 {(X[col] > 0).mean():.1%}")
        res[hid] = dict(ok=ok, verdict=verdict, **{k: v for k, v in mt.items() if k != "years"}, years=mt["years"],
                        mom=dict(ann=mm["ann"], t=mm["t"]), diff_t=dt, n=len(T), win=float((T.net > 0).mean()), leak_ok=lk)
        with open(H.LOG, "a") as f:
            f.write(json.dumps(dict(ts=time.strftime("%Y-%m-%d %H:%M:%S"), kind="cot", period=period, strategy=hid,
                                    note="第三十七批 COT 大戶指標 + 波動指標停利（外匯課程）", passed=verdict == "✓", **res[hid]),
                               ensure_ascii=False, default=float) + "\n")
    return res


if __name__ == "__main__":
    a = sys.argv[1]
    if a == "fetch":
        fetch(tuple(sys.argv[2:]) or ("cot", "px"), force="--force" in sys.argv)
    else:
        report(a, unseal="--unseal" in sys.argv)
