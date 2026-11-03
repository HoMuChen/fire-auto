"""第四十五批 FS7：新資料（個股融資維持率、外資持股、當沖）做成的特徵（FEATURE_PLAN 階段 2）。規則見 HYPOTHESES.md 第四十五批。

特徵函數 FNS[代號] = (fn(P), 說明)，介面同 hypotheses.FNS：回傳日頻分數（日期 × 股票），評估時取換股日前一交易日的值。
先呼叫 add_fs7(P)（需已跑 xs.add_chips）載入 data/v2/maint、fhold、dt。評估併入 feature_library.py。
追繳事件統計（描述用）：.venv/bin/python research/v2/feature_fs7.py --events

leak check：.venv/bin/python research/v2/feature_fs7.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402
import xs  # noqa: E402

M_MIN, LOW_WIN, LOW_MIN, CHG_WIN, DT_WIN, DT_S, DT_L = 100, 20, 15, 20, 20, 5, 60
CALL, EV_GAP, EV_H = 130.0, 20, (5, 20, 60)
PER = {"A": ("2010-01-01", "2014-12-31"), "B": ("2015-01-01", "2019-12-31"), "V": ("2020-01-01", "2023-12-31")}


def _series(kind, col, cols):
    out = {}
    for s in cols:
        f = H.D / kind / f"{s}.parquet"
        if f.exists():
            d = pd.read_parquet(f)
            if len(d) and col in d:
                out[s] = d.assign(date=pd.to_datetime(d.date)).drop_duplicates("date", keep="last").set_index("date")[col].astype(float)
    return out


def add_fs7(P):
    idx, cols = P["close"].index, P["close"].columns
    have = P["close"].notna()
    for k, kind, col in (("maint", "maint", "margin_maintenance"), ("fhold", "fhold", "ForeignInvestmentSharesRatio"),
                         ("flim", "fhold", "ForeignInvestmentUpperLimitRatio"), ("dt_vol", "dt", "Volume")):
        P[k] = pd.DataFrame(_series(kind, col, cols)).reindex(index=idx, columns=cols).where(have)
    after = P["dt_vol"].notna().cummax()                               # 該檔第一筆當沖紀錄之後，沒有列＝當日沒有當沖
    P["dt_vol"] = P["dt_vol"].mask(after & P["dt_vol"].isna() & have, 0)


def maint(P):
    return P["maint"].where(P["margin_bal"] >= M_MIN)


def maint_low(P):
    return maint(P).rolling(LOW_WIN, min_periods=LOW_MIN).min()


def fhold(P):
    return P["fhold"]


def fhold_chg(P):
    return P["fhold"] - P["fhold"].shift(CHG_WIN)


def froom(P):
    return P["flim"] - P["fhold"]


def _dt(P, w):
    v = P["volume"].where(P["dt_vol"].notna())
    den = v.rolling(w, min_periods=w).sum()
    return P["dt_vol"].rolling(w, min_periods=w).sum() / den.where(den > 0)


def dt_ratio(P):
    return _dt(P, DT_WIN)


def dt_dev(P):
    return _dt(P, DT_S) - _dt(P, DT_L)


FNS = {"MAINT": (maint, "融資維持率"), "MAINT_LOW": (maint_low, "近20日最低維持率"),
       "FHOLD": (fhold, "外資持股比率"), "FHOLD_CHG": (fhold_chg, "外資持股20日變化"), "FROOM": (froom, "外資投資空間"),
       "DT_RATIO": (dt_ratio, "當沖比率"), "DT_DEV": (dt_dev, "當沖比率偏離")}
NAMES = list(FNS)
CAT = {"MAINT": "籌碼", "MAINT_LOW": "籌碼", "FHOLD": "籌碼", "FHOLD_CHG": "籌碼", "FROOM": "籌碼",
       "DT_RATIO": "籌碼", "DT_DEV": "籌碼"}


def call_events(P):
    """追繳事件：融資 ≥ 100 張、前一日維持率 ≥ 130%、當日 < 130%；同檔 20 個交易日內只算第一次。回傳事件日 × 股票的 bool。"""
    m = maint(P)
    hit = ((m < CALL) & (m.shift(1) >= CALL)).values
    ev = np.zeros_like(hit)
    for j in range(hit.shape[1]):
        last = -EV_GAP
        for i in np.flatnonzero(hit[:, j]):
            if i - last >= EV_GAP:
                ev[i, j], last = True, i
    return pd.DataFrame(ev, index=m.index, columns=m.columns)


def event_study(P):
    """事件日隔日開盤買、持有 h 個交易日（開盤到開盤，下市以最後收盤），減同期 pit 股票池等權平均。"""
    O, C = P["open"], P["close"]
    px = O.where(O.notna(), C.ffill())                                  # 停牌或下市後：沿用最後收盤
    elig = (P["liq"].shift(1).fillna(False).astype(bool) & (P["nbars"].shift(1) >= 250) & O.notna())
    ev = call_events(P).shift(1, fill_value=False)                      # 列＝進場日
    rows = []
    for h in EV_H:
        R = px.shift(-h) / O - 1
        ex = R.sub(R.where(elig).mean(axis=1), axis=0)
        e = ev & elig & ex.notna()
        s = ex.where(e).stack().dropna()
        for p, (a, b) in PER.items():
            x = s.loc[a:b]
            if not len(x):
                continue
            mo = x.groupby(x.index.get_level_values(0).to_period("M")).mean()
            rows.append({"h": h, "period": p, "n": len(x), "months": len(mo), "mean": x.mean(), "month_mean": mo.mean(), "median": x.median(),
                         "t": mo.mean() / mo.std(ddof=1) * np.sqrt(len(mo)) if len(mo) > 1 else np.nan,
                         "win": (x > 0).mean()})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    end = "2023-12-31" if "--events" in sys.argv else "2019-12-31"
    P = xs.load_panel(end)
    xs.add_chips(P)
    add_fs7(P)
    if "--events" in sys.argv:
        r = event_study(P)
        print(r.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
        out = H.D / "results" / "fs7_call_events.csv"
        r.to_csv(out, index=False)
        print("→", out)
        sys.exit()
    for f in FNS:
        print(f, "leak_check_xs:", xs.leak_check_xs(FNS[f][0], P) or "OK", flush=True)
    for f in FNS:
        v = FNS[f][0](P).replace([np.inf, -np.inf], np.nan)
        y = v.loc["2018"]
        ok = v.notna().sum(axis=1) >= 100
        print(f"{f:<10} 有值檔數（2018 中位數）{int(y.notna().sum(axis=1).median())}  首個 ≥100 檔 "
              f"{ok.idxmax().date() if ok.any() else '—'}  2018 分位 p1/p50/p99 "
              f"{np.nanpercentile(y, 1):.4g}/{np.nanpercentile(y, 50):.4g}/{np.nanpercentile(y, 99):.4g}  "
              f"0 的比例 {float((v == 0).sum().sum() / max(v.notna().sum().sum(), 1)):.0%}")
