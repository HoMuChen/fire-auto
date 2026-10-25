"""H106：QFF 30 分 K「快速推升第一個低點 / 快速下跌第一個高點」回測進場（影片口訣量化版）。
規則見 research/v2/HYPOTHESES.md 第二十批。固定 1 口，多空各自獨立跑，空方以價格取負數套同一套邏輯。
用法：python3 research/qff_pullback.py            # 主規則 + 兩個預先宣告變體 + 對照
      python3 research/qff_pullback.py --sens     # 推升門檻 2/3/4 ATR 穩健性（看、不挑）
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
CSV = BASE / "data" / "futures" / "QFF_30min_day.csv"
LOG = BASE / "research" / "v2" / "experiments.jsonl"
MULT = 100
FEE = 20.0          # 元/口/邊
TAX = 0.00002       # 期交稅/邊
ROLL_PTS = 2.0      # 轉倉成本 點/口
ATR_N, IMP_N, IMP_K = 14, 10, 3.0
AWAY, RETEST, INVAL, STOP = 1.0, 0.25, 0.25, 0.5
EXPIRE, FIND_LOW = 100, 50


def tick(price):
    if "--noslip" in sys.argv:
        return 0.0
    p = abs(price)
    return 5.0 if p >= 1000 else 1.0 if p >= 500 else 0.5 if p >= 100 else 0.1


def load():
    df = pd.read_csv(CSV, parse_dates=["datetime"])
    con, close, n = df.contract_date.values, df.close.values, len(df)
    f, factor = np.ones(n), 1.0
    for i in range(n - 1, -1, -1):
        if i < n - 1 and con[i] != con[i + 1] and close[i] > 0:
            factor *= close[i + 1] / close[i]
        f[i] = factor
    for c in ("open", "high", "low", "close"):
        df[c + "_adj"] = df[c] * f
    df["roll"] = np.r_[False, con[1:] != con[:-1]]
    pc = df.close_adj.shift()
    tr = pd.concat([df.high_adj - df.low_adj, (df.high_adj - pc).abs(), (df.low_adj - pc).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / ATR_N, adjust=False).mean()
    df.loc[: ATR_N, "atr"] = np.nan
    df["date"] = df.datetime.dt.strftime("%Y-%m-%d")
    return df


def fractal_low(lo, k):
    return k >= 2 and k + 2 < len(lo) and lo[k] < min(lo[k - 2], lo[k - 1]) and lo[k] < min(lo[k + 1], lo[k + 2])


def simulate(df, sign=1, ladder=False, imp_k=IMP_K):
    """sign=+1 多、-1 空（價格取負）。回傳 (trades, pnl_bar 點數陣列)。"""
    o, h, l, c = (df[x + "_adj"].values * sign for x in ("open", "high", "low", "close"))
    if sign < 0:
        h, l = l, h
    raw = df.close.values
    atr, roll = df.atr.values, df.roll.values
    n = len(df)
    pnl = np.zeros(n)
    trades = []
    # 預先算碎形高點（給 ladder 用）
    fh = np.array([k >= 2 and k + 2 < n and h[k] > max(h[k - 2], h[k - 1]) and h[k] > max(h[k + 1], h[k + 2]) for k in range(n)])
    phase, t_imp, L, L_bar, confirm_bar, target, stop, stop0, entry, entry_bar = 0, -1, None, -1, -1, None, None, None, None, -1
    hi_since = -np.inf
    for i in range(ATR_N + 1, n - 1):
        a = atr[i]
        if np.isnan(a):
            continue
        # ── 持倉管理（開盤先看，整根再看）──
        if phase == 3:
            exit_px, how = None, None
            ref = entry if i == entry_bar else c[i - 1]   # 進場根以進場價為 MTM 基準
            if i != entry_bar and o[i] <= stop:
                exit_px, how = o[i], "gap_stop"
            elif l[i] <= stop:
                exit_px, how = stop, "stop"
            elif h[i] >= target:
                if ladder:
                    higher = [h[k] for k in range(max(0, entry_bar - 200), entry_bar) if fh[k] and h[k] > target + 1e-9]
                    if higher:
                        stop, target = max(stop, entry), min(higher)
                        pnl[i] += c[i] - ref
                        if roll[i]:
                            pnl[i] -= ROLL_PTS
                        continue
                exit_px, how = (o[i] if (i != entry_bar and o[i] >= target) else target), "target"
            if exit_px is not None:
                cost = (FEE * 2 + TAX * abs(raw[i]) * MULT + TAX * abs(raw[entry_bar]) * MULT) / MULT + tick(raw[i]) + tick(raw[entry_bar])
                gross = exit_px - entry
                pnl[i] += exit_px - ref - cost
                r = gross / (entry - stop0) if entry - stop0 > 0 else np.nan
                trades.append(dict(side="L" if sign > 0 else "S", entry_dt=str(df.datetime.iloc[entry_bar]), exit_dt=str(df.datetime.iloc[i]),
                                   entry=entry * sign, exit=exit_px * sign, gross=gross, net=gross - cost, R=r, bars=i - entry_bar, how=how))
                # 目標出場 → 線還在，回到 phase 1（需再離開再回測）；停損 → 線破，重找
                if how == "target" and i - confirm_bar < EXPIRE:
                    phase, hi_since = 1, max(hi_since, h[i])
                else:
                    phase = 0
                continue
            pnl[i] += c[i] - ref
            if roll[i]:
                pnl[i] -= ROLL_PTS
            continue
        # ── 無倉：規則狀態機（都用本根收盤資訊，下一根開盤進場）──
        if phase == 0:
            if c[i] - c[i - IMP_N] >= imp_k * a:
                t_imp, phase = i, 0.5
            continue
        if phase == 0.5:  # 找推升後第一個碎形低點（k ≥ t_imp+1，k+2 ≤ i 才確認）
            k = i - 2
            if k >= t_imp + 1 and fractal_low(l, k):
                L, L_bar, confirm_bar, phase, hi_since = l[k], k, i, 1, max(h[k:i + 1])
            elif i - t_imp > FIND_LOW:
                phase = 0
            continue
        # phase 1 / 2：線存在
        if c[i] < L - INVAL * a or i - confirm_bar > EXPIRE:
            phase = 0
            continue
        hi_since = max(hi_since, h[i])
        if phase == 1:
            if h[i] >= L + AWAY * a:
                phase = 2
            continue
        if phase == 2 and l[i] <= L + RETEST * a and c[i] >= L:
            tgt, ent, stp = hi_since, o[i + 1], L - STOP * a
            if tgt <= ent or ent <= stp:
                continue
            entry, entry_bar, target, stop, stop0, phase = ent, i + 1, tgt, stp, stp, 3
    return trades, pnl


def daily(df, pnl):
    return pd.Series(pnl, index=df.date.values).groupby(level=0).sum()


def stats(d, trades=None):
    eq = d.cumsum()
    dd = (eq - eq.cummax()).min()
    out = dict(total=eq.iloc[-1], sharpe=d.mean() / d.std() * np.sqrt(252) if d.std() > 0 else np.nan, mdd=dd,
               days=len(d))
    if trades is not None:
        T = pd.DataFrame(trades)
        if len(T):
            out.update(n=len(T), win=(T.net > 0).mean(), avg_net=T.net.mean(), med_net=T.net.median(), avg_R=T.R.mean(),
                       pf=T.net[T.net > 0].sum() / -T.net[T.net < 0].sum() if (T.net < 0).any() else np.inf,
                       bars=T.bars.mean(), exposure=T.bars.sum() / (len(d) * 10))
        else:
            out.update(n=0)
    return out


def fmt(name, s):
    base = f"{name:<26} 總點數 {s['total']:+8.0f} ({s['total']*MULT/1e4:+6.1f}萬)  Sharpe {s['sharpe']:+5.2f}  MDD {s['mdd']:7.0f}點"
    if s.get("n"):
        base += (f"  筆數 {s['n']:3d}  勝率 {s['win']*100:4.0f}%  均淨 {s['avg_net']:+6.1f}  中位 {s['med_net']:+6.1f}  均R {s['avg_R']:+5.2f}"
                 f"  PF {s['pf']:4.2f}  均持 {s['bars']:4.0f}根  曝險 {s['exposure']*100:3.0f}%")
    elif "n" in s:
        base += "  筆數 0"
    return base


def bench(df):
    c = df.close_adj.values
    pnl = np.r_[0.0, np.diff(c)] - np.where(df.roll.values, ROLL_PTS, 0.0)
    pnl[: ATR_N + 1] = 0
    return daily(df, pnl)


def split(d, a="2021-01-01", b="2024-01-01"):
    return d[(d.index >= a) & (d.index < b)], d[d.index >= b]


def main():
    df = load()
    sens = "--sens" in sys.argv
    print(f"QFF 30 分 K {df.date.iloc[0]} ~ {df.date.iloc[-1]}，{len(df)} 根；ATR14 中位 {df.atr.median():.1f} 點（≈{(df.atr/df.close_adj).median()*100:.2f}%）")
    B = bench(df)
    rows = {}
    ks = [2.0, 3.0, 4.0] if sens else [IMP_K]
    for imp_k in ks:
        for ladder in ([False] if sens else [False, True]):
            tag = f"k={imp_k:.0f}ATR" + (" 一層一層" if ladder else "")
            TL, PL = simulate(df, +1, ladder, imp_k)
            TS, PS = simulate(df, -1, ladder, imp_k)
            dL, dS = daily(df, PL), daily(df, PS)
            rows[f"多 {tag}"] = (dL, TL)
            rows[f"空 {tag}"] = (dS, TS)
            rows[f"多+空 {tag}"] = (dL + dS, TL + TS)
    print("\n── 全期 ──")
    print(fmt("連續做多 1 口（對照）", stats(B)))
    for k, (d, T) in rows.items():
        print(fmt(k, stats(d, T)))
    print("\n── 分段：2021-02~2023 ｜ 2024~2026-10 ──")
    for k, (d, T) in [("連續做多 1 口", (B, None))] + list(rows.items()):
        a, b = split(d)
        Ta = [t for t in (T or []) if t["exit_dt"] < "2024"]; Tb = [t for t in (T or []) if t["exit_dt"] >= "2024"]
        sa, sb = stats(a, Ta if T is not None else None), stats(b, Tb if T is not None else None)
        print(f"{k:<26} 前段 {sa['total']:+7.0f}點 Sharpe {sa['sharpe']:+5.2f} 筆 {sa.get('n','-'):>3} ｜ 後段 {sb['total']:+7.0f}點 Sharpe {sb['sharpe']:+5.2f} 筆 {sb.get('n','-'):>3}")
    if not sens:
        T = pd.DataFrame(rows[f"多+空 k={IMP_K:.0f}ATR"][1])
        print("\n── 主規則 多+空 出場方式 ──")
        print(T.groupby(["side", "how"]).agg(n=("net", "size"), net=("net", "sum"), avg_R=("R", "mean")).round(2).to_string())
        print("\n── 主規則 逐年（多+空，淨點數 / 筆數）──")
        T["yr"] = T.exit_dt.str[:4]
        print(T.groupby("yr").agg(n=("net", "size"), net=("net", "sum"), win=("net", lambda x: (x > 0).mean())).round(2).to_string())
        print("\n── 主規則 最近 8 筆 ──")
        print(T.tail(8)[["side", "entry_dt", "exit_dt", "entry", "exit", "net", "R", "how"]].round(1).to_string(index=False))
        if "--noslip" in sys.argv:
            return
        with open(LOG, "a") as f:
            for k, (d, T_) in rows.items():
                s = stats(d, T_)
                f.write(json.dumps(dict(ts=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"), kind="qff_pullback", period="2021-2026",
                                        strategy="H106", variant=k, note="第二十批 事前登記：快速推升第一個低點回測進場（QFF 30 分 K）",
                                        **{a: (None if isinstance(b, float) and np.isnan(b) else float(b) if isinstance(b, (int, float, np.floating, np.integer)) else b) for a, b in s.items()}),
                                   ensure_ascii=False) + "\n")
            s = stats(B)
            f.write(json.dumps(dict(ts=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"), kind="qff_pullback", period="2021-2026", strategy="bench_long",
                                    **{a: float(b) for a, b in s.items()}), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
