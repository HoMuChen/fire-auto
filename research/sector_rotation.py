"""台股產業輪動 / 同業老二補漲 研究（2026-10-20）

策略（事件式，逐檔模擬，隔日開盤成交，含手續費+稅，移動停損 8%）：
  S1 產業動能輪動：產業 20 日漲幅剛進全產業前 20% → 買站上 MA20 的成員
  S2 龍頭突破→老二補漲：龍頭（產業內 60 日均成交金額最大）創 60 日新高（20 日內首次）
                         → 買 20 日漲幅落後產業、仍在 MA60 上的非龍頭成員
  S2c 對照：同一事件買「已跟漲」（20 日漲幅 ≥ 產業）的成員
  S3 產業內相對超跌：產業指數在 MA60 上，個股 20 日報酬落後產業 >10%（首次）
  S4 輪動到落後產業：產業 60 日漲幅排名後 30% 且剛站上 MA20，大盤 EW 在 MA60 上 → 買站上 MA20 成員
  BASE 對照：每 20 個交易日無條件進場，同出場規則

產業 = individual_stocks.json industry_category（去掉「類」後綴；排除 電子工業/其他/其他電子 等雜項類）
驗證 = validate_strategy_groups.py 的 6 群組 × A(2020-22)/B(2023-26) 規則
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
from validate_strategy_groups import compute_stock_features, assign_groups  # noqa: E402

FEE, TAX = 0.001425, 0.003
TRAIL = 0.08
JUNK = {"電子工業", "其他", "其他電子", "其他電子業", "創新板股票"}


def load_panel(universe="liquid"):
    """universe="liquid"：individual_stocks.json low_liquidity=False（以 2024-26 量能判定，含後見之明）；
    "all"：全部個股，流動性改由呼叫端用 point-in-time 量能過濾。"""
    stocks = [s for s in json.load(open(BASE / "individual_stocks.json"))
              if universe == "all" or s["low_liquidity"] is False]
    cols = {k: {} for k in ("open", "high", "low", "close", "money", "volume")}
    ind = {}
    for s in stocks:
        f = BASE / "data/stock_prices" / f"{s['stock_id']}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f, parse_dates=["date"]).set_index("date")
        d = d[(d.close > 0) & (d.open > 0)]
        if len(d) < 250:
            continue
        sid = s["stock_id"]
        for k, c in (("open", "open"), ("high", "high"), ("low", "low"), ("close", "close"), ("money", "trading_money"), ("volume", "volume")):
            cols[k][sid] = d[c]
        cat = (s["industry_category"] or "").removesuffix("類")
        ind[sid] = None if cat in JUNK else cat
    P = {k: pd.DataFrame(v).sort_index() for k, v in cols.items()}
    return P, pd.Series(ind)


def industry_frames(P, ind):
    c = P["close"]
    ret = c.pct_change(fill_method=None).clip(-0.105, 0.105)
    valid = ind.dropna()
    cnt = valid.value_counts()
    keep = cnt[cnt >= 5].index
    members = {k: valid[valid == k].index.tolist() for k in keep}
    idx = pd.DataFrame({k: (1 + ret[m].mean(axis=1).fillna(0)).cumprod() for k, m in members.items()})
    return members, idx


def simulate(P, entry, exit_sig=None, max_hold=40):
    """entry/exit_sig: bool DataFrame（t 日收盤判斷）→ t+1 開盤成交。回傳 trades list。"""
    O, C = P["open"].values, P["close"].values
    E = entry.reindex_like(P["close"]).fillna(False).values
    X = exit_sig.reindex_like(P["close"]).fillna(False).values if exit_sig is not None else np.zeros_like(E)
    dates = P["close"].index
    trades = []
    for j, sid in enumerate(P["close"].columns):
        pos = False
        for t in range(len(dates) - 1):
            if np.isnan(C[t, j]):
                continue
            if not pos:
                if E[t, j] and not np.isnan(O[t + 1, j]):
                    pos, bi, bp, peak = True, t + 1, O[t + 1, j], C[t + 1, j]
                continue
            if t < bi:
                continue
            peak = max(peak, C[t, j])
            if C[t, j] < peak * (1 - TRAIL) or X[t, j] or (t - bi) >= max_hold:
                sp = O[t + 1, j]
                if np.isnan(sp):
                    continue
                trades.append((sid, dates[bi], sp * (1 - FEE - TAX) / (bp * (1 + FEE)) - 1, t + 1 - bi))
                pos = False
    return pd.DataFrame(trades, columns=["sid", "entry", "ret", "hold"])


def build_signals(P, ind, members, idx):
    c = P["close"]
    ma20, ma60 = c.rolling(20).mean(), c.rolling(60).mean()
    r20 = c / c.shift(20) - 1
    ind_r20 = idx / idx.shift(20) - 1
    ind_r60 = idx / idx.shift(60) - 1
    rank20 = ind_r20.rank(axis=1, pct=True)
    rank60 = ind_r60.rank(axis=1, pct=True)
    ind_ma20, ind_ma60 = idx.rolling(20).mean(), idx.rolling(60).mean()
    mkt = (1 + c.pct_change(fill_method=None).clip(-0.105, 0.105).mean(axis=1).fillna(0)).cumprod()
    mkt_up = mkt > mkt.rolling(60).mean()
    money60 = P["money"].rolling(60).mean()
    hh60 = c.rolling(60).max()

    def per_member(frame_fn):
        out = pd.DataFrame(False, index=c.index, columns=c.columns)
        for k, m in members.items():
            m = [s for s in m if s in c.columns]
            out[m] = frame_fn(k, m)
        return out

    sig = {}
    # S1
    new_top = (rank20 >= 0.8) & (rank20.shift(1) < 0.8)
    sig["S1_產業動能輪動"] = (per_member(lambda k, m: pd.DataFrame({s: new_top[k] for s in m})) & (c > ma20),
                         per_member(lambda k, m: pd.DataFrame({s: rank20[k] < 0.5 for s in m})), 60)

    # S2 / S2c：龍頭 = 產業內 60 日均成交金額最大
    lead_brk = {}
    is_leader = pd.DataFrame(False, index=c.index, columns=c.columns)
    for k, m in members.items():
        m = [s for s in m if s in c.columns]
        mm = money60[m]
        lid = mm.fillna(-1).idxmax(axis=1).where(mm.notna().any(axis=1))
        brk_all = (c[m] >= hh60[m]).values
        col = {s: i for i, s in enumerate(m)}
        brk = pd.Series([bool(brk_all[i, col[l]]) if isinstance(l, str) else False
                         for i, l in enumerate(lid.values)], index=c.index)
        first = brk & ~brk.shift(1).rolling(20, min_periods=1).max().fillna(0).astype(bool)
        lead_brk[k] = first
        for s in m:
            is_leader[s] = lid == s
    ev = per_member(lambda k, m: pd.DataFrame({s: lead_brk[k] for s in m}))
    rel = per_member(lambda k, m: pd.DataFrame({s: r20[s] - ind_r20[k] for s in m}).lt(0))
    base2 = ev & ~is_leader & (c > ma60)
    sig["S2_龍頭突破_老二補漲"] = (base2 & rel, None, 40)
    sig["S2c_對照_已跟漲成員"] = (base2 & ~rel, None, 40)

    # S3
    relv = per_member(lambda k, m: pd.DataFrame({s: r20[s] - ind_r20[k] for s in m}).astype(float).lt(-0.10))
    ind_ok = per_member(lambda k, m: pd.DataFrame({s: idx[k] > ind_ma60[k] for s in m}))
    s3 = relv & ~relv.shift(1, fill_value=False) & ind_ok
    caught = per_member(lambda k, m: pd.DataFrame({s: r20[s] - ind_r20[k] for s in m}).astype(float).ge(0))
    sig["S3_產業內相對超跌"] = (s3, caught, 40)

    # S4
    lag_turn = (rank60 <= 0.3) & (idx > ind_ma20) & (idx.shift(1) <= ind_ma20.shift(1))
    lag_turn = lag_turn & pd.DataFrame({k: mkt_up for k in idx.columns})
    sig["S4_輪動到落後產業"] = (per_member(lambda k, m: pd.DataFrame({s: lag_turn[k] for s in m})) & (c > ma20), None, 60)

    # BASE
    b = pd.DataFrame(False, index=c.index, columns=c.columns)
    b.iloc[::20] = True
    sig["BASE_每20日無條件"] = (b, None, 40)
    return sig


def group_report(tr, groups):
    tr = tr.copy()
    tr["group"] = tr.sid.map(groups)
    tr["period"] = np.where(tr.entry <= "2022-12-31", "A", "B")
    rows = []
    for g, x in tr.groupby("group"):
        a, b = x[x.period == "A"], x[x.period == "B"]
        ok = len(a) >= 30 and len(b) >= 30
        rows.append(dict(group=g, nA=len(a), avgA=a.ret.mean() * 100, nB=len(b), avgB=b.ret.mean() * 100,
                         verdict="VALID" if ok and a.ret.mean() > 0 and b.ret.mean() > 0 else ("INSUF" if not ok else "x")))
    return pd.DataFrame(rows)


def main():
    P, ind = load_panel()
    members, idx = industry_frames(P, ind)
    print(f"個股 {P['close'].shape[1]} 檔，產業 {len(members)} 個（成員≥5，排除雜項類），期間 {P['close'].index[0].date()}~{P['close'].index[-1].date()}")
    feats = {}
    for s in P["close"].columns:
        cl = P["close"][s].dropna()
        f = compute_stock_features([{"close": x} for x in cl.values])
        if f:
            feats[s] = f
    groups, _, _ = assign_groups(feats)
    sig = build_signals(P, ind, members, idx)
    summ = []
    out = BASE / "data/sector_rotation"
    out.mkdir(exist_ok=True)
    for name, (ent, ex, mh) in sig.items():
        tr = simulate(P, ent, ex, mh)
        tr.to_csv(out / f"{name}.csv", index=False)
        g = group_report(tr, groups)
        yr = tr.groupby(tr.entry.dt.year).ret.mean() * 100
        summ.append(dict(strategy=name, n=len(tr), per_yr=len(tr) / 6.8, avg=tr.ret.mean() * 100,
                         win=(tr.ret > 0).mean() * 100, hold=tr.hold.mean(),
                         valid=(g.verdict == "VALID").sum(),
                         **{f"y{y}": v for y, v in yr.items()}))
        print(f"\n=== {name}  n={len(tr)} avg={tr.ret.mean()*100:+.2f}% win={(tr.ret>0).mean()*100:.1f}% hold={tr.hold.mean():.1f}d")
        print(g.round(2).to_string(index=False))
    s = pd.DataFrame(summ)
    pd.set_option("display.width", 250)
    print("\n=== 總表（avg=每筆淨報酬%，y20xx=該年進場均報酬%）")
    print(s.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
