"""H121 條件疊加梯度：L0（pit 池）→ L1 H2≥0.95 → L2 +營收 3 月創 12 月新高 → L3 +YoY≥30% → L4 +SUR≥2。
每月換股日開盤買、下一換股日開盤賣；看各層勝率（淨報酬 > 0）相對 L0 提高多少、同月隨機抽樣 p。

用法：python3 research/v2/ladder.py dev|val
"""
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402
import xs  # noqa: E402
from hypotheses import h2_hi52  # noqa: E402

COND = {"H2≥0.95": lambda F: F.h2 >= 0.95, "營收新高": lambda F: F.r3h >= 0.999,
        "YoY≥30%": lambda F: F.yoy >= 0.30, "SUR≥2": lambda F: F.sur >= 2}
NDRAW = 5000


def add_sur(P):
    """同 xs.add_v17 的 sur（只用月營收，不碰 fin/bs/mv）。"""
    idx, cols = P["close"].index, P["close"].columns
    out = {}
    for s in cols:
        f = H.D / "revenue" / f"{s}.parquet"
        if not f.exists():
            continue
        d = pd.read_parquet(f)
        if not len(d):
            continue
        d["ym"] = pd.PeriodIndex.from_fields(year=d.revenue_year.astype(int), month=d.revenue_month.astype(int), freq="M")
        d = d.drop_duplicates("ym", keep="last").set_index("ym").revenue.astype(float)
        d = d.reindex(pd.period_range(d.index.min(), d.index.max(), freq="M"))
        dif = d - d.shift(12)
        h = dif.shift(1).rolling(24, min_periods=12)
        avail = (d.index + 1).to_timestamp() + pd.Timedelta(days=10)
        v = xs._align(pd.Series(((dif - h.mean()) / h.std()).values, index=avail), idx)
        if v is not None:
            out[s] = v
    P["sur"] = pd.DataFrame(out).reindex(index=idx, columns=cols)


def trades(P, p0, p1):
    """每個換股日 × pit 流動股一列：條件特徵（前一日）與持有一個月的淨報酬。"""
    O, C, Hh, L = P["open"], P["close"], P["high"], P["low"]
    Cf = C.ffill()
    idx = O.index
    feats = {"h2": h2_hi52(P), "r3h": P["rev3_high"], "yoy": P["rev_yoy"], "sur": P["sur"]}
    feats = {k: v.shift(1) for k, v in feats.items()}
    elig = P["liq"].shift(1).fillna(False).astype(bool) & (P["nbars"].shift(1) >= 250) & O.notna()
    lim_up = (Hh == L) & (O > Cf.shift(1))                       # 一字漲停：買不到
    rd = xs.rebalance_dates(idx, pd.Timestamp(p0), pd.Timestamp(p1))
    rows = []
    for d, d2 in zip(rd[:-1], rd[1:]):
        e = elig.loc[d] & ~lim_up.loc[d]
        names = e.index[e]
        k2 = idx.get_loc(d2)
        px2 = O.loc[d2, names].fillna(Cf.iloc[k2 - 1][names])    # 下市 / 停牌：最後收盤
        ret = (px2 * (1 - H.SELL_FEE - H.SELL_TAX) / (O.loc[d, names] * (1 + H.BUY_FEE)) - 1).clip(-0.95, 10)
        df = pd.DataFrame({"date": d, "sid": names, "ret": ret.values,
                           **{k: v.loc[d, names].values for k, v in feats.items()}})
        rows.append(df)
    T = pd.concat(rows, ignore_index=True).dropna(subset=["ret"])
    T["ex"] = T.ret - T.groupby("date").ret.transform("mean")
    return T


def matched_p(T, mask, rng):
    """同月隨機抽樣：每筆從同月 L0 抽一檔，勝率 ≥ 觀察值的比例。"""
    sel = T[mask]
    obs = (sel.ret > 0).mean()
    wins = T.assign(w=(T.ret > 0).astype(float)).groupby("date").w
    k = sel.groupby("date").size()
    # 同月不放回抽 k 檔的勝場數 = 超幾何分布
    pool = {d: wins.get_group(d).values for d in k.index}
    sims = np.zeros(NDRAW)
    for d, n in k.items():
        w = pool[d]
        sims += rng.hypergeometric(int(w.sum()), int(len(w) - w.sum()), int(n), size=NDRAW)
    return obs, (sims / k.sum() >= obs - 1e-12).mean()


def level_stats(T, mask, rng):
    sel = T[mask]
    if not len(sel):
        return dict(n=0)
    m = sel.groupby("date").ex.mean()
    win, p = matched_p(T, mask, rng)
    return dict(n=len(sel), months=sel.date.nunique(), win=win, win_ex=(sel.ex > 0).mean(), avg=sel.ret.mean(),
                med=sel.ret.median(), ex=sel.ex.mean(), t=m.mean() / m.std() * np.sqrt(len(m)) if len(m) > 2 else np.nan,
                p=p, yrs_pos=f"{int((sel.groupby(sel.date.dt.year).ex.mean() > 0).sum())}/{sel.date.dt.year.nunique()}")


def fmt(st):
    if not st["n"]:
        return "n=0"
    return (f"n={st['n']:>6} 月={st['months']:>3} 勝率={st['win']*100:5.1f}% 超額勝={st['win_ex']*100:5.1f}% "
            f"均={st['avg']*100:+5.2f}% 中位={st['med']*100:+5.2f}% 超額={st['ex']*100:+5.2f}% t={st['t']:5.2f} "
            f"p={st['p']:.4f} 正年={st['yrs_pos']}")


def main(period):
    if period == "holdout":
        raise PermissionError("holdout 封存中")
    p0, p1 = H.PERIODS[period]
    p0 = max(p0, "2011-02-01")
    P = xs.load_panel(p1, fundamentals=True)
    add_sur(P)
    T = trades(P, p0, p1)
    rng = np.random.default_rng(121)
    names = list(COND)
    ok = {k: COND[k](T).fillna(False).values for k in names}
    base = level_stats(T, np.ones(len(T), bool), rng)
    print(f"H121 {period} {p0}~{p1}\nL0 pit 池     {fmt(base)}")
    ladder, mask = [], np.ones(len(T), bool)
    for i, k in enumerate(names, 1):
        mask = mask & ok[k]
        st = level_stats(T, mask, rng)
        ladder.append(st)
        print(f"L{i} +{k:<9} {fmt(st)}")
    deep = max([i for i, st in enumerate(ladder) if st["n"] >= 100], default=None)
    if deep is not None:
        st = ladder[deep]
        wins = [base["win"]] + [x["win"] for x in ladder[:deep + 1]]
        mono = all(b >= a - 1e-12 for a, b in zip(wins, wins[1:]))
        need_p = 0.01 if period == "dev" else 0.05
        ok_ = (st["win"] - base["win"] >= 0.10) and st["p"] <= need_p and st["avg"] > 0 and (mono or period != "dev")
        print(f"\n主判定層 L{deep + 1}（n≥100 最深層）：勝率 {st['win']*100:.1f}% vs 基準 {base['win']*100:.1f}%"
              f"（+{(st['win'] - base['win'])*100:.1f}pp）p {st['p']:.4f} 單調 {mono} → {'✓' if ok_ else '✗'}")
    print("\n附帶描述：16 種組合（不判定）")
    for r in range(1, len(names) + 1):
        for combo in itertools.combinations(names, r):
            m = np.logical_and.reduce([ok[k] for k in combo])
            sel = T[m]
            if len(sel):
                print(f"  {' & '.join(combo):<40} n={len(sel):>6} 勝率={(sel.ret > 0).mean()*100:5.1f}% "
                      f"均={sel.ret.mean()*100:+5.2f}% 中位={sel.ret.median()*100:+5.2f}%")
    with open(H.LOG, "a") as f:
        f.write(json.dumps(dict(ts=time.strftime("%Y-%m-%d %H:%M:%S"), kind="ladder", period=period,
                                strategy="H121 條件疊加梯度", note="事前登記；勝=淨報酬>0", base=base,
                                levels={f"L{i}": st for i, st in enumerate(ladder, 1)}),
                           ensure_ascii=False, default=float) + "\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dev")
