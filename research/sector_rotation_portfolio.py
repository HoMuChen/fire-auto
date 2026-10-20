"""S1 產業動能輪動（120 日漲幅剛進全產業前 10%）投組模擬。

規則（沿用 sector_rotation.py 事件定義，不調參）：
  訊號：產業 EW 指數 120 日報酬排名剛進前 10% → 該產業站上 MA20 的成員為候選
  候選排序：個股 120 日報酬由高到低；最多同時持有 N 檔、每檔 1/N 權益
  出場：收盤 < 持有期最高收盤 ×(1−8%)、或產業 120 日排名 < 50%、或持有 60 日
  成交：t 日收盤判斷 → t+1 開盤；含手續費 0.1425%×2 + 證交稅 0.3%
對照：大盤等權（EW，無成本）、三策略分池組合（portfolio_sim.combined_daily_equity next_open）
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sector_rotation as SR  # noqa: E402

FEE, TAX, TRAIL, MAX_HOLD = SR.FEE, SR.TAX, 0.08, 60


def metrics(eq):
    eq = eq.dropna()
    r = eq.pct_change().dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    return dict(CAGR=(eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1, MDD=(eq / eq.cummax() - 1).min(),
                Sharpe=r.mean() / r.std() * np.sqrt(252))


def run(N=10, lb=120, th=0.9, pit=False):
    """pit=True：全部個股 + 當時近 60 日均量 ≥ 200 張 才算流動性（產業指數與候選都用），去除後見之明。"""
    P, ind = SR.load_panel("all" if pit else "liquid")
    liq = P["volume"].rolling(60).mean() >= 200_000 if pit else None
    if pit:  # 產業指數只用當時有流動性的成員
        P_idx = dict(P, close=P["close"].where(liq))
        members, idx = SR.industry_frames(P_idx, ind)
    else:
        members, idx = SR.industry_frames(P, ind)
    c, o = P["close"], P["open"]
    rank = (idx / idx.shift(lb) - 1).rank(axis=1, pct=True)
    newtop = (rank >= th) & (rank.shift(1) < th)
    ma20 = c.rolling(20).mean()
    mom = c / c.shift(lb) - 1
    dates = c.index
    C, O = c.values, o.values
    col = {s: j for j, s in enumerate(c.columns)}
    sid_ind = {s: ind.get(s) for s in c.columns}

    cash, pos, eq, trades, expo = 1.0, {}, [], [], []
    for t in range(len(dates) - 1):
        d = dates[t]
        # 收盤估值
        val = cash + sum(p["sh"] * (C[t, col[s]] if not np.isnan(C[t, col[s]]) else p["last"]) for s, p in pos.items())
        eq.append(val)
        expo.append(len(pos) / N)
        # 出場判斷
        sells = []
        for s, p in pos.items():
            px = C[t, col[s]]
            if np.isnan(px):
                continue
            p["last"] = px
            p["peak"] = max(p["peak"], px)
            k = sid_ind[s]
            if px < p["peak"] * (1 - TRAIL) or rank.at[d, k] < 0.5 or t - p["bi"] >= MAX_HOLD:
                sells.append(s)
        # 候選
        cands = []
        for k in newtop.columns[newtop.loc[d].values]:
            for s in members[k]:
                if s in col and s not in pos and C[t, col[s]] > ma20.iat[t, col[s]] and not np.isnan(O[t + 1, col[s]]) \
                        and (liq is None or liq.iat[t, col[s]]):
                    cands.append((mom.iat[t, col[s]], s))
        cands.sort(reverse=True)
        # t+1 開盤成交：先賣後買
        for s in sells:
            sp = O[t + 1, col[s]]
            if np.isnan(sp):
                continue
            p = pos.pop(s)
            cash += p["sh"] * sp * (1 - FEE - TAX)
            trades.append((s, dates[p["bi"]], sp * (1 - FEE - TAX) / (p["bp"] * (1 + FEE)) - 1))
        slot_val = val / N
        for _, s in cands:
            if len(pos) >= N:
                break
            bp = O[t + 1, col[s]]
            amt = min(slot_val, cash)
            if amt < slot_val * 0.5:
                break
            pos[s] = dict(sh=amt / (bp * (1 + FEE)), bp=bp, peak=bp, bi=t + 1, last=bp)
            cash -= amt
    eq = pd.Series(eq, index=dates[:-1])
    mc = c.where(liq) if pit else c
    mkt = (1 + mc.pct_change(fill_method=None).clip(-0.105, 0.105).mean(axis=1).fillna(0)).cumprod()
    return eq, pd.DataFrame(trades, columns=["sid", "entry", "ret"]), np.mean(expo), mkt


def main():
    pit = "--pit" in sys.argv
    eq, tr, expo, mkt = run(pit=pit)
    print("股票池：", "全部個股 + point-in-time 流動性" if pit else "現行流動性名單（含後見之明）")
    start = eq.index[eq.ne(eq.iloc[0]).argmax()]  # 第一次交易日（120 日暖身後）
    rows = []
    for lab, a, b in (("全期", start, None), ("A 2020-22", start, "2022-12-31"), ("B 2023-26", "2023-01-01", None)):
        e = eq.loc[a:b]; m = mkt.loc[a:b]
        rows.append(dict(期間=lab, **{f"S1_{k}": v for k, v in metrics(e).items()}, **{f"EW_{k}": v for k, v in metrics(m).items()}))
    print(f"S1 產業動能(120d top10%) N=10：{len(tr)} 筆（每年 {len(tr)/((eq.index[-1]-start).days/365.25):.0f}），"
          f"每筆淨 {tr.ret.mean()*100:+.2f}%，勝率 {(tr.ret>0).mean()*100:.0f}%，平均曝險 {expo:.0%}")
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    yr = eq.resample("YE").last().pct_change(); yr.iloc[0] = eq.resample("YE").last().iloc[0] / eq.loc[start] - 1
    ym = mkt.resample("YE").last().pct_change()
    print("逐年 S1 vs EW:", " ".join(f"{d.year}:{a*100:+.0f}/{b*100:+.0f}" for d, a, b in zip(yr.index, yr.values, ym.values)))
    try:
        import portfolio_sim as PS
        dd, vv = PS.combined_daily_equity(exec_mode="next_open")
        tri = pd.Series(vv, index=pd.to_datetime(dd))
        j = pd.concat([eq.pct_change(), tri.pct_change(), mkt.pct_change()], axis=1, keys=["S1", "三策略", "EW"]).dropna()
        print("日報酬相關：\n", j.corr().round(2).to_string())
        both = (0.5 * (1 + j.S1).cumprod() + 0.5 * (1 + j["三策略"]).cumprod())
        only = (1 + j["三策略"]).cumprod()
        print("三策略單獨:", {k: round(v, 3) for k, v in metrics(only).items()},
              " 50/50 混合:", {k: round(v, 3) for k, v in metrics(both).items()})
    except Exception as ex:  # noqa: BLE001
        print("三策略對照略過：", ex)


if __name__ == "__main__":
    main()
