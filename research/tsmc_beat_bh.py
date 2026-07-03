"""
台積電專屬策略研究：價量 + 籌碼 + 券商分點，目標跑贏 Buy & Hold
- next-bar 成交（t 日訊號 → t+1 開盤價成交）
- 成本：買 0.1425%、賣 0.1425% + 稅 0.3%
- long-only 無槓桿，訊號 in=滿倉 / out=空手
- 共同窗口 2021-06-30 ~ 2026-07-02（券商分點起點）
"""
import json
import math
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
BUY_COST = 0.001425
SELL_COST = 0.001425 + 0.003

START = "2021-06-30"
END = "2026-07-02"


# ─── 載入資料 ───

def load_price():
    df = pd.read_csv(BASE / "data/stock_prices/2330.csv", parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_inst():
    df = pd.read_csv(BASE / "data/institutional/2330.csv", parse_dates=["date"])
    piv = df.pivot_table(index="date", columns="investor_name",
                         values=["buy", "sell"], aggfunc="sum").fillna(0)
    out = pd.DataFrame(index=piv.index)
    for name in ["Foreign_Investor", "Investment_Trust", "Dealer_self", "Dealer_Hedging"]:
        if ("buy", name) in piv.columns:
            out[f"net_{name}"] = piv[("buy", name)] - piv[("sell", name)]
    out["net_all"] = out.sum(axis=1)
    return out.reset_index()


def load_margin():
    df = pd.read_csv(BASE / "data/margin/2330.csv", parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["margin_chg"] = df["margin_purchase_today_balance"] - df["margin_purchase_yesterday_balance"]
    df["short_chg"] = df["short_sale_today_balance"] - df["short_sale_yesterday_balance"]
    return df[["date", "margin_purchase_today_balance", "short_sale_today_balance",
               "margin_chg", "short_chg"]]


def load_broker_conc():
    df = pd.read_parquet(BASE / "data/broker_concentration.parquet")
    df = df[df["stock_id"] == "2330"].copy()
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "buy_conc", "total_buy"]].sort_values("date").reset_index(drop=True)


def build_dataset():
    px = load_price()
    inst = load_inst()
    marg = load_margin()
    conc = load_broker_conc()

    df = px.merge(inst, on="date", how="left") \
           .merge(marg, on="date", how="left") \
           .merge(conc, on="date", how="left")

    # ── 價量指標 ──
    c = df["close"]
    for n in [5, 10, 20, 60, 120, 200]:
        df[f"ma{n}"] = c.rolling(n).mean()
    df["mom60"] = c / c.shift(60) - 1
    df["mom20"] = c / c.shift(20) - 1
    df["hh20"] = df["high"].rolling(20).max()
    df["ll10"] = df["low"].rolling(10).min()
    df["hh55"] = df["high"].rolling(55).max()
    df["ll20"] = df["low"].rolling(20).min()
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    # ATR(14)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - c.shift()).abs(),
        (df["low"] - c.shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()
    # RSI(14)
    delta = c.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    dn = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi14"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))

    # ── 籌碼指標（volume-normalized，rolling） ──
    vol_shares = df["volume"]
    for col, short in [("net_Foreign_Investor", "fi"), ("net_Investment_Trust", "it")]:
        ratio = df[col] / vol_shares
        df[f"{short}_r5"] = ratio.rolling(5).sum()
        df[f"{short}_r10"] = ratio.rolling(10).sum()
        df[f"{short}_r20"] = ratio.rolling(20).sum()
    df["margin_chg_r10"] = (df["margin_chg"] * 1000 / vol_shares).rolling(10).sum()

    # ── 券商分點集中度 ──
    df["conc_ma5"] = df["buy_conc"].rolling(5).mean()
    df["conc_pct120"] = df["buy_conc"].rolling(120).rank(pct=True)

    return df


# ─── 回測引擎 ───

def simulate(df, signal, start=START, end=END):
    """signal: bool Series（t 日收盤判斷），t+1 開盤成交。回傳績效 dict。"""
    d = df.copy()
    d["sig"] = signal.fillna(False).astype(bool)
    d = d[(d["date"] >= start) & (d["date"] <= end)].reset_index(drop=True)

    cash = 1.0
    shares = 0.0
    in_pos = False
    equity = []
    trades = 0
    entry_val = 0.0
    wins = 0
    trade_rets = []

    for i in range(len(d)):
        o = d.loc[i, "open"]
        cl = d.loc[i, "close"]
        prev_sig = d.loc[i - 1, "sig"] if i > 0 else False

        if not in_pos and prev_sig:
            shares = cash * (1 - BUY_COST) / o
            entry_val = cash
            cash = 0.0
            in_pos = True
            trades += 1
        elif in_pos and not prev_sig:
            cash = shares * o * (1 - SELL_COST)
            ret = cash / entry_val - 1
            trade_rets.append(ret)
            if ret > 0:
                wins += 1
            shares = 0.0
            in_pos = False

        equity.append(cash + shares * cl)

    eq = pd.Series(equity, index=d["date"])
    if in_pos:  # 期末未平倉，用市值計
        pass

    total = eq.iloc[-1] / eq.iloc[0] - 1
    yrs = (d["date"].iloc[-1] - d["date"].iloc[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    dd = (eq / eq.cummax() - 1).min()
    ret_d = eq.pct_change().dropna()
    sharpe = ret_d.mean() / ret_d.std() * math.sqrt(252) if ret_d.std() > 0 else 0
    exposure = d["sig"].mean()

    return {
        "total": total, "cagr": cagr, "mdd": dd, "sharpe": sharpe,
        "trades": trades, "exposure": exposure,
        "win_rate": wins / len(trade_rets) if trade_rets else None,
        "equity": eq,
    }


def bench_bh(df, start=START, end=END):
    d = df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)
    entry = d["open"].iloc[1] * (1 + BUY_COST)  # 同樣 next-bar 開盤買進
    eq = d["close"] / entry
    eq.index = d["date"]
    total = eq.iloc[-1] / eq.iloc[0] - 1
    yrs = (d["date"].iloc[-1] - d["date"].iloc[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    dd = (eq / eq.cummax() - 1).min()
    ret_d = eq.pct_change().dropna()
    sharpe = ret_d.mean() / ret_d.std() * math.sqrt(252)
    return {"total": total, "cagr": cagr, "mdd": dd, "sharpe": sharpe,
            "trades": 1, "exposure": 1.0, "win_rate": None, "equity": eq}


# ─── 策略定義（訊號 = 持有狀態 bool） ───

def strategies(df):
    c = df["close"]
    s = {}

    # === 價量 ===
    s["MA60 濾網"] = c > df["ma60"]
    s["MA120 濾網"] = c > df["ma120"]
    s["MA200 濾網"] = c > df["ma200"]
    s["雙均線 20/60"] = df["ma20"] > df["ma60"]
    s["雙均線 50/200 黃金交叉"] = df["ma5"].rolling(50).mean() > df["ma200"]  # 近似
    s["動量 mom60>0"] = df["mom60"] > 0
    # 唐奇安：突破 N 高進、跌破 M 低出 → 狀態機
    def donchian(hh_col, ll_col):
        hh_prev = df[hh_col].shift(1)
        ll_prev = df[ll_col].shift(1)
        states = []
        state = False
        for i in range(len(df)):
            if not state and c.iloc[i] >= hh_prev.iloc[i]:
                state = True
            elif state and c.iloc[i] <= ll_prev.iloc[i]:
                state = False
            states.append(state)
        return pd.Series(states, index=df.index)

    don = donchian("hh20", "ll10")
    don2 = donchian("hh55", "ll20")
    s["唐奇安 20/10"] = don
    s["唐奇安 55/20"] = don2

    # === 籌碼 ===
    s["外資10日買超>0"] = df["fi_r10"] > 0
    s["外資20日買超>0"] = df["fi_r20"] > 0
    s["投信10日買超>0"] = df["it_r10"] > 0
    s["外資或投信10日>0"] = (df["fi_r10"] > 0) | (df["it_r10"] > 0)
    s["融資10日減少(反向)"] = df["margin_chg_r10"] < 0

    # === 券商分點 ===
    s["集中度>120日中位"] = df["conc_pct120"] > 0.5
    s["集中度>120日70%"] = df["conc_pct120"] > 0.7

    # === 組合 ===
    s["MA60 + 外資10日"] = (c > df["ma60"]) & (df["fi_r10"] > 0)
    s["MA60 或 外資20日"] = (c > df["ma60"]) | (df["fi_r20"] > 0)
    s["MA120 + 外資20日"] = (c > df["ma120"]) & (df["fi_r20"] > 0)
    s["MA60 + 集中度>中位"] = (c > df["ma60"]) & (df["conc_pct120"] > 0.5)
    s["MA200濾網 + 外資10日擇時"] = (c > df["ma200"]) & (df["fi_r10"] > -0.05)
    s["唐奇安20/10 + 外資10日"] = don & (df["fi_r10"] > 0)

    # === 第二輪：反向集中度（權值股高集中度=負向 → 低集中度=正向？） ===
    s["低集中度(<中位)"] = df["conc_pct120"] < 0.5
    s["低集中度 + MA60"] = (df["conc_pct120"] < 0.5) & (c > df["ma60"])
    s["低集中度 或 MA200"] = (df["conc_pct120"] < 0.5) | (c > df["ma200"])

    # === 第二輪：預設持有、極端惡化才退（risk-off only） ===
    # 出場需同時滿足多個惡化條件，否則一直抱著 → 曝險極高
    below200 = c < df["ma200"]
    below120 = c < df["ma120"]
    fi_neg = df["fi_r20"] < 0
    fi_neg10 = df["fi_r10"] < 0
    s["退場=破MA200且外資20日賣"] = ~(below200 & fi_neg)
    s["退場=破MA120且外資20日賣"] = ~(below120 & fi_neg)
    s["退場=破MA200且外資10日賣"] = ~(below200 & fi_neg10)

    # 遲滯版：破 MA200+外資賣 → 出場；站回 MA120 → 才回場
    def hysteresis(exit_cond, entry_cond):
        states, state = [], True  # 預設在場
        ec, en = exit_cond.fillna(False), entry_cond.fillna(False)
        for i in range(len(df)):
            if state and ec.iloc[i]:
                state = False
            elif not state and en.iloc[i]:
                state = True
            states.append(state)
        return pd.Series(states, index=df.index)

    s["遲滯:出=破200+外資賣/回=站上120"] = hysteresis(below200 & fi_neg, c > df["ma120"])
    s["遲滯:出=破200+外資賣/回=站上60"] = hysteresis(below200 & fi_neg, c > df["ma60"])
    s["遲滯:出=破120+外資賣/回=站上120"] = hysteresis(below120 & fi_neg, c > df["ma120"])

    # === 第二輪：融資過熱退場（散戶槓桿高漲=風險） ===
    margin_hot = df["margin_chg_r10"] > df["margin_chg_r10"].rolling(120).quantile(0.9)
    s["退場=融資過熱(120日90%)"] = ~margin_hot
    s["退場=融資過熱或破200"] = ~(margin_hot | (below200 & fi_neg))

    return s


def fmt(name, r, bh):
    beat = "✓" if r["total"] > bh["total"] else " "
    wr = f"{r['win_rate']:.0%}" if r["win_rate"] is not None else "  - "
    return (f"  {beat} {name:<24} 總報酬 {r['total']*100:>+7.1f}%  CAGR {r['cagr']*100:>+6.1f}%  "
            f"MDD {r['mdd']*100:>6.1f}%  Sharpe {r['sharpe']:>5.2f}  "
            f"交易 {r['trades']:>3}  曝險 {r['exposure']:>4.0%}  勝率 {wr}")


def main():
    df = build_dataset()
    bh = bench_bh(df)
    print(f"回測窗口: {START} ~ {END}（next-bar 開盤成交，含成本）")
    print(f"\n  Buy & Hold 基準:        總報酬 {bh['total']*100:+7.1f}%  CAGR {bh['cagr']*100:+6.1f}%  "
          f"MDD {bh['mdd']*100:6.1f}%  Sharpe {bh['sharpe']:5.2f}")
    print("\n" + "=" * 110)

    results = {}
    for name, sig in strategies(df).items():
        r = simulate(df, sig)
        results[name] = r
        print(fmt(name, r, bh))

    print("=" * 110)
    beat = {k: v for k, v in results.items() if v["total"] > bh["total"]}
    print(f"\n跑贏 B&H 總報酬的策略數: {len(beat)}/{len(results)}")
    for k, v in sorted(beat.items(), key=lambda x: -x[1]["total"]):
        print(f"  {k}: {v['total']*100:+.1f}% (B&H {bh['total']*100:+.1f}%)")


if __name__ == "__main__":
    main()
