"""
小型台積電期貨(QFF) 再掛單網格收租 — 30 分 K 版

沿用 research/tsmc_grid_income.py 的收租邏輯（買=價≤近H高×(1-STEP)、每批漲+TAKE 賣、
最多 N 批、±STEP 內不重複、套牢不認賠），但資料改用真實 QFF 日盤 30 分 K
（data/futures/QFF_30min_day.csv，近月連續合約）。

與股票版差異：
- 期貨：1 口 = 100 股，1 點 = NT$100；每批 = contracts_per_lot 口。
- 換月：對近月連續序列做「比例(ratio)還原」消除轉倉價差跳空（保留 % 走勢，供 % 網格用）。
- 成本：期交稅 0.002%/邊（契約金額）+ 手續費/口/邊；轉倉另計每口點數成本。
- 無槓桿檢查：回報實際槓桿（總名目/權益），套牢不強平（現金流策略前提）。
- 視窗：30 分 K 僅 2025-01 起（~1.6 年，且為強多頭段），與股票版 6.5 年不可直接比。

用法：
  python3 research/tsmc_futures_grid_30min.py
  python3 research/tsmc_futures_grid_30min.py monthly
  python3 research/tsmc_futures_grid_30min.py --max-lots 20 --take 0.01 --step 0.01 --h-days 10
"""
import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).parent.parent
CSV = BASE / "data" / "futures" / "QFF_30min_day.csv"

MULTIPLIER = 100                 # 1 口 = 100 股；1 點 = NT$100
BARS_PER_DAY = 10
FUTURES_TAX = 0.00002            # 期交稅（股價類期貨）每邊
DEFAULT_FEE_PER_SIDE = 20.0      # 手續費 元/口/邊（保守）
DEFAULT_ROLL_COST_POINTS = 2.0   # 每口每次轉倉成本（點；價差+手續費估）


def load_bars(start=None):
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    if start:
        rows = [r for r in rows if r["datetime"][:10] >= start]
    dt = [r["datetime"] for r in rows]
    con = [r["contract_date"] for r in rows]
    close = [float(r["close"]) for r in rows]
    # 比例還原（ratio back-adjust）：把舊合約價按 roll 比例縮放到最新合約尺度，消跳空
    n = len(close)
    factor = 1.0
    adj = [0.0] * n
    for i in range(n - 1, -1, -1):
        if i < n - 1 and con[i] != con[i + 1] and close[i] > 0:
            factor *= close[i + 1] / close[i]
        adj[i] = close[i] * factor
    # 轉倉點：contract 改變的 bar index（開倉部位在此付轉倉成本）
    rolls = set(i for i in range(1, n) if con[i] != con[i - 1])
    return dt, con, close, adj, rolls


def run(p):
    dt, con, real, adj, rolls = load_bars(getattr(p, "start", None))
    n = len(adj)
    H = p.h_days * BARS_PER_DAY

    capital_base = p.capital      # 投入本金（補錢模式下會隨追繳增加）
    total_deposit = 0.0           # 累計補錢
    deposit_events = 0
    worst_margin = (9.99, "")     # 最緊的 (equity/exposure, 日期)；越接近維持保證金越危險
    realized = 0.0
    lots = []                     # {price(adj), contracts}
    curve = []
    mcf = defaultdict(float)      # 每月已實現現金流(NT$)
    mbuy = defaultdict(int); msell = defaultdict(int)
    max_lev = 0.0
    max_lots = 0
    end_lots = {}

    # 趨勢濾網用的移動平均（bars）
    RM = p.regime_ma
    ma_series = [None] * n
    if RM:
        run_sum = 0.0
        for i in range(n):
            run_sum += adj[i]
            if i >= RM:
                run_sum -= adj[i - RM]
            if i >= RM - 1:
                ma_series[i] = run_sum / RM

    # 波動率序列（近100根30分報酬 std，×√10 ≈ 日波動），供動態格距用
    vol_series = [None] * n
    if getattr(p, "atr_mult", 0):
        VW = 100
        rets = [0.0] + [(adj[i] / adj[i - 1] - 1) if adj[i - 1] > 0 else 0.0
                        for i in range(1, n)]
        for i in range(n):
            if i >= VW:
                w = rets[i - VW + 1:i + 1]
                mean = sum(w) / len(w)
                sd = (sum((r - mean) ** 2 for r in w) / (len(w) - 1)) ** 0.5
                vol_series[i] = sd * (10 ** 0.5)

    # RSI 序列（超賣過濾用），週期 p.rsi_period（預設 70 根≈7日）
    rsi_series = [None] * n
    if getattr(p, "rsi_gate", 0):
        per = getattr(p, "rsi_period", 70)
        gains = [0.0] * n
        losses = [0.0] * n
        for i in range(1, n):
            ch = adj[i] - adj[i - 1]
            gains[i] = max(ch, 0.0)
            losses[i] = max(-ch, 0.0)
        ag = al = 0.0
        for i in range(1, n):
            if i <= per:
                ag += gains[i] / per
                al += losses[i] / per
            else:
                ag = (ag * (per - 1) + gains[i]) / per
                al = (al * (per - 1) + losses[i]) / per
            if i >= per:
                rsi_series[i] = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)

    # 長期高點（等比加寬 / 深度加碼 用「跌多深」衡量）
    LW = 600  # 60 日
    ref_long = [0.0] * n
    if getattr(p, "widen", 0) or getattr(p, "depth_scale", 0):
        for i in range(n):
            ref_long[i] = max(adj[max(0, i - LW):i + 1])

    def ref_long_ds(i):
        return ref_long[i]

    def dyn_step(i):
        """有效格距：① atr_mult×日波動；② 等比加寬(跌越深越寬)；否則固定 p.step。"""
        s = p.step
        if getattr(p, "atr_mult", 0) and vol_series[i]:
            s = max(0.004, min(0.025, p.atr_mult * vol_series[i]))
        if getattr(p, "widen", 0) and ref_long[i] > 0:
            drop = max(0.0, (ref_long[i] - adj[i]) / ref_long[i])
            s = s * (1 + p.widen * drop)     # 跌 drop 深→格距放大
        return min(s, 0.05)

    bought_today = False
    for i in range(n):
        c = adj[i]
        ym = dt[i][:7]
        day = dt[i][:10]
        if i == 0 or dt[i - 1][:10] != day:
            bought_today = False              # 換日重置
        is_close_bar = (i == n - 1) or (dt[i + 1][:10] != day)

        # 轉倉成本（開倉部位）
        if i in rolls and lots:
            roll_cost = p.roll_cost_points * MULTIPLIER * sum(l["contracts"] for l in lots)
            realized -= roll_cost
            mcf[ym] -= roll_cost

        step_i = dyn_step(i)
        take_i = step_i if getattr(p, "atr_mult", 0) else p.take

        # 賣：逐批漲到買價 +TAKE 平倉；移動停利模式則達標後改追蹤高點回落
        def do_close(lot, price):
            nonlocal realized
            gross = (price - lot["price"]) * MULTIPLIER * lot["contracts"]
            cost = (adj_notional(lot["price"], lot["contracts"]) +
                    adj_notional(price, lot["contracts"])) * FUTURES_TAX \
                + p.fee_per_side * lot["contracts"] * 2
            pnl = gross - cost
            realized += pnl
            mcf[ym] += pnl
            msell[ym] += 1

        rem = []
        for lot in lots:
            if getattr(p, "trail_tp", 0):
                # 移動停利：達 +take 才啟動，之後追蹤高點、回落 trail_tp 才賣（讓贏家跑）
                if c >= lot["price"] * (1 + take_i):
                    lot["armed"] = True
                if lot.get("armed"):
                    lot["peak"] = max(lot.get("peak", c), c)
                    if c <= lot["peak"] * (1 - p.trail_tp):
                        do_close(lot, c)
                        continue
                rem.append(lot)
            else:
                if c >= lot["price"] * (1 + take_i):
                    do_close(lot, c)
                else:
                    rem.append(lot)
        lots[:] = rem

        # 買：跌破近H高×(1-STEP)、±STEP 內無持倉、批數/槓桿夠、（可選）趨勢濾網
        ref = max(adj[max(0, i - H):i + 1])
        equity = capital_base + realized + unreal(lots, c)
        notional_after = notional(lots, c) + adj_notional(c, p.contracts_per_lot)
        lev_ok = notional_after <= equity * p.max_leverage
        regime_ok = (not RM) or (ma_series[i] is not None and c > ma_series[i])
        # 當天已買過 → 只在收盤那根再檢查（放慢單日填倉）
        timing_ok = (not p.intraday_once) or (not bought_today) or is_close_bar
        # RSI 超賣過濾：只在 RSI < gate 時吃網格買點
        rsi_ok = (not getattr(p, "rsi_gate", 0)) or \
            (rsi_series[i] is not None and rsi_series[i] < p.rsi_gate)
        # 深度加碼：跌越深（相對長期高）每格買越多口（受控反馬丁，上限 3 口）
        cpl = p.contracts_per_lot
        if getattr(p, "depth_scale", 0) and getattr(p, "widen", 0) == 0 and ref_long_ds(i) > 0:
            drop = max(0.0, (ref_long_ds(i) - c) / ref_long_ds(i))
            cpl = min(3 * p.contracts_per_lot,
                      int(round(p.contracts_per_lot * (1 + p.depth_scale * drop))))
            cpl = max(cpl, p.contracts_per_lot)
        notional_after = notional(lots, c) + adj_notional(c, cpl)
        lev_ok = notional_after <= equity * p.max_leverage
        if (c <= ref * (1 - step_i) and len(lots) < p.max_lots and lev_ok and regime_ok
                and timing_ok and rsi_ok
                and not any(abs(l["price"] / c - 1) < step_i for l in lots)):
            realized -= p.fee_per_side * cpl  # 買進手續費
            mcf[ym] -= p.fee_per_side * cpl
            lots.append({"price": c, "contracts": cpl})
            mbuy[ym] += 1
            bought_today = True

        equity = capital_base + realized + unreal(lots, c)

        # 追蹤最緊的保證金餘裕（equity/exposure，越低越接近被追繳的維持保證金）
        if lots:
            exp0 = notional(lots, c)
            if exp0 > 0 and equity / exp0 < worst_margin[0]:
                worst_margin = (equity / exp0, dt[i])

        # 補錢模式：權益跌破維持保證金 → 補到原始保證金水位（不強制平倉）
        if p.topup and lots:
            exp = notional(lots, c)
            if equity < exp * p.maintenance_margin:
                deposit = exp * p.initial_margin - equity
                if deposit > 0:
                    capital_base += deposit
                    total_deposit += deposit
                    deposit_events += 1
                    equity += deposit

        curve.append(equity)
        if equity > 0:
            max_lev = max(max_lev, notional(lots, c) / equity)
        max_lots = max(max_lots, len(lots))
        end_lots[ym] = len(lots)

    return dict(dt=dt, curve=curve, realized=realized, mcf=mcf, mbuy=mbuy,
                msell=msell, max_lev=max_lev, max_lots=max_lots, end_lots=end_lots,
                lots_open=len(lots), total_deposit=total_deposit,
                deposit_events=deposit_events, peak_committed=p.capital + total_deposit,
                worst_margin=worst_margin)


def adj_notional(price, contracts):
    return price * MULTIPLIER * contracts


def notional(lots, price):
    return sum(adj_notional(price, l["contracts"]) for l in lots)


def unreal(lots, price):
    return sum((price - l["price"]) * MULTIPLIER * l["contracts"] for l in lots)


def months_of(dt):
    seen, s = [], set()
    for x in dt:
        ym = x[:7]
        if ym not in s:
            s.add(ym); seen.append(ym)
    return seen


def maxdd(curve):
    peak = curve[0]; dd = 0.0
    for e in curve:
        peak = max(peak, e); dd = min(dd, e / peak - 1)
    return dd * 100


def summary(p):
    r = run(p)
    dt, curve = r["dt"], r["curve"]
    ms = months_of(dt)
    cap = p.capital
    vals = [r["mcf"].get(m, 0.0) / cap * 100 for m in ms]
    pos = sum(1 for v in vals if v > 0.05)
    dry = mx = 0
    for v in vals:
        dry = dry + 1 if v < 0.3 else 0; mx = max(mx, dry)
    years = len(dt) / (BARS_PER_DAY * 252)
    tot = (curve[-1] / cap - 1) * 100
    cagr = ((curve[-1] / cap) ** (1 / years) - 1) * 100 if years > 0 else 0
    print(f"小型台積電期(QFF) 30分K 網格收租（{dt[0][:10]} ~ {dt[-1][:10]}，{len(ms)}月，"
          f"{len(dt)}根K）")
    tp_desc = f"移動停利{p.trail_tp:.1%}" if getattr(p, "trail_tp", 0) else f"固定+{p.take:.1%}"
    print(f"  參數：max {p.max_lots} 批 × {p.contracts_per_lot} 口／格{p.step:.1%}／"
          f"{tp_desc}／近{p.h_days}日高／本金{cap:,.0f}／槓桿上限{p.max_leverage}x")
    print(f"  總報酬 {tot:+.1f}%   CAGR {cagr:+.1f}%   最大回撤 {maxdd(curve):.1f}%")
    print(f"  已實現現金流 {r['realized']:+,.0f} 元（{r['realized']/cap*100:+.1f}% 本金）"
          f"｜期末未平倉 {r['lots_open']} 批")
    print(f"  月現金流：中位 {statistics.median(vals):+.2f}%  平均 {statistics.mean(vals):+.2f}%  "
          f"有現金流月 {pos}/{len(ms)}={pos/len(ms)*100:.0f}%  最長乾旱 {mx}月")
    print(f"  峰值槓桿 {r['max_lev']:.2f}x｜最多同時持倉 {r['max_lots']} 批")
    if p.topup:
        peak = r["peak_committed"]
        print(f"  ★補錢模式：追繳 {r['deposit_events']} 次，累計補錢 {r['total_deposit']:,.0f} 元"
              f"｜峰值投入資金 {peak:,.0f} 元（初始 {cap:,.0f} + 補 {r['total_deposit']:,.0f}）")
        print(f"  以峰值投入資金計：已實現現金流 {r['realized']/peak*100:+.1f}%"
              f"／年化約 {r['realized']/peak/years*100:+.1f}%")
    tb, ts = sum(r["mbuy"].values()), sum(r["msell"].values())
    print(f"  交易：買 {tb} / 賣 {ts}（月均 買{tb/len(ms):.1f}/賣{ts/len(ms):.1f}）")


def monthly(p):
    r = run(p)
    dt = r["dt"]; cap = p.capital
    print(f"{'月份':<9}{'買':>5}{'賣':>5}{'月底批':>7}{'現金流%':>10}")
    print("-" * 38)
    cur = None; ytot = 0.0
    for ym in months_of(dt):
        if cur and ym[:4] != cur:
            print(f"  {cur} 全年{'':>13}{ytot:>+9.2f}%"); print("-" * 38); ytot = 0.0
        cur = ym[:4]
        cf = r["mcf"].get(ym, 0.0) / cap * 100; ytot += cf
        print(f"{ym:<9}{r['mbuy'].get(ym,0):>5}{r['msell'].get(ym,0):>5}"
              f"{r['end_lots'][ym]:>7}{cf:>+9.2f}%")
    print(f"  {cur} 全年{'':>13}{ytot:>+9.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="summary", choices=["summary", "monthly"])
    ap.add_argument("--capital", type=float, default=1_000_000)
    ap.add_argument("--max-lots", type=int, default=20, dest="max_lots")
    ap.add_argument("--contracts-per-lot", type=int, default=1, dest="contracts_per_lot")
    ap.add_argument("--step", type=float, default=0.01)
    ap.add_argument("--take", type=float, default=0.01)
    ap.add_argument("--h-days", type=int, default=10, dest="h_days")
    ap.add_argument("--fee-per-side", type=float, default=DEFAULT_FEE_PER_SIDE, dest="fee_per_side")
    ap.add_argument("--roll-cost-points", type=float, default=DEFAULT_ROLL_COST_POINTS, dest="roll_cost_points")
    ap.add_argument("--max-leverage", type=float, default=1.0, dest="max_leverage")
    ap.add_argument("--regime-ma", type=int, default=0, dest="regime_ma",
                    help="趨勢濾網：僅在 adj收盤 > 近N根均線時開新倉（0=關；600≈60日）")
    ap.add_argument("--intraday-once", action="store_true", dest="intraday_once",
                    help="當天已買過後，只在收盤那根(13:15)再檢查是否加碼")
    ap.add_argument("--topup", action="store_true", dest="topup",
                    help="補錢模式：權益跌破維持保證金就補到原始保證金（不強平）")
    ap.add_argument("--initial-margin", type=float, default=0.135, dest="initial_margin")
    ap.add_argument("--maintenance-margin", type=float, default=0.1035, dest="maintenance_margin")
    ap.add_argument("--start", default=None, help="回測起始日 YYYY-MM-DD（策略從此日啟動）")
    ap.add_argument("--atr-mult", type=float, default=0.0, dest="atr_mult",
                    help="動態格距：格距=atr_mult×近期日波動（0=固定 --step）")
    ap.add_argument("--trail-tp", type=float, default=0.007, dest="trail_tp",
                    help="移動停利：達+take後追蹤高點、回落此比例才賣（預設0.7%；設0=固定停利）")
    ap.add_argument("--widen", type=float, default=0.0,
                    help="等比加寬網格：格距=step×(1+widen×跌深)，跌越深格距越寬（0=均勻）")
    ap.add_argument("--rsi-gate", type=float, default=0.0, dest="rsi_gate",
                    help="RSI超賣過濾：只在 RSI<此值 才吃網格買點（0=關）")
    ap.add_argument("--rsi-period", type=int, default=70, dest="rsi_period",
                    help="RSI 週期(30分K根數，預設70≈7日)")
    ap.add_argument("--depth-scale", type=float, default=0.0, dest="depth_scale",
                    help="深度加碼：口數=base×(1+depth_scale×跌深)，上限3口（0=固定口數）")
    p = ap.parse_args()
    (monthly if p.mode == "monthly" else summary)(p)


if __name__ == "__main__":
    main()
