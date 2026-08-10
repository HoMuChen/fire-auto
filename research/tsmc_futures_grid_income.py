"""
小型台積電期貨(QF) 再掛單網格收租策略回測。

這是 `research/tsmc_grid_income.py` 的期貨口數版：
- 價格先用 2330 現股收盤當期貨代理價，暫不處理基差、夜盤、轉倉轉約。
- 小型台積電期貨 1 口 = 100 股，1 點 = NT$100。
- 每次信號買 `--contracts-per-trade` 口；每個 lot 漲到買價 +TAKE 就整批平倉。
- 買進不扣名目本金，採每日盯市概念；用「名目曝險 / 權益」限制槓桿。
- max_leverage=1.0 代表無槓桿：總名目曝險不可超過帳戶權益。
- 若權益低於維持保證金，不強制平倉；從外部補錢到原始保證金水位。

用法：
  python3 research/tsmc_futures_grid_income.py
  python3 research/tsmc_futures_grid_income.py monthly --leverage 1.25
  python3 research/tsmc_futures_grid_income.py monthly --trades-out /tmp/tsmc_qf_trades.csv
  python3 research/tsmc_futures_grid_income.py --contracts-per-trade 2
  python3 research/tsmc_futures_grid_income.py --start 2024-01
  python3 research/tsmc_futures_grid_income.py --capital 1500000
  python3 research/tsmc_futures_grid_income.py --initial-margin 0.135
  python3 research/tsmc_futures_grid_income.py --roll-cost-points 10
  python3 research/tsmc_futures_grid_income.py --roll-cost-pct 0.003
  python3 research/tsmc_futures_grid_income.py --fee 20
"""
import argparse
import csv
import re
import statistics
import sys
import unicodedata
from collections import defaultdict

sys.path.insert(0, "/Users/mu/fire-auto")
import backtest as bt


STOCK = "2330"
DEFAULT_CAPITAL = 1_000_000
MULTIPLIER = 100
STEP = 0.01
TAKE = 0.01
H = 10

# 期交稅：股價類期貨通常以契約金額十萬分之二估算；買賣雙邊都收。
FUTURES_TAX = 0.00002
# 期貨手續費依券商不同。預設用 20 元/口/邊做保守敏感度測試，可用 --fee 調整。
DEFAULT_FEE_PER_SIDE = 20.0
DEFAULT_INITIAL_MARGIN = 0.135
DEFAULT_MAINTENANCE_MARGIN = 0.1035


def contract_notional(price, contracts=1):
    return price * MULTIPLIER * contracts


def trade_cost(price, fee_per_side, contracts=1):
    return contract_notional(price, contracts) * FUTURES_TAX + fee_per_side * contracts


def unrealized_pnl(lots, price):
    return sum((price - lot["price"]) * MULTIPLIER * lot["contracts"] for lot in lots)


def exposure(lots, price):
    return sum(contract_notional(price, lot["contracts"]) for lot in lots)


def total_contracts(lots):
    return sum(lot["contracts"] for lot in lots)


def normalize_start_date(value):
    if not value:
        return None
    if re.fullmatch(r"\d{4}-\d{2}", value):
        return value + "-01"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return value
    raise ValueError("--start 請使用 YYYY-MM 或 YYYY-MM-DD")


def filter_prices(prices, start_date):
    if not start_date:
        return prices
    filtered = [p for p in prices if p["date"] >= start_date]
    if not filtered:
        raise ValueError(f"--start {start_date} 晚於本地資料最後日期")
    return filtered


def positive_int(value):
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("必須是正整數")
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必須是正整數")
    return parsed


def validate_roll_cost_args(roll_cost_points, roll_cost_pct):
    if roll_cost_points and roll_cost_pct:
        raise ValueError("--roll-cost-points 和 --roll-cost-pct 只能擇一使用")
    if roll_cost_points < 0 or roll_cost_pct < 0:
        raise ValueError("轉倉成本不可為負數")


def display_width(value):
    text = str(value)
    return sum(2 if unicodedata.east_asian_width(ch) in ("F", "W") else 1 for ch in text)


def pad_cell(value, width, align="right"):
    text = str(value)
    padding = max(0, width - display_width(text))
    if align == "left":
        return text + " " * padding
    return " " * padding + text


def table_row(cells):
    return "".join(pad_cell(value, width, align) for value, width, align in cells)


def run(
    capital=DEFAULT_CAPITAL,
    max_leverage=1.0,
    fee_per_side=DEFAULT_FEE_PER_SIDE,
    initial_margin=DEFAULT_INITIAL_MARGIN,
    maintenance_margin=DEFAULT_MAINTENANCE_MARGIN,
    start_date=None,
    contracts_per_trade=1,
    roll_cost_points=0.0,
    roll_cost_pct=0.0,
):
    prices = filter_prices(bt.read_prices(STOCK), start_date)
    closes = [p["close"] for p in prices]

    cash = capital
    lots = []  # {price, contracts, open_cost}
    curve = []
    mbuy = defaultdict(int)
    msell = defaultdict(int)
    mcf = defaultdict(float)
    end_lots = {}
    end_equity = {}
    leverage_curve = []
    skipped_by_leverage = defaultdict(int)
    margin_deposit = defaultdict(float)
    margin_events = 0
    total_margin_deposit = 0.0
    max_single_deposit = 0.0
    trades = []

    for i, p in enumerate(prices):
        c = p["close"]
        ym = p["date"][:7]
        is_month_end = i == len(prices) - 1 or prices[i + 1]["date"][:7] != ym
        ref = max(closes[max(0, i - H):i + 1])

        # 平倉：逐 lot 漲到買價 +TAKE 就整批賣。
        rem = []
        for lot in lots:
            if c >= lot["price"] * (1 + TAKE):
                gross = (c - lot["price"]) * MULTIPLIER * lot["contracts"]
                costs = lot["open_cost"] + trade_cost(c, fee_per_side, lot["contracts"])
                pnl = gross - costs
                cash += pnl
                mcf[ym] += pnl
                msell[ym] += lot["contracts"]
                trades.append({
                    "date": p["date"],
                    "action": "sell",
                    "price": c,
                    "contracts": lot["contracts"],
                    "buy_price": lot["price"],
                    "gross_pnl": gross,
                    "cost": costs,
                    "net_pnl": pnl,
                    "cash_after": cash,
                    "equity_after": "",
                    "open_contracts_after": "",
                    "note": "take_profit",
                })
            else:
                rem.append(lot)
        lots[:] = rem

        # 權益先含未實現損益，再用槓桿上限判斷能不能新增一批口數。
        equity_before_buy = cash + unrealized_pnl(lots, c)
        can_add_exposure = exposure(lots, c) + contract_notional(c, contracts_per_trade) <= equity_before_buy * max_leverage

        if c <= ref * (1 - STEP):
            near = any(abs(lot["price"] / c - 1) < STEP for lot in lots)
            if not near and equity_before_buy > 0:
                if can_add_exposure:
                    open_cost = trade_cost(c, fee_per_side, contracts_per_trade)
                    cash -= open_cost
                    lots.append({"price": c, "contracts": contracts_per_trade, "open_cost": open_cost})
                    mbuy[ym] += contracts_per_trade
                    trades.append({
                        "date": p["date"],
                        "action": "buy",
                        "price": c,
                        "contracts": contracts_per_trade,
                        "buy_price": c,
                        "gross_pnl": "",
                        "cost": open_cost,
                        "net_pnl": -open_cost,
                        "cash_after": cash,
                        "equity_after": "",
                        "open_contracts_after": "",
                        "note": "grid_buy",
                    })
                else:
                    skipped_by_leverage[ym] += contracts_per_trade
                    trades.append({
                        "date": p["date"],
                        "action": "skip_buy",
                        "price": c,
                        "contracts": contracts_per_trade,
                        "buy_price": c,
                        "gross_pnl": "",
                        "cost": "",
                        "net_pnl": "",
                        "cash_after": cash,
                        "equity_after": "",
                        "open_contracts_after": total_contracts(lots),
                        "note": "leverage_limit",
                    })

        equity = cash + unrealized_pnl(lots, c)
        current_exposure = exposure(lots, c)
        open_contracts = total_contracts(lots)
        if is_month_end and open_contracts and (roll_cost_points or roll_cost_pct):
            roll_points = roll_cost_points or c * roll_cost_pct
            roll_cost = roll_points * MULTIPLIER * open_contracts
            cash -= roll_cost
            equity -= roll_cost
            mcf[ym] -= roll_cost
            trades.append({
                "date": p["date"],
                "action": "roll_cost",
                "price": c,
                "contracts": open_contracts,
                "buy_price": "",
                "gross_pnl": "",
                "cost": roll_cost,
                "net_pnl": -roll_cost,
                "cash_after": cash,
                "equity_after": equity,
                "open_contracts_after": open_contracts,
                "note": f"monthly_roll_cost_{roll_points:g}_points",
            })
            current_exposure = exposure(lots, c)
        maintenance_required = current_exposure * maintenance_margin
        if lots and equity < maintenance_required:
            target_equity = current_exposure * initial_margin
            deposit = max(0.0, target_equity - equity)
            if deposit > 0:
                cash += deposit
                equity += deposit
                margin_deposit[ym] += deposit
                total_margin_deposit += deposit
                max_single_deposit = max(max_single_deposit, deposit)
                margin_events += 1
                trades.append({
                    "date": p["date"],
                    "action": "margin_deposit",
                    "price": c,
                    "contracts": total_contracts(lots),
                    "buy_price": "",
                    "gross_pnl": "",
                    "cost": "",
                    "net_pnl": deposit,
                    "cash_after": cash,
                    "equity_after": equity,
                    "open_contracts_after": total_contracts(lots),
                    "note": "top_up_to_initial_margin",
                })
        curve.append(equity)
        leverage_curve.append(current_exposure / equity if equity > 0 else float("inf"))
        end_lots[ym] = total_contracts(lots)
        end_equity[ym] = equity
        if trades:
            for trade in reversed(trades):
                if trade["date"] != p["date"] or trade["equity_after"] != "":
                    break
                trade["equity_after"] = equity
                trade["open_contracts_after"] = total_contracts(lots)

    margin_info = {
        "deposit_by_month": margin_deposit,
        "total_deposit": total_margin_deposit,
        "events": margin_events,
        "max_single_deposit": max_single_deposit,
    }
    return prices, curve, mbuy, msell, mcf, end_lots, end_equity, leverage_curve, skipped_by_leverage, margin_info, trades


def months_of(prices):
    seen, s = [], set()
    for p in prices:
        ym = p["date"][:7]
        if ym not in s:
            s.add(ym)
            seen.append(ym)
    return seen


def maxdd(curve):
    peak = curve[0]
    dd = 0.0
    for e in curve:
        peak = max(peak, e)
        dd = min(dd, e / peak - 1)
    return dd * 100


def write_trades(path, trades):
    fields = [
        "date",
        "action",
        "price",
        "contracts",
        "buy_price",
        "gross_pnl",
        "cost",
        "net_pnl",
        "cash_after",
        "equity_after",
        "open_contracts_after",
        "note",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(trades)


def summarize(capital, max_leverage, fee_per_side, initial_margin, maintenance_margin, start_date, contracts_per_trade, roll_cost_points, roll_cost_pct):
    prices, curve, mbuy, msell, mcf, end_lots, end_equity, lev, skipped, margin_info, trades = run(
        capital, max_leverage, fee_per_side, initial_margin, maintenance_margin, start_date, contracts_per_trade, roll_cost_points, roll_cost_pct
    )
    ms = months_of(prices)
    cf_amounts = [mcf.get(m, 0.0) for m in ms]
    vals = [mcf.get(m, 0.0) / capital * 100 for m in ms]
    pos = sum(1 for v in vals if v > 0.05)
    dry = mx = 0
    for v in vals:
        dry = dry + 1 if v < 0.3 else 0
        mx = max(mx, dry)
    tot = (curve[-1] / capital - 1) * 100
    total_capital_used = capital + margin_info["total_deposit"]
    adjusted_tot = (curve[-1] / total_capital_used - 1) * 100
    cagr = ((curve[-1] / capital) ** (252 / len(prices)) - 1) * 100
    by_year = defaultdict(float)
    for m in ms:
        by_year[m[:4]] += mcf.get(m, 0.0) / capital * 100
    tb, ts = sum(mbuy.values()), sum(msell.values())
    return {
        "start": prices[0]["date"],
        "end": prices[-1]["date"],
        "months": len(ms),
        "capital": capital,
        "max_leverage": max_leverage,
        "fee_per_side": fee_per_side,
        "total_return": tot,
        "adjusted_return": adjusted_tot,
        "cagr": cagr,
        "maxdd": maxdd(curve),
        "median_cf": statistics.median(vals),
        "median_cf_amount": statistics.median(cf_amounts),
        "avg_cf": statistics.mean(vals),
        "avg_cf_amount": statistics.mean(cf_amounts),
        "positive_months": pos,
        "drought": mx,
        "year_cf": dict(sorted(by_year.items())),
        "buys": tb,
        "sells": ts,
        "final_equity": curve[-1],
        "final_contracts": end_lots[ms[-1]],
        "peak_leverage": max(lev),
        "peak_contracts": max(end_lots.values()) if end_lots else 0,
        "skipped": sum(skipped.values()),
        "margin_deposit": margin_info["total_deposit"],
        "margin_events": margin_info["events"],
        "max_single_deposit": margin_info["max_single_deposit"],
        "trades": trades,
    }


def roll_cost_label(roll_cost_points, roll_cost_pct):
    if roll_cost_pct:
        return f"{roll_cost_pct:.3%}/月/口"
    return f"{roll_cost_points:g}點/月/口"


def print_summary(capital, leverage_values, fee_per_side, initial_margin, maintenance_margin, start_date, contracts_per_trade, roll_cost_points, roll_cost_pct):
    first = summarize(capital, leverage_values[0], fee_per_side, initial_margin, maintenance_margin, start_date, contracts_per_trade, roll_cost_points, roll_cost_pct)
    print("小型台積電期貨 再掛單網格收租")
    print(f"  期間：{first['start']} ~ {first['end']}，{first['months']}月")
    print(f"  本金：{capital:,.0f} 元")
    print(f"  價格代理：2330 現股收盤；暫不處理基差/轉倉；1口=100股；每次信號買{contracts_per_trade}口")
    print(
        f"  參數：格{STEP:.0%} / 賣+{TAKE:.0%} / 近{H}日高 / "
        f"原始保證金{initial_margin:.1%} / 維持保證金{maintenance_margin:.1%} / "
        f"轉倉成本{roll_cost_label(roll_cost_points, roll_cost_pct)} / 期交稅{FUTURES_TAX*100:.3f}%/邊 / "
        f"手續費{fee_per_side:.0f}元/口/邊\n"
    )
    print(f"{'槓桿上限':>8} {'總報酬':>8} {'實報酬':>8} {'最大DD':>8} {'月中位CF':>11} {'期末權益':>12} {'買/賣口':>9} {'峰值口數':>8} {'峰值槓桿':>9} {'補錢':>10} {'最大單補':>10} {'補次':>5} {'略過口':>6}")
    print("-" * 138)
    for i, lev in enumerate(leverage_values):
        r = first if i == 0 else summarize(capital, lev, fee_per_side, initial_margin, maintenance_margin, start_date, contracts_per_trade, roll_cost_points, roll_cost_pct)
        print(
            f"{lev:>7.2f}x "
            f"{r['total_return']:>+7.0f}% "
            f"{r['adjusted_return']:>+7.0f}% "
            f"{r['maxdd']:>+7.1f}% "
            f"{r['median_cf_amount']:>+10,.0f} "
            f"{r['final_equity']:>11,.0f} "
            f"{r['buys']:>3}/{r['sells']:<3} "
            f"{r['peak_contracts']:>8} "
            f"{r['peak_leverage']:>8.2f}x "
            f"{r['margin_deposit']:>9,.0f} "
            f"{r['max_single_deposit']:>9,.0f} "
            f"{r['margin_events']:>5} "
            f"{r['skipped']:>6}"
        )
    return first


def print_monthly(capital, max_leverage, fee_per_side, initial_margin, maintenance_margin, start_date, contracts_per_trade, roll_cost_points, roll_cost_pct):
    prices, curve, mbuy, msell, mcf, end_lots, end_equity, lev, skipped, margin_info, trades = run(
        capital, max_leverage, fee_per_side, initial_margin, maintenance_margin, start_date, contracts_per_trade, roll_cost_points, roll_cost_pct
    )
    deposits = margin_info["deposit_by_month"]
    print(
        f"小型台積電期貨網格逐月明細（本金 {capital:,.0f}，每次{contracts_per_trade}口，槓桿上限 {max_leverage:.2f}x，"
        f"原始保證金 {initial_margin:.1%}，維持保證金 {maintenance_margin:.1%}，"
        f"轉倉成本 {roll_cost_label(roll_cost_points, roll_cost_pct)}，手續費 {fee_per_side:.0f}/口/邊）"
    )
    widths = {
        "month": 12,
        "buy": 6,
        "sell": 6,
        "lots": 10,
        "cashflow": 14,
        "deposit": 12,
        "equity": 14,
        "skip": 8,
    }
    print(table_row([
        ("月份", widths["month"], "left"),
        ("買口", widths["buy"], "right"),
        ("賣口", widths["sell"], "right"),
        ("月底口數", widths["lots"], "right"),
        ("現金流", widths["cashflow"], "right"),
        ("補錢", widths["deposit"], "right"),
        ("月底權益", widths["equity"], "right"),
        ("略過口", widths["skip"], "right"),
    ]))
    print("-" * sum(widths.values()))
    cur = None
    ytot = 0.0
    ydeposit = 0.0
    for ym in months_of(prices):
        if cur and ym[:4] != cur:
            print(table_row([
                (f"{cur} 全年", widths["month"], "left"),
                ("", widths["buy"], "right"),
                ("", widths["sell"], "right"),
                ("", widths["lots"], "right"),
                (f"{ytot:+,.0f}", widths["cashflow"], "right"),
                (f"{ydeposit:,.0f}", widths["deposit"], "right"),
                ("", widths["equity"], "right"),
                ("", widths["skip"], "right"),
            ]))
            print("-" * sum(widths.values()))
            ytot = 0.0
            ydeposit = 0.0
        cur = ym[:4]
        cf = mcf.get(ym, 0.0)
        ytot += cf
        dep = deposits.get(ym, 0.0)
        ydeposit += dep
        print(table_row([
            (ym, widths["month"], "left"),
            (mbuy.get(ym, 0), widths["buy"], "right"),
            (msell.get(ym, 0), widths["sell"], "right"),
            (end_lots[ym], widths["lots"], "right"),
            (f"{cf:+,.0f}", widths["cashflow"], "right"),
            (f"{dep:,.0f}", widths["deposit"], "right"),
            (f"{end_equity[ym]:,.0f}", widths["equity"], "right"),
            (skipped.get(ym, 0), widths["skip"], "right"),
        ]))
    print(table_row([
        (f"{cur} 全年", widths["month"], "left"),
        ("", widths["buy"], "right"),
        ("", widths["sell"], "right"),
        ("", widths["lots"], "right"),
        (f"{ytot:+,.0f}", widths["cashflow"], "right"),
        (f"{ydeposit:,.0f}", widths["deposit"], "right"),
        ("", widths["equity"], "right"),
        ("", widths["skip"], "right"),
    ]))
    total_capital_used = capital + margin_info["total_deposit"]
    adjusted_return = (curve[-1] / total_capital_used - 1) * 100
    print(
        f"\n期末權益 {curve[-1]:,.0f}，累計補錢 {margin_info['total_deposit']:,.0f}，"
        f"實報酬 {adjusted_return:+.1f}%，峰值槓桿 {max(lev):.2f}x"
    )
    return trades


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs="?", choices=["summary", "monthly"], default="summary")
    parser.add_argument("--start", default=None, help="起始月份或日期，格式 YYYY-MM 或 YYYY-MM-DD")
    parser.add_argument("--capital", type=float, default=DEFAULT_CAPITAL, help="初始本金")
    parser.add_argument("--contracts-per-trade", type=positive_int, default=1, help="每次買入信號下單口數")
    parser.add_argument("--trades-out", default=None, help="輸出交易明細 CSV；summary 模式輸出 leverage-list 第一組")
    parser.add_argument("--roll-cost-points", type=float, default=0.0, help="每月轉倉成本點數，按月底未平倉口數扣除")
    parser.add_argument("--roll-cost-pct", type=float, default=0.0, help="每月轉倉成本比例，按月底價格與未平倉口數扣除，例如 0.003 = 0.3%")
    parser.add_argument("--leverage", type=float, default=1.0, help="monthly 模式使用的槓桿上限")
    parser.add_argument("--fee", type=float, default=DEFAULT_FEE_PER_SIDE, help="每口每邊期貨手續費")
    parser.add_argument("--initial-margin", type=float, default=DEFAULT_INITIAL_MARGIN, help="原始保證金率；補錢補到此水位")
    parser.add_argument("--maintenance-margin", type=float, default=DEFAULT_MAINTENANCE_MARGIN, help="維持保證金率；低於此水位觸發補錢")
    parser.add_argument(
        "--leverage-list",
        default="1.0,1.25,1.5,2.0",
        help="summary 模式比較的槓桿上限，逗號分隔",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        start_date = normalize_start_date(args.start)
        validate_roll_cost_args(args.roll_cost_points, args.roll_cost_pct)
    except ValueError as exc:
        raise SystemExit(str(exc))
    if args.mode == "monthly":
        try:
            trades = print_monthly(
                args.capital,
                args.leverage,
                args.fee,
                args.initial_margin,
                args.maintenance_margin,
                start_date,
                args.contracts_per_trade,
                args.roll_cost_points,
                args.roll_cost_pct,
            )
            if args.trades_out:
                write_trades(args.trades_out, trades)
                print(f"交易明細已輸出：{args.trades_out}")
        except ValueError as exc:
            raise SystemExit(str(exc))
    else:
        leverage_values = [float(x) for x in args.leverage_list.split(",") if x.strip()]
        try:
            first = print_summary(
                args.capital,
                leverage_values,
                args.fee,
                args.initial_margin,
                args.maintenance_margin,
                start_date,
                args.contracts_per_trade,
                args.roll_cost_points,
                args.roll_cost_pct,
            )
            if args.trades_out:
                write_trades(args.trades_out, first["trades"])
                print(f"交易明細已輸出：{args.trades_out}")
        except ValueError as exc:
            raise SystemExit(str(exc))
