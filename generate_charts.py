"""批次產生 watchlist 股票的策略圖表 HTML"""
import json, sys
sys.path.insert(0, '.')
from backtest import *
from pathlib import Path

STRATEGIES_DIR = Path("strategies")

# Watchlist: (stock_id, stock_name, strategy_name)
WATCHLIST = [
    ("5225", "東科-KY", "雙底反彈"),
    ("7516", "清淨海", "威廉%R(14)"),
    ("6621", "華宇藥", "ATR通道回歸"),
    ("3227", "原相", "RSI+Bollinger"),
    ("6114", "久威", "布林觸底→中線"),
    ("8105", "凌巨", "RSI階梯[40,30,20]→55"),
    ("2362", "藍天", "ATR通道回歸"),
    ("3708", "上緯投控", "RSI階梯[40,30,20]→55"),
    ("4749", "新應材", "RSI階梯[40,30,20]→55"),
    ("2637", "慧洋-KY", "RSI背離反彈"),
    ("6829", "千附精密", "RSI(7)快閃"),
    ("6165", "浪凡", "布林觸底→中線"),
    ("7825", "和亞智慧", "快速網格(3%/4%)"),
]

# Indicator overlay config per strategy
def get_overlays(strategy_name, prices):
    closes = [p["close"] for p in prices]
    n = len(prices)

    if strategy_name in ("ATR通道回歸",):
        sma20 = calc_sma(closes, 20)
        atr_vals = calc_atr(prices, 14)
        lower = [None]*n
        for i in range(n):
            if sma20[i] and atr_vals[i]:
                lower[i] = round(sma20[i] - 1.5 * atr_vals[i], 2)
        return {
            "lines": [
                {"name": "SMA(20)", "color": "#f59e0b", "data": [round(v, 2) if v else None for v in sma20]},
                {"name": "下軌 (SMA−1.5×ATR)", "color": "rgba(99,102,241,0.5)", "style": 2, "data": lower},
            ]
        }

    if strategy_name in ("RSI+Bollinger", "布林觸底→中線"):
        upper, lower, mid = calc_bollinger(closes, 20, 2)
        return {
            "lines": [
                {"name": "布林中線", "color": "#f59e0b", "data": [round(v, 2) if v else None for v in mid]},
                {"name": "布林上軌", "color": "rgba(239,68,68,0.4)", "style": 2, "data": [round(v, 2) if v else None for v in upper]},
                {"name": "布林下軌", "color": "rgba(34,197,94,0.4)", "style": 2, "data": [round(v, 2) if v else None for v in lower]},
            ]
        }

    # Default: just SMA(20)
    sma20 = calc_sma(closes, 20)
    return {
        "lines": [
            {"name": "SMA(20)", "color": "#f59e0b", "data": [round(v, 2) if v else None for v in sma20]},
        ]
    }


def run_simulation(prices, strategy_name):
    cfg = STRATEGIES[strategy_name]
    signals = cfg["fn"](prices)
    stop_config = cfg["stop"]
    exit_rule = cfg.get("exit", {"type": "signal"})

    capital = INITIAL_CAPITAL
    shares = 0
    lots = []
    position_open = False
    trades = []
    equity_curve = []
    stop_price = 0
    peak_price = 0
    entry_idx = 0

    atr = None
    if stop_config["type"] == "atr":
        atr = calc_atr(prices, stop_config.get("period", 14))

    def calc_avg_buy():
        if shares == 0: return 0
        return sum(l["shares"] * l["buy_price"] for l in lots) / shares

    def update_stop(bar_idx, avg_price):
        nonlocal stop_price, peak_price
        if stop_config["type"] == "fixed_pct":
            stop_price = avg_price * (1 - stop_config["pct"])
        elif stop_config["type"] == "atr":
            entry_atr = atr[bar_idx] if (atr and atr[bar_idx] is not None) else avg_price * 0.05
            stop_price = avg_price - stop_config["multiplier"] * entry_atr
            stop_price = max(stop_price, 0)
        elif stop_config["type"] == "trailing_pct":
            if peak_price == 0: peak_price = avg_price
            stop_price = peak_price * (1 - stop_config["pct"])

    def close_pos(close_price, date, reason):
        nonlocal capital, shares, lots, position_open, stop_price, peak_price
        avg_buy = calc_avg_buy()
        rev = shares * close_price
        fee = int(rev * SELL_FEE)
        tax = int(rev * SELL_TAX)
        capital += (rev - fee - tax)
        ret_pct = (close_price / avg_buy - 1) * 100 if avg_buy > 0 else 0
        trades.append({
            "buy_date": lots[0]["buy_date"], "buy_price": round(avg_buy, 2),
            "sell_date": date, "sell_price": close_price,
            "ret_pct": round(ret_pct, 2), "reason": reason,
        })
        shares = 0; lots = []; position_open = False; stop_price = 0; peak_price = 0

    for i, p in enumerate(prices):
        close = p["close"]
        sig = signals[i]
        if isinstance(sig, tuple):
            action, fraction = sig[0], sig[1]
        elif sig is not None:
            action = sig; fraction = 1.0
        else:
            action = None; fraction = 0

        # Trailing stop update
        if position_open and stop_config["type"] == "trailing_pct":
            if close > peak_price:
                peak_price = close
                stop_price = peak_price * (1 - stop_config["pct"])

        # 1. Stop loss
        if position_open and close <= stop_price:
            close_pos(close, p["date"], "stop_loss")
            equity_curve.append(capital)
            continue

        # 2. Exit rule
        if position_open:
            should_exit = False
            etype = exit_rule["type"]
            if etype == "signal":
                if action == "sell": should_exit = True
            elif etype == "hold_days":
                if (i - entry_idx) >= exit_rule["days"]: should_exit = True
            elif etype == "profit_or_hold":
                avg_buy = calc_avg_buy()
                if close >= avg_buy * (1 + exit_rule["profit_pct"]): should_exit = True
                elif (i - entry_idx) >= exit_rule["days"]: should_exit = True
            elif etype == "gap_fill":
                if entry_idx > 0:
                    gap_top = prices[entry_idx - 1]["close"]
                    if close >= gap_top: should_exit = True
                if (i - entry_idx) >= exit_rule["days"]: should_exit = True
            if should_exit:
                close_pos(close, p["date"], "signal")

        # 3. Entry
        if not position_open and action == "buy" and close > 0 and capital > 0:
            spend = capital * fraction
            cost_per = close * (1 + BUY_FEE)
            new_shares = int(spend / cost_per / 1000) * 1000
            if new_shares <= 0: new_shares = int(spend / cost_per)
            if new_shares > 0:
                buy_cost = new_shares * close
                fee = int(buy_cost * BUY_FEE)
                capital -= (buy_cost + fee)
                shares += new_shares
                lots.append({"shares": new_shares, "buy_price": close, "buy_date": p["date"]})
                position_open = True
                entry_idx = i
                update_stop(i, calc_avg_buy())

        # 4. Add-on (RSI ladder)
        elif position_open and isinstance(sig, tuple) and sig[0] == "buy" and close > 0 and capital > 0:
            frac = sig[1]
            spend = capital * frac
            cost_per = close * (1 + BUY_FEE)
            new_shares = int(spend / cost_per / 1000) * 1000
            if new_shares <= 0: new_shares = int(spend / cost_per)
            if new_shares > 0:
                buy_cost = new_shares * close
                fee = int(buy_cost * BUY_FEE)
                capital -= (buy_cost + fee)
                shares += new_shares
                lots.append({"shares": new_shares, "buy_price": close, "buy_date": p["date"]})
                update_stop(i, calc_avg_buy())

        if position_open:
            equity_curve.append(capital + shares * close)
        else:
            equity_curve.append(capital)

    # Force close
    if position_open and shares > 0:
        close_pos(prices[-1]["close"], prices[-1]["date"], "forced_close")
        equity_curve[-1] = capital

    return trades, equity_curve


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{TITLE}}</title>
<script src="https://cdn.jsdelivr.net/npm/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #0a0e17; color: #d1d5db; font-family: -apple-system, 'Noto Sans TC', sans-serif; }
  .header { padding: 24px 32px 16px; border-bottom: 1px solid #1e2433; }
  .header h1 { font-size: 22px; color: #f0f0f0; font-weight: 600; }
  .header .subtitle { color: #8892a4; font-size: 13px; margin-top: 4px; }
  .stats-bar { display: flex; gap: 32px; padding: 16px 32px; border-bottom: 1px solid #1e2433; flex-wrap: wrap; }
  .stat { display: flex; flex-direction: column; }
  .stat .label { font-size: 11px; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px; }
  .stat .value { font-size: 18px; font-weight: 600; margin-top: 2px; }
  .stat .value.green { color: #22c55e; }
  .stat .value.red { color: #ef4444; }
  .stat .value.blue { color: #3b82f6; }
  .chart-section { padding: 16px 32px 8px; }
  .chart-section h2 { font-size: 14px; color: #8892a4; margin-bottom: 8px; font-weight: 500; }
  .chart-container { border: 1px solid #1e2433; border-radius: 8px; overflow: hidden; background: #0f1320; }
  .legend { display: flex; gap: 20px; padding: 10px 16px; background: #0f1320; border-bottom: 1px solid #1e2433; font-size: 12px; flex-wrap: wrap; }
  .legend-item { display: flex; align-items: center; gap: 6px; }
  .legend-line { width: 16px; height: 2px; }
  .legend-marker { width: 0; height: 0; border-left: 5px solid transparent; border-right: 5px solid transparent; }
  .legend-marker.buy { border-bottom: 8px solid #22c55e; }
  .legend-marker.sell { border-top: 8px solid #ef4444; }
  .legend-marker.stop { border-top: 8px solid #f59e0b; }
  .trades-table { padding: 16px 32px 32px; }
  .trades-table h2 { font-size: 14px; color: #8892a4; margin-bottom: 8px; font-weight: 500; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { text-align: left; padding: 8px 12px; background: #141825; color: #6b7280; font-weight: 500; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 1px solid #1e2433; }
  td { padding: 8px 12px; border-bottom: 1px solid #1a1f2e; }
  tr:hover td { background: #141825; }
  .win { color: #22c55e; }
  .lose { color: #ef4444; }
  .stop-tag { display: inline-block; padding: 1px 6px; border-radius: 3px; background: #f59e0b22; color: #f59e0b; font-size: 11px; }
  .signal-tag { display: inline-block; padding: 1px 6px; border-radius: 3px; background: #3b82f622; color: #3b82f6; font-size: 11px; }
  .forced-tag { display: inline-block; padding: 1px 6px; border-radius: 3px; background: #6b728022; color: #6b7280; font-size: 11px; }
  .nav { padding: 12px 32px; border-bottom: 1px solid #1e2433; display: flex; gap: 8px; flex-wrap: wrap; }
  .nav a { color: #3b82f6; text-decoration: none; font-size: 12px; padding: 4px 10px; border: 1px solid #1e2433; border-radius: 4px; }
  .nav a:hover { background: #1e2433; }
  .nav a.current { background: #3b82f622; border-color: #3b82f6; }
</style>
</head>
<body>

<div class="nav">{{NAV_LINKS}}</div>

<div class="header">
  <h1>{{TITLE}}</h1>
  <div class="subtitle">{{SUBTITLE}}</div>
</div>

<div class="stats-bar" id="stats-bar"></div>

<div class="chart-section">
  <h2>股價走勢 + 進出場點位</h2>
  <div class="chart-container">
    <div class="legend">
      <div class="legend-item"><div class="legend-line" style="background:#3b82f6"></div> 收盤價</div>
      {{LEGEND_ITEMS}}
      <div class="legend-item"><div class="legend-marker buy"></div> 買入</div>
      <div class="legend-item"><div class="legend-marker sell"></div> 賣出</div>
      <div class="legend-item"><div class="legend-marker stop"></div> 停損</div>
    </div>
    <div id="price-chart" style="height: 420px;"></div>
  </div>
</div>

<div class="chart-section">
  <h2>資金成長曲線</h2>
  <div class="chart-container">
    <div id="equity-chart" style="height: 260px;"></div>
  </div>
</div>

<div class="trades-table">
  <h2>交易明細</h2>
  <table id="trades-tbody"></table>
</div>

<script>
const DATA = {{DATA_JSON}};

const stats = document.getElementById('stats-bar');
const totalRet = ((DATA.equity[DATA.equity.length-1] - 1000000) / 1000000 * 100).toFixed(1);
const wins = DATA.trades.filter(t => t.ret_pct > 0).length;
const losses = DATA.trades.length - wins;
const stops = DATA.trades.filter(t => t.reason === 'stop_loss').length;
const winRate = (wins / DATA.trades.length * 100).toFixed(1);
const avgRet = (DATA.trades.reduce((s,t) => s + t.ret_pct, 0) / DATA.trades.length).toFixed(2);

const statItems = [
  { label: '總報酬', value: `+${totalRet}%`, cls: 'green' },
  { label: '最終資金', value: DATA.equity[DATA.equity.length-1].toLocaleString(), cls: 'blue' },
  { label: '交易次數', value: DATA.trades.length, cls: '' },
  { label: '勝率', value: `${winRate}%`, cls: parseFloat(winRate) >= 50 ? 'green' : 'red' },
  { label: '勝 / 負 / 停損', value: `${wins} / ${losses} / ${stops}`, cls: '' },
  { label: '平均報酬', value: `${avgRet}%`, cls: parseFloat(avgRet) >= 0 ? 'green' : 'red' },
];
stats.innerHTML = statItems.map(s =>
  `<div class="stat"><span class="label">${s.label}</span><span class="value ${s.cls}">${s.value}</span></div>`
).join('');

// Price Chart
const priceEl = document.getElementById('price-chart');
const priceChart = LightweightCharts.createChart(priceEl, {
  layout: { background: { color: '#0f1320' }, textColor: '#6b7280', fontSize: 11 },
  grid: { vertLines: { color: '#1a1f2e' }, horzLines: { color: '#1a1f2e' } },
  crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
  rightPriceScale: { borderColor: '#1e2433' },
  timeScale: { borderColor: '#1e2433', timeVisible: false },
});

const closeSeries = priceChart.addLineSeries({ color: '#3b82f6', lineWidth: 2, priceLineVisible: false, lastValueVisible: false });
closeSeries.setData(DATA.dates.map((d, i) => ({ time: d, value: DATA.closes[i] })));

// Overlay lines
DATA.overlays.forEach(ov => {
  const s = priceChart.addLineSeries({
    color: ov.color, lineWidth: 1, lineStyle: ov.style || 0,
    priceLineVisible: false, lastValueVisible: false,
  });
  s.setData(DATA.dates.map((d, i) => ov.data[i] !== null ? { time: d, value: ov.data[i] } : null).filter(Boolean));
});

// Markers
const buyMarkers = DATA.trades.map(t => ({
  time: t.buy_date, position: 'belowBar', color: '#22c55e', shape: 'arrowUp',
  text: `B ${t.buy_price}`,
}));
const sellMarkers = DATA.trades.map(t => ({
  time: t.sell_date, position: 'aboveBar',
  color: t.reason === 'stop_loss' ? '#f59e0b' : '#ef4444', shape: 'arrowDown',
  text: `${t.reason === 'stop_loss' ? 'SL' : 'S'} ${t.sell_price}`,
}));
closeSeries.setMarkers([...buyMarkers, ...sellMarkers].sort((a,b) => a.time < b.time ? -1 : 1));
priceChart.timeScale().fitContent();

// Equity Chart
const eqEl = document.getElementById('equity-chart');
const eqChart = LightweightCharts.createChart(eqEl, {
  layout: { background: { color: '#0f1320' }, textColor: '#6b7280', fontSize: 11 },
  grid: { vertLines: { color: '#1a1f2e' }, horzLines: { color: '#1a1f2e' } },
  crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
  rightPriceScale: { borderColor: '#1e2433' },
  timeScale: { borderColor: '#1e2433', timeVisible: false },
});
const eqSeries = eqChart.addAreaSeries({
  topColor: 'rgba(34,197,94,0.3)', bottomColor: 'rgba(34,197,94,0.02)',
  lineColor: '#22c55e', lineWidth: 2, priceLineVisible: false, lastValueVisible: true,
});
eqSeries.setData(DATA.dates.map((d, i) => ({ time: d, value: DATA.equity[i] })));
const baseSeries = eqChart.addLineSeries({
  color: 'rgba(107,114,128,0.3)', lineWidth: 1, lineStyle: 2,
  priceLineVisible: false, lastValueVisible: false,
});
baseSeries.setData(DATA.dates.map(d => ({ time: d, value: 1000000 })));
eqChart.timeScale().fitContent();

// Trades Table
const tableEl = document.getElementById('trades-tbody');
let html = `<thead><tr><th>#</th><th>買入日期</th><th>買入價</th><th>賣出日期</th><th>賣出價</th><th>持有天數</th><th>報酬</th><th>出場</th><th>累計資金</th></tr></thead><tbody>`;
DATA.trades.forEach((t, i) => {
  const holdDays = Math.round((new Date(t.sell_date) - new Date(t.buy_date)) / 86400000);
  const sellIdx = DATA.dates.indexOf(t.sell_date);
  const eqAtSell = sellIdx >= 0 ? DATA.equity[sellIdx].toLocaleString() : '';
  const retCls = t.ret_pct > 0 ? 'win' : 'lose';
  const tagMap = { stop_loss: 'stop-tag">停損', signal: 'signal-tag">訊號', forced_close: 'forced-tag">強制平倉' };
  const tag = tagMap[t.reason] || 'signal-tag">' + t.reason;
  html += `<tr><td>${i+1}</td><td>${t.buy_date}</td><td>${t.buy_price.toFixed(2)}</td><td>${t.sell_date}</td><td>${t.sell_price.toFixed(2)}</td><td>${holdDays}天</td><td class="${retCls}">${t.ret_pct > 0 ? '+' : ''}${t.ret_pct.toFixed(2)}%</td><td><span class="${tag}</span></td><td>${eqAtSell}</td></tr>`;
});
html += '</tbody>';
tableEl.innerHTML = html;

function handleResize() {
  priceChart.applyOptions({ width: priceEl.clientWidth });
  eqChart.applyOptions({ width: eqEl.clientWidth });
}
window.addEventListener('resize', handleResize);
handleResize();
</script>
</body>
</html>"""


def generate_chart(stock_id, stock_name, strategy_name):
    prices = read_prices(stock_id)
    cfg = STRATEGIES[strategy_name]

    # Run simulation
    trades, equity = run_simulation(prices, strategy_name)

    # Get overlays
    overlays = get_overlays(strategy_name, prices)

    # Build data object
    data = {
        "dates": [p["date"] for p in prices],
        "closes": [p["close"] for p in prices],
        "overlays": overlays["lines"],
        "equity": [round(v) for v in equity],
        "trades": trades,
    }

    # Build nav links
    nav_links = []
    for sid, sname, sn in WATCHLIST:
        fname = f"{sid}_{sname}_chart.html"
        cls = ' class="current"' if sid == stock_id else ''
        nav_links.append(f'<a href="{fname}"{cls}>{sname}({sid})</a>')
    nav_html = "\n".join(nav_links)

    # Build legend items for overlays
    legend_items = []
    for ov in overlays["lines"]:
        style = "opacity:0.5;" if ov.get("style") == 2 else ""
        legend_items.append(f'<div class="legend-item"><div class="legend-line" style="background:{ov["color"]};{style}"></div> {ov["name"]}</div>')
    legend_html = "\n      ".join(legend_items)

    title = f'{stock_name}({stock_id}) — {strategy_name}'
    subtitle = f'{cfg.get("desc", "")} ｜ {cfg.get("stop_desc", "")} ｜ 2024-02-15 ~ 2026-02-25'

    html = HTML_TEMPLATE
    html = html.replace("{{TITLE}}", title)
    html = html.replace("{{SUBTITLE}}", subtitle)
    html = html.replace("{{NAV_LINKS}}", nav_html)
    html = html.replace("{{LEGEND_ITEMS}}", legend_html)
    html = html.replace("{{DATA_JSON}}", json.dumps(data, ensure_ascii=False))

    out_path = STRATEGIES_DIR / f"{stock_id}_{stock_name}_chart.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path, len(trades)


if __name__ == "__main__":
    for sid, sname, sn in WATCHLIST:
        try:
            path, n_trades = generate_chart(sid, sname, sn)
            print(f"  {sname}({sid}) — {sn}: {n_trades} trades → {path.name}")
        except Exception as e:
            print(f"  {sname}({sid}) ERROR: {e}")
    print(f"\nDone! Generated {len(WATCHLIST)} chart files.")
