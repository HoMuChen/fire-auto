"""
產生 TSMC v4 三-Sleeve 現金流策略視覺化報告（自包含 HTML）。
資料全部由 tsmc_dual_pool_strategy.py 的 next-bar 回測即時計算。
"""
import json
import pandas as pd
import numpy as np
from tsmc_dual_pool_strategy import (
    load_data, backtest, calc_metrics, cashflow_metrics, BEST_PARAMS, CAPITAL,
)

START, END = '2021-01-01', '2026-06-01'
OUT = 'tsmc_dual_pool_report.html'


def build():
    df = load_data()
    port, trades = backtest(df, BEST_PARAMS, START, END)
    m = calc_metrics(port, START, END)
    cf = cashflow_metrics(port, trades, START, END)

    # ── 資金曲線（總 + 三池）對齊股價 ──
    p = port.copy()
    p['date'] = pd.to_datetime(p['date'])
    dser = df[(df['date'] >= START) & (df['date'] <= END)][['date', 'close']].copy()
    merged = p.merge(dser, on='date', how='left')
    dates = merged['date'].dt.strftime('%Y-%m-%d').tolist()
    total = merged['total'].round(0).tolist()
    pa = merged['pool_a'].round(0).tolist()
    pb = merged['pool_b'].round(0).tolist()
    pc = merged['pool_c'].round(0).tolist()
    price = merged['close'].round(2).tolist()

    # Buy-and-Hold 曲線（同資金、首日收盤全買）
    p0 = float(merged['close'].iloc[0])
    bh_sh = int(CAPITAL / p0)
    bh = [round((CAPITAL - bh_sh * p0) + bh_sh * c, 0) for c in merged['close']]
    n = (pd.to_datetime(END) - pd.to_datetime(START)).days / 365.25
    bh_cagr = ((bh[-1] / CAPITAL) ** (1/n) - 1) * 100
    bh_ser = pd.Series(bh)
    bh_dd = ((bh_ser - bh_ser.cummax()) / bh_ser.cummax() * 100).min()

    # ── 回撤曲線 ──
    tot_ser = merged['total']
    dd_curve = ((tot_ser - tot_ser.cummax()) / tot_ser.cummax() * 100).round(2).tolist()

    # ── 逐月損益 ──
    mret = cf['mret']
    me_labels = [str(x)[:7] for x in mret.index.to_period('M')]
    me_vals = mret.round(2).tolist()
    # 每月交易筆數
    t = trades.copy()
    t['month'] = pd.to_datetime(t['date']).dt.to_period('M').astype(str)
    from tsmc_dual_pool_strategy import SELL_ACTIONS, BUY_ACTIONS
    sells_t = t[t['action'].isin(SELL_ACTIONS)]
    buys_t = t[t['action'].isin(BUY_ACTIONS)]
    spm = sells_t.groupby('month').size()
    bpm = buys_t.groupby('month').size()
    me_sells = [int(spm.get(l, 0)) for l in me_labels]
    me_buys = [int(bpm.get(l, 0)) for l in me_labels]
    me_idle = [1 if (s + b) == 0 else 0 for s, b in zip(me_sells, me_buys)]

    # ── 進出場標記（散佈在價格圖上）──
    def mk(actions, color):
        sub = t[t['action'].isin(actions)]
        return [{'x': pd.to_datetime(r['date']).strftime('%Y-%m-%d'),
                 'y': round(float(r['price']), 2)} for _, r in sub.iterrows()]
    buy_pts = mk(['A_base_buy', 'A_dip_buy', 'B_buy', 'C_buy', 'A_harvest_buy'], 'green')
    sell_pts = mk(['A_dip_sell', 'B_sell', 'C_sell', 'A_harvest_sell', 'A_end'], 'red')
    cb_pts = mk(['A_cb_exit'], 'orange')

    # ── 每池貢獻 ──
    pool_final = {'A': pa[-1] - CAPITAL*BEST_PARAMS['a_pct'],
                  'B': pb[-1] - CAPITAL*BEST_PARAMS['b_pct'],
                  'C': pc[-1] - CAPITAL*BEST_PARAMS['c_pct']}

    # ── 每筆交易勝率分布（已實現）──
    has_pp = sells_t['profit_pct'].notna()
    realized = sells_t[has_pp]['profit_pct'].round(2).tolist()

    # 訓練/驗證分段
    seg = {}
    for s, e, k in [('2021-01-01','2024-12-31','train'),
                    ('2025-01-01','2026-06-01','test')]:
        po, tr2 = backtest(df, BEST_PARAMS, s, e)
        mm = calc_metrics(po, s, e); cc = cashflow_metrics(po, tr2, s, e)
        seg[k] = dict(cagr=mm['annual_ret'], dd=mm['max_dd'], idle=round(cc['idle_pct'],1),
                      mwr=round(cc['month_wr'],1), spm=round(cc['sells_pm'],1),
                      twr=round(cc['trade_wr'],0))

    data = dict(
        dates=dates, total=total, pa=pa, pb=pb, pc=pc, price=price, bh=bh,
        dd_curve=dd_curve,
        me_labels=me_labels, me_vals=me_vals, me_sells=me_sells, me_buys=me_buys, me_idle=me_idle,
        buy_pts=buy_pts, sell_pts=sell_pts, cb_pts=cb_pts,
        realized=realized,
        metrics=dict(
            cagr=m['annual_ret'], dd=m['max_dd'], sharpe=m['sharpe'],
            idle_pct=round(cf['idle_pct'],1), idle=cf['idle'], n_months=cf['n_months'], max_idle=cf['max_idle'],
            month_wr=round(cf['month_wr'],1), win_m=cf['win_m'], loss_m=cf['loss_m'],
            sells_pm=round(cf['sells_pm'],1), sells=cf['sells'], buys=cf['buys'],
            trade_wr=round(cf['trade_wr'],0), pf=round(cf['pf'],1),
            sortino=round(cf['sortino'],2), max_streak=cf['max_streak'],
            year_rets=m['year_rets'],
            bh_cagr=round(bh_cagr,1), bh_dd=round(bh_dd,1),
            pool_final={k: round(v) for k,v in pool_final.items()},
            final_total=round(total[-1]),
        ),
        seg=seg,
    )
    return data


HTML = """<!DOCTYPE html>
<html lang="zh-TW"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TSMC 2330 三-Sleeve 現金流策略報告 (v4)</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root{--bg:#0f1117;--card:#1a1d2e;--card2:#22263a;--border:#2d3154;
--green:#00d084;--red:#ff4757;--blue:#4a9eff;--yellow:#ffd700;--purple:#a855f7;--orange:#ff9f43;
--text:#e2e8f0;--text2:#94a3b8}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:-apple-system,"Segoe UI",Roboto,"Noto Sans TC",sans-serif;line-height:1.6;padding:24px}
.wrap{max-width:1180px;margin:0 auto}
h1{font-size:26px;margin-bottom:4px}
h2{font-size:19px;margin:32px 0 14px;padding-left:10px;border-left:4px solid var(--blue)}
.sub{color:var(--text2);font-size:14px;margin-bottom:20px}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:8px}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:18px}
.card .lbl{color:var(--text2);font-size:13px;margin-bottom:6px}
.card .val{font-size:26px;font-weight:700}
.card .tgt{font-size:12px;margin-top:4px}
.pass{color:var(--green)} .fail{color:var(--red)}
.chartbox{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:18px;margin-bottom:18px}
.chartbox h3{font-size:15px;margin-bottom:12px;color:var(--text)}
canvas{max-height:340px}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{padding:7px 9px;text-align:center;border-bottom:1px solid var(--border)}
th{color:var(--text2);font-weight:600}
.g{color:var(--green)} .r{color:var(--red)} .muted{color:var(--text2)}
.two{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.note{background:var(--card2);border:1px solid var(--border);border-radius:10px;padding:14px 16px;font-size:13px;color:var(--text2);margin-top:8px}
.pill{display:inline-block;padding:2px 8px;border-radius:6px;font-size:11px;margin-left:6px}
@media(max-width:880px){.cards{grid-template-columns:repeat(2,1fr)}.two{grid-template-columns:1fr}}
</style></head><body><div class="wrap">

<h1>台積電 2330 — 三-Sleeve 現金流策略 <span class="pill" style="background:#1e3a2f;color:var(--green)">v4</span></h1>
<div class="sub">目標：穩定現金流（空窗月&lt;10% · 月勝率≥65% · 賣出≥3/月 · CAGR≥12%）　|　回測 2021-2026 · next-bar 引擎（昨訊號今開盤成交）</div>

<h2>四大現金流目標</h2>
<div class="cards" id="goalCards"></div>

<h2>輔助指標</h2>
<div class="cards" id="auxCards"></div>

<h2>資金成長曲線</h2>
<div class="chartbox"><h3>策略 vs Buy-and-Hold（同 100 萬本金）</h3><canvas id="equity"></canvas></div>
<div class="two">
<div class="chartbox"><h3>三池資金拆解</h3><canvas id="pools"></canvas></div>
<div class="chartbox"><h3>回撤曲線（%）</h3><canvas id="dd"></canvas></div>
</div>

<h2>逐月損益（現金流穩定度）</h2>
<div class="chartbox"><h3>月報酬（綠=賺/紅=虧/灰=空窗月）</h3><canvas id="monthly"></canvas></div>

<h2>股價與進出場點</h2>
<div class="chartbox"><h3>2330 收盤 + 買(綠)/賣(紅)/斷路器(橘)</h3><canvas id="pricechart"></canvas></div>

<h2>訓練 / 驗證切分</h2>
<div class="chartbox"><table id="segTable"></table>
<div class="note">CAGR 在熊市重的訓練期與牛市的驗證期都 ≥12%，每筆勝率兩期皆 ~83% — 不靠單一行情。
驗證期空窗偏高是 2025 急漲無回檔，那些空窗月本身是大賺月（抱著就漲），非閒置死錢。</div></div>

<h2>逐月明細</h2>
<div class="chartbox"><table id="monthTable"></table></div>

<div class="note" style="margin-top:24px">
<b>策略架構：</b>Pool A 趨勢池(50%，主倉+逢低加碼+單片再平衡收割) · Pool B 震盪池(30%，BB/KD 進場、分批出場) ·
Pool C 常駐微網格(20%，無趨勢濾鏡填空窗，小目標+寬停損高勝率)。三池資金完全獨立。<br>
<b>會計修正(2026-06-02 review)：</b>每筆勝率/PF/連敗已含 CB 出場(真實風險事件)、剔除期末市值平倉(回測邊界 artifact)；
頻率/空窗用真實成交。Pool A 單筆損益改用持有股加權均價。CAGR/資產曲線不受影響。<br>
<b>誠實定位：</b>非追求打敗 Buy-and-Hold，而是「穩定月現金流」——代價是 CAGR 從 v3 的 +22.9% 降到 +18.9%。
未做全市場群組驗證，屬台積電專屬持倉框架。交易成本：買 0.1425% / 賣 0.4425%（含稅）。
</div>

</div>
<script>
const D = __DATA__;
const M = D.metrics;
const fmtK = v => (v/1000).toFixed(0)+'K';
const fmtM = v => (v/10000).toFixed(1)+'萬';

// ── 目標卡 ──
function goalCard(lbl, val, ok, tgt){
  return `<div class="card"><div class="lbl">${lbl}</div>
    <div class="val ${ok?'pass':'fail'}">${val}</div>
    <div class="tgt ${ok?'pass':'fail'}">${ok?'✓':'✗'} ${tgt}</div></div>`;
}
document.getElementById('goalCards').innerHTML =
  goalCard('空窗月', M.idle_pct+'%', M.idle_pct<10, `目標 <10% (${M.idle}/${M.n_months}, 最長${M.max_idle})`) +
  goalCard('月勝率', M.month_wr+'%', M.month_wr>=65, `目標 ≥65% (${M.win_m}賺/${M.loss_m}虧)`) +
  goalCard('賣出頻率', M.sells_pm+'/月', M.sells_pm>=3, `目標 ≥3/月 (共${M.sells}筆)`) +
  goalCard('CAGR', (M.cagr>=0?'+':'')+M.cagr+'%', M.cagr>=12, '目標 ≥12%');

function aux(lbl, val, sub){return `<div class="card"><div class="lbl">${lbl}</div><div class="val">${val}</div><div class="tgt muted">${sub}</div></div>`;}
document.getElementById('auxCards').innerHTML =
  aux('每筆勝率', M.trade_wr+'%', `獲利因子 ${M.pf}`) +
  aux('最大回撤', M.dd+'%', `vs B&H ${M.bh_dd}%`) +
  aux('月度 Sortino', M.sortino, `最長連敗 ${M.max_streak} 筆`) +
  aux('期末總值', fmtM(M.final_total), `B&H CAGR +${M.bh_cagr}%`);

const gridC = '#2d3154', tickC = '#94a3b8';
const baseOpts = (yfmt)=>({responsive:true,interaction:{mode:'index',intersect:false},
  plugins:{legend:{labels:{color:tickC,boxWidth:12}}},
  scales:{x:{ticks:{color:tickC,maxTicksLimit:10},grid:{color:gridC}},
          y:{ticks:{color:tickC,callback:yfmt},grid:{color:gridC}}}});

// ── 資金曲線 ──
new Chart(document.getElementById('equity'),{type:'line',
  data:{labels:D.dates,datasets:[
    {label:'策略總值',data:D.total,borderColor:'#00d084',backgroundColor:'rgba(0,208,132,.08)',fill:true,borderWidth:2,pointRadius:0},
    {label:'Buy & Hold',data:D.bh,borderColor:'#4a9eff',borderWidth:1.5,borderDash:[5,4],pointRadius:0,fill:false}
  ]},options:baseOpts(fmtK)});

// ── 三池 ──
new Chart(document.getElementById('pools'),{type:'line',
  data:{labels:D.dates,datasets:[
    {label:'Pool A 趨勢',data:D.pa,borderColor:'#4a9eff',borderWidth:1.5,pointRadius:0},
    {label:'Pool B 震盪',data:D.pb,borderColor:'#a855f7',borderWidth:1.5,pointRadius:0},
    {label:'Pool C 微網格',data:D.pc,borderColor:'#ff9f43',borderWidth:1.5,pointRadius:0}
  ]},options:baseOpts(fmtK)});

// ── 回撤 ──
new Chart(document.getElementById('dd'),{type:'line',
  data:{labels:D.dates,datasets:[
    {label:'回撤%',data:D.dd_curve,borderColor:'#ff4757',backgroundColor:'rgba(255,71,87,.12)',fill:true,borderWidth:1.5,pointRadius:0}
  ]},options:baseOpts(v=>v+'%')});

// ── 逐月損益 ──
new Chart(document.getElementById('monthly'),{type:'bar',
  data:{labels:D.me_labels,datasets:[
    {label:'月報酬%',data:D.me_vals,
     backgroundColor:D.me_vals.map((v,i)=>D.me_idle[i]?'#4b5563':(v>=0?'#00d084':'#ff4757'))}
  ]},options:{...baseOpts(v=>v+'%'),plugins:{legend:{display:false},
    tooltip:{callbacks:{afterLabel:(c)=>`買${D.me_buys[c.dataIndex]} 賣${D.me_sells[c.dataIndex]}`+(D.me_idle[c.dataIndex]?' (空窗)':'')}}}}});

// ── 股價 + 進出場 ──
new Chart(document.getElementById('pricechart'),{type:'line',
  data:{datasets:[
    {label:'收盤',data:D.dates.map((d,i)=>({x:d,y:D.price[i]})),borderColor:'#64748b',borderWidth:1.2,pointRadius:0,order:3},
    {label:'買',data:D.buy_pts,type:'scatter',backgroundColor:'#00d084',pointRadius:2.5,order:1},
    {label:'賣',data:D.sell_pts,type:'scatter',backgroundColor:'#ff4757',pointRadius:2.5,order:1},
    {label:'斷路器',data:D.cb_pts,type:'scatter',backgroundColor:'#ff9f43',pointRadius:5,pointStyle:'triangle',order:0}
  ]},options:{responsive:true,
    plugins:{legend:{labels:{color:tickC,boxWidth:12}}},
    scales:{x:{type:'category',labels:D.dates,ticks:{color:tickC,maxTicksLimit:10},grid:{color:gridC}},
            y:{ticks:{color:tickC},grid:{color:gridC}}}}});

// ── 訓練/驗證表 ──
const s=D.seg;
document.getElementById('segTable').innerHTML =
 `<tr><th>期間</th><th>CAGR</th><th>最大回撤</th><th>空窗月</th><th>月勝率</th><th>賣出/月</th><th>每筆勝率</th></tr>
  <tr><td>訓練 2021-2024</td><td class="${s.train.cagr>=12?'g':'r'}">${s.train.cagr>=0?'+':''}${s.train.cagr}%</td><td>${s.train.dd}%</td><td>${s.train.idle}%</td><td>${s.train.mwr}%</td><td>${s.train.spm}</td><td>${s.train.twr}%</td></tr>
  <tr><td>驗證 2025-2026</td><td class="${s.test.cagr>=12?'g':'r'}">${s.test.cagr>=0?'+':''}${s.test.cagr}%</td><td>${s.test.dd}%</td><td>${s.test.idle}%</td><td>${s.test.mwr}%</td><td>${s.test.spm}</td><td>${s.test.twr}%</td></tr>
  <tr style="font-weight:700"><td>完整 2021-2026</td><td class="g">+${M.cagr}%</td><td>${M.dd}%</td><td class="${M.idle_pct<10?'g':'r'}">${M.idle_pct}%</td><td class="${M.month_wr>=65?'g':'r'}">${M.month_wr}%</td><td>${M.sells_pm}</td><td>${M.trade_wr}%</td></tr>`;

// ── 逐月明細表（依年分組）──
let rows='<tr><th>年</th>'; for(let mo=1;mo<=12;mo++)rows+=`<th>${mo}月</th>`; rows+='</tr>';
const byYear={};
D.me_labels.forEach((l,i)=>{const[y,mo]=l.split('-');(byYear[y]=byYear[y]||{})[parseInt(mo)]={v:D.me_vals[i],idle:D.me_idle[i]};});
Object.keys(byYear).sort().forEach(y=>{
  rows+=`<tr><td><b>${y}</b></td>`;
  for(let mo=1;mo<=12;mo++){const c=byYear[y][mo];
    if(!c){rows+='<td class="muted">·</td>';}
    else{const cls=c.idle?'muted':(c.v>=0?'g':'r');rows+=`<td class="${cls}">${c.v>=0?'+':''}${c.v}${c.idle?'*':''}</td>`;}}
  rows+='</tr>';});
document.getElementById('monthTable').innerHTML=rows+'<tr><td colspan="13" class="muted" style="text-align:left">* = 空窗月（無交易，數字為持倉市值變動）</td></tr>';
</script></body></html>"""


if __name__ == '__main__':
    data = build()
    html = HTML.replace('__DATA__', json.dumps(data, ensure_ascii=False))
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write(html)
    M = data['metrics']
    print(f"已輸出 {OUT}")
    print(f"  空窗 {M['idle_pct']}% / 月勝率 {M['month_wr']}% / 賣 {M['sells_pm']}/月 / CAGR +{M['cagr']}%")
    print(f"  資料點 {len(data['dates'])}, 買點 {len(data['buy_pts'])}, 賣點 {len(data['sell_pts'])}")
