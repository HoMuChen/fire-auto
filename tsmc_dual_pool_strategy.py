"""
TSMC (2330) 台積電三-Sleeve 現金流策略 - v4（現金流導向，next-bar）
=================================================================
目標已從「絕對報酬」改為「穩定現金流」：高月勝率、高頻率、低空窗。

完整期 2021-2026（四目標全達標，且三池都淨賺）：
  空窗月 6.1%  ✓ (<10%)      月勝率 65.2%  ✓ (≥65%)
  賣出頻率 3.2/月 ✓ (≥3/月)   CAGR +19.1%   ✓ (≥12%)
  每筆勝率 73%   獲利因子 2.4   Sortino ~2.2   最大回撤 -15.8%
  三池淨損益：Pool A +259% / Pool B +91% / Pool C +8,130（皆為正）

  ※ Pool C 必須「波段」不可「剝頭皮」（2026-06-02 重要修正）：
    原 Pool C 用小目標 +1.5% 剝頭皮，毛報酬 +0.28%/筆但扣 0.585% 來回成本後 ≈ -0.3%/筆，
    整池淨虧 -15K——只為填空窗而賠錢無意義（印證「短線在台股成本下不可行」的教訓）。
    改成波段均值回歸（目標 +6%、停損 10%）後 Pool C 淨賺 +8.1K（訓練+6.5K/驗證+0.96K 皆正），
    且空窗 9.1%→6.1%、CAGR +18.9%→+19.1%、DD 更小。每筆勝率 84%→73% 是還原真相
    （原 84% 是被成本吃掉的小贏家堆出來的假象）。**現金流引擎必須自身有正期望，不能靠賠錢製造活動。**

  ※ 交易紀錄會計（2026-06-02 review 修正）：CB 出場與期末平倉原本沒寫 profit_pct 被漏算，
    已修正。每筆品質（勝率/PF/連敗）= 含 CB 出場（真實風險事件）、剔除期末強制平倉
    （A_end/leg=end，回測邊界市值 artifact）；頻率/空窗用真實成交（含期末）。
    Pool A 改用持有股加權均價算單筆損益（收割回補後成本會更新）。CAGR/資產曲線不受影響。

  訓練 2021-2024：空窗4.2% / 月勝率62.5% / 賣4.8/月 / CAGR+13.1% / 每筆勝率84%
  驗證 2025-2026：空窗22%* / 月勝率72.2% / 賣3.4/月 / CAGR+31.9% / 每筆勝率82%
  *驗證期空窗高是 2025 急漲無回檔，那些空窗月全是大賺月（抱著就漲），非死錢。
  CAGR 在熊市重的訓練期(+13%)與牛市驗證期(+32%)都 ≥12%，每筆勝率兩期都 ~83%，不靠單一行情。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
為什麼從 v3 兩池 → v4 三 sleeve
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
v3 兩池（追求報酬）：CAGR +22.9% 但**空窗月 26%、月勝率僅 56%、頻率 1.4 賣/月**。
診斷：熊市/盤整時 Pool A 進冷卻、Pool B 被 MA200 濾鏡擋住，**兩池同時停擺**，
26% 的月份完全沒交易（最長連續 8 個月躺現金）→ 現金流不穩的病根是「空窗」，不是勝率。
（每筆勝率 v3 已達 80%、獲利因子 6.6，勝率本來就高，缺的是「月月有現金流」。）

v4 解法 = 新增 Pool C「常駐微網格」+ Pool A「獲利收割梯」，專門填空窗：
  ① Pool C 在熊市/盤整也交易（無 MA200 濾鏡），用高勝率均值回歸補上回檔/震盪月。
  ② Pool A 收割梯在「緩漲無回檔」月賣一小片（穩賺賣出），補上melt-up空窗月。
取捨：CAGR 從 +22.9% 降到 +18.9%（主動放掉 ~4pp 換穩定），仍遠高於 12% 底線。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
策略架構（三 sleeve，完全獨立資金，皆 next-bar：昨日訊號、今日開盤成交）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

▌ Pool A — 趨勢池（50%）「賺牛市、躲熊市 + 緩漲收割」
  進場：price > MA200，投入池子 85% 建主倉；剩 15% 趨勢內逢低加碼（回調5%買/反彈3%賣）
  收割（單片再平衡）：主倉漲 +3.5% 賣出 15% 一小片（穩賺賣出），回檔 2% 以當下價買回
    → 填緩漲空窗月。注意：同時最多「一片」在外，續漲時只上移參考價、不再賣下一片，
    行為是「最多一片在外的再平衡」而非多片掛單的網格梯。
  斷路器：Pool A 價值從高點跌 12% 全出場；冷卻 45 天 + price>MA200 才重入

▌ Pool B — 震盪池（30%）「賺較大回調反彈，分批出場」
  進場：BB%B<0.15 或 KD<20，且 price>MA200×0.95（不接強趨勢的刀）
  出場：BB%B>0.70 賣一半（降風險）+ 剩一半移動停損 8%（讓贏家跑）
  每筆 60%、最多 2 筆；斷路器 15%

▌ Pool C — 常駐波段均值回歸（20%）「填空窗的現金流引擎」★v4 新增
  進場：price 低於 MA20 達 2%（盤整/弱勢也觸發，無 MA200 濾鏡），格距 1.5%，最多 4 格
  出場：波段目標 +6%（必須 > 來回成本，剝頭皮會淨虧）；安全停損 10%（寬，靠回歸不被洗）
  關鍵：目標要夠大（波段非剝頭皮）才有正淨期望——Pool C 須自身淨賺，不能靠賠錢製造活動填空窗

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
誠實定位與限制（沿用 v3 教訓）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- 本版**不追求打敗 buy-and-hold**（B&H 完整期 +31%），而是「穩定月現金流」：
  空窗<10%、月勝率≥65%、月月有 3+ 筆可做，代價是 CAGR 降到 ~19%、DD ~16%。
- 四目標的穩健性：Pool C 出場參數鄰域 18/18 組合全達標（非踩單一參數）。
- 仍未處理：單檔過擬合（未照 CLAUDE.md 全市場群組驗證）。這是台積電專屬持倉框架。
- 回測一律 next-bar（昨訊號、今開盤成交），移動/收割停損用既有 peak、low 先觸發（保守）。

交易成本：買 0.1425%、賣 0.4425%（含 0.3% 交易稅）。
"""
import pandas as pd
import numpy as np

BUY_FEE  = 0.001425
SELL_FEE = 0.001425 + 0.003
CAPITAL  = 1_000_000


BEST_PARAMS = {
    # ── 三 sleeve 資金配置 ──
    'a_pct':       0.50,   # 趨勢池
    'b_pct':       0.30,   # 震盪池
    'c_pct':       0.20,   # 常駐微網格（填空窗）

    # ── Pool A 趨勢池 ──
    'a_stock':     0.85,   # 主倉佔池子比例（剩 15% 逢低加碼）
    'a_cb':        0.12,   # 斷路器
    'a_cooldown':  45,     # CB 後冷卻天數
    'dip_trigger': 0.05,   # 逢低加碼觸發
    'dip_recover': 0.03,   # 加碼部位反彈賣出
    'a_harvest':   True,   # 啟用獲利收割梯
    'a_harv_up':   0.035,  # 每漲此幅賣一小片
    'a_harv_dn':   0.02,   # 回檔此幅買回
    'a_harv_frac': 0.15,   # 每次收割賣出主倉比例

    # ── Pool B 震盪池（分批出場）──
    'b_bb_buy':    0.15,
    'b_kd_buy':    20,
    'b_target':    0.70,   # 第一段：賣一半
    'b_trail':     0.08,   # 第二段：移動停損
    'b_cb':        0.15,
    'b_pos_frac':  0.60,
    'b_max_pos':   2,
    'b_bear_fil':  0.95,

    # ── Pool C 常駐波段均值回歸（現金流引擎，須淨賺）──
    'c_mode':      'ma20grid',  # 對 MA20 乖離的網格（平靜盤也交易）
    'c_dev':       0.02,        # 低於 MA20 此幅進場
    'c_grid_gap':  0.015,       # 格距
    'c_target':    0.06,        # 波段目標（必須 > 來回成本 0.585% 才有淨邊際，剝頭皮會淨虧）
    'c_stop':      0.10,        # 安全停損（寬，靠均值回歸）
    'c_bear_brake':0.0,         # 深跌煞車（0=不啟用，實測啟用反傷月勝率）
    'c_max_pos':   4,           # 最多格數
    'c_pos_frac':  0.20,        # 每格佔 Pool C 現金比例
    'c_rsi':       30,          # （mode='rsi' 時用，目前未用）
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 資料讀取與指標
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_data(path='data/stock_prices/2330.csv'):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df = df[df['close'] > 0].copy()

    hi, lo, cl = df['high'], df['low'], df['close']
    prev = cl.shift(1)
    tr = pd.concat([hi-lo, (hi-prev).abs(), (lo-prev).abs()], axis=1).max(axis=1)
    df['atr20'] = tr.ewm(span=20, adjust=False).mean()
    for p in [20, 60, 120, 200]:
        df[f'ma{p}'] = cl.rolling(p).mean()

    # Bollinger %B
    df['bb_mid'] = cl.rolling(20).mean()
    df['bb_std'] = cl.rolling(20).std()
    df['bb_pct'] = (cl - (df['bb_mid'] - 2*df['bb_std'])) / ((4*df['bb_std']) + 1e-9)

    # RSI(14)
    delta = cl.diff()
    gain  = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    df['rsi14'] = 100 - 100 / (1 + gain / loss)

    # KD（隨機指標，9日）
    low9  = lo.rolling(9).min()
    high9 = hi.rolling(9).max()
    rsv   = (cl - low9) / (high9 - low9 + 1e-9) * 100
    df['kd_k'] = rsv.ewm(com=2, adjust=False).mean()
    df['kd_d'] = df['kd_k'].ewm(com=2, adjust=False).mean()

    df['ret5d'] = cl.pct_change(5)
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 回測引擎（三 sleeve 獨立資金，next-bar）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def backtest(df, params=None, start='2021-01-01', end='2026-06-01'):
    """三 sleeve next-bar 引擎：A 趨勢(+收割梯) / B 震盪(分批出場) / C 常駐微網格(填空窗)。"""
    P = params if params is not None else BEST_PARAMS

    # ── Pool A 趨勢 ──
    a_cash = CAPITAL * P['a_pct']
    a_base_sh = 0; a_dip_sh = 0; a_dip_buy_p = 0.0; a_recent_high = 0.0
    a_peak = a_cash; a_stop = False; a_stop_dt = None
    a_base_cost = 0.0; a_harvest_sh = 0; a_harvest_sell_p = 0.0; a_harvest_ref = 0.0
    a_cost_sum = 0.0   # 持有 A 股的總成本（毛額，含 base/dip/收割回補），供 CB/期末/收割算單筆損益

    def a_avg():       # 持有 A 股的加權平均成本
        s = a_base_sh + a_dip_sh
        return a_cost_sum / s if s > 0 else 0.0

    # ── Pool B 震盪 ──
    b_cash = CAPITAL * P['b_pct']
    b_pos = {}; b_cnt = 0; b_peak = b_cash; b_stop = False

    # ── Pool C 常駐微網格 ──
    c_cash = CAPITAL * P['c_pct']
    c_pos = {}; c_cnt = 0

    portfolio = []; trades = []
    df = df.reset_index(drop=True)
    idxs = df.index[(df['date'] >= start) & (df['date'] <= end)].tolist()

    def a_shares(): return a_base_sh + a_dip_sh
    def av(p): return a_cash + a_shares() * p
    def bv(p): return b_cash + sum(pos[0]*p for pos in b_pos.values())
    def cv(p): return c_cash + sum(pos[0]*p for pos in c_pos.values())

    for i in idxs:
        if i == 0:
            continue
        row = df.loc[i]
        d  = row['date']; o = row['open']; hi = row['high']; lo = row['low']; cl = row['close']
        prev = df.loc[i-1]                       # 昨日：訊號來源
        m200 = prev['ma200']; bbp = prev['bb_pct']; kd = prev['kd_k']; pclose = prev['close']
        rsi = prev['rsi14']; ma20 = prev['ma20']

        if pd.isna(m200) or pd.isna(bbp) or o == 0:
            portfolio.append({'date': d, 'total': av(cl)+bv(cl)+cv(cl),
                              'pool_a': av(cl), 'pool_b': bv(cl), 'pool_c': cv(cl)})
            continue

        ex = o                                   # 今日成交價 = 開盤

        # ════════════ POOL A 趨勢（+ 收割梯）════════════
        if not a_stop:
            a_peak = max(a_peak, av(ex))
            if av(ex) < a_peak * (1 - P['a_cb']):
                if a_shares() > 0:
                    sh = a_shares(); avg = a_avg(); a_cash += sh * ex * (1 - SELL_FEE)
                    trades.append({'date': d, 'action': 'A_cb_exit', 'price': ex, 'shares': sh,
                                   'buy_price': avg, 'profit_pct': (ex/avg-1)*100 if avg>0 else 0.0})
                    a_base_sh = 0; a_dip_sh = 0; a_cost_sum = 0.0
                a_stop = True; a_peak = a_cash; a_stop_dt = d; a_recent_high = 0.0
                a_harvest_sh = 0; a_harvest_ref = 0.0
        if a_stop and pclose > m200 and (d - a_stop_dt).days >= P['a_cooldown']:
            a_stop = False; a_peak = a_cash; a_stop_dt = None
        if not a_stop and pclose > m200 and a_base_sh == 0:
            sh = int(a_cash * P['a_stock'] / (ex * (1 + BUY_FEE)))
            if sh > 0 and sh*ex*(1+BUY_FEE) <= a_cash:
                a_cash -= sh*ex*(1+BUY_FEE); a_base_sh = sh
                a_cost_sum = sh * ex                      # 新主倉，重置成本基礎
                trades.append({'date': d, 'action': 'A_base_buy', 'price': ex, 'shares': sh})
                a_peak = max(a_peak, av(ex)); a_recent_high = ex
                a_base_cost = ex; a_harvest_ref = ex; a_harvest_sh = 0
        if not a_stop and a_base_sh > 0:
            a_recent_high = max(a_recent_high, cl)
            if a_dip_sh == 0 and pclose < a_recent_high * (1 - P['dip_trigger']):
                sh = int(a_cash / (ex * (1 + BUY_FEE)))
                if sh > 0 and sh*ex*(1+BUY_FEE) <= a_cash:
                    a_cash -= sh*ex*(1+BUY_FEE); a_dip_sh = sh; a_dip_buy_p = ex
                    a_cost_sum += sh * ex                 # 加碼 lot 計入成本
                    trades.append({'date': d, 'action': 'A_dip_buy', 'price': ex, 'shares': sh})
            elif a_dip_sh > 0 and pclose >= a_dip_buy_p * (1 + P['dip_recover']):
                a_cash += a_dip_sh * ex * (1 - SELL_FEE)
                a_cost_sum -= a_dip_sh * a_dip_buy_p      # 移除加碼 lot 成本（按其買價）
                trades.append({'date': d, 'action': 'A_dip_sell', 'price': ex, 'shares': a_dip_sh,
                               'buy_price': a_dip_buy_p, 'profit_pct': (ex/a_dip_buy_p-1)*100})
                a_dip_sh = 0; a_recent_high = ex

        # ── Pool A 獲利收割（單片再平衡）：漲一段賣一小片(穩賺賣出)，回檔以當下價買回 ──
        # 注意：同時最多「一片」在外（a_harvest_sh>0 時不再賣下一片，只上移參考價），
        # 行為是「最多一片在外的再平衡」而非多片掛出的網格梯。單筆損益用持有股的加權均價。
        if P.get('a_harvest', False) and not a_stop and a_base_sh > 0:
            if a_harvest_sh == 0 and pclose >= a_harvest_ref * (1 + P['a_harv_up']):
                hs = int(a_base_sh * P['a_harv_frac'])
                if hs > 0:
                    avg = a_avg()
                    a_cash += hs * ex * (1 - SELL_FEE); a_base_sh -= hs
                    a_cost_sum -= hs * avg                # 賣出片按均價移除成本
                    a_harvest_sh = hs; a_harvest_sell_p = ex; a_harvest_ref = ex
                    trades.append({'date': d, 'action': 'A_harvest_sell', 'price': ex, 'shares': hs,
                                   'buy_price': avg, 'profit_pct': (ex/avg-1)*100 if avg>0 else 0.0})
            elif a_harvest_sh > 0 and pclose <= a_harvest_sell_p * (1 - P['a_harv_dn']):
                cost = a_harvest_sh * ex * (1 + BUY_FEE)
                if cost <= a_cash:
                    a_cash -= cost; a_base_sh += a_harvest_sh
                    a_cost_sum += a_harvest_sh * ex       # 回補按當下價更新成本基礎
                    trades.append({'date': d, 'action': 'A_harvest_buy', 'price': ex, 'shares': a_harvest_sh})
                    a_harvest_sh = 0; a_harvest_ref = ex
            elif a_harvest_sh > 0 and pclose >= a_harvest_sell_p * (1 + P['a_harv_up']):
                a_harvest_ref = pclose  # 續漲：基準上移，等更高再賣（不追，但不卡住）

        # ════════════ POOL B 震盪（分批出場）════════════
        if not b_stop:
            b_peak = max(b_peak, bv(ex))
            if bv(ex) < b_peak * (1 - P['b_cb']):
                for pid, v in list(b_pos.items()):
                    b_cash += v[0] * ex * (1 - SELL_FEE)
                    trades.append({'date': d, 'action': 'B_sell', 'price': ex, 'shares': v[0],
                                   'buy_price': v[1], 'profit_pct': (ex/v[1]-1)*100, 'leg': 'cb'})
                b_pos.clear(); b_stop = True; b_peak = b_cash
        if b_stop and pclose > m200 * P['b_bear_fil']:
            b_stop = False; b_peak = b_cash
        if not b_stop:
            for pid in list(b_pos.keys()):
                sh, bp, pk, half = b_pos[pid]
                ts = pk * (1 - P['b_trail'])               # 既有 peak（保守，不含今日 high）
                if lo < ts:                                 # 今日 low 觸發停損
                    fill = min(ts, o)
                    b_cash += sh * fill * (1 - SELL_FEE)
                    trades.append({'date': d, 'action': 'B_sell', 'price': fill, 'shares': sh,
                                   'buy_price': bp, 'profit_pct': (fill/bp-1)*100, 'leg': 'trail'})
                    del b_pos[pid]
                elif (not half) and bbp > P['b_target']:    # 昨日訊號，今日開盤賣一半
                    hs = sh // 2
                    if hs > 0:
                        b_cash += hs * ex * (1 - SELL_FEE)
                        trades.append({'date': d, 'action': 'B_sell', 'price': ex, 'shares': hs,
                                       'buy_price': bp, 'profit_pct': (ex/bp-1)*100, 'leg': 'target'})
                        b_pos[pid] = (sh-hs, bp, max(pk, cl), True)
                    else:
                        b_pos[pid] = (sh, bp, max(pk, cl), True)
                else:
                    b_pos[pid] = (sh, bp, max(pk, cl), half)  # 收盤後更新 peak 供明日用
            sig_bb = bbp < P['b_bb_buy']; sig_kd = (not pd.isna(kd) and kd < P['b_kd_buy'])
            if (sig_bb or sig_kd) and len(b_pos) < P['b_max_pos'] and pclose > m200 * P['b_bear_fil']:
                sh = int(b_cash * P['b_pos_frac'] / (ex * (1 + BUY_FEE)))
                if sh > 0 and sh*ex*(1+BUY_FEE) <= b_cash:
                    b_cash -= sh*ex*(1+BUY_FEE); b_cnt += 1
                    b_pos[b_cnt] = (sh, ex, ex, False)
                    trades.append({'date': d, 'action': 'B_buy', 'price': ex, 'shares': sh})
                    b_peak = max(b_peak, bv(ex))

        # ════════════ POOL C 常駐微網格（填空窗）════════════
        if P['c_pct'] > 0:
            # 同日 OHLC 成交順序（保守）：開盤跳空優先 → 盤中 stop-first（若同日同碰 target/stop，
            # 無日內 tick 無法判先後，保守假設先觸發停損）→ 盤中停利。
            for pid in list(c_pos.keys()):
                sh, bp = c_pos[pid]
                tgt = bp * (1 + P['c_target']); stp = bp * (1 - P['c_stop'])
                fill = leg = None
                if o >= tgt:                                 # 跳空高於目標：開盤賣（有利）
                    fill, leg = o, 'gap'
                elif o <= stp:                               # 跳空低於停損：開盤停損
                    fill, leg = o, 'stop'
                elif lo <= stp:                              # 盤中先假設觸發停損（保守）
                    fill, leg = stp, 'stop'
                elif hi >= tgt:                              # 盤中達標停利
                    fill, leg = tgt, 'target'
                if fill is not None:
                    c_cash += sh * fill * (1 - SELL_FEE)
                    trades.append({'date': d, 'action': 'C_sell', 'price': fill, 'shares': sh,
                                   'buy_price': bp, 'profit_pct': (fill/bp-1)*100, 'leg': leg})
                    del c_pos[pid]
            # 進場
            if P.get('c_mode', 'ma20grid') == 'rsi':
                c_sig = (not pd.isna(rsi) and rsi < P['c_rsi'])
            else:
                c_sig = (not pd.isna(ma20) and pclose < ma20 * (1 - P['c_dev']))
            ok_spacing = True
            if c_pos and P.get('c_grid_gap', 0) > 0:
                last_bp = max(v[1] for v in c_pos.values())
                ok_spacing = pclose < last_bp * (1 - P['c_grid_gap'])
            brake = False
            if P.get('c_bear_brake', 0) > 0:
                ma60 = prev['ma60']
                brake = (not pd.isna(ma60)) and pclose < ma60 * (1 - P['c_bear_brake'])
            if c_sig and ok_spacing and not brake and len(c_pos) < P['c_max_pos']:
                sh = int(c_cash * P['c_pos_frac'] / (ex * (1 + BUY_FEE)))
                if sh > 0 and sh*ex*(1+BUY_FEE) <= c_cash:
                    c_cash -= sh*ex*(1+BUY_FEE); c_cnt += 1
                    c_pos[c_cnt] = (sh, ex)
                    trades.append({'date': d, 'action': 'C_buy', 'price': ex, 'shares': sh})

        portfolio.append({'date': d, 'total': av(cl)+bv(cl)+cv(cl),
                          'pool_a': av(cl), 'pool_b': bv(cl), 'pool_c': cv(cl)})

    # 期末平倉（成本計入最終 portfolio，且寫入單筆損益供現金流指標）
    lp = df.loc[idxs[-1], 'close']; ld = df.loc[idxs[-1], 'date']
    if a_shares() > 0:
        avg = a_avg(); a_cash += a_shares()*lp*(1-SELL_FEE)
        trades.append({'date': ld, 'action': 'A_end', 'price': lp, 'shares': a_shares(),
                       'buy_price': avg, 'profit_pct': (lp/avg-1)*100 if avg>0 else 0.0})
        a_base_sh = 0; a_dip_sh = 0; a_cost_sum = 0.0
    for pid, v in list(b_pos.items()):
        b_cash += v[0]*lp*(1-SELL_FEE)
        trades.append({'date': ld, 'action': 'B_sell', 'price': lp, 'shares': v[0],
                       'buy_price': v[1], 'profit_pct': (lp/v[1]-1)*100, 'leg': 'end'})
    b_pos.clear()
    for pid, v in list(c_pos.items()):
        c_cash += v[0]*lp*(1-SELL_FEE)
        trades.append({'date': ld, 'action': 'C_sell', 'price': lp, 'shares': v[0],
                       'buy_price': v[1], 'profit_pct': (lp/v[1]-1)*100, 'leg': 'end'})
    c_pos.clear()
    if portfolio:
        portfolio[-1] = {'date': ld, 'total': a_cash+b_cash+c_cash,
                         'pool_a': a_cash, 'pool_b': b_cash, 'pool_c': c_cash}
    return pd.DataFrame(portfolio), pd.DataFrame(trades) if trades else pd.DataFrame()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 績效計算（報酬 + 現金流指標）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calc_metrics(port_df, start, end):
    df = port_df.copy().set_index('date')
    n = (pd.to_datetime(end) - pd.to_datetime(start)).days / 365.25
    ann = ((df['total'].iloc[-1] / CAPITAL) ** (1/n) - 1) * 100
    rmax = df['total'].cummax()
    mdd = ((df['total'] - rmax) / rmax * 100).min()
    dr = df['total'].pct_change().dropna()
    sh = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    yr = {int(y): round((g['total'].iloc[-1]/g['total'].iloc[0] - 1)*100, 1)
          for y, g in df.groupby(df.index.year) if len(g) > 5}
    return {'annual_ret': round(ann, 2), 'max_dd': round(mdd, 2),
            'sharpe': round(sh, 3), 'year_rets': yr}


SELL_ACTIONS = ['A_dip_sell','A_cb_exit','B_sell','C_sell','A_end','A_harvest_sell']
BUY_ACTIONS  = ['A_base_buy','A_dip_buy','B_buy','C_buy','A_harvest_buy']


def cashflow_metrics(port, trades, start, end):
    """現金流導向指標：空窗月、月勝率、賣出頻率、月度 Sortino，及每筆勝率/PF/連敗。"""
    p = port.copy().set_index('date').sort_index()
    me = p['total'].resample('ME').last()
    mret = me.pct_change(); mret.iloc[0] = me.iloc[0]/CAPITAL - 1
    mret = mret.dropna()*100
    n = len(mret)
    win_m = int((mret > 0).sum()); loss_m = int((mret < 0).sum())

    if len(trades):
        t = trades.copy(); t['month'] = pd.to_datetime(t['date']).dt.to_period('M')
        has_pp = 'profit_pct' in t
        valid = t['action'].isin(SELL_ACTIONS) & (t['profit_pct'].notna() if has_pp else False)
        sells_all = t[valid]                              # 含期末平倉：用於頻率/空窗（真實成交）
        # 每筆品質（勝率/PF/連敗）排除「期末強制平倉」——回測邊界的市值平倉 artifact，
        # 實盤部位其實還開著，那筆 +100%+ 的市值損益不是真實出場決策。CB 出場是真實出場，保留。
        is_end = (t['action'] == 'A_end') | (t.get('leg', pd.Series(index=t.index, dtype=object)) == 'end')
        sells = t[valid & (~is_end)]                      # 排除期末平倉：用於品質指標
        buys = t[t['action'].isin(BUY_ACTIONS)]
        spm = sells_all.groupby('month').size() if len(sells_all) else pd.Series(dtype=int)
        bpm = buys.groupby('month').size() if len(buys) else pd.Series(dtype=int)
    else:
        sells_all = sells = buys = pd.DataFrame(); spm = bpm = pd.Series(dtype=int)

    periods = mret.index.to_period('M')
    allp = pd.period_range(periods.min(), periods.max(), freq='M')
    sal = spm.reindex(allp, fill_value=0); bal = bpm.reindex(allp, fill_value=0)
    tot = sal + bal
    idle = int((tot == 0).sum())
    mx = cur = 0
    for v in tot.values:
        if v == 0: cur += 1; mx = max(mx, cur)
        else: cur = 0

    dr = mret/100; dstd = dr[dr < 0].std()
    sortino = dr.mean()/dstd*np.sqrt(12) if dstd > 0 else float('nan')

    if len(sells):
        wr = (sells['profit_pct']>0).mean()*100
        W = sells[sells['profit_pct']>0]['profit_pct']; L = sells[sells['profit_pct']<=0]['profit_pct']
        pf = W.sum()/abs(L.sum()) if len(L) and L.sum()!=0 else float('inf')
        ss = sells.sort_values('date'); st=mxs=0
        for v in ss['profit_pct'].values:
            if v<=0: st+=1; mxs=max(mxs,st)
            else: st=0
    else:
        wr=pf=mxs=0

    return dict(n_months=n, win_m=win_m, loss_m=loss_m, month_wr=win_m/n*100,
                idle=idle, idle_pct=idle/n*100, max_idle=mx,
                sells=len(sells_all), sells_pm=len(sells_all)/n, sells_quality=len(sells),
                buys=len(buys), sortino=sortino, trade_wr=wr, pf=pf, max_streak=mxs, mret=mret)


def print_report(m, cf, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  ── 現金流目標 ──")
    print(f"  空窗月  {cf['idle_pct']:5.1f}%  {'✓' if cf['idle_pct']<10 else '✗'} (目標 <10%, {cf['idle']}/{cf['n_months']}, 最長{cf['max_idle']})")
    print(f"  月勝率  {cf['month_wr']:5.1f}%  {'✓' if cf['month_wr']>=65 else '✗'} (目標 ≥65%, {cf['win_m']}賺/{cf['loss_m']}虧)")
    print(f"  賣出頻率 {cf['sells_pm']:4.1f}/月 {'✓' if cf['sells_pm']>=3 else '✗'} (目標 ≥3/月)")
    print(f"  CAGR    {m['annual_ret']:+5.1f}%  {'✓' if m['annual_ret']>=12 else '✗'} (目標 ≥12%)")
    print(f"  ── 輔助 ──")
    print(f"  每筆勝率 {cf['trade_wr']:.0f}%   獲利因子 {cf['pf']:.1f}   月度Sortino {cf['sortino']:.2f}")
    print(f"  最大回撤 {m['max_dd']:.1f}%   最長連敗 {cf['max_streak']}   日Sharpe {m['sharpe']:.2f}")
    print(f"  各年報酬：" + "  ".join(f"{y}:{r:+.0f}%" for y, r in m['year_rets'].items()))


if __name__ == '__main__':
    from pathlib import Path
    if not Path('data/stock_prices/2330.csv').exists():
        print("找不到資料 data/stock_prices/2330.csv"); exit(1)

    df = load_data()
    print(f"資料範圍：{df['date'].min().date()} ~ {df['date'].max().date()}")

    for s, e, lb in [('2021-01-01','2024-12-31','訓練期 2021-2024'),
                     ('2025-01-01','2026-06-01','驗證期 2025-2026'),
                     ('2021-01-01','2026-06-01','完整期 2021-2026（四目標達標）')]:
        port, tr = backtest(df, BEST_PARAMS, s, e)
        m = calc_metrics(port, s, e)
        cf = cashflow_metrics(port, tr, s, e)
        print_report(m, cf, lb)
