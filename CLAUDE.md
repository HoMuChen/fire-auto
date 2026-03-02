# Fire-Auto

交易策略研究專案，使用 Supabase 儲存台股股票列表與歷史股價資料。

## 環境設定

- `.env.local` — Supabase credentials、FinMind API token、Cron secret

## Supabase Database

**URL**: `https://jnikspnudxhecsqthlbo.supabase.co`

### `stocks` 表 — 股票列表（3,047 筆）

| Column | Type | Description |
|--------|------|-------------|
| `stock_id` | string (PK) | 股票代號（如 `2330`, `3687`, `00835B`） |
| `stock_name` | string | 股票名稱 |
| `industry_category` | string | 產業類別（如 數位雲端類、電子零組件業） |
| `type` | string | 市場類型：`twse`（上市）/ `tpex`（上櫃） |
| `sync_status` | string | 同步狀態（`synced`） |
| `updated_at` | timestamp | 最後更新時間 |

統計：上市 431、上櫃 522，其餘為 ETF 等。

### `stock_prices` 表 — 日K線股價（~1,264,702 筆）

| Column | Type | Description |
|--------|------|-------------|
| `id` | integer (PK) | 自增主鍵 |
| `stock_id` | string (FK) | 股票代號，關聯 `stocks.stock_id` |
| `date` | date | 交易日期 |
| `open` | float | 開盤價 |
| `high` | float | 最高價 |
| `low` | float | 最低價 |
| `close` | float | 收盤價 |
| `volume` | integer | 成交股數 |
| `trading_money` | integer | 成交金額 |
| `trading_turnover` | integer | 成交筆數 |
| `spread` | float | 漲跌幅 |

日期範圍：2024-02-15 ~ 2026-02-25（約兩年日K線）。

### `institutional_investors` 表 — 三大法人買賣超

| Column | Type | Description |
|--------|------|-------------|
| `id` | integer (PK) | 自增主鍵 |
| `stock_id` | string (FK) | 股票代號 |
| `date` | date | 交易日期 |
| `investor_name` | string | 法人類別 |
| `buy` | integer | 買進股數 |
| `sell` | integer | 賣出股數 |

investor_name 值：`Foreign_Investor`（外資）、`Investment_Trust`（投信）、`Dealer_self`（自營商自行）、`Dealer_Hedging`（自營商避險）、`Foreign_Dealer_Self`
日期範圍：2020-01-02 ~ 2026-02-25

### `margin_trading` 表 — 融資融券

| Column | Type | Description |
|--------|------|-------------|
| `id` | integer (PK) | 自增主鍵 |
| `stock_id` | string (FK) | 股票代號 |
| `date` | date | 交易日期 |
| `margin_purchase_buy` | integer | 融資買進 |
| `margin_purchase_sell` | integer | 融資賣出 |
| `margin_purchase_cash_repayment` | integer | 融資現償 |
| `margin_purchase_yesterday_balance` | integer | 融資前日餘額 |
| `margin_purchase_today_balance` | integer | 融資今日餘額 |
| `short_sale_buy` | integer | 融券買進 |
| `short_sale_sell` | integer | 融券賣出 |
| `short_sale_cash_repayment` | integer | 融券現償 |
| `short_sale_yesterday_balance` | integer | 融券前日餘額 |
| `short_sale_today_balance` | integer | 融券今日餘額 |

日期範圍：2020-01-02 ~ 2026-02-25

### `monthly_revenue` 表 — 月營收

| Column | Type | Description |
|--------|------|-------------|
| `id` | integer (PK) | 自增主鍵 |
| `stock_id` | string (FK) | 股票代號 |
| `date` | date | 日期 |
| `revenue_year` | integer | 營收年份 |
| `revenue_month` | integer | 營收月份 |
| `revenue` | bigint | 營收金額（元） |

### `dividends` 表 — 除權息

| Column | Type | Description |
|--------|------|-------------|
| `stock_id` | string (FK) | 股票代號 |
| `date` | date | 除權息日 |
| `year` | integer | 年度 |
| `cash_dividend` | float | 現金股利 |
| `stock_dividend` | float | 股票股利 |

### `financial_statements` 表 — 財報

| Column | Type | Description |
|--------|------|-------------|
| `stock_id` | string (FK) | 股票代號 |
| `date` | date | 財報日期 |
| `statement_type` | string | 報表類型（income, balance_sheet, ...） |
| `item_name` | string | 科目名稱 |
| `value` | float | 金額 |

## 本地資料

### `individual_stocks.json` — 個股清單（2,518 檔）

從 Supabase `stocks` 表篩選，排除以下非個股類別：
ETF、上櫃ETF、上櫃指數股票型基金(ETF)、ETN、指數投資證券(ETN)、受益證券、存託憑證、所有證券、Index、大盤

每筆欄位：`stock_id`, `stock_name`, `industry_category`, `type`, `avg_daily_vol_張`, `low_liquidity`

- `avg_daily_vol_張`：過去兩年平均每日成交張數（無本地股價資料者為 `null`）
- `low_liquidity`：`true` 表示平均日成交張數 < 200 張，視為流動性不足，**回測與 Watchlist 應排除**；`false` 為可用標的；`null` 表示無本地股價資料

分佈：上市 1,231 / 上櫃 913 / 興櫃 374（其中 low_liquidity=false 可用標的 1,439 檔）

### `data/stock_prices/` — 本地股價快取（CSV）

- 每檔股票一個 CSV 檔（如 `2330.csv`），共 2,369 檔（149 檔在 Supabase 無股價資料）
- CSV 欄位：`date,open,high,low,close,volume,trading_money,trading_turnover,spread`
- 日期範圍：2024-02-15 ~ 2026-02-25（約兩年日K線）
- 總大小約 65 MB

### `data/institutional/` — 本地三大法人快取（CSV）

- 每檔股票一個 CSV 檔（如 `2330.csv`）
- CSV 欄位：`date,investor_name,buy,sell`
- 每個交易日有多筆（每種法人一筆）
- 日期範圍：2020-01-02 ~ 2026-02-25

### `data/margin/` — 本地融資融券快取（CSV）

- 每檔股票一個 CSV 檔（如 `2330.csv`）
- CSV 欄位：`date,margin_purchase_buy,margin_purchase_sell,margin_purchase_cash_repayment,margin_purchase_yesterday_balance,margin_purchase_today_balance,short_sale_buy,short_sale_sell,short_sale_cash_repayment,short_sale_yesterday_balance,short_sale_today_balance`
- 每個交易日一筆
- 日期範圍：2020-01-02 ~ 2026-02-25

### `data/cache_meta.json` — 股價快取 metadata

記錄每檔股票在本地的最新快取日期，格式：`{"2330": {"start": "2020-01-02", "end": "2026-02-25"}, ...}`

### `data/chip_cache_meta.json` — 籌碼快取 metadata

記錄三大法人和融資融券的快取狀態，格式：`{"inst:2330": {"start": "...", "end": "..."}, "margin:2330": {...}, ...}`

## 股價快取模組 `stock_cache.py`

從 Supabase 抓股價並快取在本地 CSV，下次直接讀本地檔案。

```python
from stock_cache import StockCache
cache = StockCache()

# 取得股價（有快取讀快取，過期自動增量更新）
rows = cache.get("2330")

# 強制重新抓
rows = cache.get("2330", force_update=True)

# 批次更新全部
cache.update_all()

# 查看快取狀態
cache.info()
```

快取邏輯：
- 本地最新日期 = 今天 → 直接讀 CSV
- 本地最新日期 < 今天 → 增量更新（只抓缺少的日期）
- 本地無資料 → 全量抓取

## 籌碼快取模組 `chip_cache.py`

從 Supabase 抓三大法人買賣超和融資融券，快取在本地 CSV。

```python
from chip_cache import ChipCache

cache = ChipCache()

# 取得三大法人買賣超
rows = cache.get_institutional("2330")
# -> [{"date": "2020-01-02", "investor_name": "Foreign_Investor", "buy": "26970999", "sell": "31163268"}, ...]

# 取得融資融券
rows = cache.get_margin("2330")
# -> [{"date": "2020-01-02", "margin_purchase_today_balance": "21635", "short_sale_today_balance": "236", ...}, ...]

# 批次更新所有有流動性個股
cache.update_all()

# 查看快取狀態
cache.info()
```

快取邏輯同 `stock_cache.py`（增量更新）。只快取有流動性的 1,439 檔個股。
資料存放：`data/institutional/`（法人）、`data/margin/`（融資融券）。

## 策略研究原則

### 核心哲學

- **參數穩健性優先**：好的策略不會因為參數從 10 調到 20 就產生截然不同的結果。如果微調參數能大幅改變績效，代表策略本身不穩固，應該捨棄而非調參。
- **一開始就用合理數字**：不做參數最佳化，直接採用市場慣用的合理預設值（例如均線用 20/60/120，RSI 用 14，ATR 用 14）。參數是用來描述邏輯的，不是用來擬合歷史的。
- **全市場驗證**：每個策略都必須對 `individual_stocks.json` 中的全部個股回測，不能只挑幾檔跑。一個策略要有意義，必須在大量股票上呈現統計優勢。

### 策略類型

不限於以下分類，可自行上網搜尋或自由發想新策略來回測。順勢與逆勢至少各一，其餘類型多多益善：

**順勢（Trend Following）**
- 核心邏輯：趨勢一旦形成，傾向延續
- 方向：突破買進、跌破賣出
- 典型工具：均線、通道突破、動量指標

**逆勢（Mean Reversion）**
- 核心邏輯：價格偏離均值後，傾向回歸
- 方向：超跌買進、超漲賣出
- 典型工具：RSI、布林通道、乖離率

**其他可探索方向（不限於此）**
- 量價結合：量縮突破、量增回測等
- 波動率策略：波動收斂後的爆發、ATR 通道
- 型態辨識：跳空缺口、N日新高/新低
- 事件驅動：除權息、月營收公布等時間點
- 統計套利：產業內相對強弱、配對交易
- 自由發想：任何有邏輯基礎的想法都值得回測驗證

### 回測規範

- **標的範圍**：`individual_stocks.json` 全部個股（有股價資料的 2,369 檔）
- **資料來源**：`data/stock_prices/` 本地 CSV（透過 `stock_cache.py` 讀取）
- **資料期間**：2024-02-15 ~ 2026-02-25
- **交易成本**：必須計入手續費（0.1425%）與交易稅（賣出 0.3%）
- **滑價**：無需額外假設（用收盤價模擬）
- **部位**：預設每筆交易等金額投入，不加槓桿
- **停損**：所有策略都必須有停損機制，但不使用統一固定比例。每個策略應根據自身邏輯設計專屬停損條件：
  - 逆勢短線策略（如 RSI）→ ATR 倍數停損（依波動自適應）
  - 機械式策略（如網格）→ 固定百分比（如 2 倍網格間距）
  - 短持策略（如連 N 黑）→ 緊停損（持有期短，快速止損）
  - 可追蹤趨勢的策略（如 KD 交叉）→ 移動停損（追蹤高點）
  - 停損類型：`fixed_pct`（固定百分比）、`atr`（ATR 倍數）、`trailing_pct`（移動停損）
- **績效指標**：勝率、平均報酬、最大回撤、獲利因子（Profit Factor）、夏普比率

### 策略驗證標準（必須遵守）

**一個策略要被保留，必須通過「群組驗證」**：

1. 用事前可觀察特徵（波動率、趨勢強度）將 1,281 檔流動性個股分成 6 個群組
2. 對每個「策略 × 群組」配對，計算群組內**所有股票的平均每筆交易淨報酬**（扣手續費+稅）
3. 將資料切分為 Period A（2020-2022）和 Period B（2023-2026）
4. **兩個時段的平均每筆交易報酬都 > 0，且每期至少 30 筆交易**，才算 VALID
5. 策略不需要對所有群組有效，但**至少要對一個群組整體有正期望值**

這意味著：
- ✗ 「這個策略在某一檔股票上賺很多」不算有效（可能是巧合）
- ✓ 「這個策略在某類股票上，平均每筆交易有正期望」才算有效
- 驗證工具：`validate_strategy_groups.py`

### 工作流程

1. 定義策略邏輯（進場/出場條件）
2. 用合理預設參數實作
3. 對全部個股跑回測
4. 用 `validate_strategy_groups.py` 做群組驗證
5. 只保留至少對一個群組 VALID 的策略
6. 若結果有意義，檢驗參數穩健性（微調參數看結果是否劇烈變動，變動大就捨棄）
7. 保留穩健的策略，記錄結論

### 股票分群（6 組）

用波動率（三分位）× 趨勢強度（R² > 0.3）分群：

| 群組 | 波動率 | 趨勢 | 股票數 | 有效策略數 |
|------|--------|------|--------|-----------|
| med_vol/trending | 中 | 有 | 209 | 13（全部） |
| low_vol/trending | 低 | 有 | 234 | 12 |
| med_vol/sideways | 中 | 無 | 220 | 8 |
| high_vol/trending | 高 | 有 | 218 | 5 |
| high_vol/sideways | 高 | 無 | 209 | 4 |
| low_vol/sideways | 低 | 無 | 194 | 0 |

## 回測引擎 `backtest.py`

唯一的回測腳本，所有策略都在這裡定義與執行。

架構（v2）：
- **策略函數無狀態**：不追蹤持倉，只輸出 buy/sell 條件
- **`simulate()` 統一管理**：持倉、停損、出場規則全由引擎控制
- 出場規則（exit_rule）：`signal`（指標賣訊）、`hold_days`（持 N 天）、`profit_or_hold`（獲利或持 N 天）、`gap_fill`（缺口回補或持 N 天）

包含 14 個通過群組驗證的策略（逆勢 3 / 順勢 8 / 量價 2 / 波動率 1）：

| 策略 | 類別 | 有效群組數 | 最佳群組均報酬/筆 |
|------|------|-----------|----------------|
| 唐奇安改良 | 順勢 | 2 | 4.15%（med_vol/trending） |
| AD背離 | 量價 | 3 | 3.68%（low_vol/trending） |
| MACD改良 | 順勢 | 5 | 3.44%（med_vol/trending） |
| 波動率擠壓 | 順勢 | 3 | 2.92%（med_vol/trending） |
| 唐奇安+ADX | 順勢 | 5 | 2.63%（high_vol/trending） |
| N日新高改良 | 順勢 | 5 | 2.11%（high_vol/trending） |
| KD(9) 30/70 | 逆勢 | 3 | 1.84%（med_vol/trending） |
| 雙均線改良 | 順勢 | 3 | 1.75%（med_vol/trending） |
| 超跌反彈 | 逆勢 | 2 | 1.65%（med_vol/trending） |
| KD(5)快速 | 逆勢 | 2 | 1.44%（med_vol/trending） |
| N日新高+量 | 順勢 | 3 | 1.32%（med_vol/trending） |
| 量價突破 | 量價 | 3 | 1.26%（med_vol/trending） |
| 雙均線(5/20) | 順勢 | 2 | 1.00%（med_vol/trending） |
| MACD快速(8,17,9) | 順勢 | 4 | 0.96%（med_vol/trending） |

策略函數仍保留在 `backtest.py` 中（供參考），但 `STRATEGIES` dict 只包含通過驗證的 14 個。

### 最佳策略組合（投組模擬結果）

目標群組：**med_vol/trending**（中波動趨勢股）。
達標門檻：年化 ≥30%, DD ≤15%, Sharpe ≥1.5, 交易/年 ≥50。

#### 方案 A：三策略分池 + 各自篩選 [4/4]（最佳）

| 指標 | 值 |
|------|-----|
| 年化報酬 | **+34.0%** ✓ |
| 最大回撤 | **9.1%** ✓ |
| Sharpe | **2.58** ✓ |
| 交易/年 | **114** ✓ |

三策略各 1/3 資金，**獨立管理持倉**（分池），**每個策略各自篩選適合的股票**：
- **波動率擠壓**：125 檔, max=7, alloc=1/7, trail=4%（不限每日新倉/擁擠過濾）
- **超跌反彈**：100 檔, max=7, alloc=1/7, 每日限買 1, 同日>3 全跳過, trail=8%
- **AD背離**：116 檔, max=5, alloc=1/5, 每日限買 1, 同日>3 全跳過, trail=8%

**個股篩選規則**：用 Period A（2020-2022）的每策略每檔股票平均交易報酬 > 0 篩選。
每個策略邏輯不同，適合的股票不同（三策略交集僅 33 檔，只適合單一策略的有 60 檔）。
篩選腳本：`/tmp/stock_filter_per_strategy.py`

對照無篩選基準（209 檔同名單）：+30.8%, DD 12.3%, Sharpe 2.11, 150/yr [4/4]

#### 方案 B：純波動率擠壓 [4/4]

| 指標 | 值 |
|------|-----|
| 年化報酬 | **+32.4%** ✓ |
| 最大回撤 | **13.9%** ✓ |
| Sharpe | **1.81** ✓ |
| 交易/年 | **55** ✓ |

單策略配置：max=5 持倉, alloc=1/5, 每日最多 3 新倉, 同日 >3 信號全跳過, trailing stop 4%

#### 三策略個別績效（各用 100% 資金）

| 策略 | 年化 | DD | Sharpe | 交易/年 | 均報酬/筆 | 平均持有 |
|------|------|-----|--------|---------|----------|---------|
| 超跌反彈 | +36.6% | 17.7% | 1.65 | 58 | +4.66% | 29.9 天 |
| AD背離 | +31.1% | 23.0% | 1.62 | 27 | +6.06% | 46.0 天 |
| 波動率擠壓 (max=7) | +22.9% | 11.9% | 1.67 | 65 | +2.49% | 13.5 天 |

三策略都是正報酬。分池組合後 DD 從 17-23% 降到 12.3%（分散效果）。
加上各自篩選後 DD 進一步降到 9.1%，Sharpe 從 2.11 升到 2.58。

### 投組模擬方法（重要警告）

多策略投組模擬**必須用分池管理**（每策略獨立資金池、獨立持倉）。

**絕對不可以用「共享池」**（所有策略信號丟進同一個池搶位置）：
- 長持有策略（超跌 30 天、AD 46 天）會堵住位置，短持有策略（擠壓 13.5 天）進不了場
- 共享池會嚴重低估績效（同樣三策略，分池 +30.8% vs 共享池 +18.7%）
- 分池的正確實作參考：`/tmp/random_50_test.py` 的 `triple_combo()`

### 籌碼策略研究結論（2026-03-01）

- 13 個 volume-normalized 籌碼策略全部通過群組驗證，但投組年化最高 ~14%，**單獨無法達到 30% 門檻**
- 正確的籌碼資料用法：volume-normalized ratio（net_buy/volume）、rolling window 累積、1-day lag（收盤後才公布）
- 最有價值的發現：Williams %R(14) < -90 超賣反彈 + 投信 5 日過濾 = med_vol/trending +10.96%/筆，但信號太少（~16/年）且持有期長（~100天），無法與擠壓組合
- 資料來源：`data/institutional/`（三大法人）、`data/margin/`（融資融券）

## `strategies/` — 策略資料

- `filtered_stock_lists.json` — 三策略各自篩選後的股票名單（擠壓 125 / 超跌 100 / AD 116 檔），含每檔的 A/B 期交易數和均報酬
- `strategy_group_validation.json` — 群組驗證結果（14 策略 × 6 群組）

### `WATCHLIST.md` — 觀察清單（根目錄）

三策略分池組合的配置摘要、進出場條件、個股篩選說明。

## 資料來源

- **FinMind API** — token 存於 `.env.local`（`FINMIND_API_TOKEN`）
- **Supabase** — 股票列表 + 歷史股價，本地快取於 `data/stock_prices/`
