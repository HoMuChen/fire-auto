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

### `data/cache_meta.json` — 快取 metadata

記錄每檔股票在本地的最新快取日期，格式：`{"2330": "2026-02-25", ...}`

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

### 工作流程

1. 定義策略邏輯（進場/出場條件）
2. 用合理預設參數實作
3. 對全部個股跑回測
4. 彙總統計結果
5. 若結果有意義，檢驗參數穩健性（微調參數看結果是否劇烈變動，變動大就捨棄）
6. 保留穩健的策略，記錄結論

## 回測引擎 `backtest.py`

唯一的回測腳本，所有策略都在這裡定義與執行。

架構（v2）：
- **策略函數無狀態**：不追蹤持倉，只輸出 buy/sell 條件
- **`simulate()` 統一管理**：持倉、停損、出場規則全由引擎控制
- 出場規則（exit_rule）：`signal`（指標賣訊）、`hold_days`（持 N 天）、`profit_or_hold`（獲利或持 N 天）、`gap_fill`（缺口回補或持 N 天）

包含 24 個策略（逆勢 18 / 順勢 3 / 量價 1 / 型態 2）。

執行：
```bash
python backtest.py
```
- 讀取 `/tmp/sideways_volatile.json`（367 檔高波動盤整股）
- 對每檔股票跑全部 24 策略
- 篩選報酬 > 50% 的組合，寫入 `strategies/*.json`

## 策略研究結果 `strategies/`

回測後可行的策略記錄在 `strategies/` 目錄，**每檔股票一個 JSON 檔**。

- 檔名格式：`{stock_id}_{股票名稱}.json`（如 `3615_安可.json`）
- 每個檔案包含：股票基本資訊、回測期間、買入持有報酬、可行策略清單
- 每個策略記錄：名稱、類別（順勢/逆勢）、進出場條件描述、報酬率、交易次數、勝率、平均持有天數、最大回撤、Sharpe ratio
- 只記錄達到目標的策略（如回測報酬 > 50%），不記錄失敗的
- 每個 JSON 檔包含 `backtest_engine` 和 `engine_version` 欄位，記錄產出來源

完成一檔股票的策略研究後，將可行策略寫入對應的 JSON 檔。

### `WATCHLIST.md` — 觀察清單（根目錄）

風險調整後的觀察清單，篩選條件：Sharpe ≥ 1.4、回撤 < 20%、勝率 ≥ 60%。
連結指向 `strategies/*.json`（策略檔）和 `charts/*_chart.html`（圖表）。

### `charts/` — 策略回測圖表

每檔 watchlist 股票一個 HTML 圖表，檔名 `{stock_id}_{股票名稱}_chart.html`。
包含股價走勢、技術指標 overlay、進出場標記、資金成長曲線、交易明細表。
產生方式見 `.claude/skills/generate-strategy-chart.md`。

## 資料來源

- **FinMind API** — token 存於 `.env.local`（`FINMIND_API_TOKEN`）
- **Supabase** — 股票列表 + 歷史股價，本地快取於 `data/stock_prices/`
