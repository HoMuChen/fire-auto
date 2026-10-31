# 台股資料湖（Cloudflare R2）

fire-auto 的研究級台股資料（2009/2010 起、含下市股、還原權息、財報、籌碼、大盤與期權），以 parquet 長表放在 R2，供各專案**唯讀**。
寫入只由 fire-auto 這台機器的 `lake/lake.py` 執行。資料來源 FinMind，**僅限個人使用，bucket 不公開、不外流**。

## 讀者：三步驟

1. 取得**唯讀** token（Cloudflare → R2 → Manage API tokens → Object Read only，限定 bucket `trading-homuchen-com`），放進自己專案的環境變數：
   `R2_ACCOUNT_ID`、`R2_READ_ACCESS_KEY_ID`、`R2_READ_SECRET_ACCESS_KEY`
2. 安裝 DuckDB（≥ 1.1；Python `pip install duckdb`，或 CLI / Node / R 版皆可）
3. 掛載目錄檔，直接用表名查：

```python
import os, duckdb
con = duckdb.connect()
con.sql(f"""CREATE SECRET (TYPE r2, KEY_ID '{os.environ["R2_READ_ACCESS_KEY_ID"]}',
            SECRET '{os.environ["R2_READ_SECRET_ACCESS_KEY"]}', ACCOUNT_ID '{os.environ["R2_ACCOUNT_ID"]}')""")
con.sql("ATTACH 'r2://trading-homuchen-com/v1/catalog.duckdb' AS tw (READ_ONLY)")

df = con.sql("""SELECT date, stock_id, close, adj_close, volume
                FROM tw.price WHERE stock_id = '2330' AND date >= '2024-01-01' ORDER BY date""").df()
con.sql("SELECT * FROM tw.market.taiex ORDER BY date DESC LIMIT 5").show()
con.sql("SELECT schema_name, view_name, comment FROM duckdb_views() WHERE database_name = 'tw'").show()   # 表清單與說明
```

DuckDB 2.0 注意：一律用 `con.sql(...)` 再接 `.df()` / `.fetchall()` / `.show()`；2.0 的查詢結果已沒有 `fetchone()`，
而 `con.execute()` 的結果沒讀完前，同一個連線不能再下其他指令。

DuckDB 只會抓需要的欄位與年份分區（`WHERE date >= …` 會跳過其他年份的檔）；下載流量免費。

**不用 DuckDB**：`_manifest.json` 列出每個資料集的所有檔案路徑，polars / pandas / pyarrow 用 S3 介面讀即可
（endpoint `https://<R2_ACCOUNT_ID>.r2.cloudflarestorage.com`、region `auto`）。

**快取由讀者自行決定**。要整批放本機：`rclone sync r2:trading-homuchen-com/v1/price ./cache/price`（只下載有變的檔）。
內容沒變的檔絕不改寫；有沒有變以 `_manifest.json` 各檔的 `sha256` 為準。平常每天只有今年的分區與小檔會變，
但**新股上市（回補歷史）與除權息（`adj_*` 整段重新還原）會動到過去年份**——除權息旺季 `price` 可能每天全部更新（約 280 MB，下載免費）。

## 資料集

| 表 | 內容 | 分區 | 起 |
|---|---|---|---|
| `universe` | 股票池 2,594 檔：stock_id、name、industry、market（twse/tpex）、delist_date | — | — |
| `price` | 日線：原始 `open/high/low/close`（close=0 為無成交）＋還原 `adj_*`；`volume` 股、`money` 元、`turnover` 筆 | year | 2010 |
| `per` | `PER`、`PBR`、`dividend_yield` | year | 2009 |
| `revenue` | 月營收（元）＋ `avail_date`（次月 11 日） | — | 2009 |
| `inst` | 三大法人買賣（股），`name` 區分法人別 | year | 2009（2012-05 起完整） |
| `margin` | 融資融券（張），FinMind 原欄位 | year | 2009 |
| `fin` / `bs` / `cf` | 損益（單季）/ 資產負債（季底）/ 現金流量（**年初累計**），長表 `type`/`value`，元；`avail_date`＝法定截止 +1 天 | — | 2009 / 2012Q4 / 2012Q1 |
| `mv` | 日市值（元）⚠ FinMind int32 溢位錯值未修 | year | 2009 |
| `holding` | 集保股權分散（級距、人數、%、股） | year | 2010 |
| `dividend` | 股利政策與除權息日（FinMind 原欄位） | — | 2009 |
| `market.*` | `taiex`、`taiex_tr`（報酬指數）、`mtx_daily`、`txo_chain`、`txo_weekly`、`txo_final_settle`、`tx_large_traders`、`mkt_inst`、`mkt_margin`、`gspc`；自建：`txo_vix`、`txo_oi`、`ad`、`breadth`、`mkt_maint` | — | 2005~2012 |
| `futures_30min.qff` / `.sef` | 小型台積電 / 小型智邦期 30 分 K（日盤、連續近月；datetime 台北時間、K 棒起始） | — | 2021 / 2024 |

慣例：`date` 為 date 型別、`stock_id` 為字串；除 `price` 外欄位名沿用 FinMind 原名（可對照 FinMind 文件）。

## 研究防前視提醒

- **財報 / 營收要用 `avail_date` 對齊，不要用 `date`**（`date` 是期末日 / 營收月份，當時還沒公布）。
- 還原價 `adj_*` 是**事後**還原（之後的除權息會回頭改動歷史值）；成交量、流動性、漲跌停判斷用原始價。
- 股票池含下市股（`universe.delist_date`），回測要用 point-in-time 名單，不要只取現存股票。
- **最新一季財報可能被改寫**：FinMind 剛公布時常先給「年初累計」，之後才改成單季（2026Q2 曾有近千檔是 H1 累計值）。
  每日更新會重抓最近 3 季並覆蓋；用最新一季前請確認 `_manifest.json` 的更新時間，或與上一季比對量級。
- 日資料最近 10 個交易日會每天重抓覆蓋（FinMind 會事後修正成交量等），所以「最近幾天」的值可能小幅變動。

## 發布端（只在 fire-auto 這台機器）

```
python3 lake/lake.py build      # data/v2 等 → data/lake/v1（內容沒變的檔不改寫）
python3 lake/lake.py verify     # 筆數、2330 與 research/v2/harness.load_stock 比對
python3 lake/lake.py upload     # 只傳變動檔 → 連 R2 建 catalog.duckdb → _manifest.json → 刪孤兒檔
python3 lake/lake.py status
```

寫入 token（Object Read & Write）只放這台機器的 `.env.local`：`R2_ACCOUNT_ID`、`R2_WRITE_ACCESS_KEY_ID`、`R2_WRITE_SECRET_ACCESS_KEY`（`R2_BUCKET` 預設 `trading-homuchen-com`）。
發布順序保證讀者不會看到半套：資料檔先傳，`_manifest.json` 最後才換；孤兒檔在 manifest 切換後才刪。

## 每日更新（第 2 階段）

`python3 lake/lake.py daily`＝`lake/update.py`（FinMind 依日期一次抓全市場，增量更新 `data/v2`）→ `build` → `verify` → `upload`；失敗發 Telegram。

| 資料 | 抓法 | 覆蓋規則 |
|---|---|---|
| raw / adj / per / inst / margin / mv | 最後日期 − 10 天起逐個平日 bulk | 當天有資料就覆蓋同日；股價 0 筆＝休市跳過 |
| adj 除權息 | 重疊日新舊收盤不一致 | 該檔整段重抓（後復權會改歷史） |
| revenue | 最近 3 個月逐月 bulk | 依（年, 月）覆蓋 |
| fin / bs / cf | 最近 3 個季底 bulk | 依季底覆蓋（跟到 FinMind 的累計 → 單季修正） |
| dividend | 近 120 天 | 依（date, year）覆蓋 |
| holding | 最後日 − 3 天起逐個平日（公布日不固定） | 依日期覆蓋 |
| universe | TaiwanStockInfo + Delisting | 新股逐檔抓完整歷史；下市標 delist_date |
| market.* | 從最後日期所在月 / 年重抓 | 依日期覆蓋；自建 txo_vix / ad / breadth / mkt_maint 整段重算 |

建議 cron（由使用者加；放在 broker 22:00 之後、避開 API 尖峰）：
`30 23 * * 1-5 cd /Users/mu/fire-auto && /usr/local/bin/python3 lake/lake.py daily >> data/lake_daily.log 2>&1`

第 3 階段（未做）：分點（`data/broker`）、訂單簿（`data/orderbook`）。
