# Fire-Auto

台股交易研究與自動化：FinMind API 抓資料 → 本地快取 → 防前視回測研究 → 少數通過驗證的策略上實盤（Telegram 通知 / 永豐 Shioaji 期貨下單）。

> **新 session / 新 agent 先讀這段**
> 1. **現在實盤只有兩條線**：股票 **H136 營收驚喜＋強勢月換股**（`h2_monthly.py`，2026-11-02 由 H29 換成 H136、首次實盤前切換；只發通知、人工下單）、期貨 **QFF 小型台積電網格**（`futures_live.py`，**真實自動下單中**）。另有 SEF 小型智邦網格「提醒」與總經儀表。
> 2. **舊三策略（擠壓/超跌/AD）、方案 A/B、14 技術策略、季節性、籌碼、分點過濾全部已失效**，程式與文件在 `legacy/`（見 `legacy/README.md`）。使用者說「跑 scan」時要先提醒這是已停用的舊系統。
> 3. **新策略研究一律走 `research/v2/`**（防前視框架、事前登記、三段期間），規則見下方「研究規範」。**holdout（2024-）只有使用者明確同意才能開封。**
> 4. 不要動使用者的期貨 live 設定、cron 與帳本；不要 commit 使用者自己未提交的研究檔（`git status` 裡不是你改的檔）。

## 目錄地圖

| 位置 | 用途 |
|------|------|
| `h2_monthly.py` | ★ 實盤：H136 月換股選股 + Telegram（檔名沿用；`--h29` 退回前任 H29、`--h2` 算舊 H2 對照） |
| `futures_live.py` | ★ 實盤：QFF 網格真實下單執行器（9 道安全機制，見檔頭） |
| `futures_grid_monitor.py` | 期貨網格純提醒版（現用於 SEF；`--symbol QFF/SEF`） |
| `macro_dashboard.py` | 每日總經預測市場儀表（Polymarket + `claude -p` 解讀，參考用、非 alpha） |
| `notify.py` | Telegram：`send(text, md=False, intraday=False)`；`intraday=True` 才會多發盤中名單 |
| `stock_cache.py` / `chip_cache.py` / `broker_cache.py` | 每日資料快取（股價 / 法人+融資券 / 券商分點） |
| `backtest.py`、`validate_strategy_groups.py` | 舊函式庫（技術指標、成本常數）；TSMC 網格研究與 `research/v2/run_legacy14.py` 仍 import，**不要搬** |
| `tsmc_dual_pool_strategy.py`、`gen_tsmc_report.py` | TSMC v4 三 sleeve 現金流策略與報告（台積電專屬，未經 v2 驗證） |
| `research/v2/` | ★ 現行研究框架與所有假設紀錄（見 `research/v2/README.md`） |
| `lake/` | 資料湖：`lake.py build/verify/upload` 把 `data/v2` 等轉長表 parquet 發布到 Cloudflare R2（bucket `trading-homuchen-com`、前綴 `v1/`，含 `catalog.duckdb`），供其他專案唯讀；讀法見 `lake/README.md`。`lake/update.py` 每日增量更新 `data/v2`（`lake.py daily` 00:30）；另含分點（`data/broker`，月分區）、訂單簿（`data/orderbook`，一天一檔）、台指期日盤 1 分 K（`data/v2/futures/tx1m`，2011 起）與上櫃整體法人／融資（`market.mkt_inst_otc`／`mkt_margin_otc`，櫃買官網 `research/v2/tpex_mkt.py`，2008 起；FinMind 的 `mkt_inst`／`mkt_margin` 只有上市） |
| `research/build_data_v2.py` | 重建 `data/v2/`（含下市股、還原權息、2010 起） |
| `research/futures_cache*.py`、`research/tsmc_*`、`research/grid_*`、`research/txo_vix.py` | 期貨 / TSMC 網格研究與 30 分 K 快取（使用者主導的研究線） |
| `strategies/` | 仍有效的研究筆記：TSMC 網格收租、外資棄守退場、外資大賣→長紅（統計）、美國數據公布前（迷思） |
| `analysis/` | 個股研究清單：`analysis/watchlist.json`＝**單一真實來源**。**不分追蹤／不追蹤**（2026-11-07 使用者決定），只分「做過研究」和「沒做過」：做過綜合分析的股票一律放 `tracking`（`passed` 留空、只為相容 trading.homuchen.com 的讀取；`removed` 只放下市、合併），每檔有論點（可以是負面結論）、持續關注的點（質化問題、檢查點）、價位帶「便宜／合理／昂貴」、結論要改的條件（`exit_conditions`）。**增量更新只在使用者說「更新研究」時跑、不排程**：每月（月營收公布後）逐檔看營收、到期檢查點、更新質化判斷、對照價位帶並寫紀錄；每季（財報截止日後）寫新日期的報告、重算價位帶。**一律編輯 `analysis/watchlist.json`，改完執行 `analysis/publish.py` 自動驗證並重繪 `analysis/WATCHLIST.md` 與發布至 R2 `analysis/v1/`，勿直接手改 `WATCHLIST.md` 以免被覆蓋**（本機純檢查用 `publish.py --check-only` 或 `render_watchlist.py`）；`analysis/screen_quality.py`＝找新研究對象的一種來源（本業體質好、股價弱的篩選，只是候選，不是訊號） |
| `legacy/` | ⛔ 已停用的舊系統與舊研究（只供查閱） |
| `data/` | 快取與狀態（大多 gitignore，見下） |

## 實盤系統

### 股票：H136「營收驚喜＋強勢」月換股（`h2_monthly.py`；2026-11-02 使用者決定由 H29 換成 H136，首次實盤前切換）
- 規則（**凍結，不可改**；與回測 `research/v2/feature_combo.py` FS2 引擎一致）：每月第一個 ≥12 日的交易日**開盤**換股；股票池＝上市/上櫃普通股、近 60 交易日均量 ≥200 張、≥250 根 K；分數＝H2（前一交易日還原收盤 / 近 252 交易日最高還原收盤）、SUR（月營收驚喜＝當月 YoY 金額差減前 24 個月平均、除以標準差；m 月營收視為 m+1 月 11 日才可得，75 天未更新視為缺）、H97（前一交易日還原收盤 / 上月最後交易日還原收盤 − 1）三者**各自在股票池內的百分位，取有值者平均（≥2 項）**；取最高 **20 檔**等權；無停損、無濾網；平手依代號小→大。
- 流程：cron 平日 20:30 跑，只有「下一平日是本月第 1~3 個 ≥12 日的平日（第 2~3 個＝失敗補發）且本月尚未換股」才計算（API：近 ~95 天全市場原始價 + 流動股還原價 + 40 次全市場月營收）並發 Telegram 到 **`TELEGRAM_CHAT_ID` 一個人**（使用者明確要求：不發盤中名單）。使用者收到後人工下單。
- 狀態：`data/h2_state.json`（檔名沿用；每月名單與 `strategy`；`capital` 設了訊息才會算張數）。首次實盤換股 2026-11-11 晚通知。
- 手動：`python3 h2_monthly.py --force --no-send`（立刻算、不存狀態、不發）；`--asof YYYY-MM-DD` 用指定日收盤重算；`--h29` 前任 H29、`--h2` 舊 H2。
- 績效（top20，扣成本，月報酬；同引擎）：年化超額 dev +28.0% / val +49.2% / holdout 2024-01~2026-09 **+52.4%（t 3.66）**，H29 同期 +18.3% / +34.0% / +39.9%。對 H29 月差：holdout +1.04%（t 1.18，3/3 年），2011~2026/9 合計 +0.90%（t 3.04，13/16 年）；⚠ H136 是 FS2 3330 組的 dev 第一名，dev 有選擇偏差，乾淨證據只有 holdout。月換手 ~85%（每月約換 17 檔，H29 約 16 檔）。
- **失效判準 M1（2026-11-07 上線前寫定，`research/v2/h136_monitor.py check`，每月換股後跑）**：從 2026-11 換股起，第 12／24／36 個實盤月的累積超額低於 −9.3%／−5.9%／+1.7%，或 net 從高點回落超過 −30.2% → **檢討**（先查執行與資料，停不停由使用者決定）；沒觸發前不改規則、不重算門檻。⚠ 12 個月內回撤 −20% 屬正常；實盤一年幾乎分辨不出策略有沒有壞（零邊際也只有 36% 會在第 12 月觸發）。
- **前任 H29「雙新高」**（2026-10-28 ~ 11-02，未實際下單過）：H2 與 H10（近 3 月營收 / 12 個月內最高 3 月）各自在全上市櫃普通股的百分位平均；三段皆過（年化 dev +22.8% / val +66.9% / holdout +74.1%），`--h29` 可退回。
- **前任 H2「52 週高點接近度」**：只看股價分數，三段皆過（dev +19.0% / val +58.4% / holdout +57.6%）；62% 換股日超過 20 檔同分、依代號取 → H29 用營收新高打破同分。

### 研究候選 / 已淘汰（未上線）
- ~~H56 小台指「特定法人淨部位」多/空手~~：dev/val 過，但 **holdout（2026-10-21 開封）Sharpe 0.92 < 加權報酬指數 1.65 → ✗ 淘汰**（`research/v2/mtx.py`；小台指 13 個假設目前無三段皆過者）。
- **H75 做多小台指 + 每週賣 1σ 價外買權**（`research/v2/options.py`）：dev 0.93 / val 0.95 過，但疊加層單獨無溢價、只是降波動（勝做多 ~0.03）；holdout 門檻 1.65 預期必敗，**未開封**。
- **H46 產業中性營收新高多空**（個股期貨，`research/v2/neutral.py`）：名目 1 倍 dev +5.8% / val +5.6%、MDD −9.5% / −4.5%、與大盤 ρ≈−0.1。**使用者決定只記錄、不開封、不實盤。**

### 期貨：QFF 小型台積電網格（`futures_live.py`）
- **`data/LIVE_QFF_ENABLED` 存在 = 真實下單模式**（目前開著）；刪掉 = 提醒模式；建 `data/AUTO_OFF` = 全停（緊急煞車）。
- 帳本 `data/live_ledger_QFF.json`（逐口建倉價，驅動停利；**不可用券商均價取代**）。
- 需用 `.venv/bin/python`（含 shioaji）；憑證與金鑰在 `.env.local`（`SJ_*`）、`Sinopac.pfx`。

### 期貨記帳與 QFF 參數

`data/futures_trades.csv`：**與股票策略系統完全獨立**，純為個人期貨成交紀錄，手動維護。

欄位：`成交日期, 商品, 買賣, 倉別, 盤別, 成交均價, 成交口數, 委託書號`（忠實對應券商「期權-歷史成交回報」）

- 商品如 `小型台積電期202607`（202607 = 結算月份）
- 損益計算：小型台積電期 = **100 股/口**，1 點（價格漲跌 1 元）= 每口 NT$100；FIFO 配對進出場
- 手續費/期交稅依券商而定，未內含於檔內

**⚠️ 記帳分工（2026-09-18 定案，三邊不重疊，勿混）**：

| 帳戶/系統 | 記在哪 | 維護者 |
|-----------|--------|--------|
| **台新帳戶（手動操作）** | 本檔 `data/futures_trades.csv` | 手動 |
| **永豐 QFF live（自動）逐筆成交** | 券商「歷史成交回報」（不另存本地檔） | 券商 |
| **永豐 QFF live（自動）網格狀態** | `data/live_ledger_QFF.json` | `futures_live.py` 自動 |

- **本檔（`futures_trades.csv`）自 2026-09-18 起專指「台新帳戶的手動操作」**（與永豐自動帳戶完全分離），`futures_live.py` 完全不碰它。
- QFF live 的逐筆成交以券商紀錄為權威，**不在本地重複記**（券商保留期有限，需永久留檔時才偶爾手動抄進本檔）。
- `data/live_ledger_QFF.json` 是網格**運作狀態**（逐口建倉價 → 驅動停利/下一買點），**非歷史流水，也不可用券商均價取代**（券商轉倉會壓平成單一均價，算不出各格獨立停利）。

**QFF live 網格參數（`futures_live.py` / 帳本 config）**：本金 30萬／格距 1.2%／停利 +1.2%／槓桿上限 **5x**（2026-10-15 由 3x 調高，使用者明確要求；對真實 `margin.equity_amount` 卡，非 config 資金）／`intraday_once`（一天最多買 2 次，當天買過後只剩 13:15 收盤根能再買）／硬上限 12 口。⚠ 5x 滿載再跌 ~11% 即踩追繳線，使用者配套＝跌勢延續時補保證金。改動要同步帳本 `config.max_leverage` 與 `default_ledger()` 預設。
- **選用功能（2026-11-03 加，預設關閉）**：帳本 config `trail_tp`（移動停利，0＝關）、`lev_gate`（`{"lo": 2.0, "full_drop": 0.30}`＝槓桿上限隨距近一年高跌幅從 lo 放大到 max_leverage，null＝關）。缺這兩個 key＝舊行為。回測（`research/tsmc_futures_grid_30min.py --trail-tp/--lev-gate-lo/--lev-gate-drop`）與風險檢查見 `strategies/tsmc_grid_income.md` 最後一節；**是否開啟由使用者決定，不要代開**。

### 其他
- SEF 小型智邦網格：`futures_grid_monitor.py --symbol SEF` 純提醒，狀態 `data/futures_grid_state_SEF.json`，使用者手動 `buy/sell` 回報。SEF 期貨 2024-02 才上市、無空頭樣本，只用 1x。
- 總經儀表 `macro_dashboard.py`：每日 08:01 發 Telegram，參考用。

## 排程（crontab，平日）

| 時間 | 指令 | 性質 | Log |
|------|------|------|-----|
| 08:01 | `macro_dashboard.py` | 參考儀表 | `data/macro_dashboard.log` |
| 08:40 | `.venv/bin/python research/orderbook_recorder.py record`（錄到 13:50）→ `pack` | 資料：QFF/TXF/2330 五檔+逐筆 | `data/orderbook_recorder.log` |
| 09:15~13:45 每 30 分 | `.venv/bin/python futures_live.py` | ★ QFF 真實下單 | `data/futures_live.log` |
| 09:15~13:45 每 30 分 | `.venv/bin/python futures_grid_monitor.py --symbol SEF` | SEF 提醒 | `data/futures_monitor_sef.log` |
| 18:00 | `stock_cache.py today` | 資料：股價（**broker_cache 靠 `data/stock_prices/2330.csv` 判斷交易日，不可刪**） | `data/stock_update.log` |
| 18:30 / 18:35 | `research/futures_cache.py` / `research/futures_cache_sef.py` | 資料：QFF / SEF 30 分 K | `data/futures_cache*.log` |
| 20:30 | `h2_monthly.py` | ★ H136 換股通知 | `data/h2_monthly.log` |
| 22:00 | `broker_cache.py today`（約 50~80 分） | 資料：分點（進資料湖） | `data/broker_daily.log` |
| 00:30（週二~六） | `lake/lake.py daily` | 資料湖：增量更新 `data/v2` → build → verify → 上傳 R2 | `data/lake_daily.log` |

股票腳本用 `/usr/local/bin/python3`（系統 `/usr/bin/python3` 沒有 pandas）；期貨下單/提醒用 `.venv/bin/python`。2026-10-20 已移除 `scan.py` 08:00 與 `monitor.py` 盤中兩個 cron；2026-10-31 移除 `chip_cache` 19:00（無人使用），分點回補完成後 22:00 只跑 `today`，`lake.py daily` 排在午夜後（`2-6`＝處理前一個平日）。

## 資料

- **FinMind API** 是唯一資料來源（token：`.env.local` 的 `FINMIND_API_TOKEN`；使用者方案 **6000 次/小時**）。Supabase 已停用。
- 打網路的指令在沙盒內會被擋 → 需 `dangerouslyDisableSandbox`。
- **`data/v2/`（研究專用，gitignore，`research/build_data_v2.py` 重建約 65 分鐘；2026-10-31 起由 `lake/update.py` 增量更新，不再是凍結快照——最近 10 個交易日、最新 3 季財報會被覆蓋修正，除權息會改寫 adj 歷史）**：`universe.csv`（2,594 檔普通股，含 58 檔不在 TaiwanStockInfo 的下市股，`delist_date` 標 2010 後下市）、`raw/{id}.parquet`（原始價：流動性、漲跌停）、`adj/{id}.parquet`（還原權息：報酬；下市股自建）、`revenue/ per/ inst/ margin/ fin/ bs/`（`research/v2/fetch_fundamentals.py`）、`taiex_tr.parquet`（加權報酬指數）。raw 有 close=0 停牌日需防護。
- **每日快取（gitignore，可重抓）**：`data/stock_prices/{id}.csv`（近 2 年未還原、無下市股，**不可用於研究**）、`data/institutional/`、`data/margin/`、`data/broker/{id}.parquet`（2021-06 起分點）、`data/futures/`（QFF/SEF 30 分 K、TXO 日資料與自建台指 VIX）。用法：`StockCache().get/update_today()`、`ChipCache().get_institutional/get_margin()`、`BrokerCache().get()`。
- `individual_stocks.json`：舊個股清單（快取模組用來決定抓哪些股票）；`data/low_liquidity_stocks.json`：舊流動性名單。
- **進版控的個人紀錄**：`data/trades.csv`（股票成交，舊三策略時期）、`data/futures_trades.csv`（台新帳戶手動期貨）。其餘 data 檔一律 gitignore。

## 研究規範（`research/v2/`，2026-10-20 起）

- **期間固定**：dev 2010-01-01~2019-12-31（開發）／val 2020-2023（規則凍結後才跑）／holdout 2024-（`unseal=True`，**需使用者同意、只看一次**）。
- **特徵庫**：`research/v2/FEATURES.md`（`feature_library.py` 產生）。新研究先從這裡依用途取原料；特徵只看自己夠不夠好，不以改進現有實盤策略為門檻。找新特徵的分階段計畫（交接用）在 `research/v2/FEATURE_PLAN.md`。用特徵找策略的分階段計畫在 `research/v2/STRATEGY_PLAN.md`（目標：比基準好即可，不和既有策略比）。
- **事前登記**：跑 dev 前先把假設、規則、過關標準寫進 `research/v2/HYPOTHESES.md`；dev 沒過的不准跑 val；每次執行自動記到 `research/v2/experiments.jsonl`（進版控）。
- **防前視**：`leak_check` / `leak_check_xs`（截斷資料重算訊號須一致）；pit 流動性股票池（60 日均量 ≥200 張、上市櫃、含下市股、不含興櫃、≥250 根 K）；**次日開盤成交**；一字漲跌停無法成交（2015-06-01 前 7%、後 10%）；下市以最後收盤出場。營收視為次月 11 日可得；財報 Q1→5/16、Q2→8/15、Q3→11/15、Q4→次年 4/1。
- **成本**：買 0.1425%、賣 0.1425% + 證交稅 0.3%（來回 0.585%）；期貨每腳單邊約 0.05~0.1%。
- **橫斷面判定**（主判定 top20 等權月換股）：超額（vs pit 等權大盤）t ≥ 2、dev ≥7/9 年正、分位大致單調；val 再過一次。
- **核心哲學**：參數用市場慣用值、不做參數搜索；改邏輯不調參；微調參數結果劇變就捨棄；必須全市場（整個 pit 池）有效，不挑個股；看過結果才提出的組合標「事後」並打折。
- **已知規律**（100 個假設的結論，詳見 `research/v2/README.md`）：
  - 台股橫斷面是「**延續**」而非「反轉」：1 月反轉、彩券效應、盤中反向、產業內反轉、配對交易全部失敗或反向。
  - 可交易的只有「**強勢延續（H2）**」與「**營收創新高（H10 系）**」；價值、殖利率、低波動、非流動性、籌碼、ROE/毛利、SUE、季節性、大盤擇時、絕對門檻、期貨做空避險都不過。
  - 事件型策略（營收突破）每筆超額為正但分布像彩券（中位數 < 0），壓到 10~20 檔後失效。
  - 營收類訊號（YoY、創新高、月營收驚喜 SUR）分位都很強，但單獨集中成 20 檔都不過；只有跟 H2 合成（H29）才過，在 H29 上再加 SUR 也沒增量；但把 H10 換成 SUR＋營收公布窗報酬（H136，FS2 隨機組合找到）三段都較好、2026-11-02 上線。財報品質（F-score、應計、現金流殖利率）、集保籌碼、小型股、改良動能（殘差、產業、FIP）都不過。
  - H2 有 62% 的換股日超過 20 檔同分（創新高），凍結規則依代號小→大取；改用突破幅度排序（H95）val 反而更差。
  - 6~8 月「去年同期有除息」的股票整體有溢酬（毛 +0.72%/月，事後診斷），但挑高殖利率反而最差（H100 ✗）。
  - 短線（持有 3~10 天）在現股不可行：0.585% 來回成本吃掉邊際。
  - 小台指擇時（19 個）與台指選擇權賣方（月選 6 個、週選 4 個）都贏不了單純做多：MTX 連續做多含逆價差收益、Sharpe 高；選擇權溢價小、左尾重，價外買權零溢價。
  - 分點籌碼（2021-07 起才有資料，只能探索、無法走三段）：課程式條件（主力大買、集中度、量增、大戶加碼）扣掉動能後無效；唯一有東西的是「全市場分點能力排名」有延續性（前 10% 分點大買 → 5 日 +0.1～0.2%、兩段都正），但幅度小於成本，詳見 `research/v2/README.md`「探索」。
  - 市值前 20 大（個股期貨做多，第五十九～六十二批）：短線反轉、網格、波動目標化、除息前後、池內輪動、ADR 隔夜資訊全部 ✗。「跌了買」類 dev 有、2020 後消失；池內只有 H2 延續方向穩（val t 1.64），但和 H136 同因子。ADR 新資訊約一半在開盤反映、盤中延續小於成本（ADR 日 K 在 `data/v2/us/`，`fetch_adr.py`）。
  - 高波動池（60 日波動前 100 檔，現股，第六十三批 `vpool_lab.py`）：池本身 dev 每年輸全池 6.7%；擇時「趨勢裡買回檔」dev 有、val 消失（同 B 池）；池內輪動「營收＋相對低波動」dev、val 都成立，但和 H136 同因子（相關 0.83～0.85），不能合併。
  - 跨資產（第六十四批 `xasset_lab.py`，美國 ETF 台幣代理）：美股和 H136 的相關 dev 只有 0.19（val 0.35），70：30 組合 dev 過、val 輸 H136（1.81 vs 1.87）；美債在 H136 最差月份平均 +1.8%，但 2015–19 報酬太低。
  - 台指期日內（第六十五批 `intraday_lab.py`，1 分 K 在 `data/v2/futures/tx1m`）：開盤區間突破有毛邊際（每筆約 4 點），但微台來回成本約 5.4 點、小台約 3.4 點，扣完 ≤ 0。
  - 全池特徵組合大搜尋（第六十六批 `combo_search.py`，163 特徵的單一／兩兩／三三 × N 10／20，共 144 萬組，目標「自己夠好」）：val 12／20 過，但全部含強勢或營收成分、和 H136 相關 0.58～0.88，val 超額 t 都輸 H136；加價值面、和 H136 較獨立的組合 val 大多失敗（6 個只過 1 個）。holdout（使用者同意，測 3 個）2／3 過：H2＋H99＋H10／N10（t 2.91，和 H136 相關 0.87）、H97＋H10＋H81／N20（t 2.10，相關 0.48）；三段的 Sharpe、t 都沒有贏 H136。
  - 存貨（第六十七批 `inventory_lab.py`；存貨 2012-Q4 起才有，dev 只有 2014-06～2019）：週轉天數水準／年變化、相對同業、產業庫存拐點單獨用都沒有東西；產業輪動、全市場與半導體庫存擇時全部 ✗。組合裡 dev 有 10 個靠存貨拉高 t，val 只有 3 個延續（都是個股週轉天數下降＋營收新高或強勢），7 個反轉；這 3 個 holdout 全部 ✗、都輸大盤。
  - 池 × 事件模型（第六十八批 `pool_event_lab.py`，10 格資金、訊號觸發進出）：週報酬自我相關定義的「趨勢／回歸性格」幾乎不延續（等級相關 +0.06），搭配預測兩個方向都相反；逆勢事件（RSI、布林下軌）扣成本後全部大幅為負；順勢事件最好的是 250 日新高持有 60 日，Sharpe 0.90、MDD −56%，dev 0／252。
  - 第六十九批（`pool_event2_lab.py`）：5 個價格性格指標（週／月自我相關、變異數比、趨勢效率、產業自我相關）延續性最高 0.076，不能拿來分池；順勢事件拉長持有（120 日、吊燈停利、跌破 60 日線）或改用營收排序都更差，dev 0／48。
  - 事件化 H136（第七十批 `h136_event_lab.py`，10 格）：「名次進前 N 買、掉出前 M 賣」每月檢查的版本 dev、val 都自己夠好，但沒有贏月換股 H136（dev 月差 t 1.76）；**每日檢查**名次洗進洗出、每年 ~290 筆，全面變差；60 日線停損也變差。乾淨引擎下緩衝每年只多 0.5～0.8%（H97 換得快）。⚠ 10 格各自複利幾年後會集中在少數幾格，格子模擬的報酬差要用 FS2 引擎核對。
  - 股價淨值比 < 1 池的逆勢（第七十一批 `pb_lab.py`）：池本身沒有溢酬（對大盤 −0.8%／年）；短線逆勢事件、每月反轉輪動 dev 0／78，平均比全流動池更差；逆勢每筆毛報酬約 +0.1%，遠小於成本。低 PBR 在排序、疊加強勢、逆勢三種用法都失敗。
  - 自身淨值比便宜線（第七十二批 `pbband_lab.py`，過去 5 年自己的 10／20 分位）：跌破後買進、回到中位數或 80 分位賣，dev 0／24；淨值比很少在一年內回到中位數（大多被 250 日上限賣掉），報酬輸大盤。「每檔適合的線不同」：最佳線延續 t 2.07，但打亂核對後只剩 +0.55%（p 0.26），逐股學習版 dev 0／6。⚠ 用 `pool_event_lab.simulate` 時，自訂出場代號別用 X1、X2、X5、Y3（會被當成固定持有期）和 X4、X6（會觸發吊燈停利）。
  - ⚠ FinMind 還原價**沒有還原減資**：停牌約兩週後復牌的減資股，價格會像一天大漲 50%～130%。高波動、小型股研究要先修正（`vpool_lab.gap_fix`：停牌 ≥ 5 日後復牌的跨停牌報酬設為 0）。
  - Kullamägi 領漲股整理突破（第七十三批 `kq_lab.py`，美股動能交易法）：台股合乎型態的機會每年只有約 12 筆、持股率 1～3%，dev 盤中突破版 alpha t 1.68、收盤確認版 0.62 ✗；勝率 33%、少數大贏家的形狀和文章一樣，但邊際全在突破當天盤中，隔天開盤買就沒有。提前進場（第七十四批 `kq2_lab.py`：整理中買、NR7／內包收斂日買）dev 0／4：買到的股票 53～71% 之後會突破，但 33～63% 在突破前就被短線出場規則洗掉。
  - 足跡圖「賣壓被吸收」反轉（第七十五批 `absorb_lab.py`，台指期 1 分 K，Delta 用 tick rule 推估，`data/v2/futures/tx1md`）：大額主動賣打到當天新低、3 根內沒跌破又出現大額主動買 → 反向進場，dev 9 年 155 筆、每筆毛 −0.4 點、0/9 年正 ✗；確認時已反彈約 13.5 點，之後和隨機一樣；只看價量的同型態也沒有。「真跌破」和「被吸收」之後的走勢都接近 0。
  - ICT 聰明錢（第七十六批 `ict_lab.py`，台指期 1 分 K）：掃前日日盤高低 → 結構轉變 → FVG 回測限價、停損掃蕩極值、2R，dev 547 筆、每筆毛 +0.1 點、淨 t −4.33、2/9 年正 ✗；6 個穩健性全負。單純「掃了就收回」毛 +1.7 點反而最好，加 MSS、FVG 越加越差（FVG 回測成交有逆選擇）。台指期上「掃高低點後反轉」的各種確認方式（H127、H188、H189）毛報酬都在 ±2 點內。
  - 訂單流「泡泡」當過濾（第七十七批 `flow_lab.py`，∞FLOW 的用法）：在 ICT、掃前日低收回上加「Delta 同向／沒有反向大 Delta／掃蕩後累積 Delta」，dev 0／6 ✗；用 K 棒實體做同樣過濾效果相同或更好（每分鐘 Delta 和 K 棒漲跌相關 0.73），過濾後扣成本全負。
  - 台指期每日 Delta 擇時（第七十八批 `dflow_lab.py`，持有拉長到日線）：交易規則 dev 0／3 ✗（Sharpe 最高 0.56 vs 小台做多 0.89）。但資訊回歸 N＝5 過：控制價格後，5 日主動買賣偏多、價格卻沒漲 → 之後 5 日偏漲（每 1σ +19bp，t 2.30）；只看 Delta 本身 ≈ 0。是唯一一個 Delta 在價格以外有資訊的結果，幅度只夠微調部位、不夠擇時（「一直做多＋依殘差 Delta 加減碼」是事後想法，未登記）。
- 研究方向想法不限：順勢、逆勢、量價、波動率、型態、事件、統計套利、季節性、基本面皆可，但一律走上述流程。

## TSMC / 期貨研究線（使用者主導，未經 v2 框架驗證）
- 台積電專屬、非全市場：外資棄守退場（`strategies/tsmc_foreign_exit.md`）、v4 三 sleeve 現金流（`tsmc_dual_pool_strategy.py`）、網格收租（`research/tsmc_grid_income.py`、`strategies/tsmc_grid_income.md`）、期貨 30 分 K 網格（`research/tsmc_futures_grid_30min.py`，QFF live 的研究依據）。
- 實驗存檔：`research/tsmc_grid_experiments/`；VIX 濾網、跳空停利等變體：`research/grid_*.py`、`research/txo_vix.py`（使用者未提交的 WIP，勿代為 commit）。

## 環境
- `.env.local`：`FINMIND_API_TOKEN`、`TELEGRAM_*`、`SJ_*`（永豐）。
- Python：股票/研究用 `/usr/local/bin/python3`（`python` 指令不存在）；macOS 多行程要用 `multiprocessing.get_context("fork")`。
- `pyproject.toml` + `uv.lock` 管 `.venv`（shioaji 需 Python 3.12）。
