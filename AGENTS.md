# Fire-Auto

台股交易研究與自動化：FinMind API 抓資料 → 本地快取 → 防前視回測研究 → 少數通過驗證的策略上實盤（Telegram 通知 / 永豐 Shioaji 期貨下單）。

> **新 session / 新 agent 先讀這段**
> 1. **現在實盤只有兩條線**：股票 **H29 雙新高月換股**（`h2_monthly.py`，2026-10-28 由 H2 換成 H29；只發通知、人工下單）、期貨 **QFF 小型台積電網格**（`futures_live.py`，**真實自動下單中**）。另有 SEF 小型智邦網格「提醒」與總經儀表。
> 2. **舊三策略（擠壓/超跌/AD）、方案 A/B、14 技術策略、季節性、籌碼、分點過濾全部已失效**，程式與文件在 `legacy/`（見 `legacy/README.md`）。使用者說「跑 scan」時要先提醒這是已停用的舊系統。
> 3. **新策略研究一律走 `research/v2/`**（防前視框架、事前登記、三段期間），規則見下方「研究規範」。**holdout（2024-）只有使用者明確同意才能開封。**
> 4. 不要動使用者的期貨 live 設定、cron 與帳本；不要 commit 使用者自己未提交的研究檔（`git status` 裡不是你改的檔）。

## 目錄地圖

| 位置 | 用途 |
|------|------|
| `h2_monthly.py` | ★ 實盤：H29 月換股選股 + Telegram（檔名沿用；`--h2` 算舊 H2 對照） |
| `futures_live.py` | ★ 實盤：QFF 網格真實下單執行器（9 道安全機制，見檔頭） |
| `futures_grid_monitor.py` | 期貨網格純提醒版（現用於 SEF；`--symbol QFF/SEF`） |
| `macro_dashboard.py` | 每日總經預測市場儀表（Polymarket + `claude -p` 解讀，參考用、非 alpha） |
| `notify.py` | Telegram：`send(text, md=False, intraday=False)`；`intraday=True` 才會多發盤中名單 |
| `stock_cache.py` / `chip_cache.py` / `broker_cache.py` | 每日資料快取（股價 / 法人+融資券 / 券商分點） |
| `backtest.py`、`validate_strategy_groups.py` | 舊函式庫（技術指標、成本常數）；TSMC 網格研究與 `research/v2/run_legacy14.py` 仍 import，**不要搬** |
| `tsmc_dual_pool_strategy.py`、`gen_tsmc_report.py` | TSMC v4 三 sleeve 現金流策略與報告（台積電專屬，未經 v2 驗證） |
| `research/v2/` | ★ 現行研究框架與所有假設紀錄（見 `research/v2/README.md`） |
| `research/build_data_v2.py` | 重建 `data/v2/`（含下市股、還原權息、2010 起） |
| `research/futures_cache*.py`、`research/tsmc_*`、`research/grid_*`、`research/txo_vix.py` | 期貨 / TSMC 網格研究與 30 分 K 快取（使用者主導的研究線） |
| `strategies/` | 仍有效的研究筆記：TSMC 網格收租、外資棄守退場、外資大賣→長紅（統計）、美國數據公布前（迷思） |
| `analysis/` | 個股分析快照 |
| `legacy/` | ⛔ 已停用的舊系統與舊研究（只供查閱） |
| `data/` | 快取與狀態（大多 gitignore，見下） |

## 實盤系統

### 股票：H29「雙新高」月換股（`h2_monthly.py`；2026-10-28 使用者決定由 H2 換成 H29，首次實盤前切換）
- 規則（**凍結，不可改**）：每月第一個 ≥12 日的交易日**開盤**換股；股票池＝上市/上櫃普通股、近 60 交易日均量 ≥200 張、≥250 根 K；分數＝H2（前一交易日還原收盤 / 近 252 交易日最高還原收盤）與 H10（近 3 月營收 / 12 個月內最高 3 月營收；m 月營收視為 m+1 月 11 日才可得）**各自在全上市櫃普通股的百分位，平均**；取股票池中最高 **20 檔**等權；無停損、無濾網；平手依代號小→大。
- 流程：cron 平日 20:30 跑，只有「下一平日是本月第 1~3 個 ≥12 日的平日（第 2~3 個＝失敗補發）且本月尚未換股」才計算（約 25~30 分鐘，~2,100 次 API：全市場還原價算百分位 + 18 次全市場月營收）並發 Telegram 到 **`TELEGRAM_CHAT_ID` 一個人**（使用者明確要求：不發盤中名單）。使用者收到後人工下單。
- 狀態：`data/h2_state.json`（檔名沿用；每月名單與 `strategy`；`capital` 設了訊息才會算張數）。首次實盤換股 2026-11-11 晚通知。
- 手動：`python3 h2_monthly.py --force --no-send`（立刻算、不存狀態、不發）；`--asof YYYY-MM-DD` 用指定日收盤重算；`--h2` 算舊 H2。
- 績效（top20，含成本，未扣一字漲停買不到 ≈ −1%/年）：dev +22.8% / val +66.9% / holdout 2024- +74.1% 年化，日線 MDD −28.6% / −28.5% / −31.7%（大盤等權 +4.3/+20.2/+19.7%）。每年皆勝 H2，但月差 t 1.33 不顯著、屬事後組合。
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

### 其他
- SEF 小型智邦網格：`futures_grid_monitor.py --symbol SEF` 純提醒，狀態 `data/futures_grid_state_SEF.json`，使用者手動 `buy/sell` 回報。SEF 期貨 2024-02 才上市、無空頭樣本，只用 1x。
- 總經儀表 `macro_dashboard.py`：每日 08:01 發 Telegram，參考用。

## 排程（crontab，平日）

| 時間 | 指令 | 性質 | Log |
|------|------|------|-----|
| 08:01 | `macro_dashboard.py` | 參考儀表 | `data/macro_dashboard.log` |
| 09:15~13:45 每 30 分 | `.venv/bin/python futures_live.py` | ★ QFF 真實下單 | `data/futures_live.log` |
| 09:15~13:45 每 30 分 | `.venv/bin/python futures_grid_monitor.py --symbol SEF` | SEF 提醒 | `data/futures_monitor_sef.log` |
| 18:00 | `stock_cache.py today` | 資料：全市場股價 | `data/stock_update.log` |
| 18:30 / 18:35 | `research/futures_cache.py` / `research/futures_cache_sef.py` | 資料：QFF / SEF 30 分 K | `data/futures_cache*.log` |
| 19:00 | `ChipCache().update_all(sleep=1.2)` | 資料收集（無策略使用） | `data/chip_update.log` |
| 20:30 | `h2_monthly.py` | ★ H29 換股通知 | `data/h2_monthly.log` |
| 22:00 | `broker_cache.py today`（先停、後重啟 backfill） | 資料收集（無策略使用） | `data/broker_daily.log` |

股票腳本用 `/usr/local/bin/python3`（系統 `/usr/bin/python3` 沒有 pandas）；期貨下單/提醒用 `.venv/bin/python`。2026-10-20 已移除 `scan.py` 08:00 與 `monitor.py` 盤中兩個 cron。

## 資料

- **FinMind API** 是唯一資料來源（token：`.env.local` 的 `FINMIND_API_TOKEN`；使用者方案 **6000 次/小時**）。Supabase 已停用。
- 打網路的指令在沙盒內會被擋 → 需 `dangerouslyDisableSandbox`。
- **`data/v2/`（研究專用，gitignore，`research/build_data_v2.py` 重建約 65 分鐘）**：`universe.csv`（2,594 檔普通股，含 58 檔不在 TaiwanStockInfo 的下市股，`delist_date` 標 2010 後下市）、`raw/{id}.parquet`（原始價：流動性、漲跌停）、`adj/{id}.parquet`（還原權息：報酬；下市股自建）、`revenue/ per/ inst/ margin/ fin/ bs/`（`research/v2/fetch_fundamentals.py`）、`taiex_tr.parquet`（加權報酬指數）。raw 有 close=0 停牌日需防護。
- **每日快取（gitignore，可重抓）**：`data/stock_prices/{id}.csv`（近 2 年未還原、無下市股，**不可用於研究**）、`data/institutional/`、`data/margin/`、`data/broker/{id}.parquet`（2021-06 起分點）、`data/futures/`（QFF/SEF 30 分 K、TXO 日資料與自建台指 VIX）。用法：`StockCache().get/update_today()`、`ChipCache().get_institutional/get_margin()`、`BrokerCache().get()`。
- `individual_stocks.json`：舊個股清單（快取模組用來決定抓哪些股票）；`data/low_liquidity_stocks.json`：舊流動性名單。
- **進版控的個人紀錄**：`data/trades.csv`（股票成交，舊三策略時期）、`data/futures_trades.csv`（台新帳戶手動期貨）。其餘 data 檔一律 gitignore。

## 研究規範（`research/v2/`，2026-10-20 起）

- **期間固定**：dev 2010-01-01~2019-12-31（開發）／val 2020-2023（規則凍結後才跑）／holdout 2024-（`unseal=True`，**需使用者同意、只看一次**）。
- **事前登記**：跑 dev 前先把假設、規則、過關標準寫進 `research/v2/HYPOTHESES.md`；dev 沒過的不准跑 val；每次執行自動記到 `research/v2/experiments.jsonl`（進版控）。
- **防前視**：`leak_check` / `leak_check_xs`（截斷資料重算訊號須一致）；pit 流動性股票池（60 日均量 ≥200 張、上市櫃、含下市股、不含興櫃、≥250 根 K）；**次日開盤成交**；一字漲跌停無法成交（2015-06-01 前 7%、後 10%）；下市以最後收盤出場。營收視為次月 11 日可得；財報 Q1→5/16、Q2→8/15、Q3→11/15、Q4→次年 4/1。
- **成本**：買 0.1425%、賣 0.1425% + 證交稅 0.3%（來回 0.585%）；期貨每腳單邊約 0.05~0.1%。
- **橫斷面判定**（主判定 top20 等權月換股）：超額（vs pit 等權大盤）t ≥ 2、dev ≥7/9 年正、分位大致單調；val 再過一次。
- **核心哲學**：參數用市場慣用值、不做參數搜索；改邏輯不調參；微調參數結果劇變就捨棄；必須全市場（整個 pit 池）有效，不挑個股；看過結果才提出的組合標「事後」並打折。
- **已知規律**（100 個假設的結論，詳見 `research/v2/README.md`）：
  - 台股橫斷面是「**延續**」而非「反轉」：1 月反轉、彩券效應、盤中反向、產業內反轉、配對交易全部失敗或反向。
  - 可交易的只有「**強勢延續（H2）**」與「**營收創新高（H10 系）**」；價值、殖利率、低波動、非流動性、籌碼、ROE/毛利、SUE、季節性、大盤擇時、絕對門檻、期貨做空避險都不過。
  - 事件型策略（營收突破）每筆超額為正但分布像彩券（中位數 < 0），壓到 10~20 檔後失效。
  - 營收類訊號（YoY、創新高、月營收驚喜 SUR）分位都很強，但單獨集中成 20 檔都不過；只有跟 H2 合成（H29）才過，再加 SUR 也沒增量。財報品質（F-score、應計、現金流殖利率）、集保籌碼、小型股、改良動能（殘差、產業、FIP）都不過。
  - H2 有 62% 的換股日超過 20 檔同分（創新高），凍結規則依代號小→大取；改用突破幅度排序（H95）val 反而更差。
  - 6~8 月「去年同期有除息」的股票整體有溢酬（毛 +0.72%/月，事後診斷），但挑高殖利率反而最差（H100 ✗）。
  - 短線（持有 3~10 天）在現股不可行：0.585% 來回成本吃掉邊際。
  - 小台指擇時（19 個）與台指選擇權賣方（月選 6 個、週選 4 個）都贏不了單純做多：MTX 連續做多含逆價差收益、Sharpe 高；選擇權溢價小、左尾重，價外買權零溢價。
- 研究方向想法不限：順勢、逆勢、量價、波動率、型態、事件、統計套利、季節性、基本面皆可，但一律走上述流程。

## TSMC / 期貨研究線（使用者主導，未經 v2 框架驗證）
- 台積電專屬、非全市場：外資棄守退場（`strategies/tsmc_foreign_exit.md`）、v4 三 sleeve 現金流（`tsmc_dual_pool_strategy.py`）、網格收租（`research/tsmc_grid_income.py`、`strategies/tsmc_grid_income.md`）、期貨 30 分 K 網格（`research/tsmc_futures_grid_30min.py`，QFF live 的研究依據）。
- 實驗存檔：`research/tsmc_grid_experiments/`；VIX 濾網、跳空停利等變體：`research/grid_*.py`、`research/txo_vix.py`（使用者未提交的 WIP，勿代為 commit）。

## 環境
- `.env.local`：`FINMIND_API_TOKEN`、`TELEGRAM_*`、`SJ_*`（永豐）。
- Python：股票/研究用 `/usr/local/bin/python3`（`python` 指令不存在）；macOS 多行程要用 `multiprocessing.get_context("fork")`。
- `pyproject.toml` + `uv.lock` 管 `.venv`（shioaji 需 Python 3.12）。
