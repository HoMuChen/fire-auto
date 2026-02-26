---
name: generate-strategy-chart
description: 產生股票策略回測視覺化 HTML 圖表。當使用者要求「畫圖」「chart」「視覺化」「進出場圖」或針對某檔股票/策略要看圖表時使用此 skill。
---

# 策略回測圖表產生器

為指定的股票+策略組合產出互動式 HTML 圖表，包含股價走勢、技術指標、進出場標記、資金成長曲線、交易明細表。

## 觸發時機

- 使用者要求某檔股票的策略圖表/chart/視覺化
- 使用者要求看某策略的進出場點位圖
- 使用者要求批次產生 watchlist 所有圖表
- 使用者要求更新現有圖表（策略或數據有變動時）

## 輸入

必要資訊（從使用者指令或 `strategies/*.json` 取得）：
- `stock_id` — 股票代號（如 `2362`）
- `stock_name` — 股票名稱（如 `藍天`）
- `strategy_name` — 策略名稱，必須與 `backtest.py` 中 `STRATEGIES` dict 的 key 完全一致

## 產出流程

### Step 1: 執行回測模擬

使用 `backtest.py` 的回測引擎，收集逐日資料：

```python
import sys
sys.path.insert(0, '.')
from backtest import *

prices = read_prices(stock_id)
cfg = STRATEGIES[strategy_name]
signals = cfg["fn"](prices)
```

模擬邏輯完全複製 `backtest.py` 的 `simulate()` 函數，但額外收集：
- 每筆交易的 `buy_date`, `buy_price`, `sell_date`, `sell_price`, `ret_pct`, `reason`
- 逐日 `equity_curve`（持倉時 = 現金 + 持股市值，空倉時 = 現金）

關鍵：必須正確處理所有出場規則（signal / hold_days / profit_or_hold / gap_fill）和停損類型（fixed_pct / atr / trailing_pct），以及 RSI 階梯的分批加碼（`("buy", fraction)` tuple 訊號）。

### Step 2: 計算技術指標疊加線

根據策略類型選擇適當的 overlay：

| 策略 | 疊加指標 |
|------|----------|
| ATR通道回歸 | SMA(20) 橙線 + 下軌 SMA−1.5×ATR 紫虛線 |
| RSI+Bollinger, 布林觸底→中線 | 布林中線(橙) + 上軌(紅虛) + 下軌(綠虛) |
| 其他所有策略 | SMA(20) 橙線 |

使用 `backtest.py` 的 `calc_sma()`, `calc_atr()`, `calc_bollinger()` 函數計算。

如果有新策略需要新的 overlay，在 `generate_charts.py` 的 `get_overlays()` 函數中加入新分支。

### Step 3: 組裝 HTML

使用 `generate_charts.py` 中的 `HTML_TEMPLATE`，替換以下佔位符：

| 佔位符 | 內容 |
|--------|------|
| `{{TITLE}}` | `{stock_name}({stock_id}) — {strategy_name}` |
| `{{SUBTITLE}}` | 策略描述 + 停損描述 + 日期範圍 |
| `{{NAV_LINKS}}` | 所有 watchlist 股票的導航連結 |
| `{{LEGEND_ITEMS}}` | 技術指標圖例 |
| `{{DATA_JSON}}` | 完整資料 JSON（dates, closes, overlays, equity, trades） |

### Step 4: 輸出

檔名格式：`strategies/{stock_id}_{stock_name}_chart.html`

## HTML 結構

圖表使用 [Lightweight Charts v4.1.0](https://cdn.jsdelivr.net/npm/lightweight-charts@4.1.0) (TradingView)，暗色主題。

頁面由上到下：
1. **導航列** — 所有 watchlist 股票的快速切換連結
2. **標題** — 股票名稱 + 策略名稱 + 策略描述
3. **統計列** — 總報酬、最終資金、交易次數、勝率、勝/負/停損、平均報酬
4. **股價圖** (420px) — 收盤價藍線 + 技術指標 overlay + 買入(綠箭頭)/賣出(紅箭頭)/停損(黃箭頭) markers
5. **資金曲線** (260px) — 綠色面積圖 + 100 萬基準線
6. **交易明細表** — #、買入日期/價、賣出日期/價、持有天數、報酬%、出場原因、累計資金

配色：
- 背景: `#0a0e17`, 圖表區: `#0f1320`
- 收盤價: `#3b82f6` (藍)
- 買入: `#22c55e` (綠), 賣出: `#ef4444` (紅), 停損: `#f59e0b` (黃)
- 資金曲線: `#22c55e` (綠漸層)

## 執行方式

### 單檔
```bash
python3 -c "
from generate_charts import generate_chart
path, n = generate_chart('2362', '藍天', 'ATR通道回歸')
print(f'{path}: {n} trades')
"
```

### 批次（全部 watchlist）
```bash
python3 generate_charts.py
```

### 更新 watchlist 清單

如果 watchlist 成員有變動，需同步更新 `generate_charts.py` 頂部的 `WATCHLIST` 列表。

## 注意事項

- 策略名稱必須與 `backtest.py` 的 `STRATEGIES` dict key 完全一致（含括號、斜線等特殊字元）
- 股價資料來自 `data/stock_prices/{stock_id}.csv`，透過 `read_prices()` 讀取
- 所有模擬邏輯（停損、出場規則、交易成本）與 `backtest.py` 完全一致
- HTML 為 self-contained 單檔（資料內嵌 JSON），唯一外部依賴是 CDN 的 lightweight-charts JS
- 產出後用 `open {path}` 在瀏覽器中預覽
