# MOPS 重大訊息監控

輪詢公開資訊觀測站的即時重大訊息，存進 SQLite，隔天再和官方日期清單對帳。

**狀態（2026-11-24）：** 11/23 第一個完整交易日對帳通過（官方 192 筆、輪詢抓到 192、漏 0）。資料每天由 `lake.py daily` 發布到 R2 的 `mops_news`。2026-11-24 起，有影響的重訊會即時寄研究報告（`mops_alert.py`）。

## 做法

1. **輪詢**：每 5 分鐘抓一次即時清單。清單只有當天公告，所以要常駐，週末和盤後也跑。
2. **入庫**：用 `co_id|YYYYMMDD|HHMMSS|sha1(去空白主旨)[:10]` 去重，新公告記下 `first_seen`。
3. **補全文**：每次最多補 40 筆，每筆最多試 5 次，請求間隔 1.5 秒。
4. **對帳**：每天 00:20 後抓前一天的官方日期清單，漏抓的補進資料庫並標 `source=reconcile`。

官方日期清單的 D 日包含 D-1 日 17:30 之後的公告，對帳以這個口徑為準。監控啟動前的公告會顯示為漏抓，判讀 2026-11-23 時要扣除週日 17:30～20:14 的部分。2026-11-22 的對帳必有缺口，可忽略。

## MOPS 端點

都是 POST，要帶 User-Agent 和 Referer，回傳 HTML。基底網址是 `https://mopsov.twse.com.tw/mops/web/`。

| 用途 | 端點 | 參數 |
|------|------|------|
| 即時清單 | `ajax_t05sr01_1` | `step=0&TYPEK=all` |
| 依日期（對帳基準） | `ajax_t05st02` | `encodeURIComponent=1&step=1&step00=0&firstin=1&off=1&TYPEK=all&year=115&month=11&day=20` |
| 單一公司 | `ajax_t05st01` | `encodeURIComponent=1&step=1&firstin=1&off=1&queryName=co_id&inpuType=co_id&TYPEK=all&co_id=4127&year=115&month=&b_date=&e_date=` |
| 全文 | `ajax_t05st01` | `step=2&off=1&firstin=1&seq_no=..&spoke_time=..&spoke_date=西元YYYYMMDD&co_id=..&TYPEK=all` |

全文頁第一個 `<pre>` 是主旨，最長的 `<pre>` 才是說明全文。

存託憑證（如 911608 明輝-DR）的全文走 `ajax_t59sb01`：`firstin=true&step=2b&SKEY=..&DATE1=西元YYYYMMDD&co_id=..&TYPEK=all&YEAR=115&MONTH=11`，頁面只有一個 `<pre>`。程式在 `raw_time` 存 `t59sb01` 標記這種公告。

## 部署（ai-dev）

放在 `/Users/mu/mops-monitor/`，和實盤 repo 分開，不動 crontab。`mops_monitor.py` 和 `mops_monitor_run.sh` 分別部署成該目錄的 `mops_monitor.py`、`run.sh`，資料庫在 `data/mops.sqlite`。

```bash
nohup ~/mops-monitor/run.sh >> ~/mops-monitor/monitor.log 2>&1 &   # 啟動
kill $(cat ~/mops-monitor/run.pid)                                  # 停止
tail ~/mops-monitor/monitor.log                                     # 看輪詢
MOPS_DB=~/mops-monitor/data/mops.sqlite /usr/local/bin/python3 \
  ~/mops-monitor/mops_monitor.py status --date 20261123             # 看對帳
```

launchd 在 ai-dev 上無法載入（`mu` 不是 console 使用者，bootstrap 回錯誤 125、5），所以改用背景程序。

## 已知限制

- ai-dev 重開機後不會自動恢復，要手動啟動。
- 程序掛掉沒有通知。
- MOPS 偶爾回 502 或逾時（一天 1～4 次），下一輪會補上，對帳也會再檢查。官方的頻率上限還沒測過。

## R2

`lake/lake.py` 的 `build_mops` 每次從 SQLite（預設 `~/mops-monitor/data/mops.sqlite`，可用 `MOPS_DB` 改）重建 `v1/mops_news/year=YYYY/month=MM/data.parquet`，view 是 `tw.main.mops_news`。內容沒變的月份 sha256 相同、不會重傳。沒有資料庫的機器會跳過。不直接同步 `.sqlite`，避免 WAL 造成檔案不一致。

`lake.py daily` 只在週二～六 00:30 跑，所以週末的公告要到週二凌晨才會上 R2。

## 重訊研究信（`mops_alert.py`）

`run.sh` 每輪輪詢後在背景跑 `mops_alert.py run`，有鎖，上一輪沒跑完就略過。只處理 `first_seen` 在近 3 小時內的公告，全文最多等 30 分鐘。

1. **規則篩：** 不是上市櫃普通股（興櫃、DR、公發）記 `skip_market`；例行公告（法說會、注意交易資訊、iXBRL、公司治理主管、債券發行條件、背書保證…，見 `ROUTINE`）記 `skip_rule`。都不呼叫模型。
2. **判斷：** 其餘的每 25 則一批交給 `claude -p`，回傳有沒有明顯影響、方向、理由。無影響記 `no`；呼叫失敗不記，下一輪重試。
3. **報告：** 有影響的從 `~/fire-auto/data/v2` 組資料，包括股價與估值、近 10 日、月營收、季財報、法人、融資，再加上 MOPS 近兩個月的重訊。這些交給 `claude -p` 寫分析（JSON），組成 HTML。報告附三張 matplotlib 圖：股價與成交量、月營收與年增率、法人每日買賣超，信裡用 cid 內嵌，因為 Gmail 不顯示 SVG 和 data URI；存檔版則改成 data URI。顏色照台股慣例紅漲綠跌。報告存到 `reports/`，再用 Resend 寄出，記 `yes`；失敗記 `fail`。一則約 80 秒。
4. **設定：** 放在 `~/mops-monitor/.env`（權限 600）：`RESEND_API_KEY`（只能寄信的 key）、`MOPS_MAIL_TO`（多個收件者用逗號、分號或空白分隔，例如 `MOPS_MAIL_TO=a@x.com,b@y.com`，寫在同一行；一封最多 50 人，所有人會看到彼此的地址）；寄件人預設 `alert@mail.homuchen.com`（`MOPS_MAIL_FROM`）。

股數用「淨值 × 淨值比 ÷ 收盤」回推，不用「股本 ÷ 10」，因為有些公司的面額不是 10 元，例如尚凡是 1 元。股價資料只到前一個交易日收盤。報告用來快速掌握消息，不是交易訊號。

```bash
python3 mops_alert.py status [--date YYYYMMDD]      # 當天各階段筆數與有影響的清單
python3 mops_alert.py report KEY [--send]           # 對單一公告重出報告（預設不寄）
tail ~/mops-monitor/alert.log
```
