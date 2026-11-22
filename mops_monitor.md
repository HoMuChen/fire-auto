# MOPS 重大訊息監控

輪詢公開資訊觀測站的即時重大訊息，存進 SQLite，隔天再和官方日期清單對帳。

**狀態（2026-11-22）：試跑中。** 只收資料，不發通知、不上 R2。2026-11-23（週一）是第一個完整交易日，2026-11-24 00:20 後看對帳結果。

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
- 平日尖峰的完整度和官方頻率限制還沒驗證。

## 對帳通過後

- 每日匯出按月分區的 Parquet 到 R2（`v1/mops_news/year=YYYY/month=MM/`），加入 `catalog.duckdb` 的 `mops_news`，接在 `lake.py daily` 後，重寫當月和上月。
- 不直接同步 `.sqlite`，避免 WAL 造成檔案不一致。
- Telegram 通知直接從 ai-dev 的 SQLite 發，是否要做由使用者決定。
