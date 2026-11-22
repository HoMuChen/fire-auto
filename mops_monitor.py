#!/usr/bin/env python3
"""公開資訊觀測站重大訊息監控（只用標準函式庫；試跑期不發通知、不上 R2）。

  python3 mops_monitor.py poll                     抓即時清單，新公告入庫並補全文（排程每 5 分鐘）
  python3 mops_monitor.py reconcile [--date YYYYMMDD]  和官方「依日期查詢」對帳，漏抓的補進資料庫
  python3 mops_monitor.py status [--date YYYYMMDD]     看輪詢與對帳結果

資料：data/mops/mops.sqlite（可用環境變數 MOPS_DB 改路徑）。
官方日期清單的 D 日包含 D-1 日 17:30 之後的公告，對帳以這個口徑為準。
announcements.source：poll＝輪詢即時抓到；reconcile＝輪詢漏掉、對帳才補到。
"""
import argparse
import fcntl
import hashlib
import html
import json
import os
import re
import sqlite3
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE = "https://mopsov.twse.com.tw/mops/web/"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/126 Safari/537.36"
DB_PATH = Path(os.environ.get("MOPS_DB", Path(__file__).parent / "data" / "mops" / "mops.sqlite"))
TW = timezone(timedelta(hours=8))
DELAY = 1.5
MAX_DETAILS_PER_RUN = 40
MAX_DETAIL_TRIES = 5

SCHEMA = """
CREATE TABLE IF NOT EXISTS announcements(
  key TEXT PRIMARY KEY, co_id TEXT, name TEXT, spoke_date TEXT, spoke_time TEXT, subject TEXT,
  seq_no TEXT, raw_time TEXT, detail TEXT, detail_tries INTEGER DEFAULT 0,
  source TEXT, first_seen TEXT);
CREATE INDEX IF NOT EXISTS idx_ann_date ON announcements(spoke_date);
CREATE INDEX IF NOT EXISTS idx_ann_co ON announcements(co_id, spoke_date);
CREATE TABLE IF NOT EXISTS polls(ts TEXT, ok INTEGER, rows INTEGER, new INTEGER, error TEXT, ms INTEGER);
CREATE TABLE IF NOT EXISTS reconciles(
  ts TEXT, date TEXT, official INTEGER, matched INTEGER, missing INTEGER, missing_rows TEXT);
"""


def now():
    return datetime.now(TW)


def post(page, data, referer):
    req = urllib.request.Request(
        BASE + page,
        urllib.parse.urlencode(data).encode(),
        {"User-Agent": UA, "Referer": BASE + referer, "Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(req, timeout=45) as r:
        raw = r.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("cp950", "replace")
    if re.search(r"過於頻繁|Overrun|安全性考量|CANNOT BE ACCESSED", text, re.I):
        raise RuntimeError("MOPS 擋下請求")
    return text


def clean(s):
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", "", s)
    s = html.unescape(s).replace("\xa0", " ")
    return re.sub(r"[ \t\r]+", " ", s).strip()


def roc_to_ad(s):
    y, m, d = s.split("/")
    return f"{int(y) + 1911}{int(m):02d}{int(d):02d}"


def make_key(co_id, date, hhmmss, subject):
    norm = re.sub(r"\s+", "", subject)
    return f"{co_id}|{date}|{hhmmss}|{hashlib.sha1(norm.encode()).hexdigest()[:10]}"


def parse_rows(page):
    """解析即時清單（代號在前）與日期清單（日期在前）的列。"""
    out = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", page, re.S | re.I):
        if "詳細資料" not in tr:
            continue
        cells = [clean(c) for c in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S | re.I)]
        if len(cells) < 5:
            continue
        if re.fullmatch(r"\d{2,3}/\d{2}/\d{2}", cells[0]):
            date, tm, co_id, name, subject = cells[:5]
        elif re.fullmatch(r"\d{2,3}/\d{2}/\d{2}", cells[2]):
            co_id, name, date, tm, subject = cells[:5]
        else:
            continue
        subject = re.sub(r"\s*\n\s*", "", subject)
        hhmmss = tm.replace(":", "").zfill(6)
        ad = roc_to_ad(date)
        keys = {k.lower(): v for k, v in re.findall(r"\.(\w+)\.value=['\"]([^'\"]*)", tr)}
        out.append({
            "key": make_key(co_id, ad, hhmmss, subject), "co_id": co_id, "name": name,
            "spoke_date": ad, "spoke_time": hhmmss, "subject": subject,
            "seq_no": keys.get("seq_no"), "raw_time": keys.get("spoke_time"),
        })
    return out


def check_list(page):
    if "公司代號" not in page and "查無" not in page and "無資料" not in page:
        raise RuntimeError("回傳內容不是重訊清單")


def fetch_realtime():
    page = post("ajax_t05sr01_1", {"step": "0", "TYPEK": "all"}, "t05sr01_1")
    check_list(page)
    return parse_rows(page)


def fetch_by_date(date):
    y, m, d = int(date[:4]) - 1911, date[4:6], date[6:]
    page = post("ajax_t05st02", {
        "encodeURIComponent": "1", "step": "1", "step00": "0", "firstin": "1", "off": "1",
        "TYPEK": "all", "year": str(y), "month": m, "day": d,
    }, "t05st02")
    check_list(page)
    return parse_rows(page)


def lookup_seq(row):
    """日期清單沒有 seq_no，從單一公司清單用時間與主旨對回來。"""
    y = int(row["spoke_date"][:4]) - 1911
    page = post("ajax_t05st01", {
        "encodeURIComponent": "1", "step": "1", "firstin": "1", "off": "1", "queryName": "co_id",
        "inpuType": "co_id", "TYPEK": "all", "co_id": row["co_id"], "year": str(y),
        "month": str(int(row["spoke_date"][4:6])), "b_date": "", "e_date": "",
    }, "t05st01")
    for r in parse_rows(page):
        if r["key"] == row["key"]:
            return r["seq_no"], r["raw_time"]
    return None, None


def fetch_detail(row):
    seq, raw_time = row["seq_no"], row["raw_time"]
    if not seq:
        seq, raw_time = lookup_seq(row)
        time.sleep(DELAY)
        if not seq:
            return None, None, None
    page = post("ajax_t05st01", {
        "step": "2", "off": "1", "firstin": "1", "seq_no": seq, "spoke_time": raw_time,
        "spoke_date": row["spoke_date"], "co_id": row["co_id"], "TYPEK": "all",
    }, "t05st01")
    # 頁面第一個 <pre> 是主旨，說明全文是最長的那個
    blocks = [clean(x) for x in re.findall(r"<pre[^>]*>(.*?)</pre>", page, re.S | re.I)]
    return (max(blocks, key=len) if len(blocks) >= 2 else None), seq, raw_time


def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    return con


def insert_new(con, rows, source):
    ts = now().isoformat(timespec="seconds")
    new = 0
    for r in rows:
        cur = con.execute(
            "INSERT OR IGNORE INTO announcements"
            "(key,co_id,name,spoke_date,spoke_time,subject,seq_no,raw_time,source,first_seen)"
            " VALUES(?,?,?,?,?,?,?,?,?,?)",
            (r["key"], r["co_id"], r["name"], r["spoke_date"], r["spoke_time"], r["subject"],
             r["seq_no"], r["raw_time"], source, ts),
        )
        new += cur.rowcount
    con.commit()
    return new


def fill_details(con):
    todo = con.execute(
        "SELECT * FROM announcements WHERE detail IS NULL AND detail_tries < ?"
        " ORDER BY first_seen DESC LIMIT ?", (MAX_DETAIL_TRIES, MAX_DETAILS_PER_RUN),
    ).fetchall()
    done = 0
    for row in todo:
        time.sleep(DELAY)
        try:
            detail, seq, raw_time = fetch_detail(dict(row))
        except Exception as e:
            print(f"detail 失敗 {row['key']}: {e}", file=sys.stderr)
            detail, seq, raw_time = None, None, None
        con.execute(
            "UPDATE announcements SET detail=?, detail_tries=detail_tries+1,"
            " seq_no=COALESCE(?,seq_no), raw_time=COALESCE(?,raw_time) WHERE key=?",
            (detail, seq, raw_time, row["key"]),
        )
        con.commit()
        done += detail is not None
    return done, len(todo)


def cmd_poll(con):
    t0 = time.time()
    ts = now().isoformat(timespec="seconds")
    try:
        rows = fetch_realtime()
        new = insert_new(con, rows, "poll")
        con.execute("INSERT INTO polls VALUES(?,?,?,?,?,?)", (ts, 1, len(rows), new, None, int((time.time() - t0) * 1000)))
        con.commit()
    except Exception as e:
        con.execute("INSERT INTO polls VALUES(?,?,?,?,?,?)", (ts, 0, 0, 0, str(e)[:300], int((time.time() - t0) * 1000)))
        con.commit()
        print(f"{ts} poll 失敗: {e}")
        return 1
    done, tried = fill_details(con)
    print(f"{ts} poll ok rows={len(rows)} new={new} detail={done}/{tried}")
    return 0


def cmd_reconcile(con, date):
    official = fetch_by_date(date)
    keys = {r["key"] for r in official}
    have = {x[0] for x in con.execute(
        f"SELECT key FROM announcements WHERE source='poll' AND key IN ({','.join('?' * len(keys))})", tuple(keys)
    )} if keys else set()
    missing = [r for r in official if r["key"] not in have]
    insert_new(con, missing, "reconcile")
    con.execute("INSERT INTO reconciles VALUES(?,?,?,?,?,?)", (
        now().isoformat(timespec="seconds"), date, len(keys), len(have), len(keys - have),
        json.dumps([{k: r[k] for k in ("co_id", "name", "spoke_date", "spoke_time", "subject")} for r in missing], ensure_ascii=False),
    ))
    con.commit()
    done, tried = fill_details(con)
    print(f"{date} 官方 {len(keys)} 筆，輪詢抓到 {len(have)}，漏 {len(keys - have)}；本次補全文 {done}/{tried}")
    for r in missing[:20]:
        print(f"  漏 {r['spoke_date']} {r['spoke_time']} {r['co_id']} {r['name']} {r['subject'][:40]}")
    return 0


def cmd_status(con, date):
    p = con.execute("SELECT count(*) n, sum(ok) ok, min(ts) a, max(ts) b FROM polls WHERE substr(replace(ts,'-',''),1,8)=?", (date,)).fetchone()
    print(f"{date} 輪詢 {p['n']} 次，成功 {p['ok'] or 0}，{p['a']} ~ {p['b']}")
    for e in con.execute("SELECT ts,error FROM polls WHERE ok=0 AND substr(replace(ts,'-',''),1,8)=? ORDER BY ts DESC LIMIT 5", (date,)):
        print("  失敗", e["ts"], e["error"])
    a = con.execute("SELECT count(*) n, sum(detail IS NOT NULL) d FROM announcements WHERE spoke_date=?", (date,)).fetchone()
    print(f"發言日 {date} 入庫 {a['n']} 筆，有全文 {a['d'] or 0}")
    for r in con.execute("SELECT * FROM reconciles WHERE date=? ORDER BY ts DESC LIMIT 3", (date,)):
        print(f"對帳 {r['ts']} 官方 {r['official']} 抓到 {r['matched']} 漏 {r['missing']}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="MOPS 重大訊息監控")
    ap.add_argument("cmd", choices=["poll", "reconcile", "status"])
    ap.add_argument("--date", default=None, help="西元 YYYYMMDD 或 yesterday，預設今天")
    a = ap.parse_args()
    if a.date == "yesterday":
        date = (now() - timedelta(days=1)).strftime("%Y%m%d")
    else:
        date = a.date or now().strftime("%Y%m%d")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if a.cmd != "status":
        lock = open(DB_PATH.with_suffix(".lock"), "w")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print("上一次執行尚未結束，略過")
            return 0
    con = connect()
    if a.cmd == "poll":
        return cmd_poll(con)
    if a.cmd == "reconcile":
        return cmd_reconcile(con, date)
    return cmd_status(con, date)


if __name__ == "__main__":
    sys.exit(main())
