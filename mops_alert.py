#!/usr/bin/env python3
"""重訊影響判斷 → 研究報告（HTML）→ Email（Resend）。接在 mops_monitor.py 輪詢後面跑。

  python3 mops_alert.py run [--no-send]       處理近 3 小時的新公告：規則篩 → claude 判斷 → 報告 → 寄信
  python3 mops_alert.py report KEY [--send]   對單一公告產生報告（測試用；預設不寄）
  python3 mops_alert.py status [--date YYYYMMDD]

流程：
  1. 規則篩：不是上市櫃普通股（興櫃、DR、公發）或例行公告（法說會、注意交易資訊…）直接略過，不呼叫模型。
  2. claude -p 一次判斷一批：會不會明顯影響股價、方向、理由。
  3. 判定有影響的：從 data/v2 組研究資料（股價、估值、月營收、季財報、法人、融資）＋近兩個月重訊，
     交給 claude -p 寫分析，組成 HTML，存到 reports/ 並寄出。
報告是整理消息用，不是交易訊號（事件型策略在 research/v2 沒有通過驗證）。

設定（同目錄 .env 或環境變數）：RESEND_API_KEY、MOPS_MAIL_TO（多個收件者用逗號、分號或空白分隔，Resend 一封最多 50 人）、MOPS_MAIL_FROM（預設 alert@mail.homuchen.com）；
沒有 key 或 --no-send → 只存 HTML。資料：FIRE_AUTO_ROOT（預設 ~/fire-auto）的 data/v2；資料庫同 mops_monitor.py（MOPS_DB）。
"""
import argparse
import fcntl
import html
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

import mops_monitor as mm

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("FIRE_AUTO_ROOT", Path.home() / "fire-auto"))
V2 = ROOT / "data" / "v2"
REPORTS = HERE / "reports"
WINDOW_H = 3            # 只處理 first_seen 在近 N 小時內的公告（部署時不會回頭處理舊公告）
WAIT_DETAIL_MIN = 30    # 全文還沒抓到時最多等多久，超過就只用主旨判斷
BATCH = 25
CLAUDE_TIMEOUT = (240, 420)   # 判斷、報告

SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts(
  key TEXT PRIMARY KEY, ts TEXT, stage TEXT, direction TEXT, reason TEXT,
  report TEXT, sent_at TEXT, error TEXT);
"""
# stage：skip_market（非上市櫃普通股）、skip_rule（例行公告）、no（模型判斷無明顯影響）、yes（有影響、已出報告）、fail（模型或報告失敗）

ROUTINE = re.compile(
    "法人說明會|法說會|受邀參加|投資展望會|產業論壇|業績發表會|注意交易資訊|公[布佈]注意|iXBRL|"
    "公司治理主管|代理發言人|補充公告發行|公司債.*(發行條件|掛牌)|背書保證|資金貸與|"
    "股東(常|臨時)會.*(召開|開會|議事)|除權息基準日|除息基準日|現金股利發放日|董事會召開日期")

TRIAGE_PROMPT = """你是台股事件分析助手。以下是公開資訊觀測站剛發布的重大訊息，請逐則判斷是否可能對該公司股價造成「明顯影響」。

會明顯影響的例子：財測或營收、獲利大幅變動，重大合約或訂單，併購、公開收購、股份轉換，處分或取得金額大的資產（相對公司規模），
現金增資、減資、私募、庫藏股，重大虧損或減損，訴訟、停工、事故、資安事件，更換會計師或會計師保留意見，董事長、總經理異動，
經營權變動，下市櫃或停止交易，財務危機、跳票，大型擴產或投資。
例行公告判為否：法說會、股東會程序、一般主管異動、代子公司例行公告、債券發行條件、金額不大的例行資產取得或處分。

只輸出 JSON 陣列，不要其他文字。每則一個物件：{"id": 編號, "impact": true 或 false, "direction": "利多" 或 "利空" 或 "不確定", "reason": "20 字內理由"}

"""

REPORT_PROMPT = """你是台股研究員。下面是一則剛發布的重大訊息，以及這家公司的現有資料（股價資料只到前一個交易日收盤）。
請判斷這則消息對股價的意涵，寫成給投資人快速閱讀的研究摘要。

要求：
- 只根據下面提供的資料，不要編造數字或背景；資料不足就直接說不足。
- 量化消息的規模：例如金額佔近四季營收、淨值或市值的比例，或對 EPS 的大約影響（每股影響＝金額 ÷ 股數）。
- 結合近期股價、營收趨勢、法人與融資動向，說明市場可能已反映多少。
- 公告時間在最後一個收盤日之後的，代表市場還沒反應；之前的，用近 10 日股價看反應。
- 不要給買賣建議，不要免責聲明，不要客套。繁體中文。

只輸出一個 JSON 物件，不要其他文字：
{"title": "15 字內標題", "direction": "利多|利空|中性|不確定", "magnitude": "大|中|小",
 "summary": "兩句內的結論", "points": ["分析重點 3～6 點，每點 1～3 句，引用數字"], "watch": ["接下來要觀察的事"]}

"""


# ─────────── 共用 ───────────
def load_env():
    env = {}
    for p in (HERE / ".env", ROOT / ".env.local"):
        if p.exists():
            for line in p.read_text().splitlines():
                m = re.match(r"\s*([A-Z_][A-Z0-9_]*)\s*=\s*['\"]?([^'\"\n]*?)['\"]?\s*$", line)
                if m and m.group(1) not in env:
                    env[m.group(1)] = m.group(2)
    env.update({k: v for k, v in os.environ.items() if k.startswith(("RESEND_", "MOPS_MAIL"))})
    return env


def connect():
    con = mm.connect()
    con.executescript(SCHEMA)
    return con


def claude(prompt, timeout):
    exe = os.environ.get("CLAUDE_BIN") or shutil.which("claude") or str(Path.home() / ".local" / "bin" / "claude")
    res = subprocess.run([exe, "-p", prompt], capture_output=True, text=True, timeout=timeout, cwd="/tmp")
    if res.returncode != 0 or not res.stdout.strip():
        raise RuntimeError(f"claude -p rc={res.returncode}: {(res.stderr or res.stdout).strip()[:200]}")
    return res.stdout.strip()


def parse_json(text, open_ch):
    close_ch = "]" if open_ch == "[" else "}"
    i, j = text.find(open_ch), text.rfind(close_ch)
    if i < 0 or j < i:
        raise ValueError(f"模型輸出不是 JSON：{text[:200]}")
    return json.loads(text[i:j + 1])


def universe():
    u = pd.read_csv(V2 / "universe.csv", dtype=str)
    u = u[u.type.isin(["twse", "tpex"]) & u.delist_date.isna()]
    return u.set_index("stock_id")


# ─────────── 研究資料 ───────────
def _read(kind, sid):
    p = V2 / kind / f"{sid}.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df.date)
    return df.sort_values("date").reset_index(drop=True) if "date" in df.columns else df


def pct(x, digits=1):
    return "—" if x is None or pd.isna(x) else f"{x * 100:+.{digits}f}%"


def num(x, digits=1):
    return "—" if x is None or pd.isna(x) else f"{x:,.{digits}f}"


def facts(sid, info):
    """回傳 (sections, text)：sections＝[(標題, 表頭, 列)] 給 HTML；text 給模型。"""
    sec, notes = [], []
    raw, adj = _read("raw", sid), _read("adj", sid)
    shares = None
    bs = _read("bs", sid)
    if len(bs):
        last = bs[bs.date == bs.date.max()].set_index("type").value
        cap, eq = last.get("CapitalStock"), last.get("EquityAttributableToOwnersOfParent", last.get("Equity"))
        shares = cap / 10 if cap else None          # 面額 10 元；面額不是 10 元的（如尚凡 1 元）下面用淨值比修正
    if len(raw) and len(adj):
        raw, adj = raw[raw.close > 0], adj[adj.close > 0].set_index("date").close
        c = raw.iloc[-1]
        per = _read("per", sid)
        p = per.iloc[-1] if len(per) else None
        if p is not None and eq and p.PBR > 0:
            shares = eq * p.PBR / c.close            # 每股淨值＝收盤 ÷ 淨值比 → 股數＝淨值 ÷ 每股淨值
        r = lambda n: adj.iloc[-1] / adj.iloc[-1 - n] - 1 if len(adj) > n else None   # noqa: E731
        hi = adj.tail(252).max()
        v20, v60 = raw.volume.tail(20).mean() / 1000, raw.volume.tail(60).mean() / 1000
        mcap = c.close * shares / 1e8 if shares else None
        rows = [("收盤（" + str(pd.Timestamp(c.date).date()) + "）", num(c.close, 2)),
                ("5／20／60 日報酬", f"{pct(r(5))}／{pct(r(20))}／{pct(r(60))}"),
                ("距 52 週高點", pct(adj.iloc[-1] / hi - 1)),
                ("20 日均量（張）", f"{num(v20, 0)}（60 日 {num(v60, 0)}）"),
                ("20 日均成交金額（億）", num(raw.money.tail(20).mean() / 1e8, 2)),
                ("市值（億）", num(mcap, 0))]
        if p is not None:
            hist = per[per.date >= per.date.max() - pd.Timedelta(days=365 * 3)].PER
            rank = (hist < p.PER).mean() if p.PER > 0 and len(hist) else None
            rows += [("本益比／淨值比／殖利率", f"{num(p.PER, 1)}／{num(p.PBR, 2)}／{num(p.dividend_yield, 2)}%"),
                     ("本益比在近 3 年的分位", "—" if rank is None else f"{rank * 100:.0f}%")]
        if shares and len(bs):
            rows.append(("股本（億）／股數（億股）／每股淨值", f"{num(cap / 1e8, 2)}／{num(shares / 1e8, 2)}／{num(eq / shares, 2)}"))
        sec.append(("股價與估值", None, rows))
        last10 = raw.tail(10).iloc[::-1]
        chg = raw.close.pct_change()
        sec.append(("近 10 日", ("日期", "收盤", "漲跌", "成交量（張）"),
                    [(str(d.date.date()), num(d.close, 2), pct(chg[i], 2), num(d.volume / 1000, 0)) for i, d in last10.iterrows()]))
    rev = _read("revenue", sid)
    if len(rev):
        rev = rev.assign(ym=rev.revenue_year * 100 + rev.revenue_month).drop_duplicates("ym", keep="last").set_index("ym").revenue
        out = []
        for ym in list(rev.index)[-6:][::-1]:
            yoy = rev[ym] / rev[ym - 100] - 1 if ym - 100 in rev.index and rev[ym - 100] else None
            prev = ym - 1 if ym % 100 != 1 else ym - 89
            mom = rev[ym] / rev[prev] - 1 if prev in rev.index and rev[prev] else None
            out.append((f"{ym // 100}/{ym % 100:02d}", num(rev[ym] / 1e8, 2), pct(yoy), pct(mom)))
        sec.append(("月營收（億）", ("月份", "營收", "年增", "月增"), out))
        last12 = rev.tail(12).sum() / 1e8
        notes.append(f"近 12 個月營收合計 {last12:,.1f} 億")
    fin = _read("fin", sid)
    if len(fin):
        f = fin[fin.type.isin(["EPS", "Revenue", "GrossProfit", "OperatingIncome", "IncomeAfterTaxes"])]
        f = f.pivot_table(index="date", columns="type", values="value", aggfunc="last").sort_index().tail(5)
        out = []
        for d, x in f.iloc[::-1].iterrows():
            q = f"{d.year}Q{(d.month - 1) // 3 + 1}"
            rv = x.get("Revenue")
            gm = x.get("GrossProfit") / rv if rv else None
            om = x.get("OperatingIncome") / rv if rv else None
            out.append((q, num(rv / 1e8 if rv else None, 2), "—" if gm is None else f"{gm * 100:.1f}%",
                        "—" if om is None else f"{om * 100:.1f}%", num(x.get("EPS"), 2)))
        sec.append(("季財報（單季）", ("季度", "營收（億）", "毛利率", "營益率", "EPS"), out))
    inst = _read("inst", sid)
    if len(inst):
        inst = inst.assign(net=(inst.buy - inst.sell) / 1000,
                           who=inst.name.map(lambda n: "外資" if n.startswith("Foreign") else "投信" if n.startswith("Investment") else "自營商"))
        days = sorted(inst.date.unique())
        out = []
        for who in ("外資", "投信", "自營商"):
            s = inst[inst.who == who].groupby("date").net.sum()
            out.append((who, num(s[s.index.isin(days[-5:])].sum(), 0), num(s[s.index.isin(days[-20:])].sum(), 0)))
        sec.append(("法人買賣超（張）", ("", "近 5 日", "近 20 日"), out))
    mg = _read("margin", sid)
    if len(mg) > 20:
        a, b = mg.iloc[-1], mg.iloc[-21]
        sec.append(("融資融券（張）", ("", "最新", "20 日前", "變化"), [
            ("融資餘額", num(a.MarginPurchaseTodayBalance, 0), num(b.MarginPurchaseTodayBalance, 0),
             pct(a.MarginPurchaseTodayBalance / b.MarginPurchaseTodayBalance - 1) if b.MarginPurchaseTodayBalance else "—"),
            ("融券餘額", num(a.ShortSaleTodayBalance, 0), num(b.ShortSaleTodayBalance, 0), "")]))
    head = f"{sid} {info['name']}（{'上市' if info['type'] == 'twse' else '上櫃'}，{info['industry']}）"
    lines = [head] + notes + ([f"流通股數約 {shares / 1e8:,.2f} 億股"] if shares else [])
    for title, cols, rows in sec:
        lines.append(f"\n【{title}】")
        if cols:
            lines.append(" | ".join(cols))
        lines += [" | ".join(str(x) for x in r) for r in rows]
    return sec, "\n".join(lines)


def history(sid, spoke_date, key):
    """近兩個月同公司的重訊主旨（MOPS 單一公司查詢）。"""
    d = datetime.strptime(spoke_date, "%Y%m%d")
    out = []
    for dt in (d, d.replace(day=1) - timedelta(days=1)):
        try:
            page = mm.post("ajax_t05st01", {
                "encodeURIComponent": "1", "step": "1", "firstin": "1", "off": "1", "queryName": "co_id",
                "inpuType": "co_id", "TYPEK": "all", "co_id": sid, "year": str(dt.year - 1911), "month": str(dt.month),
                "b_date": "", "e_date": ""}, "t05st01")
            out += [r for r in mm.parse_rows(page) if r["key"] != key]
        except Exception as e:
            print(f"  history {sid} {dt:%Y-%m} 失敗：{e}", file=sys.stderr)
        time.sleep(mm.DELAY)
    out.sort(key=lambda r: r["spoke_date"] + r["spoke_time"], reverse=True)
    return out[:15]


# ─────────── HTML ───────────
CSS_TD = "padding:4px 10px;border-bottom:1px solid #e5e7eb;font-size:14px;"
COLOR = {"利多": "#15803d", "利空": "#b91c1c"}


def table(cols, rows):
    h = "<table style='border-collapse:collapse;margin:4px 0 16px;'>"
    if cols:
        h += "<tr>" + "".join(f"<th style='{CSS_TD}text-align:left;color:#6b7280;font-weight:600;'>{html.escape(c)}</th>" for c in cols) + "</tr>"
    for r in rows:
        h += "<tr>" + "".join(f"<td style='{CSS_TD}{'text-align:right;' if i else ''}'>{html.escape(str(x))}</td>" for i, x in enumerate(r)) + "</tr>"
    return h + "</table>"


def render(ann, info, ana, sections, hist):
    e = html.escape
    color = COLOR.get(ana.get("direction"), "#374151")
    t = datetime.strptime(ann["spoke_date"] + ann["spoke_time"], "%Y%m%d%H%M%S")
    body = [f"<div style='font-family:-apple-system,\"PingFang TC\",\"Microsoft JhengHei\",sans-serif;max-width:720px;color:#111827;line-height:1.6;'>",
            f"<div style='font-size:13px;color:#6b7280;'>{e(ann['co_id'])} {e(info['name'])}・{e(info['industry'])}・{t:%Y-%m-%d %H:%M} 發布</div>",
            f"<h2 style='margin:4px 0 8px;font-size:20px;'>{e(ana.get('title') or ann['subject'][:30])}</h2>",
            f"<div style='margin-bottom:12px;'><span style='background:{color};color:#fff;border-radius:4px;padding:2px 8px;font-size:13px;'>"
            f"{e(ana.get('direction', '不確定'))}・影響{e(ana.get('magnitude', '—'))}</span></div>",
            f"<p style='font-size:15px;'><b>{e(ana.get('summary', ''))}</b></p>",
            f"<p style='font-size:14px;color:#374151;'>公告主旨：{e(ann['subject'])}</p>"]
    if ana.get("points"):
        body.append("<h3 style='font-size:16px;margin:16px 0 4px;'>分析</h3><ul style='padding-left:20px;font-size:14px;'>"
                    + "".join(f"<li>{e(p)}</li>" for p in ana["points"]) + "</ul>")
    if ana.get("watch"):
        body.append("<h3 style='font-size:16px;margin:16px 0 4px;'>接下來要觀察</h3><ul style='padding-left:20px;font-size:14px;'>"
                    + "".join(f"<li>{e(p)}</li>" for p in ana["watch"]) + "</ul>")
    body.append("<h3 style='font-size:16px;margin:20px 0 4px;'>資料</h3>")
    for title, cols, rows in sections:
        body.append(f"<div style='font-size:14px;font-weight:600;margin-top:8px;'>{e(title)}</div>" + table(cols, rows))
    if hist:
        body.append("<div style='font-size:14px;font-weight:600;margin-top:8px;'>近兩個月重訊</div>"
                    + table(("日期", "主旨"), [(f"{r['spoke_date'][4:6]}/{r['spoke_date'][6:]} {r['spoke_time'][:2]}:{r['spoke_time'][2:4]}", r["subject"][:60]) for r in hist]))
    body.append("<h3 style='font-size:16px;margin:20px 0 4px;'>公告全文</h3>"
                f"<pre style='white-space:pre-wrap;font-family:inherit;font-size:13px;background:#f9fafb;padding:12px;border-radius:6px;'>{e(ann['detail'] or '（全文未取得）')}</pre>")
    body.append("<p style='font-size:12px;color:#9ca3af;margin-top:24px;'>重訊監控自動產生：股價等資料只到前一個交易日收盤，分析由模型撰寫，"
                "用來快速掌握消息，不是交易訊號。</p></div>")
    return "\n".join(body)


# ─────────── 寄信 ───────────
def mail_to(env):
    """MOPS_MAIL_TO 可放多個收件者，用逗號、分號或空白分隔。"""
    return [x for x in re.split(r"[,;\s]+", env.get("MOPS_MAIL_TO", "")) if x]


def send_mail(env, subject, html_body):
    key, to = env.get("RESEND_API_KEY"), mail_to(env)
    if not key or not to:
        raise RuntimeError("沒有設定 RESEND_API_KEY 或 MOPS_MAIL_TO")
    body = {"from": f"重訊監控 <{env.get('MOPS_MAIL_FROM', 'alert@mail.homuchen.com')}>", "to": to,
            "subject": subject, "html": html_body}
    req = urllib.request.Request("https://api.resend.com/emails", json.dumps(body).encode(), {
        "Authorization": f"Bearer {key}", "Content-Type": "application/json", "User-Agent": "mops-alert/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read()).get("id")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Resend {e.code}: {e.read()[:200]!r}") from None


# ─────────── 流程 ───────────
def make_report(ann, info):
    sections, text = facts(ann["co_id"], info)
    hist = history(ann["co_id"], ann["spoke_date"], ann["key"])
    t = f"{ann['spoke_date']} {ann['spoke_time'][:2]}:{ann['spoke_time'][2:4]}"
    prompt = (REPORT_PROMPT + f"=== 重大訊息（{t}）===\n主旨：{ann['subject']}\n說明：\n{(ann['detail'] or '（全文未取得）')[:6000]}\n\n"
              f"=== 公司資料 ===\n{text}\n\n=== 近兩個月其他重訊 ===\n"
              + "\n".join(f"{r['spoke_date']} {r['subject'][:60]}" for r in hist))
    ana = parse_json(claude(prompt, CLAUDE_TIMEOUT[1]), "{")
    return ana, render(ann, info, ana, sections, hist)


def save(ann, page):
    REPORTS.mkdir(exist_ok=True)
    p = REPORTS / f"{ann['spoke_date']}_{ann['spoke_time']}_{ann['co_id']}.html"
    p.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(ann['co_id'])} 重訊</title></head><body>{page}</body></html>")
    return p


def record(con, key, stage, direction=None, reason=None, report=None, sent_at=None, error=None):
    con.execute("INSERT OR REPLACE INTO alerts VALUES(?,?,?,?,?,?,?,?)",
                (key, mm.now().isoformat(timespec="seconds"), stage, direction, reason, report, sent_at, error))
    con.commit()


def ready(con):
    now = mm.now()
    rows = con.execute(
        "SELECT a.* FROM announcements a LEFT JOIN alerts x ON x.key = a.key"
        " WHERE x.key IS NULL AND a.first_seen >= ? ORDER BY a.first_seen, a.spoke_time",
        ((now - timedelta(hours=WINDOW_H)).isoformat(timespec="seconds"),)).fetchall()
    wait = (now - timedelta(minutes=WAIT_DETAIL_MIN)).isoformat(timespec="seconds")
    return [dict(r) for r in rows if r["detail"] is not None or r["detail_tries"] >= mm.MAX_DETAIL_TRIES or r["first_seen"] < wait]


def triage(items):
    lines = [f"[{i}] {a['co_id']} {a['name']}｜{a['subject']}｜{(a['detail'] or '')[:800].replace(chr(10), ' ')}"
             for i, a in enumerate(items)]
    res = parse_json(claude(TRIAGE_PROMPT + "\n".join(lines), CLAUDE_TIMEOUT[0]), "[")
    return {int(r["id"]): r for r in res if "id" in r}


def cmd_run(con, send):
    env, uni = load_env(), universe()
    todo = ready(con)
    cand = []
    for a in todo:
        if a["co_id"] not in uni.index:
            record(con, a["key"], "skip_market")
        elif ROUTINE.search(a["subject"]):
            record(con, a["key"], "skip_rule")
        else:
            cand.append(a)
    hits = []
    for i in range(0, len(cand), BATCH):
        chunk = cand[i:i + BATCH]
        try:
            res = triage(chunk)
        except Exception as e:
            print(f"{mm.now():%F %T} 判斷失敗：{e}", file=sys.stderr)
            continue                                          # 不記錄 → 下一輪重試（仍受 WINDOW_H 限制）
        for j, a in enumerate(chunk):
            r = res.get(j, {})
            if r.get("impact"):
                hits.append((a, r))
            else:
                record(con, a["key"], "no", r.get("direction"), r.get("reason") or "（模型未回覆）")
    sent = 0
    for a, r in hits:
        info = uni.loc[a["co_id"]]
        try:
            ana, page = make_report(a, info)
            path = save(a, page)
            sent_at = None
            if send:
                subj = f"[重訊{ana.get('direction', r.get('direction', ''))}] {a['co_id']} {info['name']}：{ana.get('title') or a['subject'][:30]}"
                send_mail(env, subj, page)
                sent_at, sent = mm.now().isoformat(timespec="seconds"), sent + 1
            record(con, a["key"], "yes", ana.get("direction", r.get("direction")), r.get("reason"), str(path), sent_at)
        except Exception as e:
            print(f"{mm.now():%F %T} 報告失敗 {a['key']}：{e}", file=sys.stderr)
            record(con, a["key"], "fail", r.get("direction"), r.get("reason"), error=str(e)[:300])
    print(f"{mm.now().isoformat(timespec='seconds')} alert 待處理 {len(todo)}，送模型 {len(cand)}，有影響 {len(hits)}，寄出 {sent}", flush=True)


def cmd_report(con, key, send):
    a = con.execute("SELECT * FROM announcements WHERE key = ?", (key,)).fetchone()
    if not a:
        sys.exit(f"找不到 {key}")
    a = dict(a)
    info = universe().loc[a["co_id"]]
    ana, page = make_report(a, info)
    path = save(a, page)
    print(json.dumps(ana, ensure_ascii=False, indent=1))
    print(path)
    if send:
        print("寄出", send_mail(load_env(), f"[重訊{ana.get('direction', '')}] {a['co_id']} {info['name']}：{ana.get('title', '')}", page))


def cmd_status(con, date):
    rows = con.execute(
        "SELECT x.stage, x.direction, x.reason, x.sent_at, x.error, a.co_id, a.name, a.spoke_time, a.subject"
        " FROM alerts x JOIN announcements a ON a.key = x.key WHERE a.spoke_date = ? ORDER BY a.spoke_time", (date,)).fetchall()
    cnt = {}
    for r in rows:
        cnt[r["stage"]] = cnt.get(r["stage"], 0) + 1
    print(f"{date} " + "、".join(f"{k} {v}" for k, v in sorted(cnt.items())))
    for r in rows:
        if r["stage"] in ("yes", "fail"):
            print(f"  {r['stage']:4s} {r['spoke_time']} {r['co_id']} {r['name']} {r['direction'] or ''} {r['subject'][:40]}"
                  f"｜{r['reason'] or ''}{'｜已寄' if r['sent_at'] else ''}{'｜' + r['error'] if r['error'] else ''}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report", "status"])
    ap.add_argument("key", nargs="?")
    ap.add_argument("--no-send", action="store_true")
    ap.add_argument("--send", action="store_true")
    ap.add_argument("--date", default=mm.now().strftime("%Y%m%d"))
    a = ap.parse_args()
    con = connect()
    if a.cmd == "status":
        return cmd_status(con, a.date)
    lock = open(HERE / ".alert.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("上一輪 alert 還在跑，略過")
        return
    if a.cmd == "run":
        cmd_run(con, send=not a.no_send)
    else:
        cmd_report(con, a.key, a.send)


if __name__ == "__main__":
    main()
