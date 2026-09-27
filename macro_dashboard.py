#!/usr/bin/env python3
"""
每日總經預測市場儀表板

從 Polymarket（Gamma API，免金鑰）抓取對台股有間接影響的預測市場即時機率，
計算與前一交易日的機率變化（Δ 才是真正的訊號，非絕對水位），
再用 `claude -p`（headless）產生一段繁中台股總經解讀，
印終端 + 存檔 + 發早晨 Telegram（緊接在 scan.py 早晨摘要之後）。

⚠️ 定位：這是「參考儀表」，不是可回測的 alpha 訊號。
    不進 scan.py、不碰斷路器、不影響任何已驗證策略。

用法：
    python3 macro_dashboard.py              # 抓資料 + LLM 分析 + 存檔 + 發 Telegram
    python3 macro_dashboard.py --no-llm     # 跳過 claude -p（只出規則式儀表）
    python3 macro_dashboard.py --no-telegram# 不發 Telegram（只印終端 + 存檔）

排程建議（平日 08:00，排在 scan.py 之後）：
    0 8 * * 1-5 /usr/local/bin/python3 /Users/mu/fire-auto/macro_dashboard.py >> /Users/mu/fire-auto/data/macro_dashboard.log 2>&1
"""

import sys
import os
import json
import shutil
import subprocess
import urllib.request
import urllib.parse
import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent
SNAP_DIR = BASE_DIR / "data" / "macro_snapshots"
OUT_TXT = BASE_DIR / "data" / "macro_dashboard.txt"
OUT_JSON = BASE_DIR / "data" / "macro_dashboard.json"

GAMMA_SEARCH = "https://gamma-api.polymarket.com/public-search"
UA = {"User-Agent": "Mozilla/5.0 (fire-auto macro-dashboard)"}

# ── 追蹤清單 ────────────────────────────────────────────────
# 每筆：
#   key          — 穩定識別碼（用於跨日比對 Δ，不隨市場 slug 變動）
#   label        — 顯示名稱
#   query        — 丟給 Gamma public-search 的搜尋字串
#   must_contain — 市場題目（小寫）必須全部包含的關鍵字，用來精準鎖定合約
#   select       — "volume"（挑成交量最大）或 "nearest_end"（挑最近到期的活躍合約，會自動滾動）
#   tone_up      — 機率「上升」對台股的意涵：risk=風險/利空🔴, good=利多🟢, hawkish=偏空🟡
# 市場會隨時間結算/滾動，如某筆抓不到會顯示「—（無活躍合約）」，屆時調整 query/must_contain 即可。
TRACKED = [
    {"key": "fed_first_cut", "label": "Fed 下次會議前降息機率",
     "query": "fed rate cut meeting", "must_contain": ["fed rate cut by"],
     "select": "nearest_end", "tone_up": "good"},
    {"key": "fed_no_cut_2026", "label": "Fed 2026 全年不降息",
     "query": "fed rate cut 2026", "must_contain": ["no fed rate cuts"],
     "select": "volume", "tone_up": "hawkish"},
    {"key": "us_recession_2026", "label": "美國 2026 前衰退",
     "query": "recession 2026", "must_contain": ["us recession"],
     "select": "volume", "tone_up": "risk"},
    {"key": "taiwan_clash", "label": "台海軍事衝突（2027 前）",
     "query": "Taiwan military clash", "must_contain": ["taiwan military clash"],
     "select": "volume", "tone_up": "risk"},
    {"key": "us_china_tariff", "label": "美中關稅協議（年底前）",
     "query": "China tariff agreement", "must_contain": ["tariff agreement by december"],
     "select": "volume", "tone_up": "good"},
]

# Polymarket 幾乎沒有乾淨的「美國 CPI 月合約」（多為阿根廷/巴西/歐元區），
# 美國 CPI 是 Kalshi 的地盤。CPI 這類暫以下方註記呈現，待日後接 Kalshi API。
CPI_NOTE = "CPI/通膨：Polymarket 無適合美國 CPI 合約（Kalshi 為主，待接）。目前以「Fed 不降息機率」間接反映通膨黏著度。"

TONE_EMOJI = {"risk": "🔴", "good": "🟢", "hawkish": "🟡"}
BIG_MOVE_PP = 8.0  # |Δ| ≥ 此值（百分點）標注顯著異動


# ── 抓取 ────────────────────────────────────────────────────
def _http_json(url, timeout=15):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def search_live_markets(query):
    """搜尋並回傳仍活躍（未結算、0.1%<Yes<99.9%）的市場清單"""
    url = f"{GAMMA_SEARCH}?q={urllib.parse.quote(query)}&limit_per_type=10"
    try:
        data = _http_json(url)
    except Exception as e:
        print(f"  [Gamma] 搜尋失敗 {query!r}：{e}")
        return []
    markets = []
    for ev in data.get("events", []):
        ev_vol = float(ev.get("volume") or 0)
        for m in (ev.get("markets") or []):
            if m.get("closed") or not m.get("active"):
                continue
            try:
                prices = json.loads(m.get("outcomePrices") or "[]")
                yes = float(prices[0])
            except (ValueError, IndexError, TypeError):
                continue
            if yes <= 0.001 or yes >= 0.999:  # 已結算/退化
                continue
            markets.append({
                "question": m.get("question", ""),
                "yes": yes,
                "volume": float(m.get("volumeNum") or ev_vol),
                "end": m.get("endDate") or "",
            })
    return markets


def _parse_end(s):
    try:
        return datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return datetime.datetime.max.replace(tzinfo=datetime.timezone.utc)


def resolve_item(item, cache):
    """依 must_contain + select 從搜尋結果挑出目標合約"""
    q = item["query"]
    if q not in cache:
        cache[q] = search_live_markets(q)
    now = datetime.datetime.now(datetime.timezone.utc)
    cands = [
        m for m in cache[q]
        if all(kw in m["question"].lower() for kw in item["must_contain"])
    ]
    if not cands:
        return None
    if item["select"] == "nearest_end":
        future = [m for m in cands if _parse_end(m["end"]) > now]
        pool = future or cands
        return min(pool, key=lambda m: _parse_end(m["end"]))
    return max(cands, key=lambda m: m["volume"])  # 預設 volume


# ── 快照 / Δ ────────────────────────────────────────────────
def load_prev_snapshot(today_str):
    """讀取「今天之前」最近一份快照，回傳 (date_str, {key: yes})"""
    if not SNAP_DIR.exists():
        return None, {}
    files = sorted(p.stem for p in SNAP_DIR.glob("*.json"))
    prev = [f for f in files if f < today_str]
    if not prev:
        return None, {}
    try:
        return prev[-1], json.loads((SNAP_DIR / f"{prev[-1]}.json").read_text())
    except Exception:
        return None, {}


def save_snapshot(today_str, probs):
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    (SNAP_DIR / f"{today_str}.json").write_text(
        json.dumps(probs, ensure_ascii=False, indent=2))


# ── 報表 ────────────────────────────────────────────────────
def build_rows(prev_probs):
    """回傳 (rows, probs)；rows 供顯示，probs 供存快照"""
    rows, probs, cache = [], {}, {}
    for item in TRACKED:
        m = resolve_item(item, cache)
        if not m:
            rows.append({"item": item, "found": False})
            continue
        yes = m["yes"]
        probs[item["key"]] = yes
        prev = prev_probs.get(item["key"])
        delta = None if prev is None else (yes - prev) * 100
        rows.append({"item": item, "found": True, "yes": yes,
                     "delta": delta, "question": m["question"]})
    return rows, probs


def _row_line(row):
    it = row["item"]
    if not row["found"]:
        return f"  {it['label']}：—（無活躍合約，需調整 query）"
    yes = row["yes"]
    tone = TONE_EMOJI.get(it["tone_up"], "")
    d = row["delta"]
    if d is None:
        dstr = "（無前值）"
    else:
        arrow = "▲" if d > 0 else ("▼" if d < 0 else "＝")
        flag = " ⚠️異動" if abs(d) >= BIG_MOVE_PP else ""
        dstr = f"{arrow}{d:+.1f}pp{flag}"
    return f"  {tone} {it['label']}：{yes*100:.1f}%  {dstr}"


def build_report_text(rows, prev_date, llm_text):
    today = datetime.date.today().isoformat()
    lines = [f"🌐 總經預測市場儀表 {today}"]
    if prev_date:
        lines.append(f"（Δ 對比 {prev_date}｜資料：Polymarket）")
    else:
        lines.append("（首日無前值可比對｜資料：Polymarket）")
    lines.append("")
    for row in rows:
        lines.append(_row_line(row))
    lines.append("")
    lines.append(f"ℹ️ {CPI_NOTE}")
    if llm_text:
        lines.append("")
        lines.append("🧠 台股解讀（Claude）")
        lines.append(llm_text.strip())
    lines.append("")
    lines.append("※ 參考儀表，非可回測訊號；不影響 scan.py / 斷路器 / 已驗證策略。")
    return "\n".join(lines)


# ── LLM 分析（claude -p headless）────────────────────────────
def _claude_bin():
    return (os.environ.get("CLAUDE_BIN")
            or shutil.which("claude")
            or str(Path.home() / ".local" / "bin" / "claude"))


def llm_analysis(rows, timeout=150):
    """用 claude -p 產生繁中台股總經解讀；失敗回傳 None（不拋例外）"""
    facts = []
    for row in rows:
        if not row["found"]:
            continue
        it = row["item"]
        d = row["delta"]
        dstr = "" if d is None else f"（較前值 {d:+.1f}pp）"
        facts.append(f"- {it['label']}：{row['yes']*100:.1f}%{dstr}")
    if not facts:
        return None
    prompt = (
        "你是協助台股操作的總經分析助手。以下是今天 Polymarket 預測市場的即時機率"
        "（pp = 百分點，Δ 是與前一交易日的變化，變化比水位更重要）：\n\n"
        + "\n".join(facts)
        + "\n\n請用繁體中文寫 3-5 行精簡解讀，聚焦：這些機率變化對『台股』"
        "（大盤風險、外資動能、電子/半導體權值股、台海風險）短期的偏多/偏空意涵。"
        "只講重點與可操作的風險提示，不要客套、不要免責聲明、不要重複上面的數字清單。"
    )
    claude = _claude_bin()
    if not Path(claude).exists() and not shutil.which(claude):
        print(f"  [LLM] 找不到 claude 可執行檔：{claude}")
        return None
    try:
        res = subprocess.run(
            [claude, "-p", prompt],
            capture_output=True, text=True, timeout=timeout,
            cwd="/tmp",  # 中性目錄，避免載入本專案 context 拖慢
        )
    except subprocess.TimeoutExpired:
        print(f"  [LLM] claude -p 逾時（>{timeout}s），略過分析")
        return None
    except Exception as e:
        print(f"  [LLM] claude -p 失敗：{e}")
        return None
    out = (res.stdout or "").strip()
    if res.returncode != 0 or not out:
        print(f"  [LLM] claude -p 無輸出（rc={res.returncode}）：{(res.stderr or '').strip()[:200]}")
        return None
    return out


# ── 主流程 ──────────────────────────────────────────────────
def main():
    args = sys.argv[1:]
    no_llm = "--no-llm" in args
    no_tg = "--no-telegram" in args

    today = datetime.date.today().isoformat()
    prev_date, prev_probs = load_prev_snapshot(today)

    print("抓取 Polymarket 預測市場…")
    rows, probs = build_rows(prev_probs)
    save_snapshot(today, probs)

    llm_text = None if no_llm else (print("呼叫 claude -p 產生台股解讀…") or llm_analysis(rows))

    report = build_report_text(rows, prev_date, llm_text)
    print("\n" + report)

    OUT_TXT.write_text(report)
    OUT_JSON.write_text(json.dumps(
        {"date": today, "prev_date": prev_date, "probs": probs,
         "rows": [{"key": r["item"]["key"], "label": r["item"]["label"],
                   "found": r["found"], "yes": r.get("yes"),
                   "delta": r.get("delta")} for r in rows],
         "llm": llm_text},
        ensure_ascii=False, indent=2))

    if not no_tg:
        try:
            from notify import send
            if send(report):
                print("\n[Telegram] 已發送")
            else:
                print("\n[Telegram] 發送失敗")
        except Exception as e:
            print(f"\n[Telegram] 略過：{e}")


if __name__ == "__main__":
    main()
