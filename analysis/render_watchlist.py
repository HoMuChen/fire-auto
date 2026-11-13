"""驗證 analysis/watchlist.json 並重新生成唯讀檢視檔 analysis/WATCHLIST.md。

單一真實來源為 analysis/watchlist.json；請勿直接手改 WATCHLIST.md（發布時會覆蓋）。
用法：
  /usr/local/bin/python3 analysis/render_watchlist.py [--check]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

ANALYSIS_DIR = Path(__file__).resolve().parent
WATCHLIST_JSON = ANALYSIS_DIR / "watchlist.json"
WATCHLIST_MD = ANALYSIS_DIR / "WATCHLIST.md"

VALID_VALUATION_BASIS = {"normalized_pe", "forward_pe", "pb", "sotp"}
VALID_QUESTION_STATUS = {"open", "partial", "answered"}
VALID_CADENCE = {"once", "monthly", "quarterly", "ongoing"}
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

STATIC_INTRO_MD = """# 個股研究清單

**目的**：做過綜合分析的公司都在這裡，補上量化模型看不到的質化判斷（生意怎麼賺、能賺多久、哪裡會出錯）。清單**不分追蹤／不追蹤**，只分「做過研究」和「還沒做過」；每檔都寫出**持續關注的點**（質化問題、檢查點）和價位帶。股價跌到「合理」以下、而且論點沒破時，可以出手。

完整分析在同目錄的 `{代號}_{名稱}_綜合分析_{日期}.md`（單一真實來源為 `watchlist.json`，改完執行 `analysis/publish.py` 自動驗證並重繪本檔）。這份清單和實盤 H136（`h2_monthly.py`，機械式月換股）無關，H136 排名只當參考。

**用法**（使用者說「更新研究」才跑，不排程）：
- **每月增量更新**（月營收公布後，約每月 10 日）：
  - 逐檔看月營收，檢查到期的檢查點；
  - 逐條更新「質化問題」的目前判斷；
  - 對照現價落在哪個價位帶，結果寫進該檔的「紀錄」。
- **每季增量更新**（財報截止日後：5/16、8/15、11/15、隔年 4/1）：
  - 寫一份新日期的綜合分析報告（舊報告保留）；
  - 重算價位帶，論點逐條重新檢查，在紀錄中寫明改了什麼。
- **出手條件**（三項都要成立）：
  1. 現價在「合理」以下；
  2. 最近一次檢查論點沒破；
  3. 沒有未解的紅旗。

  進「合理」可以開始分批，進「便宜」再加。
- **結論可以是負面的**：例如「便宜是景氣高峰的假象」「衰退是結構性的」。這類股票一樣留在清單，「結論要改的條件」寫明什麼情況下才值得重新考慮。
- 新研究的股票：先做綜合分析，再寫論點、質化問題、價位帶、檢查點和結論要改的條件。

**價位帶怎麼算**：
- 用「正常化 EPS × 本益比區間」，再用淨值比交叉核對；每檔的假設都寫在下面。
- **景氣循環股不能用高峰 EPS**：高峰時本益比最低，看起來最便宜，其實最貴。
- 專案的量化研究顯示，「便宜」本身（低淨值比、跌到自己歷史低檔）在台股不是有效訊號（價值面假設、第七十一和七十二批都不過）。所以價位帶只有在論點成立時才有意義，**質化判斷才是這份清單的重點**。
- 價位帶是起始假設，可以依你的判斷調整。"""


def _is_num(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _check_date(val: object, label: str, errs: list[str]) -> None:
    if not isinstance(val, str) or not DATE_RE.match(val):
        errs.append(f"{label}: 日期格式必須為 YYYY-MM-DD（收到 {val!r}）")
        return
    try:
        dt.date.fromisoformat(val)
    except ValueError:
        errs.append(f"{label}: 無效日期 {val!r}")


def _check_reports(reports: object, prefix: str, analysis_dir: Path, errs: list[str]) -> None:
    if not isinstance(reports, list) or not reports:
        errs.append(f"{prefix}.reports: 必須為非空陣列")
        return
    for i, r in enumerate(reports):
        rp = f"{prefix}.reports[{i}]"
        if not isinstance(r, dict):
            errs.append(f"{rp}: 必須為物件")
            continue
        _check_date(r.get("date"), f"{rp}.date", errs)
        fn = r.get("filename")
        if not isinstance(fn, str) or not fn.endswith(".md") or "/" in fn or "\\" in fn:
            errs.append(f"{rp}.filename: 無效檔名 {fn!r}")
        elif not (analysis_dir / fn).is_file():
            errs.append(f"{rp}.filename: 找不到報告檔案 {analysis_dir / fn}")


def _validate_tracking_item(item: object, prefix: str, analysis_dir: Path, is_removed: bool, errs: list[str]) -> None:
    if not isinstance(item, dict):
        errs.append(f"{prefix}: 必須為物件")
        return
    req_str = (
        "stock_id", "name", "industry_tag", "type_label",
        "position_label", "next_check_label", "thesis_md",
    )
    for k in req_str:
        if not isinstance(item.get(k), str) or not item[k].strip():
            errs.append(f"{prefix}.{k}: 必須為非空字串")

    _check_date(item.get("added_date"), f"{prefix}.added_date", errs)
    _check_date(item.get("snapshot_date"), f"{prefix}.snapshot_date", errs)
    if item.get("added_price_date") is not None:
        _check_date(item.get("added_price_date"), f"{prefix}.added_price_date", errs)
    for k in ("added_price", "snapshot_price"):
        if not _is_num(item.get(k)) or float(item[k]) <= 0:
            errs.append(f"{prefix}.{k}: 必須為正數（收到 {item.get(k)!r}）")

    # bands
    bands = item.get("bands")
    if not isinstance(bands, dict):
        errs.append(f"{prefix}.bands: 必須為物件")
    else:
        b_nums = ("cheap_max", "fair_min", "fair_max", "expensive_min")
        if all(_is_num(bands.get(k)) for k in b_nums):
            c_max = float(bands["cheap_max"])
            f_min = float(bands["fair_min"])
            f_max = float(bands["fair_max"])
            e_min = float(bands["expensive_min"])
            if not (0 < c_max <= f_min < f_max < e_min):
                errs.append(
                    f"{prefix}.bands: 價位帶必須滿足 0 < cheap_max <= fair_min < fair_max < expensive_min "
                    f"（收到 {c_max}, {f_min}, {f_max}, {e_min}）"
                )
        else:
            for k in b_nums:
                if not _is_num(bands.get(k)):
                    errs.append(f"{prefix}.bands.{k}: 必須為數值")
        vb = bands.get("valuation_basis")
        if vb not in VALID_VALUATION_BASIS:
            errs.append(f"{prefix}.bands.valuation_basis: 必須為 {sorted(VALID_VALUATION_BASIS)} 之一（收到 {vb!r}）")
        for opt_num in ("normalized_eps", "book_value_per_share"):
            if opt_num not in bands or (bands[opt_num] is not None and not _is_num(bands[opt_num])):
                errs.append(f"{prefix}.bands.{opt_num}: 必須為數值或 null")
        if not isinstance(bands.get("assumptions_md"), str) or not bands["assumptions_md"].strip():
            errs.append(f"{prefix}.bands.assumptions_md: 必須為非空字串")

    # questions
    qs = item.get("questions")
    if not isinstance(qs, list) or not qs:
        errs.append(f"{prefix}.questions: 必須為非空陣列")
    else:
        for idx, q in enumerate(qs, 1):
            qp = f"{prefix}.questions[{idx - 1}]"
            if not isinstance(q, dict):
                errs.append(f"{qp}: 必須為物件")
                continue
            if q.get("id") != idx:
                errs.append(f"{qp}.id: 應為連續編號 {idx}（收到 {q.get('id')!r}）")
            for k in ("question", "how_to_check", "judgment"):
                if not isinstance(q.get(k), str) or not q[k].strip():
                    errs.append(f"{qp}.{k}: 必須為非空字串")
            _check_date(q.get("judged_at"), f"{qp}.judged_at", errs)
            if q.get("status") not in VALID_QUESTION_STATUS:
                errs.append(f"{qp}.status: 必須為 {sorted(VALID_QUESTION_STATUS)} 之一（收到 {q.get('status')!r}）")

    # checkpoints
    cps = item.get("checkpoints")
    if not isinstance(cps, list) or not cps:
        errs.append(f"{prefix}.checkpoints: 必須為非空陣列")
    else:
        for idx, cp in enumerate(cps):
            cpp = f"{prefix}.checkpoints[{idx}]"
            if not isinstance(cp, dict):
                errs.append(f"{cpp}: 必須為物件")
                continue
            cad = cp.get("cadence")
            if cad not in VALID_CADENCE:
                errs.append(f"{cpp}.cadence: 必須為 {sorted(VALID_CADENCE)} 之一（收到 {cad!r}）")
            due = cp.get("due_date")
            if due is not None:
                _check_date(due, f"{cpp}.due_date", errs)
                if cad in {"monthly", "quarterly", "ongoing"}:
                    errs.append(f"{cpp}.due_date: 常態監控（{cad}）的 due_date 應為 null")
            for k in ("date_label", "event", "metric", "bull_case", "bear_case"):
                if not isinstance(cp.get(k), str) or not cp[k].strip():
                    errs.append(f"{cpp}.{k}: 必須為非空字串")

    # exit_conditions & exit_note_md
    exs = item.get("exit_conditions")
    if not isinstance(exs, list) or not exs:
        errs.append(f"{prefix}.exit_conditions: 必須為非空陣列")
    else:
        for idx, ex in enumerate(exs):
            if not isinstance(ex, str) or not ex.strip():
                errs.append(f"{prefix}.exit_conditions[{idx}]: 必須為非空字串")
            elif ex.lstrip().startswith("注意："):
                errs.append(f"{prefix}.exit_conditions[{idx}]: 「注意：...」提醒請放入 exit_note_md，勿混入 exit_conditions")
    if "exit_note_md" not in item or (item["exit_note_md"] is not None and (not isinstance(item["exit_note_md"], str) or not item["exit_note_md"].strip())):
        errs.append(f"{prefix}.exit_note_md: 必須為非空字串或 null")

    # red_flags
    rfs = item.get("red_flags")
    if not isinstance(rfs, list) or any(not isinstance(x, str) or not x.strip() for x in rfs):
        errs.append(f"{prefix}.red_flags: 必須為字串陣列")

    # quant_ref
    qr = item.get("quant_ref")
    if qr is not None:
        if not isinstance(qr, dict):
            errs.append(f"{prefix}.quant_ref: 必須為物件或 null")
        else:
            for k in ("h136_rank", "pool_size"):
                if qr.get(k) is not None and (not isinstance(qr.get(k), int) or isinstance(qr.get(k), bool)):
                    errs.append(f"{prefix}.quant_ref.{k}: 必須為整數或 null")
            if qr.get("as_of") is not None:
                _check_date(qr.get("as_of"), f"{prefix}.quant_ref.as_of", errs)
            if not isinstance(qr.get("summary_md"), str) or not qr["summary_md"].strip():
                errs.append(f"{prefix}.quant_ref.summary_md: 必須為非空字串")

    # reports & logs
    _check_reports(item.get("reports"), prefix, analysis_dir, errs)
    logs = item.get("logs")
    if not isinstance(logs, list) or not logs:
        errs.append(f"{prefix}.logs: 必須為非空陣列")
    else:
        for idx, lg in enumerate(logs):
            lp = f"{prefix}.logs[{idx}]"
            if not isinstance(lg, dict):
                errs.append(f"{lp}: 必須為物件")
                continue
            _check_date(lg.get("date"), f"{lp}.date", errs)
            if not isinstance(lg.get("note_md"), str) or not lg["note_md"].strip():
                errs.append(f"{lp}.note_md: 必須為非空字串")

    if is_removed:
        _check_date(item.get("removed_date"), f"{prefix}.removed_date", errs)
        if not isinstance(item.get("removed_reason_md"), str) or not item["removed_reason_md"].strip():
            errs.append(f"{prefix}.removed_reason_md: 必須為非空字串")


def validate_watchlist(data: object, analysis_dir: Path = ANALYSIS_DIR) -> list[str]:
    """回傳所有錯誤訊息清單；空清單代表通過驗證。"""
    errs: list[str] = []
    if not isinstance(data, dict):
        return ["根節點必須為 JSON 物件"]
    if data.get("schema_version") != 1:
        errs.append(f"schema_version 必須為 1（收到 {data.get('schema_version')!r}）")
    _check_date(data.get("updated_at"), "updated_at", errs)
    if not isinstance(data.get("header_notes_md"), str) or not data["header_notes_md"].strip():
        errs.append("header_notes_md: 必須為非空字串")

    for section in ("tracking", "passed", "removed"):
        if not isinstance(data.get(section), list):
            errs.append(f"{section}: 必須為陣列")

    if errs:
        return errs

    seen_ids: dict[str, str] = {}
    for section in ("tracking", "passed", "removed"):
        for idx, item in enumerate(data[section]):
            sid = item.get("stock_id") if isinstance(item, dict) else None
            if isinstance(sid, str) and sid:
                if sid in seen_ids:
                    errs.append(f"stock_id 重複：{sid} 同時出現在 {seen_ids[sid]} 與 {section}[{idx}]")
                else:
                    seen_ids[sid] = f"{section}[{idx}]"

    for idx, item in enumerate(data["tracking"]):
        label = f"tracking[{idx}]({item.get('stock_id', '?')})" if isinstance(item, dict) else f"tracking[{idx}]"
        _validate_tracking_item(item, label, analysis_dir, is_removed=False, errs=errs)

    for idx, item in enumerate(data["passed"]):
        label = f"passed[{idx}]({item.get('stock_id', '?')})" if isinstance(item, dict) else f"passed[{idx}]"
        if not isinstance(item, dict):
            errs.append(f"{label}: 必須為物件")
            continue
        for k in ("stock_id", "name", "reason_md", "revisit_md"):
            if not isinstance(item.get(k), str) or not item[k].strip():
                errs.append(f"{label}.{k}: 必須為非空字串")
        _check_date(item.get("analyzed_date"), f"{label}.analyzed_date", errs)
        rpm = item.get("revisit_price_max")
        if "revisit_price_max" not in item or (rpm is not None and (not _is_num(rpm) or float(rpm) <= 0)):
            errs.append(f"{label}.revisit_price_max: 必須為正數或 null")
        _check_reports(item.get("reports"), label, analysis_dir, errs)

    for idx, item in enumerate(data["removed"]):
        label = f"removed[{idx}]({item.get('stock_id', '?')})" if isinstance(item, dict) else f"removed[{idx}]"
        _validate_tracking_item(item, label, analysis_dir, is_removed=True, errs=errs)

    return errs


# ─────────── Markdown 渲染 ───────────
def _fmt_num(val: float | int) -> str:
    f = float(val)
    if f.is_integer():
        return f"{int(f):,}"
    return f"{f:,.2f}".rstrip("0").rstrip(".")


def _short_md(ymd: str) -> str:
    _, m, d = ymd.split("-")
    return f"{int(m)}/{int(d)}"


def _indent_2(text: str) -> str:
    return "\n".join(("  " + line if line else "") for line in text.splitlines())


def _fmt_or_item(item: str, idx: int, is_last: bool) -> str:
    end = "。" if is_last else "；"
    if idx == 0:
        return f"  - {item.rstrip('；。')}{end}"
    body = item.rstrip("；。")
    sep = " " if body and body[0].isascii() and body[0].isalnum() else ""
    return f"  - 或{sep}{body}{end}"


def _render_tracking_detail(item: dict, is_first: bool = False) -> str:
    lines: list[str] = []
    lines.append(f"### {item['stock_id']} {item['name']}（{item['industry_tag']}）")
    lines.append("")

    rep_links = "、".join(f"[{r['filename']}]({r['filename']})" for r in item["reports"])
    lines.append(
        f"- **首次研究**：{item['added_date']}，收盤 {_fmt_num(item['added_price'])}（{_short_md(item.get('added_price_date') or item['snapshot_date'])}）。"
        f"報告：{rep_links}"
    )

    thesis = item["thesis_md"].strip()
    if thesis.startswith("- "):
        lines.append("- **論點**：")
        lines.append(_indent_2(thesis))
    else:
        t_lines = thesis.splitlines()
        lines.append(f"- **論點**：{t_lines[0]}")
        if len(t_lines) > 1:
            lines.append(_indent_2("\n".join(t_lines[1:])))

    q_note = "（量化看不出來，要持續回答）" if is_first else ""
    judged_dates = sorted({q["judged_at"] for q in item["questions"]})
    j_hdr = f"目前判斷（{judged_dates[-1]}）" if judged_dates else "目前判斷"
    lines.append(f"- **質化問題**{q_note}：")
    lines.append("")
    lines.append(f"| # | 問題 | 怎麼查 | {j_hdr} |")
    lines.append("|---|---|---|---|")
    for q in item["questions"]:
        lines.append(f"| {q['id']} | {q['question']} | {q['how_to_check']} | {q['judgment']} |")
    lines.append("")

    lines.append("- **價位帶**：")
    lines.append(_indent_2(item["bands"]["assumptions_md"].strip()))

    lines.append("- **檢查點**：")
    lines.append("")
    lines.append("| 日期 | 事件 | 要看什麼 | 正面 | 負面 |")
    lines.append("|---|---|---|---|---|")
    for cp in item["checkpoints"]:
        lines.append(
            f"| {cp['date_label']} | {cp['event']} | {cp['metric']} | {cp['bull_case']} | {cp['bear_case']} |"
        )
    lines.append("")

    lines.append("- **結論要改的條件**：")
    exs = item["exit_conditions"]
    for i, ex in enumerate(exs):
        lines.append(_fmt_or_item(ex, i, i == len(exs) - 1))
    if item.get("exit_note_md"):
        lines.append(f"  - {item['exit_note_md'].strip()}")

    rfs = item.get("red_flags") or []
    if rfs:
        lines.append("- **紅旗**：")
        for i, rf in enumerate(rfs):
            end = "。" if i == len(rfs) - 1 else "；"
            lines.append(f"  - {rf.rstrip('；。')}{end}")

    if item.get("quant_ref"):
        lines.append(f"- **量化參考**：{item['quant_ref']['summary_md'].strip()}")

    lines.append("- **紀錄**：")
    for lg in item["logs"]:
        lines.append(f"  - {lg['date']}：{lg['note_md'].strip()}")

    return "\n".join(lines)


def render_watchlist_md(data: dict) -> str:
    out: list[str] = [STATIC_INTRO_MD, "", "## 已研究", ""]

    tracking = data["tracking"]
    snap_dates = {t["snapshot_date"] for t in tracking}
    price_hdr = f"現價（{_short_md(next(iter(snap_dates)))}）" if len(snap_dates) == 1 else "現價"

    out.append(f"| 代號 | 名稱 | 類型 | {price_hdr} | 便宜 | 合理 | 昂貴 | 現在位置 | 下次檢查 |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for t in tracking:
        b = t["bands"]
        p_cell = (
            _fmt_num(t["snapshot_price"])
            if len(snap_dates) == 1
            else f"{_fmt_num(t['snapshot_price'])}（{_short_md(t['snapshot_date'])}）"
        )
        out.append(
            f"| {t['stock_id']} | {t['name']} | {t['type_label']} | {p_cell} "
            f"| ≤ {_fmt_num(b['cheap_max'])} | {_fmt_num(b['fair_min'])}～{_fmt_num(b['fair_max'])} "
            f"| ≥ {_fmt_num(b['expensive_min'])} | {t['position_label']} | {t['next_check_label']} |"
        )
    out.append("")
    out.append(data["header_notes_md"].strip())
    out.append("")
    out.append("---")

    for idx, t in enumerate(tracking):
        out.append("")
        out.append(_render_tracking_detail(t, is_first=(idx == 0)))
        out.append("")
        out.append("---")

    if data["passed"]:
        out.append("")
        out.append("## 已分析、不追蹤（舊格式）")
        out.append("")
        out.append("| 代號 | 名稱 | 分析日 | 不追的原因 | 什麼情況下重新看 | 報告 |")
        out.append("|---|---|---|---|---|---|")
        for p in data["passed"]:
            rep_links = "、".join(f"[{r['filename']}]({r['filename']})" for r in p["reports"])
            out.append(
                f"| {p['stock_id']} | {p['name']} | {p['analyzed_date']} "
                f"| {p['reason_md']} | {p['revisit_md']} | {rep_links} |"
            )

    removed = data["removed"]
    if removed:
        out.append("")
        out.append("## 已移出（下市、合併等）")
        out.append("")
        out.append("| 代號 | 名稱 | 加入日 | 移出日 | 移出原因 | 報告 |")
        out.append("|---|---|---|---|---|---|")
        for r in removed:
            rep_links = "、".join(f"[{rep['filename']}]({rep['filename']})" for rep in r["reports"])
            out.append(
                f"| {r['stock_id']} | {r['name']} | {r['added_date']} | {r['removed_date']} "
                f"| {r['removed_reason_md']} | {rep_links} |"
            )

    return "\n".join(out) + "\n"


def load_and_sync(
    json_path: Path = WATCHLIST_JSON,
    md_path: Path = WATCHLIST_MD,
    write_md: bool = True,
) -> tuple[dict, bool]:
    """載入並驗證 watchlist.json；write_md=True 時若 WATCHLIST.md 內容不同才寫入。回傳 (data, md_updated)。"""
    if not json_path.is_file():
        raise FileNotFoundError(f"找不到 {json_path}")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    errs = validate_watchlist(data, json_path.parent)
    if errs:
        msg = "watchlist.json 驗證失敗：\n  - " + "\n  - ".join(errs)
        raise ValueError(msg)

    rendered = render_watchlist_md(data)
    updated = False
    if write_md:
        old = md_path.read_text(encoding="utf-8") if md_path.is_file() else None
        if old != rendered:
            tmp = md_path.with_suffix(".md.tmp")
            tmp.write_text(rendered, encoding="utf-8")
            tmp.replace(md_path)
            updated = True
    return data, updated


def main() -> None:
    ap = argparse.ArgumentParser(description="驗證 watchlist.json 並重新生成 WATCHLIST.md")
    ap.add_argument("--check", action="store_true", help="只驗證 watchlist.json，不改寫 WATCHLIST.md")
    args = ap.parse_args()
    try:
        data, updated = load_and_sync(write_md=not args.check)
    except Exception as e:
        print(f"✗ {e}", file=sys.stderr)
        sys.exit(1)

    c_t, c_p, c_r = len(data["tracking"]), len(data["passed"]), len(data["removed"])
    if args.check:
        print(f"✓ watchlist.json 驗證通過（tracking={c_t}, passed={c_p}, removed={c_r}）")
    else:
        state = "已更新" if updated else "無變動"
        print(f"✓ watchlist.json 驗證通過（tracking={c_t}, passed={c_p}, removed={c_r}）；WATCHLIST.md {state}")


if __name__ == "__main__":
    main()
