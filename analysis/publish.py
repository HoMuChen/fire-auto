"""驗證個股質化追蹤清單與綜合分析報告，並發布至 Cloudflare R2 的 analysis/v1/ 前綴。

安全邊界：
  所有 R2 讀取、上傳與孤兒檔清理嚴格限定在 `analysis/v1/` 前綴下，絕不碰觸 `v1/`（市場資料湖）。

用法：
  /usr/local/bin/python3 analysis/publish.py [--check-only] [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ANALYSIS_DIR))
import render_watchlist  # noqa: E402

PREFIX = "analysis/v1"
STAGE = ROOT / "data" / "lake" / "analysis" / "v1"
MANIFEST = "_manifest.json"
TAIPEI_TZ = ZoneInfo("Asia/Taipei")

REPORT_FILENAME_RE = re.compile(
    r"^(?P<stock_id>[A-Za-z0-9]+)_(?P<name>.+)_綜合分析_(?P<report_date>\d{4}-\d{2}-\d{2})\.md$"
)
TITLE_LINE_RE = re.compile(r"^#\s*(?P<title_name>.+?)（(?P<stock_id>[A-Za-z0-9]+)）")
DATA_AS_OF_RE = re.compile(r"資料截至\s*(?P<data_as_of>\d{4}-\d{2}-\d{2})")


def _assert_safe_r2_key(key: str) -> None:
    """硬性防護：確保所有 R2 操作絕對限制在 analysis/v1/ 底下，絕不觸碰 v1/。"""
    if (
        not isinstance(key, str)
        or not key.startswith("analysis/v1/")
        or key == "analysis/v1/"
        or ".." in key
        or key.startswith("v1/")
    ):
        raise RuntimeError(f"拒絕不安全的 R2 key（僅允許 analysis/v1/*）：{key!r}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_if_changed(src: Path, dst: Path) -> tuple[str, int]:
    """若目標檔不存在或 sha256 不同才複製，回傳 (sha256, size)。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    src_hash = sha256_file(src)
    if not dst.is_file() or sha256_file(dst) != src_hash:
        tmp = dst.with_name(dst.name + ".tmp")
        shutil.copyfile(src, tmp)
        tmp.replace(dst)
    return src_hash, dst.stat().st_size


def scan_reports(watchlist_data: dict, analysis_dir: Path = ANALYSIS_DIR) -> list[dict]:
    """掃描 analysis/*_綜合分析_*.md 並解析檔名與檔頭資訊。"""
    status_by_sid: dict[str, str] = {}
    for section in ("tracking", "passed", "removed"):
        for item in watchlist_data.get(section, []):
            status_by_sid[item["stock_id"]] = section

    entries: list[dict] = []
    for path in sorted(analysis_dir.glob("*_綜合分析_*.md"), key=lambda p: p.name):
        m_fn = REPORT_FILENAME_RE.match(path.name)
        if not m_fn:
            raise ValueError(f"報告檔名不符合 {{stock_id}}_{{name}}_綜合分析_{{YYYY-MM-DD}}.md：{path.name}")
        sid = m_fn.group("stock_id")
        fn_name = m_fn.group("name")
        report_date = m_fn.group("report_date")

        head_lines = path.read_text(encoding="utf-8").splitlines()[:10]
        if not head_lines:
            raise ValueError(f"報告檔案為空：{path.name}")

        m_title = TITLE_LINE_RE.match(head_lines[0].strip())
        if m_title:
            if m_title.group("stock_id") != sid:
                raise ValueError(
                    f"{path.name} 第一行代號（{m_title.group('stock_id')}）與檔名（{sid}）不一致"
                )
            display_name = m_title.group("title_name").strip()
        else:
            display_name = fn_name

        data_as_of = None
        for line in head_lines[:6]:
            m_asof = DATA_AS_OF_RE.search(line)
            if m_asof:
                data_as_of = m_asof.group("data_as_of")
                break
        if not data_as_of:
            raise ValueError(f"{path.name} 檔頭找不到「資料截至 YYYY-MM-DD」")

        key = f"{PREFIX}/reports/{path.name}"
        _assert_safe_r2_key(key)
        entries.append({
            "stock_id": sid,
            "name": display_name,
            "report_date": report_date,
            "data_as_of": data_as_of,
            "filename": path.name,
            "key": key,
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
            "in_watchlist": status_by_sid.get(sid),
        })
    return entries


def build_stage(watchlist_data: dict, reports: list[dict]) -> dict:
    """將 watchlist.json、所有報告與 _manifest.json 寫入本機 stage（data/lake/analysis/v1）。"""
    STAGE.mkdir(parents=True, exist_ok=True)
    reports_stage = STAGE / "reports"
    reports_stage.mkdir(parents=True, exist_ok=True)

    wl_src = ANALYSIS_DIR / "watchlist.json"
    wl_dst = STAGE / "watchlist.json"
    wl_sha, wl_size = copy_if_changed(wl_src, wl_dst)
    wl_key = f"{PREFIX}/watchlist.json"
    _assert_safe_r2_key(wl_key)

    expected_report_names = {r["filename"] for r in reports}
    for r in reports:
        copy_if_changed(ANALYSIS_DIR / r["filename"], reports_stage / r["filename"])
    for old in reports_stage.glob("*.md"):
        if old.name not in expected_report_names:
            old.unlink()

    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(TAIPEI_TZ).isoformat(timespec="seconds"),
        "watchlist": {
            "key": wl_key,
            "sha256": wl_sha,
            "size": wl_size,
            "updated_at": watchlist_data["updated_at"],
            "counts": {
                "tracking": len(watchlist_data["tracking"]),
                "passed": len(watchlist_data["passed"]),
                "removed": len(watchlist_data["removed"]),
            },
        },
        "reports": reports,
    }
    (STAGE / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    p = ROOT / ".env.local"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    for k, v in os.environ.items():
        if k.startswith(("R2_", "TRADING_SYNC_")):
            env[k] = v
    return env


def r2_client(env: dict[str, str], allow_readonly: bool = False):
    acc = env.get("R2_ACCOUNT_ID")
    ak = env.get("R2_WRITE_ACCESS_KEY_ID") or (env.get("R2_READ_ACCESS_KEY_ID") if allow_readonly else None)
    sk = env.get("R2_WRITE_SECRET_ACCESS_KEY") or (env.get("R2_READ_SECRET_ACCESS_KEY") if allow_readonly else None)
    if not (acc and ak and sk):
        need = ("R2_ACCOUNT_ID", "R2_WRITE_ACCESS_KEY_ID", "R2_WRITE_SECRET_ACCESS_KEY")
        miss = [k for k in need if not env.get(k)]
        raise RuntimeError(f".env.local 缺 {', '.join(miss)}")

    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=f"https://{acc}.r2.cloudflarestorage.com",
        aws_access_key_id=ak,
        aws_secret_access_key=sk,
        region_name="auto",
        config=Config(retries={"max_attempts": 5, "mode": "standard"}),
    )


def _extract_remote_hashes(remote_manifest: dict) -> dict[str, str]:
    have: dict[str, str] = {}
    wl = remote_manifest.get("watchlist")
    if isinstance(wl, dict) and isinstance(wl.get("key"), str) and isinstance(wl.get("sha256"), str):
        if wl["key"].startswith(f"{PREFIX}/"):
            _assert_safe_r2_key(wl["key"])
            have[wl["key"]] = wl["sha256"]
    for r in remote_manifest.get("reports", []):
        if isinstance(r, dict) and isinstance(r.get("key"), str) and isinstance(r.get("sha256"), str):
            if r["key"].startswith(f"{PREFIX}/"):
                _assert_safe_r2_key(r["key"])
                have[r["key"]] = r["sha256"]
    return have


def notify_webhook(env: dict[str, str], manifest: dict) -> None:
    url = env.get("TRADING_SYNC_URL", "").strip()
    token = env.get("TRADING_SYNC_TOKEN", "").strip()
    if not (url and token):
        return
    body = json.dumps({
        "source": "fire-auto",
        "prefix": PREFIX,
        "generated_at": manifest["generated_at"],
    }).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "fire-auto-analysis-publish/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            print(f"Webhook 通知成功（HTTP {resp.status}）：{url}")
    except Exception as e:
        print(f"⚠ Webhook 通知失敗（不影響發布）：{e}", file=sys.stderr)


def publish(manifest: dict, dry_run: bool = False) -> None:
    env = load_env()
    bucket = env.get("R2_BUCKET", "trading-homuchen-com")
    manifest_key = f"{PREFIX}/{MANIFEST}"
    _assert_safe_r2_key(manifest_key)

    # key -> (local_stage_path, sha256, size, content_type)
    want: dict[str, tuple[Path, str, int, str]] = {}
    wl = manifest["watchlist"]
    _assert_safe_r2_key(wl["key"])
    want[wl["key"]] = (STAGE / "watchlist.json", wl["sha256"], wl["size"], "application/json")
    for r in manifest["reports"]:
        _assert_safe_r2_key(r["key"])
        want[r["key"]] = (
            STAGE / "reports" / r["filename"],
            r["sha256"],
            r["size"],
            "text/markdown; charset=utf-8",
        )

    s3 = None
    remote_exists = False
    have: dict[str, str] = {}
    try:
        s3 = r2_client(env, allow_readonly=dry_run)
        try:
            obj = s3.get_object(Bucket=bucket, Key=manifest_key)
            remote_manifest = json.loads(obj["Body"].read())
            remote_exists = True
            have = _extract_remote_hashes(remote_manifest)
        except s3.exceptions.NoSuchKey:
            remote_exists = False
            have = {}
    except Exception as e:
        if not dry_run:
            sys.exit(f"✗ 無法連線 R2：{e}")
        print(f"ℹ dry-run：本機無法連線 R2（{e}），改以空白遠端預覽；本機 stage 已寫入 {STAGE}")
        s3 = None

    todo = [k for k in sorted(want) if have.get(k) != want[k][1]]
    orphans = sorted(set(have) - set(want))
    for k in orphans:
        _assert_safe_r2_key(k)

    kb = sum(want[k][2] for k in todo) / 1024
    print(
        f"R2 {bucket}/{PREFIX}：待上傳 {len(todo)} 檔（{kb:,.1f} KB）、"
        f"刪除孤兒 {len(orphans)} 檔、不變 {len(want) - len(todo)} 檔"
    )

    if dry_run:
        for k in todo:
            print(f"  + {k} ({want[k][3]}, {want[k][2]:,} bytes)")
        for k in orphans:
            print(f"  - {k}")
        print(f"  * {manifest_key} (最後更新)")
        return

    assert s3 is not None
    if not todo and not orphans and remote_exists:
        print("遠端內容無變動，略過上傳")
        return

    def _put(key: str) -> None:
        _assert_safe_r2_key(key)
        local_path, _, _, ctype = want[key]
        s3.upload_file(str(local_path), bucket, key, ExtraArgs={"ContentType": ctype})

    if todo:
        with ThreadPoolExecutor(min(8, len(todo))) as ex:
            list(ex.map(_put, todo))

    # 所有內容檔成功後，最後才切換 _manifest.json（原子發布訊號）
    _assert_safe_r2_key(manifest_key)
    s3.put_object(
        Bucket=bucket,
        Key=manifest_key,
        Body=(STAGE / MANIFEST).read_bytes(),
        ContentType="application/json",
    )

    # _manifest.json 更新完成後才刪孤兒檔
    for key in orphans:
        _assert_safe_r2_key(key)
        s3.delete_object(Bucket=bucket, Key=key)

    print(f"✓ 已發布至 r2://{bucket}/{PREFIX}/（watchlist + {len(manifest['reports'])} 份報告）")
    notify_webhook(env, manifest)


def main() -> None:
    ap = argparse.ArgumentParser(description="驗證並發布 analysis/ 至 R2 analysis/v1/")
    ap.add_argument("--check-only", action="store_true", help="只驗證 watchlist.json 與報告並更新本地 WATCHLIST.md")
    ap.add_argument("--dry-run", action="store_true", help="建置本機 stage 並預覽 R2 變更，不實際上傳")
    args = ap.parse_args()

    try:
        wl_data, md_updated = render_watchlist.load_and_sync(write_md=True)
        reports = scan_reports(wl_data)
    except Exception as e:
        print(f"✗ {e}", file=sys.stderr)
        sys.exit(1)

    c = {
        "tracking": len(wl_data["tracking"]),
        "passed": len(wl_data["passed"]),
        "removed": len(wl_data["removed"]),
    }
    md_msg = "已同步更新" if md_updated else "已是最新"
    print(
        f"✓ 驗證通過：tracking={c['tracking']}, passed={c['passed']}, removed={c['removed']}；"
        f"報告 {len(reports)} 份；WATCHLIST.md {md_msg}"
    )

    if args.check_only:
        return

    manifest = build_stage(wl_data, reports)
    publish(manifest, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
