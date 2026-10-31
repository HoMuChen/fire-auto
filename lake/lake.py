"""台股資料湖：本機研究資料（data/v2 等）→ 長表 parquet → Cloudflare R2，供其他專案唯讀。

結構（bucket＝.env.local 的 R2_BUCKET，目前 trading-homuchen-com；前綴 v1/）：
  _manifest.json                每個資料集的欄位、筆數、日期範圍、各檔 sha256（最後才上傳＝發布完成的訊號）
  catalog.duckdb                只有 view 定義（明確列檔，不用萬用字元 → 讀者不需 LIST、看到固定的檔案組合）
  universe.parquet、revenue.parquet、fin/bs/cf/dividend.parquet
  price|per|inst|margin|mv|holding/year=YYYY/data.parquet
  market/*.parquet、futures_30min/*.parquet

用法（/usr/local/bin/python3）：
  python3 lake/lake.py build [名稱 ...]     # 重建本機 data/lake/v1（只有內容變了的檔才會改寫）
  python3 lake/lake.py verify               # 用本機目錄檔跑一致性檢查（筆數、與 harness.load_stock 比對）
  python3 lake/lake.py upload [--dry-run]   # 只傳 sha256 變了的檔 → catalog.duckdb → _manifest.json → 刪 R2 上的孤兒檔
  python3 lake/lake.py status               # 本機 manifest 摘要
  python3 lake/lake.py daily                # 排程用：lake/update.py 增量更新 data/v2 → build → verify → upload（有 token 才傳）；失敗發 Telegram
環境變數（.env.local）：R2_ACCOUNT_ID、R2_BUCKET（預設 trading-homuchen-com）、R2_WRITE_ACCESS_KEY_ID、R2_WRITE_SECRET_ACCESS_KEY
規則：只有這台機器寫入；內容沒變的檔不改寫（sha256 相同）。新股回補歷史、除權息重新還原（adj_* 整段改）時
      過去年份的分區也會變 → 讀者以 _manifest.json 的 sha256 判斷要不要重抓。
"""
import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "data" / "v2"
PREFIX = "v1"
STAGE = ROOT / "data" / "lake" / PREFIX
MANIFEST = "_manifest.json"
CATALOG = "catalog.duckdb"
CATALOG_STORAGE = "v1.0.0"          # 讓 DuckDB 1.0 以上都能開
_AVAIL_Q = {3: (5, 16, 0), 6: (8, 15, 0), 9: (11, 15, 0), 12: (4, 1, 1)}   # 同 research/v2/xs.py：法定截止 +1 天


# ─────────── 讀取來源 ───────────
def _per_stock(kind, sids=None):
    """data/v2/{kind}/{sid}.parquet 全部串起來；空檔跳過；缺 stock_id 欄的以檔名補上。"""
    out = []
    for f in sorted((V2 / kind).glob("*.parquet")):
        if sids is not None and f.stem not in sids:
            continue
        try:
            d = pd.read_parquet(f)
        except Exception:
            continue                                   # 空表（無欄位）
        if not len(d):
            continue
        d["stock_id"] = f.stem
        out.append(d)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _q_avail(dates):
    d = pd.to_datetime(dates, errors="coerce")
    m = d.dt.month
    return pd.to_datetime(dict(year=d.dt.year + (m == 12), month=m.map({k: v[0] for k, v in _AVAIL_Q.items()}),
                               day=m.map({k: v[1] for k, v in _AVAIL_Q.items()})), errors="coerce")


def b_universe():
    u = pd.read_csv(V2 / "universe.csv", dtype={"stock_id": str})
    return u.rename(columns={"type": "market"}).assign(delist_date=lambda x: pd.to_datetime(x.delist_date))


def b_price():
    raw, adj = _per_stock("raw"), _per_stock("adj")
    adj = adj[["stock_id", "date", "open", "high", "low", "close"]].rename(
        columns={c: f"adj_{c}" for c in ("open", "high", "low", "close")})
    return raw.merge(adj, on=["stock_id", "date"], how="left")


def b_per():
    return _per_stock("per")


def b_revenue():
    d = _per_stock("revenue").drop(columns=["country"], errors="ignore")
    ym = pd.to_datetime(dict(year=d.revenue_year, month=d.revenue_month, day=1))
    return d.assign(avail_date=ym + pd.offsets.MonthBegin(1) + pd.Timedelta(days=10))      # 次月 11 日


def _fin(kind):
    d = _per_stock(kind)
    return d.assign(avail_date=_q_avail(d.date))


def b_dividend():
    return _per_stock("div")


SPEC = {   # 名稱 → (建表函數, 分區, 說明)
    "universe": (b_universe, None, "股票池：上市/上櫃普通股 2,594 檔（含 2010 後下市股，delist_date）"),
    "price": (b_price, "year", "日線：原始 OHLC（open/high/low/close，close=0 為無成交）+ 還原權息 adj_*；volume 股、money 元"),
    "per": (b_per, "year", "本益比 PER、淨值比 PBR、殖利率 dividend_yield（FinMind TaiwanStockPER）"),
    "revenue": (b_revenue, None, "月營收（元）；avail_date＝次月 11 日（保守可得日）"),
    "inst": (lambda: _per_stock("inst"), "year", "三大法人買賣（股），name＝Foreign_Investor / Investment_Trust / Dealer_* 等；2012-05 起完整"),
    "margin": (lambda: _per_stock("margin"), "year", "融資融券（張），欄位沿用 FinMind TaiwanStockMarginPurchaseShortSale"),
    "fin": (lambda: _fin("fin"), None, "綜合損益表（長表 type/value，單季值，元）；avail_date＝法定截止 +1 天"),
    "bs": (lambda: _fin("bs"), None, "資產負債表（長表，季底，元；*_per 為占總資產 %）；2012Q4 起；avail_date 同上"),
    "cf": (lambda: _fin("cf"), None, "現金流量表（長表，**年初累計值**，元）；2012Q1 起；avail_date 同上"),
    "mv": (lambda: _per_stock("mv"), "year", "日市值（元）；⚠ FinMind 有 int32 溢位（2,147,483,647 股）的錯值未修正"),
    "holding": (lambda: _per_stock("hold"), "year", "集保股權分散表：HoldingSharesLevel 級距、people 人數、percent %、unit 股；2010-14 月頻、2015 起週頻"),
    "dividend": (b_dividend, None, "股利政策與除權息日（FinMind TaiwanStockDividend 原欄位）"),
}
MARKET = {   # market/{名稱}.parquet ← 來源檔（index 是日期的會轉成 date 欄）
    "taiex": ("futures/taiex.parquet", "加權指數日線（FinMind TAIEX）"),
    "taiex_tr": ("taiex_tr.parquet", "加權報酬指數"),
    "mtx_daily": ("futures/MTX_daily.parquet", "小台指期日線（全部合約月份）"),
    "txo_chain": ("futures/TXO_chain.parquet", "台指選擇權月選全序列結算價"),
    "txo_weekly": ("futures/TXO_weekly.parquet", "台指選擇權週選（2012-11 起）"),
    "txo_final_settle": ("futures/TXO_final_settle.parquet", "TXO 最後結算價（期交所）"),
    "tx_large_traders": ("futures/TX_large_traders.parquet", "台指期大額交易人未平倉"),
    "mkt_inst": ("futures/mkt_inst.parquet", "全市場三大法人買賣金額"),
    "mkt_margin": ("futures/mkt_margin.parquet", "全市場融資融券"),
    "gspc": ("futures/gspc.parquet", "S&P 500 日線"),
    "txo_vix": ("futures/TXO_vix.parquet", "自建台指 VIX（月選）"),
    "txo_oi": ("futures/TXO_oi.parquet", "TXO 買權 / 賣權總未平倉"),
    "ad": ("futures/ad.parquet", "自建上市櫃漲跌家數"),
    "breadth": ("futures/breadth.parquet", "自建市場寬度：跌離 52 週高 ≥10/20/50% 家數比例"),
    "mkt_maint": ("futures/mkt_maint.parquet", "自建大盤融資維持率"),
}
FUT30 = {"QFF": "小型台積電期 30 分 K（日盤，連續近月）", "SEF": "小型智邦期 30 分 K（日盤，連續近月）"}


# ─────────── 正規化與寫檔 ───────────
def normalize(df):
    df = df.copy()
    if isinstance(df.index, pd.DatetimeIndex) or df.index.name == "date":   # 自建指標以日期為 index
        df = df.rename_axis("date").reset_index()
    for c in df.columns:
        if c in ("date", "avail_date", "delist_date"):
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.normalize()
        elif c == "stock_id":
            df[c] = df[c].astype(str)
    keys = [c for c in ("stock_id", "date", "datetime") if c in df.columns]
    if keys:
        df = df.sort_values(keys, kind="stable").reset_index(drop=True)
    return df


def to_table(df):
    t = pa.Table.from_pandas(df, preserve_index=False)
    fields = []
    for f in t.schema:
        typ = f.type
        if f.name in ("date", "avail_date", "delist_date"):
            typ = pa.date32()
        elif pa.types.is_large_string(typ) or pa.types.is_string(typ):
            typ = pa.string()
        elif pa.types.is_timestamp(typ):
            typ = pa.timestamp("us")
        fields.append(pa.field(f.name, typ))
    return t.cast(pa.schema(fields)).replace_schema_metadata(None)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write(table, rel):
    """寫到暫存檔；內容（sha256）與現有檔相同就不覆蓋，保留舊檔 mtime。回傳檔案資訊。"""
    p = STAGE / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    pq.write_table(table, tmp, compression="zstd", row_group_size=500_000)
    h = sha256(tmp)
    if p.exists() and sha256(p) == h:
        tmp.unlink()
    else:
        tmp.replace(p)
    return {"sha256": h, "size": p.stat().st_size, "rows": table.num_rows}


def _date_range(df):
    for c in ("date", "datetime"):
        if c in df.columns and len(df):
            s = pd.to_datetime(df[c])
            return str(s.min().date()), str(s.max().date())
    return None, None


def _entry(df, files, desc, partition=None):
    lo, hi = _date_range(df)
    t = to_table(df.head(0))
    return {"description": desc, "partition": partition, "rows": int(len(df)), "min_date": lo, "max_date": hi,
            "columns": {f.name: str(f.type) for f in t.schema}, "files": files}


def build_dataset(name, ds):
    fn, part, desc = SPEC[name]
    t0 = time.time()
    df = normalize(fn())
    files = {}
    if part == "year":
        years = df["date"].dt.year
        for y in sorted(years.dropna().unique()):
            rel = f"{name}/year={int(y)}/data.parquet"
            files[rel] = write(to_table(df[years == y]), rel)
        for old in (STAGE / name).glob("year=*/data.parquet"):           # 不再存在的年份
            if str(old.relative_to(STAGE)) not in files:
                old.unlink()
    else:
        files[f"{name}.parquet"] = write(to_table(df), f"{name}.parquet")
    ds[name] = _entry(df, files, desc, part)
    print(f"  {name:10s} {len(df):>12,} 筆  {len(files):>3} 檔  {ds[name]['min_date']} ~ {ds[name]['max_date']}  {time.time()-t0:.0f}s", flush=True)


def build_market(ds):
    for name, (src, desc) in MARKET.items():
        df = normalize(pd.read_parquet(V2 / src))
        rel = f"market/{name}.parquet"
        ds[f"market.{name}"] = _entry(df, {rel: write(to_table(df), rel)}, desc)
    for sym, desc in FUT30.items():
        df = pd.read_csv(ROOT / "data" / "futures" / f"{sym}_30min_day.csv")
        df["datetime"] = pd.to_datetime(df.datetime)
        df = normalize(df)
        rel = f"futures_30min/{sym}.parquet"
        ds[f"futures_30min.{sym.lower()}"] = _entry(df, {rel: write(to_table(df), rel)}, desc + "；datetime 為台北時間、K 棒起始")
    print(f"  market / futures_30min：{len(MARKET)} + {len(FUT30)} 檔", flush=True)


def load_manifest(path=None):
    p = path or STAGE / MANIFEST
    return json.loads(Path(p).read_text()) if Path(p).exists() else {"version": PREFIX, "datasets": {}}


def build(names):
    STAGE.mkdir(parents=True, exist_ok=True)
    man = load_manifest()
    ds = man["datasets"]
    want = names or list(SPEC) + ["market"]
    print(f"build → {STAGE}")
    for n in want:
        if n == "market":
            build_market(ds)
        else:
            build_dataset(n, ds)
    man.update(version=PREFIX, generated_at=datetime.now().isoformat(timespec="seconds"),
               source="FinMind（個人使用，勿公開散布）＋ fire-auto 自建指標",
               conventions={"date": "date32；交易日", "stock_id": "字串", "avail_date": "保守可得日（研究防前視用）",
                            "units": "金額＝元、股數＝股（margin 為張）", "partition": "year=YYYY 分區；內容沒變不改寫，有變以各檔 sha256 為準（新股回補、除權息會動到過去年份）"})
    (STAGE / MANIFEST).write_text(json.dumps(man, ensure_ascii=False, indent=1))


# ─────────── DuckDB 目錄檔 ───────────
def view_name(ds_name):
    return ds_name if "." in ds_name else f"main.{ds_name}"


def write_catalog(path, base, man, setup=()):
    """只含 view；每個 view 明確列出檔案（base 可為 r2://bucket/v1、本機路徑或 http URL）。
    DuckDB 建 view 時會讀檔確認欄位 → 檔案必須已在 base 上；setup 為建 view 前要執行的 SQL（如 R2 secret）。"""
    import duckdb
    path = Path(path)
    if path.exists():
        path.unlink()
    con = duckdb.connect()
    for q in setup:
        con.execute(q)
    con.execute(f"ATTACH '{path}' AS cat (STORAGE_VERSION '{CATALOG_STORAGE}')")
    for schema in sorted({n.split(".")[0] for n in man["datasets"] if "." in n}):
        con.execute(f"CREATE SCHEMA cat.{schema}")
    for n, e in sorted(man["datasets"].items()):
        files = ", ".join(f"'{base}/{rel}'" for rel in sorted(e["files"]))
        hive = ", hive_partitioning = true, union_by_name = true" if e.get("partition") else ""
        con.execute(f"CREATE VIEW cat.{view_name(n)} AS SELECT * FROM read_parquet([{files}]{hive})")
        con.execute(f"COMMENT ON VIEW cat.{view_name(n)} IS '{e['description'].replace(chr(39), chr(39)*2)}'")
    con.execute("DETACH cat")
    con.close()


# ─────────── 驗證 ───────────
def verify():
    import duckdb
    sys.path.insert(0, str(ROOT / "research" / "v2"))
    import harness as H
    man = load_manifest()
    cat = STAGE / "_local_catalog.duckdb"
    write_catalog(cat, str(STAGE), man)
    con = duckdb.connect()
    con.execute(f"ATTACH '{cat}' AS tw (READ_ONLY)")
    bad = 0
    for n, e in sorted(man["datasets"].items()):
        got = con.execute(f"SELECT count(*) FROM tw.{view_name(n)}").fetchone()[0]
        ok = got == e["rows"] == sum(f["rows"] for f in e["files"].values())
        bad += not ok
        print(f"  {'✓' if ok else '✗'} {n:28s} {got:>12,}")
    # 與研究框架讀到的 2330 一致
    ref = H.load_stock("2330", "2026-12-31")
    q = con.execute("""SELECT date, adj_close, close AS raw_close, volume FROM tw.price
                       WHERE stock_id = '2330' AND close > 0 AND adj_close > 0 AND open > 0 ORDER BY date""").df()
    m = ref.merge(q.assign(date=pd.to_datetime(q.date)), on="date", suffixes=("", "_lake"))
    ok = len(m) == len(ref) and np.allclose(m.close, m.adj_close) and np.allclose(m.raw_close, m.raw_close_lake)
    bad += not ok
    print(f"  {'✓' if ok else '✗'} 2330 與 harness.load_stock 一致（{len(m)}/{len(ref)} 日）")
    r = con.execute("SELECT revenue_year, revenue_month, avail_date FROM tw.revenue WHERE stock_id='2330' ORDER BY revenue_year DESC, revenue_month DESC LIMIT 1").fetchone()
    print(f"    2330 最新營收 {r[0]}-{r[1]:02d} 可得日 {r[2]}")
    f = con.execute("SELECT date, avail_date FROM tw.fin WHERE stock_id='2330' ORDER BY date DESC LIMIT 1").fetchone()
    print(f"    2330 最新財報 {f[0]} 可得日 {f[1]}")
    con.close()
    cat.unlink()
    print("全部通過" if not bad else f"{bad} 項失敗")
    return bad == 0


# ─────────── 上傳 R2 ───────────
def load_env():
    env = {}
    p = ROOT / ".env.local"
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return {**env, **{k: v for k, v in os.environ.items() if k.startswith("R2_")}}


def bucket():
    return load_env().get("R2_BUCKET", "trading-homuchen-com")


def r2_client():
    import boto3
    from botocore.config import Config
    env = load_env()
    need = ("R2_ACCOUNT_ID", "R2_WRITE_ACCESS_KEY_ID", "R2_WRITE_SECRET_ACCESS_KEY")
    miss = [k for k in need if not env.get(k)]
    if miss:
        sys.exit(f".env.local 缺 {', '.join(miss)}")
    return boto3.client("s3", endpoint_url=f"https://{env['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
                        aws_access_key_id=env["R2_WRITE_ACCESS_KEY_ID"], aws_secret_access_key=env["R2_WRITE_SECRET_ACCESS_KEY"],
                        region_name="auto", config=Config(retries={"max_attempts": 5, "mode": "standard"}))


def upload(dry=False):
    man = load_manifest()
    if not man["datasets"]:
        sys.exit("本機還沒 build")
    s3, b = r2_client(), bucket()
    try:
        remote = json.loads(s3.get_object(Bucket=b, Key=f"{PREFIX}/{MANIFEST}")["Body"].read())
    except s3.exceptions.NoSuchKey:
        remote = {"datasets": {}}
    have = {rel: f["sha256"] for e in remote["datasets"].values() for rel, f in e["files"].items()}
    want = {rel: f for e in man["datasets"].values() for rel, f in e["files"].items()}
    todo = [rel for rel, f in sorted(want.items()) if have.get(rel) != f["sha256"]]
    orphans = sorted(set(have) - set(want))
    mb = sum(want[r]["size"] for r in todo) / 1e6
    print(f"R2 {b}/{PREFIX}：上傳 {len(todo)} 檔（{mb:,.1f} MB）、刪除孤兒 {len(orphans)} 檔、不變 {len(want) - len(todo)} 檔")
    if dry:
        for r in todo[:30]:
            print("  +", r)
        return
    for i, rel in enumerate(todo, 1):
        s3.upload_file(str(STAGE / rel), b, f"{PREFIX}/{rel}", ExtraArgs={"ContentType": "application/vnd.apache.parquet"})
        if i % 20 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)}", flush=True)
    env = load_env()
    secret = (f"CREATE SECRET (TYPE r2, KEY_ID '{env['R2_WRITE_ACCESS_KEY_ID']}', SECRET '{env['R2_WRITE_SECRET_ACCESS_KEY']}', "
              f"ACCOUNT_ID '{env['R2_ACCOUNT_ID']}')")
    write_catalog(STAGE / CATALOG, f"r2://{b}/{PREFIX}", man, setup=("INSTALL httpfs", "LOAD httpfs", secret))
    s3.upload_file(str(STAGE / CATALOG), b, f"{PREFIX}/{CATALOG}", ExtraArgs={"ContentType": "application/octet-stream"})
    s3.put_object(Bucket=b, Key=f"{PREFIX}/{MANIFEST}", Body=(STAGE / MANIFEST).read_bytes(), ContentType="application/json")
    for rel in orphans:                       # manifest 已切換後才刪，讀者不會指到不存在的檔
        s3.delete_object(Bucket=b, Key=f"{PREFIX}/{rel}")
    print("發布完成")


def status():
    man = load_manifest()
    print(f"{STAGE}  generated_at {man.get('generated_at')}")
    tot = 0
    for n, e in sorted(man["datasets"].items()):
        sz = sum(f["size"] for f in e["files"].values())
        tot += sz
        print(f"  {n:28s} {e['rows']:>12,} 筆 {len(e['files']):>3} 檔 {sz/1e6:>8.1f} MB  {e['min_date']} ~ {e['max_date']}")
    print(f"  合計 {tot/1e6:,.0f} MB")


def daily():
    """每日排程：增量更新 → build → verify → upload。任何一步失敗都發 Telegram（主 chat）並以非 0 結束。"""
    sys.path.insert(0, str(ROOT))
    import notify
    import traceback
    step = "update"
    try:
        sys.path.insert(0, str(ROOT / "lake"))
        import update
        st = update.run()
        step = "build"
        build([])
        step = "verify"
        if not verify():
            raise RuntimeError("verify 未通過")
        env = load_env()
        if all(env.get(k) for k in ("R2_ACCOUNT_ID", "R2_WRITE_ACCESS_KEY_ID", "R2_WRITE_SECRET_ACCESS_KEY")):
            step = "upload"
            upload()
        else:
            print("未設定 R2 token → 只更新本機，不上傳")
        man = load_manifest()
        print(f"daily 完成：price 至 {man['datasets']['price']['max_date']}；" + "；".join(f"{k} {v}" for k, v in list(st.items())[:4]))
    except Exception as e:
        msg = f"⚠️ 資料湖 daily 失敗（{step}）：{e}"
        print(msg)
        traceback.print_exc()
        notify.send(msg)
        sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "verify", "upload", "status", "daily"])
    ap.add_argument("names", nargs="*")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    {"build": lambda: build(a.names), "verify": verify, "upload": lambda: upload(a.dry_run), "status": status,
     "daily": daily}[a.cmd]()
