"""
買方分點集中度預算模組。

對每檔流動性個股、每個交易日，計算「買超前 5 大分點買進股數 ÷ 全分點總買進股數」
= 買方集中度（0~1）。集中度高 = 已有主力集中吃貨（擁擠）；低 = 籌碼分散、尚未被盯上。

輸出：data/broker_concentration.parquet
  欄位：date, stock_id, buy_conc, n_brokers, total_buy

用途：給三策略進場信號做「低集中度過濾」（保留 < 當日橫斷面中位數者）。
資料來源：data/broker/{stock_id}.parquet（chip backfill 已補完 2021-06-30 ~）

CLI:
  python3 broker_concentration.py build           # 重算全市場 -> parquet
  python3 broker_concentration.py median          # 印每日橫斷面中位數摘要
"""
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent
BROKER_DIR = BASE_DIR / "data" / "broker"
OUT_PATH = BASE_DIR / "data" / "broker_concentration.parquet"
STOCKS_PATH = BASE_DIR / "individual_stocks.json"
META_PATH = BASE_DIR / "data" / "broker_cache_meta.json"

TOP_N = 5


def _liquid_stock_ids() -> list[str]:
    """有 broker parquet 且非低流動性的個股。"""
    import json
    stocks = json.load(open(STOCKS_PATH))
    liquid = {s["stock_id"] for s in stocks if not s.get("low_liquidity")}
    have = {p.stem for p in BROKER_DIR.glob("*.parquet")}
    return sorted(liquid & have)


def _concentration_for_stock(stock_id: str, after: str = None) -> pd.DataFrame:
    """單檔：回傳 date, buy_conc, n_brokers, total_buy。after 指定只算晚於該日的資料（增量）。"""
    filters = [("date", ">", after)] if after else None
    df = pd.read_parquet(BROKER_DIR / f"{stock_id}.parquet", columns=["date", "buy"], filters=filters)
    df = df[df["buy"] > 0]
    if df.empty:
        return pd.DataFrame(columns=["date", "buy_conc", "n_brokers", "total_buy"])
    total = df.groupby("date")["buy"].sum()
    nbrk = df.groupby("date")["buy"].size()
    top = (
        df.sort_values(["date", "buy"], ascending=[True, False])
        .groupby("date")
        .head(TOP_N)
        .groupby("date")["buy"]
        .sum()
    )
    out = pd.DataFrame({"total_buy": total, "top_buy": top, "n_brokers": nbrk})
    out["buy_conc"] = out["top_buy"] / out["total_buy"]
    out = out.reset_index()
    out["stock_id"] = stock_id
    return out[["date", "stock_id", "buy_conc", "n_brokers", "total_buy"]]


def build(progress: bool = True) -> pd.DataFrame:
    ids = _liquid_stock_ids()
    frames = []
    for i, sid in enumerate(ids, 1):
        try:
            frames.append(_concentration_for_stock(sid))
        except Exception as e:  # noqa: BLE001
            print(f"  WARN {sid}: {e}")
        if progress and i % 100 == 0:
            print(f"  [{i}/{len(ids)}] {sid}")
    allc = pd.concat(frames, ignore_index=True)
    allc.to_parquet(OUT_PATH, index=False)
    print(f"完成！{len(ids)} 檔，{len(allc):,} 筆 (股,日) -> {OUT_PATH}")
    return allc


def load() -> pd.DataFrame:
    return pd.read_parquet(OUT_PATH)


def daily_median(df: pd.DataFrame = None) -> pd.Series:
    """每個交易日，全市場買方集中度的橫斷面中位數。"""
    if df is None:
        df = load()
    return df.groupby("date")["buy_conc"].median()


def _latest_broker_date() -> str | None:
    """broker 資料最新日（取 chip meta 的最大 end）。"""
    import json
    if not META_PATH.exists():
        return None
    meta = json.load(open(META_PATH))
    ends = [v.get("end") for v in meta.values() if isinstance(v, dict) and v.get("end")]
    return max(ends) if ends else None


def ensure_current(verbose: bool = False) -> str | None:
    """把 broker 新增的交易日增量補進集中度表（自包含，不依賴 cron）。回傳表中最新日。"""
    have = load() if OUT_PATH.exists() else None
    have_max = have["date"].max() if have is not None and len(have) else None
    latest = _latest_broker_date()
    if latest is None:
        return have_max
    if have_max is not None and have_max >= latest:
        return have_max  # 已是最新
    ids = _liquid_stock_ids()
    frames = []
    for sid in ids:
        try:
            d = _concentration_for_stock(sid, after=have_max)  # 只算新日
            if len(d):
                frames.append(d)
        except Exception:
            continue
    if not frames:
        return have_max
    new = pd.concat(frames, ignore_index=True)
    combined = pd.concat([have, new], ignore_index=True) if have is not None else new
    combined = combined.drop_duplicates(["stock_id", "date"], keep="last")
    combined.to_parquet(OUT_PATH, index=False)
    if verbose:
        print(f"  集中度表增量更新：{have_max} → {combined['date'].max()}（+{len(new):,} 筆）")
    return combined["date"].max()


def annotate_signals(stock_ids: list[str], date: str, df: pd.DataFrame = None) -> dict:
    """回傳 {stock_id: {"conc":x, "pct":百分位, "high":bool}}；找不到當日資料回 None。

    pct = 當日全市場集中度的百分位（0~100，越高=越集中）；high = conc > 當日全市場中位數。
    """
    if df is None:
        df = load()
    day = df[df["date"] == date]
    if day.empty:
        return {sid: None for sid in stock_ids}
    med = day["buy_conc"].median()
    concs = day["buy_conc"].to_numpy()
    n = len(concs)
    lookup = dict(zip(day["stock_id"], day["buy_conc"]))
    out = {}
    for sid in stock_ids:
        c = lookup.get(sid)
        if c is None:
            out[sid] = None
            continue
        pct = (concs <= c).sum() / n * 100
        out[sid] = {"conc": float(c), "pct": float(pct), "high": bool(c > med)}
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        build()
    elif cmd == "update":
        print("最新日：", ensure_current(verbose=True))
    elif cmd == "median":
        med = daily_median()
        print(f"交易日數: {len(med)}")
        print(f"集中度中位數 全期均值: {med.mean():.3f}  最小 {med.min():.3f}  最大 {med.max():.3f}")
        print(med.tail(10).to_string())
    else:
        print(__doc__)
