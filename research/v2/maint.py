"""
自建「大盤融資維持率」日序列（2009 起）→ data/v2/futures/mkt_maint.parquet

定義（近似交易所公布的整體市場融資維持率）：
  維持率 = Σ_個股（融資餘額張 × 1000 × 未還原收盤價）/ 全市場融資餘額金額
  分子 = 融資擔保品市值（data/v2/margin/{id}.parquet 的 MarginPurchaseTodayBalance × data/v2/raw 收盤）
  分母 = FinMind TaiwanStockTotalMarginPurchaseShortSale 的 MarginPurchaseMoney（data/v2/futures/mkt_margin.parquet）
  未納入融券擔保品與保證金（官方公式有，影響 <5 個百分點），故水準略低於官方；用於「相對自身歷史」的訊號即可。
  融資成數六成 → 新買進維持率 ≈ 166%；交易所整體維持率常態 160~170%，130~140% 為斷頭潮區。

防前視：融資餘額與收盤皆當日盤後公布，訊號只能用於 t+1 開盤。

覆蓋率：FinMind 全市場總額只含上市；v2 股票池只有普通股，上市 ETF / 存託憑證 / ETN / 特別股 的融資不在裡面
（2010 年佔 1%，2026 年佔 32%，主因 ETF），故 `fetch` 另抓這些非普通股的融資與價格到 data/v2/margin_x、raw_x。

用法：python3 research/v2/maint.py fetch   # 補抓上市非普通股（ETF 等）融資與價格
      python3 research/v2/maint.py build   # 建序列
      python3 research/v2/maint.py show    # 列出極值與關鍵日期
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent.parent
D = BASE / "data" / "v2"
OUT = D / "futures" / "mkt_maint.parquet"
SKIP_CAT = {"Index", "大盤"}


def fetch():
    sys.path.insert(0, str(BASE / "research"))
    from build_data_v2 import COLS, RateLimiter, api  # noqa: E402
    lim = RateLimiter(85)
    info = api(lim, dataset="TaiwanStockInfo")
    u = pd.read_csv(D / "universe.csv", dtype=str)
    tw = info[(info.type == "twse") & ~info.stock_id.isin(u.stock_id) & ~info.industry_category.isin(SKIP_CAT)]
    ids = sorted(set(tw.stock_id))
    (D / "margin_x").mkdir(exist_ok=True); (D / "raw_x").mkdir(exist_ok=True)
    print(f"上市非普通股 {len(ids)} 檔", flush=True)

    def one(sid):
        f = D / "margin_x" / f"{sid}.parquet"
        if not f.exists():
            m = api(lim, dataset="TaiwanStockMarginPurchaseShortSale", data_id=sid, start_date="2009-01-01")
            if m is None:
                return "fail"
            m.to_parquet(f, index=False)
        m = pd.read_parquet(f)
        if len(m) == 0 or "MarginPurchaseTodayBalance" not in m or m.MarginPurchaseTodayBalance.sum() == 0:
            return "no_margin"
        pf = D / "raw_x" / f"{sid}.parquet"
        if not pf.exists():
            p = api(lim, dataset="TaiwanStockPrice", data_id=sid, start_date="2009-01-01")
            if p is None:
                return "fail"
            p.rename(columns=COLS).to_parquet(pf, index=False)
        return "ok"

    stats = {}
    with ThreadPoolExecutor(6) as ex:
        for i, r in enumerate(ex.map(one, ids)):
            stats[r] = stats.get(r, 0) + 1
            if i % 100 == 0:
                print(f"[{i}/{len(ids)}] {stats}", flush=True)
    print("完成", stats, flush=True)


def build():
    mm = pd.read_parquet(D / "futures" / "mkt_margin.parquet")
    money = (mm[mm.name == "MarginPurchaseMoney"].assign(date=lambda x: pd.to_datetime(x.date))
             .drop_duplicates("date").set_index("date").TodayBalance.astype(float).sort_index())
    lots = (mm[mm.name == "MarginPurchase"].assign(date=lambda x: pd.to_datetime(x.date))
            .drop_duplicates("date").set_index("date").TodayBalance.astype(float).sort_index())
    idx = money.index
    collat = pd.Series(0.0, index=idx)
    covered = pd.Series(0.0, index=idx)   # 有納入計算的融資張數（檢查覆蓋率）
    u = pd.read_csv(D / "universe.csv", dtype=str).set_index("stock_id")
    n = 0
    files = [(f, D / "raw" / f.name) for f in sorted((D / "margin").glob("*.parquet"))
             if u.type.get(f.stem) in ("twse", "delisted")]          # 分母只含上市 → 分子只用上市普通股
    files += [(f, D / "raw_x" / f.name) for f in sorted((D / "margin_x").glob("*.parquet"))]
    for f, pf in files:
        if not pf.exists():
            continue
        m = pd.read_parquet(f)
        if "date" not in m.columns or len(m) == 0:
            continue
        m = m[["date", "MarginPurchaseTodayBalance"]].assign(date=pd.to_datetime(m.date)).drop_duplicates("date").set_index("date")
        bal = m.MarginPurchaseTodayBalance.astype(float).reindex(idx)
        if bal.fillna(0).abs().sum() == 0:
            continue
        p = pd.read_parquet(pf)
        if "date" not in p.columns or len(p) == 0:
            continue
        p = p[["date", "close"]].assign(date=pd.to_datetime(p.date)).drop_duplicates("date").set_index("date").close.astype(float)
        p = p.replace(0, np.nan).reindex(idx).ffill()   # 停牌日 close=0 → 用前收
        v = (bal * 1000 * p).fillna(0)
        collat += v
        covered += bal.where(p.notna(), 0).fillna(0)
        n += 1
    out = pd.DataFrame({"collateral": collat, "money": money, "lots_total": lots, "lots_covered": covered})
    out["coverage"] = out.lots_covered / out.lots_total
    out["maint"] = out.collateral / out.money * 100
    out = out[out.money > 0]
    out.to_parquet(OUT)
    print(f"寫入 {OUT}：{len(out)} 天 {out.index.min().date()}~{out.index.max().date()}，{n} 檔；"
          f"張數覆蓋率 中位 {out.coverage.median():.3f} 最低 {out.coverage.min():.3f}")
    return out


def show():
    o = pd.read_parquet(OUT)
    tx = pd.read_parquet(D / "futures" / "taiex.parquet")
    tx = tx.assign(date=pd.to_datetime(tx.date)).drop_duplicates("date").set_index("date").close.astype(float)
    o["taiex"] = tx.reindex(o.index)
    print(o.maint.describe().round(1).to_string())
    print("\n各年：最低 / 最高 維持率 與日期")
    for y, g in o.groupby(o.index.year):
        lo, hi = g.maint.idxmin(), g.maint.idxmax()
        print(f"  {y}: 低 {g.maint.min():6.1f}% ({lo.date()}, 加權 {g.taiex[lo]:.0f})  "
              f"高 {g.maint.max():6.1f}% ({hi.date()})  覆蓋 {g.coverage.mean():.2f}")
    print("\n關鍵日期")
    for d in ["2008-11-20", "2011-12-19", "2015-08-24", "2018-10-26", "2020-03-19", "2022-10-25",
              "2024-08-05", "2025-04-09", "2026-10-20"]:
        d = pd.Timestamp(d)
        if d in o.index:
            print(f"  {d.date()}  維持率 {o.maint[d]:6.1f}%  加權 {o.taiex[d]:.0f}")
    print("\n最近 10 天")
    print(o[["maint", "taiex"]].tail(10).round(1).to_string())


if __name__ == "__main__":
    {"fetch": fetch, "build": build, "show": show}[sys.argv[1] if len(sys.argv) > 1 else "build"]()
