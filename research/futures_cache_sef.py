"""小型智邦期貨（SEF）tick → 日盤 30 分 K 本地快取

沿用 futures_cache.py 的聚合邏輯（近月連續、日盤 08:45~13:45、30 分桶、排除價差合約），
僅將 data_id 改為 SEF、輸出改為 SEF_30min_day.csv。

注意：小型智邦期貨於 2024-02 才上市（不像 QFF 可回溯到 2021），故首次回補從 2024-01-01 起
（更早無資料，空日不寫入）。日盤量約 3,000 口/日，比 QFF 薄約 15 倍。

用法：
    python3 research/futures_cache_sef.py                      # 增量：從既有最後日補到今天
    python3 research/futures_cache_sef.py 2024-01-01 2026-09-01  # 指定範圍（覆寫重建）
"""
import csv
import sys
import time
from datetime import datetime, date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import futures_cache as fc

fc.FUTURES_ID = "SEF"
OUT = fc.BASE / "data" / "futures" / "SEF_30min_day.csv"
fc.OUT = OUT
FIRST_START = "2024-01-01"


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tok = fc.token()

    args = sys.argv[1:]
    if len(args) >= 2:
        start, end = args[0], args[1]
        mode = "w"                          # 指定範圍 → 覆寫重建
    else:
        last = fc.last_cached_date()
        start = (datetime.strptime(last, "%Y-%m-%d").date() + timedelta(days=1)).isoformat() \
            if last else FIRST_START
        end = date.today().isoformat()
        mode = "a"                          # 無參數 → 增量接續

    new_file = mode == "w" or not OUT.exists()
    d0 = datetime.strptime(start, "%Y-%m-%d").date()
    d1 = datetime.strptime(end, "%Y-%m-%d").date()
    print(f"補 SEF 日盤 30 分 K：{start} ~ {end}（{'重建' if mode=='w' else '增量'}）")

    with open(OUT, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fc.FIELDS)
        if new_file:
            w.writeheader()
        d = d0
        dw = bt = errs = 0
        while d <= d1:
            if d.weekday() >= 5:
                d += timedelta(days=1)
                continue
            ds = d.isoformat()
            try:
                rows = fc.bars_for_day(fc.fetch_ticks(ds, tok))
                if rows:
                    w.writerows(rows)
                    f.flush()
                    dw += 1
                    bt += len(rows)
            except Exception as e:
                errs += 1
                if errs <= 8:
                    print(f"  {ds}: {e}")
            if d.day == 1:
                print(f"  ...{ds}  交易日{dw} K{bt} 錯{errs}")
            time.sleep(0.5)
            d += timedelta(days=1)

    print(f"完成：交易日 {dw}，30分K {bt} 根，錯誤 {errs}\n輸出：{OUT}")


if __name__ == "__main__":
    main()
