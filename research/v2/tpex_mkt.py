"""上櫃市場整體三大法人買賣金額 / 融資融券餘額：櫃買中心官網 JSON（FinMind 的 TaiwanStockTotal* 只有上市）。

輸出（欄位、name、單位與上市的 mkt_inst / mkt_margin 相同，可直接相加或對照）：
  data/v2/futures/mkt_inst_otc.parquet    date, name, buy, sell（元）
      name：Foreign_Investor（外資及陸資，不含外資自營商）、Foreign_Dealer_Self（外資自營商，2018 起才拆出）、
            Investment_Trust、Dealer（2014 前自營商未拆）/ Dealer_self + Dealer_Hedging、total（三大法人合計，不含外資自營商）
      口徑：「上櫃有價證券」（prod=0：股票＋權證＋ETF＋ETN＋TDR 等，同證交所 BFI82U）。
  data/v2/futures/mkt_margin_otc.parquet  date, name, YesBalance, buy, sell, Return, TodayBalance
      name：MarginPurchase（融資，張）、MarginPurchaseMoney（融資金額，元）、ShortSale（融券，張；buy＝券買回補、sell＝券賣）
      口徑：櫃買「上櫃股票融資融券餘額」表的合計列（含可信用交易的 ETF）。
休市日兩張表都回傳空 → 不寫該日。2017 年部分日子官網沒有合計列 → 以各法人加總補（同官網：不含外資自營商）。

用法：/usr/local/bin/python3 research/v2/tpex_mkt.py backfill [起日 2008-01-01] [迄日 今天]   （斷點續抓：已有的日期跳過）
      每日更新由 lake/update.py 呼叫 update(today)。
"""
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

FD = Path(__file__).resolve().parents[2] / "data/v2/futures"
INST_F, MARGIN_F = FD / "mkt_inst_otc.parquet", FD / "mkt_margin_otc.parquet"
BASE = "https://www.tpex.org.tw/www/zh-tw"
UA = {"User-Agent": "Mozilla/5.0 (fire-auto research)"}
SLEEP = 0.4          # 每次請求間隔（秒），對官網客氣一點
INST_NAME = {        # 官網列名（去全形空白）→ mkt_inst 的 name；None＝小計列，不收
    "外資": "Foreign_Investor", "外資及陸資": "Foreign_Investor", "外資及陸資(不含自營商)": "Foreign_Investor",
    "外資自營商": "Foreign_Dealer_Self", "外資及陸資合計": None,
    "投信": "Investment_Trust",
    "自營商": "Dealer", "自營商合計": None, "自營商(自行買賣)": "Dealer_self", "自營商(避險)": "Dealer_Hedging",
    "合計": "total", "三大法人合計*": "total", "三大法人合計": "total",
}
MARGIN_COLS = ["date", "name", "YesBalance", "buy", "sell", "Return", "TodayBalance"]


def _get(path, **params):
    for i in range(4):
        try:
            r = requests.get(f"{BASE}/{path}", params={**params, "response": "json"}, headers=UA, timeout=30)
            r.raise_for_status()
            j = r.json()
            if j.get("stat") == "ok":
                return j
        except Exception:
            pass
        time.sleep(5 * (i + 1))
    raise RuntimeError(f"櫃買 {path} {params} 抓取失敗")


def _num(s):
    return int(str(s).replace(",", "").strip() or 0)


def fetch_inst(d):
    """某日上櫃三大法人買賣金額 → DataFrame（休市回傳空表）。"""
    j = _get("insti/summary", type="Daily", date=f"{d:%Y/%m/%d}", prod=0)
    rows = {}
    for r in j["tables"][0]["data"]:
        if len(r) < 3:
            continue
        key = r[0].replace("　", "").strip()
        if key not in INST_NAME:
            raise ValueError(f"{d} 未知的法人列：{r[0]!r}")
        if INST_NAME[key]:
            rows[INST_NAME[key]] = (_num(r[1]), _num(r[2]))
    if not rows:
        return pd.DataFrame(columns=["date", "name", "buy", "sell"])
    if "Dealer_self" in rows:                     # 有拆自營商時，同 mkt_inst 不留自營商合計
        rows.pop("Dealer", None)
    if "total" not in rows:                       # 2017 部分日子沒有合計列
        parts = [v for k, v in rows.items() if k != "Foreign_Dealer_Self"]
        rows["total"] = (sum(b for b, _ in parts), sum(s for _, s in parts))
    return pd.DataFrame([(d.isoformat(), k, b, s) for k, (b, s) in rows.items()], columns=["date", "name", "buy", "sell"])


def fetch_margin(d):
    """某日上櫃融資融券合計 → DataFrame（休市回傳空表）。"""
    j = _get("margin/balance", date=f"{d:%Y/%m/%d}")
    summ = {r[1]: r for t in j.get("tables", []) for r in (t.get("summary") or []) if len(r) > 14}
    if "合計(張)" not in summ:
        return pd.DataFrame(columns=MARGIN_COLS)
    s, m = summ["合計(張)"], summ.get("融資金(仟元)")
    out = [(d.isoformat(), "MarginPurchase", *(_num(s[i]) for i in (2, 3, 4, 5, 6))),      # 前資餘額、資買、資賣、現償、資餘額
           (d.isoformat(), "ShortSale", _num(s[10]), _num(s[12]), _num(s[11]), _num(s[13]), _num(s[14]))]  # 前券、券買、券賣、券償、券餘額
    if m is not None:
        out.insert(1, (d.isoformat(), "MarginPurchaseMoney", *(_num(m[i]) * 1000 for i in (2, 3, 4, 5, 6))))
    return pd.DataFrame(out, columns=MARGIN_COLS)


def _read(f):
    return pd.read_parquet(f) if f.exists() else pd.DataFrame()


def _write(f, old, new):
    """以日期覆蓋：舊檔去掉 new 有的日期 → 接上 → 依日期、原順序排序；內容沒變不寫。回傳覆蓋天數。"""
    if not len(new):
        return 0
    out = pd.concat([old[~old.date.isin(set(new.date))], new], ignore_index=True) if len(old) else new
    out = out.sort_values("date", kind="stable").reset_index(drop=True)
    for c in out.columns:
        if c not in ("date", "name"):
            out[c] = out[c].astype("int64")
    if len(old) and len(out) == len(old) and out.equals(old.reset_index(drop=True)):
        return 0
    tmp = f.with_name(f".{f.name}.tmp")
    out.to_parquet(tmp, index=False)
    tmp.replace(f)
    return new.date.nunique()


def run(a, b, skip_existing, dry=False, log=print):
    """抓 a~b 的平日；skip_existing＝兩張表都已有的日期跳過（回補用）。回傳 (inst 天數, margin 天數)。"""
    oi, om = _read(INST_F), _read(MARGIN_F)
    have = set(oi.date) & set(om.date) if skip_existing and len(oi) and len(om) else set()
    days = [a + timedelta(n) for n in range((b - a).days + 1)]
    days = [d for d in days if d.weekday() < 5 and d.isoformat() not in have]
    bi, bm, ni, nm = [], [], 0, 0
    for k, d in enumerate(days, 1):
        bi.append(fetch_inst(d))
        time.sleep(SLEEP)
        bm.append(fetch_margin(d))
        time.sleep(SLEEP)
        if k % 250 == 0 or k == len(days):           # 每 250 天落地一次，中斷也不白抓
            new_i, new_m = pd.concat(bi, ignore_index=True), pd.concat(bm, ignore_index=True)
            if not dry:
                ni += _write(INST_F, oi, new_i)
                nm += _write(MARGIN_F, om, new_m)
                oi, om = _read(INST_F), _read(MARGIN_F)
            bi, bm = [], []
            log(f"  tpex {d}：{k}/{len(days)} 天")
    return ni, nm


def update(today, overlap_days=14, dry=False):
    """每日更新：最後日期往回 overlap_days 天重抓到昨天為止（官網偶有事後修正）。"""
    old = _read(INST_F)
    last = date.fromisoformat(old.date.max()) if len(old) else date(2008, 1, 1)
    ni, nm = run(last - timedelta(days=overlap_days), today, skip_existing=False, dry=dry, log=lambda *_: None)
    mx = _read(INST_F).date.max()
    return f"至 {mx[5:]}，inst {ni} / margin {nm} 天覆蓋"


if __name__ == "__main__":
    if sys.argv[1] == "backfill":
        a = date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else date(2008, 1, 1)
        b = date.fromisoformat(sys.argv[3]) if len(sys.argv) > 3 else date.today()
        print(run(a, b, skip_existing=True))
