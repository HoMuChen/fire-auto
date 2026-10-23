"""
自建台指 VIX（2021 起）— 用 FinMind TaiwanOptionDaily 的 TXO 每日結算價，照 CBOE VIX 公式重算。

為何自建：期交所免費下載的台指 VIX 只給近 3 個月，回測網格濾網需要 2021 起的日序列。
資料：只取日盤（trading_session=position）、月選（contract_date 6 碼，不含週選）。
公式（CBOE VIX）：
  每個到期 T：F = K* + e^{rT}(C−P)（K* = |C−P| 最小的履約價）、K0 = ≤F 的最大履約價，
  σ² = 2/T Σ ΔK/K² e^{rT} Q(K) − (F/K0 − 1)²/T，Q = K<K0 用 Put、K>K0 用 Call、K0 取平均。
  取近月/次月（剩餘 ≥ MIN_DAYS 天）兩個到期，內插到 30 天 → VIX = 100√(σ²₃₀)。
  價格用每日結算價（無 bid/ask），到期日 = 該合約在資料中最後出現日（未到期者用第三個週三）。

用法：
  python3 research/txo_vix.py fetch          # 增量抓 TXO 月選日資料 → data/futures/TXO_daily.csv
  python3 research/txo_vix.py build          # 算 VIX → data/futures/TXO_vix_daily.csv
  python3 research/txo_vix.py check          # 與期交所官方近3月每日收盤 VIX 比對
"""
import csv
import datetime as dtm
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

BASE = Path(__file__).parent.parent
RAW = BASE / "data" / "futures" / "TXO_daily.csv"
OUT = BASE / "data" / "futures" / "TXO_vix_daily.csv"
START = "2021-01-01"
R = 0.015            # 無風險利率（影響極小）
MIN_DAYS = 7         # 近月剩餘天數 < 此值改用次月/次次月（避開到期週雜訊）
FIELDS = ["date", "contract_date", "strike", "cp", "settle", "close", "volume"]


def token():
    for l in open(BASE / ".env.local"):
        if l.startswith("FINMIND_API_TOKEN"):
            return l.split("=", 1)[1].strip().strip('"')


def month_ranges(start, end):
    d = dtm.date.fromisoformat(start)
    while d <= end:
        nxt = (d.replace(day=28) + dtm.timedelta(days=4)).replace(day=1)
        yield d.isoformat(), min(nxt - dtm.timedelta(days=1), end).isoformat()
        d = nxt


def fetch():
    have = set()
    if RAW.exists():
        have = {r["date"] for r in csv.DictReader(open(RAW))}
    last = max(have) if have else None
    start = START if not last else last          # 從最後一天重抓（當天可能不完整）
    rows_old = [r for r in csv.DictReader(open(RAW))] if RAW.exists() else []
    rows_old = [r for r in rows_old if r["date"] < start]
    new = []
    tok = token()
    for s, e in month_ranges(start, dtm.date.today()):
        for attempt in range(3):
            try:
                j = requests.get("https://api.finmindtrade.com/api/v4/data",
                                 params=dict(dataset="TaiwanOptionDaily", data_id="TXO",
                                             start_date=s, end_date=e, token=tok),
                                 timeout=180).json()
                break
            except Exception as ex:
                print(f"  {s} 重試 {attempt+1}: {ex}"); time.sleep(5)
        d = [x for x in j.get("data", []) if x["trading_session"] == "position"
             and re.fullmatch(r"\d{6}", x["contract_date"])]
        for x in d:
            new.append(dict(date=x["date"], contract_date=x["contract_date"],
                            strike=x["strike_price"], cp=x["call_put"][0].upper(),
                            settle=x["settlement_price"], close=x["close"], volume=x["volume"]))
        print(f"  {s}~{e}: {len(d)} 筆（{j.get('msg')}）")
        time.sleep(1.0)
    with open(RAW, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS); w.writeheader()
        w.writerows(rows_old); w.writerows(new)
    print(f"寫入 {RAW}（{len(rows_old)+len(new)} 筆）")


def third_wednesday(ym):
    y, m = int(ym[:4]), int(ym[4:])
    d = dtm.date(y, m, 15)
    return d + dtm.timedelta(days=(2 - d.weekday()) % 7)


def term_variance(chain, T):
    """chain: {strike: {'C': px, 'P': px}} → (σ², F)；資料不足回 None"""
    ks = sorted(k for k, v in chain.items() if "C" in v and "P" in v)
    if len(ks) < 5:
        return None
    kstar = min(ks, key=lambda k: abs(chain[k]["C"] - chain[k]["P"]))
    F = kstar + math.exp(R * T) * (chain[kstar]["C"] - chain[kstar]["P"])
    below = [k for k in ks if k <= F]
    if not below:
        return None
    K0 = max(below)
    # 選價外序列；往外走遇到連續兩個 0 價就停（CBOE 規則，用結算價代 bid）
    sel = {K0: (chain[K0]["C"] + chain[K0]["P"]) / 2}
    for side, seq in (("P", [k for k in reversed(ks) if k < K0]), ("C", [k for k in ks if k > K0])):
        zeros = 0
        for k in seq:
            q = chain[k][side]
            if q <= 0:
                zeros += 1
                if zeros >= 2:
                    break
                continue
            zeros = 0
            sel[k] = q
    sk = sorted(sel)
    tot = 0.0
    for i, k in enumerate(sk):
        if i == 0:
            dk = sk[1] - sk[0]
        elif i == len(sk) - 1:
            dk = sk[-1] - sk[-2]
        else:
            dk = (sk[i + 1] - sk[i - 1]) / 2
        tot += dk / k ** 2 * math.exp(R * T) * sel[k]
    var = 2 / T * tot - (F / K0 - 1) ** 2 / T
    return var, F


def build():
    by = defaultdict(lambda: defaultdict(dict))   # (date, cd) -> strike -> {C,P}
    lastday = {}
    for r in csv.DictReader(open(RAW)):
        px = float(r["settle"])
        by[(r["date"], r["contract_date"])][float(r["strike"])][r["cp"]] = px
        lastday[r["contract_date"]] = max(lastday.get(r["contract_date"], ""), r["date"])
    today = max(lastday.values())
    expiry = {cd: (dtm.date.fromisoformat(ld) if ld < today else third_wednesday(cd))
              for cd, ld in lastday.items()}
    dates = sorted({d for d, _ in by})
    out = []
    for d in dates:
        D = dtm.date.fromisoformat(d)
        cds = sorted(cd for (dd, cd) in by if dd == d and (expiry[cd] - D).days >= MIN_DAYS)
        terms = []
        for cd in cds[:2]:
            days = (expiry[cd] - D).days
            res = term_variance(by[(d, cd)], days / 365)
            if res:
                terms.append((days, res[0], res[1], cd))
        if len(terms) < 2:
            continue
        (n1, v1, F1, c1), (n2, v2, _, c2) = terms
        T1, T2 = n1 / 365, n2 / 365
        w1 = (n2 - 30) / (n2 - n1)
        v30 = (T1 * v1 * w1 + T2 * v2 * (1 - w1)) * 365 / 30
        if v30 <= 0:
            continue
        out.append(dict(date=d, vix=round(100 * math.sqrt(v30), 2), near=c1, next=c2,
                        near_days=n1, fwd=round(F1, 1)))
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, ["date", "vix", "near", "next", "near_days", "fwd"])
        w.writeheader(); w.writerows(out)
    print(f"寫入 {OUT}（{len(out)} 天，{out[0]['date']} ~ {out[-1]['date']}）")


def check():
    """與期交所官方每日收盤 VIX（僅近 4 個月有檔）比對"""
    off = {}
    d = dtm.date.today().replace(day=1)
    for _ in range(4):
        url = f"https://www.taifex.com.tw/file/taifex/Dailydownload/vix/log2data/{d:%Y%m}new.txt"
        txt = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60).content.decode("big5", "ignore")
        for line in txt.splitlines():
            f = line.split()
            if len(f) >= 3 and re.fullmatch(r"\d{8}", f[0]):
                off[f"{f[0][:4]}-{f[0][4:6]}-{f[0][6:]}"] = float(f[2])
        d = (d - dtm.timedelta(days=1)).replace(day=1)
    mine = {r["date"]: float(r["vix"]) for r in csv.DictReader(open(OUT))}
    common = sorted(set(off) & set(mine))
    if not common:
        print("找不到可比對日期；官方解析到", len(off), "天"); return
    diffs = [mine[d] - off[d] for d in common]
    for d in common[-10:]:
        print(f"  {d}  官方 {off[d]:6.2f}  自建 {mine[d]:6.2f}  差 {mine[d]-off[d]:+.2f}")
    import statistics as st
    mo = [off[d] for d in common]; mm = [mine[d] for d in common]
    print(f"比對 {len(common)} 天：平均差 {st.mean(diffs):+.2f}、平均絕對差 "
          f"{st.mean(abs(x) for x in diffs):.2f}、相關 {st.correlation(mo, mm):.3f}")


if __name__ == "__main__":
    {"fetch": fetch, "build": build, "check": check}[sys.argv[1] if len(sys.argv) > 1 else "build"]()
