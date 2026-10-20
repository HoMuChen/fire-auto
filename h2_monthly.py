"""H2「52 週高點接近度」每月選股（實盤執行版 20 檔，規則見 research/v2/HYPOTHESES.md）。

規則（凍結，不可改）：
  換股日 = 每月第一個 ≥12 日的交易日，開盤成交
  股票池 = 上市/上櫃普通股、近 60 交易日均量 ≥ 200 張、已有 ≥ 250 根 K
  分數   = 前一交易日還原收盤 / 近 252 交易日最高還原收盤；取最高 20 檔等權

排程：平日 20:30 跑（避開 19:00 籌碼更新的 API 尖峰）。只有「下一個平日 ≥12 日且本月還沒換過」時才算並發
Telegram（只發 TELEGRAM_CHAT_ID＝早晨摘要收件人，不發盤中名單）。遇假日則清單順延到下一個交易日開盤，仍有效。

用法：
  python3 h2_monthly.py                  # 排程用：非換股前夕直接結束
  python3 h2_monthly.py --force          # 不管日期，立刻算（不更新狀態）
  python3 h2_monthly.py --force --no-send
狀態：data/h2_state.json（每月選股結果 + 總資金設定 capital，設了才會算每檔張數）
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "research"))
from build_data_v2 import NON_STOCK, RateLimiter, api  # noqa: E402

TOPN, LIQ_SHARES, LIQ_WIN, MIN_BARS, HI_WIN = 20, 200_000, 60, 250, 252
STATE = BASE / "data/h2_state.json"


def next_weekday(d):
    d += timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {"capital": None, "months": {}}


def universe(lim):
    info = api(lim, dataset="TaiwanStockInfo")
    ok = info.stock_id.str.fullmatch(r"[1-9]\d{3}") & ~info.industry_category.isin(NON_STOCK) & info.type.isin(["twse", "tpex"])
    u = info[ok].drop_duplicates("stock_id")
    return dict(zip(u.stock_id, u.stock_name))


def recent_raw(lim, today):
    """近 ~90 個日曆天全市場原始價（逐日 1 call），算流動性與今日收盤。"""
    days = [(today - timedelta(days=k)).isoformat() for k in range(95, -1, -1)]
    days = [d for d in days if date.fromisoformat(d).weekday() < 5]
    with ThreadPoolExecutor(4) as ex:
        dfs = list(ex.map(lambda d: api(lim, dataset="TaiwanStockPrice", start_date=d, end_date=d), days))
    df = pd.concat([d for d in dfs if d is not None and len(d)], ignore_index=True)
    df = df[df.close > 0]
    df["date"] = pd.to_datetime(df.date)
    return df


def score_one(lim, sid, today_close, start):
    a = api(lim, dataset="TaiwanStockPriceAdj", data_id=sid, start_date=start)
    if a is None or not len(a):
        return None
    a = a[a.close > 0].assign(date=lambda x: pd.to_datetime(x.date)).set_index("date").close
    if today_close is not None and today_close[0] > a.index[-1]:
        a.loc[today_close[0]] = today_close[1]          # 還原價最新一根＝原始價（除權息當天例外，影響極小）
    if len(a) < MIN_BARS:
        return None
    return float(a.iloc[-1] / a.iloc[-HI_WIN:].max()), len(a)


def pick(today):
    lim = RateLimiter(80)
    names = universe(lim)
    raw = recent_raw(lim, today)
    last_day = raw.date.max()
    vol = raw.pivot_table(index="date", columns="stock_id", values="Trading_Volume").sort_index()
    liq = vol.iloc[-LIQ_WIN:].mean()
    cand = [s for s in names if liq.get(s, 0) >= LIQ_SHARES]
    close_today = raw[raw.date == last_day].set_index("stock_id").close
    start = (today - timedelta(days=int(MIN_BARS * 1.6) + 30)).isoformat()   # 夠 ≥250 根 K 判斷 + 252 日高
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(lambda s: (s, score_one(lim, s, (last_day, close_today.get(s)) if s in close_today else None, start)), cand))
    sc = pd.DataFrame([(s, r[0], r[1]) for s, r in res if r], columns=["sid", "score", "bars"]).set_index("sid")
    sc = sc[sc.bars >= MIN_BARS]
    sc = sc.reset_index().sort_values(["score", "sid"], ascending=[False, True])   # 平手依代號小→大（與回測 nlargest 一致）
    top = sc.set_index("sid").score.head(TOPN)
    n_tie = int((sc.score >= top.iloc[-1] - 1e-9).sum()) if len(top) else 0
    return top, names, close_today, last_day, len(cand), n_tie


def message(top, prev, names, close_today, last_day, n_pool, capital, target_day, n_tie=0):
    keep = [s for s in top.index if s in prev]
    buy = [s for s in top.index if s not in prev]
    sell = [s for s in prev if s not in top.index]
    per = capital / TOPN if capital else None

    def line(s, show_qty):
        px = close_today.get(s)
        t = f"{s} {names.get(s, '')}  {top.get(s, float('nan')):.3f}" if s in top.index else f"{s} {names.get(s, '')}"
        if px:
            t += f"  收{px:g}"
            if show_qty and per:
                sh = int(per // px)
                t += f"  ≈{sh // 1000}張{sh % 1000}股" if sh >= 1000 else f"  ≈{sh}股(零股)"
        return t

    out = [f"📈 H2 52週高點 月換股（{target_day:%m/%d} 週{'一二三四五'[target_day.weekday()]} 開盤；遇假日順延）",
           f"資料至 {last_day:%m/%d} 收盤｜股票池 {n_pool} 檔｜取前 {TOPN}" + (f"｜每檔目標 {per:,.0f} 元" if per else "｜每檔 1/20 資金")]
    if n_tie > TOPN:
        out.append(f"（{n_tie} 檔並列最低入選分數，依代號小→大取，與回測一致）")
    if not prev:
        out += ["", f"【建立部位 {len(buy)} 檔】"] + [line(s, True) for s in buy]
    else:
        out += ["", f"【賣出 {len(sell)}】"] + ([line(s, False) for s in sell] or ["（無）"])
        out += ["", f"【買進 {len(buy)}】"] + ([line(s, True) for s in buy] or ["（無）"])
        out += ["", f"【續抱 {len(keep)}】（偏離目標 ±30% 以上再調）"] + [line(s, True) for s in keep]
    out += ["", "規則：前日收盤/252日最高，無停損；分數=1.000 即收在一年新高"]
    return "\n".join(out)


def main():
    force, no_send = "--force" in sys.argv, "--no-send" in sys.argv
    today = date.today()
    target = next_weekday(today)
    st = load_state()
    month = target.strftime("%Y-%m")
    if not force and (target.day < 12 or month in st["months"]):
        return
    top, names, close_today, last_day, n_pool, n_tie = pick(today)
    prev_m = sorted(m for m in st["months"] if m < month)
    prev = st["months"][prev_m[-1]]["picks"] if prev_m else []
    msg = message(top, prev, names, close_today, last_day, n_pool, st.get("capital"), target, n_tie)
    print(msg)
    if not force:
        st["months"][month] = {"picks": list(top.index), "scores": top.round(4).to_dict(),
                               "data_date": f"{last_day:%Y-%m-%d}", "created": datetime.now().isoformat(timespec="seconds")}
        STATE.write_text(json.dumps(st, ensure_ascii=False, indent=1))
    if not no_send:
        from notify import send
        send(msg)          # 只發 TELEGRAM_CHAT_ID（早晨摘要收件人），不發盤中名單


if __name__ == "__main__":
    main()
