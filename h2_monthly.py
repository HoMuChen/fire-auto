"""H29「雙新高」每月選股（實盤執行版 20 檔，規則見 research/v2/HYPOTHESES.md；2026-10-28 由 H2 換成 H29）。

規則（凍結，不可改）：
  換股日 = 每月第一個 ≥12 日的交易日，開盤成交
  股票池 = 上市/上櫃普通股、近 60 交易日均量 ≥ 200 張、已有 ≥ 250 根 K
  H2     = 前一交易日還原收盤 / 近 252 交易日最高還原收盤（252 日內 ≥ 200 根才有值）
  H10    = 近 3 個月營收合計 / 12 個月內任一 3 個月合計最高（營收 m 月視為 m+1 月 11 日可得、超過 75 天未更新視為缺）
  分數   = H2、H10 各自在全上市櫃普通股（不限流動性，與回測 xs 一致）的百分位，平均；取股票池中最高 20 檔等權
  --h2   = 舊 H2 規則（只看股價分數），供對照

排程：平日 20:30 跑（避開 19:00 籌碼更新的 API 尖峰）。只有「下一個平日是本月第 1~3 個 ≥12 日的平日、且本月還沒換過」
時才算並發 Telegram（只發 TELEGRAM_CHAT_ID＝早晨摘要收件人，不發盤中名單）。第 2~3 個只是前一晚失敗時的補發；
月中不會發（2026-10-21 曾因狀態檔不存在而在月中誤發）。遇假日則清單順延到下一個交易日開盤，仍有效。

用法：
  python3 h2_monthly.py                  # 排程用：非換股前夕直接結束
  python3 h2_monthly.py --force          # 不管日期，立刻算（不更新狀態）
  python3 h2_monthly.py --force --no-send
  python3 h2_monthly.py --force --no-send --asof 2026-10-08   # 用指定日收盤重算（驗證與回測一致）
  python3 h2_monthly.py --force --no-send --h2                # 舊 H2 規則對照
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


def is_rebalance_eve(target, window=3):
    """target（下一個平日）是否為本月第 1~window 個 ≥12 日的平日。"""
    if target.day < 12:
        return False
    d, k = target.replace(day=12), 0
    while d <= target:
        k += d.weekday() < 5
        d += timedelta(days=1)
    return k <= window


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
    df = df[(df.close > 0) & (df.date <= today.isoformat())]
    df["date"] = pd.to_datetime(df.date)
    return df


def adj_close(lim, sid, today_close, start, today):
    a = api(lim, dataset="TaiwanStockPriceAdj", data_id=sid, start_date=start, end_date=today.isoformat())
    if a is None or not len(a):
        return None
    a = a[a.close > 0].assign(date=lambda x: pd.to_datetime(x.date)).set_index("date").close
    if not len(a):
        return None
    if today_close is not None and today_close[0] > a.index[-1]:
        a.loc[today_close[0]] = today_close[1]          # 還原價最新一根＝原始價（除權息當天例外，影響極小）
    return a


def rev_high(lim, last_day):
    """全市場 H10（近 3 月營收 / 12 個月內最高 3 月），只用 last_day 當時已可得的月份。"""
    first = pd.Period(last_day, "M")
    dates = [(first - k).to_timestamp().date().isoformat() for k in range(17, -1, -1)]   # 營收資料 date＝次月 1 日
    with ThreadPoolExecutor(4) as ex:
        dfs = list(ex.map(lambda d: api(lim, dataset="TaiwanStockMonthRevenue", start_date=d, end_date=d), dates))
    df = pd.concat([d for d in dfs if d is not None and len(d)], ignore_index=True)
    df["ym"] = pd.PeriodIndex.from_fields(year=df.revenue_year.astype(int), month=df.revenue_month.astype(int), freq="M")
    df["avail"] = (df.ym + 1).dt.to_timestamp() + pd.Timedelta(days=10)
    df = df[df.avail <= last_day]
    out = {}
    for sid, g in df.groupby("stock_id"):
        r = g.drop_duplicates("ym", keep="last").set_index("ym").revenue.astype(float)
        r = r.reindex(pd.period_range(r.index.min(), r.index.max(), freq="M"))
        r3 = r.rolling(3).sum()
        v = (r3 / r3.rolling(12, min_periods=12).max()).iloc[-1]
        stale = last_day - ((r.index[-1] + 1).to_timestamp() + pd.Timedelta(days=10)) > pd.Timedelta(days=75)
        if pd.notna(v) and not stale and v not in (float("inf"), float("-inf")):
            out[sid] = float(v)
    return pd.Series(out, dtype=float)


def pick(today, h2_only=False):
    lim = RateLimiter(80)
    names = universe(lim)
    raw = recent_raw(lim, today)
    last_day = raw.date.max()
    vol = raw.pivot_table(index="date", columns="stock_id", values="Trading_Volume").sort_index()
    liq = vol.iloc[-LIQ_WIN:].mean()
    close_today = raw[raw.date == last_day].set_index("stock_id").close
    start = (today - timedelta(days=int(MIN_BARS * 1.6) + 30)).isoformat()   # 夠 ≥250 根 K 判斷 + 252 日高
    pool = list(names) if not h2_only else [s for s in names if liq.get(s, 0) >= LIQ_SHARES]   # H29 百分位要全市場
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(lambda s: (s, adj_close(lim, s, (last_day, close_today.get(s)) if s in close_today else None, start, today)), pool))
    C = pd.DataFrame({s: a for s, a in res if a is not None}).sort_index().loc[:last_day]
    win = C.iloc[-HI_WIN:]
    h2 = (C.iloc[-1] / win.max()).where(win.notna().sum() >= 200)        # 同回測 rolling(252, min_periods=200)；當日停牌＝NaN
    bars = C.notna().sum()
    elig = [s for s in C.columns if liq.get(s, 0) >= LIQ_SHARES and bars[s] >= MIN_BARS]
    if h2_only:
        score, h10 = h2, pd.Series(dtype=float)
    else:
        h10 = rev_high(lim, last_day).reindex(C.columns)
        score = (h2.rank(pct=True) + h10.rank(pct=True)) / 2              # 同回測 _pct（axis=1 rank, pct=True）
    sc = score[elig].dropna().rename("score").reset_index().rename(columns={"index": "sid"})
    sc = sc.sort_values(["score", "sid"], ascending=[False, True])        # 平手依代號小→大（與回測 nlargest 一致）
    top = sc.set_index("sid").score.head(TOPN)
    n_tie = int((sc.score >= top.iloc[-1] - 1e-9).sum()) if len(top) else 0
    parts = pd.DataFrame({"h2": h2.reindex(top.index), "h10": h10.reindex(top.index)})
    return top, parts, names, close_today, last_day, len(elig), n_tie


def message(top, parts, prev, names, close_today, last_day, n_pool, capital, target_day, n_tie=0, h2_only=False):
    keep = [s for s in top.index if s in prev]
    buy = [s for s in top.index if s not in prev]
    sell = [s for s in prev if s not in top.index]
    per = capital / TOPN if capital else None

    def line(s, show_qty):
        px = close_today.get(s)
        if s not in top.index:
            t = f"{s} {names.get(s, '')}"
        elif h2_only:
            t = f"{s} {names.get(s, '')}  {top[s]:.3f}"
        else:
            t = f"{s} {names.get(s, '')}  價{parts.h2[s]:.2f} 營{parts.h10[s]:.2f}"
        if px:
            t += f"  收{px:g}"
            if show_qty and per:
                sh = int(per // px)
                t += f"  ≈{sh // 1000}張{sh % 1000}股" if sh >= 1000 else f"  ≈{sh}股(零股)"
        return t

    title = "H2 52週高點" if h2_only else "H29 雙新高"
    out = [f"📈 {title} 月換股（{target_day:%m/%d} 週{'一二三四五'[target_day.weekday()]} 開盤；遇假日順延）",
           f"資料至 {last_day:%m/%d} 收盤｜股票池 {n_pool} 檔｜取前 {TOPN}" + (f"｜每檔目標 {per:,.0f} 元" if per else "｜每檔 1/20 資金")]
    if n_tie > TOPN:
        out.append(f"（{n_tie} 檔並列最低入選分數，依代號小→大取，與回測一致）")
    if not prev:
        out += ["", f"【建立部位 {len(buy)} 檔】"] + [line(s, True) for s in buy]
    else:
        out += ["", f"【賣出 {len(sell)}】"] + ([line(s, False) for s in sell] or ["（無）"])
        out += ["", f"【買進 {len(buy)}】"] + ([line(s, True) for s in buy] or ["（無）"])
        out += ["", f"【續抱 {len(keep)}】（偏離目標 ±30% 以上再調）"] + [line(s, True) for s in keep]
    out += ["", "規則：前日收盤/252日最高，無停損；分數=1.000 即收在一年新高" if h2_only else
            "規則：價=前日收盤/252日最高、營=近3月營收/12月內最高3月，兩者全市場百分位平均取前20；無停損"]
    return "\n".join(out)


def main():
    force, no_send, h2_only = "--force" in sys.argv, "--no-send" in sys.argv, "--h2" in sys.argv
    asof = date.fromisoformat(sys.argv[sys.argv.index("--asof") + 1]) if "--asof" in sys.argv else None
    today = asof or date.today()
    target = next_weekday(today)
    st = load_state()
    month = target.strftime("%Y-%m")
    if not force and (not is_rebalance_eve(target) or month in st["months"]):
        return
    top, parts, names, close_today, last_day, n_pool, n_tie = pick(today, h2_only)
    prev_m = sorted(m for m in st["months"] if m < month)
    prev = st["months"][prev_m[-1]]["picks"] if prev_m else []
    msg = message(top, parts, prev, names, close_today, last_day, n_pool, st.get("capital"), target, n_tie, h2_only)
    print(msg)
    if not force:
        st["months"][month] = {"strategy": "H2" if h2_only else "H29", "picks": list(top.index), "scores": top.round(4).to_dict(),
                               "data_date": f"{last_day:%Y-%m-%d}", "created": datetime.now().isoformat(timespec="seconds")}
        STATE.write_text(json.dumps(st, ensure_ascii=False, indent=1))
    if not no_send:
        from notify import send
        send(msg)          # 只發 TELEGRAM_CHAT_ID（早晨摘要收件人），不發盤中名單


if __name__ == "__main__":
    main()
