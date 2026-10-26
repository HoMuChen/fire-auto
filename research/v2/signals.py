"""第二十三批：大盤「事件型」訊號 —— 做多勝率特別高（A 系）/ 該降槓桿的警示（W 系）。
不是擇時回測：只問「訊號出現後」的條件分布是否顯著優於 / 劣於基準。登記見 HYPOTHESES.md 第二十三批。

用法：python3 research/v2/signals.py build            # 建漲跌家數快取 data/v2/futures/ad.parquet
      python3 research/v2/signals.py dev              # 全部訊號 dev 判定
      python3 research/v2/signals.py val A1 W3 ...    # 只跑 dev 過關者（val）
      python3 research/v2/signals.py pool A1 A2 ...   # dev 段數 <8 者：2010-23 合併描述（不判定）
      python3 research/v2/signals.py check            # 用 H108 驗證工具（應重現 3/11、3/10）
      python3 research/v2/signals.py now              # 各訊號最近一次亮燈與現況（描述用）

時序：所有原料皆為當日收盤 / 盤後公布 → 訊號日 t 收盤後才知道 → **t+1 收盤進場**（保守，
避開 FinMind 指數開盤價不可靠），報酬用加權報酬指數 tr[t+1+h] / tr[t+1] − 1。
事件段：訊號日間隔 > GAP（20 交易日）才算新的一段；只用每段第一天（起點）評估。
顯著性：隨機日期抽樣（同期間、同段數、互相間隔 > GAP）10,000 次的經驗 p 值。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402

FD = H.D / "futures"
AD = FD / "ad.parquet"
LOG = Path(__file__).resolve().parent / "experiments.jsonl"
GAP, NBOOT, SEED = 20, 10_000, 19
BULL_H, WARN_H = (20, 60, 120), 120


# ---------------------------------------------------------------- 資料
def build_ad(end="2026-12-31"):
    """全部普通股（含下市）每日上漲 / 下跌家數（還原收盤 vs 前一日）。"""
    up, dn = {}, {}
    for s in H.universe():
        df = H.load_stock(s, end)
        if df is None or len(df) < 2:
            continue
        c = df.set_index("date").close
        r = c.pct_change()
        up[s], dn[s] = (r > 0), (r < 0)
    U, Dn = pd.DataFrame(up).sort_index(), pd.DataFrame(dn).sort_index()
    out = pd.DataFrame({"adv": U.sum(axis=1), "dec": Dn.sum(axis=1)})
    out = out[(out.adv + out.dec) >= 200]
    out.to_parquet(AD)
    print(f"ad {out.index[0].date()} ~ {out.index[-1].date()} {len(out)} 天")


def load():
    tx = pd.read_parquet(FD / "taiex.parquet")
    tx = tx[tx.close > 0].assign(date=lambda d: pd.to_datetime(d.date)).set_index("date").close
    tr = pd.read_parquet(H.D / "taiex_tr.parquet").assign(date=lambda d: pd.to_datetime(d.date)).set_index("date").price
    idx = tx.index.intersection(tr.index)
    F = pd.DataFrame({"taiex": tx.reindex(idx), "tr": tr.reindex(idx)})
    F["idx_dd"] = F.taiex / F.taiex.rolling(252, min_periods=252).max() - 1
    F["ma200"] = F.taiex.rolling(200, min_periods=200).mean()

    b = pd.read_parquet(FD / "breadth.parquet")
    F["B20"] = b.B20.reindex(idx)
    F["dB20"] = F.B20 - F.B20.shift(60)

    ad = pd.read_parquet(AD)
    F["adr"] = (ad.adv / (ad.adv + ad.dec)).reindex(idx)
    F["adr_ema"] = F.adr.ewm(span=10, adjust=False).mean()

    F["vix"] = pd.read_parquet(FD / "TXO_vix.parquet").vix.reindex(idx)

    mg = pd.read_parquet(FD / "mkt_margin.parquet")
    mg = mg[mg.name == "MarginPurchaseMoney"].assign(date=lambda d: pd.to_datetime(d.date)).set_index("date").TodayBalance
    F["margin"] = mg.reindex(idx)

    ins = pd.read_parquet(FD / "mkt_inst.parquet")
    ins = ins[ins.name == "Foreign_Investor"].assign(date=lambda d: pd.to_datetime(d.date)).set_index("date")
    F["fnet"] = (ins.buy - ins.sell).reindex(idx)

    lt = pd.read_parquet(FD / "TX_large_traders.parquet").assign(date=lambda d: pd.to_datetime(d.date)).set_index("date")
    lt = lt[~lt.index.duplicated(keep="last")]
    F["lt_net"] = ((lt.buy_top10_specific_open_interest - lt.sell_top10_specific_open_interest)
                   / lt.market_open_interest).reindex(idx)

    sp = pd.read_parquet(FD / "gspc.parquet").assign(date=lambda d: pd.to_datetime(d.date)).set_index("date").Close
    sp_ret = sp.pct_change()
    sp_below = sp < sp.rolling(200, min_periods=200).mean()
    # 美股日 d 收盤 → 台股「d 之後第一個交易日」可用（台北時間清晨已知）。對到台股日：
    pos = idx.searchsorted(sp.index, side="right")
    ok = pos < len(idx)
    m = pd.DataFrame({"tw": idx[pos[ok]], "ret": sp_ret.values[ok], "below": sp_below.values[ok]})
    m = m.groupby("tw").agg(ret=("ret", "min"), below=("below", "last"))
    F["sp_ret_prev"] = m.ret.reindex(idx)       # 台股日 t 開盤前已知的最近美股日報酬（取期間最差）
    F["sp_below_prev"] = m.below.reindex(idx).astype("boolean").ffill().fillna(False).astype(bool)
    return F


# ---------------------------------------------------------------- 訊號（全部只用 t 日以前資料）
def signals(F):
    S = {}
    # A1 寬度洗盤：B20 > 70% 且指數距 52 週高 < −10%（原登記 80%/−15% dev 只 2 段，看報酬前依頻率放寬一次）
    S["A1"] = (F.B20 > 0.70) & (F.idx_dd < -0.10)
    # A2 Zweig breadth thrust：adr 10 日 EMA 在 10 日內由 < 0.40 升到 > 0.615
    S["A2"] = (F.adr_ema > 0.615) & (F.adr_ema.rolling(11).min() < 0.40)
    # A3 VIX 尖峰回落：近 20 日內 VIX 創 126 日（半年）新高，且今天 ≤ 該高點 × 0.8（原 252 日 dev 只 4 段）
    vhi = F.vix.rolling(126, min_periods=126).max()
    peak20 = F.vix.rolling(20).max()
    S["A3"] = ((F.vix >= vhi).astype(float).rolling(20).max() > 0) & (F.vix <= 0.8 * peak20)
    # A4 融資急降：融資金額 20 日變化 ≤ −10%（原 −15% dev 只 2 段）
    S["A4"] = (F.margin / F.margin.shift(20) - 1) <= -0.10
    # A5 外資連 5 日賣超且 5 日合計落在過去 756 日 5 日合計的最低 5%
    f5 = F.fnet.rolling(5).sum()
    q05 = f5.rolling(756, min_periods=500).quantile(0.05).shift(1)
    S["A5"] = (F.fnet < 0).rolling(5).sum().eq(5) & (f5 <= q05)
    # A6 美股單日跌 > 3% 之後的台股第一個交易日
    S["A6"] = F.sp_ret_prev <= -0.03

    strong = F.idx_dd >= -0.03
    # W1 H108 寬度惡化型背離（對照，已完成）
    S["W1"] = strong & (F.B20 > 0.26) & (F.dB20 > 0)
    # W2 融資過熱：融資金額 60 日 +10% 且指數距高 ≤3%（原 +20% dev 0 段）
    S["W2"] = ((F.margin / F.margin.shift(60) - 1) >= 0.10) & strong
    # W3 過度延伸：指數高於 200 日線 ≥ 10%（原 20% dev 0 段）
    S["W3"] = (F.taiex / F.ma200 - 1) >= 0.10
    # W4 低波動自滿：VIX ≤ 過去 252 日 10% 分位 且 指數創 252 日新高
    vq10 = F.vix.rolling(252, min_periods=252).quantile(0.10)
    S["W4"] = (F.vix <= vq10) & (F.idx_dd >= -1e-9)
    # W5 前十大特定法人淨部位創 126 日新低且為淨空（原 252 日 dev 1 段）
    S["W5"] = (F.lt_net <= F.lt_net.rolling(126, min_periods=126).min()) & (F.lt_net < 0)
    # W6 S&P 500 收在 200 日線下（台股次一交易日可知）
    S["W6"] = F.sp_below_prev
    return {k: v.fillna(False).astype(bool) for k, v in S.items()}


NAMES = {"A1": "寬度洗盤（B20>70% 且指數距高<−10%）", "A2": "Zweig 寬度推升", "A3": "VIX 半年高後回落 20%",
         "A4": "融資 20 日急降 ≥10%", "A5": "外資 5 日連賣且創 3 年極端", "A6": "美股單日跌 >3% 隔日",
         "W1": "H108 寬度惡化型背離（對照）", "W2": "融資 60 日 +10% 且指數在高點", "W3": "指數高於 200 日線 ≥10%",
         "W4": "VIX 1 年 10% 低分位且指數創高", "W5": "大額特定法人淨空創半年新低", "W6": "S&P 500 跌破 200 日線"}


# ---------------------------------------------------------------- 評估
def starts(flag, m):
    """期間 m 內的事件段起點（位置）。"""
    days = np.where((flag & m).values)[0]
    out, prev = [], -10**9
    for d in days:
        if d - prev > GAP:
            out.append(d)
        prev = d
    return np.array(out, int)


def outcomes(F):
    tr = F.tr.values
    n = len(tr)
    O = {}
    for h in BULL_H:
        r = np.full(n, np.nan)
        r[:n - 1 - h] = tr[1 + h:] / tr[1:n - h] - 1
        O[f"r{h}"] = r
    dd = np.full(n, np.nan)          # t+1 進場後 120 日內最深跌幅（加權報酬指數）
    mae60 = np.full(n, np.nan)       # 做多進場後 60 日內最深浮虧
    for i in range(n - 1 - WARN_H):
        w = tr[i + 1:i + 2 + WARN_H]
        dd[i] = w.min() / w[0] - 1
        mae60[i] = w[:61].min() / w[0] - 1
    O["dd120"], O["mae60"] = dd, mae60
    return O


def rand_starts(pool, k, rng):
    """從 pool（位置陣列）抽 k 個互相間隔 > GAP 的日期。"""
    for _ in range(50):
        c = np.sort(rng.choice(pool, size=min(len(pool), k * 3), replace=False))
        pick, prev = [], -10**9
        for d in c:
            if d - prev > GAP:
                pick.append(d); prev = d
            if len(pick) == k:
                return np.array(pick)
    return np.array(pick)


PERIODS = dict(H.PERIODS, pool=(H.PERIODS["dev"][0], H.PERIODS["val"][1]))   # pool = 2010-23 合併，只作描述


def evaluate(F, O, key, flag, period, rng):
    p0, p1 = PERIODS[period]
    m = pd.Series((F.index >= p0) & (F.index <= p1), index=F.index)
    st = starts(flag, m)
    valid_all = np.where(m.values & ~np.isnan(O["dd120"]))[0]
    st = st[~np.isnan(O["dd120"][st])]
    res = {"key": key, "period": period, "episodes": int(len(st)), "flag_days": int((flag & m).sum())}
    if len(st) == 0:
        return res, st
    bull = key.startswith("A")
    if bull:
        for h in BULL_H:
            r = O[f"r{h}"]
            res[f"win{h}"] = float((r[st] > 0).mean()); res[f"mean{h}"] = float(np.nanmean(r[st]))
            res[f"base_win{h}"] = float((r[valid_all] > 0).mean()); res[f"base_mean{h}"] = float(np.nanmean(r[valid_all]))
        res["mae60"] = float(np.nanmean(O["mae60"][st])); res["base_mae60"] = float(np.nanmean(O["mae60"][valid_all]))
        stat = lambda idx: np.nanmean(O["r60"][idx])                      # noqa: E731
    else:
        hit = O["dd120"] <= -0.10
        res["hit"] = float(hit[st].mean()); res["base_hit"] = float(hit[valid_all].mean())
        for h in (60, 120):
            res[f"mean{h}"] = float(np.nanmean(O[f"r{h}"][st])); res[f"base_mean{h}"] = float(np.nanmean(O[f"r{h}"][valid_all]))
        res["dd120"] = float(np.nanmean(O["dd120"][st])); res["base_dd120"] = float(np.nanmean(O["dd120"][valid_all]))
        win = flag.astype(float).rolling(WARN_H).max()
        res["window_cover"] = float(win[m].mean())   # 隨機 120 日窗含訊號的機率（常亮程度）
        stat = lambda idx: hit[idx].mean()                               # noqa: E731
    obs = stat(st)
    sims = np.array([stat(rand_starts(valid_all, len(st), rng)) for _ in range(NBOOT)])
    res["p"] = float((sims >= obs).mean()) if bull else float((sims >= obs).mean())
    # 做多：mean60 越大越好 → p = P(sim ≥ obs)；警示：hit 越大越好 → p = P(sim ≥ obs)
    return res, st


PASS = {"min_ep": 8, "bull_win_lift": 0.15, "warn_hit_lift": 0.15, "p_dev": 0.01, "p_val": 0.10}


def verdict(r, period):
    if period == "pool":
        return "描述（不判定）"
    if r["episodes"] < (PASS["min_ep"] if period == "dev" else 3):
        return "樣本不足 → 只看 2010-23 合併描述"
    pmax = PASS["p_dev"] if period == "dev" else PASS["p_val"]
    if r["key"].startswith("A"):
        lift = r["win60"] - r["base_win60"]
        ok = (r["mean60"] > r["base_mean60"]) and r["p"] <= pmax and (lift >= PASS["bull_win_lift"] if period == "dev" else lift > 0)
    else:
        lift = r["hit"] - r["base_hit"]
        ok = (r["mean120"] < r["base_mean120"]) and r["p"] <= pmax and (lift >= PASS["warn_hit_lift"] if period == "dev" else lift > 0)
    if ok:
        return "✓"
    return "留意（p≤0.05）" if period == "dev" and r["p"] <= 0.05 else "✗"


def fmt(r, period):
    pc = lambda x: f"{x*100:+.1f}" if isinstance(x, float) else "—"   # noqa: E731
    pw = lambda x: f"{x*100:.0f}%" if isinstance(x, float) else "—"   # noqa: E731
    head = f"{r['key']} {NAMES[r['key']]:<28} 段 {r['episodes']:>2}（訊號日 {r['flag_days']}）"
    if r["episodes"] == 0:
        return head
    if r["key"].startswith("A"):
        body = (f"  60 日勝率 {pw(r['win60'])} vs 基準 {pw(r['base_win60'])} | 均 20/60/120 日 {pc(r['mean20'])}/{pc(r['mean60'])}/{pc(r['mean120'])}%"
                f"（基準 {pc(r['base_mean20'])}/{pc(r['base_mean60'])}/{pc(r['base_mean120'])}）| 120 日勝率 {pw(r['win120'])} vs {pw(r['base_win120'])}"
                f" | 60 日最深浮虧 {pc(r['mae60'])}%（{pc(r['base_mae60'])}）| p {r['p']:.4f}")
    else:
        body = (f"  120 日內跌 ≥10% {pw(r['hit'])} vs 基準 {pw(r['base_hit'])} | 均 60/120 日 {pc(r['mean60'])}/{pc(r['mean120'])}%"
                f"（基準 {pc(r['base_mean60'])}/{pc(r['base_mean120'])}）| 120 日最深 {pc(r['dd120'])}%（{pc(r['base_dd120'])}）"
                f" | 120 日窗含訊號 {pw(r['window_cover'])} | p {r['p']:.4f}")
    return f"{head}  → {verdict(r, period)}\n{body}"


def run(period, keys):
    if period == "holdout":
        raise SystemExit("holdout 需使用者明確同意後再開")
    F = load(); O = outcomes(F); S = signals(F)
    rng = np.random.default_rng(SEED)
    print(f"[{period} {PERIODS[period][0]}~{PERIODS[period][1]}]  t+1 收盤進場；事件段間隔 >{GAP} 日；隨機抽樣 {NBOOT} 次 seed {SEED}")
    rows = []
    for k in keys:
        r, st = evaluate(F, O, k, S[k], period, rng)
        print(fmt(r, period))
        if len(st):
            tag = "dd120" if k.startswith("W") else "r60"
            print("    起點：" + ", ".join(f"{F.index[i].strftime('%Y-%m-%d')}({O[tag][i]*100:+.0f}%)" for i in st))
        r["verdict"] = verdict(r, period); rows.append(r)
    with open(LOG, "a") as f:
        f.write(json.dumps(dict(ts=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"), kind="signals", period=period,
                                strategy="batch23", note="第二十三批 事件型訊號", results=rows), ensure_ascii=False) + "\n")


def check():
    """H108 = W1：段命中應為 dev 3/11、val 3/10（breadth.py warn 用 t 收盤、此處用 t+1，可能微差）。"""
    F = load(); O = outcomes(F); S = signals(F)
    rng = np.random.default_rng(SEED)
    for p in ("dev", "val"):
        r, st = evaluate(F, O, "W1", S["W1"], p, rng)
        print(p, f"段 {r['episodes']}，命中 {int(round(r['hit']*r['episodes']))}/{r['episodes']}，基準 {r['base_hit']*100:.0f}%，窗含訊號 {r['window_cover']*100:.0f}%")


def now():
    F = load(); S = signals(F)
    last = F.index[-1]
    print(f"資料最後一日 {last.date()}")
    for k, fl in S.items():
        on = fl[fl]
        print(f"  {k} {NAMES[k]:<28} 今天 {'亮' if fl.iloc[-1] else '—'}  最近亮燈 {on.index[-1].date() if len(on) else '從未'}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "dev"
    if cmd == "build":
        build_ad()
    elif cmd == "check":
        check()
    elif cmd == "now":
        now()
    else:
        keys = sys.argv[2:] or list(NAMES)
        run(cmd, keys)
