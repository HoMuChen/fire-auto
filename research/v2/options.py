"""第十三~十五批：台指選擇權（TXO）賣方。月選 H66~H71、週選 H72~H75；規則與過關標準見 HYPOTHESES.md。

每個到期日 E_k 收盤以下一個到期契約的結算價建倉 → 持有到 E_{k+1}，以期交所最後結算價現金結算；逐日結算價盯市。
月選只用月契約；週選用所有到期（週選 + 第三週的月選）。
部位：每 1 指數點名目配 1 口（名目 1 倍）；報酬以進場時 F 為分母。
用法：python3 research/v2/options.py fsp | weekly | dev | val [H66 ...]
"""
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402
import mtx  # noqa: E402

FD = H.D / "futures"
TAX, COMM = 0.001, 0.4                                     # 權利金期交稅、每口每邊手續費（點）


def fetch_fsp(years=None):
    """期交所「選擇權最後結算價」→ data/v2/futures/TXO_final_settle.parquet（月選）。years 給定時只抓這些年並與既有檔合併（每日增量用）。"""
    import requests
    rows = []
    for y in years or range(2009, pd.Timestamp.today().year + 1):
        r = requests.post("https://www.taifex.com.tw/cht/5/optIndxFSP", headers={"User-Agent": "Mozilla/5.0"}, timeout=60,
                          data=dict(start_year=str(y), start_month="01", end_year=str(y), end_month="12", all="all", _all="on"))
        t = pd.read_html(io.StringIO(r.content.decode("utf-8", "ignore")))[0].iloc[:, :3]
        t.columns = ["date", "contract", "txo"]
        rows.append(t)
    df = pd.concat(rows, ignore_index=True).astype({"contract": str})
    df = df[df.contract.str.fullmatch(r"\d{6}(W\d)?")].copy()
    df["date"], df["txo"] = pd.to_datetime(df.date), pd.to_numeric(df.txo, errors="coerce")
    if years and (FD / "TXO_final_settle.parquet").exists():
        df = pd.concat([pd.read_parquet(FD / "TXO_final_settle.parquet"), df], ignore_index=True)
    df = df.drop_duplicates("contract", keep="last").sort_values("date", kind="stable").reset_index(drop=True)
    df.to_parquet(FD / "TXO_final_settle.parquet")
    return df


def fetch_weekly():
    """FinMind TXO 週選（契約 YYYYMMWn）日盤結算價，2012-11 起 → data/v2/futures/TXO_weekly.parquet。"""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from build_data_v2 import RateLimiter, api
    lim, out = RateLimiter(80), []
    for m in pd.date_range("2012-11-01", pd.Timestamp.today(), freq="MS"):
        o = api(lim, dataset="TaiwanOptionDaily", data_id="TXO", start_date=f"{m:%Y-%m-%d}", end_date=f"{m + pd.offsets.MonthEnd(0):%Y-%m-%d}")
        if o is not None and len(o):
            o = o[(o.trading_session == "position") & o.contract_date.str.fullmatch(r"\d{6}W\d")]
            out.append(o[["date", "contract_date", "strike_price", "call_put", "settlement_price", "volume", "open_interest"]])
    df = pd.concat(out, ignore_index=True)
    df.to_parquet(FD / "TXO_weekly.parquet")
    print("TXO weekly", len(df), df.contract_date.nunique(), df.date.min(), df.date.max())


def tick(p):
    return np.select([p < 10, p < 50, p < 500, p < 1000], [0.1, 0.5, 1, 5], 10)


def side_cost(p):
    """每口每邊成本（點）。"""
    return TAX * p + COMM + tick(p)


class Data:
    def __init__(self, end):
        end = pd.Timestamp(end)
        ch = pd.read_parquet(FD / "TXO_chain.parquet")
        if (FD / "TXO_weekly.parquet").exists():
            ch = pd.concat([ch, pd.read_parquet(FD / "TXO_weekly.parquet")[ch.columns]], ignore_index=True)
        ch["date"] = pd.to_datetime(ch.date)
        ch = ch[ch.date <= end]
        ch["cp"] = ch.call_put.str[0].str.upper()
        self.chain = {cd: g.pivot_table(index="date", columns=["cp", "strike_price"], values="settlement_price")
                      for cd, g in ch.groupby("contract_date")}
        fs = pd.read_parquet(FD / "TXO_final_settle.parquet")
        self.exp = fs.set_index("contract").date.to_dict()
        self.sq = fs.set_index("contract").txo.to_dict()
        m = pd.read_parquet(mtx.RAW)
        m = m[m.trading_session == "position"].assign(date=lambda x: pd.to_datetime(x.date))
        self.fut = m[m.date <= end].pivot_table(index="date", columns="contract_date", values="close")
        self.vix = pd.read_parquet(FD / "TXO_vix.parquet").vix
        F = mtx.continuous(end)
        self.rv = F.S.pct_change().rolling(21).std() * np.sqrt(252)
        self.mtx = F
        self.bh = (1 + mtx.run_pos(F, *mtx.strategy("BH", F))[0]).cumprod()   # MTX 連續做多淨值（含轉倉成本）
        self.end = end


def strike(avail, target):
    return avail[np.argmin(np.abs(avail - target))]


WEEKLY = {"H72", "H73", "H74", "H75"}


def parity_fwd(px):
    """買權賣權平價推遠期：|C−P| 最小的履約價 K，F = K + C − P。"""
    k = px["C"].index.intersection(px["P"].index)
    d = px["C"][k].values - px["P"][k].values
    i = np.argmin(np.abs(d))
    return k[i] + d[i]


def legs_for(h, D, cd, e0):
    """回傳 [(cp, K, qty)]（qty<0 賣出）、期貨多單（None / "same" 同月契約 / "bh" MTX 連續做多）與 F；legs None = 該期空手。"""
    px = D.chain[cd].loc[e0].dropna()
    F = parity_fwd(px) if h in WEEKLY else D.fut.at[e0, cd]
    T = (D.exp[cd] - e0).days / 365
    sig = D.vix.get(e0, np.nan) / 100
    kc, kp = np.array(sorted(px["C"].index)), np.array(sorted(px["P"].index))
    up, dn = F * np.exp(sig * np.sqrt(T)), F * np.exp(-sig * np.sqrt(T))
    if h in ("H66", "H72"):
        return [("C", strike(kc, up), -1), ("P", strike(kp, dn), -1)], None, F
    if h in ("H67", "H73"):
        return [("P", strike(kp, F), -1)], None, F
    if h == "H68":
        up2, dn2 = F * np.exp(2 * sig * np.sqrt(T)), F * np.exp(-2 * sig * np.sqrt(T))
        return [("C", strike(kc, up), -1), ("P", strike(kp, dn), -1), ("C", strike(kc, up2), 1), ("P", strike(kp, dn2), 1)], None, F
    if h == "H69":
        if not (sig > D.rv.get(e0, np.nan)):
            return None, None, F
        return [("C", strike(kc, up), -1), ("P", strike(kp, dn), -1)], None, F
    if h == "H70":
        return [("C", strike(kc, F), -1)], "same", F
    if h == "H71":
        return [("C", strike(kc, up), -1), ("P", strike(kp, dn), -1)], "same", F
    if h == "H74":
        return [("P", strike(kp, dn), -1)], None, F
    if h == "H75":
        return [("C", strike(kc, up), -1)], "bh", F
    raise KeyError(h)


def run(h, D):
    """回傳日淨值（以 1 起算）與每月明細。"""
    weekly = h in WEEKLY
    cds = sorted((c for c in D.exp if c in D.chain and (weekly or "W" not in c)), key=D.exp.get)
    eq, out, months = 1.0, [], []
    for prev, cd in zip(cds[:-1], cds[1:]):
        e0, e1 = D.exp[prev], D.exp[cd]                      # 進場 = 上一序列到期日收盤
        if e0 > D.end:
            break
        if e0 not in D.chain[cd].index or (not weekly and e0 not in D.fut.index):
            legs = None                                      # 下一契約進場日無報價 → 空手
        else:
            legs, fut, F0 = legs_for(h, D, cd, e0)
        if legs is None:                                     # 空手期
            days = D.mtx.index[(D.mtx.index >= e0) & (D.mtx.index <= min(e1, D.end))]
            out.append(pd.Series(eq, index=days))
            months.append(dict(entry=e0, exit=e1, ret=0.0, legs="空手"))
            continue
        days = D.chain[cd].loc[e0:min(e1, D.end)].index
        P = D.chain[cd].loc[days]
        val = []
        for t in days:
            v = 0.0
            for cp, K, q in legs:
                if t == e1:
                    p = max(D.sq[cd] - K, 0) if cp == "C" else max(K - D.sq[cd], 0)
                else:
                    p = P[(cp, K)].loc[:t].ffill().iloc[-1]
                v += q * p
            if fut == "same":
                v += (D.sq[cd] if t == e1 else D.fut[cd].loc[:t].ffill().iloc[-1])
            elif fut == "bh":
                v += F0 * D.bh.loc[:t].iloc[-1] / D.bh.loc[:e0].iloc[-1]
            val.append(v)
        val = np.array(val)
        p0 = [P[(cp, K)].iloc[0] for cp, K, _ in legs]
        cost = sum(side_cost(p) for p in p0) + (F0 * (mtx.TAX + mtx.FEE) + 1 if fut == "same" else 0)   # "bh" 成本已含在淨值
        exit_cost = 0.0
        if days[-1] == e1:
            exit_cost = sum(COMM for cp, K, _ in legs if (D.sq[cd] - K if cp == "C" else K - D.sq[cd]) > 0) + (F0 * mtx.FEE if fut == "same" else 0)
        pnl = (val - val[0]) / F0                            # 以進場 F 為名目
        path = eq * (1 + pnl - cost / F0 - np.r_[np.zeros(len(days) - 1), exit_cost / F0])
        out.append(pd.Series(path, index=days))
        months.append(dict(entry=e0, exit=e1, ret=path[-1] / eq - 1, legs=" ".join(f"{'賣' if q < 0 else '買'}{cp}{K:.0f}" for cp, K, q in legs),
                           F=F0, sq=D.sq[cd] if days[-1] == e1 else np.nan, credit=-sum(q * p for (_, _, q), p in zip(legs, p0))))
        eq = path[-1]
    s = pd.concat(out)
    s = s[~s.index.duplicated(keep="last")]
    return s, pd.DataFrame(months)


NAMES = {"H66": "H66 賣1σ寬跨式", "H67": "H67 賣ATM賣權", "H68": "H68 鐵兀鷹1σ/2σ", "H69": "H69 寬跨式+VRP濾網", "H70": "H70 掩護性買權", "H71": "H71 做多+寬跨式疊加(事後)",
         "H72": "H72 週選1σ寬跨式", "H73": "H73 週選ATM賣權", "H74": "H74 週選1σ價外賣權", "H75": "H75 做多+週選1σ買權"}
WEEKLY_DEV0 = "2013-01-01"                                   # 週選 2012-11 才掛牌


def bench(D, p0, p1):
    rb = mtx.run_pos(D.mtx, *mtx.strategy("BH", D.mtx))[0].loc[p0:p1]
    tr = pd.read_parquet(H.D / "taiex_tr.parquet").assign(date=lambda x: pd.to_datetime(x.date)).set_index("date").price.pct_change().loc[p0:p1]
    mb, mt = mtx.metrics(rb), mtx.metrics(tr)
    bar = max(mb["Sharpe"], mt["Sharpe"])
    print(f"[{p0}~{p1}] 基準 MTX 做多 Sharpe {mb['Sharpe']:.2f} MDD {mb['MDD']*100:.1f}% ｜ 加權報酬指數 Sharpe {mt['Sharpe']:.2f} → 門檻 Sharpe > {bar:.2f} 且 MDD ≥ {mb['MDD']*100:.1f}%")
    return rb, mb, bar


if __name__ == "__main__":
    if sys.argv[1] in ("fsp", "weekly"):
        fetch_fsp() if sys.argv[1] == "fsp" else fetch_weekly()
        sys.exit()
    period = sys.argv[1]
    unseal = "--unseal" in sys.argv
    if period == "holdout" and not unseal:
        raise PermissionError("holdout 未開封（需使用者同意，加 --unseal）")
    want = [a for a in sys.argv[2:] if not a.startswith("--")] or list(NAMES)
    P0, p1 = H.PERIODS[period]
    D = Data(p1)
    benches = {}
    for h in want:
        p0 = max(P0, WEEKLY_DEV0) if h in WEEKLY else P0
        if p0 not in benches:
            benches[p0] = bench(D, p0, p1)
        rb, mb, bar = benches[p0]
        s, mo = run(h, D)
        r = s.pct_change().loc[p0:p1].dropna()
        m = mtx.metrics(r, rb.reindex(r.index).fillna(0))
        mr = mo[(mo.exit > p0) & (mo.entry <= p1)].ret                 # 每期（月選 = 月、週選 = 週）報酬
        ok = m["Sharpe"] > bar and m["MDD"] >= mb["MDD"]
        print(f"{NAMES[h]:16s} Sharpe {m['Sharpe']:5.2f}  CAGR {m['CAGR']*100:+6.1f}%  MDD {m['MDD']*100:6.1f}%  {'✓' if ok else '✗'}  "
              f"最差期 {mr.min()*100:+.1f}%  期勝率 {(mr > 0).mean():.0%}  期偏態 {mr.skew():+.2f}  空手期 {(mo.legs == '空手').sum()}  JK z {m['JK_z']:+.2f}")
        print("   " + "  ".join(f"{k}:{v*100:+.1f}" for k, v in m["years"].items()))
        with open(H.LOG, "a") as f:
            f.write(json.dumps(dict(ts=time.strftime("%Y-%m-%d %H:%M:%S"), kind="options", period=period, start=p0, strategy=NAMES[h],
                                    note="TXO 賣方 第十五批（週選）" if h in WEEKLY else "TXO 賣方 第十三批", bar=bar, passed=bool(ok),
                                    worst_period=mr.min(), win_period=(mr > 0).mean(),
                                    **{k: v for k, v in m.items() if k != "years"}, years=m["years"]), ensure_ascii=False, default=float) + "\n")
