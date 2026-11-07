"""實盤監控 M1：H136 失效判準（HYPOTHESES.md「實盤監控 M1 登記」，2026-11-07 事前寫定）。

bands：FS2 引擎算 H136 全期每月 net / 超額 → 平穩區塊抽樣（平均區塊 6 個月）36 個月路徑，
       三個世界（全邊際／半邊際／零邊際）→ 檢查點 12/24/36 月的累積超額門檻、回撤門檻、誤報率、偵測力。
       結果存 data/v2/results/h136_bands.json，並記到 experiments.jsonl。
check：同一個引擎重算到最新，取 2026-11 換股起的實盤月份，對照 bands 的門檻。
       這是模型的實盤期報酬，不含實際成交差異。

用法：.venv/bin/python research/v2/h136_monitor.py check（每月換股後）；bands 只在上線前算一次（已存在會拒絕，需 --force）
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import feature_combo as FC  # noqa: E402
import harness as H  # noqa: E402
import xs  # noqa: E402

FEATS = ["H2", "H87", "H97"]
H136 = (0, 1, 2)
LIVE_START = pd.Timestamp("2026-11-01")
CHECKS = (12, 24, 36)
NPATH, BLOCK, SEED = 20_000, 6, 2026
Q_KILL, Q_DD = 0.05, 0.01
BANDS = H.D / "results" / "h136_bands.json"


def series():
    """FS2 引擎 H136 每月 net、ex（2010-01 起到最後一個完整換股月）。"""
    t0 = time.time()
    p1 = pd.Timestamp.today().strftime("%Y-%m-%d")
    P = xs.load_panel(p1, fundamentals=True)
    for add in (xs.add_chips, xs.add_fin, xs.add_v16, xs.add_v17):
        add(P)
    dates, R, E, F = FC.prep(P, H.PERIODS["dev"][0], p1, FEATS)
    net, ex = FC.bt(F, R, E, H136)
    s = pd.DataFrame(dict(net=net, ex=ex), index=dates).dropna()
    print(f"載入 {time.time()-t0:.0f}s；資料最後一天 {P['close'].index[-1].date()}；"
          f"月份 {s.index[0].date()} ~ {s.index[-1].date()}（{len(s)} 個，最後一個月的報酬到下一個換股日開盤）")
    return s


def boot_idx(n, rng):
    """平穩區塊抽樣索引 [NPATH, 36]（循環接回開頭）。"""
    L = max(CHECKS)
    idx = np.empty((NPATH, L), dtype=int)
    idx[:, 0] = rng.integers(0, n, NPATH)
    jump = rng.random((NPATH, L)) < 1 / BLOCK
    fresh = rng.integers(0, n, (NPATH, L))
    for t in range(1, L):
        idx[:, t] = np.where(jump[:, t], fresh[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def mdd(net):
    """每條路徑每個月為止的 MDD（net 淨值、起點 1）。"""
    eq = np.cumprod(1 + net, axis=1)
    peak = np.maximum.accumulate(np.concatenate([np.ones((len(eq), 1)), eq], axis=1), axis=1)[:, 1:]
    return np.minimum.accumulate(eq / peak - 1, axis=1)


def simulate(s, rng):
    idx = boot_idx(len(s), rng)
    net0, ex0 = s["net"].values[idx], s["ex"].values[idx]
    mu = s["ex"].mean()
    W = {}
    for w, cut in (("全邊際", 0.0), ("半邊際", 0.5), ("零邊際", 1.0)):
        net, ex = net0 - cut * mu, ex0 - cut * mu
        W[w] = dict(cum=np.cumsum(ex, axis=1), dd=mdd(net))
    kill = {h: float(np.quantile(W["半邊際"]["cum"][:, h - 1], Q_KILL)) for h in CHECKS}
    dd_line = float(np.quantile(W["全邊際"]["dd"][:, -1], Q_DD))
    out = dict(mu_m=mu, n=len(s), first=str(s.index[0].date()), last=str(s.index[-1].date()),
               kill=kill, dd_line=dd_line, worlds={})
    for w, d in W.items():
        hit_a = np.column_stack([d["cum"][:, h - 1] < kill[h] for h in CHECKS])
        hit_b = d["dd"][:, -1] < dd_line
        out["worlds"][w] = dict(
            cum_q={h: {q: float(np.quantile(d["cum"][:, h - 1], q)) for q in (0.05, 0.25, 0.5, 0.75)} for h in CHECKS},
            dd_q={h: {q: float(np.quantile(d["dd"][:, h - 1], q)) for q in (0.5, 0.05, 0.01)} for h in CHECKS},
            p_a_by={h: float(hit_a[:, :i + 1].any(axis=1).mean()) for i, h in enumerate(CHECKS)},
            p_b=float(hit_b.mean()),
            p_any=float((hit_a.any(axis=1) | hit_b).mean()),
            p_neg12=float((d["cum"][:, 11] < 0).mean()))
    return out


def pct(v):
    return f"{v*100:+.1f}%"


def report(o, title):
    print(f"\n== {title}：{o['first']} ~ {o['last']}（{o['n']} 個月），月超額平均 {pct(o['mu_m'])}（年化 {pct(o['mu_m']*12)}）")
    print("  累積超額分布（25% / 中位 / 75%）與 M1-a 門檻（半邊際 5% 分位）：")
    for h in CHECKS:
        f, z = o["worlds"]["全邊際"]["cum_q"][h], o["worlds"]["零邊際"]["cum_q"][h]
        print(f"    {h:>2} 個月：全邊際 {pct(f[0.25])} / {pct(f[0.5])} / {pct(f[0.75])}"
              f"｜零邊際中位 {pct(z[0.5])}｜門檻 {pct(o['kill'][h])}")
    print(f"  net 回撤（全邊際，中位 / 5% / 1%）：" + "  ".join(
        f"{h} 月 {pct(o['worlds']['全邊際']['dd_q'][h][0.5])} / {pct(o['worlds']['全邊際']['dd_q'][h][0.05])} / "
        f"{pct(o['worlds']['全邊際']['dd_q'][h][0.01])}" for h in CHECKS))
    print(f"  M1-b 回撤門檻（全邊際 36 月 MDD 1% 分位）：{pct(o['dd_line'])}")
    print("  觸發機率（M1-a 到第 12 / 24 / 36 月為止｜M1-b｜任一）、12 個月累積超額為負的機率：")
    for w, d in o["worlds"].items():
        a = " / ".join(f"{d['p_a_by'][h]*100:.1f}%" for h in CHECKS)
        print(f"    {w}：{a}｜{d['p_b']*100:.1f}%｜{d['p_any']*100:.1f}%｜12 月為負 {d['p_neg12']*100:.1f}%")


def bands():
    if BANDS.exists() and "--force" not in sys.argv:
        raise SystemExit(f"{BANDS} 已存在；M1 門檻寫定後不重算（看到實盤後改線＝事後調參）。確定要重算加 --force")
    s = series()
    live = s[s.index >= LIVE_START]
    if len(live):
        print(f"⚠ 已有 {len(live)} 個實盤月，bands 只用 {LIVE_START.date()} 之前的月份")
    s = s[s.index < LIVE_START]
    main_ = simulate(s, np.random.default_rng(SEED))
    sens = simulate(s[s.index >= "2020-01-01"], np.random.default_rng(SEED))
    report(main_, "主樣本（全期）")
    report(sens, "敏感度（2020 起）")
    BANDS.parent.mkdir(parents=True, exist_ok=True)
    BANDS.write_text(json.dumps(dict(made=time.strftime("%Y-%m-%d"), live_start=str(LIVE_START.date()),
                                     main=main_, sens=sens), ensure_ascii=False, indent=1, default=str))
    FC._log(dict(kind="monitor", period="dev+val+holdout", strategy="M1 H136 失效判準 bands",
                 note="實盤監控 M1 登記；holdout 2026-11-02 已開封，只描述；不算試驗帳",
                 n=main_["n"], mu_m=main_["mu_m"], kill=main_["kill"], dd_line=main_["dd_line"],
                 p_any={w: d["p_any"] for w, d in main_["worlds"].items()},
                 sens_kill=sens["kill"], sens_dd_line=sens["dd_line"]))
    print(f"\n已存 {BANDS}")


def check():
    b = json.loads(BANDS.read_text())["main"]
    s = series()
    live = s[s.index >= LIVE_START]
    if live.empty:
        print(f"還沒有完整的實盤月（第一個實盤月要等下一次換股開盤才算完）。")
        return
    eq = (1 + live["net"]).cumprod()
    dd = float((eq / np.maximum.accumulate(np.r_[1.0, eq.values])[1:] - 1).min())
    cum = live["ex"].cumsum()
    n = len(live)
    print(f"\n== 實盤 {n} 個月（{live.index[0].date()} 起）：累積超額 {pct(cum.iloc[-1])}，net 累積 {pct(eq.iloc[-1]-1)}，MDD {pct(dd)}")
    for d, r in live.iterrows():
        print(f"  {d.date()}  net {pct(r['net'])}  超額 {pct(r['ex'])}  累積超額 {pct(cum.loc[d])}")
    flags = []
    for h in CHECKS:
        if n >= h:
            ok = cum.iloc[h - 1] >= b["kill"][str(h)]
            print(f"  M1-a 第 {h} 月：{pct(cum.iloc[h-1])} vs 門檻 {pct(b['kill'][str(h)])} → {'通過' if ok else '⚠ 觸發檢討'}")
            flags += [] if ok else [f"M1-a{h}"]
    nxt = next((h for h in CHECKS if n < h), None)
    if nxt:
        print(f"  下一個檢查點：第 {nxt} 月（門檻 {pct(b['kill'][str(nxt)])}）")
    ok = dd >= b["dd_line"]
    print(f"  M1-b 回撤：{pct(dd)} vs 門檻 {pct(b['dd_line'])} → {'通過' if ok else '⚠ 觸發檢討'}")
    flags += [] if ok else ["M1-b"]
    print(f"\n結論：{'⚠ ' + '、'.join(flags) + ' 觸發檢討（不是自動停用，先查執行與資料）' if flags else '未觸發，照規則繼續、不改規則'}")


if __name__ == "__main__":
    {"bands": bands, "check": check}[sys.argv[1] if len(sys.argv) > 1 else "check"]()
