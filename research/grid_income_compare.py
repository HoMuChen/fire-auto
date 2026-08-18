"""
期貨網格收租設計比較 — 目標：跨 2022 熊市仍穩定現金流

對症 2022「22 個月乾旱」的病根（保證金凍結 + 高檔快速買滿），比較幾種期貨專屬解法：
  1. 趨勢濾網（跌破長均線停止開新倉）
  2. 容量預留（降低 max_lots，保留保證金緩衝以能買到底）
  3. 加大格距（買得慢，攤在整段下跌）
  4. 組合
評分重點（穩定現金流）：最長乾旱↓、正報酬月%↑、MDD↓，其次 CAGR。
"""
import sys
import statistics
from types import SimpleNamespace
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
import tsmc_futures_grid_30min as g


def base(**kw):
    p = SimpleNamespace(capital=1_000_000, max_lots=20, contracts_per_lot=1,
                        step=0.01, take=0.01, h_days=10,
                        fee_per_side=20.0, roll_cost_points=2.0,
                        max_leverage=1.0, regime_ma=0, intraday_once=False,
                        topup=False, initial_margin=0.135, maintenance_margin=0.1035,
                        start=None, atr_mult=0.0, trail_tp=0.0,
                        widen=0.0, rsi_gate=0.0, rsi_period=70, depth_scale=0.0,
                        depth_ref_bars=600, shallow_boost=0.0, shallow_taper=0.15,
                        derisk_lev=0.0)
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def evaluate(p):
    r = g.run(p)
    dt, curve = r["dt"], r["curve"]
    ms = g.months_of(dt)
    cap = p.capital
    vals = [r["mcf"].get(m, 0.0) / cap * 100 for m in ms]
    pos = sum(1 for v in vals if v > 0.05)
    dry = mx = 0
    for v in vals:
        dry = dry + 1 if v < 0.3 else 0
        mx = max(mx, dry)
    years = len(dt) / (g.BARS_PER_DAY * 252)
    cagr = ((curve[-1] / cap) ** (1 / years) - 1) * 100 if years > 0 else 0
    return {
        "cagr": cagr, "mdd": g.maxdd(curve),
        "med": statistics.median(vals), "mean": statistics.mean(vals),
        "pos_pct": pos / len(ms) * 100, "drought": mx,
        "max_lots": r["max_lots"], "max_lev": r["max_lev"],
        "trades": sum(r["mbuy"].values()) + sum(r["msell"].values()),
    }


CONFIGS = [
    ("基準 20批/1%/1x", base()),
    ("趨勢濾網 MA200(20日)", base(regime_ma=200)),
    ("趨勢濾網 MA600(60日)", base(regime_ma=600)),
    ("趨勢濾網 MA1200(120日)", base(regime_ma=1200)),
    ("容量預留 8批", base(max_lots=8)),
    ("容量預留 10批", base(max_lots=10)),
    ("加大格距 2%", base(step=0.02, take=0.02)),
    ("加大格距 3%", base(step=0.03, take=0.03)),
    ("濾網MA600 + 8批", base(regime_ma=600, max_lots=8)),
    ("濾網MA600 + 格距2%", base(regime_ma=600, step=0.02, take=0.02)),
    ("濾網MA600 + 10批 + 2%", base(regime_ma=600, max_lots=10, step=0.02, take=0.02)),
    # ── 資金預留（更大本金＝更低有效槓桿＝熊市能買到底，不凍結）──
    ("資金預留 本金200萬", base(capital=2_000_000)),
    ("資金預留 本金300萬", base(capital=3_000_000)),
    ("本金200萬 + 格距2%", base(capital=2_000_000, step=0.02, take=0.02)),
    ("本金300萬 + 格距2%", base(capital=3_000_000, step=0.02, take=0.02)),
    ("本金200萬 + 濾網MA600", base(capital=2_000_000, regime_ma=600)),
    ("本金300萬 + 30批", base(capital=3_000_000, max_lots=30)),
    ("本金300萬 + 40批 + 格距1.5%", base(capital=3_000_000, max_lots=40, step=0.015, take=0.015)),
]


def main():
    print("=" * 108)
    print("  期貨網格收租設計比較（2021-02 ~ 2026-08，完整含 2022 熊市；無槓桿 1x）")
    print("=" * 108)
    print(f"  {'設計':<26}{'CAGR':>7}{'MDD':>8}{'月中位':>8}{'月均':>7}"
          f"{'正月%':>7}{'最長乾旱':>9}{'最多批':>7}{'交易':>7}")
    print("  " + "-" * 104)
    for name, p in CONFIGS:
        m = evaluate(p)
        print(f"  {name:<26}{m['cagr']:>+6.1f}%{m['mdd']:>7.1f}%{m['med']:>+7.2f}%"
              f"{m['mean']:>+6.2f}%{m['pos_pct']:>6.0f}%{m['drought']:>7.0f}月"
              f"{m['max_lots']:>6}批{m['trades']:>7}")
    print("=" * 108)
    print("  目標：最長乾旱↓、正月%↑、MDD↓（穩定現金流優先於絕對報酬）")


if __name__ == "__main__":
    main()
