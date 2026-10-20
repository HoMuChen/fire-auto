"""舊 14 策略在 v2 harness 的 dev 期（2010-2019）重驗。

這 14 個策略是在 2020-2026 資料上設計/挑出來的，從沒看過 2010-2019 → 這段對它們是真正的樣本外。
參數、停損完全照 backtest.py，不改。
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest as bt  # noqa: E402
import harness as H  # noqa: E402

OUT = H.BASE / "data/v2/results"

if __name__ == "__main__":
    period = sys.argv[1] if len(sys.argv) > 1 else "dev"
    specs = {k: {"fn": v["fn"], "stop": v["stop"]["pct"], "desc": v["desc"]} for k, v in bt.STRATEGIES.items()}
    T, S = H.run(specs, period, note="legacy14 原參數重驗")
    OUT.mkdir(exist_ok=True)
    T.to_parquet(OUT / f"legacy14_{period}.parquet")
    pd.set_option("display.width", 220)
    f = S.copy()
    for c in ("win", "avg", "ex", "edge"):
        f[c] = (f[c] * 100).round(2)
    print(f.round(2).sort_values("edge", ascending=False).to_string(index=False))
    G = H.by_group(T, specs)
    print("\n分群（edge = 超額 − 同群組隨機對照，pp）")
    print(G.pivot(index="strategy", columns="group", values="edge").round(2).to_string())
