"""第十九批橫斷面合成（事後）：H104 = H2 + H99 趨勢因子、H105 = H29 + H97 營收公布窗；對照 H29（同期、同起點）。
用法：python3 research/v2/batch19_xs.py dev [H104 H105]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hypotheses as Hy  # noqa: E402
import xs  # noqa: E402

STARTS = {"H104": "2012-01-01", "H105": None}


def compare(R, Rb, name, bname="H29"):
    a, b = R["topn"].dropna(), Rb["topn"].dropna()
    ix = a.index.intersection(b.index)
    d = a[ix] - b[ix]
    by = d.groupby(d.index.year).sum()
    return f"{name} − {bname}：月差 {d.mean()*100:+.2f}%  t {d.mean()/d.std()*np.sqrt(len(d)):+.2f}  年勝 {int((by>0).sum())}/{len(by)}"


if __name__ == "__main__":
    period = sys.argv[1]
    want = [k for k in sys.argv[2:] if not k.startswith("--")] or ["H104", "H105"]
    P, rows, cmp_lines = None, [], []
    for k in want:
        st = STARTS[k]
        res, P = xs.run({k: Hy.FNS[k], "H29": Hy.FNS["H29"], "H2": Hy.FNS["H2"]}, period, P=P,
                        note=f"第十九批 事前登記（事後合成，對照 H29，起 {st or 'dev 起'}）", topn=20, start=st)
        for nm, (R, s) in res.items():
            for col in ("top_q", "topn"):
                rows.append(dict(批=k, H=nm, name=Hy.FNS[nm][1], port=col, **{a: b for a, b in s[col].items()}))
            rows[-1]["Q"] = " / ".join(f"{v*100:+.1f}" for v in s["Q_ex_ann"].values())
        cmp_lines.append(compare(res[k][0], res["H29"][0], k))
        cmp_lines.append(compare(res[k][0], res["H2"][0], k, "H2"))
    df = pd.DataFrame(rows)
    for c in ("CAGR", "bench_CAGR", "ex_ann", "MDD", "turnover"):
        df[c] = (df[c] * 100).round(1)
    pd.set_option("display.width", 250)
    print(f"[{period}]")
    print(df.round(2).to_string(index=False))
    print("\n對照（top20 月報酬差）")
    print("\n".join("  " + x for x in cmp_lines))
