"""
篩選適合網格的「高波動 × 區間震盪」個股。

條件：
- 有 6 年資料、流動性足（low_liquidity=False）
- 高波動：年化波動率在全體前 1/3
- 區間震盪：趨勢 R² < 0.3（非單邊趨勢）
- 沒有單邊大漲大跌：期間總報酬介於 -40% ~ +80%（排除大多頭/大空頭）
排序：震盪強度（路徑長度/淨位移）由高到低 —— 越高=來回擺盪越多，網格越有得賺。
"""
import sys, json, math, csv
sys.path.insert(0, "/Users/mu/fire-auto")
import validate_strategy_groups as vsg

names = {s["stock_id"]: s["stock_name"] for s in json.load(open("/Users/mu/fire-auto/individual_stocks.json"))}


def features(prices):
    closes = [p["close"] for p in prices]
    n = len(closes)
    rets = [closes[i] / closes[i-1] - 1 for i in range(1, n) if closes[i-1] > 0]
    import statistics
    ann_vol = statistics.pstdev(rets) * math.sqrt(252)
    # 趨勢 R²（log price vs t 線性擬合）
    xs = list(range(n)); ys = [math.log(c) for c in closes if c > 0]
    if len(ys) != n:
        return None
    mx = sum(xs)/n; my = sum(ys)/n
    sxx = sum((x-mx)**2 for x in xs); sxy = sum((xs[i]-mx)*(ys[i]-my) for i in range(n))
    syy = sum((y-my)**2 for y in ys)
    r2 = (sxy*sxy)/(sxx*syy) if sxx>0 and syy>0 else 0
    total_ret = closes[-1]/closes[0] - 1
    # 震盪強度 = 路徑長度(累積絕對日變動) / |淨位移|
    path = sum(abs(r) for r in rets)
    disp = abs(math.log(closes[-1]/closes[0])) if closes[0]>0 and closes[-1]>0 else 1e-9
    chop = path / max(disp, 1e-6)
    return dict(ann_vol=ann_vol, r2=r2, total_ret=total_ret, chop=chop, avg_abs=sum(abs(r) for r in rets)/len(rets))


def main():
    stocks = json.load(open("/Users/mu/fire-auto/individual_stocks.json"))
    liquid = [s for s in stocks if s.get("low_liquidity") is False]
    rows = []
    for s in liquid:
        sid = s["stock_id"]
        path = vsg.DATA_DIR / f"{sid}.csv"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            first = next(csv.DictReader(f), None)
        if not first or first["date"] > "2020-01-31":
            continue
        try:
            prices = vsg.read_prices(sid)
        except Exception:
            continue
        if len(prices) < 1000 or prices[0]["close"] == 0:
            continue
        f = features(prices)
        if f:
            f["sid"] = sid
            rows.append(f)

    vols = sorted(r["ann_vol"] for r in rows)
    vol_p67 = vols[int(len(vols)*0.67)]
    cand = [r for r in rows
            if r["ann_vol"] >= vol_p67 and r["r2"] < 0.3 and -0.40 <= r["total_ret"] <= 0.80]
    cand.sort(key=lambda r: r["chop"], reverse=True)

    print(f"全體 {len(rows)} 檔；高波動門檻(年化) ≥ {vol_p67:.0%}；符合「高波動×低趨勢×區間」{len(cand)} 檔\n")
    print(f"{'代號':<6}{'名稱':<9}{'年化波動':>8}{'趨勢R²':>8}{'總報酬':>8}{'日均振幅':>8}{'震盪強度':>8}")
    print("-" * 60)
    for r in cand[:20]:
        print(f"{r['sid']:<6}{names.get(r['sid'],''):<9}{r['ann_vol']:>7.0%}{r['r2']:>8.2f}"
              f"{r['total_ret']:>+7.0%}{r['avg_abs']:>7.1%}{r['chop']:>8.1f}")


if __name__ == "__main__":
    main()
