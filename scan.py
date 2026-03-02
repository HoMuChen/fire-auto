"""
每日信號掃描器 — 盤後更新股價，掃描三策略篩選名單

用法：
    python3 scan.py              # 更新股價 + 掃描
    python3 scan.py --no-update  # 跳過股價更新（已更新過）
    python3 scan.py --json       # 輸出 JSON 格式
"""

import json
import sys
from datetime import datetime
from pathlib import Path

from backtest import (
    read_prices, calc_sma, calc_rsi, calc_kd, calc_ad_line,
    calc_bollinger, calc_keltner, calc_adx,
)

BASE_DIR = Path(__file__).parent
FILTERED_PATH = BASE_DIR / "strategies" / "filtered_stock_lists.json"
STOCKS_PATH = BASE_DIR / "individual_stocks.json"


# ═══════════════════════════════════════════════════════════════
#  指標 → 價格 反算
# ═══════════════════════════════════════════════════════════════

def _kd_price_threshold(prices, i, k_current, target_k=30, k_period=9):
    """估算明天收盤價需低於多少，KD(9) 才會 ≤ target_k"""
    # K_tmr = K_today * 2/3 + raw_k * 1/3
    # raw_k_needed = (target_k - K_today * 2/3) * 3 = target_k*3 - K_today*2
    raw_k_needed = target_k * 3 - k_current * 2
    if raw_k_needed < 0 or raw_k_needed > 100:
        return None
    # 明天的 9 日窗口 ≈ prices[i-7:i+1]（去最舊加明天，近似用今天的）
    window = prices[max(0, i - k_period + 2):i + 1]
    if len(window) < 2:
        return None
    high_9 = max(p["high"] for p in window)
    low_9 = min(p["low"] for p in window)
    r = high_9 - low_9
    if r <= 0:
        return None
    return round(low_9 + (raw_k_needed / 100) * r, 1)


def _rsi_price_threshold(closes, target_rsi=45, period=14):
    """估算明天收盤價需低於多少，RSI(14) 才會 ≤ target_rsi"""
    if len(closes) < period + 1:
        return None
    # 計算當前 avg_gain / avg_loss
    avg_gain = 0.0
    avg_loss = 0.0
    for j in range(1, period + 1):
        change = closes[j] - closes[j - 1]
        avg_gain += max(change, 0)
        avg_loss += max(-change, 0)
    avg_gain /= period
    avg_loss /= period
    for j in range(period + 1, len(closes)):
        change = closes[j] - closes[j - 1]
        avg_gain = (avg_gain * (period - 1) + max(change, 0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-change, 0)) / period
    # 假設明天下跌 loss: new_avg_gain = avg_gain*(p-1)/p, new_avg_loss = (avg_loss*(p-1)+loss)/p
    # RS = new_avg_gain / new_avg_loss = target_rsi / (100 - target_rsi)
    target_rs = target_rsi / (100 - target_rsi)
    if target_rs <= 0:
        return None
    loss_needed = (period - 1) * (avg_gain / target_rs - avg_loss)
    if loss_needed <= 0:
        return round(closes[-1], 1)  # 已經低於目標
    return round(closes[-1] - loss_needed, 1)


def load_stock_names():
    with open(STOCKS_PATH) as f:
        return {s["stock_id"]: s["stock_name"] for s in json.load(f)}


def load_filtered_lists():
    with open(FILTERED_PATH) as f:
        data = json.load(f)
    return {
        "squeeze": [s["stock_id"] for s in data["strategies"]["squeeze"]["kept"]],
        "oversold": [s["stock_id"] for s in data["strategies"]["oversold"]["kept"]],
        "ad_divergence": [s["stock_id"] for s in data["strategies"]["ad_divergence"]["kept"]],
    }


# ═══════════════════════════════════════════════════════════════
#  波動率擠壓掃描
# ═══════════════════════════════════════════════════════════════

def scan_squeeze(stock_ids, names):
    results = []
    for sid in stock_ids:
        try:
            prices = read_prices(sid)
        except Exception:
            continue
        if len(prices) < 30:
            continue

        closes = [p["close"] for p in prices]
        vols = [p["volume"] for p in prices]
        upper_bb, lower_bb, _ = calc_bollinger(closes, 20, 2)
        upper_kc, lower_kc, _ = calc_keltner(prices, 20, 1.5)
        adx = calc_adx(prices, 14)
        vol_ma = calc_sma([float(v) for v in vols], 20)
        n = len(prices)

        # 擠壓狀態
        in_sq = [False] * n
        for i in range(n):
            if all(v is not None for v in [upper_bb[i], lower_bb[i], upper_kc[i], lower_kc[i]]):
                in_sq[i] = upper_bb[i] < upper_kc[i] and lower_bb[i] > lower_kc[i]
        sq_count = [0] * n
        for i in range(1, n):
            sq_count[i] = sq_count[i - 1] + 1 if in_sq[i] else 0

        i = n - 1
        if i < 6:
            continue

        # 條件檢查（與 backtest.py strategy_squeeze 完全一致）
        currently_squeezing = in_sq[i]
        squeeze_days = sq_count[i]
        just_released = not in_sq[i] and sq_count[i - 1] >= 5

        bb_breakout = upper_bb[i] is not None and closes[i] > upper_bb[i]
        adx_ok = adx[i] is not None and adx[i] > 18
        vol_ok = vol_ma[i] is not None and vol_ma[i] > 0 and vols[i] > vol_ma[i] * 1.4
        vol_ratio = vols[i] / vol_ma[i] if vol_ma[i] and vol_ma[i] > 0 else 0
        mom5 = (closes[i] - closes[i - 5]) / closes[i - 5] if closes[i - 5] > 0 else 0
        mom_ok = mom5 > 0

        bb_dist = (closes[i] / upper_bb[i] - 1) if upper_bb[i] else None

        # 完全觸發
        triggered = just_released and bb_breakout and adx_ok and vol_ok and mom_ok

        # 明天觸發條件（用於 squeezing / released_miss）
        vol_threshold = vol_ma[i] * 1.4 if vol_ma[i] else 0
        vol_threshold_張 = round(vol_threshold / 1000)
        # 明天的5日動量參考價 = closes[i-4]（明天i+1, 往回5天=i-4）
        mom5_ref = closes[i - 4] if i >= 4 and closes[i - 4] > 0 else None

        if triggered:
            results.append({
                "stock_id": sid, "name": names.get(sid, ""),
                "status": "triggered", "date": prices[i]["date"],
                "close": closes[i],
                "squeeze_days": sq_count[i - 1],
                "bb_upper": round(upper_bb[i], 1) if upper_bb[i] else None,
                "bb_dist_pct": round(bb_dist * 100, 1) if bb_dist is not None else None,
                "adx": round(adx[i], 1) if adx[i] else None,
                "vol_ratio": round(vol_ratio, 1),
                "mom5_pct": round(mom5 * 100, 1),
            })
        elif currently_squeezing and squeeze_days >= 3:
            next_day = _squeeze_next_day(
                upper_bb[i], upper_kc[i], adx[i], adx_ok,
                vol_threshold_張, mom5_ref, mom_ok, mom5,
            )
            results.append({
                "stock_id": sid, "name": names.get(sid, ""),
                "status": "squeezing", "date": prices[i]["date"],
                "close": closes[i],
                "squeeze_days": squeeze_days,
                "bb_upper": round(upper_bb[i], 1) if upper_bb[i] else None,
                "bb_dist_pct": round(bb_dist * 100, 1) if bb_dist is not None else None,
                "adx": round(adx[i], 1) if adx[i] else None,
                "vol_ratio": round(vol_ratio, 1),
                "mom5_pct": round(mom5 * 100, 1),
                "next_day": next_day,
            })
        elif just_released:
            # 脫離但沒全部達標
            missing = []
            if not bb_breakout:
                missing.append("突破上軌")
            if not adx_ok:
                missing.append("趨勢不足")
            if not vol_ok:
                missing.append("量不足")
            if not mom_ok:
                missing.append("動量不足")
            next_day = _squeeze_released_next_day(
                upper_bb[i], bb_breakout, adx[i], adx_ok,
                vol_threshold_張, vol_ok, mom5_ref, mom_ok, mom5,
            )
            results.append({
                "stock_id": sid, "name": names.get(sid, ""),
                "status": "released_miss", "date": prices[i]["date"],
                "close": closes[i],
                "squeeze_days": sq_count[i - 1],
                "bb_upper": round(upper_bb[i], 1) if upper_bb[i] else None,
                "bb_dist_pct": round(bb_dist * 100, 1) if bb_dist is not None else None,
                "adx": round(adx[i], 1) if adx[i] else None,
                "vol_ratio": round(vol_ratio, 1),
                "mom5_pct": round(mom5 * 100, 1),
                "missing": missing,
                "next_day": next_day,
            })

    # 排序：觸發 > 脫離未達 > 擠壓中（按天數遞減）
    order = {"triggered": 0, "released_miss": 1, "squeezing": 2}
    results.sort(key=lambda r: (order.get(r["status"], 9), -r["squeeze_days"]))
    return results


def _squeeze_next_day(bb_upper, kc_upper, adx_val, adx_ok,
                      vol_threshold_張, mom5_ref, mom_ok, mom5):
    """擠壓中 → 明天觸發需要什麼（只列待達成條件，純價量）"""
    conds = []
    warning = None
    if bb_upper and kc_upper:
        gap_pct = (kc_upper - bb_upper) / kc_upper * 100
        conds.append(f"波動帶需先擴張（目前差{gap_pct:.1f}%）")
    conds.append(f"收盤 > ~{bb_upper:.1f}" if bb_upper else "收盤突破上軌")
    if not adx_ok:
        warning = f"趨勢強度偏弱（{adx_val:.0f}/18），即使突破也不觸發" if adx_val else "趨勢強度不足"
    conds.append(f"量 > {vol_threshold_張:,}張")
    if not mom_ok and mom5_ref is not None:
        conds.append(f"收盤 > {mom5_ref:.1f}（5日前價位）")
    return {"action": conds, "warning": warning}


def _squeeze_released_next_day(bb_upper, bb_ok, adx_val, adx_ok,
                               vol_threshold_張, vol_ok, mom5_ref, mom_ok, mom5):
    """脫離擠壓但未全達 → 列缺少條件（純價量）"""
    conds = []
    warning = None
    if not bb_ok:
        conds.append(f"收盤 > {bb_upper:.1f}" if bb_upper else "收盤突破上軌")
    if not adx_ok:
        warning = f"趨勢強度偏弱（{adx_val:.0f}/18），即使突破也不觸發" if adx_val else "趨勢強度不足"
    if not vol_ok:
        conds.append(f"量 > {vol_threshold_張:,}張")
    if not mom_ok and mom5_ref is not None:
        conds.append(f"收盤 > {mom5_ref:.1f}（5日前價位）")
    return {"action": conds, "warning": warning}


# ═══════════════════════════════════════════════════════════════
#  超跌反彈掃描
# ═══════════════════════════════════════════════════════════════

def scan_oversold(stock_ids, names):
    results = []
    for sid in stock_ids:
        try:
            prices = read_prices(sid)
        except Exception:
            continue
        if len(prices) < 65:
            continue

        closes = [p["close"] for p in prices]
        opens = [p["open"] for p in prices]
        vols = [p["volume"] for p in prices]
        sma60 = calc_sma(closes, 60)
        k, _ = calc_kd(prices, 9)
        vol_ma = calc_sma([float(v) for v in vols], 20)
        n = len(prices)
        i = n - 1

        if sma60[i] is None or k[i] is None or vol_ma[i] is None or vol_ma[i] == 0:
            continue

        # 條件檢查（與 backtest.py strategy_oversold_reversal 完全一致）
        above_ma = closes[i] > sma60[i]
        drop5 = (closes[i - 5] - closes[i - 1]) / closes[i - 5] if closes[i - 5] > 0 else 0
        kd_val = k[i]
        kd_low = kd_val < 30
        is_red = closes[i] > opens[i]
        vol_shrink = vols[i] < vol_ma[i] * 0.8
        vol_ratio = vols[i] / vol_ma[i]

        conditions = {
            "60均之上": above_ma,
            "5日跌5%": drop5 >= 0.05,
            "短線超賣": kd_low,
            "紅K": is_red,
            "量縮": vol_shrink,
        }
        met = sum(conditions.values())
        triggered = met == 5

        # 至少在均線上 + 近期有下跌 + KD 偏低才列入
        if not above_ma:
            continue
        if drop5 < 0.03 and kd_val >= 40:
            continue

        if met >= 3 or triggered:
            missing = [k for k, v in conditions.items() if not v]

            # KD → 價格門檻（明天收盤需低於多少才能讓 KD < 30）
            kd_price = None
            if not kd_low:
                kd_price = _kd_price_threshold(prices, i, kd_val, target_k=30)

            # 明天觸發條件
            next_day = []
            if not triggered:
                next_day = _oversold_next_day(
                    closes, i, sma60[i], above_ma,
                    drop5, kd_val, kd_low, kd_price, vol_ma[i],
                )

            results.append({
                "stock_id": sid, "name": names.get(sid, ""),
                "status": "triggered" if triggered else "near",
                "date": prices[i]["date"],
                "close": closes[i],
                "sma60": round(sma60[i], 1),
                "drop5d_pct": round(drop5 * 100, 1),
                "kd": round(kd_val, 0),
                "vol_ratio": round(vol_ratio, 2),
                "is_red": is_red,
                "met": met, "total": 5,
                "missing": missing,
                "next_day": next_day,
            })

    order = {"triggered": 0, "near": 1}
    results.sort(key=lambda r: (order.get(r["status"], 9), -r["met"]))
    return results


def _oversold_next_day(closes, i, sma60_val, above_ma,
                       drop5_today, kd_val, kd_low, kd_price, vol_ma_val):
    """超跌反彈 → 只列待達成條件（純價量）"""
    conds = []
    warning = None

    # 明天的5日跌幅
    if i >= 4 and closes[i - 4] > 0:
        drop5_tmr = (closes[i - 4] - closes[i]) / closes[i - 4]
        drop5_today_met = drop5_today >= 0.05
        drop5_tmr_met = drop5_tmr >= 0.05

        if not drop5_tmr_met:
            tmr_str = f"跌{drop5_tmr:.1%}" if drop5_tmr > 0 else f"反漲{abs(drop5_tmr):.1%}"
            if drop5_today_met:
                warning = f"5日跌幅窗口滑出！今天{drop5_today:.1%}達標 → 明天{tmr_str}，不再滿足"
            else:
                conds.append(f"5日跌≥5%（明天{tmr_str}，不滿足）")

    if not above_ma:
        conds.append(f"收盤 > {sma60_val:.1f}（60日均價）")

    if not kd_low:
        if kd_price is not None:
            conds.append(f"收盤 < {kd_price:.1f}（短線超賣價位）")
        else:
            conds.append(f"短線動能需再降（目前偏高 {kd_val:.0f}/30）")

    conds.append("收紅K（收盤 > 開盤）")

    vol_threshold = vol_ma_val * 0.8 if vol_ma_val else 0
    conds.append(f"量 < {vol_threshold / 1000:,.0f}張")

    return {"action": conds, "warning": warning}


# ═══════════════════════════════════════════════════════════════
#  AD背離掃描
# ═══════════════════════════════════════════════════════════════

def scan_ad_divergence(stock_ids, names):
    results = []
    for sid in stock_ids:
        try:
            prices = read_prices(sid)
        except Exception:
            continue
        if len(prices) < 65:
            continue

        closes = [p["close"] for p in prices]
        n = len(prices)
        ad = calc_ad_line(prices)
        rsi14 = calc_rsi(closes, 14)
        sma60 = calc_sma(closes, 60)
        i = n - 1

        if sma60[i] is None or rsi14[i] is None or i < 25:
            continue

        # 條件檢查（與 backtest.py strategy_ad_divergence 完全一致）
        above_ma = closes[i] > sma60[i]
        ma_rising = sma60[i - 5] is not None and sma60[i] > sma60[i - 5]
        price_min20 = min(closes[i - 20:i])
        price_at_low = closes[i] <= price_min20
        price_dist = (closes[i] / price_min20 - 1) if price_min20 > 0 else 999
        ad_min20 = min(ad[i - 20:i])
        ad_not_low = ad[i] > ad_min20
        rsi_val = rsi14[i]
        rsi_ok = rsi_val < 45

        conditions = {
            "60均之上": above_ma,
            "均線上升": ma_rising,
            "20日新低": price_at_low,
            "量價背離": ad_not_low,
            "動能轉弱": rsi_ok,
        }
        met = sum(conditions.values())
        triggered = met == 5

        # 至少在均線上 + MA上升 + 接近低點才列入
        if not (above_ma and ma_rising):
            continue
        price_near_low = price_dist <= 0.02  # 距20日低點2%以內
        if not price_near_low and not price_at_low:
            continue
        if rsi_val >= 50:
            continue

        if met >= 3 or triggered:
            missing = [k for k, v in conditions.items() if not v]

            # RSI → 價格門檻（明天收盤需低於多少才能讓 RSI < 45）
            rsi_price = None
            if not rsi_ok:
                rsi_price = _rsi_price_threshold(closes, target_rsi=45)

            # 明天觸發條件
            next_day = []
            if not triggered:
                next_day = _ad_next_day(
                    closes, i, ad, sma60[i], above_ma, ma_rising,
                    price_at_low, ad_not_low, ad_min20, rsi_val, rsi_ok, rsi_price,
                )

            results.append({
                "stock_id": sid, "name": names.get(sid, ""),
                "status": "triggered" if triggered else "near",
                "date": prices[i]["date"],
                "close": closes[i],
                "price_min20": round(price_min20, 1),
                "price_dist_pct": round(price_dist * 100, 1),
                "ad_divergence": ad_not_low,
                "rsi": round(rsi_val, 0),
                "met": met, "total": 5,
                "missing": missing,
                "next_day": next_day,
            })

    order = {"triggered": 0, "near": 1}
    results.sort(key=lambda r: (order.get(r["status"], 9), -r["met"]))
    return results


def _ad_next_day(closes, i, ad, sma60_val, above_ma, ma_rising,
                 price_at_low, ad_not_low, ad_min20, rsi_val, rsi_ok, rsi_price):
    """AD背離 → 只列待達成條件（純價量）"""
    conds = []
    warning = None

    if not above_ma:
        conds.append(f"收盤 > {sma60_val:.1f}（60日均價）")

    if not ma_rising:
        conds.append("60日均線需上升中")

    # 明天的20日新低門檻
    new_min20 = min(closes[i - 19:i + 1]) if i >= 19 else min(closes[:i + 1])
    if not price_at_low:
        conds.append(f"收盤 ≤ {new_min20:.1f}（創20日新低）")

    if not ad_not_low:
        warning = "需成交量配合：下跌時量縮（賣壓減弱），才能形成量價背離"

    if not rsi_ok:
        if rsi_price is not None:
            conds.append(f"收盤 < {rsi_price:.1f}（動能轉弱價位）")
        else:
            conds.append(f"動能需再降（目前偏高 {rsi_val:.0f}/45）")

    return {"action": conds, "warning": warning}


# ═══════════════════════════════════════════════════════════════
#  輸出格式
# ═══════════════════════════════════════════════════════════════

def _print_next_day(r):
    """印出明天觸發條件（如果有）"""
    nd = r.get("next_day")
    if not nd:
        return
    action = nd.get("action", []) if isinstance(nd, dict) else nd
    warning = nd.get("warning") if isinstance(nd, dict) else None
    if action:
        print(f"         → 進場條件：{' + '.join(action)}")
    if warning:
        print(f"         → ⚠ {warning}")


def print_squeeze(results):
    triggered = [r for r in results if r["status"] == "triggered"]
    released = [r for r in results if r["status"] == "released_miss"]
    squeezing = [r for r in results if r["status"] == "squeezing"]

    print("=" * 80)
    print("  波動率擠壓")
    print("=" * 80)

    if triggered:
        print(f"\n  *** 觸發買入 ({len(triggered)} 檔) ***")
        _print_sq_table(triggered)

    if released:
        print(f"\n  脫離擠壓但未全部達標 ({len(released)} 檔)")
        _print_sq_table(released, show_missing=True)

    if squeezing:
        print(f"\n  擠壓中，等待突破 ({len(squeezing)} 檔)")
        _print_sq_table(squeezing)

    if not results:
        print("  目前沒有個股處於擠壓狀態")

    print()


def _print_sq_table(rows, show_missing=False):
    hdr = f"  {'代號':>6} {'名稱':<6} {'收盤':>7} {'距上軌':>7} {'趨勢':>5} {'量比':>5} {'動量':>6} {'擠壓天':>5}"
    if show_missing:
        hdr += "  缺少"
    print(hdr)
    print(f"  {'-' * (len(hdr) - 2)}")
    for r in rows:
        bb_s = f"{r['bb_dist_pct']:+.1f}%" if r["bb_dist_pct"] is not None else "N/A"
        adx_s = f"{r['adx']:.0f}" if r["adx"] else "N/A"
        line = f"  {r['stock_id']:>6} {r['name']:<6} {r['close']:>7.1f} {bb_s:>7} {adx_s:>5} {r['vol_ratio']:>4.1f}x {r['mom5_pct']:>+5.1f}% {r['squeeze_days']:>5}"
        if show_missing:
            line += f"  {', '.join(r.get('missing', []))}"
        print(line)
        _print_next_day(r)


def print_oversold(results):
    triggered = [r for r in results if r["status"] == "triggered"]
    near = [r for r in results if r["status"] == "near"]

    print("=" * 80)
    print("  超跌反彈")
    print("=" * 80)

    if triggered:
        print(f"\n  *** 觸發買入 ({len(triggered)} 檔) ***")
        _print_os_table(triggered)

    if near:
        print(f"\n  接近觸發 ({len(near)} 檔)")
        _print_os_table(near)

    if not results:
        print("  目前沒有個股接近超跌反彈條件")

    print()


def _print_os_table(rows):
    print(f"  {'代號':>6} {'名稱':<6} {'收盤':>7} {'60均':>7} {'5日跌':>6} {'超賣':>4} {'量比':>6} {'紅K':>3} {'達成':>4}  缺少")
    print(f"  {'-' * 75}")
    for r in rows:
        red = "Y" if r["is_red"] else "N"
        miss = ", ".join(r["missing"])
        print(f"  {r['stock_id']:>6} {r['name']:<6} {r['close']:>7.1f} {r['sma60']:>7.1f} {r['drop5d_pct']:>5.1f}% {r['kd']:>4.0f} {r['vol_ratio']:>5.2f}x  {red:>2} {r['met']:>1}/{r['total']}  {miss}")
        _print_next_day(r)


def print_ad(results):
    triggered = [r for r in results if r["status"] == "triggered"]
    near = [r for r in results if r["status"] == "near"]

    print("=" * 80)
    print("  AD背離")
    print("=" * 80)

    if triggered:
        print(f"\n  *** 觸發買入 ({len(triggered)} 檔) ***")
        _print_ad_table(triggered)

    if near:
        print(f"\n  接近觸發 ({len(near)} 檔)")
        _print_ad_table(near)

    if not results:
        print("  目前沒有個股接近 AD 背離條件")

    print()


def _print_ad_table(rows):
    print(f"  {'代號':>6} {'名稱':<6} {'收盤':>7} {'20日低':>7} {'距低點':>7} {'量價離':>6} {'動能':>5} {'達成':>4}  缺少")
    print(f"  {'-' * 75}")
    for r in rows:
        div = "Y" if r["ad_divergence"] else "N"
        miss = ", ".join(r["missing"])
        print(f"  {r['stock_id']:>6} {r['name']:<6} {r['close']:>7.1f} {r['price_min20']:>7.1f} {r['price_dist_pct']:>+6.1f}% {div:>6} {r['rsi']:>5.0f} {r['met']:>1}/{r['total']}  {miss}")
        _print_next_day(r)


def _next_trading_date(data_date):
    """推算下一個交易日：若今天 > 資料日期，用今天；否則跳週末"""
    from datetime import timedelta
    try:
        d = datetime.strptime(data_date, "%Y-%m-%d")
    except (ValueError, TypeError):
        return "下一交易日"
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    if today > d:
        # 資料是過去的，觀察日 = 今天（或下一個工作日）
        d = today
        while d.weekday() >= 5:
            d += timedelta(days=1)
    else:
        # 資料是今天的，觀察日 = 明天（跳週末）
        d += timedelta(days=1)
        while d.weekday() >= 5:
            d += timedelta(days=1)
    return d.strftime("%Y-%m-%d")


def print_summary(sq, os, ad, data_date):
    next_date = _next_trading_date(data_date)
    print("=" * 80)
    print(f"  掃描摘要")
    print(f"  資料日期：{data_date}｜下一交易日：{next_date}")
    print("=" * 80)
    sq_t = len([r for r in sq if r["status"] == "triggered"])
    os_t = len([r for r in os if r["status"] == "triggered"])
    ad_t = len([r for r in ad if r["status"] == "triggered"])
    total_t = sq_t + os_t + ad_t

    print(f"  波動率擠壓：{sq_t} 觸發, {len(sq) - sq_t} 觀察")
    print(f"  超跌反彈：  {os_t} 觸發, {len(os) - os_t} 觀察")
    print(f"  AD背離：    {ad_t} 觸發, {len(ad) - ad_t} 觀察")
    print(f"  合計：      {total_t} 檔觸發買入")

    if sq_t + os_t + ad_t > 0:
        print(f"\n  已觸發個股（{data_date} 收盤達成所有條件）：")
        for r in sq + os + ad:
            if r["status"] == "triggered":
                print(f"    {r['stock_id']} {r['name']}")

    watch = [r for r in sq + os + ad if r["status"] != "triggered" and r.get("next_day")]
    if watch:
        print(f"\n  {next_date} 觀察重點：")
        for r in watch:
            # 只提最關鍵的缺少條件
            missing = r.get("missing", [])
            if missing:
                print(f"    {r['stock_id']} {r['name']}（差 {', '.join(missing)}）")

    print()


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    args = sys.argv[1:]
    no_update = "--no-update" in args
    json_output = "--json" in args

    # 1) 更新股價
    if not no_update:
        from stock_cache import StockCache
        cache = StockCache()
        cache.update_today()
        print()

    # 2) 載入資料
    names = load_stock_names()
    lists = load_filtered_lists()

    # 3) 掃描
    sq = scan_squeeze(lists["squeeze"], names)
    os_results = scan_oversold(lists["oversold"], names)
    ad = scan_ad_divergence(lists["ad_divergence"], names)

    # 資料日期（取第一檔的最後日期）
    data_date = "unknown"
    for sid in lists["squeeze"][:5]:
        try:
            p = read_prices(sid)
            if p:
                data_date = p[-1]["date"]
                break
        except Exception:
            continue

    next_date = _next_trading_date(data_date)

    # 4) 輸出
    if json_output:
        output = {
            "scan_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "data_date": data_date,
            "next_trading_date": next_date,
            "squeeze": sq,
            "oversold": os_results,
            "ad_divergence": ad,
            "summary": {
                "squeeze_triggered": len([r for r in sq if r["status"] == "triggered"]),
                "oversold_triggered": len([r for r in os_results if r["status"] == "triggered"]),
                "ad_triggered": len([r for r in ad if r["status"] == "triggered"]),
            },
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print(f"\n  資料日期：{data_date}｜觀察條件適用於：{next_date}\n")
        print_squeeze(sq)
        print_oversold(os_results)
        print_ad(ad)
        print_summary(sq, os_results, ad, data_date)


if __name__ == "__main__":
    main()
