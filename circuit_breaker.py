"""
系統性風險斷路器 — 與 backtest 投組模擬（/tmp/combo_2022.py 的 CFG）一致

回測規則：
    squeeze : 無限制（daily=∞, skip=∞）
    oversold: 每日限買 1 檔；同日 buy 訊號 > 3 檔 → 全跳過
    ad      : 每日限買 1 檔；同日 buy 訊號 > 3 檔 → 全跳過

邏輯（對應回測）：
    todays = [名單內、今日 buy、未持有的股]   # 依名單順序
    if len(todays) > skip: todays = []          # 同日 > skip 全跳過
    買進名單順序前 daily 檔

擠壓是順勢策略（突破本來就該追），不設斷路。
超跌 / AD 是逆勢策略（接刀），同日大量觸發代表系統性下殺，全跳過。
"""

# strategy_key -> {daily, skip}；None 表示無限制
BREAKER = {
    "squeeze":  {"daily": None, "skip": None},
    "oversold": {"daily": 1, "skip": 3},
    "ad":       {"daily": 1, "skip": 3},
}


def is_systemic(strategy_key: str, n_triggered: int) -> bool:
    """同日觸發數是否超過 skip 門檻（系統性下殺，全跳過）"""
    cfg = BREAKER.get(strategy_key)
    if not cfg or cfg["skip"] is None:
        return False
    return n_triggered > cfg["skip"]


def allowed_set(strategy_key: str, triggered_ids: list[str]) -> list[str]:
    """套用斷路規則後，允許進場 / 通知的股票代號 list。

    triggered_ids 必須**已依想要的優先順序排好**：
      - scan.py（日掃）：依濾網名單順序 → 取名單第一檔（對齊回測）
      - monitor.py（盤中）：依當日首次觸發時間順序 → 取最早觸發那檔

    回傳空 list 代表系統性下殺全跳過。
    """
    cfg = BREAKER.get(strategy_key, {"daily": None, "skip": None})
    skip, daily = cfg["skip"], cfg["daily"]
    if skip is not None and len(triggered_ids) > skip:
        return []                       # 同日 > skip → 全跳過
    if daily is None:
        return list(triggered_ids)      # 無限制（擠壓）
    return list(triggered_ids[:daily])  # 每日限 daily 檔


if __name__ == "__main__":
    # ── 自測：對齊回測語意 ──
    # 擠壓無限制
    assert allowed_set("squeeze", ["a", "b", "c", "d", "e"]) == ["a", "b", "c", "d", "e"]
    assert is_systemic("squeeze", 99) is False
    # 超跌 / AD：1 檔 → 取 1
    assert allowed_set("oversold", ["a"]) == ["a"]
    # 3 檔（= skip，未超過）→ 取第一檔
    assert allowed_set("oversold", ["a", "b", "c"]) == ["a"]
    assert is_systemic("oversold", 3) is False
    # 4 檔（> skip）→ 全跳過
    assert allowed_set("oversold", ["a", "b", "c", "d"]) == []
    assert is_systemic("oversold", 4) is True
    assert allowed_set("ad", ["x", "y", "z", "w"]) == []
    assert allowed_set("ad", ["x", "y"]) == ["x"]
    assert is_systemic("ad", 2) is False
    # 0 檔
    assert allowed_set("oversold", []) == []
    print("circuit_breaker 自測通過 ✓")
