# 台積電網格研究 — 實驗歷程存檔

定案版在 `research/tsmc_grid_income.py`、文件 `strategies/tsmc_grid_income.md`。
以下為研究過程的里程碑腳本（存檔用，部分路徑/依賴為當時 scratchpad 環境）：

- `grid_screen.py` — 篩選高波動震盪股（網格主場標的）
- `grid_2330.py` — 原始 7 批 1% 網格（大多頭慘輸 B&H）
- `grid_variants.py` — 批數/格距/馬丁格爾變體比較
- `grid_bullets.py` — 收租網格 + 子彈加碼 + 深層槓桿（測出槓桿必在最深底部、與現金流互斥）
- `grid_reanchor.py` — 再掛單（跟隨近高）突破版，填掉乾旱 → 定案基礎
