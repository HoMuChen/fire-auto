# 學術因子對照表（FEATURE_PLAN 階段 3）

把 Jensen、Kelly、Pedersen（2023，*Is There a Replication Crisis in Finance?*）的 153 個因子，逐條和本專案已測的假設（`HYPOTHESES.md`）、特徵庫（`FEATURES.md`）對照。這份表本身只是對照；第四十六批 FS8 已從表裡挑 20 個跑完（結果見 `HYPOTHESES.md`），狀態已更新成「已測」。下一批從剩下的「現有資料可做」裡挑。

- 清單來源：作者公開的 `bkelly-lab/ReplicationCrisis` GitHub 專案裡的 `GlobalFactors/Cluster Labels.csv`（153 個因子、13 群）與 `Factor Details.xlsx`（名稱、方向）。分群照 JKP 原文，所以「季節性」群裡有不少不是季節性的因子。
- 方向：JKP 的預期方向（+＝值越大報酬越高）。台股已測的結論見對應的假設或特徵；方向不一定和美股相同（例如台股是延續、不是反轉）。
- 狀態：
  - **已測**：特徵庫或某個 H 已經是同一個定義。
  - **已測（近似）**：已測的版本只差在窗長、分母、季或月等細節。照「不做參數搜尋」的原則，不再重測。
  - **現有資料可做**：`data/v2` 已有的資料就能算（財報 fin／bs／cf、月營收、股利、市值、價量、籌碼）。財報類 bs 從 2012Q4、cf 從 2012Q1 起，評估會走 B＋V。
  - **需要新資料**：需要 FinMind 沒有、或還沒抓的資料。
  - **台股做不到**：台股資料做不到，或歷史太短（資料從 2010 年起，看 6 年以上前報酬的因子在 dev 大多沒有值）。

## 總覽

| 群 | 因子數 | 已測 | 已測（近似） | 現有資料可做 | 需要新資料 | 台股做不到 |
|---|---|---|---|---|---|---|
| 投資（Investment） | 22 | 5 | 4 | 12 | 1 | 0 |
| 低風險（Low Risk） | 18 | 5 | 6 | 6 | 0 | 1 |
| 價值（Value） | 18 | 9 | 2 | 7 | 0 | 0 |
| 品質（Quality） | 17 | 2 | 1 | 14 | 0 | 0 |
| 獲利成長（Profit Growth） | 12 | 6 | 1 | 4 | 1 | 0 |
| 季節性（Seasonality） | 12 | 1 | 1 | 6 | 0 | 4 |
| 低槓桿（Low Leverage） | 11 | 3 | 0 | 5 | 1 | 2 |
| 獲利能力（Profitability） | 11 | 4 | 1 | 6 | 0 | 0 |
| 動能（Momentum） | 8 | 3 | 1 | 4 | 0 | 0 |
| 舉債（Debt Issuance） | 7 | 1 | 0 | 6 | 0 | 0 |
| 應計（Accruals） | 6 | 2 | 0 | 3 | 0 | 1 |
| 短期反轉（Short-Term Reversal） | 6 | 1 | 1 | 4 | 0 | 0 |
| 規模（Size） | 5 | 3 | 0 | 1 | 0 | 1 |
| **合計** | **153** | **45** | **18** | **78** | **3** | **9** |

- 已測（含近似）的因子集中在動能、規模、低風險、價值；這些群台股的結論已經很清楚（見 `AGENTS.md`「已知規律」）。
- 第四十六批之前，還沒測、資料又已經有的共 98 個（現在剩 78 個），大多在**投資**、**品質**、**價值**、**獲利能力**、**舉債**這幾群，也就是資產負債表的各種成長與比率，以及用市值當分母的估值比。台股過去的財報類特徵（H26、H27、H80～H83、F-score）大多不過，所以預期多數不會有東西；但 FS6 的 NET_ISSUE、CL_CHG 說明資產負債表裡仍有沒測到的資訊。
- 需要新資料的只有員工人數（emp_gr1、sale_emp_gr1）和上市日期（age）；做不到的是研發類（3 個）和超長期季節性（6 個）。

## 第四十六批（FS8）挑的 20 個

原則：一群挑一兩個代表、用原論文定義、避免和已測的高度重複。2026-11-03 跑完：估值類（EBITDA_EV、FCF_ME、EQNPO）排序 `B◎✓✓`，但 EBITDA_EV、FCF_ME 的多空、避雷在 V 反向；HL_SPREAD 排序 `◎✓✓`（價差小的好，和 JKP 相反）；NOA_GR、PPEINV 避雷 `B◎✓✓`；Z_SCORE、CASH_AT 只預測風險。

| 代號 | 名稱 | 群 | 理由 |
|---|---|---|---|
| noa_at | 淨營運資產 ÷ 資產 | 舉債 | 美股最穩的資產負債表異常之一，台股未測 |
| noa_gr1a | 淨營運資產變化 | 投資 | 和 H80 資產成長、H83 應計互補 |
| taccruals_at | 總應計 ÷ 資產 | 應計 | H83 只有營運應計，總應計多了投資、籌資部分 |
| cop_at | 現金基礎營業利潤 ÷ 資產 | 品質 | Ball 等 2016，美股比毛利資產比強 |
| ope_be | 營業利潤 ÷ 權益 | 獲利能力 | Fama-French RMW 的定義 |
| z_score | Altman Z 分數 | 低槓桿 | 破產風險；台股困境股可能是彩券 |
| o_score | Ohlson O 分數 | 獲利能力 | 同上，另一種定義 |
| cash_at | 現金 ÷ 資產 | 低槓桿 | 簡單、資料完整 |
| ebitda_mev | EBITDA ÷ 企業價值 | 價值 | 只在 H121 當成分測過；台股價值類多數不過，當確認 |
| sale_me | 營收股價比 | 價值 | 可用月營收，比季報即時 |
| fcf_me | 自由現金流 ÷ 市值 | 價值 | H81 沒扣資本支出 |
| eqnpo_me | 淨發放殖利率 | 價值 | 股利＋庫藏股，和 NET_ISSUE、H6 一起看 |
| capx_gr2 | 資本支出 2 年成長 | 投資 | FS6 的資本支出強度只預測風險，改看成長 |
| ppeinv_gr1a | 固定資產＋存貨變化 | 投資 |  |
| dsale_dinv | 營收變化 − 存貨變化 | 獲利成長 | H90 只看存貨 |
| niq_be_chg1 | 季 ROE 變化 | 獲利成長 |  |
| ret_12_7 | 中期動能（前 12～7 月） | 動能 | Novy-Marx 2012：中期動能比近期動能強；和 H1 的窗部分重疊 |
| bidaskhl_21d | 高低價價差 | 低槓桿 | 流動性成本，價量即可算 |
| betadown_252d | 下檔 beta | 低風險 | BETA 只預測風險；看下檔版是否不同 |
| coskew_21d | 共偏態 | 季節性群 |  |

組合類（QMJ、Stambaugh-Yuan 錯價因子）建議等成分測完再說。

## 投資（Investment，22 個）

| JKP 代號 | 英文名 | 中文 | JKP 方向 | 狀態 | 對應／所需資料 | 備註 |
|---|---|---|---|---|---|---|
| at_gr1 | Asset Growth | 資產成長 | − | 已測 | H80 |  |
| capx_gr2 | CAPEX growth (2 years) | 資本支出 2 年成長 | − | 已測 | CAPX_GR2（第四十六批） |  |
| inv_gr1a | Inventory change | 存貨變化 ÷ 資產 | − | 已測 | H90 |  |
| noa_gr1a | Change in net operating assets | 淨營運資產變化 | − | 已測 | NOA_GR（第四十六批） |  |
| ppeinv_gr1a | Change PPE and Inventory | 固定資產＋存貨變化 | − | 已測 | PPEINV（第四十六批） | Lyandres 等 2008 |
| inv_gr1 | Inventory growth | 存貨成長率 | − | 已測（近似） | H90 | H90 是存貨變化 ÷ 平均總資產 |
| ret_60_12 | Long-term reversal | 長期反轉（前 60～13 月） | − | 已測（近似） | H98 | H98 是前 36～13 月 |
| sale_gr1 | Sales Growth (1 year) | 營收 1 年成長 | − | 已測（近似） | H7 | H7 是月營收 YoY |
| saleq_gr1 | Sales growth (1 quarter) | 單季營收 YoY | − | 已測（近似） | H7 | 月營收比季營收更早、更細 |
| aliq_at | Liquidity of book assets | 帳面資產流動性 | − | 現有資料可做 | bs 現金、流動資產、固定資產 |  |
| be_gr1a | Change in common equity | 權益變化 ÷ 資產 | − | 現有資料可做 | bs 權益 |  |
| capx_gr1 | CAPEX growth (1 year) | 資本支出 1 年成長 | − | 現有資料可做 | cf 取得不動產廠房設備 | cf 2012Q1 起 |
| capx_gr3 | CAPEX growth (3 years) | 資本支出 3 年成長 | − | 現有資料可做 | 同上 |  |
| coa_gr1a | Change in current operating assets | 流動營運資產變化 | − | 現有資料可做 | bs |  |
| col_gr1a | Change in current operating liabilities | 流動營運負債變化 | − | 現有資料可做 | bs |  |
| lnoa_gr1a | Change in long-term net operating assets | 長期淨營運資產變化 | − | 現有資料可做 | bs |  |
| mispricing_mgmt | Mispricing factor: Management | 錯價因子：管理面（Stambaugh-Yuan） | + | 現有資料可做 | 組合：淨發行、應計、NOA、資產成長、投資 | 成分大多已測（NET_ISSUE、H83、H80） |
| ncoa_gr1a | Change in noncurrent operating assets | 非流動營運資產變化 | − | 現有資料可做 | bs |  |
| nncoa_gr1a | Change in net noncurrent operating assets | 淨非流動營運資產變化 | − | 現有資料可做 | bs |  |
| sale_gr3 | Sales Growth (3 years) | 營收 3 年成長 | − | 現有資料可做 | 月營收 |  |
| seas_2_5na | Years 2-5 lagged returns, nonannual | 2–5 年前非同月報酬 | − | 現有資料可做 | 價格 | H38 是同月版本 |
| emp_gr1 | Hiring rate | 員工人數成長 | − | 需要新資料 | 員工人數（FinMind 沒有；年報或公開資訊觀測站） |  |

## 低風險（Low Risk，18 個）

| JKP 代號 | 英文名 | 中文 | JKP 方向 | 狀態 | 對應／所需資料 | 備註 |
|---|---|---|---|---|---|---|
| betadown_252d | Downside beta | 下檔 beta | − | 已測 | BETA_DOWN（第四十六批） |  |
| ivol_capm_252d | Idiosyncratic volatility from the CAPM (252 days) | CAPM 特質波動（252 日） | − | 已測 | IVOL | ≈ H3（ρ 0.98） |
| rmax1_21d | Maximum daily return | 近 21 日最大日報酬 | − | 已測 | H13 |  |
| rvol_21d | Return volatility | 報酬波動 | − | 已測 | H3 |  |
| turnover_126d | Share turnover | 週轉率 | − | 已測 | TURN | TURN 是 60 日；≈ H91 |
| beta_60m | Market Beta | 市場 beta（60 月） | − | 已測（近似） | BETA | BETA 是 250 日日報酬 |
| betabab_1260d | Frazzini-Pedersen market beta | Frazzini-Pedersen beta | − | 已測（近似） | BETA | BAB 用 5 年相關 × 1 年波動比 |
| ivol_capm_21d | Idiosyncratic volatility from the CAPM (21 days) | CAPM 特質波動（21 日） | − | 已測（近似） | IVOL | IVOL 是 250 日 |
| ivol_ff3_21d | Idiosyncratic volatility from the Fama-French 3-factor model | FF3 特質波動 | − | 已測（近似） | IVOL | 要自建台股 SMB、HML；預期 ≈ IVOL |
| ivol_hxz4_21d | Idiosyncratic volatility from the q-factor model | q 因子特質波動 | − | 已測（近似） | IVOL | 同上 |
| rmax5_21d | Highest 5 days of return | 近 21 日最高 5 日報酬平均 | − | 已測（近似） | H13 |  |
| beta_dimson_21d | Dimson beta | Dimson beta（21 日，含落後項） | − | 現有資料可做 | 價格 |  |
| earnings_variability | Earnings variability | 盈餘變異 | − | 現有資料可做 | fin、cf（季） |  |
| ocfq_saleq_std | Cash flow volatility | 現金流波動 | − | 現有資料可做 | cf、fin（季） |  |
| zero_trades_126d | Number of zero trades with turnover as tiebreaker (6 months) | 零成交日數（6 月） | + | 現有資料可做 | 原始成交量 | 股票池要求 60 日均量 ≥ 200 張，池內幾乎沒有零成交，預期沒有變異 |
| zero_trades_21d | Number of zero trades with turnover as tiebreaker (1 month) | 零成交日數（1 月） | + | 現有資料可做 | 同上 | 同上 |
| zero_trades_252d | Number of zero trades with turnover as tiebreaker (12 months) | 零成交日數（12 月） | + | 現有資料可做 | 同上 | 同上 |
| seas_6_10na | Years 6-10 lagged returns, nonannual | 6–10 年前非同月報酬 | − | 台股做不到 |  | 資料從 2010 年起，dev 期間大多沒有值 |

## 價值（Value，18 個）

| JKP 代號 | 英文名 | 中文 | JKP 方向 | 狀態 | 對應／所需資料 | 備註 |
|---|---|---|---|---|---|---|
| be_me | Book-to-market equity | 淨值股價比 | + | 已測 | H5 |  |
| chcsho_12m | Net stock issues | 淨發行股數（12 月） | − | 已測 | NET_ISSUE |  |
| div12m_me | Dividend yield | 殖利率 | + | 已測 | H6 |  |
| ebitda_mev | Ebitda-to-market enterprise value | EBITDA ÷ 企業價值 | + | 已測 | EBITDA_EV（第四十六批） | 只在 H121 QMV 裡當成分測過 |
| eqnpo_me | Net payout yield | 淨發放殖利率 | + | 已測 | EQNPO（第四十六批） |  |
| fcf_me | Free cash flow-to-price | 自由現金流 ÷ 市值 | + | 已測 | FCF_ME（第四十六批） | H81 是營業現金流，沒扣資本支出 |
| ni_me | Earnings-to-price | 盈餘殖利率 | + | 已測 | H11 |  |
| ocf_me | Operating cash flow-to-market | 營業現金流 ÷ 市值 | + | 已測 | H81 |  |
| sale_me | Sales-to-market | 營收股價比 | + | 已測 | SALE_ME（第四十六批） | 可用月營收近 12 月合計，比季報更即時 |
| eqnetis_at | Net equity issuance | 淨權益發行 ÷ 資產 | − | 已測（近似） | NET_ISSUE | NET_ISSUE 用股數；金額版要現金增資資料（div 有 TotalNumberOfCashCapitalIncrease） |
| eqpo_me | Payout yield | 發放殖利率（股利＋庫藏股） | + | 已測（近似） | H6 | H6 只有現金股利 |
| at_me | Assets-to-market | 資產 ÷ 市值 | + | 現有資料可做 | bs、市值 |  |
| bev_mev | Book-to-market enterprise value | 企業帳面價值 ÷ 企業市值 | + | 現有資料可做 | bs、市值 |  |
| debt_me | Debt-to-market | 負債 ÷ 市值 | + | 現有資料可做 | bs、市值 |  |
| eq_dur | Equity duration | 權益存續期間 | − | 現有資料可做 | fin、bs、市值 | Dechow 等 2004；計算較繁 |
| eqnpo_12m | Equity net payout | 權益淨發放（12 月） | + | 現有資料可做 | 股利、股數 | ≈ 殖利率 − 淨發行 |
| ival_me | Intrinsic value-to-market | 內含價值 ÷ 市值 | + | 現有資料可做 | fin、bs、市值 | Frankel-Lee 1998 剩餘所得模型 |
| netis_at | Net total issuance | 淨總發行 ÷ 資產 | − | 現有資料可做 | cf 籌資活動、bs |  |

## 品質（Quality，17 個）

| JKP 代號 | 英文名 | 中文 | JKP 方向 | 狀態 | 對應／所需資料 | 備註 |
|---|---|---|---|---|---|---|
| cop_at | Cash-based operating profits-to-book assets | 現金基礎營業利潤 ÷ 資產 | + | 已測 | COP（第四十六批） | Ball 等 2016 |
| gp_at | Gross profits-to-assets | 毛利 ÷ 資產 | + | 已測 | H27 |  |
| gp_atl1 | Gross profits-to-lagged assets | 毛利 ÷ 前期資產 | + | 已測（近似） | H27 |  |
| at_turnover | Capital turnover | 資本週轉率 | + | 現有資料可做 | fin、bs |  |
| cop_atl1 | Cash-based operating profits-to-lagged book assets | 現金基礎營業利潤 ÷ 前期資產 | + | 現有資料可做 | 同上 |  |
| dgp_dsale | Change gross margin minus change sales | 毛利率變化 − 營收變化 | + | 現有資料可做 | fin |  |
| mispricing_perf | Mispricing factor: Performance | 錯價因子：績效面（Stambaugh-Yuan） | + | 現有資料可做 | 組合：O 分數、動能、毛利、ROA |  |
| ni_inc8q | Number of consecutive quarters with earnings increases | 連續盈餘成長季數 | + | 現有資料可做 | fin 淨利（季） | H89 盈餘創新高相近但不同 |
| niq_at | Quarterly return on assets | 季 ROA | + | 現有資料可做 | fin、bs |  |
| op_at | Operating profits-to-book assets | 營業利潤 ÷ 資產 | + | 現有資料可做 | fin、bs |  |
| op_atl1 | Operating profits-to-lagged book assets | 營業利潤 ÷ 前期資產 | + | 現有資料可做 | fin、bs |  |
| opex_at | Operating leverage | 營業費用 ÷ 資產 | + | 現有資料可做 | fin 營業費用、bs | Novy-Marx 2011；H123「營運槓桿」定義不同 |
| qmj | Quality minus Junk: Composite | 品質減垃圾（QMJ） | + | 現有資料可做 | 組合 | H121 QMV 是另一種品質＋動能＋價值組合 |
| qmj_growth | Quality minus Junk: Growth | QMJ：成長 | + | 現有資料可做 | 組合 |  |
| qmj_prof | Quality minus Junk: Profitability | QMJ：獲利 | + | 現有資料可做 | 組合 |  |
| qmj_safety | Quality minus Junk: Safety | QMJ：安全 | + | 現有資料可做 | 組合 |  |
| sale_bev | Asset turnover | 資產週轉率 | + | 現有資料可做 | fin、bs |  |

## 獲利成長（Profit Growth，12 個）

| JKP 代號 | 英文名 | 中文 | JKP 方向 | 狀態 | 對應／所需資料 | 備註 |
|---|---|---|---|---|---|---|
| dsale_dinv | Change sales minus change Inventory | 營收變化 − 存貨變化 | + | 已測 | DSALE_DINV（第四十六批） | H90 只看存貨 |
| dsale_drec | Change sales minus change receivables | 營收變化 − 應收變化 | − | 已測 | AR_REV | AR_REV＝應收成長 − 營收成長，方向相同 |
| niq_be_chg1 | Change in quarterly return on equity | 季 ROE 變化 | + | 已測 | ROE_CHG（第四十六批） | H88 三率三升相近 |
| niq_su | Standardized earnings surprise | 盈餘驚喜 SUE | + | 已測 | H28 |  |
| ret_12_7 | Price momentum t-12 to t-7 | 中期動能（前 12～7 月） | + | 已測 | RET_12_7（第四十六批） | Novy-Marx 2012 |
| saleq_su | Standardized Revenue surprise | 營收驚喜 | + | 已測 | H87 | H87 是月營收版 SUR |
| seas_1_1an | Year 1-lagged return, annual | 去年同月報酬 | + | 已測（近似） | H38 | H38 是過去 5 年同月平均 |
| dsale_dsga | Change sales minus change SG&A | 營收變化 − 管銷費用變化 | + | 現有資料可做 | fin 營業費用合計（沒有單獨的管銷科目） |  |
| niq_at_chg1 | Change in quarterly return on assets | 季 ROA 變化 | + | 現有資料可做 | fin、bs |  |
| ocf_at_chg1 | Change in operating cash flow to assets | 營業現金流 ÷ 資產變化 | + | 現有資料可做 | cf、bs |  |
| tax_gr1a | Tax expense surprise | 所得稅費用驚喜 | + | 現有資料可做 | fin 所得稅（TAX） |  |
| sale_emp_gr1 | Labor force efficiency | 每員工營收成長 | + | 需要新資料 | 員工人數 |  |

## 季節性（Seasonality，12 個）

| JKP 代號 | 英文名 | 中文 | JKP 方向 | 狀態 | 對應／所需資料 | 備註 |
|---|---|---|---|---|---|---|
| coskew_21d | Coskewness | 共偏態 | − | 已測 | COSKEW（第四十六批） |  |
| seas_2_5an | Years 2-5 lagged returns, annual | 2–5 年前同月報酬 | + | 已測（近似） | H38 | H38 是 1–5 年 |
| corr_1260d | Market correlation | 與市場相關（5 年） | − | 現有資料可做 | 價格 | 2015 年起才有值 |
| dbnetis_at | Net debt issuance | 淨舉債 ÷ 資產 | − | 現有資料可做 | cf 舉借、償還長期借款 |  |
| kz_index | Kaplan-Zingales index | Kaplan-Zingales 財務限制指數 | + | 現有資料可做 | fin、bs、cf、股利 |  |
| lti_gr1a | Change in long-term investments | 長期投資變化 | − | 現有資料可做 | bs 權益法投資、非流動金融資產 |  |
| pi_nix | Taxable income-to-book income | 稅前淨利 ÷ 淨利 | + | 現有資料可做 | fin |  |
| sti_gr1a | Change in short-term investments | 短期投資變化 | + | 現有資料可做 | bs 流動金融資產 |  |
| seas_11_15an | Years 11-15 lagged returns, annual | 11–15 年前同月報酬 | + | 台股做不到 |  | 歷史太短 |
| seas_11_15na | Years 11-15 lagged returns, nonannual | 11–15 年前非同月報酬 | − | 台股做不到 |  | 歷史太短 |
| seas_16_20an | Years 16-20 lagged returns, annual | 16–20 年前同月報酬 | + | 台股做不到 |  | 歷史太短 |
| seas_6_10an | Years 6-10 lagged returns, annual | 6–10 年前同月報酬 | + | 台股做不到 |  | dev 期間大多沒有值 |

## 低槓桿（Low Leverage，11 個）

| JKP 代號 | 英文名 | 中文 | JKP 方向 | 狀態 | 對應／所需資料 | 備註 |
|---|---|---|---|---|---|---|
| bidaskhl_21d | The high-low bid-ask spread | 高低價價差估計（Corwin-Schultz） | + | 已測 | HL_SPREAD（第四十六批） |  |
| cash_at | Cash-to-assets | 現金 ÷ 資產 | + | 已測 | CASH_AT（第四十六批） |  |
| z_score | Altman Z-score | Altman Z 分數 | + | 已測 | Z_SCORE（第四十六批） | 五個成分都有 |
| aliq_mat | Liquidity of market assets | 市值資產流動性 | − | 現有資料可做 | bs、市值 |  |
| at_be | Book leverage | 帳面槓桿（資產 ÷ 權益） | − | 現有資料可做 | bs | DEBT_CHG 是負債比的變化，水準未測 |
| netdebt_me | Net debt-to-price | 淨負債 ÷ 市值 | − | 現有資料可做 | bs、市值 |  |
| ni_ivol | Earnings volatility | 盈餘波動 | + | 現有資料可做 | fin 淨利（季） | 要 8 季以上 |
| tangibility | Asset tangibility | 資產有形性 | + | 現有資料可做 | bs |  |
| age | Firm age | 上市年數 | − | 需要新資料 | 上市櫃日期（data/v2 只有 2010 年後的價格，不能拿第一根 K 當上市日） |  |
| rd5_at | R&D capital-to-book assets | 研發資本 ÷ 資產 | + | 台股做不到 |  | FinMind 損益表沒有研發費用科目 |
| rd_sale | R&D-to-sales | 研發 ÷ 營收 | + | 台股做不到 |  | 同上 |

## 獲利能力（Profitability，11 個）

| JKP 代號 | 英文名 | 中文 | JKP 方向 | 狀態 | 對應／所需資料 | 備註 |
|---|---|---|---|---|---|---|
| f_score | Pitroski F-score | Piotroski F-score | + | 已測 | F_SCORE、H82 |  |
| ni_be | Return on equity | ROE | + | 已測 | H26 |  |
| o_score | Ohlson O-score | Ohlson O 分數 | − | 已測 | O_SCORE（第四十六批） |  |
| ope_be | Operating profits-to-book equity | 營業利潤 ÷ 權益 | + | 已測 | OPE（第四十六批） | Fama-French RMW |
| niq_be | Quarterly return on equity | 季 ROE | + | 已測（近似） | H26 | H26 是 TTM |
| dolvol_var_126d | Coefficient of variation for dollar trading volume | 成交金額變異係數 | − | 現有資料可做 | 原始成交金額 |  |
| ebit_bev | Return on net operating assets | 淨營運資產報酬 | + | 現有資料可做 | fin、bs |  |
| ebit_sale | Profit margin | 營業利益率 | + | 現有資料可做 | fin | H88 三率三升看的是變化 |
| ocf_at | Operating cash flow to assets | 營業現金流 ÷ 資產 | + | 現有資料可做 | cf、bs | H81 是 ÷ 市值 |
| ope_bel1 | Operating profits-to-lagged book equity | 營業利潤 ÷ 前期權益 | + | 現有資料可做 | fin、bs |  |
| turnover_var_126d | Coefficient of variation for share turnover | 週轉率變異係數 | − | 現有資料可做 | 原始成交量 |  |

## 動能（Momentum，8 個）

| JKP 代號 | 英文名 | 中文 | JKP 方向 | 狀態 | 對應／所需資料 | 備註 |
|---|---|---|---|---|---|---|
| prc_highprc_252d | Current price to high price over last year | 52 週高點接近度 | + | 已測 | H2 |  |
| resff3_12_1 | Residual momentum t-12 to t-1 | 殘差動能 12-1 | + | 已測 | H77 | H77 用市場模型殘差 |
| ret_12_1 | Price momentum t-12 to t-1 | 12-1 動能 | + | 已測 | H1 |  |
| resff3_6_1 | Residual momentum t-6 to t-1 | 殘差動能 6-1 | + | 已測（近似） | H77 |  |
| ret_3_1 | Price momentum t-3 to t-1 | 3-1 動能 | + | 現有資料可做 | 價格 | H1 的不同窗；不做參數搜尋，優先度低 |
| ret_6_1 | Price momentum t-6 to t-1 | 6-1 動能 | + | 現有資料可做 | 價格 | 同上 |
| ret_9_1 | Price momentum t-9 to t-1 | 9-1 動能 | + | 現有資料可做 | 價格 | 同上 |
| seas_1_1na | Year 1-lagged return, nonannual | 去年非同月報酬 | + | 現有資料可做 | 價格 |  |

## 舉債（Debt Issuance，7 個）

| JKP 代號 | 英文名 | 中文 | JKP 方向 | 狀態 | 對應／所需資料 | 備註 |
|---|---|---|---|---|---|---|
| noa_at | Net operating assets | 淨營運資產 ÷ 資產 | − | 已測 | NOA（第四十六批） | Hirshleifer 等 2004；台股未測 |
| capex_abn | Abnormal corporate investment | 異常資本支出 | − | 現有資料可做 | cf 取得不動產廠房設備、fin 營收 | 資本支出 ÷ 營收相對前 3 年平均；FS6 的 CAPEX_INT 只預測風險 |
| debt_gr3 | Growth in book debt (3 years) | 帳面負債 3 年成長 | − | 現有資料可做 | bs 負債 | DEBT_CHG 是負債比的 4 季變化；bs 2012Q4 起，3 年成長要 2016 才有 |
| fnl_gr1a | Change in financial liabilities | 金融負債變化 | − | 現有資料可做 | bs 短借、長借、應付公司債 |  |
| ncol_gr1a | Change in noncurrent operating liabilities | 非流動營運負債變化 | − | 現有資料可做 | bs 非流動負債 − 長借 − 應付公司債 |  |
| nfna_gr1a | Change in net financial assets | 淨金融資產變化 | + | 現有資料可做 | bs 現金、金融資產、金融負債 |  |
| ni_ar1 | Earnings persistence | 盈餘持續性 | + | 現有資料可做 | fin 淨利（季） | 要滾動多年的 AR(1) 係數，季財報 2013 起，要 2018 年後才有值，B 段太短 |

## 應計（Accruals，6 個）

| JKP 代號 | 英文名 | 中文 | JKP 方向 | 狀態 | 對應／所需資料 | 備註 |
|---|---|---|---|---|---|---|
| oaccruals_at | Operating accruals | 營運應計 ÷ 資產 | − | 已測 | H83 | H83＝（淨利 − 營業現金流）÷ 平均總資產 |
| taccruals_at | Total accruals | 總應計 ÷ 資產 | − | 已測 | TACC（第四十六批） | H83 只有營運應計 |
| cowc_gr1a | Change in current operating working capital | 流動營運資金變化 | − | 現有資料可做 | bs 流動資產、現金、流動負債、短借 | H83 低應計是現金流法的總應計，概念相近 |
| oaccruals_ni | Percent operating accruals | 營運應計 ÷ \|淨利\| | − | 現有資料可做 | fin、cf | H83 的分母換成淨利絕對值（Hafzalla 等 2011） |
| taccruals_ni | Percent total accruals | 總應計 ÷ \|淨利\| | − | 現有資料可做 | bs、cf、fin |  |
| seas_16_20na | Years 16-20 lagged returns, nonannual | 16–20 年前非同月報酬 | − | 台股做不到 |  | 需要 16 年以上的價格歷史，資料從 2010 年起 |

## 短期反轉（Short-Term Reversal，6 個）

| JKP 代號 | 英文名 | 中文 | JKP 方向 | 狀態 | 對應／所需資料 | 備註 |
|---|---|---|---|---|---|---|
| ret_1_0 | Short-term reversal | 短期反轉（上月報酬） | − | 已測 | H4 |  |
| rskew_21d | Total skewness | 報酬偏態（21 日） | − | 已測（近似） | SKEW | SKEW 是 250 日 |
| iskew_capm_21d | Idiosyncratic skewness from the CAPM | CAPM 特質偏態 | − | 現有資料可做 | 價格 | SKEW 是 250 日總偏態 |
| iskew_ff3_21d | Idiosyncratic skewness from the Fama-French 3-factor model | FF3 特質偏態 | − | 現有資料可做 | 價格（要自建因子） |  |
| iskew_hxz4_21d | Idiosyncratic skewness from the q-factor model | q 因子特質偏態 | − | 現有資料可做 | 價格（要自建因子） |  |
| rmax5_rvol_21d | Highest 5 days of return scaled by volatility | 最高 5 日報酬 ÷ 波動 | − | 現有資料可做 | 價格 | H13 相近 |

## 規模（Size，5 個）

| JKP 代號 | 英文名 | 中文 | JKP 方向 | 狀態 | 對應／所需資料 | 備註 |
|---|---|---|---|---|---|---|
| ami_126d | Amihud Measure | Amihud 非流動性 | + | 已測 | H9 |  |
| market_equity | Market Equity | 市值 | − | 已測 | H84 |  |
| prc | Price per share | 股價 | − | 已測 | LOW_PRICE |  |
| dolvol_126d | Dollar trading volume | 成交金額 | − | 現有資料可做 | 原始成交金額 | 預期 ≈ H84 小型股、H9 |
| rd_me | R&D-to-market | 研發 ÷ 市值 | + | 台股做不到 |  | 沒有研發科目 |
