# rebound_ranking.py
# 這支程式是整個模型最重的一次性運算 只需要在universe.csv更新後手動重跑
# 第一步先用低波動率定義對universe.csv做預篩選 只留下真正廣泛分散 沒有槓桿反向 沒有限定單一國家或產業的ETF
# 第二步對通過預篩選的ETF各自做grid search
# 找出讓best_rebound_ratio最大的專屬觸發門檻(trigger_zscore)和最佳持有天數(optimal_days)
# 波動率篩選也在grid search這一步做 用的是過去10年的資料 不是ETF上市以來的全部資料
# 這樣可以排除掉ETF剛上市那幾年通常比較不穩定 不確定性較高的時期
# 目前main()會把完成grid search的全部ETF直接存檔 檔名依照實際數量命名 例如candidates_194.csv
# filter_top_n這個函式保留從N支篩選到前TOP_N支的邏輯 但main()目前沒有呼叫它
# 這是刻意保留下來的 之後學到更多篩選知識後 可以在這194支的基礎上繼續往下篩選
# 這個排名不會隨著最新股價變動 因為是根據過去10年全部歷史資料算出來的 不是這幾天的資料

import pandas as pd
import numpy as np
import yfinance as yf
import time

# ---------------- 參數設定 ----------------
# 回顧幾年的歷史資料 這裡用10年 這個範圍同時也是排除掉ETF剛上市時期的依據
LOOKBACK_YEARS = 10

# 最佳持有天數的搜尋範圍 這裡搜尋1到3天 找下跌後第幾天賣出反彈報酬最高
DAYS_MAX = 3

# 觸發門檻(z_score)的搜尋範圍
# 舉例 觸發門檻是2 代表只挑z_score落在負的3(不含)到負的2(含)這個區間的日子 不是門檻以下全部累加
# 這個grid乘以DAYS_MAX共6乘3等於18種組合 每支ETF都會逐一試過這18組 取best_rebound_ratio最高的那一組
TRIGGER_ZSCORE_GRID = [1, 1.5, 2, 2.5, 3, 3.5]

# 最少事件數 一個觸發門檻區間如果篩出的下跌事件少於這個數字就不採用
# 這是為了避免用只發生兩三次的極端事件去推論一個穩定的專屬參數 造成過度配適
MIN_EVENTS = 8

# 年化波動率門檻 這是grid search這一步的最後一道保險 用過去10年的收盤對收盤報酬計算
# 舉例 某支ETF過去10年日報酬的年化標準差是0.75 代表75% 會被排除
# 這道門檻通過率會被低波動率定義預篩選機制擋掉大部分不合適的ETF 這裡只是留一道保險
VOLATILITY_THRESHOLD = 0.6

# 極端值過濾門檻 初步z_score絕對值超過這個數字的日期會被排除 不參與統計基準的計算
# 舉例 某天因為除息或財報跳空 return算出來的初步z_score是5.2 這種日子會被排除
# 排除後用剩下的資料重新算一次mean和std 這組修正過的統計基準才會拿去用在grid search和每日分析
EXTREME_Z_THRESHOLD = 4

# 這是filter_top_n函式要篩到剩下幾支的數字 目前main()沒有呼叫filter_top_n 這個常數先保留著
TOP_N = 100

# ---------------- 低波動率定義 ----------------
# 這個定義是實際比對SPY VOO QQQ ZEB.TO(應該通過)
# 和XID.TO DTRE DLR.TO TWM SKYY FDN TMV INDY IGV(應該被排除)這13支ETF的yfinance資料後歸納出來的

# 白名單分類 這些都是美股廣泛分散 沒有槓桿 沒有反向 沒有限定單一國家或產業的晨星分類
# 舉例 SPY的category是"Large Blend" QQQ的category是"Large Growth" 兩個都在這個清單裡 算通過
# SKYY的category是"Technology" INDY的category是"India Equity" 都不在清單裡 算不通過
ALLOWED_CATEGORIES = {
    "Large Blend", "Large Growth", "Large Value",
    "Mid-Cap Blend", "Mid-Cap Growth", "Mid-Cap Value",
    "Small Blend", "Small Growth", "Small Value"
}

# beta3Year的容許範圍 這是給category查不到的股票用的備用判斷
# 加拿大上市的ETF在yfinance裡常常查不到category(顯示None) 例如ZEB.TO XID.TO DLR.TO都是這樣
# 這時候改看beta3Year 也就是3年期貝他值 代表這支基金相對大盤的波動同步程度
# 舉例 ZEB.TO的beta3Year是1.1 落在範圍內算通過 XID.TO的beta3Year是0.26 跟大盤幾乎不同步 不通過
# DLR.TO是貨幣ETF 根本查不到beta3Year這個欄位 直接判定不通過
BETA_LOWER = 0.8
BETA_UPPER = 1.3

# 每次yfinance查詢之間的間隔秒數
REQUEST_DELAY = 0.3

# 進度顯示間隔
PROGRESS_INTERVAL = 20

# 輸入檔案路徑
UNIVERSE_PATH = "universe.csv"


def passes_low_volatility_definition(symbol):
    """
    判斷一支股票是否符合低波動率定義
    先看category是否落在白名單裡 如果category是None(常見於加拿大上市的ETF查不到晨星分類)
    就改看beta3Year是否落在BETA_LOWER到BETA_UPPER之間 如果兩者都查不到 就直接判定不通過
    """
    try:
        info = yf.Ticker(symbol).info
    except Exception:
        return False

    category = info.get("category")

    if category is not None:
        return category in ALLOWED_CATEGORIES

    beta = info.get("beta3Year")

    if beta is None:
        return False

    return BETA_LOWER <= beta <= BETA_UPPER


def _get_etf_category(ticker):
    """
    查詢這支ETF的類別 例如Large Blend Bond ETF等
    這個資訊是給介面分類顯示用的 查詢失敗就標記成未分類 不影響其他篩選流程
    """
    try:
        category = ticker.info.get("category")
        return category if category else "未分類"
    except Exception:
        return "未分類"


def find_best_params_for_etf(symbol, listing_years):
    """
    對單一ETF做完整的分析 包含波動率篩選 極端值過濾 和grid search
    回傳這支ETF的類別 專屬觸發門檻 最佳持有天數 best_rebound_ratio等欄位
    如果波動率超標 資料不足 或找不到符合條件的組合 回傳None
    """
    ticker = yf.Ticker(symbol)

    try:
        hist = ticker.history(period=f"{LOOKBACK_YEARS}y", auto_adjust=True)
    except Exception:
        return None

    if hist.empty or len(hist) < 500:
        # 資料太少代表可能剛上市或資料有問題 直接跳過
        return None

    # ---- 波動率篩選 用過去10年收盤對收盤的日報酬計算年化波動率 ----
    close_return = hist["Close"].pct_change().dropna()
    annualized_volatility = close_return.std() * np.sqrt(252)

    if np.isnan(annualized_volatility) or annualized_volatility > VOLATILITY_THRESHOLD:
        return None

    # ---- 準備grid search用的資料 ----
    data = hist[["Close", "High", "Open"]].copy()

    # 計算每日的開盤對前一日收盤的報酬率 這是用來偵測下跌事件的訊號
    # 舉例 昨天收盤100元 今天開盤97元 這裡算出來的return就是0.97
    data["return"] = data["Open"] / data["Close"].shift(1)
    data.dropna(subset=["return"], inplace=True)

    if len(data) < 500:
        return None

    # ---- 極端值過濾 先用全部資料算一次初步統計基準 ----
    # 舉例 某天除息跳空 return算出來的初步raw_z絕對值是5.2 超過EXTREME_Z_THRESHOLD=4 這天會被排除
    raw_mean = data["return"].mean()
    raw_std = data["return"].std()

    if raw_std == 0 or np.isnan(raw_std):
        return None

    raw_z = (data["return"] - raw_mean) / raw_std

    # 排除掉極端值之後 用剩下的資料重新算一次mean和std 這組才是正式的統計基準
    good_mask = raw_z.abs() <= EXTREME_Z_THRESHOLD
    good_data = data[good_mask].copy()

    if len(good_data) < 500:
        return None

    return_mean = good_data["return"].mean()
    return_std = good_data["return"].std()

    if return_std == 0 or np.isnan(return_std):
        return None

    # 用修正過的統計基準 對原始全部資料重新算一次z_score
    # 這樣即使某天是被排除的極端值 它的z_score還是算得出來 只是這天不會被拿去當作統計基準的來源
    data["z"] = (data["return"] - return_mean) / return_std

    total_trading_days = len(data)

    # 預先算好未來1到DAYS_MAX天的反彈報酬率
    # 舉例 rebound_2欄位的意思是 從今天收盤買進 持有到未來第2天開盤賣出的報酬率
    for day in range(1, DAYS_MAX + 1):
        data[f"rebound_{day}"] = data["Open"].shift(-day) / data["Close"]

    best_combo = None  # 格式是 (trigger_zscore, optimal_days, best_ratio, event_count)

    for trigger_zscore in TRIGGER_ZSCORE_GRID:
        # 找出z_score落在這個下跌區間內的日期 用區間分箱 不是門檻以下全部累加
        # 舉例 trigger_zscore是2 區間就是負的3(不含)到負的2(含) 只抓落在這個區間的日子
        lower_bound = -(trigger_zscore + 1)
        upper_bound = -trigger_zscore
        drop_mask = (data["z"] > lower_bound) & (data["z"] <= upper_bound)
        event_count = int(drop_mask.sum())

        if event_count < MIN_EVENTS:
            continue

        for day in range(1, DAYS_MAX + 1):
            col = f"rebound_{day}"
            valid_mask = drop_mask & data[col].notna()

            if valid_mask.sum() < MIN_EVENTS:
                continue

            avg_ratio = data.loc[valid_mask, col].mean()

            if best_combo is None or avg_ratio > best_combo[2]:
                best_combo = (trigger_zscore, day, avg_ratio, event_count)

    if best_combo is None:
        return None

    trigger_zscore, optimal_days, best_ratio, event_count = best_combo
    category = _get_etf_category(ticker)

    return {
        "symbol": symbol,
        "category": category,
        "listing_years": listing_years,
        "annualized_volatility": round(float(annualized_volatility), 4),
        # 這支ETF專屬的觸發門檻和最佳持有天數 是grid search在過去10年資料裡找出的最佳區間組合
        # 注意這裡用float而不是int 因為grid裡有1.5 2.5 3.5這種非整數值 用int會被錯誤截斷成1或2
        "trigger_zscore": float(trigger_zscore),
        "optimal_days": int(optimal_days),
        "best_rebound_ratio": round(float(best_ratio), 4),
        "event_count": int(event_count),
        "total_trading_days": int(total_trading_days),
        # 這兩個欄位是過濾掉極端值之後 這支ETF過去10年return的平均值和標準差
        # 每日分析時要用同一套統計基準去算今天的z_score 才會跟這裡的定義一致
        "return_mean": round(float(return_mean), 6),
        "return_std": round(float(return_std), 6)
    }


def filter_top_n(result_df):
    """
    這是從N支股票篩選到前TOP_N支的邏輯 依照best_rebound_ratio由大到小排序後取前TOP_N名
    目前main()沒有呼叫這個函式 直接輸出全部通過grid search的股票 不做這一層篩選
    這段邏輯保留在這裡 是為了將來學到更多篩選知識後 可以在這批股票的基礎上繼續往下篩選
    輸入的result_df必須已經是完成grid search後的結果 且尚未排序
    回傳裁切過的DataFrame 並附上排名欄位
    """
    sorted_df = result_df.sort_values("best_rebound_ratio", ascending=False).reset_index(drop=True)

    if len(sorted_df) < TOP_N:
        final_df = sorted_df.copy()
    else:
        final_df = sorted_df.head(TOP_N).copy()

    final_df["rank"] = final_df.index + 1
    return final_df


def main():
    universe = pd.read_csv(UNIVERSE_PATH)
    print(f"=== 讀入母體ETF 共 {len(universe)} 支 開始逐一檢查是否符合低波動率定義 ===")

    passed_rows = []

    for i, row in universe.iterrows():
        if i % PROGRESS_INTERVAL == 0:
            print(f"低波動率定義篩選進度 {i}/{len(universe)}")

        if passes_low_volatility_definition(row["symbol"]):
            passed_rows.append(row)

        time.sleep(REQUEST_DELAY)

    filtered_universe = pd.DataFrame(passed_rows).reset_index(drop=True)
    print(f"低波動率定義篩選完成 共 {len(filtered_universe)} 支ETF通過 準備進入grid search")

    print(f"=== 開始對通過低波動率定義的 {len(filtered_universe)} 支ETF做波動率篩選 極端值過濾與grid search ===")

    results = []

    for i, row in filtered_universe.iterrows():
        if i % PROGRESS_INTERVAL == 0:
            print(f"grid search進度 {i}/{len(filtered_universe)}")

        params = find_best_params_for_etf(row["symbol"], row["listing_years"])

        if params is not None:
            results.append(params)

        time.sleep(REQUEST_DELAY)

    result_df = pd.DataFrame(results)
    print(f"grid search完成 共 {len(result_df)} 支ETF找到有效參數組合")

    # 按照best_rebound_ratio由大到小排序 全部完成grid search的ETF都保留 不做前TOP_N的裁切
    # filter_top_n函式還在上面 只是這裡先不呼叫它 保留給以後繼續篩選用
    result_df.sort_values("best_rebound_ratio", ascending=False, inplace=True)
    result_df.reset_index(drop=True, inplace=True)

    # 標記名次 這個名次之後在介面上會用來標記每日分析結果的排名
    result_df["rank"] = result_df.index + 1

    output_count = len(result_df)

    # 檔名依照實際輸出的股票數量命名 目前應該會是candidates_194.csv這種格式
    output_path = f"candidates_{output_count}.csv"
    result_df.to_csv(output_path, index=False)
    print(f"=== 完成 已將全部 {output_count} 支ETF存到 {output_path} ===")


if __name__ == "__main__":
    main()
