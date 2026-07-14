# rebound_ranking.py
# 這支程式是整個模型最重的一次性運算 只需要在universe.csv更新後手動重跑
# 對universe.csv裡的每支ETF 各自做grid search
# 找出讓best_rebound_ratio最大的專屬觸發門檻(trigger_zscore)和最佳持有天數(optimal_days)
# 波動率篩選也在這裡做 用的是過去10年的資料 不是ETF上市以來的全部資料
# 這樣可以排除掉ETF剛上市那幾年通常比較不穩定 不確定性較高的時期
# 最後把所有ETF按照best_rebound_ratio排序 取前100名
# 這個排名不會隨著最新股價變動 因為是根據過去10年全部歷史資料算出來的 不是這幾天的資料

import pandas as pd
import numpy as np
import yfinance as yf
import time

# ---------------- 參數設定 ----------------
# 回顧幾年的歷史資料 這裡用10年 這個範圍同時也是排除掉ETF剛上市時期的依據
LOOKBACK_YEARS = 10

# 最佳持有天數的搜尋範圍 這裡搜尋1到14天 找反彈後第幾天賣出報酬最高
DAYS_MAX = 3

# 觸發門檻(z_score)的搜尋範圍 只用1 2 3 4這四個整數
# 舉例 觸發門檻是3 代表只挑當天跌幅超過歷史平均3個標準差的日子
# 這個grid乘以DAYS_MAX共4*14=56種組合 每支ETF都會逐一試過這56組 取best_rebound_ratio最高的那一組
TRIGGER_ZSCORE_GRID = [1, 1.5, 2, 2.5]

# 最少事件數 一個觸發門檻如果篩出的下跌事件少於這個數字就不採用
# 這是為了避免用只發生兩三次的極端事件去推論一個穩定的專屬參數 造成過度配適
MIN_EVENTS = 8

# 年化波動率門檻 這裡沿用60% 用過去10年的收盤對收盤報酬計算
# 舉例 某支ETF過去10年日報酬的年化標準差是0.75 代表75% 會被排除
VOLATILITY_THRESHOLD = 0.6

# 最終取排名前幾名的ETF
TOP_N = 100

# 輸入輸出檔案路徑
UNIVERSE_PATH = "universe.csv"
OUTPUT_PATH = "candidates_100.csv"

# 進度顯示間隔
PROGRESS_INTERVAL = 20

# 每次yfinance查詢之間的間隔秒數
REQUEST_DELAY = 0.3


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
    對單一ETF做完整的分析 包含波動率篩選和grid search
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

    return_mean = data["return"].mean()
    return_std = data["return"].std()

    if return_std == 0 or np.isnan(return_std):
        return None

    # 計算每一天的z_score 負值代表當天下跌
    data["z"] = (data["return"] - return_mean) / return_std

    total_trading_days = len(data)

    # 預先算好未來1到DAYS_MAX天的反彈報酬率
    # rebound_5欄位的意思是 從今天收盤買進 持有到未來第5天開盤賣出的報酬率
    for day in range(1, DAYS_MAX + 1):
        data[f"rebound_{day}"] = data["Open"].shift(-day) / data["Close"]

    best_combo = None  # 格式是 (trigger_zscore, optimal_days, best_ratio, event_count)

    for trigger_zscore in TRIGGER_ZSCORE_GRID:
        # 找出跌幅超過這個觸發門檻的所有日期
        drop_mask = data["z"] <= -trigger_zscore
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
        # 這支ETF專屬的觸發門檻和最佳持有天數 是grid search在過去10年資料裡找出的最佳組合
        "trigger_zscore": int(trigger_zscore),
        "optimal_days": int(optimal_days),
        "best_rebound_ratio": round(float(best_ratio), 4),
        "event_count": int(event_count),
        "total_trading_days": int(total_trading_days),
        # 這兩個欄位是這支ETF過去10年return的平均值和標準差
        # 每日分析時要用同一套統計基準去算今天的z_score 才會跟這裡的定義一致
        "return_mean": round(float(return_mean), 6),
        "return_std": round(float(return_std), 6)
    }


def main():
    universe = pd.read_csv(UNIVERSE_PATH)
    print(f"=== 讀入母體ETF 共 {len(universe)} 支 開始逐一做波動率篩選與grid search ===")

    results = []

    for i, row in universe.iterrows():
        if i % PROGRESS_INTERVAL == 0:
            print(f"處理進度 {i}/{len(universe)}")

        params = find_best_params_for_etf(row["symbol"], row["listing_years"])

        if params is not None:
            results.append(params)

        time.sleep(REQUEST_DELAY)

    result_df = pd.DataFrame(results)
    print(f"波動率篩選與grid search完成 共 {len(result_df)} 支ETF找到有效參數組合")

    # 按照best_rebound_ratio由大到小排序 取前TOP_N名
    result_df.sort_values("best_rebound_ratio", ascending=False, inplace=True)
    result_df = result_df.head(TOP_N).reset_index(drop=True)

    # 標記原始名次 這個名次之後在介面上會用來標記每日分析結果的排名
    result_df["rank"] = result_df.index + 1

    result_df.to_csv(OUTPUT_PATH, index=False)
    print(f"=== 完成 已將前{TOP_N}名ETF存到 {OUTPUT_PATH} ===")


if __name__ == "__main__":
    main()
