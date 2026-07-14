# rebound_ranking.py
# 這支程式是整個模型最重的一次性運算 只需要在universe.csv更新後手動重跑
# 對universe.csv裡的每支ETF 先做低波動率定義篩選 只留下真正適合這個反彈策略的ETF
# 低波動率定義是兩個條件同時成立 平均成交量至少達到MIN_VOLUME 年化波動率落在MIN_VOLATILITY到VOLATILITY_THRESHOLD之間
# 舉例 XID.TO和DTRE雖然年化波動率在合理範圍內 但平均日成交量只有幾千股 流動性太差 會被排除
# 舉例 DLR.TO是追蹤美元對加幣匯率的貨幣型ETF 年化波動率只有0.0657 明顯低於MIN_VOLATILITY 也會被排除
# 通過篩選的ETF才會進入grid search 找出讓best_rebound_ratio最大的專屬觸發門檻(trigger_zscore)和最佳持有天數(optimal_days)
# 如果通過篩選並且grid search成功的ETF數量少於100支 就把這些全部存起來 不用再篩一次
# 如果數量達到100支以上 就按照best_rebound_ratio排序取前100名 這部分邏輯跟原本完全一樣
# 輸出檔名會依照最終存進去的ETF數量命名 例如87支就存成candidates_87.csv 100支以上一律存成candidates_100.csv

import pandas as pd
import numpy as np
import yfinance as yf
import time

# ---------------- 參數設定 ----------------
# 回顧幾年的歷史資料 這裡用10年 這個範圍同時也是排除掉ETF剛上市時期的依據
LOOKBACK_YEARS = 10

# 最佳持有天數的搜尋範圍 這裡搜尋1到3天 找下跌後第幾天賣出反彈報酬最高
DAYS_MAX = 3

# 觸發門檻(z_score)的搜尋範圍 對齊ETF_test.py的np.arange(1, 4, 0.5)
# 舉例 觸發門檻是2 代表只挑z_score落在負的3(不含)到負的2(含)這個區間的日子 不是門檻以下全部累加
TRIGGER_ZSCORE_GRID = [1, 1.5, 2, 2.5, 3, 3.5]

# 最少事件數 一個觸發門檻區間如果篩出的下跌事件少於這個數字就不採用
# 這是為了避免用只發生兩三次的極端事件去推論一個穩定的專屬參數 造成過度配適
MIN_EVENTS = 8

# ---- 低波動率定義 兩個條件同時成立才算通過 ----
# 平均日成交量門檻 用來排除XID.TO DTRE這種流動性太差的ETF
# 舉例 XID.TO過去10年平均日成交量只有2826股 遠低於這個門檻 會被排除
MIN_VOLUME = 100000

# 年化波動率下限 用來排除DLR.TO這種貨幣或現金類ETF 這種ETF價格幾乎不太會有劇烈下跌後反彈的事件
# 舉例 DLR.TO的年化波動率是0.0657 低於這個下限 會被排除
MIN_VOLATILITY = 0.10

# 年化波動率上限 用來排除槓桿或反向這類波動過於劇烈的ETF 沿用原本的定義
VOLATILITY_THRESHOLD = 0.6

# 極端值過濾門檻 初步z_score絕對值超過這個數字的日期會被排除 不參與統計基準的計算
# 舉例 某天因為除息或財報跳空 return算出來的初步z_score是5.2 這種日子會被排除
# 排除後用剩下的資料重新算一次mean和std 這組修正過的統計基準才會拿去用在grid search和每日分析
EXTREME_Z_THRESHOLD = 4

# 最終最多取排名前幾名的ETF 如果通過篩選並且grid search成功的數量少於這個數字 就全部保留不做篩選
TOP_N = 100

# 輸入檔案路徑
UNIVERSE_PATH = "universe.csv"

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
    對單一ETF做完整的分析 包含低波動率定義篩選(成交量加波動率區間) 極端值過濾 和grid search
    回傳這支ETF的類別 專屬觸發門檻 最佳持有天數 best_rebound_ratio等欄位
    如果沒通過低波動率定義 資料不足 或找不到符合條件的組合 回傳None
    """
    ticker = yf.Ticker(symbol)

    try:
        hist = ticker.history(period=f"{LOOKBACK_YEARS}y", auto_adjust=True)
    except Exception:
        return None

    if hist.empty or len(hist) < 500:
        # 資料太少代表可能剛上市或資料有問題 直接跳過
        return None

    # ---- 低波動率定義篩選 ----
    # 平均日成交量 用來判斷流動性夠不夠
    # 舉例 hist["Volume"]這欄位過去10年每天的成交股數 取平均就是avg_volume
    avg_volume = hist["Volume"].mean()

    # 年化波動率 用過去10年收盤對收盤的日報酬計算
    close_return = hist["Close"].pct_change().dropna()
    annualized_volatility = close_return.std() * np.sqrt(252)

    if np.isnan(annualized_volatility) or np.isnan(avg_volume):
        return None

    # 成交量不足 或波動率超出區間(太低像貨幣ETF 或太高像槓桿反向ETF) 都不通過
    if avg_volume < MIN_VOLUME:
        return None

    if annualized_volatility < MIN_VOLATILITY or annualized_volatility > VOLATILITY_THRESHOLD:
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
        "avg_volume": round(float(avg_volume), 0),
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


def main():
    universe = pd.read_csv(UNIVERSE_PATH)
    print(f"=== 讀入母體ETF 共 {len(universe)} 支 開始逐一做低波動率定義篩選 極端值過濾與grid search ===")
    print(f"低波動率定義 平均成交量至少{MIN_VOLUME:,}股 年化波動率介於{MIN_VOLATILITY}到{VOLATILITY_THRESHOLD}之間")

    results = []

    for i, row in universe.iterrows():
        if i % PROGRESS_INTERVAL == 0:
            print(f"處理進度 {i}/{len(universe)}")

        params = find_best_params_for_etf(row["symbol"], row["listing_years"])

        if params is not None:
            results.append(params)

        time.sleep(REQUEST_DELAY)

    result_df = pd.DataFrame(results)
    final_count = len(result_df)
    print(f"篩選與grid search完成 共 {final_count} 支ETF通過低波動率定義並找到有效參數組合")

    # 按照best_rebound_ratio由大到小排序 這個排名方式不管哪種情況都一樣
    result_df.sort_values("best_rebound_ratio", ascending=False, inplace=True)

    if final_count < TOP_N:
        # 通過篩選並且grid search成功的數量不足TOP_N 全部保留 不用再篩一次
        output_df = result_df.reset_index(drop=True)
        output_path = f"candidates_{final_count}.csv"
    else:
        # 數量達到TOP_N以上 取前TOP_N名 跟原本的邏輯完全一樣
        output_df = result_df.head(TOP_N).reset_index(drop=True)
        output_path = f"candidates_{TOP_N}.csv"

    # 標記名次 這個名次之後在介面上會用來標記每日分析結果的排名
    output_df["rank"] = output_df.index + 1

    output_df.to_csv(output_path, index=False)
    print(f"=== 完成 已將 {len(output_df)} 支ETF存到 {output_path} ===")


if __name__ == "__main__":
    main()
