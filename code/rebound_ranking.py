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

# 最佳持有天數的搜尋範圍 這裡搜尋1到3天 找下跌後第幾天賣出反彈報酬最高
DAYS_MAX = 3

# 觸發門檻(z_score)的搜尋範圍 對齊ETF_test.py的np.arange(1, 4, 0.5)
# 舉例 觸發門檻是2 代表只挑z_score落在負的2.5(不含)到負的2(含)這個區間的日子 不是門檻以下全部累加
# 區間寬度跟grid間距一樣都是0.5 讓各組threshold切出來的區間完全不重疊 不會有同一批下跌事件被兩組threshold重複算到
# 這個grid乘以DAYS_MAX共6乘3等於18種組合 每支ETF都會逐一試過這18組 取best_rebound_ratio最高的那一組
TRIGGER_ZSCORE_GRID = [1, 1.5, 2, 2.5, 3, 3.5]

# 區間分箱的寬度 跟TRIGGER_ZSCORE_GRID的間距保持一致 這樣切出來的區間才會完全不重疊
BIN_WIDTH = 0.5

# 最少事件數 一個觸發門檻區間如果篩出的下跌事件少於這個數字就不採用
# 這是為了避免用只發生兩三次的極端事件去推論一個穩定的專屬參數 造成過度配適
MIN_EVENTS = 8

# 年化波動率不再拿來當篩選依據 因為它抓不到DIG這種槓桿型ETF的問題
# DIG年化波動率算出來是57.8% 沒超過原本60%的門檻 但它單日收盤對開盤變動超過3%的比例高達22.16%
# 對照ZEB.TO ZEB.TO SPY GLD這類真正低波動的ETF 這個比例都在1%以下
# 所以改用單日極端變動比例當作低波動率的判斷依據 這個指標才抓得到真正的日內劇烈震盪

# 單日收盤對開盤變動幅度超過這個比例算一次極端日
# 舉例 某天開盤64元 收盤72元 這天的intraday_return是72除以64減1約等於12.5% 超過3%算一次極端日
EXTREME_DAY_THRESHOLD = 0.03

# 過去10年裡 極端日次數占總交易天數的比例不能超過這個上限 超過就代表這支ETF平常波動就很劇烈
MAX_EXTREME_DAY_RATIO = 0.02

# 極端值過濾門檻 初步z_score絕對值超過這個數字的日期會被排除 不參與統計基準的計算
# 舉例 某天因為除息或財報跳空 return算出來的初步z_score是5.2 這種日子會被排除
# 排除後用剩下的資料重新算一次mean和std 這組修正過的統計基準才會拿去用在grid search和每日分析
EXTREME_Z_THRESHOLD = 4

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

    # ---- 波動率篩選 改用單日收盤對開盤變動幅度的極端日比例來判斷 不再只看年化標準差 ----
    # 舉例 某天開盤64元 收盤72元 intraday_return是72除以64減1約等於0.125 也就是12.5%
    intraday_return = hist["Close"] / hist["Open"] - 1
    extreme_day_ratio = (intraday_return.abs() > EXTREME_DAY_THRESHOLD).mean()

    if np.isnan(extreme_day_ratio) or extreme_day_ratio > MAX_EXTREME_DAY_RATIO:
        return None

    # 年化波動率不再當作篩選依據 但還是算出來存進輸出檔案 給介面顯示用 方便對照參考
    close_return = hist["Close"].pct_change().dropna()
    annualized_volatility = close_return.std() * np.sqrt(252)

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
        # 舉例 trigger_zscore是2 區間就是負的2.5(不含)到負的2(含) 只抓落在這個區間的日子
        # 寬度用BIN_WIDTH 跟grid間距一致 讓不同threshold之間切出來的區間不重疊
        lower_bound = -(trigger_zscore + BIN_WIDTH)
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
        # 單日收盤對開盤變動超過3%的日子占總交易天數的比例 這是現在實際拿來篩選低波動率的依據
        "extreme_day_ratio": round(float(extreme_day_ratio), 4),
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
    print(f"=== 讀入母體ETF 共 {len(universe)} 支 開始逐一做波動率篩選 極端值過濾與grid search ===")

    results = []

    for i, row in universe.iterrows():
        if i % PROGRESS_INTERVAL == 0:
            print(f"處理進度 {i}/{len(universe)}")

        params = find_best_params_for_etf(row["symbol"], row["listing_years"])

        if params is not None:
            results.append(params)

        time.sleep(REQUEST_DELAY)

    result_df = pd.DataFrame(results)
    print(f"波動率篩選 極端值過濾與grid search完成 共 {len(result_df)} 支ETF找到有效參數組合")

    # 按照best_rebound_ratio由大到小排序 取前TOP_N名
    result_df.sort_values("best_rebound_ratio", ascending=False, inplace=True)
    result_df = result_df.head(TOP_N).reset_index(drop=True)

    # 標記原始名次 這個名次之後在介面上會用來標記每日分析結果的排名
    result_df["rank"] = result_df.index + 1

    result_df.to_csv(OUTPUT_PATH, index=False)
    print(f"=== 完成 已將前{TOP_N}名ETF存到 {OUTPUT_PATH} ===")


if __name__ == "__main__":
    main()
