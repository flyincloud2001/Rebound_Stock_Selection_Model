# rebound_ranking.py
# 對universe.csv裡每支ETF做波動率篩選 短期急跌篩選 極端值過濾 再用grid search找出專屬trigger_zscore和optimal_days
# 只需要在universe.csv更新後手動重跑一次 排名依據過去10年歷史資料 不會隨最新股價變動

import pandas as pd
import numpy as np
import yfinance as yf
import time

# ---------------- 參數設定 ----------------
LOOKBACK_YEARS = 10   # 回顧年數 同時排除ETF剛上市不穩定的時期
DAYS_MAX = 3           # 最佳持有天數搜尋範圍 1到3天

# 觸發門檻(z_score)搜尋範圍 對齊ETF_test.py的np.arange(1,4,0.5) 區間寬度跟間距一致 各組threshold不重疊
# 舉例 trigger_zscore=2 代表只抓z_score落在負2.5(不含)到負2(含)這個區間的日子
TRIGGER_ZSCORE_GRID = [1, 1.5, 2, 2.5, 3, 3.5]
BIN_WIDTH = 0.5

MIN_EVENTS = 8   # 一個區間篩出的事件數少於這個數字就不採用 避免過度配適

# 低波動率篩選 不用年化標準差 因為抓不到DIG這種槓桿ETF的問題 改用單日極端變動比例
EXTREME_DAY_THRESHOLD = 0.03    # 單日收盤對開盤變動超過3%算一次極端日
MAX_EXTREME_DAY_RATIO = 0.02    # 極端日比例不能超過2%

# 短期急跌篩選 抓短時間內集中發生的跌幅 排除跟大盤同期重挫的系統性事件
RAPID_DECLINE_WINDOW = 10           # 短窗口天數
RAPID_DECLINE_THRESHOLD = -0.15     # 窗口內跌幅超過15%算一次急跌事件
BENCHMARK_SYMBOL = "SPY"            # 系統性風險判斷基準
BENCHMARK_CRISIS_THRESHOLD = -0.08  # 基準同期也跌超過這個比例 視為系統性風險 不計入違規
MAX_IDIOSYNCRATIC_CRASH_COUNT = 2   # 非系統性急跌次數上限

EXTREME_Z_THRESHOLD = 4   # 初步z_score絕對值超過這個數字的日期排除 不參與統計基準計算

TOP_N = 100
UNIVERSE_PATH = "universe.csv"
OUTPUT_PATH = "candidates_100.csv"
PROGRESS_INTERVAL = 20
REQUEST_DELAY = 0.3


def _get_etf_category(ticker):
    """查詢ETF類別 查詢失敗回傳未分類 不影響其他篩選"""
    try:
        return ticker.info.get("category") or "未分類"
    except Exception:
        return "未分類"


def _count_idiosyncratic_crashes(close, spy_close):
    """
    計算短期急跌次數中 有幾次不是跟大盤同期重挫的系統性事件
    舉例 某段10天跌了18% 同期SPY只跌了3% 這就算一次非系統性急跌
    """
    rolling_return = close / close.shift(RAPID_DECLINE_WINDOW) - 1
    crash_dates = rolling_return[rolling_return < RAPID_DECLINE_THRESHOLD].index

    count = 0
    for d in crash_dates:
        if d not in spy_close.index:
            continue
        spy_window = spy_close.loc[:d].iloc[-RAPID_DECLINE_WINDOW:]
        if len(spy_window) < RAPID_DECLINE_WINDOW:
            continue
        spy_window_return = spy_window.iloc[-1] / spy_window.iloc[0] - 1
        if spy_window_return >= BENCHMARK_CRISIS_THRESHOLD:
            count += 1
    return count


def find_best_params_for_etf(symbol, listing_years, spy_close):
    """
    對單一ETF做波動率篩選 短期急跌篩選 極端值過濾 和grid search
    回傳專屬trigger_zscore optimal_days best_rebound_ratio等欄位 不符合任一篩選條件回傳None
    """
    ticker = yf.Ticker(symbol)

    try:
        hist = ticker.history(period=f"{LOOKBACK_YEARS}y", auto_adjust=True)
    except Exception:
        return None

    if hist.empty or len(hist) < 500:
        return None

    # 單日收盤對開盤變動比例篩選
    intraday_return = hist["Close"] / hist["Open"] - 1
    extreme_day_ratio = (intraday_return.abs() > EXTREME_DAY_THRESHOLD).mean()
    if np.isnan(extreme_day_ratio) or extreme_day_ratio > MAX_EXTREME_DAY_RATIO:
        return None

    # 短期急跌篩選 排除掉太常自己單獨重挫的ETF
    idiosyncratic_crash_count = _count_idiosyncratic_crashes(hist["Close"], spy_close)
    if idiosyncratic_crash_count > MAX_IDIOSYNCRATIC_CRASH_COUNT:
        return None

    # 年化波動率不當篩選依據 只算出來給介面參考
    annualized_volatility = hist["Close"].pct_change().dropna().std() * np.sqrt(252)

    # 準備grid search資料 return是開盤對前一日收盤的報酬率 是偵測下跌事件的訊號
    data = hist[["Close", "High", "Open"]].copy()
    data["return"] = data["Open"] / data["Close"].shift(1)
    data.dropna(subset=["return"], inplace=True)
    if len(data) < 500:
        return None

    # 極端值過濾 先算初步統計基準 排除|z|>4的日子 再用剩下的資料重算正式基準
    raw_mean, raw_std = data["return"].mean(), data["return"].std()
    if raw_std == 0 or np.isnan(raw_std):
        return None
    raw_z = (data["return"] - raw_mean) / raw_std
    good_data = data[raw_z.abs() <= EXTREME_Z_THRESHOLD]
    if len(good_data) < 500:
        return None

    return_mean, return_std = good_data["return"].mean(), good_data["return"].std()
    if return_std == 0 or np.isnan(return_std):
        return None

    # 用修正過的統計基準 對原始全部資料重新算z_score
    data["z"] = (data["return"] - return_mean) / return_std
    total_trading_days = len(data)

    for day in range(1, DAYS_MAX + 1):
        data[f"rebound_{day}"] = data["Open"].shift(-day) / data["Close"]

    best_combo = None  # (trigger_zscore, optimal_days, best_ratio, event_count)

    for trigger_zscore in TRIGGER_ZSCORE_GRID:
        # 區間分箱 threshold是2的話 抓的是負2.5(不含)到負2(含)
        lower_bound = -(trigger_zscore + BIN_WIDTH)
        upper_bound = -trigger_zscore
        drop_mask = (data["z"] > lower_bound) & (data["z"] <= upper_bound)
        event_count = int(drop_mask.sum())
        if event_count < MIN_EVENTS:
            continue

        for day in range(1, DAYS_MAX + 1):
            valid_mask = drop_mask & data[f"rebound_{day}"].notna()
            if valid_mask.sum() < MIN_EVENTS:
                continue
            avg_ratio = data.loc[valid_mask, f"rebound_{day}"].mean()
            if best_combo is None or avg_ratio > best_combo[2]:
                best_combo = (trigger_zscore, day, avg_ratio, event_count)

    if best_combo is None:
        return None

    trigger_zscore, optimal_days, best_ratio, event_count = best_combo

    return {
        "symbol": symbol,
        "category": _get_etf_category(ticker),
        "listing_years": listing_years,
        "annualized_volatility": round(float(annualized_volatility), 4),
        "extreme_day_ratio": round(float(extreme_day_ratio), 4),
        "idiosyncratic_crash_count": int(idiosyncratic_crash_count),
        "trigger_zscore": float(trigger_zscore),   # float因為grid裡有1.5 2.5 3.5 用int會被截斷
        "optimal_days": int(optimal_days),
        "best_rebound_ratio": round(float(best_ratio), 3),
        "event_count": event_count,
        "total_trading_days": total_trading_days,
        "return_mean": round(float(return_mean), 6),
        "return_std": round(float(return_std), 6)
    }


def main():
    universe = pd.read_csv(UNIVERSE_PATH)
    print(f"=== 讀入母體ETF 共 {len(universe)} 支 開始篩選與grid search ===")

    # 先抓一次SPY資料 當作判斷系統性風險的共用基準 不用每支ETF都重抓
    spy_close = yf.Ticker(BENCHMARK_SYMBOL).history(period=f"{LOOKBACK_YEARS}y", auto_adjust=True)["Close"]

    results = []
    for i, row in universe.iterrows():
        if i % PROGRESS_INTERVAL == 0:
            print(f"處理進度 {i}/{len(universe)}")
        params = find_best_params_for_etf(row["symbol"], row["listing_years"], spy_close)
        if params is not None:
            results.append(params)
        time.sleep(REQUEST_DELAY)

    result_df = pd.DataFrame(results)
    print(f"完成 共 {len(result_df)} 支ETF找到有效參數組合")

    result_df.sort_values("best_rebound_ratio", ascending=False, inplace=True)
    result_df = result_df.head(TOP_N).reset_index(drop=True)
    result_df["rank"] = result_df.index + 1

    result_df.to_csv(OUTPUT_PATH, index=False)
    print(f"=== 已將前{TOP_N}名ETF存到 {OUTPUT_PATH} ===")


if __name__ == "__main__":
    main()
