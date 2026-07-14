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

# 低波動率篩選 用S&P 500 Low Volatility Index官方方法論的定義
# 波動率是過去252個交易日(近1年)每日報酬率標準差 再年化
# 25%這個門檻是實測校準出來的 VOO 12.5% SPY 12.6% ZEB.TO 13.3% QQQ 18.6%都在門檻內
# DIG這種2倍槓桿ETF算出來是42% EWV CURE等槓桿ETF都在30%以上 會被乾淨排除
VOLATILITY_WINDOW = 252
VOLATILITY_THRESHOLD = 0.25

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


def find_best_params_for_etf(symbol, listing_years):
    """
    對單一ETF做波動率篩選 極端值過濾 和grid search
    回傳專屬trigger_zscore optimal_days best_rebound_ratio等欄位 不符合任一篩選條件回傳None
    """
    ticker = yf.Ticker(symbol)

    try:
        hist = ticker.history(period=f"{LOOKBACK_YEARS}y", auto_adjust=True)
    except Exception:
        return None

    if hist.empty or len(hist) < 500:
        return None

    # 波動率篩選 用S&P官方定義 近1年每日報酬率標準差 年化後跟VOLATILITY_THRESHOLD比較
    close_return = hist["Close"].pct_change().dropna()
    annualized_volatility = close_return.tail(VOLATILITY_WINDOW).std() * np.sqrt(252)
    if np.isnan(annualized_volatility) or annualized_volatility > VOLATILITY_THRESHOLD:
        return None

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

    results = []
    for i, row in universe.iterrows():
        if i % PROGRESS_INTERVAL == 0:
            print(f"處理進度 {i}/{len(universe)}")
        params = find_best_params_for_etf(row["symbol"], row["listing_years"])
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
