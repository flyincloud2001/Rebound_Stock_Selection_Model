# app.py
# ETF反彈選股模型 Streamlit介面 全程手動觸發 不會自動排程
# tab1顯示candidates_100.csv裡的100支候選ETF 依類別分類
# tab2按下按鈕後抓每支ETF最新資料 只從「今日z_score為負」的ETF裡 找出離自己專屬觸發區間最近的前十名
# 今日z_score為負代表今天開盤確實跳空下跌 大於等於0代表今天沒有下跌 不算反彈候選

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import os
from datetime import datetime

CANDIDATES_PATH = "candidates_100.csv"
DATA_DIR = r"C:\Users\flyin\OneDrive\桌面\新代碼\Rebound Stock Selection Model\data"
TOP_N_DAILY = 10
BIN_WIDTH = 0.5   # 必須跟rebound_ranking.py裡的BIN_WIDTH保持一致

st.set_page_config(page_title="ETF反彈選股模型", layout="wide")


@st.cache_data
def load_candidates():
    return pd.read_csv(CANDIDATES_PATH)


def fetch_today_status(row):
    """
    抓一支ETF最近幾天資料 算出今天開盤跳空的z_today和此刻即時報酬的z_now
    再算出z_today跟這支ETF專屬觸發區間的距離distance
    """
    symbol = row["symbol"]

    try:
        hist = yf.Ticker(symbol).history(period="10d", auto_adjust=True)
    except Exception:
        return None

    if len(hist) < 2:
        return None

    prev_close = hist["Close"].iloc[-2]
    today_open = hist["Open"].iloc[-1]
    current_price = hist["Close"].iloc[-1]
    today_date = hist.index[-1].strftime("%Y-%m-%d")

    if prev_close == 0 or pd.isna(prev_close) or pd.isna(today_open):
        return None

    # 今天開盤對昨收的報酬率跟z_score 這是主要訊號 對齊rebound_ranking.py的定義
    today_return = today_open / prev_close
    z_today = (today_return - row["return_mean"]) / row["return_std"]

    # 此刻即時報酬跟z_score 用最新成交價算 跟z_today是不同時間點的兩組數字 不能混用
    if pd.isna(current_price) or current_price == 0:
        current_return, z_now = np.nan, np.nan
    else:
        current_return = current_price / prev_close
        z_now = (current_return - row["return_mean"]) / row["return_std"]

    # 專屬觸發區間 上界是負的trigger_zscore 下界是負的(trigger_zscore加BIN_WIDTH)
    trigger_zscore = row["trigger_zscore"]
    upper_bound = -trigger_zscore
    lower_bound = -(trigger_zscore + BIN_WIDTH)

    if z_today > upper_bound:
        distance = z_today - upper_bound
    elif z_today <= lower_bound:
        distance = lower_bound - z_today
    else:
        distance = 0.0

    return {
        "symbol": symbol,
        "category": row["category"],
        "original_rank": row["rank"],
        "trigger_zscore": trigger_zscore,
        "optimal_days": row["optimal_days"],
        "best_rebound_ratio": row["best_rebound_ratio"],
        "today_date": today_date,
        "today_return": round(float(today_return), 3),
        "z_today": round(float(z_today), 2),
        "distance": round(float(distance), 3),
        "current_return": round(float(current_return), 3) if not np.isnan(current_return) else None,
        "z_now": round(float(z_now), 2) if not np.isnan(z_now) else None
    }


def run_daily_analysis(candidates_df, log_placeholder, progress_bar):
    """對candidates_df裡每支ETF呼叫fetch_today_status 過程中更新進度條與文字紀錄"""
    records = []
    total = len(candidates_df)
    logs = []

    for i, row in candidates_df.iterrows():
        result = fetch_today_status(row)
        if result is not None:
            records.append(result)
        else:
            logs.append(f"{row['symbol']} 資料抓取失敗 已略過")

        if i % 10 == 0 or i == total - 1:
            progress_bar.progress(min((i + 1) / total, 1.0))
            logs.append(f"已處理 {i + 1}/{total} 支ETF")
            log_placeholder.text("\n".join(logs[-10:]))

    return pd.DataFrame(records)


def save_snapshot(result_df):
    """把今天完整結果存檔 檔名含日期 每天執行都會留下一份紀錄"""
    os.makedirs(DATA_DIR, exist_ok=True)
    today_str = datetime.now().strftime("%Y-%m-%d")
    snapshot_path = os.path.join(DATA_DIR, f"snapshot_{today_str}.csv")
    result_df.to_csv(snapshot_path, index=False)
    return snapshot_path


def plot_top_etfs(top_df):
    """畫出前十名ETF的今日z_score和專屬觸發區間上界對照長條圖"""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(top_df))
    ax.bar(x - 0.2, top_df["z_today"], width=0.4, label="今天的z_score")
    ax.bar(x + 0.2, -top_df["trigger_zscore"], width=0.4, label="專屬觸發區間上界")
    ax.set_xticks(x)
    ax.set_xticklabels(top_df["symbol"], rotation=45)
    ax.set_ylabel("z_score")
    ax.legend()
    ax.set_title("前十名ETF 今日z_score與專屬觸發區間上界對照")
    return fig


def main():
    st.title("ETF反彈選股模型")
    st.caption("手動觸發 不會自動排程執行")

    candidates_df = load_candidates()
    tab1, tab2 = st.tabs(["候選ETF總覽", "每日分析"])

    # ---- 分頁一 候選ETF總覽 ----
    with tab1:
        st.subheader(f"候選ETF池 共 {len(candidates_df)} 支")
        st.caption("排名依據best_rebound_ratio 來自過去10年歷史資料 已過濾極端值 不會隨最新股價變動")

        for category in sorted(candidates_df["category"].dropna().unique()):
            category_df = candidates_df[candidates_df["category"] == category].sort_values("rank")
            with st.expander(f"{category} ({len(category_df)} 支)"):
                st.dataframe(
                    category_df[[
                        "rank", "symbol", "trigger_zscore", "optimal_days", "best_rebound_ratio",
                        "event_count", "listing_years", "annualized_volatility"
                    ]].rename(columns={
                        "rank": "排名", "symbol": "代碼", "trigger_zscore": "觸發門檻震幅(Z值)",
                        "optimal_days": "最佳持有天數", "best_rebound_ratio": "最佳反彈報酬率",
                        "event_count": "歷史事件次數", "listing_years": "上市年限",
                        "annualized_volatility": "近1年年化波動率(篩選依據)"
                    }),
                    use_container_width=True
                )

    # ---- 分頁二 每日分析 ----
    with tab2:
        st.subheader("執行每日分析")
        st.write("按下按鈕後 會對所有候選ETF抓取最新資料 只從今日z_score為負的ETF裡 找出離自己專屬觸發區間最近的前十名")
        st.caption("今日z_score為負代表今天開盤確實跳空下跌 大於等於0代表今天沒有下跌 不會列入候選")

        if st.button("開始分析"):
            progress_bar = st.progress(0.0)
            log_placeholder = st.empty()

            with st.spinner("正在抓取最新資料並計算..."):
                result_df = run_daily_analysis(candidates_df, log_placeholder, progress_bar)

            st.success(f"分析完成 共成功取得 {len(result_df)} 支ETF的最新資料")

            snapshot_path = save_snapshot(result_df)
            st.write(f"完整結果已存到 {snapshot_path}")

            # 只從今日z_score為負(今天確實開盤下跌)的ETF裡挑選 正值代表今天沒有下跌 不是反彈候選
            drop_only_df = result_df[result_df["z_today"] < 0]

            top_df = drop_only_df.sort_values("distance").head(TOP_N_DAILY).copy()
            top_df = top_df.sort_values("original_rank").reset_index(drop=True)
            top_df["today_rank"] = top_df.index + 1

            st.subheader(f"今日前{TOP_N_DAILY}名 依原始排名排序")
            st.caption("觸發門檻震幅(Z)是區間裡最接近0那一邊的絕對值 例如數值2代表區間是負2.5(不含)到負2(含) 不是下界")
            st.dataframe(
                top_df[[
                    "today_rank", "original_rank", "symbol", "category", "z_today", "z_now",
                    "trigger_zscore", "distance", "optimal_days", "best_rebound_ratio",
                    "today_return", "current_return"
                ]].rename(columns={
                    "today_rank": "今日名次", "original_rank": "原始排名(前100)", "symbol": "代碼",
                    "category": "ETF類別", "z_today": "今日z_score(開盤跳空)", "z_now": "此刻z_score",
                    "trigger_zscore": "觸發門檻震幅(Z值)", "distance": "距離",
                    "optimal_days": "最佳持有天數", "best_rebound_ratio": "最佳反彈報酬率",
                    "today_return": "今日開盤對昨收報酬率", "current_return": "此刻對昨收報酬率"
                }),
                use_container_width=True
            )

            st.subheader("今日z_score與專屬觸發區間上界對照圖")
            st.pyplot(plot_top_etfs(top_df))

            st.subheader(f"完整{len(candidates_df)}支ETF今日狀態")
            st.dataframe(result_df.sort_values("distance"), use_container_width=True)


if __name__ == "__main__":
    main()
