# app.py
# 這是整個模型的Streamlit介面 全程手動觸發 沒有排程
# 分成兩個部分
# 第一部分是候選ETF總覽 顯示candidates_100.csv裡的前100名ETF 按照ETF類別分類顯示
# 第二部分是每日分析 按下按鈕後 對這100支ETF抓最新資料 找出當日z_score落在自己專屬觸發區間內或最接近的前十名

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import os
from datetime import datetime

# ---------------- 參數設定 ----------------
CANDIDATES_PATH = "candidates_100.csv"

# 資料存放資料夾 每次按下分析鍵抓到的最新資料會存在這裡
DATA_DIR = r"C:\Users\flyin\OneDrive\桌面\新代碼\Rebound Stock Selection Model\data"

# 每日分析要挑出前幾名
TOP_N_DAILY = 10

# 區間分箱的寬度 必須跟rebound_ranking.py裡的BIN_WIDTH保持一致 不然距離算出來的定義會對不起來
BIN_WIDTH = 0.5

st.set_page_config(page_title="ETF反彈選股模型", layout="wide")


# ---------------- 讀取候選ETF清單 ----------------
@st.cache_data
def load_candidates():
    """
    讀取grid search算好的candidates_100.csv
    這個檔案裡每一列是一支ETF 包含它專屬的trigger_zscore(觸發區間下界) optimal_days(最佳持有天數)等參數
    return_mean和return_std都是已經過濾掉極端值之後算出來的統計基準
    """
    df = pd.read_csv(CANDIDATES_PATH)
    return df


# ---------------- 對單一ETF抓最新資料並計算今天的z_score ----------------
def fetch_today_status(row):
    """
    對一支ETF抓最近幾天的資料 算出今天的return和z_score
    再算出今天z_score跟這支ETF專屬觸發區間的距離
    回傳一個dict 如果抓取失敗回傳None
    """
    symbol = row["symbol"]

    try:
        hist = yf.Ticker(symbol).history(period="10d", auto_adjust=True)
    except Exception:
        return None

    if len(hist) < 2:
        return None

    # 用最新一筆資料當作今天 前一筆資料的收盤價當作昨收
    # 舉例 hist最後一列的Open是150元 倒數第二列的Close是148元 today_return就是150/148約等於1.0135
    prev_close = hist["Close"].iloc[-2]
    today_open = hist["Open"].iloc[-1]
    today_date = hist.index[-1].strftime("%Y-%m-%d")

    if prev_close == 0 or pd.isna(prev_close) or pd.isna(today_open):
        return None

    today_return = today_open / prev_close

    # 用grid search時存下的return_mean和return_std 算出今天的z_score
    # 這組統計基準已經在rebound_ranking.py裡過濾過極端值 這裡要用同一套才會跟排名時的定義一致
    z_today = (today_return - row["return_mean"]) / row["return_std"]

    # 此刻的即時報酬率和z_score 用hist["Close"]最後一筆算 這個欄位在盤中會隨著最新成交價更新
    # 舉例 昨收140元 現在最新成交價140.30元 current_return就是140.30/140約等於1.0021
    # 這組數字反映的是抓資料當下的即時狀態 跟today_return(開盤跳空)是兩個不同時間點的觀察 不能混用
    current_price = hist["Close"].iloc[-1]

    if pd.isna(current_price) or current_price == 0:
        current_return = np.nan
        z_now = np.nan
    else:
        current_return = current_price / prev_close
        z_now = (current_return - row["return_mean"]) / row["return_std"]

    # 這支ETF專屬的觸發區間 下界是負的(trigger_zscore加BIN_WIDTH)不含 上界是負的trigger_zscore含
    # 舉例 trigger_zscore是2 BIN_WIDTH是0.5 區間就是負的2.5(不含)到負的2(含)
    trigger_zscore = row["trigger_zscore"]
    lower_bound = -(trigger_zscore + BIN_WIDTH)
    upper_bound = -trigger_zscore

    # distance是今天z_score(開盤跳空)跟這個區間的距離 如果今天z_score本來就落在區間內 distance算0
    # 舉例 區間是負的2.5到負的2 今天z_today是負的2.3 屬於落在區間內 distance就是0
    # 舉例 今天z_today是負的1.5 比區間上界負的2還高(跌得不夠深) distance就是1.5加負2的差 也就是0.5
    # 距離小不代表今天一定會反彈 只代表今天的狀況比較接近這支ETF歷史上表現最好的那個區間
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
        "trigger_zscore": row["trigger_zscore"],
        "optimal_days": row["optimal_days"],
        "best_rebound_ratio": row["best_rebound_ratio"],
        "today_date": today_date,
        "today_return": round(float(today_return), 4),
        "z_today": round(float(z_today), 2),
        "distance": round(float(distance), 3),
        # 此刻(抓資料當下)的即時報酬率和z_score 跟today_return z_today是不同時間點的兩組數字
        "current_return": round(float(current_return), 4) if not np.isnan(current_return) else None,
        "z_now": round(float(z_now), 2) if not np.isnan(z_now) else None
    }


# ---------------- 執行每日分析 ----------------
def run_daily_analysis(candidates_df, log_placeholder, progress_bar):
    """
    對candidates_df裡的每支ETF呼叫fetch_today_status
    過程中會更新畫面上的進度條和文字紀錄
    回傳一個完整的DataFrame 包含所有候選ETF的今日狀態
    """
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
            # 只保留最後幾行 避免畫面過長
            log_placeholder.text("\n".join(logs[-10:]))

    result_df = pd.DataFrame(records)
    return result_df


# ---------------- 儲存資料 ----------------
def save_snapshot(result_df):
    """
    把今天分析的完整結果存到指定資料夾
    檔名包含日期 這樣每天執行都會留下一份紀錄 不會覆蓋掉之前的資料
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    today_str = datetime.now().strftime("%Y-%m-%d")
    snapshot_path = os.path.join(DATA_DIR, f"snapshot_{today_str}.csv")
    result_df.to_csv(snapshot_path, index=False)
    return snapshot_path


# ---------------- 畫圖 ----------------
def plot_top_etfs(top_df):
    """
    畫出前十名ETF的今日z_score和它們專屬觸發區間上界(負的trigger_zscore)的對照長條圖
    上界是區間裡最接近0的那條邊 可以直接看出每支ETF今天實際落點跟這條邊差多少
    """
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


# ---------------- 主介面 ----------------
def main():
    st.title("ETF反彈選股模型")
    st.caption("手動觸發 不會自動排程執行")

    candidates_df = load_candidates()

    tab1, tab2 = st.tabs(["候選ETF總覽", "每日分析"])

    # ---- 分頁一 候選ETF總覽 ----
    with tab1:
        st.subheader(f"候選ETF池 共 {len(candidates_df)} 支")
        st.caption("排名依據best_rebound_ratio 來自過去10年歷史資料 已過濾極端值 不會隨最新股價變動")

        categories = sorted(candidates_df["category"].dropna().unique())

        for category in categories:
            category_df = candidates_df[candidates_df["category"] == category].sort_values("rank")
            with st.expander(f"{category} ({len(category_df)} 支)"):
                st.dataframe(
                    category_df[[
                        "rank", "symbol", "trigger_zscore", "optimal_days",
                        "best_rebound_ratio", "event_count", "listing_years",
                        "extreme_day_ratio", "annualized_volatility"
                    ]].rename(columns={
                        "rank": "排名",
                        "symbol": "代碼",
                        "trigger_zscore": "觸發門檻震幅(Z值)",
                        "optimal_days": "最佳持有天數",
                        "best_rebound_ratio": "最佳反彈報酬率",
                        "event_count": "歷史事件次數",
                        "listing_years": "上市年限",
                        "extreme_day_ratio": "單日極端變動比例(篩選依據)",
                        "annualized_volatility": "年化波動率(僅供參考)"
                    }),
                    use_container_width=True
                )

    # ---- 分頁二 每日分析 ----
    with tab2:
        st.subheader("執行每日分析")
        st.write("按下按鈕後 會對所有候選ETF抓取最新資料 找出當日z_score落在自己專屬觸發區間內或最接近的前十名")
        st.caption("提醒 距離最近不代表這支ETF在其他區間下不會有更好的表現 只代表今天的狀況最接近它歷史上表現最好的那個區間")

        if st.button("開始分析"):
            progress_bar = st.progress(0.0)
            log_placeholder = st.empty()

            with st.spinner("正在抓取最新資料並計算..."):
                result_df = run_daily_analysis(candidates_df, log_placeholder, progress_bar)

            st.success(f"分析完成 共成功取得 {len(result_df)} 支ETF的最新資料")

            # 存檔
            snapshot_path = save_snapshot(result_df)
            st.write(f"完整結果已存到 {snapshot_path}")

            # 取距離最小的前TOP_N_DAILY名 再依照原始排名重新排序並標記新名次
            top_df = result_df.sort_values("distance").head(TOP_N_DAILY).copy()
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
                    "today_rank": "今日名次",
                    "original_rank": "原始排名(前100)",
                    "symbol": "代碼",
                    "category": "ETF類別",
                    "z_today": "今日z_score(開盤跳空)",
                    "z_now": "此刻z_score",
                    "trigger_zscore": "觸發門檻震幅(Z值)",
                    "distance": "距離",
                    "optimal_days": "最佳持有天數",
                    "best_rebound_ratio": "最佳反彈報酬率",
                    "today_return": "今日開盤對昨收報酬率",
                    "current_return": "此刻對昨收報酬率"
                }),
                use_container_width=True
            )

            st.subheader("今日z_score與專屬觸發區間上界對照圖")
            fig = plot_top_etfs(top_df)
            st.pyplot(fig)

            st.subheader(f"完整{len(candidates_df)}支ETF今日狀態")
            st.dataframe(result_df.sort_values("distance"), use_container_width=True)


if __name__ == "__main__":
    main()
