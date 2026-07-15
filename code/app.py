# app.py
# 這是整個模型的Streamlit介面 全程手動觸發 沒有排程
# 分成兩個部分
# 第一部分是候選ETF總覽 顯示candidates檔案裡的ETF 按照ETF類別分類顯示
# 第二部分是每日分析 按下按鈕後 對這些ETF抓最新資料 找出昨日z_score落在自己專屬觸發區間內或最接近的前二十名
#
# 這一版的交易邏輯是盤前交易 用昨天收盤對昨天開盤的當日內漲跌當作訊號
# 這個訊號昨天收盤後就已經確定 可以在今天開盤前用這個訊號盤前掛單 用今天的開盤價買進
# 賺的是進場後這幾天內反彈的價差

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import os
import glob
from datetime import datetime

# ---------------- 參數設定 ----------------
# 資料存放資料夾 每次按下分析鍵抓到的最新資料會存在這裡
DATA_DIR = r"C:\Users\flyin\OneDrive\桌面\新代碼\Rebound Stock Selection Model\data"

# 每日分析要挑出前幾名
TOP_N_DAILY = 20

st.set_page_config(page_title="ETF反彈選股模型", layout="wide")


# ---------------- 手動分類對照表 ----------------
# category只是拿來給介面分組顯示用的 不影響trigger_zscore grid search這些核心運算
# 加拿大上市的ETF在yfinance裡常常查不到category 導致rebound_ranking.py存進candidates檔案時標記成"未分類"
# 這個對照表是針對這些查不到category的ETF 依照它們的longName和實際持有內容手動查證分類出來的
# 放在app.py這裡 讀取candidates檔案時直接套用 不需要重新跑一次rebound_ranking.py(重跑要花很久去抓194支股票的10年歷史資料)
# 如果之後universe.csv換了一批新的加拿大ETF 出現不在這個表裡的代碼 還是會顯示"未分類"
# 到時候要再手動查證新增進來
MANUAL_CATEGORY_OVERRIDES = {
    # 債券ETF
    "XCB.TO": "債券(加拿大投資等級公司債)",
    "XLB.TO": "債券(加拿大長天期債券)",
    "XRB.TO": "債券(加拿大實質報酬債券)",
    "ZFL.TO": "債券(加拿大聯邦長天期債券)",
    "CBO.TO": "債券(加拿大公司債梯型1至5年)",
    "XIG.TO": "債券(美國投資等級公司債避險)",
    "HAB.TO": "債券(主動管理公司債)",
    "ZRR.TO": "債券(實質報酬債券)",
    "XSB.TO": "債券(加拿大短天期債券)",
    "ZCM.TO": "債券(中天期公司債)",
    "ZCS.TO": "債券(短天期公司債)",
    "CLF.TO": "債券(加拿大政府債梯型1至5年)",
    "XQB.TO": "債券(高評級加拿大債券)",
    "ZPS.TO": "債券(短天期省政府債)",
    "XBB.TO": "債券(加拿大全市場債券)",
    "ZAG.TO": "債券(加拿大綜合債券)",
    # 不動產REIT
    "RIT.TO": "不動產REIT(加拿大)",
    "CGR.TO": "不動產REIT(全球)",
    "ZRE.TO": "不動產REIT(加拿大等權重)",
    "XRE.TO": "不動產REIT(加拿大市值加權)",
    # 加拿大金融股區塊
    "CEW.TO": "加拿大金融股(銀行保險等權重)",
    "ZWB.TO": "加拿大金融股(銀行備兌買權)",
    "ZEB.TO": "加拿大金融股(銀行等權重)",
    # 商品原物料
    "HUC.TO": "商品原物料(原油)",
    "XGD.TO": "商品原物料(全球黃金股)",
    "ZMT.TO": "商品原物料(全球基本金屬避險)",
    "XEG.TO": "商品原物料(加拿大能源類股)",
    "HUN.TO": "商品原物料(天然氣)",
    "GLCC.TO": "商品原物料(黃金生產商備兌買權)",
    # 單一國家或區域股票
    "XCH.TO": "單一國家或區域股票(中國)",
    "ZCH.TO": "單一國家或區域股票(中國)",
    "XEM.TO": "單一國家或區域股票(新興市場)",
    "ZEM.TO": "單一國家或區域股票(新興市場)",
    "ZDM.TO": "單一國家或區域股票(已開發市場避險)",
    # 美股廣泛指數 但在加拿大掛牌查不到category
    "XSP.TO": "美股廣泛指數(S&P500避險)",
    "ZQQ.TO": "美股廣泛指數(Nasdaq100避險)",
    "HXS.TO": "美股廣泛指數(S&P500公司股份結構)",
    "XSU.TO": "美股廣泛指數(美國小型股避險)",
    # 加拿大廣泛股票指數
    "XIU.TO": "加拿大廣泛股票指數(TSX60)",
    "ZCN.TO": "加拿大廣泛股票指數(TSX綜合指數)",
    "XCS.TO": "加拿大廣泛股票指數(TSX小型股)",
    "XCV.TO": "加拿大廣泛股票指數(價值型)",
    "XCG.TO": "加拿大廣泛股票指數(成長型)",
    "XDV.TO": "加拿大廣泛股票指數(精選股息)",
    "CDZ.TO": "加拿大廣泛股票指數(股息貴族)",
    # 全球或多元資產配置
    "XWD.TO": "全球或多元資產配置(MSCI世界指數)",
    "XBAL.TO": "全球或多元資產配置(核心平衡型)",
    "CGAA.TO": "全球或多元資產配置(全球資產配置私募池)",
    # 優先股
    "HPR.TO": "優先股(主動管理)",
    # 基礎建設
    "CIF.TO": "基礎建設(全球)"
}


# ---------------- 找出candidates檔案 ----------------
def _find_candidates_file():
    """
    找出目前資料夾裡的candidates檔案
    因為rebound_ranking.py現在會依照通過低波動率定義並完成grid search的實際股票數量來命名檔案
    例如candidates_194.csv 檔名不再固定
    這裡抓資料夾裡符合candidates_開頭的csv檔 如果有多個就取最新修改的那個
    """
    matches = glob.glob("candidates_*.csv")

    if not matches:
        raise FileNotFoundError("找不到candidates檔案 請先執行rebound_ranking.py產生結果")

    matches.sort(key=os.path.getmtime, reverse=True)
    return matches[0]


# ---------------- 讀取候選ETF清單 ----------------
@st.cache_data
def load_candidates():
    """
    讀取grid search算好的candidates檔案
    這個檔案裡每一列是一支ETF 包含它專屬的trigger_zscore(觸發區間上界 已經是負值)和optimal_days(最佳持有天數)等參數
    return_mean和return_std都是已經過濾掉極端值之後算出來的統計基準 用的是當天收盤對當天開盤的報酬率
    """
    path = _find_candidates_file()
    df = pd.read_csv(path)

    # 把category是"未分類"的股票 拿symbol去MANUAL_CATEGORY_OVERRIDES查有沒有對應的手動分類
    # 查得到就覆蓋掉 查不到就維持"未分類" 這一步不會動到CSV檔案本身 只影響畫面上顯示的結果
    def _apply_override(row):
        if row["category"] == "未分類" and row["symbol"] in MANUAL_CATEGORY_OVERRIDES:
            return MANUAL_CATEGORY_OVERRIDES[row["symbol"]]
        return row["category"]

    df["category"] = df.apply(_apply_override, axis=1)

    return df, path


# ---------------- 對單一ETF抓最新資料並計算昨天的z_score ----------------
def fetch_yesterday_status(row):
    """
    對一支ETF抓最近幾天的資料 算出昨天收盤對昨天開盤的漲跌幅和z_score
    這個z_score昨天收盤後就已經確定 用來決定今天要不要盤前掛單買進
    再算出昨天的z_score跟這支ETF專屬觸發區間的距離
    回傳一個dict 如果抓取失敗回傳None
    """
    symbol = row["symbol"]

    try:
        hist = yf.Ticker(symbol).history(period="10d", auto_adjust=True)
    except Exception:
        return None

    if len(hist) < 2:
        return None

    # hist.index是用美東時間(America/New_York)標記的 不能拿datetime.now()這種本地系統時間去比對
    # 因為Foster的電腦在台灣 datetime.now()回傳的是台灣時間 台灣比美東快12到13小時
    # 如果直接比較 會把還沒收盤的美股session誤判成已經收盤 或反過來 判斷完全錯亂
    # 正確做法是用hist.index[-1]自己帶的時區資訊 換算出"現在"在美東時間是幾點幾號
    now_in_market_tz = pd.Timestamp.now(tz=hist.index[-1].tz)
    last_row_date = hist.index[-1].date()
    today_date_in_market_tz = now_in_market_tz.date()

    # 只比較日期還不夠 因為即使美東日期還是同一天 如果已經過了美東下午4點收盤時間
    # 這一天的資料其實已經是完整的一天了 不該再當作"還在進行中的今天"
    # 舉例 台灣早上8點查詢 換算成美東是前一天晚上8點 已經收盤4小時 這時候hist最後一列就該當完整的一天處理
    market_close_time = now_in_market_tz.replace(hour=16, minute=0, second=0, microsecond=0)
    last_row_still_in_progress = (last_row_date == today_date_in_market_tz) and (now_in_market_tz < market_close_time)

    if last_row_still_in_progress:
        yesterday_open = hist["Open"].iloc[-2]
        yesterday_close = hist["Close"].iloc[-2]
        yesterday_date = hist.index[-2].strftime("%Y-%m-%d")
    else:
        yesterday_open = hist["Open"].iloc[-1]
        yesterday_close = hist["Close"].iloc[-1]
        yesterday_date = hist.index[-1].strftime("%Y-%m-%d")

    if yesterday_open == 0 or pd.isna(yesterday_open) or pd.isna(yesterday_close):
        return None

    yesterday_return = yesterday_close / yesterday_open

    # 用grid search時存下的return_mean和return_std 算出昨天的z_score
    # 這組統計基準已經在rebound_ranking.py裡過濾過極端值 這裡要用同一套才會跟排名時的定義一致
    z_yesterday = (yesterday_return - row["return_mean"]) / row["return_std"]

    # trigger_zscore現在本身就是負值 直接當作區間上界使用 不用再加負號
    # 舉例 trigger_zscore是負2 區間就是負3(不含)到負2(含)
    trigger_zscore = row["trigger_zscore"]
    upper_bound = trigger_zscore
    lower_bound = trigger_zscore - 1

    # distance是昨天z_score跟這個區間的距離 如果昨天z_score本來就落在區間內 distance算0
    # 舉例 區間是負3到負2 昨天z_yesterday是負2.4 屬於落在區間內 distance就是0
    # 舉例 昨天z_yesterday是負1.5 比區間上界負2還高(跌得不夠深) distance就是1.5減2的差 也就是0.5
    # 距離小不代表買進後一定會反彈 只代表昨天的狀況比較接近這支ETF歷史上表現最好的那個區間
    if z_yesterday > upper_bound:
        distance = z_yesterday - upper_bound
    elif z_yesterday <= lower_bound:
        distance = lower_bound - z_yesterday
    else:
        distance = 0.0

    return {
        "symbol": symbol,
        "category": row["category"],
        "original_rank": row["rank"],
        "trigger_zscore": row["trigger_zscore"],
        "optimal_days": row["optimal_days"],
        "best_rebound_ratio": row["best_rebound_ratio"],
        "yesterday_date": yesterday_date,
        "yesterday_return": round(float(yesterday_return), 4),
        "z_yesterday": round(float(z_yesterday), 2),
        "distance": round(float(distance), 3)
    }


# ---------------- 執行每日分析 ----------------
def run_daily_analysis(candidates_df, log_placeholder, progress_bar):
    """
    對candidates_df裡的每支ETF呼叫fetch_yesterday_status
    過程中會更新畫面上的進度條和文字紀錄
    回傳一個完整的DataFrame 包含所有候選ETF昨天的狀態
    """
    records = []
    total = len(candidates_df)
    logs = []

    for i, row in candidates_df.iterrows():
        result = fetch_yesterday_status(row)

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
    畫出前N名ETF昨天的z_score和它們專屬觸發區間上界的對照長條圖
    trigger_zscore本身已經是負值 直接拿來畫 不用再加負號
    上界是區間裡最接近0的那條邊 可以直接看出每支ETF昨天實際落點跟這條邊差多少
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(top_df))

    ax.bar(x - 0.2, top_df["z_yesterday"], width=0.4, label="昨天的z_score")
    ax.bar(x + 0.2, top_df["trigger_zscore"], width=0.4, label="專屬觸發區間上界")

    ax.set_xticks(x)
    ax.set_xticklabels(top_df["symbol"], rotation=45)
    ax.set_ylabel("z_score")
    ax.legend()
    ax.set_title(f"前{TOP_N_DAILY}名ETF 昨日z_score與專屬觸發區間上界對照")

    return fig


# ---------------- 主介面 ----------------
def main():
    st.title("ETF反彈選股模型")
    st.caption("手動觸發 不會自動排程執行 訊號用昨天收盤對昨天開盤算出 可在今天開盤前盤前掛單")

    candidates_df, candidates_path = load_candidates()

    tab1, tab2 = st.tabs(["候選ETF總覽", "每日分析"])

    # ---- 分頁一 候選ETF總覽 ----
    with tab1:
        st.subheader(f"候選ETF池 共 {len(candidates_df)} 支")
        st.caption(f"讀取自 {candidates_path}")
        st.caption("先通過低波動率定義篩選 再依best_rebound_ratio排序 來自過去10年歷史資料 已過濾極端值 不會隨最新股價變動")

        categories = sorted(candidates_df["category"].dropna().unique())

        for category in categories:
            category_df = candidates_df[candidates_df["category"] == category].sort_values("rank")
            with st.expander(f"{category} ({len(category_df)} 支)"):
                st.dataframe(
                    category_df[[
                        "rank", "symbol", "trigger_zscore", "optimal_days",
                        "best_rebound_ratio", "event_count", "listing_years", "annualized_volatility"
                    ]].rename(columns={
                        "rank": "排名",
                        "symbol": "代碼",
                        "trigger_zscore": "專屬觸發區間下界門檻(Z值)",
                        "optimal_days": "最佳持有天數",
                        "best_rebound_ratio": "最佳反彈報酬率(預測)",
                        "event_count": "歷史事件次數",
                        "listing_years": "上市年限",
                        "annualized_volatility": "年化波動率"
                    }),
                    use_container_width=True
                )

    # ---- 分頁二 每日分析 ----
    with tab2:
        st.subheader("執行每日分析")
        st.write(f"按下按鈕後 會對所有候選ETF抓取最新資料 找出昨日z_score落在自己專屬觸發區間內或最接近的前{TOP_N_DAILY}名")
        st.caption("提醒 距離最近不代表這支ETF買進後一定會反彈 只代表昨天的狀況最接近它歷史上表現最好的那個區間")

        if st.button("開始分析"):
            progress_bar = st.progress(0.0)
            log_placeholder = st.empty()

            with st.spinner("正在抓取最新資料並計算..."):
                result_df = run_daily_analysis(candidates_df, log_placeholder, progress_bar)

            st.success(f"分析完成 共成功取得 {len(result_df)} 支ETF的最新資料")

            # 存檔
            snapshot_path = save_snapshot(result_df)
            st.write(f"完整結果已存到 {snapshot_path}")

            # z_yesterday不可以是正的 正代表昨天其實是上漲 不符合找下跌後反彈訊號的前提 這種股票不列入候選
            # 舉例 某支ETF昨天z_yesterday是0.25 代表昨天收盤比開盤還高 即使distance數字小也要排除
            down_only_df = result_df[result_df["z_yesterday"] < 0].copy()

            # 取距離最小的前TOP_N_DAILY名 再依照原始排名重新排序並標記新名次
            top_df = down_only_df.sort_values("distance").head(TOP_N_DAILY).copy()
            top_df = top_df.sort_values("original_rank").reset_index(drop=True)
            top_df["today_rank"] = top_df.index + 1

            st.subheader(f"今日前{TOP_N_DAILY}名 依原始排名排序")
            st.caption("以下的z_score和漲跌幅都是昨天的資料 今天開盤前可以參考這份名單盤前掛單")
            st.dataframe(
                top_df[[
                    "today_rank", "original_rank", "symbol", "category", "z_yesterday",
                    "yesterday_return", "trigger_zscore", "distance", "optimal_days",
                    "best_rebound_ratio"
                ]].rename(columns={
                    "today_rank": "今日名次",
                    "original_rank": "原始排名",
                    "symbol": "代碼",
                    "category": "ETF類別",
                    "z_yesterday": "昨日z_score",
                    "yesterday_return": "昨日漲跌幅",
                    "trigger_zscore": "專屬觸發區間下界門檻(Z值)",
                    "distance": "距離",
                    "optimal_days": "最佳持有天數",
                    "best_rebound_ratio": "最佳反彈報酬率(預測)"
                }),
                use_container_width=True
            )

            st.subheader("昨日z_score與專屬觸發區間上界對照圖")
            fig = plot_top_etfs(top_df)
            st.pyplot(fig)

            st.subheader(f"完整{len(candidates_df)}支ETF昨日狀態")
            st.dataframe(result_df.sort_values("distance"), use_container_width=True)


if __name__ == "__main__":
    main()
