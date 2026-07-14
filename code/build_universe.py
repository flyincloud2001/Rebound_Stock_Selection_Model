# build_universe.py
# 這支程式負責建立候選ETF母體
# 流程是抓取美股和加股所有的ETF(不含個股) 再用上市年限和IBKR是否可交易兩個條件篩選
# 波動率的篩選移到下一支rebound_ranking.py裡做 那裡本來就要抓歷史資料做grid search 順便算波動率 不用重複抓兩次
# 這是一次性的重運算 篩選完的結果會存成CSV 供下一支grid search腳本使用

import pandas as pd
import requests
import time
import io
import yfinance as yf
from ib_insync import IB, Stock, util
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------- 參數設定 ----------------
# 上市年限門檻 單位是年 這裡設定15年
# 舉例 如果某支ETF的歷史資料最早只到2015年 而今天是2026年 代表上市11年 不會通過
LISTING_YEARS_THRESHOLD = 15

# 輸出檔案路徑 這裡先存在本地目錄 之後Foster可以自行搬移到指定資料夾
OUTPUT_PATH = "universe.csv"

# 每抓幾支就印一次進度 避免長時間跑程式時看不到進度
PROGRESS_INTERVAL = 50

# 每次yfinance查詢之間的間隔秒數 避免被伺服器限制流量
REQUEST_DELAY = 0.5

# 多執行緒的執行緒數量 同時間會有這麼多支ETF的查詢在背景平行進行
MAX_WORKERS = 6

# IBKR連線設定 用來做最後一步驗證 確認ETF真的能在IBKR交易
# 7497是TWS紙上交易帳戶預設的埠號 如果之後改用正式帳戶 埠號要改成7496
# clientId要挑一個沒有被其他程式占用的數字 避免連線衝突
IBKR_HOST = "127.0.0.1"
IBKR_PORT = 7497
IBKR_CLIENT_ID = 99


# ---------------- 第一步 抓取美股ETF清單 ----------------
def fetch_us_etf_symbols():
    """
    從nasdaqtrader.com抓取官方上市清單
    這兩個檔案裡本來就有一欄ETF標記(Y代表是ETF N代表是個股) 直接用這欄篩選就好
    不用再用代碼格式或名稱去猜測是不是ETF
    回傳一個只包含ETF代碼的list
    """
    symbols = []

    sources = [
        "http://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "http://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
    ]

    for url in sources:
        resp = requests.get(url, timeout=30)
        lines = resp.text.split("\n")
        # 第一行是欄位標題 最後兩行是檔案資訊 都要排除
        header = lines[0].split("|")

        # 找出Symbol欄和ETF欄各自的位置 兩個檔案的欄位順序不太一樣 用欄位名稱去對應比較保險
        symbol_idx = None
        etf_idx = None
        for idx, col_name in enumerate(header):
            col_name_clean = col_name.strip()
            if col_name_clean in ("Symbol", "ACT Symbol"):
                symbol_idx = idx
            if col_name_clean == "ETF":
                etf_idx = idx

        if symbol_idx is None or etf_idx is None:
            print(f"找不到Symbol或ETF欄位 請確認{url}的檔案格式是否改變")
            continue

        for line in lines[1:-2]:
            parts = line.split("|")
            if len(parts) <= max(symbol_idx, etf_idx):
                continue

            symbol = parts[symbol_idx]
            is_etf = parts[etf_idx].strip()

            # 只留下ETF欄位標記是Y的 並且代碼不能包含點號或錢字號 這些通常是特別股或權證
            if is_etf == "Y" and symbol and "." not in symbol and "$" not in symbol:
                symbols.append(symbol)

    symbols = list(set(symbols))
    print(f"美股(NASDAQ加NYSE等交易所)共抓到 {len(symbols)} 支ETF代碼")
    return symbols


# ---------------- 第二步 抓取加股ETF清單 ----------------
def fetch_ca_etf_symbols():
    """
    從eoddata.com抓取TSX和TSXV的完整上市清單
    這份清單沒有像美股那樣明確的ETF標記欄位 這裡改用名稱裡有沒有出現ETF字樣來判斷
    這個方式不是百分之百精確 但加拿大的ETF幾乎都會在名稱裡明確標示ETF 覆蓋率應該很高
    加拿大代碼在yfinance裡 TSX要加.TO後綴 TSXV要加.V後綴
    """
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    exchange_suffix = {"TSX": ".TO", "TSXV": ".V"}

    symbols = []

    for exchange, suffix in exchange_suffix.items():
        for letter in letters:
            if letter == "A":
                url = f"https://www.eoddata.com/stocklist/{exchange}.htm"
            else:
                url = f"https://www.eoddata.com/stocklist/{exchange}/{letter}.htm"

            try:
                resp = requests.get(url, headers=headers, timeout=30)
                resp.raise_for_status()
                tables = pd.read_html(io.StringIO(resp.text))
            except Exception:
                continue

            target_table = None
            for table in tables:
                if "Code" in table.columns and "Name" in table.columns:
                    target_table = table
                    break

            if target_table is None:
                continue

            for _, row in target_table.iterrows():
                code = str(row["Code"])
                name = str(row["Name"])

                # 排除特別股 債券 認股權證
                if ".DB" in code or ".PR" in code or ".WT" in code or ".RT" in code:
                    continue

                # 名稱裡有ETF字樣才留下 舉例 "Global X Silver Covered Call Hgd ETF"符合 "Air Canada"不符合
                if "ETF" not in name.upper():
                    continue

                symbols.append(f"{code}{suffix}")

            time.sleep(0.2)

    symbols = list(set(symbols))
    print(f"TSX加TSXV共抓到 {len(symbols)} 支ETF代碼")
    return symbols


# ---------------- 第三步 計算上市年限並篩選 ----------------
def _fetch_one_listing_age(symbol, max_retries=2):
    """
    對單一ETF抓取歷史資料 計算上市年限
    這個函式會被丟到多個執行緒裡平行執行 失敗會重試
    只保留上市滿LISTING_YEARS_THRESHOLD年的ETF
    """
    hist = None
    for attempt in range(max_retries + 1):
        try:
            hist = yf.Ticker(symbol).history(period="max", auto_adjust=True)
            break
        except Exception:
            if attempt < max_retries:
                time.sleep(1 + attempt)
                continue
            return None

    time.sleep(REQUEST_DELAY)

    if hist is None or hist.empty:
        return None

    try:
        first_date = hist.index[0]
        # 計算從最早資料到今天經過幾年 舉例 first_date是2008-03-01 今天是2026-07-13 代表上市約18.4年
        listing_years = (pd.Timestamp.now(tz=first_date.tz) - first_date).days / 365.25

        if listing_years < LISTING_YEARS_THRESHOLD:
            return None

        return {
            "symbol": symbol,
            "listing_years": round(listing_years, 1)
        }
    except Exception:
        return None


def screen_listing_age(symbols):
    """
    對所有ETF用多執行緒平行抓取歷史股價 只做上市年限這一個篩選
    波動率的篩選留到rebound_ranking.py 那裡本來就要重新抓一次10年資料做grid search
    回傳通過篩選的DataFrame 欄位包含 symbol listing_years
    """
    results = []
    total = len(symbols)
    completed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_fetch_one_listing_age, s): s for s in symbols}

        for future in as_completed(futures):
            completed += 1
            result = future.result()

            if result is not None:
                results.append(result)

            if completed % PROGRESS_INTERVAL == 0:
                print(f"上市年限篩選進度 {completed}/{total}")

    df = pd.DataFrame(results)
    print(f"上市年限篩選後剩下 {len(df)} 支ETF")
    return df


# ---------------- 第四步 用IBKR驗證ETF是否真的能交易 ----------------
def verify_with_ibkr(df):
    """
    這是最後一道篩選 前面用的是交易所官方清單 只代表這支ETF有在交易所掛牌
    不代表IBKR一定開放交易 這裡實際連上IBKR 對每支ETF查詢合約細節
    查不到合約細節的會被排除 確保最後留下的都是真的能在IBKR下單的ETF
    執行這一步之前 電腦上的TWS或IB Gateway要先開著 並且已經允許API連線
    """
    try:
        util.startLoop()

        ib = IB()
        ib.connect(IBKR_HOST, IBKR_PORT, clientId=IBKR_CLIENT_ID)
    except Exception as e:
        print("=== 無法連上IBKR 請確認TWS或IB Gateway已經開啟並允許API連線 ===")
        print(f"錯誤訊息 {e}")
        print("這裡直接回傳原本的df 沒有做IBKR驗證")
        return df

    verified_rows = []
    total = len(df)

    for i, row in df.iterrows():
        symbol = row["symbol"]

        # 加拿大ETF代碼要去掉.TO或.V後綴 並且用IBKR認得的合約格式表示
        if symbol.endswith(".TO"):
            ib_symbol = symbol[:-3]
            contract = Stock(ib_symbol, "SMART", "CAD", primaryExchange="TSE")
        elif symbol.endswith(".V"):
            ib_symbol = symbol[:-2]
            contract = Stock(ib_symbol, "SMART", "CAD", primaryExchange="VENTURE")
        else:
            contract = Stock(symbol, "SMART", "USD")

        try:
            details = ib.reqContractDetails(contract)
        except Exception:
            details = []

        if details:
            verified_rows.append(row)

        if i % PROGRESS_INTERVAL == 0:
            print(f"IBKR驗證進度 {i}/{total}")

        time.sleep(0.1)

    ib.disconnect()

    verified_df = pd.DataFrame(verified_rows)
    print(f"IBKR驗證後 剩下 {len(verified_df)} 支ETF真的能在IBKR交易")
    return verified_df


# ---------------- 主程式 ----------------
def main():
    print("=== 開始建立ETF候選母體 ===")

    us_symbols = fetch_us_etf_symbols()
    ca_symbols = fetch_ca_etf_symbols()
    all_symbols = us_symbols + ca_symbols
    print(f"合併後共 {len(all_symbols)} 支ETF代碼 準備篩選上市年限")

    age_df = screen_listing_age(all_symbols)
    age_df = age_df.reset_index(drop=True)

    final_df = verify_with_ibkr(age_df)
    final_df = final_df.reset_index(drop=True)

    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"=== 完成 最終ETF母體共 {len(final_df)} 支 已存到 {OUTPUT_PATH} ===")


if __name__ == "__main__":
    main()
