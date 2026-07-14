import numpy as np
import pandas as pd
import yfinance as yf
from itertools import product

START = '2006-01-01'
END = '2026-07-14'
SYMBOL = 'CQQQ'

raw = yf.download(SYMBOL, start=START, end=END, auto_adjust=True)
data = raw[['Close', 'High', 'Open']].copy()
data['return'] = data['Open']/data['Close'].shift(1)
data.dropna(inplace=True)

#================ Get Rid of Extreme Cases ============================
return_mean = data['return'].mean()
return_std = data['return'].std()
z_score = (data['return'] - return_mean) / return_std

mask = z_score.abs() <= 4
good_data = data[mask].copy()

good_return_mean = good_data['return'].mean()
good_return_std = good_data['return'].std()
good_z_score = (good_data['return'] - good_return_mean) / good_return_std

#================ Print Out the Result =================================
days = list(range(1, 4, 1))
thresholds = np.arange(1, 4, 0.5).tolist()

informations = []
for day, threshold in product(days, thresholds):

    # 找出 z_score 落在 (-(threshold+1), -threshold] 區間的日期
    condition = (good_z_score > -(threshold + 1)) & (good_z_score <= -threshold)
    target_dates = good_data[condition].index

    # 取出各 target_dates 往後推 day 天的 return
    shifted_return = good_data['return'].shift(-day)
    rebound_ratio = (shifted_return.loc[target_dates] / good_data['return'].loc[target_dates]).mean()

    information = {'rebound_ratios': rebound_ratio,
                   'days': day,
                   'thresholds': threshold}
    informations.append(information)

# 轉成 DataFrame 才能用欄位篩選與比較
df_info = pd.DataFrame(informations)

print('=== Best Rebound Ratio ===')
best_rebound_ratio = df_info['rebound_ratios'].max()
print(f'{best_rebound_ratio:.2f}')

best_days = df_info.loc[df_info['rebound_ratios'] == best_rebound_ratio, 'days'].tolist()
best_threshold = df_info.loc[df_info['rebound_ratios'] == best_rebound_ratio, 'thresholds'].tolist()

# 印出負向區間，與 condition 的定義一致
lower_bound = [-(t + 1) for t in best_threshold]
upper_bound = [-t for t in best_threshold]
print(f'{best_days} days, with {lower_bound} < z_score <= {upper_bound}')