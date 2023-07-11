## Draw the graph of the timeseries Data
import pandas as pd
import matplotlib.pyplot as plt

# https://matplotlib.org/stable/api/dates_api.html
import matplotlib.dates as mdates

# https://dateutil.readthedocs.io/en/stable/tz.html#module-dateutil.tz
from dateutil import tz
from datetime import datetime

# Define the CVE number
cve = 'CVE-2023-23397'

# read csv file
df = pd.read_csv('labeled_data/51420_labeled_data_rounded_with_metrics_20230326.csv', index_col =0, parse_dates = True)
df.drop(['num', 'created_at',
       'edit_history_tweet_ids', 'public_metrics', 'baseSeverity',
       'baseScore', 'flag_CR', 'flag_H', 'public_metrics_dict'], inplace = True, axis = 1)

# count tweets for each CVE
df_counted = pd.DataFrame(df.groupby(['rd_created_at','CVE'])['CVE'].count())
df_imp_summed = pd.DataFrame(df.groupby(['rd_created_at','CVE'])['impression_count'].sum())

df_imp_cve = df_imp_summed.loc[pd.IndexSlice[:, cve], :]
df_imp_cve = df_imp_cve.reset_index().set_index('rd_created_at')

# df_counted.rename(columns = {'CVE':'tweets'}, inplace = True)

# count tweets for all CVEs
# df_all_counted = pd.DataFrame(df.groupby(['rd_created_at']).count())
# df_all_counted.rename(columns={'CVE':'tweets'}, inplace = True)


jp = tz.gettz('Japan/Tokyo')

# Matplotlib 時系列データの軸設定
# https://www.yutaka-note.com/entry/matplotlib_time_axis

fig, ax = plt.subplots()
plt.xticks(rotation=30)
ax.plot(df_imp_cve.index, df_imp_cve['impression_count'])
ax.set_title(f'{cve}')
ax.set_xlabel('Date and Time')
ax.set_ylabel('impression_count')
ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=([7,14,21,28]), interval=1, tz=jp))
# ax.xaxis.set_major_locator(mdates.AutoDateLocator(tz=jp))
ax.xaxis.set_major_locator(mdates.AutoDateLocator(tz=jp))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%y/%m/%d\n%H:%M"))
fig.subplots_adjust(bottom=0.2)
plt.savefig(f'pngfiles/Impression_count_timeseries_of_{cve}.png')
plt.show()