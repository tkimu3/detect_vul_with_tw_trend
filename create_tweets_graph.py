## Draw the graph of the timeseries Data
import pandas as pd
import matplotlib.pyplot as plt

# https://matplotlib.org/stable/api/dates_api.html
import matplotlib.dates as mdates

# https://dateutil.readthedocs.io/en/stable/tz.html#module-dateutil.tz
from dateutil import tz
from datetime import datetime

df = pd.read_csv('labeled_data/51420_labeled_data_rounded_20230326.csv', index_col =1, parse_dates = True)
df.drop(columns = ['Unnamed: 0'], inplace = True)

# count tweets for each CVE
# df_counted = pd.DataFrame(df.groupby(['rd_created_at','CVE'])['CVE'].count())
# df_counted.rename(columns = {'CVE':'tweets'}, inplace = True)

# count tweets for all CVEs
df_all_counted = pd.DataFrame(df.groupby(['rd_created_at']).count())
df_all_counted.rename(columns={'CVE':'tweets'}, inplace = True)


jp = tz.gettz('Japan/Tokyo')

# Matplotlib 時系列データの軸設定
# https://www.yutaka-note.com/entry/matplotlib_time_axis

fig, ax = plt.subplots()
plt.xticks(rotation=30)
ax.plot(df_all_counted.index, df_all_counted['tweets'])
ax.set_title('All tweets from Feb 7 to Mar 26')
ax.set_xlabel('Date and Time')
ax.set_ylabel('tweets')
ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=([7,14,21,28]), interval=1, tz=jp))
# ax.xaxis.set_major_locator(mdates.AutoDateLocator(tz=jp))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%y/%m/%d\n%H:%M"))
fig.subplots_adjust(bottom=0.2)
plt.savefig('pngfiles/全ツイートの時系列推移データ.png')
plt.show()