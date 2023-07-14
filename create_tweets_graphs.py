## Draw the graph of the timeseries Data
import pandas as pd
import matplotlib.pyplot as plt

# https://matplotlib.org/stable/api/dates_api.html
import matplotlib.dates as mdates

# https://dateutil.readthedocs.io/en/stable/tz.html#module-dateutil.tz
from dateutil import tz
import datetime
from datetime import timedelta

# Define the CVE number
cve = 'CVE-2023-23397'

# read csv file
df = pd.read_csv('labeled_data/51420_labeled_data_rounded_with_metrics_20230326.csv', index_col =0, parse_dates = True)
all_columns_list = list(df.columns.values)
column_list = ['CVE', 'text', 'reply_count', 'like_count', 'quote_count', 'impression_count']
# print(all_columns_list)

## remove columns from the list except for  'CVE', 'retweet_count', 'reply_count', 'like_count', 'quote_count', 'impression_count'
# removed 'num', 'created_at', 'id', 'text', 'author_id','edit_history_tweet_ids', 'public_metrics', 'baseSeverity','baseScore', 'flag_CR', 'flag_H', 'public_metrics_dict'
# drop_columns_list = columns_list.remove(['CVE', 'retweet_count', 'reply_count', 'like_count', 'quote_count', 'impression_count'])
drop_columns_list = [ c for c in all_columns_list if not c in(column_list) ]
# print(drop_columns_list)
# columns_list.remove(['CVE', 'retweet_count', 'reply_count', 'like_count', 'quote_count', 'impression_count'])

# df.drop(['num', 'created_at',
#        'edit_history_tweet_ids', 'public_metrics', 'baseSeverity',
#        'baseScore', 'flag_CR', 'flag_H', 'public_metrics_dict'], inplace = True, axis = 1)
df.drop(drop_columns_list, inplace = True, axis = 1)
# print(df)

df_concat = pd.DataFrame()

## count tweets for each CVE
feature_list = ['tweet_count','retweet_count', 'reply_count', 'like_count', 'quote_count', 'impression_count']
set_list = zip(column_list, feature_list)
# print(list(set_list))

for (col, feature) in list(set_list):
    #  print(f"{col}&{feature}")
     # define the dataframe for the iteration
     df_tmp = df.copy()

     if (feature == 'retweet_count'):
        df_tmp = df_tmp[df_tmp['text'].str.startswith('RT')]

     # Define temporaly drop-columns list
     column_list_tmp = column_list
     column_list_tmp.remove(col)
     df_tmp = df_tmp.drop(column_list_tmp, axis = 1)

    # count the 'CVE' columns if the feature is tweet_count or retweet_count
     if(col == 'CVE' or col == 'text'):
      #  df[k] = pd.DataFrame(df.groupby(['rd_created_at','CVE'])['CVE'].count())
       df_tmp = pd.DataFrame(df_tmp.groupby(['rd_created_at','CVE'])[col].count())


    #  elif (feature == 'retweet_count'):
    #    df_tmp = df_tmp[df_tmp['text']]
    #    df_tmp = pd.DataFrame(df_tmp.groupby(['rd_created_at','CVE'])[col].count())

     else:
       df_tmp = pd.DataFrame(df_tmp.groupby(['rd_created_at','CVE'])[col].sum())

    #  df_tmp = pd.DataFrame(df_tmp.groupby(['rd_created_at','CVE'])[col].count())

     df_tmp.rename(columns = {col: feature}, inplace = True)

     ## drop CVEs except for the CVE
     df_tmp = df_tmp.loc[pd.IndexSlice[:, cve], :]

     ## set the timestamp as index
     df_tmp = df_tmp.reset_index().set_index('rd_created_at')

     df_tmp.drop(['CVE'], inplace = True, axis = 1)

     ## Define the start & end time (48hours) for the cve-id
     start_time = df_tmp.index[0]
     end_time = start_time + datetime.timedelta(hours = 48)

     ## Create a Dataframe within the 48_hours timeflame
     df_tmp_48hour = df_tmp.loc[start_time:end_time]

     df_concat = pd.concat([df_concat, df_tmp_48hour], axis = 1)
    #  print(df_concat.isnull().sum())
     df_concat.fillna(0)

    #  print(df_concat)


# # 01_tweet_count
# df_counted = pd.DataFrame(df.groupby(['rd_created_at','CVE'])['CVE'].count())
# df_counted.rename(columns = {'CVE':'01_tweet_count'}, inplace = True)

# #

# # 06_impression_count
# df_imp_summed = pd.DataFrame(df.groupby(['rd_created_at','CVE'])['impression_count'].sum())

# df_imp_cve = df_imp_summed.loc[pd.IndexSlice[:, cve], :]
# df_imp_cve = df_imp_cve.reset_index().set_index('rd_created_at')

# df_counted.rename(columns = {'CVE':'tweets'}, inplace = True)

# count tweets for all CVEs
# df_all_counted = pd.DataFrame(df.groupby(['rd_created_at']).count())
# df_all_counted.rename(columns={'CVE':'tweets'}, inplace = True)


jp = tz.gettz('Japan/Tokyo')

# # Matplotlib 時系列データの軸設定
# # https://www.yutaka-note.com/entry/matplotlib_time_axis

# nrows,ncolsの設定
nrows = int(len(feature_list)/2)
ncols = 2

fig, axes = plt.subplots(nrows, ncols, sharex="all", squeeze = False, tight_layout = True)
# fig, ax = plt.subplots()
# plt.xticks(rotation=30)
count = 0

for i in range(nrows):
  for j in range(ncols):
    axes[i,j].plot(df_concat.index, df_concat.iloc[:, count])
    axes[i,j].set_title(feature_list[count])
    for tick in axes[i, j].get_xticklabels():
      tick.set_rotation(30)
    # axes[i,j].set_xlabel('Date and Time')
    # axes[i,j].set_ylabel(feature[count])
    axes[i,j].xaxis.set_major_locator(mdates.DayLocator(bymonthday=([7,14,21,28]), interval=1, tz=jp))
    # axes.xaxis.set_major_locator(mdates.AutoDateLocator(tz=jp))
    axes[i,j].xaxis.set_major_locator(mdates.AutoDateLocator(tz=jp))
    axes[i,j].xaxis.set_major_formatter(mdates.DateFormatter("%m/%d%H:%M"))
    # axes[i,j].xaxis.set_major_formatter(mdates.DateFormatter("%y/%m/%d\n%H:%M"))
    count += 1
fig.subplots_adjust(bottom=0.2)
plt.savefig(f'pngfiles/timeseries_of_{cve}.png')
plt.show()