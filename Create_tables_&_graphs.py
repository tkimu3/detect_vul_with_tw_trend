#!/usr/bin/env python
# coding: utf-8

# ## Create a table of the number of CVSS Scores

# In[2]:


# The table of all the 1677 CVEs (Over 10 authors & CVSS already released)
# Label (Critical, High, Medium, Low) of CVEs,
# and the number of CVEs for each label


# In[4]:


import pandas as pd


# In[9]:


df = pd.read_csv("CVEs_Over10authors_20230326_except_NonScores_with_flags.csv", index_col = 0)
df


# In[10]:


df.columns


# In[8]:


df1 = df.drop(columns = ['Unnamed: 0', 'num_of_author', 'baseScore', 'flag_CR',
       'flag_H'])
df1


# In[12]:


df1.groupby(['baseSeverity'])['CVE'].count()


# ## Draw the graph of the timeseries Data

# In[14]:


import matplotlib.pyplot as plt


# In[16]:


df2 = pd.read_csv('51420_labeled_data_rounded_20230326.csv', index_col =1, parse_dates = True)
df2


# In[17]:


df2.index


# In[19]:


df_test = df2


# In[21]:


df_test


# In[22]:


df_test.drop(columns = ['Unnamed: 0'], inplace = True)
df_test


# In[24]:


df_counted = pd.DataFrame(df_test.groupby(['rd_created_at','CVE'])['CVE'].count())
df_counted


# In[25]:


df_counted.shape


# In[27]:


df_counted.rename(columns = {'CVE':'tweets'}, inplace = True)
df_counted


# In[33]:


df_counted.index.dtypes


# In[48]:


print(df_counted.loc[pd.IndexSlice[:,'CVE-2021-27365'], :])


# In[41]:


df_all_counted = pd.DataFrame(df_test.groupby(['rd_created_at']).count())
df_all_counted


# In[42]:


df_all_counted.rename(columns={'CVE':'tweets'}, inplace = True)
df_all_counted


# In[80]:


# https://matplotlib.org/stable/api/dates_api.html
import matplotlib.dates as mdates


# In[81]:


# https://dateutil.readthedocs.io/en/stable/tz.html#module-dateutil.tz
from dateutil import tz
from datetime import datetime


# In[82]:


jp = tz.gettz('Japan/Tokyo')
jp


# In[ ]:


# Matplotlib 時系列データの軸設定
# https://www.yutaka-note.com/entry/matplotlib_time_axis


# In[102]:


fig, ax = plt.subplots()
plt.xticks(rotation=30)
ax.plot(df_all_counted.index, df_all_counted['tweets'])
ax.set_title('All tweets')
ax.set_xlabel('Date and Time')
ax.set_ylabel('tweets')
ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=([7,14,21,28]), interval=1, tz=jp))
# ax.xaxis.set_major_locator(mdates.AutoDateLocator(tz=jp))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%y/%m/%d\n%H:%M"))
plt.savefig('Tweets_timeseries.png')
plt.show()


# In[51]:


print( df_counted.loc[pd.IndexSlice[:,'CVE-2021-27365'], 'tweets'])


# In[ ]:





# In[53]:


df_27365 = df_counted.loc[pd.IndexSlice[:,'CVE-2021-27365'], :]
df_27365


# In[62]:


df_27365.reset_index().set_index('rd_created_at')


# In[63]:


df_27365 = df_27365.reset_index().set_index('rd_created_at')


# In[99]:


fig, ax = plt.subplots()
plt.xticks(rotation=90)
ax.plot(df_27365.index, df_27365['tweets'])
ax.set_xlabel('Date and Time')
ax.set_ylabel('tweets')
ax.set_title('CVE-2021-27365')
ax.xaxis.set_major_locator(mdates.AutoDateLocator(tz=jp))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%y/%m/%d\n%H:%M"))
plt.savefig('Tweets_timeseries_of_CVE-2021-27365.png')
plt.show()


# In[66]:


df_23397 = df_counted.loc[pd.IndexSlice[:,'CVE-2023-23397'], :]
df_23397


# In[67]:


df_23397 = df_23397.reset_index().set_index('rd_created_at')
df_23397


# In[101]:


fig, ax = plt.subplots()
plt.xticks(rotation=30)
ax.plot(df_23397.index, df_23397['tweets'])
ax.set_xlabel('Date and Time')
ax.set_ylabel('tweets')
ax.set_title('CVE-2023-23397')
# ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=(list(range(14, 28,2))), interval=1, tz=jp))
ax.xaxis.set_major_locator(mdates.AutoDateLocator(tz=jp))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%y/%m/%d\n%H:%M"))
plt.savefig('Tweets_timeseries_of_CVE-2023-23397.png')
plt.show()


# In[103]:


df_39952 = df_counted.loc[pd.IndexSlice[:,'CVE-2022-39952'], :]
df_39952 = df_39952.reset_index().set_index('rd_created_at')
fig, ax = plt.subplots()
plt.xticks(rotation=30)
ax.plot(df_39952.index, df_39952['tweets'])
ax.set_xlabel('Date and Time')
ax.set_ylabel('tweets')
ax.set_title('CVE-2022-39952')
# ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=(list(range(14, 28,2))), interval=1, tz=jp))
ax.xaxis.set_major_locator(mdates.AutoDateLocator(tz=jp))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%y/%m/%d\n%H:%M"))
plt.savefig('Tweets_timeseries_of_CVE-2022-39952.png')
plt.show()


# In[104]:


df_21716 = df_counted.loc[pd.IndexSlice[:,'CVE-2023-21716'], :]
df_21716 = df_21716.reset_index().set_index('rd_created_at')
fig, ax = plt.subplots()
plt.xticks(rotation=30)
ax.plot(df_21716.index, df_21716['tweets'])
ax.set_xlabel('Date and Time')
ax.set_ylabel('tweets')
ax.set_title('CVE-2023-21716')
# ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=(list(range(14, 28,2))), interval=1, tz=jp))
ax.xaxis.set_major_locator(mdates.AutoDateLocator(tz=jp))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%y/%m/%d\n%H:%M"))
plt.savefig('Tweets_timeseries_of_CVE-2023-21716.png')
plt.show()

