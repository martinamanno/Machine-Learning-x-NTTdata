"""
Machine Learning techniques to increase profitability. 
NTTData x Luiss
Martina Manno, Martina Crisafulli, Olimpia Sannucci, Hanna Carucci Viterbi, Tomas Ryen
"""

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import seaborn as sns
from collections import Counter
import os
import plotly.express as px
from functools import reduce
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import squarify

# Load the dataset, corresponding to the final of the clustering script
df = pd.read_csv("python_merged.csv", index_col=0)
df.dropna(inplace=True)

# Data cleaning and management
products_bought = df.groupby(['customer_unique_id']).agg({'order_item_sequence_id':'sum',
                                                          'product_category_name':'count',
                                                          'price':'sum','transaction_value':'sum',
                                                          'shipping_cost':'sum', 'review_score':'mean', 
                                                          'ts_order_purchase':'max', 'order_id':'count'}).reset_index()
all_data = products_bought.sort_values(by = ['product_category_name'], ascending=False)

# Data cleaning and addition of recency and frequency to the dataset 
all_data = all_data.rename(columns={"product_category_name": "amount_prod_categories"})
all_data['ts_order_purchase'] = pd.to_datetime(all_data['ts_order_purchase'])
all_data['today'] = all_data['ts_order_purchase'].max()
all_data['recency'] = all_data['today'] - all_data['ts_order_purchase']
all_data['recency'] = all_data['recency'].dt.days
all_data.drop(['ts_order_purchase','today'], inplace= True, axis=1)
all_data = all_data.rename(columns={"order_id": "frequency"})

# Plotting Recency
plt.figure(figsize=(18, 4))
sns.distplot(all_data['recency'])

plt.xlabel('Days since the last order')
plt.title('Recency',fontsize=16);
plt.show()

# Plotting Frequency
plt.figure(figsize=(15, 3))
sns.distplot(all_data['frequency'], kde=False);

plt.xlabel('Number of orders')
plt.title('Frequency',fontsize=16);
plt.show()
print(all_data['frequency'].value_counts())

# Plotting Monetary
plt.figure(figsize=(15, 3))
sns.distplot(all_data['transaction_value']);

plt.xlabel('Total purchase value')
plt.title('Monetary', fontsize=16);
plt.show()

# Compute ratings for each product category
rating_count = (df.groupby(by = ['product_category_name'])['review_score'].mean().reset_index().rename(columns = {'review_score':'review_count'})[['product_category_name','review_count']])

# sort products by review, from highest to lowest
product_rew = rating_count.sort_values(by=['review_count'], ascending=False)
best_products = product_rew[:10]
best_products = best_products[['product_category_name','review_count']]
best_products[:10]

# Plotting the best categories for review score
prodnote_hist = sns.barplot(y = best_products["product_category_name"], x = best_products["review_count"], palette = "BrBG_r");
sns.set_context("talk")
sns.set_style("whitegrid")
sns.set(rc = {'figure.figsize':(10,8)})
plt.title('Best 10 product categories for review score', fontsize = 16)
plt.xlabel('Product category avarage review score', fontsize = 14)
plt.ylabel('Product category', fontsize = 14)
plt.savefig('best10productreviw.png')

# ### RFM ANALYSIS

# Select data for RFM analysis
df_RFM = all_data[['customer_unique_id', 'frequency', 'recency', 'transaction_value']]
df_RFM = df_RFM.reset_index(drop=True)
df_RFM.head()

# computation of quantiles
quintiles = df_RFM[['recency', 'frequency', 'transaction_value']].quantile([.2, .4, .6, .8]).to_dict()

# Defining functions for R score and FM score
def r_score(x):
    if x <= quintiles['recency'][.2]:
        return 5
    elif x <= quintiles['recency'][.4]:
        return 4
    elif x <= quintiles['recency'][.6]:
        return 3
    elif x <= quintiles['recency'][.8]:
        return 2
    else:
        return 1
    
def fm_score(x, c):
    if x <= quintiles[c][.2]:
        return 1
    elif x <= quintiles[c][.4]:
        return 2
    elif x <= quintiles[c][.6]:
        return 3
    elif x <= quintiles[c][.8]:
        return 4
    else:
        return 5   

# Computation of the R, F and M values
df_RFM['R'] = df_RFM['recency'].apply(lambda x: r_score(x))
df_RFM['F'] = df_RFM['frequency'].apply(lambda x: fm_score(x, 'frequency'))
df_RFM['M'] = df_RFM['transaction_value'].apply(lambda x: fm_score(x, 'transaction_value'))
df_RFM['RFM Score'] = df_RFM['R'].map(str) + df_RFM['F'].map(str) + df_RFM['M'].map(str)

# Segments
# One time buyers: low recency, low frequency 
# About to Sleep: low recency, high frequency: the ones that made some purchases but are not active anymore
# Potential Loyalist: high recency, low frequency: new customers who recently made a purchase and might become loyal customers
# Loyal Customers: medium recency, high frequency: we eant them to become Champions
# Champions: high recency, high frequency; higest score 
# Strangers: medium recency, low frequency

# Build a segmentation map
segm1_map = {
    r'[1-2]1': 'One Time',
    r'[1-2][4-5]': 'About to Sleep',
    r'5[1-2]': 'Potential Loyalist',
    r'[3-4]5': 'Loyal Customers',
    r'[4-5]5': 'Champions',
    r'[3-4][1-3]': 'Strangers'
}

df_RFM['Segment1'] = df_RFM['R'].map(str) + df_RFM['F'].map(str)
df_RFM['Segment1'] = df_RFM['Segment1'].replace(segm1_map, regex=True)
all_data['R'] = all_data['recency'].apply(lambda x: r_score(x))
all_data['F'] = all_data['frequency'].apply(lambda x: fm_score(x, 'frequency'))
all_data['M'] = all_data['transaction_value'].apply(lambda x: fm_score(x, 'transaction_value'))
all_data['RFM Score'] = all_data['R'].map(str) + all_data['F'].map(str) + all_data['M'].map(str)
all_data['Segment1'] = all_data['R'].map(str) + all_data['F'].map(str)
all_data['Segment1'] = all_data['Segment1'].replace(segm1_map, regex=True)

# All spenders
# Segments classified according to how much they spend
segment1s_counts = df_RFM['Segment1'].value_counts().sort_values(ascending=True)

fig, ax = plt.subplots()

bars = ax.barh(range(len(segment1s_counts)),
              segment1s_counts,
              color='cornflowerblue')
ax.set_frame_on(False)
ax.tick_params(left=False,
               bottom=False,
               labelbottom=False)
ax.set_yticks(range(len(segment1s_counts)))
ax.set_yticklabels(segment1s_counts.index,  fontsize = 14)

for i, bar in enumerate(bars):
        value = bar.get_width()
        ax.text(value,
                bar.get_y() + bar.get_height()/2,
                ' {:,} ({:}%)'.format(int(value),
                                   int(value*100/segment1s_counts.sum())),
                va='center',
                ha='left'
               )
sns.set(rc={'figure.figsize':(5,10)})
plt.savefig('rfm_segments.png')


# Low spenders

# Plot for low-spending people
low_spenders = df_RFM[df_RFM['M'] <= 2]
low_spenders['Segment_big'] = low_spenders['R'].map(str) + low_spenders['F'].map(str)
low_spenders['Segment_big'] = low_spenders['Segment_big'].replace(segm1_map, regex=True)

segment1s_counts = low_spenders['Segment_big'].value_counts().sort_values(ascending=True)

fig, ax = plt.subplots()

bars = ax.barh(range(len(segment1s_counts)),
              segment1s_counts,
              color='b')
ax.set_frame_on(False)
ax.tick_params(left=False,
               bottom=False,
               labelbottom=False)
ax.set_yticks(range(len(segment1s_counts)))
ax.set_yticklabels(segment1s_counts.index,  fontsize = 14)

for i, bar in enumerate(bars):
        value = bar.get_width()
        ax.text(value,
                bar.get_y() + bar.get_height()/2,
                ' {:,} ({:}%)'.format(int(value),
                                   int(value*100/segment1s_counts.sum())),
                va='center',
                ha='left'
               )
sns.set(rc={'figure.figsize':(5,10)})
plt.show()


# Average spenders
# Plot for average-spending people

avg_spenders = df_RFM[df_RFM['M'] == 3]
avg_spenders['Segment_big'] = avg_spenders['R'].map(str) + avg_spenders['F'].map(str)
avg_spenders['Segment_big'] = avg_spenders['Segment_big'].replace(segm1_map, regex=True)

segment1s_counts = avg_spenders['Segment_big'].value_counts().sort_values(ascending=True)

fig, ax = plt.subplots()

bars = ax.barh(range(len(segment1s_counts)),
              segment1s_counts,
              color='g')
ax.set_frame_on(False)
ax.tick_params(left=False,
               bottom=False,
               labelbottom=False)
ax.set_yticks(range(len(segment1s_counts)))
ax.set_yticklabels(segment1s_counts.index,  fontsize = 14)

for i, bar in enumerate(bars):
        value = bar.get_width()
        ax.text(value,
                bar.get_y() + bar.get_height()/2,
                ' {:,} ({:}%)'.format(int(value),
                                   int(value*100/segment1s_counts.sum())),
                va='center',
                ha='left'
               )
sns.set(rc = {'figure.figsize':(5,10)})
plt.show()


# High spenders

# Plot for high-spending people
high_spenders = df_RFM[df_RFM['M'] >= 4]
print(len(high_spenders))

high_spenders['Segment_big'] = high_spenders['R'].map(str) + high_spenders['F'].map(str)
high_spenders['Segment_big'] = high_spenders['Segment_big'].replace(segm1_map, regex=True)

segment1s_counts = high_spenders['Segment_big'].value_counts().sort_values(ascending=True)

fig, ax = plt.subplots()

bars = ax.barh(range(len(segment1s_counts)),
              segment1s_counts,
              color = 'r')
ax.set_frame_on(False)
ax.tick_params(left = False,
               bottom = False,
               labelbottom = False)
ax.set_yticks(range(len(segment1s_counts)))
ax.set_yticklabels(segment1s_counts.index,  fontsize = 14)

for i, bar in enumerate(bars):
        value = bar.get_width()
        ax.text(value,
                bar.get_y() + bar.get_height()/2,
                ' {:,} ({:}%)'.format(int(value),
                                   int(value*100/segment1s_counts.sum())),
                va = 'center',
                ha = 'left'
               )
sns.set(rc = {'figure.figsize':(5,10)})
plt.show()


# Values for each Segment

# Group customers by segments
df_rfm2 = df_RFM.groupby('Segment1').agg(RecencyMean = ('recency', 'mean'),
                                          FrequencyMean = ('frequency', 'mean'),
                                          MonetaryMean = ('transaction_value', 'mean'),
                                          GroupSize = ('recency', 'size'))

# Plot of segments size
font = {'family' : 'Dejavu Sans',
        'weight' : 'normal',
        'size'   : 18}

plt.rc('font', **font)


fig = plt.gcf()
ax = fig.add_subplot()
fig.set_size_inches(15, 10)
squarify.plot(sizes = df_rfm2['GroupSize'], 
              label = df_rfm2.index,
              color = ['blue','firebrick', 'tomato', 'teal', 'olive', 'gold'],
              alpha = 0.5)
plt.title("RFM Segments",fontsize=18,fontweight="bold")
plt.axis('off')
plt.savefig('rfm_square.png')

# Best 10 products for review score
prodnote_hist = sns.barplot( y=best_products["product_category_name"], x=best_products["review_count"],palette = "BuPu_r");
sns.set_context("talk")
sns.set_style("whitegrid")
sns.set(rc = {'figure.figsize':(10,8)})
plt.title('Best 10 product categories for review score', fontsize = 16)
plt.xlabel('Product category avarage review score', fontsize = 14)
plt.ylabel('Product category', fontsize = 14)
plt.savefig('best10product per review.png')

# 10 mosr ordered product categories
prodcat = df.groupby(['product_category_name']).sum().reset_index() 
prodcat = prodcat[prodcat['product_category_name'] != 'None']
prodcat = prodcat.sort_values(by=['order_item_sequence_id'], ascending=False)
prodcat0 = prodcat[:10]
sns.set_context("talk")
sns.set_style("white")
sns.set(rc={'figure.figsize':(10,8)})
catorderhist = sns.barplot( y=prodcat0["product_category_name"], x=prodcat0["order_item_sequence_id"] ,palette = 'YlGn_r');
plt.title('10 most ordered product categories', fontsize = 16)
plt.xlabel('Product category', fontsize = 14)
plt.ylabel('Ammount of orders per category', fontsize = 14)
plt.savefig('10 most ordered product categories.png' )

# Number of customers by region
plt.figure(figsize = (10,5))
plt.title('Customers Per Region')
plt.ylabel('Regions')
plt.xlabel('Number of customers')
sns.barplot(y = df['customer_autonomous_community'].value_counts().index,
            x = df['customer_autonomous_community'].value_counts().values, palette = 'cool_r')
plt.xticks(rotation = 90)
plt.savefig('customers per region.png')


# Time Data Managing

# We will work only with the orders with status == delivered
data__orders=df.loc[df['order_status'] == 'delivered']
data__orders['ts_order_purchase'] = pd.to_datetime(data__orders['ts_order_purchase'])
data__orders['ts_order_estimated_delivery'] = pd.to_datetime(data__orders['ts_order_estimated_delivery'])
data__orders['ts_order_delivered_customer'] = pd.to_datetime(data__orders['ts_order_delivered_customer'])
data__orders['dif_exp_delivery'] = ((data__orders['ts_order_estimated_delivery'])-(data__orders['ts_order_delivered_customer'])).dt.days
data__orders['dif_exp_delivery'].fillna(0,inplace=True)
data__orders.dropna(inplace = True)
data__orders['dif_exp_delivery'] = data__orders['dif_exp_delivery'].astype('int64')

# New Columns to add to the dataset
data__orders['year_purch'] = data__orders['ts_order_purchase'].dt.year
data__orders['month_purch'] = data__orders['ts_order_purchase'].dt.month
data__orders['day_purch'] = data__orders['ts_order_purchase'].dt.day
data__orders['week_day'] = data__orders['ts_order_purchase'].dt.weekday

# Assign a number for each day of the week
day_name = []
for d in data__orders.week_day:
    if d == 6:
        d = 'Sun'
    elif d == 0:
        d = 'Mon'
    elif d == 1:
        d = 'Tue'
    elif d == 2:
        d = 'Wed'
    elif d == 3:
        d = 'Thur'
    elif d == 4:
        d = 'Fri'
    else:
        d = 'Sat'
    day_name.append(d)
data__orders['week_day'] = day_name

# Convert to datetime and compute difference between estimate and actual delivery
data__orders['ts_order_purchase'] = pd.to_datetime(data__orders['ts_order_purchase'])
data__orders['ts_order_estimated_delivery'] = pd.to_datetime(data__orders['ts_order_estimated_delivery'])
data__orders['ts_order_delivered_customer'] = pd.to_datetime(data__orders['ts_order_delivered_customer'])
data__orders['dif_exp_delivery']=((data__orders['ts_order_estimated_delivery'])-(data__orders['ts_order_delivered_customer'])).dt.days

# Time data modifications
data__orders['ts_order_purchase'] = pd.to_datetime(data__orders['ts_order_purchase'])
data__orders['ts_order_purchase'] = data__orders['ts_order_purchase'].dt.date
data__orders['hour_purch'] = data__orders['ts_order_purchase'].dt.round('360min')
data__orders['hour_purch'] = pd.to_datetime(data__orders['hour_purch'])
data__orders['hour_purch'] = data__orders['hour_purch'].dt.time
data__orders['hour_purch'] = data__orders['ts_order_purchase'].dt.round('360min')
data__orders['month_year_purch'] = data__orders['ts_order_purchase'].dt.strftime('%m-%Y')
data__orders.drop(['ts_order_purchase'],axis = 1, inplace=True)

#Drop some data we will not use
"""data__orders.drop(['ts_order_approved'],axis=1,inplace=True)
data__orders.drop(['ts_order_delivered_carrier'],axis=1,inplace=True)
data__orders.drop(['ts_order_delivered_customer'],axis=1,inplace=True)
data__orders.drop(['ts_order_estimated_delivery'],axis=1,inplace=True)
df.drop(['product_name_lenght'],axis=1,inplace=True)
df.drop(['product_photos_quantity'],axis=1,inplace=True)
df.drop(['product_description_lenght'],axis=1,inplace=True)
df.drop(['product_weight_gr'],axis=1,inplace=True)
df.drop(['product_length_cm'],axis=1,inplace=True)
df.drop(['product_height_cm'],axis=1,inplace=True)
df.drop(['product_width_cm'],axis=1,inplace=True)"""

# Data without the dropped columns
filtro  = data__orders['month_purch'].isin([1,2,3,4,5,6,7,8])
data_aux= data__orders[filtro]
data__orders.to_csv('data__orders.csv')

# Distribution of purchases per year
plt.figure(figsize=(20,7))
plt.subplot(111)
year_seasonality = data__orders.groupby(['year_purch'])['order_id'].nunique().sort_values(ascending=True).reset_index()
total=len(data__orders)
g = sns.barplot(x='year_purch', y='order_id', data=year_seasonality, palette='Set2')
g.set_title("Distribution of purchases per year", fontsize=20)
g.set_xlabel("Year", fontsize=17)
g.set_ylabel("Purchases quantity", fontsize=17)
sizes = []
for p in g.patches:
    height = p.get_height()
    sizes.append(height)
    g.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/total*100),
            ha="center", fontsize=14) 
g.set_ylim(0, max(sizes) * 1.1)
plt.savefig('distribution purchases per year.png')

# Distribution of purchases during the week
plt.figure(figsize=(15,7))
weekle_seasonality = data__orders.groupby(['week_day'])['order_id'].nunique().sort_values(ascending=False).reset_index()
sns.barplot(x='week_day',y='order_id', data=weekle_seasonality, palette = 'Set3')
plt.title("Distirbution of purchases during the week", fontsize=20)
plt.xlabel('day of the week')
plt.ylabel('Order quantity')
plt.savefig('distr purchases during week.png')
