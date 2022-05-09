#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mode,mean,median
import collections


# In[2]:


# Importing the same dataset that we generated in the RFM analysis
data = pd.read_csv("data__mktingstrategy.csv",sep = ",")
data = data.drop(columns = ['Unnamed: 0'])


# In[3]:


# check how many observations we have for each year
data.groupby("year_purch").size()


# # Idea behind the analysis

# The marketing analysis will be carried out dividing the data in two years 2018 and 2019 to better detect customer trends and weak spot on which to work on. 

# ## Monthly activated users

# In[4]:


# Creating a dataframe with the observations of year 2018
df_2018 = data.loc[(data["year_purch"] == 2018)]


# In[5]:


# Creating a dataframe with the observations of year 2019
df_2019 = data.loc[(data["year_purch"] == 2019)]


# In[6]:


# Compute the number of customers that were active daily in the E-commerce for 2018
df_2018["month_year_purch"] = pd.to_datetime(df_2018["month_year_purch"], format = "%m-%Y")
df_mau = pd.concat([df_2018["month_year_purch"], df_2018["customer_unique_id"]], axis = 1)
df_mau.set_index("month_year_purch")
count_mau = df_mau.groupby("month_year_purch").size().to_frame("Customer unique id")


# In[7]:


# Plotting the previous data
plt.figure(figsize = (25,5))
sns.lineplot(data = count_mau, x = "month_year_purch", y = "Customer unique id")
plt.title("Monthly activated users in 2018")
#plt.savefig("MAU.png") 


# In[8]:


# Repeating the analysis for year 2019
df_mau_new = pd.concat([df_2019["month_year_purch"], df_2019["customer_unique_id"]], axis=1)
df_mau_new.set_index("month_year_purch")
count_mau_new = df_mau_new.groupby("month_year_purch").size().to_frame("Customer unique id")
plt.figure(figsize=(25,5))
sns.lineplot(data=count_mau_new, x="month_year_purch", y="Customer unique id")
plt.title("Monthly activated users in 2019")
#plt.savefig("MAU 2019.png") 


# ## Daily Spending and Daily Spending per Visitor (RPV)

# In[9]:


# Changing object into datetime and Dropping duplicates for 2018
df_2018["hour_purch"] = pd.to_datetime(df_2018["hour_purch"], format = "%Y-%m-%d")
df_2018 = df_2018.drop_duplicates()


# In[10]:


# Daily spending for each unique user in 2018
revenues_day = df_2018.groupby(["hour_purch", "customer_unique_id"], as_index=False)["price"].sum()


# In[11]:


# Computing the number of visitor per day in 2018
n_visitor_day = revenues_day.groupby("hour_purch").hour_purch.value_counts().to_frame()
n_visitor_day = n_visitor_day.rename(columns = {'hour_purch': "visitor/day"})
n_visitor_day = n_visitor_day.reset_index(level = 1, drop = True)


# In[12]:


# Computing the daily spending per user each day in 2018
revenues_day = revenues_day.set_index("hour_purch")
revenues_day_new = pd.merge(revenues_day, n_visitor_day, left_index=True, right_index=True)
revenues_day_new["rpv"] = revenues_day_new["price"]/ revenues_day_new["visitor/day"]


# In[13]:


# Plotting the results for 2018
plt.figure(figsize = (30,5))
plt.ylabel('Revenue / Visitor')
plt.title("Revenues per visitor each day 2018")
sns.lineplot(data=revenues_day_new, x = 'hour_purch', y = 'rpv')
plt.savefig("Revenues per visitor 2018.png") 


# In[14]:


# Changing object into datetime and Dropping duplicates for 2019
df_2019["hour_purch"] = pd.to_datetime(df_2019["hour_purch"], format = "%Y-%m-%d")


# In[15]:


# Daily spending for each unique user in 2019
revenues_day_19 = df_2019.groupby(["hour_purch", "customer_unique_id"], as_index = False)["price"].sum()


# In[16]:


# Computing the number of visitor per day in 2019
n_visitor_day_19 = revenues_day_19.groupby("hour_purch").hour_purch.value_counts().to_frame()
n_visitor_day_19 = n_visitor_day_19.rename(columns = {'hour_purch': "visitor/day"})
n_visitor_day_19 = n_visitor_day_19.reset_index(level = 1, drop = True)


# In[17]:


# Computing the daily spending per user each day in 2019
revenues_day_19 = revenues_day_19.set_index("hour_purch")
revenues_day_new19 = pd.merge(revenues_day_19, n_visitor_day_19, left_index=True, right_index=True)
revenues_day_new19["rpv"] = revenues_day_new19["price"]/ revenues_day_new19["visitor/day"]


# In[18]:


# Plotting the results for 2019
plt.figure(figsize = (30,5))
plt.ylabel('Revenue / Visitor')
plt.title("Revenues per visitor each day 2019")
sns.lineplot(data = revenues_day_new19, x = 'hour_purch', y = 'rpv')
plt.savefig("Revenues per visitor 2019.png") 


# In[19]:


# Revenues per day 2018
plt.figure(figsize = (30,5))
plt.ylabel('Revenue/day')
plt.title("Revenues per day 2018")
sns.lineplot(data = revenues_day_new, x = 'hour_purch', y = 'price')
plt.savefig("Revenues per day 2018.png") 


# In[20]:


# Revenues per day 2019
plt.figure(figsize = (30,5))
plt.ylabel('Revenue/day')
plt.title("Revenues per day 2019")
sns.lineplot(data = revenues_day_new19, x = 'hour_purch', y = 'price')
plt.savefig("Revenues per day 2019.png") 


# ## Purchase Trend

# In[21]:


# Purchase Trend in 2018
trend = df_2018.groupby(["hour_purch","order_id"], as_index = False)["order_id"].size()
trend = trend.groupby("hour_purch").hour_purch.value_counts().to_frame("Order per day")
trend = trend.reset_index(level = 1, drop = True)


# In[22]:


# Plotting the purchase trend 2018
plt.figure(figsize = (12,6))
plt.ylabel('order count 2018')
plt.title("Purchase trend 2018")
sns.lineplot(data = trend,x = 'hour_purch',y = 'Order per day')
plt.savefig("Purchase trend 2018.png") 


# In[23]:


# Purchase Trend in 2019
trend19 = df_2019.groupby(["hour_purch","order_id"], as_index = False)["order_id"].size()
#df_2018.groupby(["hour_purch"]).order_id.value_counts().to_frame()
trend19 = trend19.groupby("hour_purch").hour_purch.value_counts().to_frame("Order per day")
trend19 = trend19.reset_index(level = 1, drop = True)


# In[24]:


# Plotting the purchase trend 2019
plt.figure(figsize = (12,6))
plt.ylabel('order count 2019')
plt.title("Purchase trend 2019")
sns.lineplot(data = trend19,x = 'hour_purch',y = 'Order per day')
plt.savefig("Purchase trend 2019.png") 


# ## Best Sellers

# In[25]:


# Best sellers in 2018 (printed only the first 15)
best_sellers = df_2018.groupby(["seller_id", 'product_category_name']).price.sum().to_frame()
best_sellers = best_sellers.sort_values("price", ascending= False)
best_sellers[1:15]


# In[26]:


##best sellers in 2019 (printed only the first 15)
best_sellers = df_2019.groupby(["seller_id", 'product_category_name']).price.sum().to_frame()
best_sellers = best_sellers.sort_values("price", ascending= False)
best_sellers[1:15]


# ## Best Customers

# In[27]:


# Best customers 2018 (printed only the first 15)
best_customers = df_2018.groupby(["customer_unique_id", 'product_category_name']).price.sum().to_frame()
best_customers = best_customers.sort_values("price", ascending = False)
best_customers[1:15]


# In[28]:


# Best customers 2019 (printed only the first 15)
best_customers = df_2019.groupby(["customer_unique_id", 'product_category_name']).price.sum().to_frame()
best_customers = best_customers.sort_values("price", ascending = False)
best_customers[1:15]


# ## Spending by state

# In[453]:


# Customer spending by autonomous communities in 2018
rev_by_state2018 = df_2018.groupby("customer_autonomous_community").price.sum().sort_values(ascending = False).to_frame()
sns.barplot(data = rev_by_state2018, x = 'price', y = rev_by_state2018.index, palette = "BuPu_r" )
plt.title("Spending by autonomous community 2018")
#plt.savefig("Spending by autonomous community 2018.png")


# In[29]:


# Customer spending by autonomous communities in 2019
rev_by_state2019 = df_2019.groupby("customer_autonomous_community").price.sum().sort_values(ascending = False).to_frame()
sns.barplot(data = rev_by_state2019, x = 'price', y = rev_by_state2019.index, palette= "BuPu_r" )
plt.title("Spending by autonomous community 2019")
#plt.savefig("Spending by autonomous community 2019.png")

