#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mode,mean,median


# In[2]:


geo= pd.read_csv("/Users/martina/Desktop/DATA SCIENCE/Machine Learning /NTTData/20220308_dataset_LUISS_NTT/01.geo.csv",encoding='cp1252',sep=";")
custom= pd.read_csv("/Users/martina/Desktop/DATA SCIENCE/Machine Learning /NTTData/20220308_dataset_LUISS_NTT/02.customers.csv",encoding='cp1252',sep=";")
sel= pd.read_csv("/Users/martina/Desktop/DATA SCIENCE/Machine Learning /NTTData/20220308_dataset_LUISS_NTT/03.sellers.csv",encoding='cp1252',sep=";")
ord_status= pd.read_csv("/Users/martina/Desktop/DATA SCIENCE/Machine Learning /NTTData/20220308_dataset_LUISS_NTT/04.order_status.csv",encoding='cp1252',sep=";")
ord_items= pd.read_csv("/Users/martina/Desktop/DATA SCIENCE/Machine Learning /NTTData/20220308_dataset_LUISS_NTT/05.order_items.csv",encoding='cp1252',sep=";")
ord_pay= pd.read_csv("/Users/martina/Desktop/DATA SCIENCE/Machine Learning /NTTData/20220308_dataset_LUISS_NTT/06.order_payments.csv",encoding='cp1252',sep=";")
prod_rev= pd.read_csv("/Users/martina/Desktop/DATA SCIENCE/Machine Learning /NTTData/20220308_dataset_LUISS_NTT/07.product_reviews.csv",encoding='cp1252',sep=";")
prod= pd.read_csv("/Users/martina/Desktop/DATA SCIENCE/Machine Learning /NTTData/20220308_dataset_LUISS_NTT/08.products.csv",encoding='cp1252',sep=";")


# In[ ]:


#pd.set_option('display.max_rows', None)
#ord_status
prod.isna().sum()


# In[3]:


#clean df
ord_status[["ts_order_delivered_customer","ts_order_delivered_carrier","ts_order_approved","order_id","ts_order_estimated_delivery"]] = ord_status[["ts_order_delivered_customer","ts_order_delivered_carrier","ts_order_approved","order_id","ts_order_estimated_delivery"]].replace(np.nan, "Undefined")
ord_items["order_id"]= ord_items["order_id"].fillna("Undefined")
ord_pay["order_id"]= ord_pay["order_id"].fillna("Undefined")
prod["product_category_name"]= prod["product_category_name"].fillna(mode(prod["product_category_name"]))
prod[["product_photo_quantity","product_width_cm","product_height_cm",
      "product_length_cm","product_weight_gr"]] = prod[["product_photo_quantity",
                                                        "product_width_cm","product_height_cm","product_length_cm",
                                                        "product_weight_gr"]].replace(np.nan, median(prod[["product_photo_quantity","product_width_cm","product_height_cm","product_length_cm","product_weight_gr"]]))
                                                                   
                                                                     
                                                                    


# In[4]:


lol= geo.groupby("geo_autonomous_community").size().to_frame('Count')
g = sns.barplot(x = lol.index, y = 'Count', data = lol, color="darkblue" )
g.set(title='Count plot of autonomous communities in Spain of the dataset')
g.set_xticklabels(g.get_xticklabels(), rotation=45, horizontalalignment='right')
g.set_ylabel('Count')


# In[5]:


sns.scatterplot(data = geo, x = "geo_latitude", y = "geo_longitude", hue="geo_city")
plt.legend(bbox_to_anchor=(1.5, 1))


# In[6]:


custom.groupby(["customer_autonomous_community","customer_city"]).size().to_frame()


# In[12]:


count_custom= custom.groupby("customer_autonomous_community").size().to_frame("Count")
k= sns.barplot(x=count_custom.index, y="Count",data=count_custom, color= "pink")
k.set(title='Count plot of customers in autonomous communities of Spain')
k.set_xticklabels(g.get_xticklabels(), rotation=45, horizontalalignment='right')
k.set_ylabel('Count')


# In[17]:


count_sel.sum()


# In[51]:


geo


# In[55]:


import pandas as pd
from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame

geometry = [Point(xy) for xy in zip(geo['geo_longitude'], geo['geo_latitude'])]
gdf = GeoDataFrame(geo, geometry=geometry)   

#this is a simple map that goes with geopandas
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
gdf.plot(ax=world.plot(figsize=(10, 6)), marker='o', color='red', markersize=15)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[37]:


ord_status== custom


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[13]:


sel


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


#bins_pie = pd.cut(values,bins=[0,1000,2000,3000,4000,5000,6000], labels=["0-1000","1001-2000","2001-3000","3001-4000", "4001-5000","5001-6000"])
#pie_custom= count_custom.groupby(bins_pie).size().to_frame('Count')
#h=pie_custom.plot.pie(y = 'Count', autopct="%.1f%%", color)
#h.set_title("% of customers inside autonomous communities")
#plt.legend(title="Intervals of customers inside autonomous communities", bbox_to_anchor=(2,1))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:



ok= sel.groupby("seller_autonomous_community").size().to_frame('Count')
g = sns.barplot(x = ok.index, y = 'Count', data = ok, color="darkblue" )
g.set(title='Count plot of sellers in autonomous communities in Spain of the dataset')
g.set_xticklabels(g.get_xticklabels(), rotation=45, horizontalalignment='right')
g.set_ylabel('Count')

lol= geo.groupby("geo_autonomous_community").size().to_frame('Count')
m = sns.barplot(x = lol.index, y = 'Count', data = lol, color="darkblue" )
m.set(title='Count plot of autonomous communities in Spain of the dataset')
m.set_xticklabels(m.get_xticklabels(), rotation=45, horizontalalignment='right')
m.set_ylabel('Count')

m
g


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


g.set(title='Count plot of autonomous communities in Spain of the dataset')
g.set_xticklabels(g.get_xticklabels(), rotation=45, horizontalalignment='right')
g.set_ylabel('Count')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


custom


# In[ ]:





# In[ ]:


custom.groupby(["customer_autonomous_community","customer_city"]).size().to_frame('Count')


# In[ ]:


custom.groupby(["customer_autonomous_community","customer_city"]).size().to_frame('Count')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# denominations in the json file
import json
communities_geo = '/Users/martina/Desktop/map.geojson'

# open the json file - json.load() methods returns a python dictionary
with open(communities_geo) as geo["geo_autonomous_community"]:
    communities_json = json.load(geo["geo_autonomous_community"])

# we loop through the dictionary to obtain the name of the communities in the json file
denominations_json = []
for index in range(len(communities_json['cities'])):
    denominations_json.append(communities_json['ciy'][index]['properties']['name'])
    
denominations_json


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




