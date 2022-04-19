#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mode,mean,median
import collections


# In[ ]:


geo= pd.read_csv("01.geo.csv",encoding='cp1252',sep=";")
custom= pd.read_csv("/Users/martina/Desktop/DATA SCIENCE/Machine Learning /NTTData/20220308_dataset_LUISS_NTT/02.customers.csv",encoding='cp1252',sep=";")
sel= pd.read_csv("/Users/martina/Desktop/DATA SCIENCE/Machine Learning /NTTData/20220308_dataset_LUISS_NTT/03.sellers.csv",encoding='cp1252',sep=";")
ord_status= pd.read_csv("/Users/martina/Desktop/DATA SCIENCE/Machine Learning /NTTData/20220308_dataset_LUISS_NTT/04.order_status.csv",encoding='cp1252',sep=";")
ord_items= pd.read_csv("/Users/martina/Desktop/DATA SCIENCE/Machine Learning /NTTData/20220308_dataset_LUISS_NTT/05.order_items.csv",encoding='cp1252',sep=";")
ord_pay= pd.read_csv("/Users/martina/Desktop/DATA SCIENCE/Machine Learning /NTTData/20220308_dataset_LUISS_NTT/06.order_payments.csv",encoding='cp1252',sep=";")
prod_rev= pd.read_csv("/Users/martina/Desktop/DATA SCIENCE/Machine Learning /NTTData/20220308_dataset_LUISS_NTT/07.product_reviews.csv",encoding='cp1252',sep=";")
prod= pd.read_csv("/Users/martina/Desktop/DATA SCIENCE/Machine Learning /NTTData/20220308_dataset_LUISS_NTT/08.products.csv",encoding='cp1252',sep=";")


# In[3]:


#pd.set_option('display.max_rows', None)
#ord_status
ord_status.isna().sum()


# In[4]:


ord_items.isna().sum()


# In[5]:


ord_pay.isna().sum()


# In[6]:


prod.isna().sum()
prod["product_category_name"]= prod["product_category_name"].fillna("Undefined")


# In[7]:


prod.isna().sum()


# In[8]:


prod_rev.isna().sum()


# In[9]:


custom.isna().sum()


# In[10]:


ord_status.dropna(subset=["order_id"], inplace=True)


# In[11]:


#to check whether customer_id is equal in both 
#for i in custom["customer_id"]:
    #for j in ord_status["customer_id"]:
        #if i==j:
            #print(True)
        
#print(i,j)


# In[12]:


#no duplicates in order_id of ord_status

#list_ido = [item for item, count in collections.Counter(ord_status['order_id']).items() if count > 1]
#list_ido


# In[13]:


#9804 duplicates in order_id of ord_status, repeated more than once 
#list_2 = [item for item, count in collections.Counter(ord_items['order_id']).items() if count > 1]


# In[14]:


#2288 observations are repeated more than twice 
#list_ = [item for item, count in collections.Counter(ord_items['order_id']).items() if count > 20]


# In[15]:


#list_3 = [item for item, count in collections.Counter(ord_pay['order_id']).items() if count > 1]


# In[ ]:


#list_4 = [item for item, count in collections.Counter(prod_rev['order_id']).items() if count > 1]


# In[16]:


max_vals = ord_items.groupby("order_id")["order_item_sequence_id"].max().to_dict()
ord_items["max_order"] = ord_items["order_id"].map(max_vals)
ord_items


# In[17]:


ord_items= ord_items.drop_duplicates(['order_id'])


# In[ ]:


#comparison_column = np.where(prod_rev['product_id'] == prod['product_id'], True, False)
prod_rev


# In[ ]:


#to_be_cleaned= prod.groupby(["product_category_name", "product_id"]).size().tolist()


# In[ ]:


#the product_id are always equal in both dataframes, this means that prod dataset has no duplicates for product_id
prod_rev['product_id'].isin(prod['product_id']).value_counts()


# In[18]:


df_merged= pd.merge(prod_rev, prod, on='product_id')


# In[19]:


ord_merged= ord_status.merge(ord_pay, on='order_id').merge(ord_items,on= "order_id")


# In[20]:


all_merged= pd.merge(ord_merged, df_merged, on=['order_id','product_id'])


# In[21]:


all_merged.columns


# In[22]:


final_df= pd.merge(all_merged, custom, on='customer_id')


# In[23]:


final_df.columns


# In[24]:


final_df =final_df[~final_df.isin([np.nan, np.inf, -np.inf]).any(1)]
X= final_df[["payment_method","price",'shipping_cost','max_order'
            ,'review_score','product_category_name','product_photo_quantity','customer_city']]


# In[25]:


X


# In[ ]:


#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder_1 = LabelEncoder() # create object LabelEncoder
#X.iloc[:, 0] = labelencoder_1.fit_transform(X.iloc[:, 0])
#labelencoder_2 = LabelEncoder() # create object LabelEncoder
#X.iloc[:, 5] = labelencoder_2.fit_transform(X.iloc[:, 5])
#labelencoder_3 = LabelEncoder() # create object LabelEncoder
#X.iloc[:, 7] = labelencoder_3.fit_transform(X.iloc[:, 7])
#onehotencoder = OneHotEncoder(categorical_features=[4]) # Dummy encode column 3 ie 0 0 1
#X = onehotencoder.fit_transform(X.iloc[:, [0,5,7]])


# In[ ]:


#X["shipping_cost"]  = [float(str(i).replace(",", ".")) for i in X["shipping_cost"] ]


# In[ ]:


#from sklearn.compose import ColumnTransformer
#ct = ColumnTransformer([("X", OneHotEncoder(dtype = int), [1])], remainder = 'passthrough')
#X = ct.fit_transform(X)
#X = X[:, 1:]


# In[ ]:


final_df


# In[ ]:


#from sklearn.cluster import KMeans
#wcss = []
#for i in range(1, 11):
    #kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    #kmeans.fit(X)
    #wcss.append(kmeans.inertia_)
#plt.plot(range(1, 11), wcss)
#plt.title('The Elbow Method')
#plt.xlabel('Number of clusters')
#plt.ylabel('WCSS')
#plt.show()


# In[ ]:


#km = KMeans(n_clusters=5)
#km.fit(X)
#km.predict(X)
#labels = km.labels_


# In[ ]:


X


# In[26]:


X_copy = X.copy()


# In[52]:


X_copy= pd.DataFrame(X_copy)


# In[57]:


X_copy


# In[ ]:





# In[59]:


#converting numerical columns datatype as float
X_copy[1]  = [float(str(i).replace(",", ".")) for i in X_copy[1] ]
#X_copy[:, 1] = X_copy[:,1].astype(float)
X_copy[2]  = [float(str(i).replace(",", ".")) for i in X_copy[2] ]
#X_copy[:, 2] = X_copy[:,2].astype(float)


# In[61]:


# Function for plotting elbow curve
def plot_elbow_curve(start, end, data):
    no_of_clusters = list(range(start, end+1))
    cost_values = []
    
    for k in no_of_clusters:
        test_model = KPrototypes(n_clusters=k, init='Huang', random_state=42)
        test_model.fit_predict(X_copy, categorical=[0,5,7])
        cost_values.append(test_model.cost_)
        
    sns.set_theme(style="whitegrid", palette="bright", font_scale=1.2)
    
    plt.figure(figsize=(15, 7))
    ax = sns.lineplot(x=no_of_clusters, y=cost_values, marker="o", dashes=False)
    ax.set_title('Elbow curve', fontsize=18)
    ax.set_xlabel('No of clusters', fontsize=14)
    ax.set_ylabel('Cost', fontsize=14)
    ax.set(xlim=(start-0.1, end+0.1))
    plt.plot()
plot_elbow_curve(2,10, X_copy)


# In[64]:


no_of_clusters = list(range(1, 11))
cost_values = []
    
for k in no_of_clusters:
    test_model = KPrototypes(n_clusters=k, init='Huang', random_state=42)
    test_model.fit_predict(X_copy, categorical=[0,5,7])
    cost_values.append(test_model.cost_)
        
sns.set_theme(style="whitegrid", palette="bright", font_scale=1.2)
    
plt.figure(figsize=(15, 7))
ax = sns.lineplot(x=no_of_clusters, y=cost_values, marker="o", dashes=False)
ax.set_title('Elbow curve', fontsize=18)
ax.set_xlabel('No of clusters', fontsize=14)
ax.set_ylabel('Cost', fontsize=14)
ax.set(xlim=(start-0.1, end+0.1))
plt.plot()


# In[ ]:


X_cost = pd.DataFrame({'Cluster':range(1, 11), 'Cost':cost})
X_cost.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### VECCHIO SCRIPT

# In[ ]:


#clean df
ord_items["order_id"]= ord_items["order_id"].fillna("Undefined")
ord_pay["order_id"]= ord_pay["order_id"].fillna("Undefined")
prod["product_category_name"]= prod["product_category_name"].fillna(mode(prod["product_category_name"]))
prod[["product_photo_quantity","product_width_cm","product_height_cm",
      "product_length_cm","product_weight_gr"]] = prod[["product_photo_quantity",
                                                        "product_width_cm","product_height_cm","product_length_cm",
                                                        "product_weight_gr"]].replace(np.nan, median(prod[["product_photo_quantity","product_width_cm","product_height_cm","product_length_cm","product_weight_gr"]]))
                                                                   
                                                                     
                                                                    


# In[ ]:





# In[ ]:


lol= geo.groupby("geo_autonomous_community").size().to_frame('Count')
g = sns.barplot(x = lol.index, y = 'Count', data = lol, color="darkblue" )
g.set(title='Count plot of autonomous communities in Spain of the dataset')
g.set_xticklabels(g.get_xticklabels(), rotation=45, horizontalalignment='right')
g.set_ylabel('Count')


# In[ ]:


sns.scatterplot(data = geo, x = "geo_latitude", y = "geo_longitude", hue="geo_city")
plt.legend(bbox_to_anchor=(1.5, 1))


# In[ ]:


custom.groupby(["customer_autonomous_community","customer_city"]).size().to_frame()


# In[ ]:


count_custom= custom.groupby("customer_autonomous_community").size().to_frame("Count")
k= sns.barplot(x=count_custom.index, y="Count",data=count_custom, color= "pink")
k.set(title='Count plot of customers in autonomous communities of Spain')
k.set_xticklabels(g.get_xticklabels(), rotation=45, horizontalalignment='right')
k.set_ylabel('Count')


# In[ ]:


count_sel.sum()


# In[ ]:


geo


# In[ ]:


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


pd.set_option('display.max_colwidth', None)
prod.groupby("product_category_name").size().to_frame()


# In[ ]:


def checkIfDuplicates_1(listOfElems):
    if len(listOfElems) == len(set(listOfElems)):
        return False
    else:
        return True


# In[ ]:


checkIfDuplicates_1(custom["customer_id"])


# In[ ]:


checkIfDuplicates_1(ord_status["order_id"])
from collections import Counter

[k for k,v in Counter(ord_status["order_id"]).items() if v>1]


# In[ ]:


custom.loc[custom["customer_id"]=="472acc24324ad4cee482fe4ef5910dc1"]


# In[ ]:





# In[ ]:





# In[ ]:


ord_status== custom


# In[ ]:


#bins_pie = pd.cut(values,bins=[0,1000,2000,3000,4000,5000,6000], labels=["0-1000","1001-2000","2001-3000","3001-4000", "4001-5000","5001-6000"])
#pie_custom= count_custom.groupby(bins_pie).size().to_frame('Count')
#h=pie_custom.plot.pie(y = 'Count', autopct="%.1f%%", color)
#h.set_title("% of customers inside autonomous communities")
#plt.legend(title="Intervals of customers inside autonomous communities", bbox_to_anchor=(2,1))


# In[ ]:


g.set(title='Count plot of autonomous communities in Spain of the dataset')
g.set_xticklabels(g.get_xticklabels(), rotation=45, horizontalalignment='right')
g.set_ylabel('Count')


# In[ ]:


custom.groupby(["customer_autonomous_community","customer_city"]).size().to_frame('Count')


# In[ ]:


custom.groupby(["customer_autonomous_community","customer_city"]).size().to_frame('Count')


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




