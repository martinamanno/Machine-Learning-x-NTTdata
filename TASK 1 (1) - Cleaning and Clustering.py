"""
Machine Learning techniques to increase profitability. 
NTTData x Luiss
Martina Manno, Martina Crisafulli, Olimpia Sannucci, Hanna Carucci Viterbi, Tomas Ryen
"""

# import the libraries needed for the analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mode, mean, median
import collections
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Loading the datasets
geo = pd.read_csv("01.geo.csv", encoding='cp1252', sep=";")
custom = pd.read_csv("02.customers.csv", encoding='cp1252', sep=";")
sel = pd.read_csv("03.sellers.csv", encoding='cp1252', sep=";")
ord_status = pd.read_csv("04.order_status.csv", encoding='cp1252', sep=";")
ord_items = pd.read_csv("05.order_items.csv", encoding='cp1252', sep=";")
ord_pay = pd.read_csv("06.order_payments.csv", encoding='cp1252', sep=";")
prod_rev = pd.read_csv("07.product_reviews.csv", encoding='cp1252', sep=";")
prod = pd.read_csv("08.products.csv", encoding='cp1252', sep=";")

# Data understanding: Geo dataset
#plotting customers by city
customer= custom[['customer_city']]
customer.rename(columns = {'customer_city':'geo_city'}, inplace = True)
geo_customer = pd.merge(geo, customer, on ='geo_city')
geo_customer['count'] = geo_customer.groupby('geo_city')['geo_city'].transform('count')
fig = px.scatter_mapbox(geo_customer, lat="geo_latitude", lon="geo_longitude", hover_name="geo_city", hover_data=['count', "geo_autonomous_community", "geo_admin1_code"],color_discrete_sequence=["blue"], zoom=3, height=300)
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()
fig.write_html("Cust_map.html")

#plotting sellers by city
seller = sel[['seller_city']]
seller.rename(columns = {'seller_city':'geo_city'}, inplace = True)
geo_seller = pd.merge(geo, seller, on = 'geo_city')
geo_seller['count'] = geo_seller.groupby('geo_city')['geo_city'].transform('count')
fig = px.scatter_mapbox(geo_seller, lat="geo_latitude", lon="geo_longitude", hover_name="geo_city", hover_data=['count', "geo_autonomous_community", "geo_admin1_code"],color_discrete_sequence=["blue"], zoom=3, height=300)
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()
fig.write_html("sel_map.html")


#Data Cleaning and Preparation
# We notice there are 610 NaN values in product_category_name,
# for the purpose of our analysis we substitute them with the category "Others"
prod["product_category_name"].isna().sum()
prod["product_category_name"] = prod["product_category_name"].fillna("Others")

# We added one additional value identifying the number of elements for each order
max_vals = ord_items.groupby("order_id")["order_item_sequence_id"].max().to_dict()
ord_items["max_order"] = ord_items["order_id"].map(max_vals)

# Cleaning the customers df by dropping NaN and duplicates
custom_df = pd.merge(custom, ord_status, on='customer_id')
custom_df.dropna(subset=['order_id'], inplace=True)
custom_df = custom_df.drop_duplicates(['customer_unique_id'])

# Drop NaN from ord_items in column order_id
ord_items.dropna(subset=['order_id'])

# Cleaning and merging the datasets by dropping duplicates
new = pd.merge(ord_items, prod_rev, on=['order_id', 'product_id'])
new = pd.merge(new, prod, on=['product_id'])
final = pd.merge(new, custom_df, on=['order_id'])
final = final.drop_duplicates()
final = final.drop(columns=['order_item_sequence_id'])
final.drop_duplicates()

# Completing the merging process
ord_pay = ord_pay.dropna()
final_new = pd.merge(final, ord_pay, on=['order_id'])
final_new = final_new.drop(columns=['ts_order_estimated_delivery',
                                    'ts_order_delivered_carrier',
                                    'ts_order_purchase',
                                    'ts_order_approved',
                                    'ts_order_delivered_customer'])
final_new = final_new.drop(columns=['product_weight_gr',
                                    'product_length_cm',
                                    'product_height_cm',
                                    'product_width_cm'])
final_new = final_new.drop_duplicates()
final_new = final_new.drop(columns=['max_shipping_seller_date',
                                    'review_date',
                                    'order_status',
                                    'payment_method_sequence_id'])

# before applying any algorithm we make sure the dataset does not contain NaN
final_new = final_new[~final_new.isin([np.nan, np.inf, -np.inf]).any(1)]

# Definition of the independent variables used in the analysis
X = final_new[["payment_method",
               "price",
               'shipping_cost',
               'max_order',
               'review_score',
               'product_category_name',
               'product_photo_quantity',
               'customer_autonomous_community',
               "payment_installments_quantity",
               "transaction_value"]]

# Copy of the dataset before applying models
X_copy = X.copy()
X_copy = pd.DataFrame(X_copy)

# Converting numerical columns datatype as float
X_copy["price"] = [float(str(i).replace(",", ".")) for i in X_copy["price"]]
X_copy["shipping_cost"] = [float(str(i).replace(",", ".")) for i in X_copy["shipping_cost"]]
X_copy["transaction_value"] = [float(str(i).replace(",", ".")) for i in X_copy["transaction_value"]]

# Define categorical variables to be encoded
categorical_cols = ['payment_method',
                    'product_category_name', 
                    'customer_autonomous_community']

# Numerical columns to be standardized 
cols_stand = ['price',
             'shipping_cost',
             'transaction_value',
             'max_order',
             'product_photo_quantity',
             'review_score',
             'payment_installments_quantity']

# Transform the categorical values into dummies
X_copy = pd.get_dummies(X_copy, columns = categorical_cols, drop_first= True)

# Standardization of independent variables
sc_X = StandardScaler()
X_copy[cols_stand] = sc_X.fit_transform(X_copy[cols_stand])

# Plotting the elbow curve to decide how many clusters to take
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X_copy)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# The optimal number of cluster is 6, we fit the data to the K-Means algorithm to find the clusters
km = KMeans(n_clusters=6)
km.fit(X_copy)
km.predict(X_copy)
labels = km.labels_

# Predicting the cluster segments
y_kmeans = km.fit_predict(X_copy)

# Add the cluster to the dataframe
X_copy['Cluster Labels'] = labels
X_copy['Segment'] = X_copy['Cluster Labels'].map({0:'First', 1:'Second',2:'Third',3:'Fourth',4:'Fifth',5: 'Sixth'})
# Order the clusters
X_copy['Segment'] = X_copy['Segment'].astype('category')
X_copy['Segment'] = X_copy['Segment'].cat.reorder_categories(['First','Second','Third','Fourth','Fifth','Sixth'])
X_copy.rename(columns = {'Cluster Labels':'Total'}, inplace = True)

# Transform back the standardized features
X_copy[cols_stand] = sc_X.inverse_transform(X_copy[cols_stand])

# Six clusters segment
cc0 = X_copy[X_copy['Segment'] == 'First']
cc1 = X_copy[X_copy['Segment'] == 'Second']
cc2 = X_copy[X_copy['Segment'] == 'Third']
cc3 = X_copy[X_copy['Segment'] == 'Fourth']
cc4 = X_copy[X_copy['Segment'] == 'Fifth']
cc5 = X_copy[X_copy['Segment'] == 'Sixth']

# Undummy function applied to the encoded categorical variables
def undummy(d):
    return d.dot(d.columns)
X_copy= X_copy.assign(Category=X_copy.filter(regex='^product_category_name').pipe(undummy),
                      Payment=X_copy.filter(regex='^payment_method').pipe(undummy),
                      Location=X_copy.filter(regex='^customer_autonomous_community').pipe(undummy))

# Drop columns 
cols_to_drop= ["payment_method_credit_card",
               'payment_method_debit_card',
               'payment_method_voucher',
               'product_category_name_automotive',
               'product_category_name_bakeware',
               'product_category_name_beauty & personal care',
               'product_category_name_bedroom decor',
               'product_category_name_book',
               'product_category_name_business office',
               'product_category_name_camera & photo',
               'product_category_name_cd vinyl',
               'product_category_name_ceiling fans',
               'product_category_name_cell phones',
               'product_category_name_cleaning supplies',
               'product_category_name_coffee machines',
               'product_category_name_comics',
               'product_category_name_computer accessories',
               'product_category_name_computers tablets',
               'product_category_name_diet sports nutrition',
               'product_category_name_dvd',
               'product_category_name_event & party supplies',
               'product_category_name_fabric',
               'product_category_name_fashion & shoes',
               'product_category_name_film & photography',
               'product_category_name_fire safety',
               'product_category_name_food',
               'product_category_name_fragrance',
               'product_category_name_furniture',
               'product_category_name_handbags & accessories',
               'product_category_name_hardware',
               'product_category_name_headphones',
               'product_category_name_health household',
               'product_category_name_home accessories',
               'product_category_name_home appliances',
               'product_category_name_home audio',
               'product_category_name_home emergency kits',
               'product_category_name_home lighting',
               'product_category_name_home security systems',
               'product_category_name_jewelry',
               'product_category_name_kids',
               'product_category_name_kids fashion',
               'product_category_name_kitchen & dining',
               'product_category_name_lawn garden',
               'product_category_name_light bulbs',
               'product_category_name_luggage',
               'product_category_name_mattresses & pillows',
               'product_category_name_medical supplies',
               "product_category_name_men's fashion",
               'product_category_name_model hobby building',
               'product_category_name_monitors',
               'product_category_name_music instruments',
               'product_category_name_office products',
               'product_category_name_oral care',
               'product_category_name_painting',
               'product_category_name_pet food',
               'product_category_name_pet supplies',
               'product_category_name_safety apparel',
               'product_category_name_seasonal decor',
               'product_category_name_sofa',
               'product_category_name_sport outdoors',
               'product_category_name_television & video',
               'product_category_name_tools home improvement',
               'product_category_name_toys games',
               'product_category_name_underwear',
               'product_category_name_videogame',
               'product_category_name_videogame console',
               'product_category_name_wall art',
               'product_category_name_watches',
               'product_category_name_wellness & relaxation',
               "product_category_name_woman's fashion",
               'customer_autonomous_community_Aragón',
               'customer_autonomous_community_Baleares',
               'customer_autonomous_community_Cantabria',
               'customer_autonomous_community_Castilla y León',
               'customer_autonomous_community_Castilla-La Mancha',
               'customer_autonomous_community_Cataluña',
               'customer_autonomous_community_Comunidad Foral de Navarra',
               'customer_autonomous_community_Comunidad Valenciana',
               'customer_autonomous_community_Comunidad de Madrid',
               'customer_autonomous_community_Extremadura',
               'customer_autonomous_community_Galicia',
               'customer_autonomous_community_Islas Canarias',
               'customer_autonomous_community_La Rioja',
               'customer_autonomous_community_País Vasco',
               'customer_autonomous_community_Principado de Asturias',
               'customer_autonomous_community_Región de Murcia']
X_copy= X_copy.drop(columns= cols_to_drop)

# Summary of the clusters, choosing mean for numerical columns and mode for categorical ones
clusters3 = X_copy.groupby('Segment').agg(
    {
        'Total':'count',
        'Category': lambda x: x.value_counts().index[0],
        'Payment': lambda x: x.value_counts().index[0],
        'Location':lambda x: x.value_counts().index[0],
        'price': 'mean',
        'shipping_cost': 'mean',
        'max_order': 'mean',
        'review_score': 'mean',
        'product_photo_quantity': 'mean',
        'transaction_value':'mean',
        'payment_installments_quantity':'mean'
    }
).reset_index()
clusters3 

# Create boxplots to the distribution of the variables in each cluster
fig, axes = plt.subplots(1, 6, figsize=(30,15))
ax = sns.boxplot(ax=axes[0], x="Segment", y="price", data=X_copy)
ax.title.set_text('Price in All Clusters')
ax2 = sns.boxplot(ax=axes[1], x="Segment", y="review_score", data=X_copy)
ax2.title.set_text('Review Score in All Clusters')
ax3 = sns.boxplot(ax=axes[2], x="Segment", y="product_photo_quantity", data=X_copy)
ax3.title.set_text('Photo Quantity  in All Clusters')
ax4 = sns.boxplot(ax=axes[3], x="Segment", y="payment_installments_quantity", data=X_copy)
ax4.title.set_text('Installments in All Clusters')
ax5 = sns.boxplot(ax=axes[4], x="Segment", y="transaction_value", data=X_copy)
ax5.title.set_text('Transaction Value in All Clusters')
ax6 = sns.boxplot(ax=axes[5], x="Segment", y="shipping_cost", data=X_copy)
ax6.title.set_text('Shipping Cost  in All Clusters')
plt.show()
