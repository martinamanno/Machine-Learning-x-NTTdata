import pandas as pd
import numpy as np

#Import the different datasets 
geo= pd.read_csv("DATASETS/01.geo.csv",encoding='cp1252',sep=";")
custom= pd.read_csv("DATASETS/02.customers.csv",encoding='cp1252',sep=";")
sel= pd.read_csv("DATASETS/03.sellers.csv",encoding='cp1252',sep=";")
ord_status= pd.read_csv("DATASETS/04.order_status.csv",encoding='cp1252',sep=";")
ord_items= pd.read_csv("DATASETS/05.order_items.csv",encoding='cp1252',sep=";")
ord_pay= pd.read_csv("DATASETS/06.order_payments.csv",encoding='cp1252',sep=";")
prod_rev= pd.read_csv("DATASETS/07.product_reviews.csv",encoding='cp1252',sep=";")
prod= pd.read_csv("DATASETS/08.products.csv",encoding='cp1252',sep=";")

#check for the number of missing values (NaN)
geo.isna().sum()
custom.isna().sum()
sel.isna().sum()
ord_status.isna().sum()
ord_items.isna().sum()
ord_pay.isna().sum()
prod_rev.isna().sum()
prod.isna().sum()

#clean datasets 
ord_status[["ts_order_delivered_customer","ts_order_delivered_carrier","ts_order_approved","order_id","ts_order_estimated_delivery"]] = ord_status[["ts_order_delivered_customer","ts_order_delivered_carrier","ts_order_approved","order_id","ts_order_estimated_delivery"]].replace(np.nan, "Undefined")
ord_items["order_id"]= ord_items["order_id"].fillna("Undefined")
ord_pay["order_id"]= ord_pay["order_id"].fillna("Undefined")
prod["product_category_name"]= prod["product_category_name"].fillna(mode(prod["product_category_name"]))
prod[["product_photo_quantity","product_width_cm","product_height_cm",
      "product_length_cm","product_weight_gr"]] = prod[["product_photo_quantity",
                                                        "product_width_cm","product_height_cm","product_length_cm",
                                                        "product_weight_gr"]].replace(np.nan, median(prod[["product_photo_quantity","product_width_cm","product_height_cm","product_length_cm","product_weight_gr"]]))
                                                                   
import plotly.express as px
fig = px.scatter_mapbox(geo, lat="geo_latitude", lon="geo_longitude", hover_name="geo_city", hover_data=["geo_autonomous_community", "geo_admin1_code"],color_discrete_sequence=["fuchsia"], zoom=3, height=300)
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()
fig.write_html("Geo_map.html")

###### ORDERS 
order_status['order_status'].value_counts()
order_items['price'] = order_items['price'].apply(lambda x: x.replace(',','.'))
order_items['shipping_cost'] = order_items['shipping_cost'].apply(lambda x: x.replace(',','.'))
order_items['price'] = order_items['price'].astype('float')
order_items['shipping_cost'] = order_items['shipping_cost'].astype('float')
order_items['price'].mean()
order_items['shipping_cost'].mean()
order_items['price'].min()
order_items['price'].max()
order_items['order_item_sequence_id'].value_counts()
order_payments['payment_method'].value_counts()
order_payments['transaction_value'] = order_payments['transaction_value'].apply(lambda x: x.replace(',','.'))
order_payments['transaction_value'] = order_payments['transaction_value'].astype('float')
