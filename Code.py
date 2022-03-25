import pandas as pd

geo= pd.read_csv("DATASETS/01.geo.csv",encoding='cp1252',sep=";")
custom= pd.read_csv("DATASETS/02.customers.csv",encoding='cp1252',sep=";")
sel= pd.read_csv("DATASETS/03.sellers.csv",encoding='cp1252',sep=";")
ord_status= pd.read_csv("DATASETS/04.order_status.csv",encoding='cp1252',sep=";")
ord_items= pd.read_csv("DATASETS/05.order_items.csv",encoding='cp1252',sep=";")
ord_pay= pd.read_csv("DATASETS/06.order_payments.csv",encoding='cp1252',sep=";")
prod_rev= pd.read_csv("DATASETS/07.product_reviews.csv",encoding='cp1252',sep=";")
prod= pd.read_csv("DATASETS/08.products.csv",encoding='cp1252',sep=";")
