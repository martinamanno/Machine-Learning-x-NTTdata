#!/usr/bin/env python
# coding: utf-8

# In[46]:


# import the libraries needed for the analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mode,mean,median
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from lightfm import LightFM


# In[16]:


# loading the datasets
geo = pd.read_csv("01.geo.csv", encoding='cp1252', sep=";")
custom = pd.read_csv("02.customers.csv", encoding='cp1252', sep=";")
sel = pd.read_csv("03.sellers.csv", encoding='cp1252', sep=";")
ord_status = pd.read_csv("04.order_status.csv", encoding='cp1252', sep=";")
ord_items = pd.read_csv("05.order_items.csv", encoding='cp1252', sep=";")
ord_pay = pd.read_csv("06.order_payments.csv", encoding='cp1252', sep=";")
prod_rev = pd.read_csv("07.product_reviews.csv", encoding='cp1252', sep=";")
prod = pd.read_csv("08.products.csv", encoding='cp1252', sep=";")


# In[17]:


# We notice there are 610 NaN values in product_category_name,
# for the purpose of our analysis we substitute them with the category "Others"
prod["product_category_name"] = prod["product_category_name"].fillna("Others")
max_vals = ord_items.groupby("order_id")["order_item_sequence_id"].max().to_dict()
ord_items["max_order"] = ord_items["order_id"].map(max_vals)


# In[18]:


# Merging datasets and data cleaning
pr = pd.merge(prod, prod_rev, on='product_id')
custom_df = pd.merge(custom, ord_status, on='customer_id')
custom_df.dropna(subset=['order_id'], inplace=True)
custom_df = custom_df.drop_duplicates(['customer_unique_id'])
ord_items.dropna(subset=['order_id'])


# In[19]:


# Cleaning and merging the datasets by dropping duplicates
new = pd.merge(ord_items, pr, on=['order_id', 'product_id'])
final = pd.merge(new, custom_df, on=['order_id'])
final = final.drop(columns=['order_item_sequence_id'])
final.drop_duplicates()


# In[20]:


# Compute the number of unique users and products
n_users = final["customer_unique_id"].unique().shape[0]
n_items = final["product_id"].unique().shape[0]
print("Number of users= "+ str(n_users) + "| Number of products= "+ str(n_items))


# In[21]:


# Select columns needed for the recommendation systems
cols = ['product_category_name', 
        'customer_unique_id', 
        'review_score', 
        'product_id']
final = final[cols]


# In[22]:


# Extract a sample on which to perform the recommendation system
final_sample= final.sample(10000)


# In[23]:


# Select users and products from sample
users = final_sample['customer_unique_id']
products = final_sample['product_id']


# In[24]:


# Encode categorical values corresponding to the user and product ids
le = LabelEncoder() # create object LabelEncoder
final_sample.iloc[:, 1] = le.fit_transform(final_sample.iloc[:, 1])
le1 = LabelEncoder() # create object LabelEncoder
final_sample.iloc[:, 3] = le1.fit_transform(final_sample.iloc[:, 3])


# In[25]:


# Dictionary containing customer unique id and corresponding encoding value
final_sample.set_index(users)['customer_unique_id'].to_dict()


# In[26]:


# Dictionary containing product id and corresponding encoding value
final_sample.set_index(products)['product_id'].to_dict()


# # Recommendation- First approach

# In[28]:


# Creating a matrix containinf customers and products
n_users = final_sample.customer_unique_id.unique().shape[0]
n_items = final_sample.product_id.unique().shape[0]
n_items = final_sample['product_id'].max()
A = np.zeros((n_users,n_items))

print("Original rating matrix : ",A)


# In[32]:


# Creation of the sparse matrix needed as input for the Nearest Neighbors algorithm
for i in range(len(A)):
    for j in range(len(A[0])):
        if A[i][j] >= 3:
            A[i][j] = 1
        else:
            A[i][j] = 0
csr_sample = csr_matrix(A)
print(csr_sample)


# In[33]:


# Application of the algorithm, it computes the distance among customers according to the cosine similarity
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=3, n_jobs=-1)
knn.fit(csr_sample)


# In[34]:


# Printing what the user likes
dataset_sort_des = final_sample.sort_values(['customer_unique_id'], ascending=[True])
filter1 = dataset_sort_des[dataset_sort_des['customer_unique_id'] == 768].product_id
filter1 = filter1.tolist()
filter1 = filter1[:20]
print("Items liked by user: ",filter1)


# In[35]:


# Printing the recommended items to the user
distances1 = []
indices1 = []
for i in filter1:
    distances , indices = knn.kneighbors(csr_sample[i],n_neighbors=3)
    indices = indices.flatten()
    indices = indices[1:]
    indices1.extend(indices)
print("Items to be recommended: ",indices1)


# # Recommendation- Second approach

# In[39]:


# creating a rating matrix transposed needed as input in the algorithm
rating_crosstab = final_sample.pivot_table(values='review_score', 
                                           index='customer_unique_id', 
                                           columns='product_category_name', 
                                           fill_value=0)
rating_crosstab.head()
X = rating_crosstab.T


# In[40]:


#Applying the Truncated SVD for linear dimensionality reduction
SVD = TruncatedSVD(n_components=12, random_state=5)
resultant_matrix = SVD.fit_transform(X)
resultant_matrix.shape
corr_mat = np.corrcoef(resultant_matrix)
corr_mat.shape


# In[44]:


# Computation of similarity among products based on correlation 
col_idx = rating_crosstab.columns.get_loc("toys games")
corr_specific = corr_mat[col_idx]
pd.DataFrame({'corr_specific':corr_specific, 'Product': rating_crosstab.columns}).sort_values('corr_specific', ascending=False).head(10)


# # Recommendation- Third Approach
# 
# 

# In[47]:


# Splitting the dataset into train and test
train, test = train_test_split(final_sample,test_size= 0.25, random_state=1)


# In[48]:


# Create data to insert into the algorithm
item_dict = {}
df = final_sample[['product_id', 'product_category_name']].sort_values('product_id').reset_index()
for i in range(df.shape[0]):
    item_dict[(df.loc[i,'product_id'])] = df.loc[i,'product_category_name']
# Dummify categorical features
final_sample_transformed = final_sample.drop(columns="customer_unique_id")
final_sample_transformed = pd.get_dummies(final_sample, columns = ['review_score', 'product_category_name'])
final_sample_transformed = final_sample_transformed.sort_values('product_id').reset_index().drop('index', axis=1)
final_sample_transformed.head(5)
# Convert to csr matrix
final_csr = csr_matrix(final_sample_transformed.drop('product_id', axis=1).values)


# In[49]:


# Create another a rating matrix using products and reviews
user_book_interaction = pd.pivot_table(final_sample, index='customer_unique_id', columns='product_id', values='review_score')
# Fill missing values with 0
user_book_interaction = user_book_interaction.fillna(0)
user_id = list(user_book_interaction.index)
user_dict = {}
counter = 0 
for i in user_id:
    user_dict[i] = counter
    counter += 1
# Convert to csr matrix
user_book_interaction_csr = csr_matrix(user_book_interaction.values)
user_book_interaction_csr


# In[50]:


# LightFM algorithm for recommendation
model = LightFM(loss='warp',
                random_state=2016,
                learning_rate=0.90,
                no_components=150,
                user_alpha=0.000005)
model = model.fit(user_book_interaction_csr,
                  epochs=100,
                  num_threads=16, verbose=False)


# In[51]:


# Use a function to summarize the results of the algorithm
def sample_recommendation_user(model, final_sample, customer_unique_id, user_dict, 
                               item_dict,threshold = 0,nrec_items = 5, show = True):
    n_users, n_items = final_sample.shape
    user_x = user_dict[customer_unique_id]
    scores = pd.Series(model.predict(user_x,np.arange(n_items), item_features=final_csr))
    scores.index = final_sample.columns
    scores = list(pd.Series(scores.sort_values(ascending = False).index))
    
    known_items = list(pd.Series(final_sample.loc[customer_unique_id,:]                                  [final_sample.loc[customer_unique_id,:] > threshold].index).sort_values(ascending = False))
    
    scores = [x for x in scores if x not in known_items]
    return_score_list = scores[0:nrec_items]
    known_items = list(pd.Series(known_items).apply(lambda x: item_dict[x]))
    scores = list(pd.Series(return_score_list).apply(lambda x: item_dict[x]))
    if show == True:
        print ("User: " + str(customer_unique_id))
        print("Known Likes:")
        
    counter = 1
    for i in known_items:
        print(str(counter) + '- ' + i)
        counter += 1
    print("\n Recommended Items:")
    for i in scores:
        print(str(counter) + '- ' + i)
        counter += 1


# In[52]:


# Result of the algorithm for user 768
sample_recommendation_user(model, user_book_interaction,768, user_dict, item_dict)

