import os
import sys 
import time
from pdb import set_trace as bp
from scipy.sparse import coo_matrix
import numpy as np 
import pandas as pd
import pickle 

def predict_rating(user_index, item_index, ratings, item_similarity, k=20):
    
    numerator = 0
    denominator = 0
    rated_items = np.where(ratings[user_index] != 0)[0]
    simm_items = np.argsort(item_similarity[item_index])[::-1].tolist()
    cnt_k = 0
    for i in range(len(simm_items)):
        if i != item_index  and ratings[user_index][i] != 0:
            cnt_k = cnt_k + 1
            numerator += ratings[user_index][i] * item_similarity[item_index][i]
            denominator += item_similarity[item_index][i]
        if cnt_k==k:
          break
    if denominator == 0:
        return 0
    else:
        return numerator / denominator

def content_item_item_cf(saved_train_matrix, item_similarity, ratings_per_user_id):
    train_matrix_shape = saved_train_matrix.shape 
    train_matrix_shape = [x for x in train_matrix_shape]
    train_matrix_shape[0] = train_matrix_shape[0] + 1 #I have added the 1 to num_users 
    new_train_matrix = np.zeros(train_matrix_shape)
    new_train_matrix[:train_matrix_shape[0]-1, :train_matrix_shape[1]]=saved_train_matrix[:,:]
    """
    for user_i in range(train_matrix_shape[0]-1):
        for item_j in range(train_matrix_shape[1]):
            new_train_matrix[user_i][item_j] = saved_train_matrix[user_i][item_j]
    
    for idx in ratings_per_user_id.nonzero()[0]:
        new_train_matrix[train_matrix_shape[0]-1][idx] = ratings_per_user_id[idx]
    """
    for idx in ratings_per_user_id.nonzero()[0]:
        new_train_matrix[train_matrix_shape[0]-1][idx] = ratings_per_user_id[idx]
    
    computed_new_train_matrix = new_train_matrix.copy()
    
    for item_j in range(train_matrix_shape[1]):
        user_idx = train_matrix_shape[0]-1
        computed_new_train_matrix[user_idx][item_j] = predict_rating(user_idx, item_j, new_train_matrix, item_similarity)
    return computed_new_train_matrix
    
animes_data_path = "preprocessed_animes.csv"
profiles_data_path = "preprocessed_profiles.csv"
reviews_data_path = "preprocessed_reviews.csv"
animes_data = pd.read_csv(animes_data_path)
profiles_data = pd.read_csv(profiles_data_path)
reviews_data = pd.read_csv(reviews_data_path, engine='python', sep=',', error_bad_lines=False)

import pickle 
vit_50_dict = {}
with open("dict_1000_vit.pickle", "rb") as f:
  vit_50_dict = pickle.load(f)

q1 = reviews_data['anime_uid'].unique().tolist()
q2 = [x for x in vit_50_dict.keys()]
s1 = set(q1)
s2 = set(q2)
s3 = s1-s2 
l3 = list(s3)

animes = animes_data.copy()
profiles = profiles_data.copy()
reviews = reviews_data.copy()
for x in l3:
  reviews = reviews[reviews['anime_uid'] != x]

np.random.seed(0)
unique_AnimeID = reviews['anime_uid'].unique()
unique_users = reviews['profile'].unique()
j = 0
user_old2new_id_dict = dict()
for u in unique_users:
    user_old2new_id_dict[u] = j
    j += 1
j = 0
movie_old2new_id_dict = dict()
for i in unique_AnimeID:
    movie_old2new_id_dict[i] = j
    j += 1

with open("user_old2new_id_dict.pickle", "wb") as f:
    pickle.dump(user_old2new_id_dict, f)
with open("anime_old2new_id_dict.pickle", "wb") as f:
    pickle.dump(movie_old2new_id_dict, f)


# Then, use the generated dictionaries to reindex UserID and MovieID in the data_df
user_list = reviews['profile'].values
movie_list = reviews['anime_uid'].values
for j in range(len(reviews)):
    user_list[j] = user_old2new_id_dict[user_list[j]]
    movie_list[j] = movie_old2new_id_dict[movie_list[j]]
reviews['profile'] = user_list
reviews['anime_uid'] = movie_list

# generate train_df with 70% samples and test_df with 30% samples, and there should have no overlap between them.
train_index = np.random.random(len(reviews)) <= 0.7

# generate train_mat and test_mat
num_user = len(reviews['profile'].unique())
num_items = len(reviews['anime_uid'].unique())

saved_mat = coo_matrix((reviews['score'].values, (reviews['profile'].values, reviews['anime_uid'].values)), shape=(num_user, num_items)).astype(float).toarray()
with open("saved_train_data.pickle", "wb") as f:
    pickle.dump(saved_mat, f)

user_rating0 = np.zeros(saved_mat[123].shape)
user_rating0[121] = 9
user_rating0[721] = 5
user_rating0[521] = 2
item_similarity = np.zeros(saved_mat.shape)
start_time = time.time()

content_item_item_cf(saved_mat, item_similarity, user_rating0)
end_time = time.time()

print("Time Needed=" + str(end_time-start_time))
