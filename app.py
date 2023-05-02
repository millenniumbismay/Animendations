from flask import Flask, render_template, url_for, request
import pandas as pd
import jsonify
import pickle
import json
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app=Flask(__name__)

def predict_rating(item_index, user_rating, item_similarity, k=20):
    numerator = 0
    denominator = 0

    simm_items = np.argsort(item_similarity[item_index])[::-1].tolist()
    cnt_k = 0
    for i in simm_items:
        if i != item_index and user_rating[0][i] != 0:
            cnt_k = cnt_k + 1
            numerator += user_rating[0][i] * item_similarity[item_index][i]
            denominator += item_similarity[item_index][i]
        if cnt_k==k:
          break
    if denominator == 0:
        return 0
    else:
        return numerator / denominator

def compute_content_based_similarity(user_rating, item_similarity):
    count = 0
    topitems = {}
    rated_items = np.where(user_rating != 0)[0]
    for uid in range(len(item_similarity)):
        if count not in rated_items:
            topitems[uid] = predict_rating(count, user_rating, item_similarity, 2)
        count += 1
    top_items = sorted(topitems.items(), key=lambda x: x[1], reverse=True)[0:5]
    top_items = [x[0] for x in top_items]
    return top_items

def compute_less_MF(P_User, Q, user_rating, num_epochs=5):
  latent_factors = 30

  a = 0.01
  b = 0.1 
  train_loss = []
  test_loss = []
  new_user_p = P_User
  nonzero_indices_train = np.nonzero(user_rating) 

  for epoch in range(num_epochs):
      for j in range(Q.shape[1]):
          if user_rating[0][j] > 0:
              diff = user_rating [0][j] - np.dot(new_user_p, Q[:, j])
              new_user_p += a * (diff * Q[:, j] - b * new_user_p)
              
  predicted_matrix = np.dot(new_user_p, Q)

  return predicted_matrix, new_user_p
     

@app.route('/')
def index():
    return render_template('index.html')

def clean_genres(genres):
  """
  Coinverts the str(list) -> list format
  """
  genres = genres[1:-1].strip().split(',')
  for genre in range(5):
      genres[genre] = "/static/zipped_index_based_images/index_based_images/" + genres[genre].strip() + ".jpg"
  return genres[0:5]

def gettitlefromgenre(genres):
    genres = genres[1:-1].strip().split(',')
    #app.logger.warning(genres)
    #app.logger.warning(type(genres))
    anime_data = pd.read_csv('new_preprocessed_animes.csv')
    print(anime_data.head())
    for genre in range(5):
        #app.logger.warning(int(genres[genre][1:-1]))
        data = anime_data[anime_data['uid']==int(genres[genre].replace(" ", ""))]
        #app.logger.warning(data)
        result2 = data['title'].values
        #print(result2)
        #app.logger.warning(type(result2))
        #app.logger.warning(result2[0])
        genres[genre] = result2[0]
        #app.logger.warning(genres[0:5])
    return genres[0:5]


@app.route('/search')
def search():
    query = request.args.get('query')
    #app.logger.warning(query)
    popularity_data = pd.read_csv('popular_animes_by_genre.csv')
    genre_list = []
    imgurl_list = []
    title_list = []
    #query = request.args.get('query')
    #app.logger.warning(query)

    anime_data = pd.read_csv('new_preprocessed_animes.csv')
    anime_data['new_title'] = anime_data.apply(lambda x: x['title'].lower(), axis = 1)
    val = ""

    # anime_data['new_title'] = anime_data.apply(lambda x: x['title'].lower(), axis = 1)
    # df_new = anime_data[anime_data['title'].str.contains(query.lower())]
    
    df_new = anime_data[anime_data['new_title'].str.contains(query.lower())]
    vals = df_new['title'].values
    anime_ids = df_new['uid'].values
    genres = df_new['genre'].values
    
    print(type(genres))
    # l1 = []
    # for val in vals:
    #     l1.append(val)
    # if len(l1) != 0:
    #     title_list.append(l1)
    

    l1 = []
    genres = np.unique(genres)
    for val in genres:
        #print(type(val))
        genre_list.append(val[2:-2].strip())
        l2 = []
        l3 = []
        temp = df_new[df_new['genre']==val]
        temp1 = temp['title'].values
        for vals in temp1:
            l2.append(vals)
        title_list.append(l2)
        temp2 = temp['uid'].values
        for vals in temp2:
            l3.append("../static/zipped_index_based_images/index_based_images/" + str(vals) + ".jpg")
        imgurl_list.append(l3)

    app.logger.warning(genre_list)
    app.logger.warning(imgurl_list)
    app.logger.warning(title_list)
    return render_template('popular_animes.html', genre=genre_list, imglink=imgurl_list, titles=title_list) 

@app.route('/genres', methods=['POST'])
def genres():
    genres_selected = dict()
    genre_prefs = json.loads(request.form['jsonval'])
    i = 0
    for temp in genre_prefs:
        genres_selected[genre_prefs['genre' + str(i+1)]] = []
        i = i + 1

    popularity_data = pd.read_csv('popular_animes_by_genre.csv')
    genre_list = []
    imgurl_list = []
    title_list = []

    for idx in range(len(popularity_data)):
        if popularity_data['genre'][idx] in genres_selected:
           imgurl_list.append(clean_genres(popularity_data['Popular Animes'][idx]))
           genre_list.append(popularity_data['genre'][idx])
           title_list.append(gettitlefromgenre(popularity_data['Popular Animes'][idx]))
    
    print(genre_list)
    return render_template('popular_animes.html', genre=genre_list, imglink=imgurl_list, titles=title_list) 


@app.route('/recommendations', methods=['POST'])
def recommendations():
    ratings = json.loads(request.form['jsonval'])
    Q = {}
    
    with open("utils/Q_MF_NCF.pickle", "rb") as f:
        Q = pickle.load(f)
        Q = Q.weight.data.numpy()

    vit_50_dict = {}
    with open("dict_1000_vit.pickle", "rb") as f:
        vit_50_dict = pickle.load(f)
    
    anime_data = pd.read_csv('new_preprocessed_animes.csv')
    
    unique_AnimeID = anime_data['uid'].unique().tolist()
    
    j = 0
    movie_old2new_id_dict = dict()
    movie_new2old_id_dict = dict()
    for i in unique_AnimeID:
        movie_old2new_id_dict[i] = j
        movie_new2old_id_dict[j] = i
        j += 1
    
    user_rating = np.zeros((1, len(unique_AnimeID)))
    num_items = len(unique_AnimeID)

    

    key_vit_50_dict = [x for x in vit_50_dict.keys()]
    embeddings = [[] for _ in range(num_items)]
    for i in key_vit_50_dict:
        if i in movie_old2new_id_dict:
            embeddings[movie_old2new_id_dict[i]] = vit_50_dict[i].reshape(-1) 
    
    for idx in range(len(embeddings)):
        if embeddings[idx] == []:
            embeddings[idx] = np.zeros((1, 1000)).reshape(-1)
  
   
    simm_item_ij = cosine_similarity(embeddings)
    
    
    for title in ratings.keys():
        uid = int(anime_data.loc[anime_data['title'] == title, 'uid'].item())
        user_rating[0][movie_old2new_id_dict[uid]] = ratings[title]
    
    P_User = np.zeros((1, 24)) 
    predicted_matrix, new_user_p = compute_less_MF(P_User, Q.T, user_rating)
    top_similar_animes = compute_content_based_similarity(user_rating, simm_item_ij)
    
    sorted_matrix = np.argsort(predicted_matrix, axis=None, kind='quicksort')[::-1]
    imgurl = [[], []]
    titles = [[], []]
    
    count = 0
    for uid in sorted_matrix:
        if user_rating[0][uid] == 0:
            if len(imgurl[0]) < 5:
                imgurl[0].append("/static/zipped_index_based_images/index_based_images/" + str(int(movie_new2old_id_dict[uid])) + ".jpg")
                titles[0].append(anime_data.loc[anime_data['uid'] == movie_new2old_id_dict[uid], 'title'].item())
            else:
                imgurl[1].append("/static/zipped_index_based_images/index_based_images/" + str(int(movie_new2old_id_dict[uid])) + ".jpg")
                titles[1].append(anime_data.loc[anime_data['uid'] == movie_new2old_id_dict[uid], 'title'].item())
            count += 1
        if count == 10:
            break
        
    img_content = []
    titles_content = []
    for uid in top_similar_animes:
        img_content.append("/static/zipped_index_based_images/index_based_images/" + str(int(movie_new2old_id_dict[uid])) + ".jpg")
        titles_content.append(anime_data.loc[anime_data['uid'] == movie_new2old_id_dict[uid], 'title'].item())

    return render_template('recommended_animes.html', imglink=imgurl, titles=titles, imgcontent=img_content, titlescontent=titles_content)
        

    


        
        



if __name__ == '__main__':
    app.run(debug=True)