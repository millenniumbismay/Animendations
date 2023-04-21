from flask import Flask, render_template, url_for, request
import pandas as pd
import jsonify
import json

app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def clean_genres(genres):
  """
  Coinverts the str(list) -> list format
  """
  genres = genres[1:-1].strip().split(',')
  for genre in range(5):
      genres[genre] = "/static/index_based_images/" + genres[genre].strip() + ".jpg"
  return genres[0:5]

def gettitlefromgenre(genres):
    genres = genres[1:-1].strip().split(',')
    #app.logger.warning(genres)
    #app.logger.warning(type(genres))
    anime_data = pd.read_csv('new_preprocessed_animes.csv')
    for genre in range(5):
        #app.logger.warning(int(genres[genre][1:-1]))
        data = anime_data[anime_data['uid']==int(genres[genre].replace(" ", ""))]
        #app.logger.warning(data)
        result2 = data['title'].values
        #print(result2)
        #app.logger.warning(type(result2))
        app.logger.warning(result2[0])
        genres[genre] = result2[0]
        #app.logger.warning(genres[0:5])
    return genres[0:5]


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

if __name__ == '__main__':
    app.run(debug=True)