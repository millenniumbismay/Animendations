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
      genres[genre] = "/static/zipped_index_based_images/index_based_images/" + genres[genre].strip() + ".jpg"
  return genres[0:5]

@app.route('/genres', methods=['POST'])
def genres():
    genre_prefs = json.loads(request.form['jsonval'])
    genre1 = genre_prefs['genre1']
    genre2 = genre_prefs['genre2']
    genre3 = genre_prefs['genre3']

    genres_selected = {genre1: [], genre2: [], genre3: []}


    popularity_data = pd.read_csv('popular_animes_by_genre.csv')
    genre_list = []
    imgurl_list = []

    for idx in range(len(popularity_data)):
        if popularity_data['genre'][idx] in genres_selected:
           imgurl_list.append(clean_genres(popularity_data['Popular Animes'][idx]))
           genre_list.append(popularity_data['genre'][idx])
    
    print(genre_list)
    return render_template('popular_animes.html', genre=genre_list, imglink=imgurl_list) 

if __name__ == '__main__':
    app.run(debug=True)