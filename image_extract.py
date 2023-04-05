import PIL
from PIL import Image
import io
import os
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from pdb import set_trace as bp


animes_data_path = "preprocessed_animes.csv"
profiles_data_path = "preprocessed_profiles.csv"
reviews_data_path = "preprocessed_reviews.csv"




animes = pd.read_csv(animes_data_path)
profiles = pd.read_csv(profiles_data_path)
reviews = pd.read_csv(reviews_data_path, engine='python', sep=',', error_bad_lines=False)

anime_title_names = animes["title"].to_numpy()
anime_title_names = np.array([x.replace(" ", "_") for x in anime_title_names])
img_urls = animes["img_url"].to_numpy()
with open('anime_title_names.pickle', 'wb') as handle:
    pickle.dump(anime_title_names, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('img_urls.pickle', 'wb') as handle:
    pickle.dump(img_urls, handle, protocol=pickle.HIGHEST_PROTOCOL)



failed_titles = []
no_urls = []
for i in range(len(anime_title_names)):
  url = img_urls[i]
  title = anime_title_names[i]
  title = title.replace("/", "__") 
  if not isinstance(url, str):
    print(f"Index = {i} and Url not there for Title: {title}")
    no_urls.append(i)
    continue
  title = title+ "." + url.split(".")[-1]
  if(i%500==0):
    print(i)
  try:
    PIL.Image.open(io.BytesIO(urllib.request.urlopen(url).read())).save(title)
  except:
    failed_titles.append(i)
    print(f"Failed at Index = {i} for Title: {title}")




print("failed_titles\n")
print(failed_titles)

print("\n")

print("no_urls\n")
print(no_urls)
