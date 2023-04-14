# Animendations
#### Project for ISR(CSCE670) - Spring'23 - Texas A&M University 
Anime Recommendations using CBCF, VBPR and NLPBPR

## Pre-processing
1. Dropped duplicates values wrt uid in animes
2. Dropped duplicates values wrt profile in profiles
3. Dropped duplicates wrt profile and uid in reviews
4. Dropped the birthday column -> We are not using age (50% NaN)
5. Removing genres which are only Hentai
6. Removed all users with less than 3 ratings and corresponding reviews and any unique anime to them
7. Cleaned 'genre' in animes table
8. Cleaned 'favorites_anime' in profiles table
9. Cleaned 'scores' in reviews table
10. Removed unnecessary review texts and stored them in 'text' in reviews table

## Synopsis Embeddings
Synopsis Embeddings has been created to cater to multiple downstream tasks such as sentence similarity, compressing to use for NLPBPR, Model with CBCF to fill some missing values, etc. Following steps have been taken:

### Synopsis Pre-processing
1. Strips HTML Tags
2. remove accented characters from text, e.g. café
3. expand shortened words, e.g. don't to do not
4. Lemmatize words wrt POS tags
5. Generalizes the text by converting words to numbers whenever possible
6. Removed stopwords except 'no' and 'not'
7. Removes Symbols - ```"!@#$%^*_-+=|\;:<>?/.,\'`\"(){}[]"```

### Synopsis Embedding
1. Tried multiple models and finalized Huggingface ```all-distilroberta-v1``` model for generating sentence embeddings
2. Pre-processed synopsis truncated to ```512 tokens```
3. Output embedding is tensor of size ```(1, 768)```

### Training Autoencoder to compress synopsis embeddings
1. Trained an autoencoder network ```[768 > 256 > 128 > 64 > 50 > 64 > 128 > 256 > 768]```
```
Architecture: AE(
  (encoder): Sequential(
    (0): Linear(in_features=768, out_features=256, bias=True)
    (1): ReLU()
    (2): Linear(in_features=256, out_features=128, bias=True)
    (3): ReLU()
    (4): Linear(in_features=128, out_features=64, bias=True)
    (5): ReLU()
    (6): Linear(in_features=64, out_features=50, bias=True)
    (7): ReLU()
  )
  (decoder): Sequential(
    (0): Linear(in_features=50, out_features=64, bias=True)
    (1): ReLU()
    (2): Linear(in_features=64, out_features=128, bias=True)
    (3): ReLU()
    (4): Linear(in_features=128, out_features=256, bias=True)
    (5): ReLU()
    (6): Linear(in_features=256, out_features=768, bias=True)
    (7): Sigmoid()
  )
)
```
2. Optimizer: ```Adam(model.parameters(), lr = 0.1, weight_decay = 1e-5)```
3. Scheduler: ```ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)```
4. Number of epochs = ```50```
5. Final Training Loss: ```0.001113589```
6. Model, Embeddings and Compressed embeddings are pickled and present in ```synopsis_autoencoder``` folder. The embeddings are in dictionary format: ```sentence_embeddings_dict.pickle: {uid: [embedding tensor of 1x786]}``` and ```compressed_synopsis_embeddings.pickle: {uid: [embedding tensor of 50]}```
7. Name of Model: ```synopsis_autoencoder_50_epochs_Adam.pickle```

The model will be updated with better parameters and reported here if any changes are done
