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