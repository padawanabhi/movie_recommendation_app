import numpy as np
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

RATINGS_DF = pd.read_csv('./data/ratings_filtered_matrix.csv', index_col=0)

INDEX = RATINGS_DF.index.max()

COLUMNS = list(RATINGS_DF.columns)


def calculate_similarity(rating_dict, columns):
    user_ratings = pd.DataFrame(rating_dict, columns=columns, index=[INDEX+1])
    user_original = user_ratings.copy() 
    new_user_df = pd.concat([RATINGS_DF, user_ratings])
    new_user_df = new_user_df.fillna(0)
    cosine_table=pd.DataFrame(cosine_similarity(new_user_df), 
                                index=new_user_df.index,
                                columns=new_user_df.index
                            )
    
    user_ratings_transpose = new_user_df.T

    unseen_movies = list(user_ratings_transpose.index[user_ratings_transpose[INDEX+1] == 0])
    neighbours = list(cosine_table[INDEX].sort_values(ascending=False).index[:10])

    predicted_ratings_movies = []
    for movie in unseen_movies:
        
        #list people who watched the the unseen movies
        others = list(user_ratings_transpose.columns[user_ratings_transpose.loc[movie] > 0])
        numerator = 0
        denominator = 0.000001
        # go through users who are similar but watched the film
        for user in neighbours:
            if user in others:
            #  extract the ratings and similarities for similar users
                rating = user_ratings_transpose.loc[movie, user]
                similarity = cosine_table.loc[INDEX, user]
                
            # predict rating based on the (weighted)
            # averaged rating of the neighbours
            # sum(ratings)/no.users OR 
            # sum(ratings*similarity)/sum(similarities)
                numerator = numerator + rating * similarity
                denominator = denominator + similarity
        predicted_ratings = round(numerator/denominator, 1)
        predicted_ratings_movies.append([predicted_ratings, movie])

    predicted_ratings_movies.sort(key = lambda x: x[0], reverse=True)
    predicted_movies = []
    for prediction in predicted_ratings_movies[:10]:
        predicted_movies.append(prediction[1])

    return predicted_movies



def get_most_watched(dataframe=RATINGS_DF) -> list:
    return list(dataframe.notna().sum().sort_values(ascending=False).index[:10])


def get_high_rated(dataframe=RATINGS_DF) -> list:

    return list(dataframe.mean().sort_values(ascending=False).index[:10])