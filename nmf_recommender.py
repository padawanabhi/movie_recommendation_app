import pickle
import pandas as pd
import numpy as np

columns = pickle.load(open('./columns.sav', 'rb'))

Q_matrix = pd.read_csv('./data/Q_matrix.csv', index_col=0)

IMDB_BASE_LINK = 'https://www.imdb.com/title/tt'


def recommend_movies(rating_dict, model_name):

    user_ratings = pd.DataFrame(rating_dict, columns=columns, index=[0])
    user_original = user_ratings.copy()

    model = pickle.load(open(f'./models/{model_name}', 'rb'))

    fill_value = float(user_ratings.mean(axis=1))

    user_ratings = user_ratings.fillna(fill_value)

    P_user = model.transform(user_ratings)

    R_user = np.dot(P_user, Q_matrix)

    user_df = pd.DataFrame(R_user, columns=columns)

    boolean_mark = user_original.isna()

    unrated_movies_df = user_df[boolean_mark]


    recommendations = unrated_movies_df.T.sort_values(by=0, ascending=False)

    return list(recommendations.index[:10])



def get_movie_id(movie_name):
    movie_df = pd.read_csv('./data/movies.csv', index_col=0)

    movie_id = movie_df[movie_df['title'] == movie_name].index[0]

    return str(movie_id)




def get_movie_name(movie_id):
    movie_df = pd.read_csv('./data/movies.csv', index_col=0)

    movie_name = movie_df[movie_df.index == int(movie_id)]['title'].values[0]

    return movie_name




def get_movie_link(movie_id):
    link_df = pd.read_csv('./data/links.csv', index_col=0)

    imdb_id = link_df[link_df.index == int(movie_id)]['imdbId'].values[0]
    if len(str(imdb_id)) == 7:
        imdb_link = IMDB_BASE_LINK + str(imdb_id) + '/'
    else:
        buffer = ['0' for _ in range(7 - len(str(imdb_id)))]
        buffer = "".join(buffer)
        imdb_link = IMDB_BASE_LINK + buffer + str(imdb_id) + '/'

    return imdb_link
