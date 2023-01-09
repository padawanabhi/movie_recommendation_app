from flask import Flask, render_template, request
from nmf_recommender import recommend_movies, get_movie_id, get_movie_name, get_movie_link
import random
from similarity_recommender import COLUMNS, calculate_similarity, get_most_watched, get_high_rated

app = Flask(__name__)


@app.route('/')
def main_page():
    film_names = [random.choice(COLUMNS) for i in range(5)]
    film_ids = [get_movie_id(movie) for movie in film_names]
    return render_template('index.html', title='Recommendation - Home', film_list=zip(film_names, film_ids))


@app.route('/recommendations')
def make_recommendations():

    watched = get_most_watched()

    rated = get_high_rated()
    
    watched_links =[ get_movie_link(get_movie_id(recommendation))  for recommendation in watched]

    rated_links =[ get_movie_link(get_movie_id(recommendation))  for recommendation in rated]
    
    try:
        user_input_ratings = dict(request.args)
        new_dict = {}
        for key, value in user_input_ratings.items():
            new_key = get_movie_name(key)
            new_dict[new_key] = float(value)
        
        filter_factor = recommend_movies(new_dict, 'nmf_users_filtered.sav')

        similar = calculate_similarity(new_dict, COLUMNS)

        nmf_links =[ get_movie_link(get_movie_id(recommendation))  for recommendation in filter_factor]
        
        sim_links =[ get_movie_link(get_movie_id(recommendation))  for recommendation in similar]

        return render_template('recommendations.html', title="Recommendations - Results", 
                                films=zip(filter_factor, nmf_links), 
                                movies = zip(similar, sim_links), 
                                popular=zip(watched, watched_links),
                                high_rated = zip(rated, rated_links))
    except:
        return render_template('base_recommendations.html', 
                                popular=zip(watched, watched_links),
                                high_rated = zip(rated, rated_links))


@app.route('/categories')
def get_categories():
    return render_template('categories.html', title="Categories")


if __name__ == '__main__':
    app.run(debug=True, port=5000)
