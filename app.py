from flask import Flask, render_template, request
from nmf_recommender import recommend_movies, get_movie_id, get_movie_name, get_movie_link
import random
from similarity_recommender import COLUMNS, calculate_similarity

app = Flask(__name__)


@app.route('/')
def main_page():
    film_names = [random.choice(COLUMNS) for i in range(5)]
    film_ids = [get_movie_id(movie) for movie in film_names]
    return render_template('index.html', title='Recommendation - Home', film_list=zip(film_names, film_ids))


@app.route('/recommendations')
def make_recommendations():
    user_input_ratings = dict(request.args)
    new_dict = {}
    for key, value in user_input_ratings.items():
        new_key = get_movie_name(key)
        new_dict[new_key] = float(value)
    
    #recommendations = recommend_movies(new_dict, 'nmf_moviemean.sav')
    
    recommendations = calculate_similarity(new_dict, COLUMNS)

    links =[ get_movie_link(get_movie_id(recommendation))  for recommendation in recommendations]

    return render_template('recommendations.html', title="Recommendations - Results", films=zip(recommendations, links))



if __name__ == '__main__':
    app.run(debug=True, port=5000)
