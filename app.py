import flask
import csv
from flask import Flask, render_template, request
import difflib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

app = flask.Flask(__name__, template_folder='templates')

df_movies = pd.read_csv('movies.csv')
df_ratings = pd.read_csv('ratings.csv')

rating = df_ratings.groupby('movieId').mean("rating")["rating"]
count_movie = df_ratings.groupby('movieId').count()
movie_merge = pd.merge(
  df_movies,
  rating, 
  on="movieId"
)
movie_merge = pd.merge(
  movie_merge,
  count_movie['userId'],
  on="movieId"
)
movie_merge.rename(
  columns={"rating":"average_rating", "userId":"number_user"}, 
  inplace = True
)
df_movies = df_movies.head(1000)
df_ratings = df_ratings.head(1000)

# filtrage collaboratif item-based: Méthode basée sur le contenu (pas besoin de données sur les utilisateurs)

count = CountVectorizer(stop_words='english')
genres_matrix = count.fit_transform(df_movies['genres'])

cosine_sim_genres = cosine_similarity(genres_matrix, genres_matrix)

def get_recommendations_genres(movie_title):
    movie_index = df_movies[df_movies['title'] == movie_title].index[0]
    similar_movies = list(enumerate(cosine_sim_genres[movie_index]))
    similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)
    similar_movies = similar_movies[1:6]
    similar_movie_indices = [movie[0] for movie in similar_movies]
    similar_movie_titles = df_movies.iloc[similar_movie_indices]['title']
    return similar_movie_titles


# filtrage collaboratif user-based: Méthode de filtrage collaboratif

# Construction de la matrice de notation utilisateur-item
ratings_matrix = df_ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Calcul de la similarité entre les utilisateurs
user_similarity = cosine_similarity(ratings_matrix)

def get_similar_users(user_id):
    similar_users = []
    user_index = user_id - 1  # Conversion de l'ID utilisateur en index (commence à 0)
    user_scores = list(enumerate(user_similarity[user_index]))
    user_scores = sorted(user_scores, key=lambda x: x[1], reverse=True)
    similar_users = [score[0] + 1 for score in user_scores]  # Conversion de l'index en ID utilisateur
    return similar_users

def get_recommendations_collaborative(user_id, title):
    similar_users = get_similar_users(user_id)
    recommended_movies = []
    watched_movies = set(df_ratings[df_ratings['userId'] == user_id]['movieId'])
    
    for user in similar_users:
        movie_list = set(df_ratings[df_ratings['userId'] == user]['movieId']) - watched_movies
        recommended_movies.extend(movie_list)
        
        if len(recommended_movies) >= 10:
            break
    
    recommended_movies = list(recommended_movies)[:10]
    df = df_movies.iloc[recommended_movies]
    if title:
      return df['title']
    else: 
      return df['genres']

# Méthode basée sur la popularité de films. Nous allons utilisé l'indice # suivant WR=(v/v+m)R+(m/v+m)C pour classer les films
def get_recommendations_popularity():
    C = movie_merge["average_rating"].mean()
    m = movie_merge["number_user"].quantile(0.95)
    v = movie_merge["number_user"]
    R = movie_merge["average_rating"]
    filtered = movie_merge[movie_merge["number_user"] >= m]
    # created another column to indicate this evaluation
    filtered['wr'] =  (v/(v+m) * R) + (m/(m+v) * C)
    filtered.sort_values(by=['wr'], inplace= True, ascending=False)
    popular_movies = filtered.head(10)['title']
    return list(popular_movies)


@app.route("/")
def index():
    return render_template('index.html')
  

@app.route("/recommendations", methods=['GET', 'POST'])
def recommendations():
    if request.method == 'POST':
        user_id = int(request.form['user_id'])
        movie_title = request.form['movie_title']
        recommended_movies_popularity = get_recommendations_popularity()
        recommended_movies_genres = get_recommendations_genres(movie_title)
        interv = zip(get_recommendations_collaborative(user_id = user_id, title = True), get_recommendations_collaborative(user_id = user_id,title =False))
        
        return render_template('recommendations.html',
                               user_id=user_id,
                               movie_title=movie_title,
                               df_movies = df_movies,
                               movie_merge = movie_merge,
                               recommended_movies_popularity=recommended_movies_popularity,
                               recommended_movies_genres=recommended_movies_genres,
                              interv = interv)
    




if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
