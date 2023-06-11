import flask
import csv
from flask import Flask, render_template, request
import difflib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import requests

app = flask.Flask(__name__, template_folder='templates')

df_movies_full = pd.read_csv('movies.csv')
df_ratings_full = pd.read_csv('ratings.csv')
df_links = pd.read_csv('links.csv')
df_movies_links = pd.read_csv("movies_links_img_desc.csv")
df_movies_ratings = pd.merge(df_movies_full, df_movies_links, on="movieId")
rating = df_ratings_full.groupby('movieId').mean("rating")["rating"]
count_movie = df_ratings_full.groupby('movieId').count()
movie_merge = pd.merge(df_movies_full, rating, on="movieId")
movie_merge = pd.merge(movie_merge, count_movie['userId'], on="movieId")
movie_merge.rename(columns={
  "rating": "average_rating",
  "userId": "number_user"
},
                   inplace=True)
movie_merge['average_rating'] = np.round(movie_merge['average_rating'], 1)

df_movies = df_movies_full.head(1000)
df_ratings = df_ratings_full.head(1000)

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
ratings_matrix = df_ratings.pivot(index='userId',
                                  columns='movieId',
                                  values='rating').fillna(0)

# Calcul de la similarité entre les utilisateurs
user_similarity = cosine_similarity(ratings_matrix)


def get_similar_users(user_id):
  similar_users = []
  user_index = user_id - 1  # Conversion de l'ID utilisateur en index (commence à 0)
  user_scores = list(enumerate(user_similarity[user_index]))
  user_scores = sorted(user_scores, key=lambda x: x[1], reverse=True)
  similar_users = [score[0] + 1 for score in user_scores
                   ]  # Conversion de l'index en ID utilisateur
  return similar_users


def get_recommendations_collaborative(user_id, title):
  similar_users = get_similar_users(user_id)
  recommended_movies = []
  watched_movies = set(df_ratings[df_ratings['userId'] == user_id]['movieId'])

  for user in similar_users:
    movie_list = set(
      df_ratings[df_ratings['userId'] == user]['movieId']) - watched_movies
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
  filtered['wr'] = (v / (v + m) * R) + (m / (m + v) * C)
  filtered.sort_values(by=['wr'], inplace=True, ascending=False)
  popular_movies = filtered.head(10)['title']
  popular_action = filtered[filtered['genres'].apply(
    lambda x: "Action" in x)].head(10)['title']
  popular_adventure = filtered[filtered['genres'].apply(
    lambda x: 'Adventure' in x)].head(10)['title']
  popular_comedy = filtered[filtered['genres'].apply(
    lambda x: 'Comedy' in x)].head(10)['title']

  return dict({
    "all_movie": list(popular_movies),
    "action": list(popular_action),
    "adventure": list(popular_adventure),
    "comedy": list(popular_comedy)
  })


@app.route("/")
def home():
  recommended_movies_popularity = get_recommendations_popularity()['all_movie']
  recommended_popularity_comedy = get_recommendations_popularity()['comedy']
  recommended_popularity_adventure = get_recommendations_popularity(
  )['adventure']
  recommended_popularity_action = get_recommendations_popularity()['action']
  return render_template(
    'home.html',
    recommended_movies_popularity=recommended_movies_popularity,
    recommended_popularity_comedy=recommended_popularity_comedy,
    recommended_popularity_action=recommended_popularity_action,
    recommended_popularity_adventure=recommended_popularity_adventure,
    df_movies_links=df_movies_links,
    movie_merge=movie_merge)


@app.route("/page_identification_based_item")
def page_identification_based_item():
  return render_template('page_id_recommender_based_item.html')


@app.route("/recommendations_based_item", methods=['GET', 'POST'])
def recommendations_item():
  if request.method == 'POST':
    movie_title = request.form['movie_title']
    recommended_movies_genres = list(get_recommendations_genres(movie_title))

    return render_template('recommender_based_item.html',
                           movie_title=movie_title,
                           df_movies_links=df_movies_links,
                           movie_merge=movie_merge,
                           recommended_movies_genres=recommended_movies_genres)


@app.route("/page_identification_based_users")
def page_identification_based_users():
  return render_template('page_id_recommender_based_users.html')


@app.route("/recommendations_based_users", methods=['GET', 'POST'])
def recommendations_users():
  if request.method == 'POST':
    user_id = int(request.form['user_id'])
    interv = zip(
      get_recommendations_collaborative(user_id=user_id, title=True),
      get_recommendations_collaborative(user_id=user_id, title=False))

    return render_template('recommender_based_user.html',
                           user_id=user_id,
                           df_movies_links=df_movies_links,
                           interv=interv,
                          movie_merge=movie_merge)


@app.route("/recommender_based_popularity", methods=['GET', 'POST'])
def recommendations_popularity():
  if request.method == 'POST':
    user_id = int(request.form['user_id'])
    movie_title = request.form['movie_title']
    recommended_movies_popularity = get_recommendations_popularity()

    return render_template(
      'recommender_based_popularity.html',
      user_id=user_id,
      movie_title=movie_title,
      recommended_movies_popularity=recommended_movies_popularity,
      df_movies_links=df_movies_links)


if __name__ == '__main__':
  app.run(host="0.0.0.0", debug=True)
