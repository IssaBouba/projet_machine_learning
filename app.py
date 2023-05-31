import flask
import csv
from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

movies_data = pd.read_csv('movies.csv')
movies_data = movies_data.head(2000)
count = CountVectorizer(stop_words='english')
genres_matrix = count.fit_transform(movies_data['genres'])

cosine_sim = cosine_similarity(genres_matrix, genres_matrix)


def get_recommendations(movie_title):
  movie_index = movies_data[movies_data['title'] == movie_title].index[0]
  similar_movies = list(enumerate(cosine_sim[movie_index]))
  similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)
  similar_movies = similar_movies[1:6]
  similar_movie_indices = [movie[0] for movie in similar_movies]
  similar_movie_titles = movies_data.iloc[similar_movie_indices][
    'title'].tolist()
  return similar_movie_titles


@app.route("/")
def index():
  return render_template('index.html')


@app.route("/recommendations", methods=['GET', 'POST'])
def recommendations():
  if request.method == 'POST':
    movie_title = request.form['movie_title']
    if movie_title in movies_data['title'].values:
      recommended_movies = get_recommendations(movie_title)
      movie_genre = movies_data
      return render_template('recommendations.html',
                             movie_genre = movie_genre,
                             movie_title=movie_title,
                             recommended_movies=recommended_movies)
   

if __name__ == '__main__':
  app.run(host="0.0.0.0", debug=True)
