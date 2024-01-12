import zipfile
import requests
import pandas as pd
from io import BytesIO


def download_movielens_100k():
    url = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
    response = requests.get(url)
    zip_content = BytesIO(response.content)

    with zipfile.ZipFile(zip_content, 'r') as zip_ref:
        zip_ref.extractall("movielens_100k")


def download_movielens_1m():
    url = 'https://files.grouplens.org/datasets/movielens/ml-1m.zip'
    response = requests.get(url)
    zip_content = BytesIO(response.content)

    with zipfile.ZipFile(zip_content, 'r') as zip_ref:
        zip_ref.extractall("movielens_1m")


download_movielens_100k()

users_columns = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
movies_columns = ['movie_id', 'movie_title', 'release_date', 'video_release_date',
                  'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
                  'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                  'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                  'Thriller', 'War', 'Western']
ratings_columns = ['user_id', 'movie_id', 'rating', 'timestamp']

users = pd.read_csv('movielens_100k/ml-100k/u.user', sep='|', names=users_columns, encoding='latin-1')
movies = pd.read_csv('movielens_100k/ml-100k/u.item', sep='|', names=movies_columns, encoding='latin-1',
                     usecols=range(24))
ratings = pd.read_csv('movielens_100k/ml-100k/u.data', sep='\t', names=ratings_columns, encoding='latin-1')

missing_users = users.isnull().sum().sum()
missing_ratings = ratings.isnull().sum().sum()
missing_movies = movies.isnull().sum().sum()

print(f"Movielens 100K Missing values in 'users': {missing_users}")
print(f"Movielens 100K Missing values in 'ratings': {missing_ratings}")
print(f"Movielens 100K Missing values in 'movies': {missing_movies}")
print(f"Movielens 100K Total missing values: {missing_users + missing_ratings + missing_movies}")

download_movielens_1m()

users_columns = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
movies_columns = ['movie_id', 'movie_title', 'release_date', 'video_release_date',
                  'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
                  'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                  'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                  'Thriller', 'War', 'Western']
ratings_columns = ['user_id', 'movie_id', 'rating', 'timestamp']

users = pd.read_csv('movielens_1m/ml-1m/users.dat', sep='::', engine='python',
                    names=['user_id', 'gender', 'age', 'occupation', 'zip'])
movies = pd.read_csv('movielens_1m/ml-1m/movies.dat', sep='::', engine='python', names=['movie_id', 'title', 'genres'],
                     encoding='ISO-8859-1')
ratings = pd.read_csv('movielens_1m/ml-1m/ratings.dat', sep='::', engine='python',
                      names=['user_id', 'movie_id', 'rating', 'timestamp'])

missing_users = users.isnull().sum().sum()
missing_ratings = ratings.isnull().sum().sum()
missing_movies = movies.isnull().sum().sum()
print()
print(f"Movielens 1M Missing values in 'users': {missing_users}")
print(f"Movielens 1M Missing values in 'ratings': {missing_ratings}")
print(f"Movielens 1M Missing values in 'movies': {missing_movies}")
print(f"Movielens 1M Total missing values: {missing_users + missing_ratings + missing_movies}")
