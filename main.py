import pandas as pd
import multiprocessing
from utils import Utils
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.cluster import DBSCAN
from scipy.sparse import csr_matrix
from catboost import CatBoostRegressor
from comparison import Comparison
from dataset import Dataset

ds = Dataset.movielens_100k

u = Utils(ds)
users, ratings, movies = u.prepare()

print(users.head())
print(ratings.head())
print(movies.head())

user_item_matrix = ratings.pivot(
    index='movie_id',
    columns='user_id',
    values='rating'
)

user_features = ratings.groupby('user_id').agg({
    'rating': ['mean', 'count']
}).reset_index()
user_features.columns = ['user_id', 'user_avg_rating', 'user_rating_count']

movie_features = ratings.groupby('movie_id').agg({
    'rating': ['mean', 'count']
}).reset_index()
movie_features.columns = ['movie_id', 'movie_avg_rating', 'movie_rating_count']

# Median Imputation for User-Item Matrix
user_item_matrix = ratings.pivot(
    index='movie_id',
    columns='user_id',
    values='rating'
).apply(lambda x: x.fillna(x.median()), axis=0)

sparse_matrix = csr_matrix(user_item_matrix.values)

dbscan = DBSCAN(eps=0.5, min_samples=5, metric='euclidean', n_jobs=-1)

clusters = dbscan.fit_predict(sparse_matrix.toarray().T)
user_cluster_ids = pd.Series(clusters, index=user_item_matrix.columns).reset_index()
user_cluster_ids.columns = ['user_id', 'cluster_id']

users = users.merge(user_cluster_ids, on='user_id')
ratings = ratings.merge(users[['user_id', 'cluster_id']], on='user_id')
ratings = ratings.merge(user_features, on='user_id', how='left')
ratings = ratings.merge(movie_features, on='movie_id', how='left')

train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=7)

model_catboost = CatBoostRegressor(
    iterations=5000,
    learning_rate=0.01,
    depth=8,
    random_seed=7,
    thread_count=multiprocessing.cpu_count(),
    bootstrap_type='MVS',
    subsample=0.8,
    task_type='GPU',
    early_stopping_rounds=10,
    l2_leaf_reg=3
)

# fixme:: run grid search with more params to get better results.
# current best rmse= 0.916 mae= 0.721
param_grid = {
    'learning_rate': [0.01, 0.03, 0.05],
    'depth': [6, 8, 10],
    'l2_leaf_reg': [1, 3, 5]
}


grid_search = GridSearchCV(model_catboost, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(
    train_data[['user_id', 'movie_id', 'cluster_id', 'user_avg_rating', 'user_rating_count', 'movie_avg_rating',
                'movie_rating_count']],
    train_data['rating'],
    verbose=False
)

best_model = grid_search.best_estimator_
print(grid_search.best_params_)
predictions = best_model.predict(test_data[['user_id', 'movie_id', 'cluster_id', 'user_avg_rating', 'user_rating_count',
                                            'movie_avg_rating', 'movie_rating_count']])

rmse = mean_squared_error(test_data['rating'], predictions, squared=False)
mae = mean_absolute_error(test_data['rating'], predictions)

print(f'RMSE: {rmse}')
print(f'MAE: {mae}')

c = Comparison(ds)
c.show_plot('my_method', mae, rmse, True)
