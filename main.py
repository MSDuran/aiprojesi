import pandas as pd
import multiprocessing
from utils import Utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.cluster import DBSCAN
from scipy.sparse import csr_matrix
from catboost import CatBoostRegressor
from comparison import Comparison
from dataset import Dataset

ds = Dataset.movielens_1m

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

# Median Imputation (MD) method
user_item_matrix = user_item_matrix.apply(lambda x: x.fillna(x.median()), axis=0)

sparse_matrix = csr_matrix(user_item_matrix.values)

dbscan = DBSCAN(eps=0.5, min_samples=5, metric='euclidean', n_jobs=-1)
user_item_matrix_np = sparse_matrix.toarray()
clusters = dbscan.fit_predict(user_item_matrix_np.T)

user_cluster_ids = pd.Series(clusters, index=user_item_matrix.columns).reset_index()
user_cluster_ids.columns = ['user_id', 'cluster_id']

users = users.merge(user_cluster_ids, on='user_id')

ratings = ratings.merge(users[['user_id', 'cluster_id']], on='user_id')

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

model_catboost.fit(
    train_data[['user_id', 'movie_id', 'cluster_id']],
    train_data['rating'],
    verbose=False
)

predictions = model_catboost.predict(test_data[['user_id', 'movie_id', 'cluster_id']])
rmse = mean_squared_error(test_data['rating'], predictions, squared=False)
mae = mean_absolute_error(test_data['rating'], predictions)

print(f'RMSE: {rmse}')
print(f'MAE: {mae}')

c = Comparison(ds)
c.show_plot('my_method', mae, rmse, True)
