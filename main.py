import multiprocessing

from utils import Utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.cluster import DBSCAN
from scipy.sparse import csr_matrix
from catboost import CatBoostRegressor
import torch

u = Utils()
users, ratings, movies = u.prepare()
print(users.head())
print(ratings.head())
print(movies.head())

user_item_matrix = ratings.pivot(
    index='movie_id',
    columns='user_id',
    values='rating'
).fillna(0)

sparse_matrix = csr_matrix(user_item_matrix.values)

dbscan = DBSCAN(eps=0.5, min_samples=5, metric='euclidean', n_jobs=-1)
user_item_matrix_np = sparse_matrix.toarray()
clusters = dbscan.fit_predict(user_item_matrix_np.T)

train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=7)

model_catboost = CatBoostRegressor(
    iterations=700,
    learning_rate=0.05,
    depth=3,
    random_seed=7,
    thread_count=multiprocessing.cpu_count(),
    bootstrap_type='Bernoulli',
    subsample=0.7,
    colsample_bylevel=0.7,
    # task_type='GPU' if torch.cuda.is_available() else 'CPU'
)

model_catboost.fit(
    train_data[['user_id', 'movie_id']],
    train_data['rating'],
    verbose=False
)

predictions = model_catboost.predict(test_data[['user_id', 'movie_id']])
rmse = mean_squared_error(test_data['rating'], predictions, squared=False)
print(f'RMSE: {rmse}')
