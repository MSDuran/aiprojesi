from utils import Utils
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import torch
import multiprocessing

# https://www.kaggle.com/code/christine12/movielens-1m-dataset-python-pandas
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# from sklearn.neighbors import KNeighborsClassifier
# file:///C:/Users/MSD/Desktop/ai%20papers/2021_Collaborative%20filtering%20recommendation%20algorithm.pdf
# np.corrcoef pearson

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

model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
model_knn.fit(sparse_matrix)


def predict_ratings(user_id, movie_id):
    distances, indices = model_knn.kneighbors(
        user_item_matrix.loc[movie_id].values.reshape(1, -1), n_neighbors=5)

    neighbor_ratings = user_item_matrix.iloc[indices[0]].loc[:, user_id]
    predicted_rating = neighbor_ratings.mean()
    return predicted_rating


train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=7)

dtrain = xgb.DMatrix(train_data[['user_id', 'movie_id']], label=train_data['rating'])
dtest = xgb.DMatrix(test_data[['user_id', 'movie_id']], label=test_data['rating'])

params = {
    'seed': 7,
    'learning_rate': 0.05,
    'max_depth': 3,
    'num_boost_round': 700,
    'eta': 0.3,
    'nthread': 4,
    'objective': 'reg:squarederror',
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'tree_method': 'hist'
}
if torch.cuda.is_available():
    params['device'] = 'cuda'
params['nthread'] = multiprocessing.cpu_count()

model_xgb = xgb.train(params, dtrain, num_boost_round=700)

predictions = model_xgb.predict(dtest)
rmse = mean_squared_error(test_data['rating'], predictions, squared=False)
print(f'RMSE: {rmse}')
