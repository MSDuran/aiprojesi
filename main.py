import pandas as pd
import multiprocessing
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.cluster import DBSCAN
from scipy.sparse import csr_matrix
from catboost import CatBoostRegressor
from comparison import Comparison
from dataset import Dataset
from utils import Utils
from joblib import Parallel, delayed

ds = Dataset.movielens_1m

u = Utils(ds)
users, ratings, movies = u.prepare()

# users = users.head(10000)
# ratings = ratings.head(10000)
# movies = movies.head(10000)

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

default_rating_value = ratings['rating'].mean()

bayesian_data = pd.merge(ratings, user_features, on='user_id', how='left')
bayesian_data = pd.merge(bayesian_data, movie_features, on='movie_id', how='left')
bayesian_data = bayesian_data[['user_avg_rating', 'movie_avg_rating', 'rating']]
# bayesian_data = bayesian_data.sample(frac=0.01, random_state=7)

bayesian_data['user_avg_rating'] = bayesian_data['user_avg_rating'].round()
bayesian_data['movie_avg_rating'] = bayesian_data['movie_avg_rating'].round()

model_structure = [('user_avg_rating', 'rating'), ('movie_avg_rating', 'rating')]
bayesian_model = BayesianNetwork(model_structure)
bayesian_model.fit(bayesian_data, estimator=MaximumLikelihoodEstimator)
inference = VariableElimination(bayesian_model)

def predict_rating(bn_inference, user_avg, movie_avg):
    user_avg_discrete = round(user_avg)
    movie_avg_discrete = round(movie_avg)
    try:
        query_result = bn_inference.query(variables=['rating'], evidence={'user_avg_rating': user_avg_discrete, 'movie_avg_rating': movie_avg_discrete})
        probabilities = query_result.values
        ratings = range(1, len(probabilities) + 1)
        expected_rating = sum(rating * prob for rating, prob in zip(ratings, probabilities))
        return expected_rating
    except KeyError:
        return default_rating_value

def update_rating(movie_id, user_id):
    if pd.isnull(user_item_matrix.at[movie_id, user_id]):
        user_avg = user_features.loc[user_features['user_id'] == user_id, 'user_avg_rating'].values[0]
        movie_avg = movie_features.loc[movie_features['movie_id'] == movie_id, 'movie_avg_rating'].values[0]
        predicted_rating = predict_rating(inference, user_avg, movie_avg)
        return movie_id, user_id, predicted_rating
    return None, None, None


tasks = [(movie_id, user_id) for movie_id in user_item_matrix.index for user_id in user_item_matrix.columns]
results = Parallel(n_jobs=-1)(delayed(update_rating)(movie_id, user_id) for movie_id, user_id in tasks)

for movie_id, user_id, predicted_rating in results:
    if predicted_rating is not None:
        user_item_matrix.at[movie_id, user_id] = predicted_rating

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

model_catboost.fit(
    train_data[['user_id', 'movie_id', 'cluster_id', 'user_avg_rating', 'user_rating_count', 'movie_avg_rating',
                'movie_rating_count']],
    train_data['rating'],
    verbose=False
)

predictions = model_catboost.predict(test_data[['user_id', 'movie_id', 'cluster_id', 'user_avg_rating', 'user_rating_count',
                                            'movie_avg_rating', 'movie_rating_count']])

rmse = mean_squared_error(test_data['rating'], predictions, squared=False)
mae = mean_absolute_error(test_data['rating'], predictions)

print(f'RMSE: {rmse}')
print(f'MAE: {mae}')

c = Comparison(ds)
c.show_plot('my_method', mae, rmse, True)
