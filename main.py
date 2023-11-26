from utils import Utils
# https://www.kaggle.com/code/christine12/movielens-1m-dataset-python-pandas
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# from sklearn.neighbors import KNeighborsClassifier

# np.corrcoef pearson

u = Utils()
users, ratings, movies = u.prepare()
print(users.head())
print(ratings.head())
print(movies.head())
