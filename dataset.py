from enum import Enum


class Dataset(Enum):
    movielens_100k = 'ml-100k', 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
    movielens_1m = 'ml-1m', 'https://files.grouplens.org/datasets/movielens/ml-1m.zip'
