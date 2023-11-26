import os
import zipfile
import requests
import pandas as pd


class Utils:
    def __init__(self):
        self.__folder = 'ml-1m'
        self.__zip_file = self.__folder + '.zip'
        self.__current_dir = os.path.dirname(os.path.abspath(__file__))
        self.__file_path = os.path.join(self.__current_dir, self.__zip_file)
        self.__dataset_path = os.path.join(self.__current_dir, self.__folder)
        self.__link = 'https://files.grouplens.org/datasets/movielens/ml-1m.zip'

    def __download_ml_1m_dataset(self):
        if os.path.exists(self.__zip_file):
            print("Dataset already exists")
        else:
            response = requests.get(self.__link)
            response.raise_for_status()
            if response.ok:
                print("Successfully downloaded the dataset")
            print("Writing to a file in the directory")
            with open(self.__file_path, 'wb') as file:
                file.write(response.content)

    def __unzip_dataset(self):
        if os.path.exists(self.__zip_file) and zipfile.is_zipfile(self.__zip_file):
            with zipfile.ZipFile(self.__zip_file, 'r') as f:
                f.extractall('.')

    def get_dataset_path(self):
        if os.path.exists(self.__dataset_path):
            return self.__dataset_path
        else:
            print("Dataset does not exist")

    def download_and_unzip_dataset(self):
        self.__download_ml_1m_dataset()
        self.__unzip_dataset()

    def __read_tables(self):
        headings = [['user_id', 'gender', 'age', 'occupation', 'zip'],
                    ['user_id', 'movie_id', 'rating', 'timestamp'],
                    ['movie_id', 'title', 'genres']]

        users = pd.read_table(os.path.join(self.__dataset_path, 'users.dat'), sep='::',
                              header=None, names=headings[0], engine='python')
        ratings = pd.read_table(os.path.join(self.__dataset_path, 'ratings.dat'), sep='::',
                                header=None, names=headings[1], engine='python')
        movies = pd.read_table(os.path.join(self.__dataset_path, 'movies.dat'), sep='::',
                               header=None, names=headings[2], engine='python', encoding='ISO-8859-1')

        return users, ratings, movies

    def prepare(self):
        self.__download_ml_1m_dataset()
        self.__unzip_dataset()
        return self.__read_tables()
