import os
import zipfile
import requests
import pandas as pd
from dataset import Dataset


class Utils:
    def __init__(self, dataset=Dataset.movielens_1m):
        self.__dataset = dataset
        self.__folder, self.__link = self.__dataset.value
        self.__prepare_folders(self.__folder, self.__link)

    def __prepare_folders(self, folder, link):
        self.__folder = folder
        self.__zip_file = self.__folder + '.zip'
        self.__current_dir = os.path.dirname(os.path.abspath(__file__))
        self.__file_path = os.path.join(self.__current_dir, self.__zip_file)
        self.__dataset_path = os.path.join(self.__current_dir, self.__folder)
        self.__link = link


    def __download_ml_dataset(self):
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

    def __read_tables(self):
        users_heading = ['user_id', 'age', 'gender', 'occupation', 'zip']
        ratings_heading = ['user_id', 'movie_id', 'rating', 'timestamp']
        movies_heading = ['movie_id', 'title', 'genres']

        if self.__folder == 'ml-100k':
            users = pd.read_table(os.path.join(self.__dataset_path, 'u.user'), sep='|',
                                  header=None, names=users_heading, engine='python')

            ratings = pd.read_table(os.path.join(self.__dataset_path, 'u.data'), sep='\t',
                                    header=None, names=ratings_heading, engine='python')

            movies = pd.read_table(os.path.join(self.__dataset_path, 'u.item'), sep='|',
                                   header=None, names=movies_heading, usecols=[0, 1, 2], engine='python',
                                   encoding='ISO-8859-1')
        else:
            users = pd.read_table(os.path.join(self.__dataset_path, 'users.dat'), sep='::',
                                  header=None, names=users_heading, engine='python')

            ratings = pd.read_table(os.path.join(self.__dataset_path, 'ratings.dat'), sep='::',
                                    header=None, names=ratings_heading, engine='python')

            movies = pd.read_table(os.path.join(self.__dataset_path, 'movies.dat'), sep='::',
                                   header=None, names=movies_heading, engine='python', encoding='ISO-8859-1')

        return users, ratings, movies

    def prepare(self):
        self.__download_ml_dataset()
        self.__unzip_dataset()
        return self.__read_tables()
