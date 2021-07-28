# Libraries

import pandas as pd
import numpy as np
from itertools import groupby
import random
from tqdm import tqdm


def get_sample_rating_data(df_rating):
    user_movies = df_rating.groupby("userId")["movieId"].count().reset_index()
    freq = {key: len(list(group)) for key, group in groupby(np.sort(user_movies.movieId))}
    df_movie_freq = pd.DataFrame([freq]).transpose().reset_index()
    df_movie_freq.columns = ["movies", "no_of_users"]

    return user_movies, df_movie_freq


def get_data_batch(df_rating, batch_no, sample_percentage):
    random.seed(42)
    user_movies, df_movie_freq = get_sample_rating_data(df_rating)

    batch_user_list = []
    for i in range(batch_no):
        print(len(batch_user_list))
        df_random_user_rating = pd.DataFrame()
        user_movies = user_movies[~user_movies.userId.isin(batch_user_list)]
        print("start of sample creation")
        for j, movie_cnt in tqdm(enumerate(df_movie_freq.movies.unique())):
            user_numbers = int(round(df_movie_freq.no_of_users[j] * sample_percentage, 0))
            user_list = list(user_movies[user_movies.movieId == movie_cnt]["userId"])
            random_user_list = random.sample(user_list, user_numbers)
            df_random_rating = df_rating[df_rating.userId.isin(random_user_list)]
            df_random_user_rating = df_random_user_rating.append(df_random_rating)

        selected_users = list(df_random_user_rating.userId.unique())
        df_random_user_rating.to_pickle("archive/df_random_rating_" + str(i) + ".pickle")
        batch_user_list.extend(selected_users)


def get_mapped_tag_relevance_score(df_tag, df_genome_tags, df_genome_scores):
    df_tag_relevance = pd.merge(pd.merge(df_tag, df_genome_tags, on="tag"), df_genome_scores, on=["movieId", "tagId"])
    return df_tag_relevance



