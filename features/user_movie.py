import pandas as pd
import numpy as np


class UserMovie:

    def __init__(self,
                 user_id,
                 movie_id,
                 timestamp,
                 rating,
                 rating_df,
                 movie_df,
                 tag_df):
        self.user_id = user_id
        self.movie_id = movie_id
        self.genre = movie_df[movie_df.movieId == self.movie_id]["genres"].values[0]
        self.timestamp = timestamp
        self.rating = rating
        self.peer_movie_rating = self.get_peer_movie_rating(rating_df)
        self.user_genre_rating = self.get_user_genre_rating(rating_df, movie_df)
        self.peer_genre_rating = self.get_peer_genre_rating(rating_df, movie_df)
        self.similar_movie_rating = self.get_similar_movie_rating(rating_df, tag_df)
        self.feature_dict = self.get_features_dict()

    def get_peer_movie_rating(self, rating_df):
        mean_rating = np.mean(rating_df[(rating_df.movieId == self.movie_id) &
                                        (rating_df.timestamp <= self.timestamp) &
                                        (rating_df.userId != self.user_id)]["rating"])

        return mean_rating

    def get_user_genre_rating(self, rating_df, movie_df):
        rating_df = rating_df[rating_df.userId == self.user_id].reset_index(drop=True)

        rating_df = pd.merge(rating_df, movie_df, on="movieId")

        mean_rating = np.mean(rating_df[(rating_df.genres == self.genre) & (rating_df.movieId != self.movie_id)
                                        & (rating_df.timestamp <= self.timestamp)]["rating"])
        return mean_rating

    def get_peer_genre_rating(self, rating_df, movie_df):
        mean_rating = np.mean(rating_df[(rating_df.movieId.isin(
            list(movie_df[movie_df.genres == self.genre]["movieId"])))
                                        & (rating_df.userId != self.user_id)
                                        & (rating_df.movieId != self.movie_id) & (
                                                rating_df.timestamp <= self.timestamp)]["rating"])

        return mean_rating

    def get_similar_movie_rating(self, rating_df, tag_df, tag_relevance_th=0.5, tag_match=4):
        # get the tags and their relevance for the movie by the user
        df_user_movie_tag = tag_df[(tag_df["movieId"] == self.movie_id)
                                   & (tag_df.timestamp <= self.timestamp)]

        df_user_movie_imp_tags = df_user_movie_tag[["tag", "relevance"]] \
            [df_user_movie_tag.relevance >= 0.5].drop_duplicates().reset_index(drop=True)

        # get the tags of other movies seen before the user saw the movie
        # check the movies which got similar tags
        df_movies_tags = tag_df[(tag_df["movieId"] != self.movie_id) & (tag_df.timestamp <= self.timestamp)
                                & (tag_df.tag.isin(list(df_user_movie_imp_tags["tag"])))]

        # get all movieids and their tags above a certain relevance by various users
        df_sim_movie_tags = df_movies_tags[df_movies_tags.relevance >= tag_relevance_th][
            ["movieId", "tag", "relevance"]].drop_duplicates()
        df_sim_movie_tags = pd.DataFrame(
            df_sim_movie_tags.groupby(["movieId", "tag"])["relevance"].mean()).reset_index()

        # get the count of tags per movie
        df_sim_movie_tag_cnt = df_sim_movie_tags.groupby(["movieId"])["tag"].count().reset_index()
        # select movies with a certain level of tag matches
        # movies with more tag match to the main movie are similar in nature
        similar_movies = df_sim_movie_tag_cnt[df_sim_movie_tag_cnt["tag"] >= tag_match]

        all_similar_movies = rating_df[(rating_df.movieId.isin(list(similar_movies.movieId)))
                                       & (rating_df.timestamp <= self.timestamp)].groupby("movieId")[
            "rating"].mean().to_dict()

        return all_similar_movies

    def get_features_dict(self):
        feature_dict = {"user_id": self.user_id,
                        "movie_id": self.movie_id,
                        "time_stamp": self.timestamp,
                        "genre": self.genre,
                        "orig_rating": self.rating,
                        "peer_movie_rating": self.peer_movie_rating,
                        "peer_genre_rating": self.peer_genre_rating,
                        "user_genre_rating": self.user_genre_rating,
                        "similar_movie_rating": self.similar_movie_rating,
                        "rating _flag": 1 if self.rating >= 4 else 0}

        return feature_dict
