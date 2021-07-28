import pickle
from tqdm import tqdm
import pandas as pd
import multiprocessing as mp
from features.user_movie import UserMovie


def get_user_features(params):
    user_movie_feature = UserMovie(params[0],
                                   params[1],
                                   params[2],
                                   params[3],
                                   params[4],
                                   params[5],
                                   params[6])
    return user_movie_feature.feature_dict


def get_val_index(df_rating_val, val_params):
    val_index = df_rating_val[(df_rating_val.userId == val_params["userId"])
                              & (df_rating_val.movieId == val_params["movieId"])
                              & (df_rating_val.timestamp == val_params["timestamp"])].index.values[0]
    return val_index


class UserDetails:

    def __init__(self, start_index, end_index):
        self.start_index = start_index
        self.end_index = end_index

    def get_user_details_bytime(self, df_rating_user, df_movie,
                                df_tag_relevance):
        train_data = []
        test_data = []
        valid_data = []
        for i, user in tqdm(enumerate(df_rating_user.userId.unique()[self.start_index:self.end_index])):
            df_sub = df_rating_user[df_rating_user.userId == user].sort_values("timestamp").reset_index(drop=True)
            # getting upto validation data_creation and transforming them to the model
            input_args = ((items[1], items[2], items[4], items[3], df_rating_user, df_movie, df_tag_relevance)
                          for items in df_sub[:len(df_sub) - 1].itertuples())

            with mp.Pool(8) as pool:
                result = pool.imap(get_user_features, input_args, chunksize=10)
                train_output = [x for x in result]

            val_data_params = df_sub.iloc[len(df_sub) - 2]
            test_data_params = df_sub.iloc[len(df_sub) - 1]

            val_index = get_val_index(df_rating_user, val_data_params)

            df_test_rating = df_rating_user.drop(df_rating_user.index[val_index])

            test_feature = UserMovie(user_id=test_data_params[0],
                                     movie_id=test_data_params[1],
                                     timestamp=test_data_params[3],
                                     rating=test_data_params[2],
                                     rating_df=df_test_rating,
                                     movie_df=df_movie,
                                     tag_df=df_tag_relevance).feature_dict

            train_data.append(train_output[:-1])
            test_data.append(test_feature)
            valid_data.append(train_output[-1])

        with open('archive/data_creation/train_data_' + str(self.start_index) + '.pickle', 'wb') as f:
            pickle.dump(train_data, f)
        pd.DataFrame(test_data).to_pickle('archive/data_creation/test_data' + str(self.start_index) + '.pickle')
        pd.DataFrame(valid_data).to_pickle('archive/data_creation/valid_data' + str(self.start_index) + '.pickle')

    def get_user_details_latest(self, df_rating_user, df_movie,
                                df_tag_relevance, batch_no):
        train_data = []
        test_data = []
        for i, user in tqdm(enumerate(df_rating_user.userId.unique()[self.start_index:self.end_index])):
            df_sub = df_rating_user[df_rating_user.userId == user].sort_values("timestamp").reset_index(drop=True)

            val_data_params = df_sub.iloc[len(df_sub) - 2]
            test_data_params = df_sub.iloc[len(df_sub) - 1]

            val_feature = UserMovie(user_id=val_data_params[0],
                                    movie_id=val_data_params[1],
                                    timestamp=val_data_params[3],
                                    rating=val_data_params[2],
                                    rating_df=df_rating_user,
                                    movie_df=df_movie,
                                    tag_df=df_tag_relevance).feature_dict

            val_index = get_val_index(df_rating_user, val_data_params)

            df_test_rating = df_rating_user.drop(df_rating_user.index[val_index])

            test_feature = UserMovie(user_id=test_data_params[0],
                                     movie_id=test_data_params[1],
                                     timestamp=test_data_params[3],
                                     rating=test_data_params[2],
                                     rating_df=df_test_rating,
                                     movie_df=df_movie,
                                     tag_df=df_tag_relevance).feature_dict

            test_data.append(test_feature)
            train_data.append(val_feature)

        pd.DataFrame(test_data).to_pickle(
            'archive/data_creation/test_data_latest_' + str(batch_no) + '_' + str(self.start_index) + "_" + str(
                self.end_index) + '.pickle')
        pd.DataFrame(train_data).to_pickle(
            'archive/data_creation/train_data_latest_' + str(batch_no) + '_' + str(self.start_index) + "_" + str(
                self.end_index) + '.pickle')
