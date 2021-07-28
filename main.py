import glob

import click
import pandas as pd

from data_creation.data_batching import get_data_batch, get_mapped_tag_relevance_score
from data_creation.data_creation import UserDetails
from model.building_model import Model
from model.classifier_mapping import CLASSIFIERS


@click.command()
@click.option("--start_index", default=0, prompt="start_val", help="starting value of the for loop")
@click.option("--end_index", default=0, prompt="end_val", help="ending value of the for loop")
@click.option("--run_model", default=True, prompt="run model")
@click.option("--run_feature_extraction", default=False, prompt="run feature extraction")
@click.option("--run_data_sampling", default=False, prompt="run data sampling")
@click.option("--feature-cache-path",
              is_eager=True,
              required=True,
              help="Absolute path to the cache folder",
              type=click.Path(),
              default="archive/df_random_rating_", )
@click.option("--tag_cache", default=True, prompt="tag_relevance_score", help="use the tag relevance score mapping")
@click.option("--batch_size", default=5, prompt="batch_size", help="no of batches to be created")
@click.option("--sample_percentage", default=0.1, prompt="sample_percentage", help="percentage of users in each batch")
def main(start_index, end_index,
         run_model,
         run_feature_extraction,
         run_data_sampling,
         feature_cache_path,
         tag_cache,
         batch_size=5,
         sample_percentage=0.1
         ):
    if run_feature_extraction:
        df_movie = pd.read_csv("archive/movie.csv")
        if tag_cache:
            df_tag = pd.read_pickle("archive/df_tag_relevance.pickle")
        else:
            df_genome_scores = pd.read_csv("archive/genome_scores.csv")
            df_genome_tags = pd.read_csv("archive/genome_tags.csv")
            df_tag = pd.read_csv("archive/tag.csv")
            df_tag = get_mapped_tag_relevance_score(df_tag, df_genome_tags, df_genome_scores)
        print("running feature extraction on all the data sampled")
        for f in glob.glob(feature_cache_path + "**.pickle", recursive=True):
            df_rating = pd.read_pickle(f)
            if end_index == 0:
                end_index = len(df_rating.userId.unique())
            print(end_index)
            user_details = UserDetails(start_index=start_index, end_index=end_index)
            user_details.get_user_details_latest(df_rating, df_movie, df_tag, batch_no=5)

    elif run_data_sampling:
        print("running sampling to extract data in batches")
        # For the entire set of users it selects stratified random samples of N batches
        # Each batch is r% of unique users and can be decided by the user
        df_rating = pd.read_csv("archive/rating.csv")
        get_data_batch(df_rating=df_rating, batch_no=batch_size, sample_percentage=sample_percentage)

    elif run_model:
        print("running classification model")
        # create classfier object dictionary
        classifier_object = CLASSIFIERS
        classifier_model = Model(train_path="archive/data/train_data_latest",
                                 test_path="archive/data/test_data_latest",
                                 top_n_movies=5,
                                 feature_labels="rating _flag",
                                 feature_columns="features",
                                 classifiers=classifier_object
                                 )

        print(classifier_model.test_data["preds"].value_counts())


if __name__ == "__main__":
    main()
