import pandas as pd
import numpy as np
import glob
import operator
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.utils import check_random_state
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, make_scorer


def get_data_from_files(path):
    df = pd.DataFrame()
    for f in glob.glob(path + '**.pickle', recursive=True):
        df = df.append(pd.read_pickle(f))
    print("total no of records ", len(df))
    return df


def get_feature_values(row, n):
    top_5 = list(
        dict(sorted(row["similar_movie_rating"].items(), key=operator.itemgetter(1), reverse=True)[:n]).values())
    if len(top_5) < 5:
        top_5.extend([0] * (n - len(top_5)))

    feature_list = [row.peer_movie_rating, row.peer_genre_rating, row.user_genre_rating]
    feature_list.extend(top_5)

    max_val = np.nanmax(feature_list)
    features_list = [max_val if (pd.isna(x)) or (x == 0) else x for x in feature_list]

    return features_list


class Model:

    def __init__(self, train_path, test_path, top_n_movies, feature_columns, feature_labels, classifiers):
        self.feature_columns = feature_columns
        self.feature_labels = feature_labels
        self.train_data = get_data_from_files(train_path)
        self.test_data = get_data_from_files(test_path)
        self.train_data["features"] = self.train_data.apply(lambda row: get_feature_values(row, top_n_movies), axis=1)
        self.test_data["features"] = self.test_data.apply(lambda row: get_feature_values(row, top_n_movies), axis=1)
        self.classifiers = classifiers
        self.best_model, self.models_report = self.build_model()
        self.predict_model()

    def build_model(self):
        # Building the baseline model
        f1_score = 0
        rng = check_random_state(0)
        kf = KFold(n_splits=5, random_state=42, shuffle=True)  # Doing a 4 fold Cross Validation
        fold = []
        scr = []
        models_report = pd.DataFrame(columns=['Model', 'Precision_score', 'Recall_score', 'F1_score', 'Accuracy'])

        df_final = self.train_data
        for i, (train_index, test_index) in tqdm(enumerate(kf.split(df_final))):
            training = df_final.iloc[train_index, :]
            valid = df_final.iloc[test_index, :]
            training_feats = np.asarray(list(training[self.feature_columns].values))  # defined above
            training_label = np.asarray(list(training[self.feature_labels].values))

            valid_feats = np.asarray(list(valid[self.feature_columns].values))
            valid_label = np.asarray(list(valid[self.feature_labels].values))
            for clf, clf_name in zip(self.classifiers.values(), self.classifiers.keys()):
                model = BaggingClassifier(base_estimator=clf,
                                          random_state=rng,
                                          n_estimators=10)
                pred = model.fit(training_feats, training_label).predict(valid_feats)
                score = accuracy_score(y_true=valid_label, y_pred=pred)
                fold.append(i + 1)
                scr.append(score)

                t = pd.Series({
                    'Model': clf_name,
                    'Precision_score': metrics.precision_score(valid_label, pred),
                    'Recall_score': metrics.recall_score(valid_label, pred),
                    'F1_score': metrics.f1_score(valid_label, pred),
                    'Accuracy': metrics.accuracy_score(valid_label, pred),
                    "AUC_Score": metrics.roc_auc_score(valid_label, pred),
                    'validation_fold': i}
                )

                models_report = models_report.append(t, ignore_index=True)
                if metrics.f1_score(valid_label, pred) > f1_score:
                    f1_score = metrics.f1_score(valid_label, pred)
                    best_model = model
        print(best_model)
        print(models_report)

        feature_importances = pd.DataFrame({"feature_name": ["peer_movie_rating",
                                                             "peer_genre_rating",
                                                             "user_genre_rating",
                                                             "similar_movie1",
                                                             "similar_movie2",
                                                             "similar_movie3",
                                                             "similar_movie4",
                                                             "similar_movie4"],
                                            "feature_importance_coeff": list(np.mean([
                                                tree.coef_ for tree in best_model.estimators_
                                            ], axis=0)[0])})
        print(feature_importances)
        return best_model, models_report

    def predict_model(self):
        test_feats = np.asarray(list(self.test_data[self.feature_columns].values))  # defined above
        test_label = np.asarray(list(self.test_data[self.feature_labels].values))
        self.test_data["preds"] = list(self.best_model.predict(test_feats))

        unique_label = np.unique([test_label, self.test_data["preds"]])
        cmtx = pd.DataFrame(
            metrics.confusion_matrix(test_label, self.test_data["preds"], labels=unique_label),
            index=['true:{:}'.format(x) for x in unique_label],
            columns=['pred:{:}'.format(x) for x in unique_label]
        )
        print(cmtx)

        return self.test_data["preds"].value_counts()
