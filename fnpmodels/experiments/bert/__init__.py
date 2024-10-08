import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

from fnpmodels.experiments.experiment import AbstractSklearnExperiment
from fnpmodels.models.sklearn import VotingClassifier


class BertLogitsCombinedExperiment(AbstractSklearnExperiment):

    def build_model(self, hp):
        model_type = hp.Choice(
            "Sklearn.model_type",
            [LogisticRegression.__name__, SVC.__name__, RandomForestClassifier.__name__, XGBClassifier.__name__,
             VotingClassifier.__name__]
        )

        if model_type == VotingClassifier.__name__:
            return VotingClassifier()

        return self.select_sklearn_model(hp)

    def cv_data_transformer(self, x_train, y_train, x_test, y_test):
        bert_model = self.fit_bert(x_train, y_train, x_test, y_test)
        x_train, y_train, x_test, y_test = self.predict_user_data(bert_model, x_train, y_train, x_test, y_test)
        x_train = tf.math.sigmoid(x_train)
        x_test = tf.math.sigmoid(x_test)
        x_train, x_test = self.pool_combined(x_train, x_test)
        return self.to_float64(x_train, y_train, x_test, y_test)

    @staticmethod
    def to_float64(x_train, y_train, x_test, y_test):
        # RandomForestClassifier doesn't support float32
        return (tf.cast(x_train, tf.float64), tf.cast(y_train, tf.float64), tf.cast(x_test, tf.float64),
                tf.cast(y_test, tf.float64))

    def pool_combined(self, x_train, x_test):
        # x_.shape == (num_users, TWEET_FEED_LEN, -1)
        pooler = self.hyperparameters.get("Combined.pooler")
        if pooler == "concat":
            x_train = tf.keras.layers.Flatten()(x_train)
            x_test = tf.keras.layers.Flatten()(x_test)
        elif pooler == "max":
            x_train = tf.keras.layers.GlobalMaxPool1D()(x_train)
            x_test = tf.keras.layers.GlobalMaxPool1D()(x_test)
        elif pooler == "average":
            x_train = tf.keras.layers.GlobalAveragePooling1D()(x_train)
            x_test = tf.keras.layers.GlobalAveragePooling1D()(x_test)
        else:
            raise ValueError("Invalid value for `Combined.pooler`")

        return x_train, x_test

    @classmethod
    def preprocess_cv_data(cls, hp, x_train, y_train, x_test, y_test):
        return hp, x_train, y_train, x_test, y_test

    def fit_bert(self, x_train, y_train, x_test, y_test):
        pass

    def predict_user_data(self, bert_model, x_train, y_train, x_test, y_test):
        pass
