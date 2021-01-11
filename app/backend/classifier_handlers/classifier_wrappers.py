from backend.classifiers.timeline_classifier import BertModel


class BertTimelineClassifierWrapper:
    BASE_MODEL_URL = "https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3"
    BASE_MODEL_SIZE = 500

    def __init__(self, classifier_weights_path):
        self.classifier = BertModel(self.BASE_MODEL_URL, self.BASE_MODEL_SIZE, "feed")
        self.classifier.build()

        if classifier_weights_path:  # TODO - FIX ME
            self.classifier.model.load_weights(classifier_weights_path)

    def predict_fake_news_spreader_prob(self, preprocessed_tweets):
        """ Return the probability that the owner of some tweets is a fake news spreader """
        prediction = self._predict("feed", preprocessed_tweets)
        return prediction.flatten().mean() * 100

    def predict_tweet_with_highest_prob(self, original_tweets, preprocessed_tweets):
        """ Return the tweet which is most likely to contain fake news, along with the probability """
        predictions = self._predict("individual", preprocessed_tweets)
        tweet_index = predictions.flatten().argmax()
        return (original_tweets[tweet_index], predictions[tweet_index])

    def _predict(self, classifier_mode, preprocessed_tweets):
        self.classifier.set_encoding("individual")
        return self.classifier.predict(preprocessed_tweets)
