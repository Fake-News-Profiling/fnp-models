from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .sentiment import AbstractSentimentAnalysisWrapper, Sentiment


class VaderSentimentAnalysisWrapper(AbstractSentimentAnalysisWrapper):
    """ Sentiment analysis wrapper for VADER """

    def __init__(self):
        analyser = SentimentIntensityAnalyzer()
        super().__init__(analyser)

    def sentiment(self, text: str) -> Sentiment:
        scores = self.analyser.polarity_scores(text)
        compound = scores["compound"]
        if compound >= 0.05:
            classification = "positive"
        elif compound <= -0.05:
            classification = "negative"
        else:
            classification = "neutral"

        return Sentiment(
            negative=scores["neg"],
            neutral=scores["neu"],
            positive=scores["pos"],
            compound=compound,
            classification=classification,
        )