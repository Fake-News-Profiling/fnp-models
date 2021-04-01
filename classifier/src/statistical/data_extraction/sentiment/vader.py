from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from statistical.data_extraction.sentiment import AbstractSentimentAnalysisWrapper, Sentiment


class VaderSentimentAnalysisWrapper(AbstractSentimentAnalysisWrapper):
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
            compound=compound,
            classification=classification
        )