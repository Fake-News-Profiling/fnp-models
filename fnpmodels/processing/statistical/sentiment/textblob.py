from textblob import TextBlob

from .sentiment import AbstractSentimentAnalysisWrapper, Sentiment


class TextBlobSentimentAnalysisWrapper(AbstractSentimentAnalysisWrapper):
    """ Sentiment analysis wrapper for TextBlob """

    def __init__(self):
        analyser = TextBlob
        super().__init__(analyser)

    def sentiment(self, text: str) -> Sentiment:
        compound = self.analyser(text).polarity
        if compound >= 0.05:
            classification = "positive"
        elif compound <= -0.05:
            classification = "negative"
        else:
            classification = "neutral"

        return Sentiment(
            compound=compound,
            classification=classification,
        )