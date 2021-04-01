from textblob import TextBlob

from statistical.data_extraction.sentiment import AbstractSentimentAnalysisWrapper, Sentiment


class TextBlobSentimentAnalysisWrapper(AbstractSentimentAnalysisWrapper):
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
            classification=classification
        )