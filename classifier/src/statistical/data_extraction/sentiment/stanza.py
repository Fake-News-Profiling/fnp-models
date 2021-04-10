import numpy as np
import stanza

from statistical.data_extraction.sentiment import AbstractSentimentAnalysisWrapper, Sentiment


class StanzaSentimentAnalysisWrapper(AbstractSentimentAnalysisWrapper):
    def __init__(self):
        analyser = stanza.Pipeline("en", processors="tokenize,sentiment")
        super().__init__(analyser)
        self.counter = 0

    def sentiment(self, text: str) -> Sentiment:
        self.counter += 1
        if self.counter % 1000 == 0:
            print("%d done" % self.counter)

        compound = float(np.mean([sentence.sentiment - 1 for sentence in self.analyser(text).sentences]))
        if compound >= 0.25:
            classification = "positive"
        elif compound <= -0.25:
            classification = "negative"
        else:
            classification = "neutral"

        return Sentiment(
            compound=compound,
            classification=classification,
        )