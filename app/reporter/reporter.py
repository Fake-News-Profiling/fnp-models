import os
from webbrowser import open_new_tab
from jinja2 import Environment, FileSystemLoader
from classifiers import timeline_classifier, timeline_preprocessor

class ReportBuilder():
    """ Handles processing of Twitter data using various classifiers """

    def __init__(self, timeline_classifier_weights_path: str):
        self.classifier = timeline_classifier.load_model(timeline_classifier_weights_path)
        template_env = Environment(loader=FileSystemLoader('./reporter/templates'))
        self.template = template_env.get_template('report.html')
    
    def generate_report(self, username: str, timeline: list):
        # Preprocess the tweet timeline
        preprocessor = timeline_preprocessor.InputPreprocessor(
            [tweet['text'] for tweet in timeline])
        preprocessor.preprocess()

        # Predict the probability that they are a fake news spreader
        overall_prediction = self.classifier.predict(preprocessor.to_tweet_feed_dataset())
        spreader_probability = overall_prediction.flatten().mean()*100

        # Find their tweet which is most likely to be fake
        self.classifier.set_encoding('individual')
        tweet_predictions = self.classifier.predict(preprocessor.to_individual_tweets_dataset())
        self.classifier.set_encoding('feed')
        tweet_index = tweet_predictions.flatten().argmax()

        # Generate the report
        report = self.template.render(
            username=username,
            num_tweets=len(timeline),
            spreader_probability=spreader_probability,
            fake_tweet=timeline[tweet_index]['text'],
            fake_tweet_id=timeline[tweet_index]['id'])

        filename = username+'_fake_news_report.html'
        os.makedirs('reports', exist_ok=True)
        filepath = os.path.join('reports', filename)
        report_file = open(filepath, 'w')
        report_file.write(report)
        report_file.close()
        print(f"Created report: {filename}")

        open_new_tab(filepath)
