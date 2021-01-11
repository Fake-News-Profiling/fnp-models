import os
from webbrowser import open_new_tab

from backend.classifier_handlers.profiler import FakeNewsProfiler
from jinja2 import Environment, FileSystemLoader


class ReportBuilder:
    """ Handles processing of Twitter data using various classifiers """

    def __init__(self, profiler):
        self.profiler = profiler
        template_env = Environment(loader=FileSystemLoader("frontend/templates"))
        self.template = template_env.get_template("report.html")

    def generate_report(self, username):
        """Generate a fake news spreader report for the given Twitter user, using a
        history of their tweets"""
        # Profile the user
        profile = self.profiler.classify_user_timeline(username)
        report = self.template.render(
            username=profile.username,
            num_tweets_assessed=profile.num_tweets_assessed,
            spreader_probability=profile.spreader_probability,
            spreader_tweet=profile.spreader_tweet.text,
            spreader_tweet_id=profile.spreader_tweet.id,
            spreader_tweet_probability=profile.spreader_tweet_probability,
        )

        # Save the HTML report
        filename = username + "_fake_news_report.html"
        os.makedirs("reports", exist_ok=True)
        filepath = os.path.join("reports", filename)

        with open(filepath, 'w', encoding='utf-8') as report_file:
            print(report, file=report_file)
            print(f"Created report: {filename}")
            open_new_tab(filepath)
