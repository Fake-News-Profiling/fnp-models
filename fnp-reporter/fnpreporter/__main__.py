import argparse
from json import JSONDecodeError

import requests

from fnpreporter.template import UserProfilerTemplateHandler


def parse_program_args() -> argparse.Namespace:
    """ Parse the inputted program arguments """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("api_url", help="URL of the Fake News Profiling API", type=str)
    arg_parser.add_argument("report_dir", help="Path to the report directory", type=str)
    return arg_parser.parse_args()


def main():
    args = parse_program_args()
    template_handler = UserProfilerTemplateHandler(args.report_dir)

    print("Starting up the program. Type 'exit' at any time to stop.")
    while True:
        username = input("Enter the Twitter username of the user to profile: ")
        if username == "exit":
            return 0

        response = requests.get(args.api_url, params={"username": username})
        try:
            data = response.json()

            template_handler.generate_report(f"{username}_report", {
                "is_fake_news_spreader": data["Ensemble.predict"][0] == 1,
                "Ensemble_predict_proba": data["Ensemble.predict_proba"][0],
                "Bert_predict_proba": data["Bert.predict_proba"][0],
                "Readability_predict_proba": data["Readability.predict_proba"][0],
                "Ner_predict_proba": data["Ner.predict_proba"][0],
                "Sentiment_predict_proba": data["Sentiment.predict_proba"][0],
                "num_tweets_used": data["num_tweets_used"],
                "username": data["username"],
            })
        except JSONDecodeError:
            print("Error: unable to decode response from the server -", response)


if __name__ == "__main__":
    main()
