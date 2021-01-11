import argparse
import sys

from backend.api_handlers.twitter_handler import TwitterHandler
from backend.classifier_handlers.classifier_wrappers import BertTimelineClassifierWrapper
from backend.classifier_handlers.profiler import FakeNewsProfiler
from frontend.reporter import ReportBuilder


def parse_program_args():
    """ Parse the inputted program arguments """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--api_key", "-k", help="The Twitter API key", type=str)
    arg_parser.add_argument("--api_secret", "-s", help="The Twitter API secret", type=str)
    arg_parser.add_argument(
        "--classifier_weights",
        "-w",
        help="Filepath to the classifiers weights checkpoint - a file ending in .ckpt",
    )
    args = vars(arg_parser.parse_args())

    if not (args["api_key"] and args["api_secret"] and args["classifier_weights"]):
        raise Exception(
            "Invalid command-line arguments. Use '--help' to see which arguments the program should be run with."
        )
    return args


def main():
    # Parse command-line args
    args = parse_program_args()
    profiler = FakeNewsProfiler(
        twitter_handler=TwitterHandler(args["api_key"], args["api_secret"]),
        classifier_handler=BertTimelineClassifierWrapper(args["classifier_weights"]),
    )
    reporter = ReportBuilder(profiler)

    # Start taking user input
    print("Starting up the program. Type 'exit' at any time to stop.")
    while True:
        username = input("Enter the Twitter username of the user to profile: ")
        if username == "exit":
            break
        if len(username) == 0:
            continue

        try:
            reporter.generate_report(username)
        except Exception as err:
            print(f"An error occurred: {err}")


if __name__ == "__main__":
    main()
