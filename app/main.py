import argparse
import os

from data.data_handler import DataHandler
from services.user_profiler.user_profiler_service import UserProfilerService
from template.template_handler import UserProfilerTemplateHandler


def parse_program_args():
    """ Parse the inputted program arguments """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--data_config", "-d", help="Filepath to the data handler configuration file", type=str
    )
    arg_parser.add_argument(
        "--services_config", "-s", help="Filepath to the services configuration file", type=str
    )
    args = vars(arg_parser.parse_args())

    if not (args["data_config"] and args["services_config"]):
        raise Exception(
            "Invalid command-line arguments. Use '--help' to see which arguments the program should be run with."
        )
    return args


def main():
    args = parse_program_args()

    user_profiler_service = UserProfilerService(
        args["services_config"],
        DataHandler(args["data_config"]),
    )
    template_handler = UserProfilerTemplateHandler()

    print("Starting up the program. Type 'exit' at any time to stop.")
    while True:
        username = input("Enter the Twitter username of the user to profile: ")
        if username == "exit":
            return 0

        try:
            data = user_profiler_service.profile_twitter_user_from_tweet_feed(username)
            template_handler.generate_report(f"{username}_report", data)
        except Exception as ex:
            print(f"An error occurred: {ex}")


if __name__ == "__main__":
    main()
