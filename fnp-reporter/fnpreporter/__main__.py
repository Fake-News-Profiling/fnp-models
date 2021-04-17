import argparse
from json import JSONDecodeError

import requests

from .template import UserProfilerTemplateHandler


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
            template_handler.generate_report(f"{username}_report", data)
        except JSONDecodeError:
            print("Error: unable to decode response from the server -", response)


if __name__ == "__main__":
    main()
