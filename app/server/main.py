import logging
import argparse
import json

from flask import Flask

from services.user_profiler import UserProfilerService


def parse_program_args():
    """ Parse the inputted program arguments """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("config", help="Filepath to the services configuration file", type=str)
    return arg_parser.parse_args()


def main():
    # Parse program arguments
    args = parse_program_args()

    # Startup the Flask server
    app = Flask("Fake News Profiling API")

    # Load services and register them with the Flask server
    with open(args.config, "r") as file:
        services_config = json.load(file)

    service_cls = {
        UserProfilerService.__name__: UserProfilerService,
    }
    for config in services_config:
        cls = service_cls[config["name"]]
        logging.info("Loading service", cls.__name__)
        service = cls(config)
        service.register_with_server(app)

    # Run the Flask server
    app.run(host="0.0.0.0", port=8000, debug=False)


if __name__ == "__main__":
    main()
