import logging
import argparse
import json
import importlib

from flask import Flask

from fnpserver import AbstractService
from fnpserver.services import DataEncoder


def parse_program_args() -> argparse.Namespace:
    """ Parse the inputted program arguments """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("config", help="Filepath to the services configuration file", type=str)
    arg_parser.add_argument("server_port", help="The server's port", type=int, default=8080)
    return arg_parser.parse_args()


def dynamically_load_class(import_path: str) -> AbstractService.__class__:
    """ Given the path to a Python class, this dynamically loads the class and returns it """
    import_path = import_path.split(".")
    module_path = ".".join(import_path[:-1])
    cls_name = import_path[-1]

    service_module = importlib.import_module(module_path)
    return getattr(service_module, cls_name)


def main():
    # Parse program arguments
    args = parse_program_args()

    # Startup the Flask server
    app = Flask("Fake News Profiling API")
    app.json_encoder = DataEncoder

    # Load services and register them with the Flask server
    with open(args.config, "r") as file:
        services_config = json.load(file)

    for service_info in services_config:
        # Dynamically load service
        logging.info("Loading service from", service_info["import_path"])
        cls = dynamically_load_class(service_info["import_path"])

        # Instantiate and register service
        service = cls(service_info["config"], service_info["endpoint"])
        service.register_with_server(app)

    # Run the Flask server
    app.run(host="0.0.0.0", port=args.server_port, debug=False)


if __name__ == "__main__":
    main()
