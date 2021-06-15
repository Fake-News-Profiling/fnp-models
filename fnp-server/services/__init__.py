import json
import logging
from abc import ABC
from typing import Dict
from dataclasses import dataclass

from dacite import from_dict
from flask import Flask
import numpy as np

from fnpserver.data.data_handler import DataHandler, DataHandlerConfig


@dataclass
class ModelConfig:
    hyperparameters: str


@dataclass
class ServiceConfig:
    models: Dict[str, ModelConfig]
    data_handler: DataHandlerConfig


class AbstractService(ABC):
    """ A fake-news-profiling service which provides endpoints to various models """

    def __init__(self, config_dict: dict, endpoint: str):
        self.route_methods = {}
        self.config = from_dict(ServiceConfig, config_dict)
        self.data_handler = DataHandler(self.config.data_handler)
        self.endpoint = endpoint.replace('_', '-')

    def register_with_server(self, app: Flask):
        """ Register all methods of this class which start with 'route_' """
        for attribute in dir(self):
            if attribute.startswith("route_") and callable(getattr(self, attribute)):

                options = {}
                if attribute in self.route_methods:
                    options["methods"] = self.route_methods[attribute]

                route = attribute.replace('route_', '').lower().replace("_", "-")
                url = f"/{self.endpoint}/{route}"
                logging.info("Registering route", url)

                app.add_url_rule(
                    url,
                    endpoint=route,
                    view_func=getattr(self, attribute),
                    **options
                )


class DataEncoder(json.JSONEncoder):
    """ JSON data encoder which can handle Numpy data types"""

    def default(self, data):
        if isinstance(data, np.bool_):
            return bool(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()

        return json.JSONEncoder.default(self, data)
