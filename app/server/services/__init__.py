import logging
from abc import ABC
from typing import Dict
from dataclasses import dataclass

from dacite import from_dict
from flask import Flask

from data import DataHandlerConfig
from data.data_handler import DataHandler


@dataclass
class ModelConfig:
    load_path: str


@dataclass
class ServiceConfig:
    name: str
    models: Dict[str, ModelConfig]
    data_handler: DataHandlerConfig


class AbstractService(ABC):
    route_methods = {}

    def __init__(self, config_dict: dict, endpoint_root: str):
        self.config = from_dict(ServiceConfig, config_dict)
        self.data_handler = DataHandler(self.config.data_handler)
        self.endpoint_root = endpoint_root.replace('_', '-')

    def register_with_server(self, app: Flask):
        """ Register all methods of this class which start with 'route_' """
        for attribute in dir(self):
            if attribute.startswith("route_") and callable(getattr(self, attribute)):

                options = {}
                if attribute in self.route_methods:
                    options["methods"] = self.route_methods[attribute]

                route = attribute.replace('route_', '').lower().replace("_", "-")
                url = f"/{self.endpoint_root}/{route}"
                logging.info("Registering route", url)
                app.add_url_rule(
                    url,
                    endpoint=route,
                    view_func=getattr(self, attribute),
                    **options
                )
