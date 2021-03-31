from dataclasses import dataclass


@dataclass
class TwitterApiConfig:
    """ Represents a configuration file for a TwitterApiHandler instance """

    api_key: str
    api_secret: str


@dataclass
class DataHandlerConfig:
    """ Represents a configuration file for a DataHandler instance """

    twitter_api: TwitterApiConfig
