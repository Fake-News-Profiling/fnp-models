from typing import Union, Optional

from dataclasses import dataclass

from .twitter import TwitterHandlerConfig, AbstractTwitterHandler, TwitterApiHandler, TwitterDatabaseHandler


@dataclass
class DataHandlerConfig:
    """ Represents a configuration file for a DataHandler instance """

    twitter_config: Optional[TwitterHandlerConfig] = None


class DataHandler:
    """ Loads all data/api handlers """

    def __init__(self, config: DataHandlerConfig):
        self.twitter_handler = self._load_twitter_handler(config.twitter_config)

    @staticmethod
    def _load_twitter_handler(twitter_config: TwitterHandlerConfig) -> Union[AbstractTwitterHandler, None]:
        if twitter_config.twitter_dir is not None:
            return TwitterDatabaseHandler(twitter_config)
        elif twitter_config.api_key is not None and twitter_config.api_secret is not None:
            return TwitterApiHandler(twitter_config)

        return None

    def get_twitter_handler(self) -> AbstractTwitterHandler:
        """
        Returns a TwitterHandler instance, or raises a NotLoadedError if the handler was not loaded
        """
        if self.twitter_handler:
            return self.twitter_handler
        else:
            raise NotLoadedError(
                "A Twitter Handler was not loaded, due to missing information in the config file"
            )


class NotLoadedError(RuntimeError):
    """ A requested data/api handler was not loaded """
    pass
