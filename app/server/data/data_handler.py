from data import DataHandlerConfig
from data.twitter_api_handler import TwitterApiHandler


class DataHandler:
    """ Loads all data/api handlers """

    def __init__(self, config: DataHandlerConfig):
        self.twitter_api_handler = TwitterApiHandler(config.twitter_api) if config.twitter_api else None

    def get_twitter_api(self) -> TwitterApiHandler:
        """
        Returns a TwitterApiHandler instance, or raises a NotLoadedError if the
        handler was not loaded
        """
        if self.twitter_api_handler:
            return self.twitter_api_handler
        else:
            raise NotLoadedError(
                "A Twitter API Handler was not loaded, due to missing information in the config file"
            )


class NotLoadedError(RuntimeError):
    """ A requested data/api handler was not loaded """
    pass
