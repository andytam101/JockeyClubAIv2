from abc import ABC, abstractmethod
from database import init_engine, get_session


class Loader(ABC):
    def __init__(self):
        init_engine()
        self.session = get_session()

    @abstractmethod
    def load(self):
        """
        Reads data from database and transforms it into useful data for training.
        :return:
        """
        pass

    def close(self):
        self.session.close()
