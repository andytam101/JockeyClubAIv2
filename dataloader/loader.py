import numpy as np
import os
from abc import ABC, abstractmethod

from database import init_engine, get_session


class Loader(ABC):
    def __init__(self, save_dir=None):
        init_engine()
        self.session = get_session()
        self.save_dir = save_dir

    @abstractmethod
    def load(self):
        """
        Reads data from database and transforms it into useful data for training.
        :return:
        """
        pass

    def save(self, data_x, data_y):
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            data_x_file = os.path.join(self.save_dir, 'data_x.npy')
            data_y_file = os.path.join(self.save_dir, 'data_y.npy')
            np.save(data_x_file, data_x)
            np.save(data_y_file, data_y)
            return True
        else:
            return False

    def close(self):
        self.session.close()
