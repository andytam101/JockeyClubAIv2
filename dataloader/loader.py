import numpy as np
import os
from abc import ABC, abstractmethod

class Loader(ABC):
    def load_train(self, session, output_dir, start_date=None, end_date=None):
        # start_date and end_date decides range of races to include
        x, y = self._load_from_db(session, start_date, end_date)
        session.close()
        self._save(output_dir, x, y)

    @abstractmethod
    def _load_from_db(self, session, start_date=None, end_date=None):
        raise NotImplementedError()

    @abstractmethod
    def load_predict(self, fetch, data):
        raise NotImplementedError()

    @abstractmethod
    def train_normalize(self, train_x):
        raise NotImplementedError()

    @abstractmethod
    def normalize(self, x, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def _save(output_dir, data_x, data_y):
        os.makedirs(output_dir, exist_ok=True)
        data_x_file = os.path.join(output_dir, 'data_x.npy')
        data_y_file = os.path.join(output_dir, 'data_y.npy')
        np.save(data_x_file, data_x)
        np.save(data_y_file, data_y)
