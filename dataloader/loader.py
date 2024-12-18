import numpy as np
import os
from abc import ABC, abstractmethod

from database import init_engine, get_session


class Loader(ABC):
    def __init__(self, save_dir=None):
        init_engine()
        self.session = get_session()
        self.save_dir = save_dir

    def load_train(self, cv_size, directory=None):
        if directory is not None:
            x, y = self._load_from_dir(directory)
        else:
            x, y = self._load_from_db()
            self.save(x, y)

        self._shuffle(x, y)

        m = x.shape[0]
        cv_idx = int(m * (1 - cv_size))
        train_x = x[:cv_idx]
        train_y = y[:cv_idx]
        cv_x = x[cv_idx:]
        cv_y = y[cv_idx:]

        return train_x, train_y, cv_x, cv_y

    @abstractmethod
    def _load_from_dir(self, directory) -> (np.ndarray, np.ndarray):
        raise NotImplementedError()

    @abstractmethod
    def _load_from_db(self):
        raise NotImplementedError()

    @abstractmethod
    def load_predict(self, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def train_normalize(self, train_x):
        raise NotImplementedError()

    @abstractmethod
    def normalize(self, x, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def _shuffle(x, y):
        mx, nx = x.shape[0], x.shape[1]
        my, ny = y.shape[0], y.shape[1]

        assert mx == my     # same set of data
        combined = np.zeros((mx, nx + ny), dtype=np.float32)
        combined[:, :nx] = x
        combined[:, nx:] = y
        np.random.shuffle(combined)

        return combined[:, :nx], combined[:, nx:]

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
