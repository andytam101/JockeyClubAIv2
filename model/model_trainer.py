import torch
import numpy as np
import math
import os

from ._model import _Model
from utils.config import device


class ModelTrainer:
    def __init__(self, model, model_dir=None):
        self.model: _Model = model
        self.model.to(device)

        self.optimizer = self.model.optimizer()
        self.dataloader = self.model.dataloader
        self.params = {}

        if model_dir is not None:
            self.load(model_dir)

    def load(self, model_dir):
        model_state_dict_path = os.path.join(model_dir, 'model_state_dict.pth')
        optimizer_state_dict_path = os.path.join(model_dir, 'optimizer_state_dict.pth')
        model_state_dict     = torch.load(model_state_dict_path, map_location=device)
        optimizer_state_dict = torch.load(optimizer_state_dict_path, map_location=device)
        self.model.load_state_dict(model_state_dict)
        self.optimizer.load_state_dict(optimizer_state_dict)

    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        model_state_dict = self.model.state_dict()
        optimizer_state_dict = self.optimizer.state_dict()

        model_state_dict_path = os.path.join(output_dir, "model_state_dict.pth")
        optimizer_state_dict_path = os.path.join(output_dir, "optimizer_state_dict.pth")

        torch.save(model_state_dict, model_state_dict_path)
        torch.save(optimizer_state_dict, optimizer_state_dict_path)

        # save normalization stuff
        self.model.save_normalization(output_dir, **self.params)

    @staticmethod
    def _split_data(x, y, cv_size):
        m = x.shape[0]
        cv_idx = int(m * (1 - cv_size))
        return x[:cv_idx], y[:cv_idx], x[cv_idx:], y[cv_idx:]

    def train_model(self, data_dir, cv_size=0.2, epochs=10000, batch_size=2048):
        x = np.load(os.path.join(data_dir, 'data_x.npy'))
        y = np.load(os.path.join(data_dir, 'data_y.npy'))   # ranking only

        y = self.model.process_y(y)

        x, y = self._shuffle(x, y)
        train_x, train_y, cv_x, cv_y = self._split_data(x, y, cv_size)

        normalize_params = self.dataloader.train_normalize(train_x)
        self.dataloader.normalize(cv_x, **normalize_params)
        self.params.update(normalize_params)

        train_x = torch.tensor(train_x, dtype=torch.float, device=device)
        train_y = torch.tensor(train_y, dtype=torch.float, device=device)
        cv_x = torch.tensor(cv_x, dtype=torch.float, device=device)
        cv_y = torch.tensor(cv_y, dtype=torch.float, device=device)

        # do the training
        train_hist = []
        cv_hist = []

        optimizer = self.optimizer
        criterion = self.model.criterion()
        acc_func = self.model.accuracy

        # show initial loss
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(train_x)
            loss = criterion(predictions, train_y)
            accuracy = acc_func(predictions, train_y)

            cv_predictions = self.model(cv_x)
            cv_loss = criterion(cv_predictions, cv_y)
            cv_accuracy = acc_func(cv_predictions, cv_y)

            train_hist.append((loss.item(), accuracy))
            cv_hist.append((cv_loss.item(), cv_accuracy))

            print(f"Initial: train loss = {loss}, cv loss = {cv_loss}")

        batch_numbers = math.ceil(train_x.size()[0] / batch_size)

        print(f"Training model: {self.model} for {epochs} epochs")
        for epoch in range(epochs):
            for i in range(batch_numbers):
                # set model to train mode
                self.model.train()

                optimizer.zero_grad()
                predictions = self.model(train_x[batch_size * i:batch_size * (i + 1)])
                loss = criterion(predictions, train_y[batch_size * i:batch_size * (i + 1)])

                loss.backward()
                optimizer.step()

            # set model to evaluate mode
            self.model.eval()

            with torch.no_grad():
                predictions = self.model(train_x)
                loss = criterion(predictions, train_y)
                train_accuracy = acc_func(predictions, train_y)

                cv_predictions = self.model(cv_x)
                cv_loss = criterion(cv_predictions, cv_y)

                cv_accuracy = acc_func(cv_predictions, cv_y)

            train_hist.append((loss.item(), train_accuracy))
            cv_hist.append((cv_loss.item(), cv_accuracy))

            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}: train loss = {loss}, cv loss = {cv_loss}")

        return train_hist, cv_hist

    @staticmethod
    def _shuffle(x, y):
        mx, nx = x.shape[0], x.shape[1]
        my, ny = y.shape[0], y.shape[1]

        assert mx == my  # same set of data
        combined = np.zeros((mx, nx + ny), dtype=np.float32)
        combined[:, :nx] = x
        combined[:, nx:] = y
        np.random.shuffle(combined)

        return combined[:, :nx], combined[:, nx:]
