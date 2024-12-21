import torch
from torch import nn
from abc import ABC, abstractmethod
import os

from dataloader.loader import Loader as DataLoader
from utils.config import device


class _Model(nn.Module, ABC):
    def __init__(self, output_model_dir, data_path=None):
        super(_Model, self).__init__()
        self.optimizer = None
        self.data_path = data_path
        self.normalization = None
        self.input_model_path  = None
        self.output_model_path = output_model_dir
        self.dataloader: DataLoader = self._dataloader()

    def load(self, model_dir=None):
        """Function must be called, can maybe be integrated into __init__"""
        self.optimizer = self._optimizer()
        if model_dir is None:
            return

        model_state_dict_path = os.path.join(model_dir, "model_state.pth")
        optimizer_state_dict_path = os.path.join(model_dir, "optimizer_state.pth")
        self.load_state_dict(torch.load(model_state_dict_path, map_location=device))
        self.optimizer.load_state_dict(torch.load(optimizer_state_dict_path, map_location=device))
        self.input_model_path = model_dir

        # children classes will do more (with normalisation stuff)
        self._load_normalization(model_dir)

    def save(self):
        os.makedirs(self.output_model_path, exist_ok=True)
        model_state_path = os.path.join(self.output_model_path, "model_state.pth")
        optimizer_state_dict_path = os.path.join(self.output_model_path, "optimizer_state.pth")
        torch.save(self.state_dict(), model_state_path)
        torch.save(self.optimizer.state_dict(), optimizer_state_dict_path)

        self.dataloader.close()

        # children classes will do more (with normalisation stuff)
        self._save_normalization(self.output_model_path)

    def train_model(self, cv_size, epochs=10000):
        train_x, train_y, cv_x, cv_y = self.dataloader.load_train(cv_size, directory=self.data_path)
        self.normalization = self.dataloader.train_normalize(train_x)     # returns **kwargs
        self.dataloader.normalize(cv_x, **self.normalization)

        self.to(device)

        optimizer = self.optimizer
        criterion = self.criterion()
        acc_func  = self.accuracy

        train_hist = []
        cv_hist = []

        print(f"Training model: {self} for {epochs} epochs")
        for epoch in range(epochs):
            # set model to train mode
            self.train()

            optimizer.zero_grad()
            predictions = self(train_x)
            loss = criterion(predictions, train_y)

            loss.backward()
            optimizer.step()

            train_accuracy = acc_func(predictions, train_y)

            # set model to evaluate mode
            self.eval()

            with torch.no_grad():
                cv_predictions = self(cv_x)
                cv_loss = criterion(cv_predictions, cv_y)

                cv_accuracy = acc_func(cv_predictions, cv_y)

            train_hist.append((loss.item(), train_accuracy))
            cv_hist.append((cv_loss.item(), cv_accuracy))

            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}: train loss = {loss}, cv loss = {cv_loss}")

        self.save()
        return train_hist, cv_hist

    def predict(self, data):
        x = self.dataloader.load_predict(data)
        self.dataloader.normalize(x, **self.normalization)
        x = torch.tensor(x, device=device)
        predictions = self(x)
        return self.reformat_predictions(predictions)

    @staticmethod
    @abstractmethod
    def _dataloader():
        """
        Returns basic unloaded type of dataloader required by each model
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def _optimizer(self):
        raise NotImplementedError

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def criterion():
        raise NotImplementedError

    @abstractmethod
    def accuracy(self, output, target):
        raise NotImplementedError

    @staticmethod
    def reformat_predictions(predictions):
        return predictions

    @abstractmethod
    def _load_normalization(self, model_directory):
        raise NotImplementedError

    @abstractmethod
    def _save_normalization(self, model_directory):
        raise NotImplementedError
