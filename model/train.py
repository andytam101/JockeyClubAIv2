import os

import torch

from model._model import _Model
from utils.config import device


def split_data(x, y, train_size):
    m = x.size(dim=0)
    cv_idx = int(m * train_size)
    train_x = x[:cv_idx]
    train_y = y[:cv_idx]
    cv_x = x[cv_idx:]
    cv_y = y[cv_idx:]

    return train_x, train_y, cv_x, cv_y


def save_model(path, model, optimizer):
    model_path = os.path.join(path, 'model_state.pth')
    optimizer_path = os.path.join(path, 'optimizer_state.pth')
    torch.save(model.state_dict(), model_path)
    torch.save(optimizer.state_dict(), optimizer_path)


def train_model(
    model: _Model,
    x: torch.Tensor,
    y: torch.Tensor,
    train_size: float,
    epochs: int,
    model_save_dir: str,
):
    """
    Provides a generic function to train any model.
    :param model:
    :param x:
    :param y:
    :param train_size:
    :param epochs:
    :param model_save_dir:
    :return:
    """
    train_x, train_y, cv_x, cv_y = split_data(x, y, train_size)

    model = model.to(device)
    optimizer = model.optimizer()
    criterion = model.criterion()
    acc_func = model.accuracy

    train_hist = []
    cv_hist    = []

    print(f"Training model: {model} for {epochs} epochs")
    for epoch in range(epochs):
        # set model to train mode
        model.train()

        optimizer.zero_grad()
        predictions = model(train_x)
        loss = criterion(predictions, train_y)

        loss.backward()
        optimizer.step()

        train_accuracy = acc_func(predictions, train_y)

        # set model to evaluate mode
        model.eval()

        with torch.no_grad():
            cv_predictions = model(cv_x)
            cv_loss = criterion(cv_predictions, cv_y)

            cv_accuracy = acc_func(cv_predictions, cv_y)

        train_hist.append((loss.item(), train_accuracy))
        cv_hist.append((cv_loss.item(), cv_accuracy))

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}: train loss = {loss}, cv loss = {cv_loss}")

    os.makedirs(model_save_dir, exist_ok=True)
    torch.save(model.state_dict(), model_save_dir)
    return train_hist, cv_hist
