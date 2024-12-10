import torch
from tqdm import tqdm

from utils.config import device


def train_model(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    train_size: float,
    epochs: int
):
    """
    Provides a generic function to train any model.
    :param model:
    :param x:
    :param y:
    :param train_size:
    :param epochs:
    :return:
    """
    m = x.size(dim=0)

    cv_idx  = int(m * train_size)
    train_x = x[:cv_idx]
    train_y = y[:cv_idx]
    cv_x    = x[cv_idx:]
    cv_y    = y[cv_idx:]

    model = model.to(device)
    optimizer = model.optimizer()
    criterion = model.criterion()

    train_hist = []
    cv_hist    = []

    for epoch in tqdm(range(epochs), desc="Training AI"):
        # set model to train mode
        model.train()

        optimizer.zero_grad()
        predictions = model(train_x)
        loss = criterion(predictions, train_y)

        loss.backward()
        optimizer.step()

        train_accuracy = (torch.round(predictions) == train_y).float().mean().item()

        # set model to evaluate mode
        model.eval()

        with torch.no_grad():
            cv_predictions = model(cv_x)
            cv_loss = criterion(cv_predictions, cv_y)

            cv_accuracy = (torch.round(cv_predictions) == cv_y).float().mean().item()

        train_hist.append((loss.item(), train_accuracy))
        cv_hist.append((cv_loss.item(), cv_accuracy))

    return train_hist, cv_hist
