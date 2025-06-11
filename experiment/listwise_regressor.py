import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import os
import random

from sympy.physics.units import momentum
from tqdm import tqdm

from utils.config import device


class ListwiseRegressor(nn.Module):
    def __init__(self, k):
        super(ListwiseRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(k*5, k*2),
            nn.ReLU(),
            nn.Linear(k*2, k),
        )

    def forward(self, x):
        return self.model(x)


def normalise_outputs(grouped_outputs):
    mean = torch.mean(grouped_outputs, dim=0)
    std = torch.std(grouped_outputs, dim=0)
    return (grouped_outputs - mean) / std


def normalise_pointwise_outputs(pointwise_output):
    with torch.no_grad():
        # softmax for win and place chance, (X - mean) / std for (1 - relative ranking) and log score
        pointwise_output[:, 3] = 1 - pointwise_output[:, 3]
        pointwise_output[:, :2] = torch.softmax(pointwise_output[:, :2], dim=0)
        pointwise_output[:, 2:] = normalise_outputs(pointwise_output[:, 2:])


def build_one_data_point(pointwise_output, listwise_output, win_odds, horse_nums, k):
    horse_nums = torch.tensor(horse_nums, dtype=torch.int, device=device)
    normalise_pointwise_outputs(pointwise_output)
    output_sum = torch.sum(pointwise_output, dim=1)  # can replace with more accurate estimator
    _, top_indices = torch.topk(output_sum, k=k, largest=True, sorted=True)

    this_win_odds = win_odds[top_indices].view(-1)
    this_x = torch.cat([pointwise_output[top_indices].view(-1), this_win_odds], dim=0)
    this_y = listwise_output[top_indices]
    this_horse_nums = horse_nums[top_indices]

    return this_x, this_y, this_horse_nums


def build_data(pointwise_outputs, listwise_output, old_y, horse_nums, wins, k):
    race_ids = list(pointwise_outputs.keys())
    random.shuffle(race_ids)

    m = len(race_ids)
    data_x = torch.zeros((m, k * 5), dtype=torch.float64, device=device)
    data_y = torch.zeros((m, k), dtype=torch.float64, device=device)
    data_horse_nums = torch.zeros((m, k), dtype=torch.int, device=device)
    data_winners = torch.zeros((m, 1), dtype=torch.int, device=device)

    counter = 0
    for race_id in tqdm(race_ids, desc="Building data"):
        this_output = listwise_output[race_id][:, 1]
        this_win_odds = torch.tensor(old_y[race_id][:, 3], dtype=torch.float64, device=device)
        this_x, this_y, this_horse_nums = build_one_data_point(pointwise_outputs[race_id], this_output, this_win_odds, horse_nums[race_id], k)
        data_x[counter] = this_x
        data_y[counter] = this_y
        data_horse_nums[counter] = this_horse_nums
        if wins[race_id].shape[0] >= 1:
            data_winners[counter] = wins[race_id][0]
        else:
            data_winners[counter] = 0
        counter += 1

    return data_x, data_y, data_horse_nums, data_winners


def detach_tensors(*tensors):
    result = []
    for tensor in tensors:
        result.append(tensor.detach())
    return tuple(result)


def split_train_cv(data_x, data_y, cv_split=0.2):
    m = data_x.size(0)
    cv_idx = int(m * (1 - cv_split))

    train_x = data_x[:cv_idx]
    train_y = data_y[:cv_idx]
    cv_x = data_x[cv_idx:]
    cv_y = data_y[cv_idx:]

    return train_x, train_y, cv_x, cv_y


def train_listwise_regressor(model, data_x, data_y):
    data_x, data_y = detach_tensors(data_x, data_y)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    epochs = 10000

    train_x, train_y, cv_x, cv_y = split_train_cv(data_x, data_y)
    for epoch in range(epochs):
        model.train()
        pred = model(train_x)
        loss = criterion(pred, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        cv_pred = model(cv_x)
        cv_loss = criterion(cv_pred, cv_y)

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs}: train loss = {loss.item()}, cv loss = {cv_loss.item()}")


def get_accuracy(model, data_x, horse_nums, true_winners):
    pred = model(data_x)
    idx = torch.argmax(pred, dim=1)
    predicted_winner = horse_nums[torch.arange(horse_nums.size(0)), idx]
    true_winners = true_winners.flatten()

    correct = (true_winners == predicted_winner).float()
    return torch.mean(correct)


def main():
    grouped_pw_outputs = torch.load("../final_grouped_outputs/grouped_outputs.pt", map_location=device)
    grouped_results = torch.load("../final_grouped_outputs/grouped_results.pt", map_location=device)
    horse_nums = np.load("../final_loaded_data/location_ST/weighed/train/horse_nums.npz")
    wins = np.load("../final_loaded_data/location_ST/weighed/train/wins.npz")
    old_y = np.load("../final_loaded_data/location_ST/weighed/train/data_y.npz")

    data_x, data_y, horse_nums, winners = build_data(grouped_pw_outputs, grouped_results, old_y, horse_nums, wins, k=4)

    model = ListwiseRegressor(4).to(device).double()
    train_listwise_regressor(model, data_x, data_y)

    acc = get_accuracy(model, data_x, horse_nums, winners)
    print(f"Accuracy: {acc}")


if __name__ == "__main__":
    main()
