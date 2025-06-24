import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from itertools import combinations

from utils.config import device
from tqdm import tqdm

from argparse import ArgumentParser
import json
import os


def build_listwise_race_x(race_x, mean, std, n):
    race_x = torch.tensor(race_x, device="cpu")
    race_x = (race_x - mean) / std
    m = race_x.size(0)

    groups = list(combinations(range(m), n))

    result = torch.stack([
        race_x[list(group)].reshape(-1) for group in groups
    ])
    return result


def build_listwise_x(data_x, mean, std, n):
    race_ids = list(dict(data_x).keys())
    result_x = []
    for race_id in tqdm(race_ids, desc="Building data x"):
        race = data_x[race_id]
        this_x = build_listwise_race_x(race, mean, std, n)
        result_x.append(this_x)
    return torch.cat(result_x, dim=0)


def build_listwise_race_y(race_y, n):
    race_y = torch.tensor(race_y, device="cpu")[:, 0]
    m = race_y.size(0)
    groups = list(combinations(range(m), n))
    result = torch.stack([
        race_y[list(group)].reshape(-1) for group in groups
    ])

    return torch.argmin(result, dim=1)


def build_listwise_y(data_y, n):
    race_ids = list(data_y.keys())
    result = []
    for race_id in tqdm(race_ids, desc="Building data y"):
        this_y = build_listwise_race_y(data_y[race_id], n)
        result.append(this_y)

    return torch.cat(result, dim=0)


def is_place(ranking, number):
    return (ranking <= 2) or (ranking <= 3 and number > 7)


def get_mean_std(data_x):
    race_ids = list(data_x.keys())
    grouped = []
    for race_id in race_ids:
        grouped.append(torch.tensor(data_x[race_id], device="cpu", dtype=torch.float32))
    grouped = torch.cat(grouped, dim=0)
    mean = torch.mean(grouped, dim=0)
    std = torch.std(grouped, dim=0)
    std[std == 0] = 1
    return mean, std


class ListwiseModel(nn.Module):
    def __init__(self, n):
        super(ListwiseModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(64 * n, 16 * n),
            nn.ReLU(),
            nn.Linear(16 * n,
            #           16 * n),
            # nn.ReLU(),
            # nn.Linear(16 * n,
                      4 * n),
            nn.ReLU(),
            nn.Linear(4 * n, n),
        )

    def forward(self, x):
        return self.model(x)


def shuffle_indices(m, cv_ratio=0.2):
    indices = torch.randperm(m)
    cv_idx = int(m * (1 - cv_ratio))
    return indices[:cv_idx], indices[cv_idx:]


def train_model(model, data_x, data_y, epochs, weight_decay):
    m = data_x.size(0)
    train_idx, cv_idx = shuffle_indices(m)
    train_x = data_x[train_idx]
    train_y = data_y[train_idx]
    cv_x = data_x[cv_idx]
    cv_y = data_y[cv_idx]

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=weight_decay)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(train_x)
        loss = criterion(output, train_y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            model.eval()
            cv_output = model(cv_x)
            cv_loss = criterion(cv_output, cv_y)
            print(f"Epoch {epoch + 1}/{epochs}: train loss = {loss.item()}, cv loss = {cv_loss.item()}")


def aggregate_scores(m, scores, n):
    scores_sum = torch.zeros(m, device=device)
    scores_count = torch.zeros(m, device=device)

    combos = list(combinations(range(m), n))

    for i, idxs in enumerate(combos):
        for j, horse_idx in enumerate(idxs):
            scores_sum[horse_idx] += scores[i, j]
            scores_count[horse_idx] += 1

    average_scores = scores_sum / scores_count
    return average_scores


def prediction(model, race_x, mean, std, n):
    m = race_x.shape[0]
    race_listwise = build_listwise_race_x(race_x, mean, std, n)
    race_listwise = race_listwise.float().to(device)
    model.eval()
    output = model(race_listwise)
    scores = torch.softmax(output, dim=1)
    aggregated_scores = aggregate_scores(m, scores, n)

    return aggregated_scores

def test_accuracy(model, data_x, data_y, mean, std, n):
    race_ids = list(data_x.keys())
    win = 0
    place = 0
    total = 0
    for race_id in tqdm(race_ids, desc="Testing accuracy"):
        this_y = data_y[race_id]
        scores = prediction(model, data_x[race_id], mean, std, n)
        pred_winner = torch.argmax(scores, dim=0).item()

        if this_y[pred_winner, 0] == 1:
            win += 1
        if is_place(this_y[pred_winner, 0], this_y[pred_winner, 4]):
            place += 1

        total += 1

    return win, place, total


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("path_name")
    return parser.parse_args()


def main():
    args = parse_args()
    path_name = args.path_name
    n = 4

    races_x = np.load(f"final_loaded_data/{path_name}/weighed/train/data_x.npz")
    races_y = np.load(f"final_loaded_data/{path_name}/weighed/train/data_y.npz")
    print(f"Training on {len(races_x)} number of races")

    test_x = np.load(f"final_loaded_data/{path_name}/weighed/test/data_x.npz")
    test_y = np.load(f"final_loaded_data/{path_name}/weighed/test/data_y.npz")

    mean, std = get_mean_std(races_x)
    result_x = build_listwise_x(races_x, mean, std, n)
    result_y = build_listwise_y(races_y, n)

    data_x = result_x.to(device).float()
    data_y = result_y.to(device).long()

    model = ListwiseModel(n).to(device).float()
    train_model(model, data_x, data_y, epochs=1000, weight_decay=0.005)

    win, place, total = test_accuracy(model, races_x, races_y, mean, std, n)
    print(f"=========== TRAIN =============")
    print(f"WIN accuracy: {win / total}")
    print(f"PLACE accuracy: {place / total}")
    test_win, test_place, test_total = test_accuracy(model, test_x, test_y, mean, std, n)
    print(f"============ TEST =============")
    print(f"WIN accuracy: {test_win / test_total}")
    print(f"PLACE accuracy: {test_place / test_total}")

    output_path = os.path.join("final_true_listwise_models", path_name)
    os.makedirs(output_path, exist_ok=True)

    accuracy = {
        "train_win": win,
        "train_place": place,
        "train_total": total,
        "train_win_acc": win / total,
        "train_place_acc": place / total,
        "test_win": test_win,
        "test_place": test_place,
        "test_total": test_total,
        "test_win_acc": test_win / test_total,
        "test_place_acc": test_place / test_total,
    }

    acc_file = os.path.join(output_path, f"accuracy.json")
    with open(acc_file, "w") as f:
        json.dump(accuracy, f)

    model_params = model.state_dict()
    torch.save(model_params, os.path.join(output_path, f"model_params.pth"))


if __name__ == "__main__":
    main()
