import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from utils.config import device as DEVICE

import json
import os

from argparse import ArgumentParser
import warnings

class ListwiseWinPlace(nn.Module):
    def __init__(self, n):
        super(ListwiseWinPlace, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4 * n, n),
            nn.ReLU(),
            nn.Linear(n, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


def is_place(ranking, total):
    return ((total > 7) & (ranking <= 3)) | (ranking <= 2)


def normalise_race_x(race_x):
    return torch.softmax(race_x, dim=0)


def build_race_x(race_x, n):
    normalised_x = normalise_race_x(race_x)
    pts_sum = torch.sum(normalised_x, dim=1)
    _, indices = torch.topk(pts_sum, n)

    result_x = race_x[indices]
    return result_x.flatten(), indices


def build_data_x(data_x, n):
    data_x_dict = dict(data_x)
    race_ids = list(data_x_dict.keys())
    res_top_n = {}

    result = torch.zeros((len(race_ids), 4 * n), device=DEVICE, dtype=torch.float32)

    counter = 0
    for race_id in race_ids:
        race_x, top_n_indices = build_race_x(data_x_dict[race_id], n)
        res_top_n[race_id] = top_n_indices
        result[counter] = race_x
        counter += 1

    result = result[:counter]
    return result, res_top_n


def build_race_y(race_y, top_n_idx, mode):
    race_y = torch.tensor(race_y, device=DEVICE, dtype=torch.float32)
    race_top_1 = race_y[top_n_idx[0]]

    if mode == "WIN":
        return race_top_1[0] == 1
    elif mode == "PLACE":
        return is_place(race_top_1[0], race_top_1[4])
    else:
        raise ValueError("Mode must be 'WIN' or 'PLACE'")

def build_data_y(data_y, top_n_indices, mode):
    race_ids = list(top_n_indices.keys())
    result = torch.zeros(len(race_ids), dtype=torch.float32, device=DEVICE)

    for (idx, race_id) in enumerate(race_ids):
        race_y = build_race_y(data_y[race_id], top_n_indices[race_id], mode)
        result[idx] = race_y

    return result


def detach_tensors(*tensors):
    result = []
    for tensor in tensors:
        result.append(tensor.detach())
    return tuple(result)


def shuffle_indices(m, cv_ratio=0.2):
    indices = torch.randperm(m)
    cv_idx = int(m * (1 - cv_ratio))
    return indices[:cv_idx], indices[cv_idx:]


def test_accuracy(model, data_x, data_y, threshold):
    model.eval()
    output = model(data_x).flatten()
    prediction = output > threshold

    tp = torch.count_nonzero((prediction == 1) & (data_y == 1))
    fn = torch.count_nonzero((prediction == 0) & (data_y == 1))
    fp = torch.count_nonzero((prediction == 1) & (data_y == 0))
    tn = torch.count_nonzero((prediction == 0) & (data_y == 0))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return precision.item(), recall.item(), accuracy.item()


def train_model(model, data_x, data_y, threshold):
    data_x, data_y = detach_tensors(data_x, data_y)
    m = data_x.size(0)
    train_idx, cv_idx = shuffle_indices(m)

    train_x = data_x[train_idx]
    train_y = data_y[train_idx]
    cv_x = data_x[cv_idx]
    cv_y = data_y[cv_idx]

    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001)

    epochs = 1000
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        pred = model(train_x).flatten()
        loss = criterion(pred, train_y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            model.eval()
            cv_pred = model(cv_x).flatten()
            cv_loss = criterion(cv_pred, cv_y)
            print(f"Epoch {epoch + 1}/{epochs}: train loss = {loss.item():.6f}, cv loss = {cv_loss.item():.6f}")

    train_precision, train_recall, train_acc = test_accuracy(model, train_x, train_y, threshold)
    cv_precision, cv_recall, cv_acc = test_accuracy(model, cv_x, cv_y, threshold)

    if train_recall == 0 or cv_recall == 0:
        return None
    else:
        return {
            "train_size": train_x.size(0),
            "train_precision": train_precision,
            "train_recall": train_recall,
            "train_acc": train_acc,
            "cv_size": cv_x.size(0),
            "cv_precision": cv_precision,
            "cv_recall": cv_recall,
            "cv_acc": cv_acc,
        }


def calculate_f_score(precision, recall, beta):
    return (1 + beta * beta) * (precision * recall) / (beta * beta * precision + recall)


def display_accuracy(precision, recall, acc, header):
    header_text = f" {header} ".center(50, "-")
    print(header_text)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Accuracy: {acc}")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("path_name")
    parser.add_argument("mode")
    return parser.parse_args()


def main():
    warnings.simplefilter(action="ignore", category=FutureWarning)
    args = parse_args()

    path_name = args.path_name
    mode = args.mode

    pw_outputs = torch.load(f"final_grouped_outputs/{path_name}/grouped_outputs.pt")
    races_y = np.load(f"final_loaded_data/{path_name}/weighed/train/data_y.npz")

    data_x, top_n_indices = build_data_x(pw_outputs, n=4)
    data_y = build_data_y(races_y, top_n_indices, mode)

    if mode == "WIN":
        threshold = torch.mean(data_y) * 1.1  # dynamic threshold, 10% above proportion of actual positives
    else:
        threshold = torch.mean(data_y)


    test_outputs = torch.load(f"final_grouped_outputs/{path_name}/test_grouped_outputs.pt")
    test_y = np.load(f"final_loaded_data/{path_name}/weighed/test/data_y.npz")
    test_x, test_top_n_indices = build_data_x(test_outputs, n=4)
    test_y = build_data_y(test_y, test_top_n_indices, mode)

    model = ListwiseWinPlace(n=4).to(DEVICE)

    accuracy = train_model(model, data_x, data_y, threshold)
    while accuracy is None:
        model = ListwiseWinPlace(n=4).to(DEVICE)
        accuracy = train_model(model, data_x, data_y, threshold)

    test_acc = test_accuracy(model, test_x, test_y, threshold)

    save_dir = f"final_trained_listwise/{path_name}/{mode.lower()}"
    os.makedirs(save_dir, exist_ok=True)

    accuracy.update({
        "test_precision": test_acc[0],
        "test_recall": test_acc[1],
        "test_acc": test_acc[2],
    })

    print(f"=" * 50)
    print(f"Model for {path_name}, {mode}")

    display_accuracy(
        accuracy["train_precision"], accuracy["train_recall"], accuracy["train_acc"], "Train"
    )
    display_accuracy(
        accuracy["cv_precision"], accuracy["cv_recall"], accuracy["cv_acc"], "CV"
    )

    display_accuracy(
        *test_acc, header="Test"
    )
    print(f"=" * 50)

    with open(os.path.join(save_dir, "accuracy.json"), "w") as f:
        json.dump(accuracy, f, indent=4)

    torch.save(model.state_dict(), os.path.join(save_dir, "model_params.pt"))


if __name__ == "__main__":
    main()
