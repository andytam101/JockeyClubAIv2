import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from tqdm import tqdm
from utils.config import device
import random


class BetClassifier(nn.Module):
    def __init__(self):
        super(BetClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(12, 7),
            nn.ReLU(),
            nn.Linear(7, 3),
        )

    def forward(self, x):
        return self.model(x)


class BetBinary(nn.Module):
    def __init__(self):
        super(BetBinary, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(12, 6),
            nn.ReLU(),
            nn.Linear(6, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


def normalise_outputs(grouped_outputs):
    mean = torch.mean(grouped_outputs, dim=0)
    std = torch.std(grouped_outputs, dim=0)
    return (grouped_outputs - mean) / std


def split_train_cv(data_x, data_y):
    m = data_x.size(0)
    cv_proportion = 0.2
    cv_idx = m - int(m * cv_proportion)
    train_x = data_x[:cv_idx]
    train_y = data_y[:cv_idx]
    cv_x = data_x[cv_idx:]
    cv_y = data_y[cv_idx:]

    return train_x, train_y, cv_x, cv_y


def build_one_data_point(grouped_output, horse_nums, win_horses, place_horses, k=4):
    with torch.no_grad():
        # softmax for win and place chance, (X - mean) / std for (1 - relative ranking) and log score
        grouped_output[:, 3] = 1 - grouped_output[:, 3]
        grouped_output[:, :2] = torch.softmax(grouped_output[:, :2], dim=0)
        grouped_output[:, 2:] = normalise_outputs(grouped_output[:, 2:])
    output_sum = torch.sum(grouped_output, dim=1)  # can replace with more accurate estimator
    _, top_indices = torch.topk(output_sum, k=k, largest=True, sorted=True)

    this_x = grouped_output[top_indices].view(-1)
    top_horse_nums = horse_nums[top_indices]
    win_vector = torch.isin(top_horse_nums, win_horses)
    place_vector = torch.isin(top_horse_nums, place_horses)

    return this_x, torch.any(win_vector), win_vector.double(), place_vector.double(), top_horse_nums


def build_data_points(grouped_outputs, horse_nums, win_horses, place_horses, k=3):
    n = len(grouped_outputs)
    counter = 0

    data_x = torch.zeros((n, k * 4), dtype=torch.float64, device=device)
    data_binary = torch.zeros(n, dtype=torch.float64, device=device)
    data_wins = torch.zeros((n, k), dtype=torch.float64, device=device)
    data_places = torch.zeros((n, k), dtype=torch.float64, device=device)
    data_horse_nums = torch.zeros((n, k), device=device)

    race_keys = list(grouped_outputs.keys())
    random.shuffle(race_keys)
    for race_id in tqdm(race_keys, desc="Loading grouped data"):
        this_horse_nums = torch.tensor(horse_nums[race_id], device=device)
        this_win_horses = torch.tensor(win_horses[race_id], device=device)
        this_place_horses = torch.tensor(place_horses[race_id], device=device)
        this_x, this_binary, this_win, this_place, this_horse_nums = (
            build_one_data_point(grouped_outputs[race_id], this_horse_nums, this_win_horses, this_place_horses, k=k))
        data_x[counter] = this_x
        data_binary[counter] = this_binary
        data_wins[counter] = this_win
        data_places[counter] = this_place
        data_horse_nums[counter] = this_horse_nums

        counter += 1

    return data_x, data_binary.unsqueeze(-1), data_wins, data_places, data_horse_nums, race_keys


def get_accuracy(model, data_x, data_y):
    model.eval()
    output = model(data_x)
    predictions = output.argmax(dim=1)
    correct = (predictions == data_y).float()
    return torch.mean(correct)


def get_accuracy_binary(model, data_x, data_y, threshold=0.6):
    model.eval()
    output = model(data_x)
    guesses = (output > threshold).double().flatten()
    data_y = data_y.flatten()
    correct = (guesses == data_y).float()

    accuracy = torch.mean(correct)

    # split into True Positive, False Positive, False Negative and True Negative
    true_positive = torch.mean(((guesses == 1) & (data_y == 1)).float())
    false_positive = torch.mean(((guesses == 1) & (data_y == 0)).float())
    false_negative = torch.mean(((guesses == 0) & (data_y == 1)).float())
    true_negative = torch.mean(((guesses == 0) & (data_y == 0)).float())

    tp_index = torch.where((guesses == 1) & (data_y == 1))

    return accuracy, true_positive, false_positive, false_negative, true_negative, tp_index


def display_binary_accuracy(accuracy, true_positive, false_positive, false_negative, true_negative, *args):
    print(f"Overall accuracy: {accuracy:.4f}")
    print(f"True positive: {true_positive:.4f}")
    print(f"False positive: {false_positive:.4f}")
    print(f"False negative: {false_negative:.4f}")
    print(f"True negative: {true_negative:.4f}")
    print(f"Precision: {true_positive / (true_positive + false_positive):.4f}")
    print(f"Recall: {true_positive / (true_positive + false_negative):.4f}")


def train_binary_model(model, data_x, data_y, criterion, optimizer, threshold=0.5):
    data_x = data_x.detach()
    data_y = data_y.detach()
    epochs = 10000

    train_x, train_y, cv_x, cv_y = split_train_cv(data_x, data_y)

    model.eval()
    print(f"Initial train loss: {criterion(model(train_x), train_y).item():.4f}")
    print(f"Initial cv loss: {criterion(model(cv_x), cv_y).item():.4f}")
    print("=" * 100)
    bin_acc = get_accuracy_binary(model, train_x, train_y, threshold=threshold)
    bin_cv_acc = get_accuracy_binary(model, cv_x, cv_y, threshold=threshold)
    print(f"Initial train accuracy:")
    display_binary_accuracy(*bin_acc)
    print("-" * 100)
    print(f"Initial cv accuracy:")
    display_binary_accuracy(*bin_cv_acc)
    print("=" * 100)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(train_x)
        loss = criterion(pred, train_y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 1000 == 0:
            cv_pred = model(cv_x)
            cv_loss = criterion(cv_pred, cv_y)
            print(f"Epoch {epoch + 1} / {epochs}: train = {loss.item():.4f}, cv = {cv_loss.item():.4f}")
            model.eval()


    accuracy = get_accuracy_binary(model, train_x, train_y, threshold=threshold)
    print("=" * 100)
    print("Final train accuracy:")
    display_binary_accuracy(*accuracy)
    print("-" * 100)

    cv_accuracy = get_accuracy_binary(model, cv_x, cv_y, threshold=threshold)
    print("Final cv accuracy:")
    display_binary_accuracy(*cv_accuracy)

    true_positive = accuracy[1]
    false_positive = accuracy[2]
    precision = true_positive / (true_positive + false_positive)

    cv_true_positive = cv_accuracy[1]
    cv_true_negative = cv_accuracy[2]
    cv_precision = cv_true_positive / (cv_true_positive + cv_true_negative)

    return precision, cv_precision, accuracy[5][0]


def train_classifier_model(model, tp_x, tp_y, criterion, optimizer):
    tp_x = tp_x.detach()
    tp_y = tp_y.detach()
    train_x, train_y, cv_x, cv_y = split_train_cv(tp_x, tp_y)

    model.eval()
    epochs = 10000
    init_accuracy = get_accuracy(model, train_x, train_y)
    init_cv_accuracy = get_accuracy(model, cv_x, cv_y)
    print(f"Initial accuracy: {init_accuracy:.4f}")
    print(f"Initial cv accuracy: {init_cv_accuracy:.4f}")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(train_x)
        loss = criterion(pred, train_y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 500 == 0:
            model.eval()
            cv_pred = model(cv_x)
            cv_loss = criterion(cv_pred, cv_y)
            print(f"Epoch {epoch + 1} / {epochs}: train = {loss.item():.4f}, cv = {cv_loss.item():.4f}")


    model.eval()
    final_accuracy = get_accuracy(model, train_x, train_y)
    final_cv_accuracy = get_accuracy(model, cv_x, cv_y)
    print(f"Final accuracy: {final_accuracy:.4f}")
    print(f"Final cv accuracy: {final_cv_accuracy:.4f}")

    return final_accuracy, final_cv_accuracy


def convert_to_index_labels(one_hot_encoding):
    return torch.argmax(one_hot_encoding, dim=1)


def main():
    grouped_outputs = torch.load("../final_grouped_outputs/grouped_outputs.pt")
    horse_nums = np.load("../final_loaded_data/location_ST/weighed/train/horse_nums.npz")
    win_horses = np.load("../final_loaded_data/location_ST/weighed/train/wins.npz")
    place_horses = np.load("../final_loaded_data/location_ST/weighed/train/places.npz")

    data_x, data_binary, data_wins, data_places, data_horse_nums, race_ids = (
        build_data_points(grouped_outputs, horse_nums, win_horses, place_horses))

    threshold = 0.53

    model = BetBinary().to(device).double()
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    level_1_precision, level_1_cv_precision, tp_index = train_binary_model(model, data_x, data_binary, criterion, optimizer, threshold=threshold)

    print("=" * 100)
    print(f"True Positive count: {tp_index.size(0)}")
    print("=" * 100)

    tp_x = data_x[tp_index]
    tp_y = convert_to_index_labels(data_wins[tp_index])

    classifier = BetClassifier().to(device).double()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(classifier.parameters(), lr=0.1, momentum=0.9)
    level_2_accuracy, level_2_cv_accuracy = train_classifier_model(classifier, tp_x, tp_y, criterion, optimizer)

    overall_wins = level_1_precision * level_2_accuracy
    overall_losses = 1 - overall_wins

    overall_cv_wins = level_1_cv_precision * level_2_cv_accuracy
    overall_cv_losses = 1 - overall_cv_wins

    print("=" * 100)
    print("Overall train stats:")
    print(f"Overall wins: {overall_wins:.4f}")
    print(f"Overall loss: {overall_losses:.4f}")
    print(f"Overall accuracy: {overall_wins / (overall_wins + overall_losses):.4f}")
    print("-" * 100)
    print(f"Overall cv stats:")
    print(f"Overall wins: {overall_cv_wins:.4f}")
    print(f"Overall loss: {overall_cv_losses:.4f}")
    print(f"Overall accuracy: {overall_cv_wins / (overall_cv_wins + overall_cv_losses):.4f}")


if __name__ == "__main__":
    main()
