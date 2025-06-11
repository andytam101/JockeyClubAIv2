import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from tqdm import tqdm
import os
import random
import warnings

from utils.config import device


class LayerOneBinary(nn.Module):
    def __init__(self, k):
        super(LayerOneBinary, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4 * k, 6),
            nn.ReLU(),
            nn.Linear(6, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class LayerTwoClassifier(nn.Module):
    def __init__(self, k):
        super(LayerTwoClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4 * k, 6),
            nn.ReLU(),
            nn.Linear(6, k),
        )

    def forward(self, x):
        return self.model(x)


class LayerThreeConfidence(nn.Module):
    def __init__(self, k):
        super(LayerThreeConfidence, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(k + 1, 2),
            nn.ReLU(),
            nn.Linear(2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


def detach_tensors(*tensors):
    result = []
    for tensor in tensors:
        result.append(tensor.detach())
    return tuple(result)


def split_train_cv(data_x, data_binary, data_wins, data_places, cv_split=0.2):
    m = data_x.size(0)
    cv_idx = int(m * (1 - cv_split))

    train_x = data_x[:cv_idx]
    train_binary = data_binary[:cv_idx]
    train_wins = data_wins[:cv_idx]
    train_places = data_places[:cv_idx]
    cv_x = data_x[cv_idx:]
    cv_binary = data_binary[cv_idx:]
    cv_wins = data_wins[cv_idx:]
    cv_places = data_places[cv_idx:]

    return train_x, train_binary, train_wins, train_places, cv_x, cv_binary, cv_wins, cv_places


def load_pointwise_outputs(filepath):
    pointwise_outputs = torch.load(filepath, map_location=device)
    return pointwise_outputs


def load_y(filepath):
    # ignore win odds
    horse_nums_path = os.path.join(filepath, "horse_nums.npz")
    wins_path = os.path.join(filepath, "wins.npz")
    place_path = os.path.join(filepath, "places.npz")

    horse_nums = np.load(horse_nums_path)
    wins = np.load(wins_path)
    places = np.load(place_path)

    return horse_nums, wins, places


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


def one_hot_to_index(one_hot_encoding):
    return torch.argmax(one_hot_encoding, dim=1)


def build_one_data_point(pointwise_output, horse_nums, wins, places, k):
    normalise_pointwise_outputs(pointwise_output)
    output_sum = torch.sum(pointwise_output, dim=1)  # can replace with more accurate estimator
    _, top_indices = torch.topk(output_sum, k=k, largest=True, sorted=True)

    this_x = pointwise_output[top_indices].view(-1)
    top_horse_nums = horse_nums[top_indices]
    win_vector = torch.isin(top_horse_nums, wins)
    place_vector = torch.isin(top_horse_nums, places)

    return this_x, torch.any(win_vector), win_vector, place_vector, top_horse_nums


def build_data(pointwise_outputs, horse_nums, wins, places, k, display=True):
    race_ids = list(pointwise_outputs.keys())
    random.shuffle(race_ids)

    m = len(race_ids)
    data_x = torch.zeros((m, k * 4), dtype=torch.float64, device=device)
    data_binary = torch.zeros(m, dtype=torch.float64, device=device)
    data_wins = torch.zeros((m, k), dtype=torch.float64, device=device)
    data_places = torch.zeros((m, k), dtype=torch.float64, device=device)

    counter = 0
    iterator = tqdm(race_ids, desc="Loading data") if display else race_ids
    for race_id in iterator:
        this_horse_nums = torch.tensor(horse_nums[race_id], dtype=torch.int, device=device)
        this_wins = torch.tensor(wins[race_id], dtype=torch.int, device=device)
        this_places = torch.tensor(places[race_id], dtype=torch.int, device=device)
        this_x, this_binary, this_win, this_place, this_top_horse_nums = (
            build_one_data_point(pointwise_outputs[race_id], this_horse_nums, this_wins, this_places, k))
        data_x[counter] = this_x
        data_binary[counter] = this_binary
        data_wins[counter] = this_win
        data_places[counter] = this_place
        counter += 1

    return data_x, data_binary, data_wins, data_places


def get_tp_fp_fn_tn(prediction, ground_truth, threshold):
    guesses = (prediction > threshold)
    tp = guesses & (ground_truth == 1)
    fp = guesses & (ground_truth == 0)
    fn = (~guesses) & (ground_truth == 1)
    tn = (~guesses) & (ground_truth == 0)

    return tp, fp, fn, tn


def calculate_precision_recall(tp, fp, fn):
    tp_proportion = torch.mean(tp.float())
    fp_proportion = torch.mean(fp.float())
    fn_proportion = torch.mean(fn.float())
    precision = tp_proportion / (tp_proportion + fp_proportion)
    recall = tp_proportion / (tp_proportion + fn_proportion)

    return precision, recall


def display_precision_recall(tp, fp, fn):
    tp_proportion = torch.mean(tp.float())
    fp_proportion = torch.mean(fp.float())
    fn_proportion = torch.mean(fn.float())
    precision = tp_proportion / (tp_proportion + fp_proportion)
    recall = tp_proportion / (tp_proportion + fn_proportion)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Percentage of bets: {tp_proportion + fp_proportion}")

    return precision, recall


def train_layer_one_model(model, train_x, train_y, cv_x, cv_y, threshold, epochs, display=True):
    train_x, train_y, cv_x, cv_y = detach_tensors(train_x, train_y, cv_x, cv_y)

    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(train_x).flatten()
        loss = criterion(pred, train_y)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            cv_pred = model(cv_x).flatten()
            cv_loss = criterion(cv_pred, cv_y).item()

        if display and (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch + 1} / {epochs}: train = {loss.item()}, cv = {cv_loss}")

    # get true positive and false positives
    model.eval()
    train_pred = model(train_x).flatten()
    train_tp, train_fp, train_fn, train_tn = get_tp_fp_fn_tn(train_pred, train_y, threshold)
    cv_pred = model(cv_x).flatten()
    cv_tp, cv_fp, cv_fn, cv_tn = get_tp_fp_fn_tn(cv_pred, cv_y, threshold)

    if display:
        print("=" * 100)
        print("Train result")
        train_precision, train_recall = display_precision_recall(train_tp, train_fp, train_fn)
        print("-" * 100)
        print("CV result")
        cv_precision, cv_recall = display_precision_recall(cv_tp, cv_fp, cv_fn)
        print("=" * 100)
    else:
        train_precision, train_recall = calculate_precision_recall(train_tp, train_fp, train_fn)
        cv_precision, cv_recall = calculate_precision_recall(cv_tp, cv_fn, cv_tn)

    return train_tp, train_fp, cv_tp, cv_fp, train_precision, train_recall, cv_precision, cv_recall


def extract_layer_2_correct(predictions, ground_truth):
    pred_indices = torch.argmax(predictions, dim=1)
    correct = pred_indices == ground_truth

    return correct


def filter_tensor(t, booleans):
    # pre: they are both 1d
    idx = torch.where(booleans)[0]
    return t[idx]


def train_layer_two_model(model, train_x, train_y, cv_x, cv_y, epochs, display=True):
    train_x, train_y, cv_x, cv_y = detach_tensors(train_x, train_y, cv_x, cv_y)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    train_y = one_hot_to_index(train_y)
    cv_y = one_hot_to_index(cv_y)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(train_x)
        loss = criterion(pred, train_y)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            cv_pred = model(cv_x)
            cv_loss = criterion(cv_pred, cv_y).item()

        if display and (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch + 1} / {epochs}: train = {loss.item()}, cv = {cv_loss}")

    # get correct ones
    model.eval()
    train_pred = model(train_x)
    train_correct = extract_layer_2_correct(train_pred, train_y)
    cv_pred = model(cv_x)
    cv_correct = extract_layer_2_correct(cv_pred, cv_y)

    train_acc = torch.mean(train_correct.float()).item()
    cv_acc = torch.mean(cv_correct.float()).item()

    if display:
        print("=" * 100)
        print(f"Train accuracy: {train_acc}")
        print(f"CV accuracy: {cv_acc}")
        print("=" * 100)

    return (
        train_pred[torch.where(train_correct)[0]],
        train_pred[torch.where(~train_correct)[0]],
        cv_pred[torch.where(cv_correct)[0]],
        cv_pred[torch.where(~cv_correct)[0]],
        train_acc,
        cv_acc
    )


def sort_tensor(t):
    return torch.sort(t, dim=1, descending=True).values


def build_layer_3_data(l2_correct_train, l2_incorrect_train, l2_correct_cv, l2_incorrect_cv, l1_fp_train, l1_fp_cv):
    l2_correct_idx = torch.argmax(l2_correct_train, dim=1).unsqueeze(1)
    l2_correct_train = torch.cat([l2_correct_idx, sort_tensor(l2_correct_train)], dim=1)
    l2_incorrect_idx = torch.argmax(l2_incorrect_train, dim=1).unsqueeze(1)
    l2_incorrect_train = torch.cat([l2_incorrect_idx, sort_tensor(l2_incorrect_train)], dim=1)
    l2_correct_cv_idx = torch.argmax(l2_correct_cv, dim=1).unsqueeze(1)
    l2_correct_cv = torch.cat([l2_correct_cv_idx, sort_tensor(l2_correct_cv)], dim=1)
    l2_incorrect_cv_idx = torch.argmax(l2_incorrect_cv, dim=1).unsqueeze(1)
    l2_incorrect_cv = torch.cat([l2_incorrect_cv_idx, sort_tensor(l2_incorrect_cv)], dim=1)
    l1_fp_train_idx = torch.argmax(l1_fp_train, dim=1).unsqueeze(1)
    l1_fp_train = torch.cat([l1_fp_train_idx, sort_tensor(l1_fp_train)], dim=1)
    l1_fp_cv_idx = torch.argmax(l1_fp_cv, dim=1).unsqueeze(1)
    l1_fp_cv = torch.cat([l1_fp_cv_idx, sort_tensor(l1_fp_cv)], dim=1)

    train_correct = l2_correct_train.size(0)
    train_incorrect = l2_incorrect_train.size(0) + l1_fp_train.size(0)
    cv_correct = l2_correct_cv.size(0)
    cv_incorrect = l2_incorrect_cv.size(0) + l1_fp_cv.size(0)
    l3_train_x = torch.cat([l2_correct_train, l2_incorrect_train, l1_fp_train], dim=0)
    l3_train_y = torch.cat([torch.ones(train_correct, device=device, dtype=torch.float64),
                            torch.zeros(train_incorrect, device=device, dtype=torch.float64)], dim=0)
    l3_cv_x = torch.cat([l2_correct_cv, l2_incorrect_cv, l1_fp_cv], dim=0)
    l3_cv_y = torch.cat([torch.ones(cv_correct, device=device, dtype=torch.float64),
                         torch.zeros(cv_incorrect, device=device, dtype=torch.float64)], dim=0)

    return l3_train_x, l3_train_y, l3_cv_x, l3_cv_y


def train_layer_three_model(model, train_x, train_y, cv_x, cv_y, threshold, epochs):
    train_x, train_y, cv_x, cv_y = detach_tensors(train_x, train_y, cv_x, cv_y)

    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(train_x).flatten()
        loss = criterion(pred, train_y)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            cv_pred = model(cv_x).flatten()
            cv_loss = criterion(cv_pred, cv_y)

        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch + 1} / {epochs}: train = {loss.item()}, cv = {cv_loss}")

    model.eval()
    train_pred = model(train_x)
    train_accuracy = get_tp_fp_fn_tn(train_pred, train_y, threshold)
    print("=" * 100)
    print("Train result: ")
    display_precision_recall(*train_accuracy[:-1])
    cv_pred = model(cv_x)
    print("-" * 100)
    print("CV result: ")
    cv_accuracy = get_tp_fp_fn_tn(cv_pred, cv_y, threshold)
    display_precision_recall(*cv_accuracy[:-1])
    print("=" * 100)


def train_layers(top_k, layer_1_threshold, display):
    # configs
    layer_1_epochs = 10000
    layer_2_epochs = 2000
    layer_3_threshold = 0.35
    layer_3_epochs = 10000

    pointwise_outputs = load_pointwise_outputs("../final_grouped_outputs/grouped_outputs.pt")
    horse_nums, wins, places = load_y("../final_loaded_data/location_ST/weighed/train/")

    data = build_data(pointwise_outputs, horse_nums, wins, places, top_k, display=display)
    train_x, train_binary, train_wins, train_places, cv_x, cv_binary, cv_wins, cv_places \
        = split_train_cv(*data, cv_split=0.2)

    layer_one_binary_model = LayerOneBinary(top_k).to(device).double()
    train_tp, train_fp, cv_tp, cv_fp, train_precision, train_recall, cv_precision, cv_recall = train_layer_one_model(
        layer_one_binary_model, train_x, train_binary, cv_x, cv_binary, layer_1_threshold, layer_1_epochs, display=display)

    l2_train_x = filter_tensor(train_x, train_tp)
    l2_train_y = filter_tensor(train_wins, train_tp)
    l2_cv_x = filter_tensor(cv_x, cv_tp)
    l2_cv_y = filter_tensor(cv_wins, cv_tp)

    layer_two_classifier_model = LayerTwoClassifier(top_k).to(device).double()
    train_correct, train_incorrect, cv_correct, cv_incorrect, train_acc, cv_acc = train_layer_two_model(
        layer_two_classifier_model, l2_train_x, l2_train_y, l2_cv_x, l2_cv_y, layer_2_epochs, display=display
    )

    # l1_fp_train = filter_tensor(train_x, train_fp)
    # l1_fp_cv = filter_tensor(cv_x, cv_fp)

    # layer_two_classifier_model.eval()
    # l1_fp_train = layer_two_classifier_model(l1_fp_train)
    # l1_fp_cv = layer_two_classifier_model(l1_fp_cv)

    overall_train_acc = train_acc * train_precision
    overall_cv_acc = cv_acc * cv_precision

    if display:
        print("Overall:")
        print(f"Train accuracy: {overall_train_acc}")
        print(f"CV accuracy: {overall_cv_acc}")

    # l3_train_x, l3_train_y, l3_cv_x, l3_cv_y = build_layer_3_data(train_correct, train_incorrect, cv_correct,
    #                                                               cv_incorrect, l1_fp_train, l1_fp_cv)
    #
    # layer_three_confidence_model = LayerThreeConfidence(top_k).to(device).double()
    # train_layer_three_model(layer_three_confidence_model, l3_train_x, l3_train_y, l3_cv_x, l3_cv_y, layer_3_threshold,
    #                         layer_3_epochs)

    return overall_train_acc, overall_cv_acc


def main():
    highest_acc = 0
    highest_cv_acc = 0
    iterations = 35
    top_k = 3
    layer_1_threshold = 0.53

    for i in range(iterations):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            train_acc, cv_acc = train_layers(top_k, layer_1_threshold, display=False)

        if abs(train_acc - cv_acc) < 0.05:
            this_acc = (train_acc + cv_acc) / 2
            if this_acc > highest_acc:
                highest_acc = this_acc
                highest_cv_acc = cv_acc

        print(f"Iteration {i + 1}/{iterations}: highest accuracy = {highest_acc}, highest cv accuracy = {highest_cv_acc}")


if __name__ == '__main__':
    main()
