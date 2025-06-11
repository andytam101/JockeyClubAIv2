import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from math import factorial
from itertools import permutations

from utils.config import device
from tqdm import tqdm


class BinaryClassifier(nn.Module):
    def __init__(self, n, layer_2_size):
        super(BinaryClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4 * n, layer_2_size),
            nn.ReLU(),
            nn.Linear(layer_2_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class DataPaths:
    def __init__(
        self,
        pw_outputs,
        race_outputs,
        horse_nums
    ):
        self.pw_outputs = pw_outputs
        self.race_outputs = race_outputs
        self.horse_nums = horse_nums

def normalise_outputs(outputs):
    mean = torch.mean(outputs, dim=0)
    std = torch.std(outputs, dim=0)
    return (outputs - mean) / std

def normalise_pw_outputs(pw_outputs):
    result = torch.zeros_like(pw_outputs)
    with torch.no_grad():
        # softmax for win and place chance, (X - mean) / std for (1 - relative ranking) and log score
        result[:, :2] = torch.softmax(pw_outputs[:, :2], dim=0)
        result[:, 2:] = normalise_outputs(pw_outputs[:, 2:])
    return result


def to_tensor(arr, dtype=torch.float64):
    return torch.tensor(arr, dtype=dtype, device=device)


def shuffle_data(data_x, data_y):
    perm = torch.randperm(len(data_x))
    return data_x[perm], data_y[perm]


def split_train_cv(data, cv_ratio):
    cv_idx = int(data.size(0) * (1 - cv_ratio))
    train_data = data[:cv_idx]
    cv_data = data[cv_idx:]
    return train_data, cv_data


def train_model(model, x, y, epochs, criterion, optimizer, cv_ratio):
    x, y = shuffle_data(x, y)
    train_x, cv_x = split_train_cv(x, cv_ratio)
    train_y, cv_y = split_train_cv(y, cv_ratio)

    for epoch in range(epochs):
        model.train()

        optimizer.zero_grad()
        pred = model(train_x).flatten()
        loss = criterion(pred, train_y)
        loss.backward()
        optimizer.step()

        model.eval()
        cv_pred = model(cv_x).flatten()
        cv_loss = criterion(cv_pred, cv_y)

        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch + 1}/{epochs}: train loss = {loss.item()}, cv loss = {cv_loss.item()}")


class TopKInNModel:
    def __init__(self, n, k, data_paths):
        self.n = n
        self.k = k
        self.layer_1 = BinaryClassifier(n, n).to(device).double()
        self.layer_2 = BinaryClassifier(n, 4 * n).to(device).double()

        self.race_outputs = np.load(data_paths.race_outputs)
        self.pw_outputs = torch.load(data_paths.pw_outputs, map_location=device)
        self.horse_nums = np.load(data_paths.horse_nums)

    def load_pw_output_in_tensor(self, tensor, idx, pw_outputs):
        normalised_pw_outputs = normalise_pw_outputs(pw_outputs)
        sum_normalised_outputs = torch.sum(normalised_pw_outputs, dim=1)
        _, top_k_idx = torch.topk(sum_normalised_outputs, k=self.n, dim=0)
        top_k_normalised_score = normalised_pw_outputs[top_k_idx]
        tensor[idx] = top_k_normalised_score.flatten()

        return top_k_idx

    def is_valid_race(self, tensor):
        return tensor.size(0) >= self.n

    def is_valid_race_id(self, race_id):
        return self.is_valid_race(self.pw_outputs[race_id])

    def build_layer_1_data(self):
        pw_outputs = self.pw_outputs
        race_outputs = self.race_outputs
        race_ids = list(pw_outputs.keys())
        m = len(race_ids)

        layer_1_input = torch.zeros((m, 4 * self.n), dtype=torch.float64, device=device)
        layer_1_output = torch.zeros(m, dtype=torch.float64, device=device)

        counter = 0
        for race_id in race_ids:
            this_race_output = to_tensor(race_outputs[race_id])
            if not self.is_valid_race(pw_outputs[race_id]):
                continue
            top_n_idx = self.load_pw_output_in_tensor(layer_1_input, counter, pw_outputs[race_id])
            top_k_ranking = this_race_output[top_n_idx][:, 0]
            layer_1_output[counter] = torch.any(top_k_ranking <= self.k)
            counter += 1

        layer_1_input = layer_1_input[:counter]
        layer_1_output = layer_1_output[:counter]

        return layer_1_input, layer_1_output

    def build_layer_2_data_for_race(self, top_n_vector, race_ranking):
        k = self.k
        n = self.n

        input_size = factorial(n)
        sub_input_size = input_size // n

        result_input = torch.zeros((input_size, 4 * n), dtype=torch.float64, device=device)
        result_output = torch.zeros((input_size, 1), dtype=torch.float64, device=device)

        for i in range(n):
            competitors = set(range(n)) - {i}
            all_ps = list(permutations(competitors))
            for j in range(sub_input_size):
                row_index = i * sub_input_size + j
                first_part = top_n_vector[4 * i:4 * (i + 1)]
                result_input[row_index, :4] = first_part
                this_p = all_ps[j]
                for k, p in enumerate(this_p):
                    result_input[row_index, 4 * (k + 1): 4 * (k + 2)] = top_n_vector[4 * p : 4 * (p + 1)]

            if race_ranking is not None:
                result_output[i * sub_input_size : (i + 1) * sub_input_size] = race_ranking[i] <= k

        return result_input, (result_output if race_ranking is not None else None)

    def build_layer_2_data(self, number_of_races):
        n = self.n
        pw_outputs = self.pw_outputs
        race_outputs = self.race_outputs
        # horse_nums = self.horse_nums

        race_ids = list(pw_outputs.keys())

        m = factorial(n) * number_of_races
        layer_2_input = torch.zeros((m, 4 * n), dtype=torch.float64, device=device)
        layer_2_output = torch.zeros(m, dtype=torch.float64, device=device)

        counter = 0

        for race_id in race_ids:
            this_race_output = to_tensor(race_outputs[race_id])
            # this_horse_nums = to_tensor(horse_nums[race_id], dtype=torch.float64)
            if not self.is_valid_race(pw_outputs[race_id]):
                continue

            # extract top n horses
            top_n_vector = torch.zeros((1, 4 * n), dtype=torch.float64, device=device)
            top_n_idx = self.load_pw_output_in_tensor(top_n_vector, 0, pw_outputs[race_id])
            race_input, race_output = self.build_layer_2_data_for_race(top_n_vector[0], this_race_output[top_n_idx][:, 0])

            start_idx = counter * factorial(n)
            end_idx = (counter + 1) * factorial(n)

            layer_2_input[start_idx:end_idx, :] = race_input
            layer_2_output[start_idx:end_idx] = race_output.flatten()

            counter += 1

        return layer_2_input, layer_2_output

    def train_layer_1_model(self, layer_1_input, layer_1_output, epochs):
        train_model(
            self.layer_1,
            layer_1_input,
            layer_1_output,
            epochs,
            criterion=nn.BCELoss(),
            optimizer=optim.SGD(self.layer_1.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4),
            cv_ratio=0.2
        )

    def train_layer_2_model(self, layer_2_input, layer_2_output, epochs):
        train_model(
            self.layer_2,
            layer_2_input,
            layer_2_output,
            epochs,
            criterion=nn.BCELoss(),
            optimizer=optim.SGD(self.layer_2.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4),
            cv_ratio=0.2
        )

    def accuracy(self, layer_one, data_input, data_output, threshold=0.5):
        model = self.layer_1 if layer_one else self.layer_2
        model.eval()
        output = model(data_input).flatten()
        pred = (output >= threshold).float()
        return torch.mean((pred == data_output).float()).item()

    def train_models(self):
        print("Loading layer 1 data")
        layer_1_input, layer_1_output = self.build_layer_1_data()
        print("Loading layer 2 data")
        layer_2_input, layer_2_output = self.build_layer_2_data(layer_1_input.size(0))
        print("=" * 100)
        print("Training layer 1 model")
        self.train_layer_1_model(layer_1_input, layer_1_output, epochs=10000)
        print("=" * 100)
        print("Training layer 2 model")
        self.train_layer_2_model(layer_2_input, layer_2_output, epochs=1000)

        layer_1_accuracy = self.accuracy(True, layer_1_input, layer_1_output)
        layer_2_accuracy = self.accuracy(False, layer_2_input, layer_2_output)

        print(f"Accuracies: ")
        print(layer_1_accuracy)
        print(layer_2_accuracy)

    def predict(self, race_id):
        pw_output = self.pw_outputs[race_id]
        return self.perform_prediction(pw_output)

    def perform_prediction(self, pw_outputs):
        top_n_vector = torch.zeros((1, 4 * self.n), dtype=torch.float64, device=device)
        top_n_idx = self.load_pw_output_in_tensor(top_n_vector, 0, pw_outputs)

        top_n_vector = top_n_vector[0]
        layer_1_scalar = self.predict_layer_1(top_n_vector)
        layer_2_vector = self.predict_layer_2(top_n_vector)

        # combined = layer_1_scalar * layer_2_vector
        return layer_1_scalar, layer_2_vector, top_n_idx

    def predict_layer_1(self, top_n_vector):
        self.layer_1.eval()
        layer_1_output = self.layer_1(top_n_vector)

        return layer_1_output.item()

    def predict_layer_2(self, top_n_vector):
        n = self.n
        layer_2_input, _ = self.build_layer_2_data_for_race(top_n_vector, None)
        self.layer_2.eval()
        layer_2_output = self.layer_2(layer_2_input)

        sub_size = factorial(n - 1)
        grouped_outputs = layer_2_output.view(n, sub_size)
        scores = grouped_outputs.mean(dim=1)

        return scores


def overall_accuracy(model, horse_nums, wins, race_output, threshold):
    correct = 0
    total = 0
    winnings = 0
    for race_id in tqdm(horse_nums.keys()):
        if not model.is_valid_race_id(race_id):
            continue
        layer_1, layer_2, top_n_idx = model.predict(race_id)
        this_horse_nums = to_tensor(horse_nums[race_id], dtype=torch.int)
        this_top_n_horses = this_horse_nums[top_n_idx]
        this_wins = wins[race_id]

        this_win_odds = to_tensor(race_output[race_id][:, 3])[top_n_idx]
        expected_values = this_win_odds * layer_2

        highest_idx = torch.argmax(expected_values, dim=0)

        if layer_1 < threshold:
            continue

        pred_winner = this_top_n_horses[highest_idx].item()
        if pred_winner in this_wins:
            winnings += this_win_odds[highest_idx].item()
            correct += 1
        total += 1

    profit = round(winnings - total, 2)

    return correct, total, len(horse_nums), profit


def main():
    path_name = "location_ST_1600"

    data_paths = DataPaths(
        f"final_grouped_outputs/{path_name}/grouped_outputs.pt",
        f"final_loaded_data/{path_name}/weighed/train/data_y.npz",
        f"final_loaded_data/{path_name}/weighed/train/horse_nums.npz"
    )

    horse_nums = np.load(f"final_loaded_data/{path_name}/weighed/train/horse_nums.npz")
    wins = np.load(f"final_loaded_data/{path_name}/weighed/train/wins.npz")
    places = np.load(f"final_loaded_data/{path_name}/weighed/train/places.npz")
    race_output = np.load(f"final_loaded_data/{path_name}/weighed/train/data_y.npz")

    model = TopKInNModel(3, 1, data_paths)
    model.train_models()

    acc = overall_accuracy(model, horse_nums, wins, race_output, threshold=0.55)
    print("=" * 100)
    print(f"Overall Accuracy: {acc}")
    if acc[1] > 0:
        print(f"Overall Accuracy: {acc[0] / acc[1]}")


if __name__ == '__main__':
    main()
