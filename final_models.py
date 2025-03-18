import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import random
import matplotlib.pyplot as plt


class PointwiseModel(nn.Module):
    def __init__(self):
        super(PointwiseModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.model(x)

    def optimizer(self):
        return optim.SGD(self.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)


def calculate_cost(model, target_keys, data_x, data_y):
    criterion = nn.MSELoss()
    total_cost = 0
    for key in target_keys:
        this_x = torch.tensor(data_x[key], dtype=torch.float64, device="cuda")
        this_y = torch.tensor(data_y[key], dtype=torch.float64, device="cuda")
        this_y = (this_y[:, 0] / this_y[:, 4])

        this_mean = torch.mean(this_x, dim=0)
        this_std = torch.std(this_x, dim=0)
        this_std[this_std == 0] = 1
        normalized_x = (this_x - this_mean) / this_std

        output = model(normalized_x)
        output = output.flatten()
        this_y = this_y.flatten()
        loss = criterion(output, this_y)

        assert not torch.isnan(normalized_x).any(), "normalized x contains nan"
        try:
            assert not torch.isnan(output).any(), "output contains nan"
        except AssertionError:
            print(output)
            print("=" * 100)
            print(normalized_x)
            print("=" * 100)
            print(model(normalized_x))
            assert False, "output contains nan"
        total_cost += loss.item()

    return total_cost / len(target_keys)


def perform_prediction(model, normalized_x, horse_nums):
    model.eval()
    with torch.no_grad():
        predictions = model(normalized_x)
        predictions = torch.softmax(predictions, dim=0)
    horse_nums = horse_nums.tolist()
    horse_predictions = list(zip(horse_nums, predictions.tolist()))
    horse_predictions.sort(key=lambda x: x[1], reverse=False)

    return horse_predictions[0][0], horse_predictions[1][0]


def test_accuracy(model, data_x, horse_nums, winner_horses, place_horses):
    with torch.no_grad():
        winner_count = 0
        place_count = 0
        q_place_count = 0
        total_count = 0
        for key in data_x:
            this_x = torch.tensor(data_x[key], dtype=torch.float64, device="cuda")

            this_mean = torch.mean(this_x, dim=0)
            this_std = torch.std(this_x, dim=0)
            this_std[this_std == 0] = 1
            normalized_x = (this_x - this_mean) / this_std

            first, second = perform_prediction(model, normalized_x, horse_nums[key])

            if first in winner_horses[key]:
                winner_count += 1
            if first in place_horses[key]:
                place_count += 1
                if second in place_horses[key]:
                    q_place_count += 1
            total_count += 1

    return winner_count, place_count, q_place_count, total_count


def train_model(model, data, results, settings):
    data_x = data["data_x"]
    data_y = data["data_y"]
    # test_data = data["test_data"]
    horse_nums = data["horse_nums"]
    test_horse_nums = data["test_h_nums"]
    train_keys = settings["train_keys"]
    cv_keys = settings["cv_keys"]
    epochs = settings["epochs"]

    winner_horses = results["winner_horses"]
    place_horses = results["place_horses"]
    # test_winner_horses = results["test_winner_horses"]
    # test_place_horses = results["test_place_horses"]

    model.eval()
    initial_train_cost = calculate_cost(model, train_keys, data_x, data_y)
    initial_cv_cost = calculate_cost(model, cv_keys, data_x, data_y)
    print(f"Initial cost: {initial_train_cost}. CV cost: {initial_cv_cost}")

    # max_test_accuracy = 0
    # max_train_accuracy = 0

    train_hist = []
    cv_hist = []

    criterion = nn.MSELoss()
    optimizer = model.optimizer()
    for epoch in range(epochs):
        model.train()
        for key in train_keys:
            this_x = torch.tensor(data_x[key], dtype=torch.float64, device="cuda")
            this_y = torch.tensor(data_y[key], dtype=torch.float64, device="cuda")
            this_y = this_y[:, 0] / this_y[:, 4]

            this_mean = torch.mean(this_x, dim=0)
            this_std = torch.std(this_x, dim=0)
            this_std[this_std == 0] = 1
            normalized_x = (this_x - this_mean) / this_std

            output = model(normalized_x)
            output = output.flatten()
            this_y = this_y.flatten()
            loss = criterion(output, this_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        train_cost = calculate_cost(model, train_keys, data_x, data_y)
        cv_cost = calculate_cost(model, cv_keys, data_x, data_y)
        train_hist.append(train_cost)
        cv_hist.append(cv_cost)

        this_train_count = test_accuracy(model, data_x, horse_nums, winner_horses, place_horses)
        this_train_accuracy = this_train_count[0] / this_train_count[3]
        # this_test_count = test_accuracy(model, test_data, test_horse_nums, test_winner_horses, test_place_horses)
        # this_test_accuracy = this_test_count[0] / this_test_count[3]

        # if this_test_accuracy > max_test_accuracy:
        #     max_test_accuracy = this_test_accuracy
        #     max_train_accuracy = this_train_accuracy

        print(f"Epoch {epoch + 1}/{epochs} - train cost = {train_cost:.5f}, cv cost: {cv_cost:.5f}, train accuracy: {this_train_accuracy * 100:.2f}%")

    return train_hist, cv_hist


def get_train_cv_split(data_keys, cv_ratio=0.1):
    m = len(data_keys)
    train_size = round(m * (1 - cv_ratio))
    train_keys = set(random.sample(data_keys, train_size))
    cv_keys = set(data_keys) - set(train_keys)
    return list(train_keys), list(cv_keys)


def display_accuracy(header, accuracy):
    win_count, place_count, q_place_count, total_count = accuracy
    print(f"======== Stats for {header} ========")
    print(f"Winner accuracy: {win_count / total_count * 100:.2f}%")
    print(f"Place accuracy: {place_count / total_count * 100:.2f}%")
    print(f"Q Place accuracy: {q_place_count / total_count * 100:.2f}%")


def plot_cost_history_graph(train_hist, cv_hist):
    x_axis = list(range(len(train_hist)))
    plt.plot(x_axis, train_hist, label="Training data")
    plt.plot(x_axis, cv_hist, label="Test data")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.show()


def group_settings(train_keys, cv_keys, epochs):
    return {
        "train_keys": train_keys,
        "cv_keys": cv_keys,
        "epochs": epochs,
    }


def group_data(data_x, data_y, test_data, horse_nums, test_h_nums):
    return {
        "data_x": data_x,
        "data_y": data_y,
        "test_data": test_data,
        "horse_nums": horse_nums,
        "test_h_nums": test_h_nums,
    }


def group_results(winner_horses, place_horses, test_winner_horses, test_place_horses):
    return {
        "winner_horses": winner_horses,
        "place_horses": place_horses,
        "test_winner_horses": test_winner_horses,
        "test_place_horses": test_place_horses,
    }


def main():
    directory = "final_data"

    data_x_path = f"{directory}/data_x.npz"
    data_y_path = f"{directory}/data_y.npz"
    horse_nums_path = f"{directory}/horse_nums.npz"
    winner_horses_path = f"{directory}/wins.npz"
    place_horses_path = f"{directory}/places.npz"

    data_x = np.load(data_x_path)
    data_y = np.load(data_y_path)
    horse_nums = np.load(horse_nums_path)
    winner_horses = np.load(winner_horses_path)
    place_horses = np.load(place_horses_path)

    data = group_data(data_x, data_y, None, horse_nums, None)
    results = group_results(winner_horses, place_horses, None, None)

    model = PointwiseModel().to("cuda").double()
    train_keys, cv_keys = get_train_cv_split(list(data_x.keys()))
    settings = group_settings(train_keys, cv_keys, epochs=100)

    train_hist, cv_hist = train_model(model, data, results, settings)
    plot_cost_history_graph(train_hist, cv_hist)
    data_accuracy = test_accuracy(model, data_x, horse_nums, winner_horses, place_horses)
    display_accuracy("Train data", data_accuracy)


if __name__ == "__main__":
    main()
