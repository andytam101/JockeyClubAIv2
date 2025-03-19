import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import os


class KFoldTrainer:
    def __init__(self, model_factory, data, results, k_folds):
        self.model_factory = model_factory
        self.model = model_factory()
        self.data = data
        self.results = results

        self.k_folds = k_folds
        self.folds = []

        self.trained_model_dicts = []
        self.cv_costs = []

    def fold_data(self, data_keys):
        data_keys_list = list(data_keys)
        random.shuffle(data_keys_list)

        n = len(data_keys_list)
        step_count = n // self.k_folds
        for i in range(0, n, step_count):
            self.folds.append(data_keys_list[i:i + step_count])


    def train_model(self, batch_size=1024, max_epoch=200, overfitting_threshold=3):
        data_x = self.data["data_x"]
        self.fold_data(data_x.keys())

        for number in range(self.k_folds):
            self.model = self.model_factory()
            print(f"Training fold {number + 1}")
            _, this_cv_loss, _, _ = self.train_one_fold(number, batch_size, max_epoch, overfitting_threshold)
            self.trained_model_dicts.append(copy.deepcopy(self.model.state_dict()))
            self.cv_costs.append(this_cv_loss)

        idx = np.argmin(self.cv_costs).item()
        best_model = self.trained_model_dicts[idx]

        self.model = self.model_factory()
        self.model.load_state_dict(best_model)

        print(f"\n======= Testing final accuracy with fold {idx} =======")
        train_accuracy = self.test_accuracy(is_test=False)
        test_accuracy = self.test_accuracy(is_test=True)
        display_accuracy("Train data", train_accuracy)
        display_accuracy("Test data", test_accuracy)

    def calculate_loss(self, data_x, data_y):
        model = self.model
        criterion = model.criterion()

        with torch.no_grad():
            output = model(data_x)
            output = output.flatten()
            data_y = data_y.flatten()
            loss = criterion(output, data_y)

        return loss.item()


    def train_one_fold(self, fold_number, batch_size=1024, max_epoch=200, overfitting_threshold=3):
        train_keys, cv_keys = self.get_train_cv_keys(fold_number)
        train_x, train_y = self.concatenate_data(train_keys)
        cv_x, cv_y = self.concatenate_data(cv_keys)

        model = self.model
        criterion = self.model.criterion()
        optimizer = self.model.optimizer()

        previous_train_loss = torch.inf
        previous_cv_loss = torch.inf
        overfitting_counter = 0

        train_hist = []
        cv_hist = []

        for epoch in range(max_epoch):
            model.train()
            # training by batches
            for i in range(0, train_x.size(0), batch_size):
                this_x = train_x[i:i + batch_size]
                this_y = train_y[i:i + batch_size]

                prediction = model(this_x)
                prediction = prediction.flatten()
                this_y = this_y.flatten()
                loss = criterion(prediction, this_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            loss = self.calculate_loss(train_x, train_y)
            cv_loss = self.calculate_loss(cv_x, cv_y)

            train_hist.append(float(loss))
            cv_hist.append(float(cv_loss))

            # auto-detecting to quit or not
            if loss < previous_train_loss and cv_loss >= previous_cv_loss:
                overfitting_counter += 1
            else:
                overfitting_counter = 0

            previous_train_loss = loss
            previous_cv_loss = cv_loss

            if (epoch + 1) % 10 == 0:
                # train_acc, _, _, train_count = self.test_accuracy(is_test=False)
                # test_acc, _, _, test_count = self.test_accuracy(is_test=True)
                #
                # train_acc = train_acc / train_count
                # test_acc = test_acc / test_count

                # train acc = {train_acc:.4f}, test acc = {test_acc:.4f}
                print(f"Epoch {epoch + 1} train loss = {loss:.6f}, cv loss = {cv_loss:.6f}")

            if overfitting_counter >= overfitting_threshold:
                print(f"Overfitting detected. Stopped training at epoch {epoch + 1}.")
                break
        else:
            print(f"Epoch {max_epoch} reached without detecting overfitting.")

        return previous_train_loss, previous_cv_loss, train_hist, cv_hist

    def get_train_cv_keys(self, fold_number):
        cv_keys = self.folds[fold_number]
        train_keys = []
        for i in range(len(self.folds)):
            if i == fold_number:
                continue
            train_keys += self.folds[i]
        return train_keys, cv_keys


    def concatenate_data(self, data_keys):
        format_y = self.model.format_y
        data_x = self.data["data_x"]
        data_y = self.data["data_y"]

        total_size = 0
        for race_id in data_keys:
            total_size += data_x[race_id].shape[0]

        result_x = torch.zeros((total_size, 64), dtype=torch.float64, device="cuda")
        result_y = torch.zeros((total_size, 1), dtype=torch.float64, device="cuda")

        counter = 0
        for race_id in tqdm(data_keys, desc="Concatenating data"):
            this_size = data_x[race_id].shape[0]
            race_x = torch.tensor(data_x[race_id], dtype=torch.float64, device="cuda")
            result_x[counter: counter + this_size] = normalise_by_race(race_x)
            result_y[counter: counter + this_size] = format_y(torch.tensor(data_y[race_id], dtype=torch.float64, device="cuda"))
            counter += this_size

        return result_x, result_y

    def test_accuracy(self, is_test):
        model = self.model

        if is_test:
            data_x = self.data["test_data"]
            horse_nums = self.data["test_h_nums"]
            winner_horses = self.results["test_winner_horses"]
            place_horses = self.results["test_place_horses"]
        else:
            data_x = self.data["data_x"]
            horse_nums = self.data["horse_nums"]
            winner_horses = self.results["winner_horses"]
            place_horses = self.results["place_horses"]

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

                first, second = self.perform_prediction(normalized_x, horse_nums[key])

                if first in winner_horses[key]:
                    winner_count += 1
                if first in place_horses[key]:
                    place_count += 1
                    if second in place_horses[key]:
                        q_place_count += 1
                total_count += 1

        return winner_count, place_count, q_place_count, total_count

    def perform_prediction(self, normalized_x, horse_nums):
        model = self.model
        model.eval()
        with torch.no_grad():
            predictions = model(normalized_x)
            predictions = torch.softmax(predictions, dim=0)
        horse_nums = horse_nums.tolist()
        horse_predictions = list(zip(horse_nums, predictions.tolist()))
        horse_predictions.sort(key=lambda x: x[1], reverse=(not self.model.reverse_points))

        return horse_predictions[0][0], horse_predictions[1][0]


def normalise_by_race(race_data):
    mean = torch.mean(race_data, dim=0)
    std = torch.std(race_data, dim=0)
    std[std == 0] = 1
    return (race_data - mean) / std


def load_data(train_directory, test_directory):
    train_dir = os.path.join(train_directory, "train")
    test_dir = os.path.join(test_directory, "test")

    train_x = np.load(os.path.join(train_dir, "data_x.npz"))
    train_y = np.load(os.path.join(train_dir, "data_y.npz"))
    test_x = np.load(os.path.join(test_dir, "data_x.npz"))

    train_horse_nums = np.load(os.path.join(train_dir, "horse_nums.npz"))
    test_horse_nums = np.load(os.path.join(test_dir, "horse_nums.npz"))

    train_wins = np.load(os.path.join(train_dir, "wins.npz"))
    train_places = np.load(os.path.join(train_dir, "places.npz"))
    test_wins = np.load(os.path.join(test_dir, "wins.npz"))
    test_places = np.load(os.path.join(test_dir, "places.npz"))

    data = group_data(train_x, train_y, test_x, train_horse_nums, test_horse_nums)
    results = group_results(train_wins, train_places, test_wins, test_places)
    return data, results


def main():
    data, results = load_data("distance_1600/weighed", "distance_1600/weighed")
    trainer = KFoldTrainer(lambda: PointwiseModel().to("cuda").double(), data, results, k_folds=5)
    trainer.train_model(batch_size=512, max_epoch=1000, overfitting_threshold=5)


class PointwiseModel(nn.Module):
    def __init__(self):
        super(PointwiseModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )
        # reverse_points mean lower points = better (i.e. ranking)
        self.reverse_points = True
        self.normalise_by_race = False

    def forward(self, x):
        return self.model(x)

    def optimizer(self):
        return optim.SGD(self.parameters(), lr=0.001, weight_decay=0.0005, momentum=0.9)

    @staticmethod
    def criterion():
        return nn.MSELoss()

    @staticmethod
    def format_y(y):
        return ((y[:, 0] - 1) / (y[:, 4] - 1)).double().unsqueeze(1)


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


# def main():
#     directory = "final_data"
#
#     data_x_path = f"{directory}/data_x.npz"
#     data_y_path = f"{directory}/data_y.npz"
#     horse_nums_path = f"{directory}/horse_nums.npz"
#     winner_horses_path = f"{directory}/wins.npz"
#     place_horses_path = f"{directory}/places.npz"
#
#     data_x = np.load(data_x_path)
#     data_y = np.load(data_y_path)
#     horse_nums = np.load(horse_nums_path)
#     winner_horses = np.load(winner_horses_path)
#     place_horses = np.load(place_horses_path)
#
#     data = group_data(data_x, data_y, None, horse_nums, None)
#     results = group_results(winner_horses, place_horses, None, None)
#
#     model = PointwiseModel().to("cuda").double()
#     train_keys, cv_keys = get_train_cv_split(list(data_x.keys()))
#     settings = group_settings(train_keys, cv_keys, epochs=100)
#
#     train_hist, cv_hist = train_model(model, data, results, settings)
#     plot_cost_history_graph(train_hist, cv_hist)
#     data_accuracy = test_accuracy(model, data_x, horse_nums, winner_horses, place_horses)
#     display_accuracy("Train data", data_accuracy)


if __name__ == "__main__":
    main()
