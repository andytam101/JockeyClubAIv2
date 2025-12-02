import argparse

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import random
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import copy
import os

from load_data import is_place
from utils.config import device


ODDS_ANOMALY_THRESHOLD = np.inf


class KFoldTrainer:
    def __init__(self, model_factory, data, results, k_folds):
        self.model_factory = model_factory
        self.model = model_factory()
        self.data = data
        self.results = results

        self.k_folds = k_folds
        self.folds = []

        self.trained_model_dicts = []
        self.mean_stds = []
        self.cv_costs = []

    def fold_data(self, data_y):
        anomaly_count = 0
        data_keys_list = []
        for key in data_y:
            race_y = data_y[key]
            winner_idx = torch.argmin(race_y[:, 0])
            if race_y[winner_idx, 3] < ODDS_ANOMALY_THRESHOLD:
                data_keys_list.append(key)
            else:
                anomaly_count += 1

        random.shuffle(data_keys_list)

        n = len(data_keys_list)
        step_count = n // self.k_folds
        for i in range(0, n, step_count):
            self.folds.append(data_keys_list[i:i + step_count])

        print(f"Removed {anomaly_count}/{n + anomaly_count} anomalies")

    def train_model(self, batch_size=1024, max_epoch=200, overfitting_threshold=3, display=True):
        data_x = self.data["data_x"]
        self.fold_data(self.data["data_y"])

        for number in range(self.k_folds):
            self.model = self.model_factory()
            if display:
                print(f"Training fold {number + 1}")
            _, this_cv_loss, _, _ = self.train_one_fold(number, batch_size, max_epoch, overfitting_threshold, display)
            self.trained_model_dicts.append(copy.deepcopy(self.model.state_dict()))
            self.cv_costs.append(this_cv_loss)

        idx = np.argmin(self.cv_costs).item()
        best_model = self.trained_model_dicts[idx]

        self.model = self.model_factory()
        self.model.load_state_dict(best_model)

        train_accuracy = self.test_accuracy(idx, is_test=False)
        test_accuracy = self.test_accuracy(idx, is_test=True)
        return best_model, train_accuracy, test_accuracy

    def calculate_loss(self, data_x, data_y):
        model = self.model
        criterion = model.criterion()

        with torch.no_grad():
            output = model(data_x)
            output = output.flatten()
            data_y = data_y.flatten()
            loss = criterion(output, data_y)

        return loss.item()


    def train_one_fold(self, fold_number, batch_size=1024, max_epoch=200, overfitting_threshold=3, display=True):
        train_keys, cv_keys = self.get_train_cv_keys(fold_number)
        train_x, train_y = self.concatenate_data(train_keys, pairwise=self.model.pairwise, display=display)
        cv_x, cv_y = self.concatenate_data(cv_keys, pairwise=self.model.pairwise, display=display)

        model = self.model
        criterion = self.model.criterion()
        optimizer = self.model.optimizer()

        if not model.normalise_by_race:
            train_mean = torch.mean(train_x, dim=0)
            train_std = torch.std(train_x, dim=0)
            train_x = (train_x - train_mean) / train_std
            cv_x = (cv_x - train_mean) / train_std

            self.mean_stds.append((train_mean, train_std))

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

            if display and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1} train loss = {loss:.6f}, cv loss = {cv_loss:.6f}")

            if display and overfitting_counter >= overfitting_threshold:
                print(f"Overfitting detected. Stopped training at epoch {epoch + 1}.")
                break
        else:
            if display:
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

    def format_pairwise_data(self, race_x, race_y):
        format_y = self.model.format_y
        this_size = race_x.shape[0] * (race_x.shape[0] - 1)
        n = race_x.shape[0]
        this_x = torch.zeros((this_size, 128), dtype=torch.float64, device="cuda")
        this_y = torch.zeros((this_size, 1), dtype=torch.float64, device="cuda")
        inner_counter = 0
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                this_x[inner_counter, :64] = race_x[i]
                this_x[inner_counter, 64:] = race_x[j]

                this_y[inner_counter, 0] = format_y(race_y[i], race_y[j])
                inner_counter += 1
        assert not torch.isnan(this_y).any()

        return this_x, this_y

    def concatenate_data(self, data_keys, pairwise, display=True):
        format_y = self.model.format_y
        data_x = self.data["data_x"]
        data_y = self.data["data_y"]

        total_size = 0
        for race_id in data_keys:
            if pairwise:
                total_size += data_x[race_id].shape[0] * (data_x[race_id].shape[0] - 1)
            else:
                total_size += data_x[race_id].shape[0]

        result_x = torch.zeros((total_size, 128 if pairwise else 64), dtype=torch.float64, device="cuda")
        result_y = torch.zeros((total_size, 1), dtype=torch.float64, device="cuda")

        counter = 0

        iterator = tqdm(data_keys, desc="Concatenating data") if display else data_keys

        for race_id in iterator:
            if pairwise:
                this_size = data_x[race_id].shape[0] * (data_x[race_id].shape[0] - 1)
                race_x = data_x[race_id]
                if self.model.normalise_by_race:
                    race_x = normalise_by_race(race_x)

                this_x, this_y = self.format_pairwise_data(race_x, data_y[race_id])

                result_x[counter: counter + this_size] = this_x
                result_y[counter: counter + this_size] = this_y
                counter += this_size
            else:
                this_size = data_x[race_id].shape[0]
                race_x = data_x[race_id]
                if self.model.normalise_by_race:
                    result_x[counter: counter + this_size] = normalise_by_race(race_x)
                else:
                    result_x[counter: counter + this_size] = race_x
                result_y[counter: counter + this_size] = format_y(data_y[race_id])
                counter += this_size

        return result_x, result_y

    def test_accuracy(self, fold_num, is_test):
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
                this_x = data_x[key]
                if self.model.normalise_by_race:
                    this_mean = torch.mean(this_x, dim=0)
                    this_std = torch.std(this_x, dim=0)
                    this_std[this_std == 0] = 1
                    this_x = (this_x - this_mean) / this_std

                if self.model.pairwise:
                    dummy_y = torch.zeros((this_x.size(0), 5), dtype=torch.float64, device="cuda")
                    this_x, _ = self.format_pairwise_data(this_x, dummy_y)
                    if self.model.normalise_by_race:
                        normalized_x = this_x
                    else:
                        mean, std = self.mean_stds[fold_num]
                        normalized_x = (this_x - mean) / std
                else:
                    if self.model.normalise_by_race:
                        normalized_x = this_x
                    else:
                        mean, std = self.mean_stds[fold_num]
                        normalized_x = (this_x - mean) / std

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

        horse_nums = horse_nums.tolist()
        if model.pairwise:
            n = len(horse_nums)
            prediction_matrix = torch.eye(n, dtype=torch.float64, device="cuda")
            counter = 0
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    prediction_matrix[i, j] = predictions[counter]
                    counter += 1

            # probabilities of it defeating other horses
            row_score = torch.sum(prediction_matrix, dim=1)

            # probabilities of other horses defeating it
            column_score = torch.sum(prediction_matrix, dim=0)

            result_vector = row_score - column_score

            # normalisation
            result_vector = torch.softmax(result_vector, dim=0)
            horse_predictions = list(zip(horse_nums, result_vector.tolist()))
        else:
            horse_predictions = list(zip(horse_nums, predictions.tolist()))
        horse_predictions.sort(key=lambda x: x[1], reverse=(not model.reverse_points))

        return horse_predictions[0][0], horse_predictions[1][0]


def normalise_by_race(race_data):
    mean = torch.mean(race_data, dim=0)
    std = torch.std(race_data, dim=0)
    std[std == 0] = 1
    return (race_data - mean) / std


def convert_np_dict_to_torch_dict(npz):
    result = {}
    for key in npz:
        result[key] = torch.tensor(npz[key], device="cuda", dtype=torch.float64)
    return result


def load_data(train_directory, *test_directories):
    train_dir = os.path.join(train_directory, "train")
    train_x = np.load(os.path.join(train_dir, "data_x.npz"))
    train_y = np.load(os.path.join(train_dir, "data_y.npz"))
    train_horse_nums = np.load(os.path.join(train_dir, "horse_nums.npz"))
    train_wins = np.load(os.path.join(train_dir, "wins.npz"))
    train_places = np.load(os.path.join(train_dir, "places.npz"))

    test_x = {}
    test_horse_nums = {}
    test_wins = {}
    test_places = {}
    for test_directory in test_directories:
        test_dir = os.path.join(test_directory, "test")
        test_x.update(np.load(os.path.join(test_dir, "data_x.npz")))
        test_horse_nums.update(np.load(os.path.join(test_dir, "horse_nums.npz")))
        test_wins.update(np.load(os.path.join(test_dir, "wins.npz")))
        test_places.update(np.load(os.path.join(test_dir, "places.npz")))

    data = group_data(train_x, train_y, test_x, train_horse_nums, test_horse_nums)
    results = group_results(train_wins, train_places, test_wins, test_places)
    return data, results


def default_factory(f):
    return lambda: f().to(device).double()


def model_ability(win_count, total_count, test_win_count, test_total_count):
    win_acc = (win_count / total_count) * 100
    test_acc = (test_win_count / test_total_count) * 100

    # subtract difference square to reduce penalise overfitting
    diff_sq = np.square(win_acc - test_acc)
    return win_acc + test_acc - diff_sq


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_dir", type=str)
    parser.add_argument("test_dir", type=str)
    parser.add_argument("save_dir", type=str)

    return parser.parse_args()


def main():
    args = parse_args()
    train_dir = args.train_dir
    test_dir = args.test_dir
    directory = args.save_dir

    data, results = load_data(
        train_dir, test_dir,
    )
    all_model_factories = map(default_factory, [PWWinnerBinary, PWPlaceBinary, PWRankingScore, PWRelativeRanking])
    all_model_factories = list(all_model_factories)

    iterations = 1
    os.makedirs(directory, exist_ok=True)

    for model_factory in all_model_factories:
        best_acc = -np.inf
        best_acc_info = None
        best_acc_model = None
        trainer = KFoldTrainer(model_factory, data, results, k_folds=5)
        print(f"Training {trainer.model.name} model")

        iterator = tqdm(range(iterations)) if iterations > 1 else range(iterations)

        for _ in iterator:
            best_model, train_acc, test_acc = trainer.train_model(batch_size=1024, max_epoch=400, overfitting_threshold=3, display=iterations == 1)

            acc_score = model_ability(train_acc[0], train_acc[3], test_acc[0], test_acc[3])

            if acc_score > best_acc:
                best_acc = acc_score
                best_acc_model = best_model
                best_acc_info = (train_acc, test_acc)

        train_acc, test_acc = best_acc_info
        display_accuracy("Train data", train_acc)
        display_accuracy("Test data", test_acc)

        torch.save(best_acc_model, f"{directory}/{"_".join(trainer.model.name.split(" "))}.pth")

        if train_acc[3] > 0:
            train_accuracy = {
                "win": train_acc[0] / train_acc[3],
                "place": train_acc[1] / train_acc[3],
                "q_place": train_acc[2] / train_acc[3]
            }
        else:
            train_accuracy = {}

        if test_acc[3] > 0:
            test_accuracy = {
                "win": test_acc[0] / test_acc[3],
                "place": test_acc[1] / test_acc[3],
                "q_place": test_acc[2] / test_acc[3]
            }
        else:
            test_accuracy = {}

        accuracy_data = {
            "train_acc": train_accuracy,
            "test_acc": test_accuracy,
        }

        with open(f"{directory}/{"_".join(trainer.model.name.split(" "))}.json", "w") as f:
            json.dump(accuracy_data, f, indent=4)


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
        "data_x": convert_np_dict_to_torch_dict(data_x),
        "data_y": convert_np_dict_to_torch_dict(data_y),
        "test_data": convert_np_dict_to_torch_dict(test_data),
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


def get_score_from_ranking(ranking, decay_sharpness=1.5):
    base_log = np.log(1 + decay_sharpness)
    a = np.log(decay_sharpness) * base_log / (base_log - np.log(decay_sharpness))
    return a / torch.log(ranking + decay_sharpness) - a / base_log


class PWRelativeRanking(nn.Module):
    def __init__(self):
        super(PWRelativeRanking, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

        self.pairwise = False
        self.reverse_points = False
        self.normalise_by_race = False

        self.name = "Relative Ranking"

    def forward(self, x):
        return self.model(x)

    def optimizer(self):
        return optim.SGD(self.parameters(), lr=0.001, weight_decay=0.0005, momentum=0.9)

    @staticmethod
    def criterion():
        return nn.MSELoss()

    @staticmethod
    def format_y(y):
        return (1 - (y[:, 0] - 1) / (y[:, 4] - 1)).double().unsqueeze(1)


class PWRankingScore(nn.Module):
    def __init__(self):
        super(PWRankingScore, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )
        self.pairwise = False
        self.reverse_points = False
        self.normalise_by_race = False

        self.name = "Ranking Score"

    def forward(self, x):
        return self.model(x)

    def optimizer(self):
        return optim.SGD(self.parameters(), lr=0.001, weight_decay=0.0005, momentum=0.9)

    @staticmethod
    def criterion():
        return nn.MSELoss()

    @staticmethod
    def format_y(y):
        return (get_score_from_ranking((y[:, 0] - 1) / (y[:, 4] - 1))).double().unsqueeze(1)

class PWWinnerBinary(nn.Module):
    def __init__(self):
        super(PWWinnerBinary, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(64, 6),
            nn.ReLU(),
            nn.Linear(6, 1),
            nn.Sigmoid(),
        )

        self.pairwise = False
        self.reverse_points = False
        self.normalise_by_race = False

        self.name = "Winner Binary"

    def forward(self, x):
        return self.model(x)

    def optimizer(self):
        return optim.SGD(self.parameters(), lr=0.001, weight_decay=0.0005, momentum=0.9)

    @staticmethod
    def criterion():
        return nn.BCELoss()

    @staticmethod
    def format_y(y):
        return (y[:, 0] == 1).double().unsqueeze(1)


class PWPlaceBinary(nn.Module):
    def __init__(self):
        super(PWPlaceBinary, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(64, 5),
            nn.ReLU(),
            nn.Linear(5, 1),
            nn.Sigmoid(),
        )

        self.name = "Place Binary"

        self.pairwise = False
        self.reverse_points = False
        self.normalise_by_race = False

    def forward(self, x):
        return self.model(x)

    def optimizer(self):
        return optim.SGD(self.parameters(), lr=0.001, weight_decay=0.0005, momentum=0.9)

    @staticmethod
    def criterion():
        return nn.BCELoss()

    @staticmethod
    def format_y(y):
        return is_place(y[:, 0], y[:, 4]).double().unsqueeze(1)


class PWOddsAdjustedScore(nn.Module):
    def __init__(self):
        super(PWOddsAdjustedScore, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

        self.name = "Odds Adjusted Score"

        self.pairwise = False
        self.reverse_points = False
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
        win_result = (y[:,0] == 1) * torch.log(y[:, 3])
        place_result = (is_place(y[:, 0], y[:, 4]).double()) * torch.log(1 + ((y[:, 3] - 1) / 3))   # estimate place odds by dividing by 3
        return place_result.double().unsqueeze(1)


class PairBinary(nn.Module):
    def __init__(self):
        super(PairBinary, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid(),
        )

        self.name = "Pairwise Binary"
        self.pairwise = True
        self.reverse_points = False
        self.normalise_by_race = False

    def forward(self, x):
        return self.model(x)

    def optimizer(self):
        return optim.SGD(self.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

    @staticmethod
    def criterion():
        return nn.BCELoss()

    @staticmethod
    def format_y(y1, y2):
        return float(y1[0] < y2[0])


class PairRScoreDiff(nn.Module):
    def __init__(self):
        super(PairRScoreDiff, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(128, 12),
            nn.ReLU(),
            nn.Linear(12, 1),
        )

        self.name = "Pair Ranking Score Difference"
        self.pairwise = True
        self.reverse_points = False
        self.normalise_by_race = False

    def forward(self, x):
        return self.model(x)

    def optimizer(self):
        return optim.SGD(self.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

    @staticmethod
    def criterion():
        return nn.MSELoss()

    @staticmethod
    def format_y(y1, y2):
        s1 = get_score_from_ranking((y1[0] - 1) / (y1[4] - 1))
        s2 = get_score_from_ranking((y2[0] - 1) / (y2[4] - 1))

        return float(s1 - s2)


class PairSScoreDiff(nn.Module):
    def __init__(self):
        super(PairSScoreDiff, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(128, 12),
            nn.ReLU(),
            nn.Linear(12, 1),
        )

        self.name = "Pair Speed Score Difference"
        self.pairwise = True
        self.reverse_points = False
        self.normalise_by_race = False

    def forward(self, x):
        return self.model(x)

    def optimizer(self):
        return optim.SGD(self.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

    @staticmethod
    def criterion():
        return nn.MSELoss()

    @staticmethod
    def format_y(y1, y2):
        return float(y1[2] - y2[2])


class PairRankingDiff(nn.Module):
    def __init__(self):
        super(PairRankingDiff, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(128, 6),
            nn.ReLU(),
            nn.Linear(6, 1),
        )

        self.name = "Pair Ranking Difference"
        self.pairwise = True
        self.reverse_points = False
        self.normalise_by_race = False

    def forward(self, x):
        return self.model(x)

    def optimizer(self):
        return optim.SGD(self.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

    @staticmethod
    def criterion():
        return nn.MSELoss()

    @staticmethod
    def format_y(y1, y2):
        r1 = y1[0] - y1[4]
        r2 = y2[0] - y2[4]
        return r2 - r1


if __name__ == "__main__":
    main()
