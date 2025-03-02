import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import random
import matplotlib.pyplot as plt


def get_mean_std(data_x):
    mean = torch.mean(data_x, dim=0)
    std = torch.std(data_x, dim=0)
    std[std == 0] = 1
    return mean, std


def convert_ranking_to_relevance_score(ranking, decay_sharpness=2):
    return 1 / np.log(ranking + decay_sharpness)


def transform_to_pairwise_diff(x, rankings):
    m, n = x.shape
    result_x = np.zeros((m * (m - 1), n * 2))
    if rankings is not None:
        result_y = np.zeros((m * (m - 1), 1))
    idx = 0
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            result_x[idx, :n] = x[i]
            result_x[idx, n:] = x[j]
            if rankings is not None:
                fst_score = convert_ranking_to_relevance_score(rankings[i])
                snd_score = convert_ranking_to_relevance_score(rankings[j])
                result_y[idx] = fst_score - snd_score
            idx += 1
    if rankings is not None:
        return result_x, result_y
    else:
        return result_x


def get_train_cv_split(data_keys, cv_ratio=0.1):
    m = len(data_keys)
    train_size = round(m * (1 - cv_ratio))
    train_keys = set(random.sample(data_keys, train_size))
    cv_keys = set(data_keys) - set(train_keys)
    return list(train_keys), list(cv_keys)


def calculate_accuracy(predictions, actual):
    return torch.mean((torch.sgn(predictions) == torch.sgn(actual)).float()).item()


def train_model(model, data_x, data_y, train_keys, cv_keys, epochs):
    criterion = nn.MSELoss()
    optimizer = model.optimizer()

    train_hist = []
    cv_hist = []

    for epoch in range(epochs):
        train_loss = 0
        for race_id in train_keys:
            model.train()
            this_x, this_y = transform_to_pairwise_diff(data_x[race_id], data_y[race_id])
            this_x = torch.tensor(this_x, dtype=torch.float32, device="cuda")
            this_y = torch.tensor(this_y, dtype=torch.float32, device="cuda")
            mean, std = get_mean_std(this_x)
            this_x = (this_x - mean) / std

            output = model(this_x)
            this_loss = criterion(output, this_y)
            train_loss += this_loss.item()
            optimizer.zero_grad()
            this_loss.backward()
            optimizer.step()
        train_loss /= len(train_keys)

        cv_loss = 0
        for race_id in cv_keys:
            model.eval()
            this_x, this_y = transform_to_pairwise_diff(data_x[race_id], data_y[race_id])
            this_x = torch.tensor(this_x, dtype=torch.float32, device="cuda")
            this_y = torch.tensor(this_y, dtype=torch.float32, device="cuda")

            mean, std = get_mean_std(this_x)
            this_x = (this_x - mean) / std

            output = model(this_x)
            this_cv_loss = criterion(output, this_y).item()
            cv_loss += this_cv_loss

        cv_loss /= len(cv_keys)

        train_hist.append(train_loss)
        cv_hist.append(cv_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch: {epoch + 1} / {epochs}: Train loss = {train_loss:.5f}, CV loss = {cv_loss:.5f}")

    return train_hist, cv_hist


def build_prediction_matrix(predictions, n):
    result = torch.zeros((n, n), dtype=torch.float32)
    counter = 0

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            result[i, j] = predictions[counter]
            counter += 1

    assert counter == (n * (n - 1))
    return result


def test_accuracy(model, data_x, horse_nums, top_3):
    win_count = 0
    place_count = 0
    total_count = 0
    for race_id in data_x:
        this_x = transform_to_pairwise_diff(data_x[race_id], None)
        this_x = torch.tensor(this_x, dtype=torch.float32, device="cuda")

        mean, std = get_mean_std(this_x)
        this_x = (this_x - mean) / std

        this_top_3 = top_3[race_id].tolist()
        this_horse_nums = horse_nums[race_id].tolist()

        model.eval()
        predictions = model(this_x)
        matrix = build_prediction_matrix(predictions, len(this_horse_nums))

        win_score = torch.sum(matrix, dim=1)
        lose_score = torch.sum(matrix, dim=0)

        total_score = (win_score - lose_score).tolist()
        corresponding = list(zip(this_horse_nums, total_score))
        corresponding.sort(key=lambda x: x[1], reverse=True)

        winner = corresponding[0]

        if winner[0] == this_top_3[0]:
            win_count += 1
        if winner[0] in this_top_3:
            place_count += 1
        total_count += 1

    return win_count, place_count, total_count


class PairwiseDiffBin(nn.Module):
    def __init__(self):
        super(PairwiseDiffBin, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(96, 9),
            nn.ReLU(),
            nn.Linear(9, 1),
        )

    def forward(self, x):
        return self.model(x)


    def optimizer(self):
        return optim.SGD(self.parameters(), lr=0.001, momentum=0.9)


def main():
    data_x = np.load("data2/data_x.npz")
    data_y = np.load("data2/data_y.npz")
    horse_nums = np.load("data2/horse_nums.npz")
    top_3 = np.load("data2/top_3.npz")

    train_keys, cv_keys = get_train_cv_split(sorted(data_x.keys()), cv_ratio=0.2)

    model = PairwiseDiffBin()
    model.to("cuda")

    epochs = 100
    train_hist, cv_hist = train_model(model, data_x, data_y, train_keys, cv_keys, epochs=epochs)

    x_axis = list(range(epochs))
    plt.plot(x_axis, train_hist, label="Train")
    plt.plot(x_axis, cv_hist, label="CV")
    plt.legend()
    plt.show()

    w, p, t = test_accuracy(model, data_x, horse_nums, top_3)

    print("=" * 100)
    print(f"{t} total races")
    print(f"Win: accuracy = {w/t*100:.2f}%")
    print(f"Place: accuracy = {w/p*100:.2f}%")


if __name__ == "__main__":
    main()
