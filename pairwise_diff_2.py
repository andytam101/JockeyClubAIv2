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
    # return (14 - ranking) / 14
    return 1 / np.log(ranking + decay_sharpness)


def convert_dataset(data_x, data_y, target_keys):
    total_size = 0
    for race_key in target_keys:
        this_size = data_x[race_key].shape[0]
        total_size += this_size * (this_size - 1)

    result_x = np.zeros((total_size, 96), dtype=np.float32)
    result_y = np.zeros((total_size, 1), dtype=np.float32)
    pointer = 0

    for race_key in target_keys:
        this_x, this_y = transform_to_pairwise_diff(data_x[race_key], data_y[race_key])
        this_size = this_x.shape[0]
        result_x[pointer:pointer+this_size] = this_x
        result_y[pointer:pointer+this_size] = this_y
        pointer += this_size

    return result_x, result_y


def transform_to_pairwise_diff(x, speeds):
    m, n = x.shape
    result_x = np.zeros((m * (m - 1), n * 2))
    if speeds is not None:
        result_y = np.zeros((m * (m - 1), 1))
    idx = 0
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            result_x[idx, :n] = x[i] - x[j]
            result_x[idx, n:] = x[i] + x[j]
            if speeds is not None:
                fst_score = convert_ranking_to_relevance_score(speeds[i])
                snd_score = convert_ranking_to_relevance_score(speeds[j])
                result_y[idx] = fst_score - snd_score
            idx += 1
    if speeds is not None:
        return result_x, result_y
    else:
        return result_x


def get_train_cv_split(data_keys, cv_ratio=0.1):
    m = len(data_keys)
    train_size = round(m * (1 - cv_ratio))
    train_keys = set(random.sample(data_keys, train_size))
    cv_keys = set(data_keys) - set(train_keys)
    return list(train_keys), list(cv_keys)


def train_model(model, train_x, train_y, cv_x, cv_y, epochs):
    criterion = nn.MSELoss()
    optimizer = model.optimizer()

    train_hist = []
    cv_hist = []

    model.eval()
    with torch.no_grad():
        initial_train_loss = criterion(model(train_x), train_y).item()
        initial_cv_loss = criterion(model(cv_x), cv_y).item()

    print(f"Initial: Train loss = {initial_train_loss:.5f}, CV loss = {initial_cv_loss:.5f}")
    for epoch in range(epochs):
        model.train()
        output = model(train_x)
        loss = criterion(output, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            cv_pred = model(cv_x)
            cv_loss = criterion(cv_pred, cv_y)
            cv_loss = cv_loss.item()

        train_hist.append(loss.item())
        cv_hist.append(cv_loss)

        if (epoch + 1) % 1000 == 0:
            print(f"Epoch: {epoch + 1} / {epochs}: Train loss = {loss.item():.5f}, CV loss = {cv_loss:.5f}")

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


def test_accuracy(model, data_x, horse_nums, top_3, train_mean, train_std):
    win_count = 0
    place_count = 0
    q_place_count = 0
    total_count = 0
    for race_id in data_x:
        this_x = transform_to_pairwise_diff(data_x[race_id], None)
        this_x = torch.tensor(this_x, dtype=torch.float32, device="cuda")
        this_x = (this_x - train_mean) / train_std

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
        second = corresponding[1]

        if winner[0] == this_top_3[0]:
            win_count += 1
        if winner[0] in this_top_3:
            place_count += 1
            if second[0] in this_top_3:
                q_place_count += 1
        total_count += 1

    return win_count, place_count, q_place_count, total_count


class PairwiseDiffBin(nn.Module):
    def __init__(self):
        super(PairwiseDiffBin, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(96, 9),
            nn.ReLU(),
            nn.Linear(9, 1)
        )

    def forward(self, x):
        return self.model(x)


    def optimizer(self):
        return optim.SGD(self.parameters(), lr=0.01, weight_decay=5e-4, momentum=0.9)


def main():
    directory = "data2"
    data_x = np.load(f"{directory}/data_x.npz")
    data_y = np.load(f"{directory}/data_y.npz")
    horse_nums = np.load(f"{directory}/horse_nums.npz")
    top_3 = np.load(f"{directory}/top_3.npz")
    
    test_data_x = np.load(f"{directory}/test_data_x.npz")
    test_horse_nums = np.load(f"{directory}/test_horse_nums.npz")
    test_top_3 = np.load(f"{directory}/test_top_3.npz")    

    model = PairwiseDiffBin().to("cuda")

    train_keys, cv_keys = get_train_cv_split(sorted(data_x.keys()), cv_ratio=0.2)
    print("Converting dataset...")
    train_x, train_y = convert_dataset(data_x, data_y, train_keys)
    cv_x, cv_y = convert_dataset(data_x, data_y, cv_keys)

    train_x = torch.tensor(train_x, dtype=torch.float32, device="cuda")
    train_y = torch.tensor(train_y, dtype=torch.float32, device="cuda")
    cv_x = torch.tensor(cv_x, dtype=torch.float32, device="cuda")
    cv_y = torch.tensor(cv_y, dtype=torch.float32, device="cuda")

    train_mean = torch.mean(train_x, dim=0)
    train_std = torch.std(train_x, dim=0)

    # no need to validate train_std != 0 because the feature should be removed if std == 0
    train_x = (train_x - train_mean) / train_std
    cv_x = (cv_x - train_mean) / train_std

    epochs = 10000
    train_hist, cv_hist = train_model(model, train_x, train_y, cv_x, cv_y, epochs=epochs)

    x_axis = list(range(epochs))
    plt.plot(x_axis, train_hist, label="Train")
    plt.plot(x_axis, cv_hist, label="CV")
    plt.legend()
    plt.show()

    w, p, q, t = test_accuracy(model, data_x, horse_nums, top_3, train_mean, train_std)
    print("=" * 100)
    print(f"Total count = {t}")
    print(f"Win: count = {w}, accuracy = {w / t * 100:.2f}%")
    print(f"Place: count = {p}, accuracy = {p / t * 100:.2f}%")
    print(f"Q Place: count = {q}, accuracy = {q / t * 100:.2f}%")

    w, p, q, t = test_accuracy(model, test_data_x, test_horse_nums, test_top_3, train_mean, train_std)
    print("=" * 100)
    print(f"Total count = {t}")
    print(f"Win: count = {w}, accuracy = {w / t * 100:.2f}%")
    print(f"Place: count = {p}, accuracy = {p / t * 100:.2f}%")
    print(f"Q Place: count = {q}, accuracy = {q / t * 100:.2f}%")


if __name__ == "__main__":
    main()
