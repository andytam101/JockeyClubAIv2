import torch
import torch.nn as nn
import numpy as np


def convert_ranking_to_relevance_score(ranking, decay_sharpness=2):
    return 1 / torch.log(ranking + decay_sharpness)


def calculate_cost(model, target_keys, data_x, data_y, settings):
    criterion = nn.BCELoss()
    train_mean = settings["train_mean"]
    train_std = settings["train_std"]
    total_cost = 0
    for key in target_keys:
        this_x = torch.tensor(data_x[key], dtype=torch.float32, device="cuda")
        this_y = torch.tensor(data_y[key], dtype=torch.float32, device="cuda")
        this_y = convert_ranking_to_relevance_score(this_y)
        normalized_x = (this_x - train_mean) / train_std

        output = model(normalized_x)
        output = output.flatten()
        this_y = this_y.flatten()
        loss = criterion(output, this_y)
        try:
            assert not torch.isnan(loss).any()
        except AssertionError:
            print(loss)
            print(output)
            print(this_y)
            input()
        total_cost += loss.item()

    return total_cost / len(target_keys)


def perform_prediction(model, normalized_x, horse_nums):
    model.eval()
    with torch.no_grad():
        predictions = model(normalized_x)
        predictions = torch.softmax(predictions, dim=0)
    horse_nums = horse_nums.tolist()
    horse_predictions = list(zip(horse_nums, predictions.tolist()))
    horse_predictions.sort(key=lambda x: x[1], reverse=True)

    return horse_predictions[0][0], horse_predictions[1][0]


def test_accuracy(model, data_x, horse_nums, winner_horses, place_horses, settings):
    with torch.no_grad():
        train_mean = settings["train_mean"]
        train_std = settings["train_std"]
        winner_count = 0
        place_count = 0
        q_place_count = 0
        total_count = 0
        for key in data_x:
            this_x = torch.tensor(data_x[key], dtype=torch.float32, device="cuda")
            normalized_x = (this_x - train_mean) / train_std
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
    test_data = data["test_data"]
    horse_nums = data["horse_nums"]
    test_horse_nums = data["test_h_nums"]
    train_mean = settings["train_mean"]
    train_std = settings["train_std"]
    train_keys = settings["train_keys"]
    cv_keys = settings["cv_keys"]
    epochs = settings["epochs"]

    winner_horses = results["winner_horses"]
    place_horses = results["place_horses"]
    test_winner_horses = results["test_winner_horses"]
    test_place_horses = results["test_place_horses"]

    model.eval()
    initial_train_cost = calculate_cost(model, train_keys, data_x, data_y, settings)
    initial_cv_cost = calculate_cost(model, cv_keys, data_x, data_y, settings)
    print(f"Initial cost: {initial_train_cost}. CV cost: {initial_cv_cost}")

    max_test_accuracy = 0
    max_train_accuracy = 0

    train_hist = []
    cv_hist = []

    criterion = nn.BCELoss()
    optimizer = model.optimizer()
    for epoch in range(epochs):
        model.train()
        for key in train_keys:
            this_x = torch.tensor(data_x[key], dtype=torch.float32, device="cuda")
            this_y = torch.tensor(data_y[key], dtype=torch.float32, device="cuda")
            this_y = convert_ranking_to_relevance_score(this_y)
            normalized_x = (this_x - train_mean) / train_std
            output = model(normalized_x)
            output = output.flatten()
            this_y = this_y.flatten()
            loss = criterion(output, this_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        train_cost = calculate_cost(model, train_keys, data_x, data_y, settings)
        cv_cost = calculate_cost(model, cv_keys, data_x, data_y, settings)
        train_hist.append(train_cost)
        cv_hist.append(cv_cost)

        this_train_count = test_accuracy(model, data_x, horse_nums, winner_horses, place_horses, settings)
        this_train_accuracy = this_train_count[0] / this_train_count[3]
        this_test_count = test_accuracy(model, test_data, test_horse_nums, test_winner_horses, test_place_horses, settings)
        this_test_accuracy = this_test_count[0] / this_test_count[3]

        if this_test_accuracy > max_test_accuracy:
            max_test_accuracy = this_test_accuracy
            max_train_accuracy = this_train_accuracy

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs} - train cost = {train_cost:.5f}, cv cost: {cv_cost:.5f}, train accuracy: {this_train_accuracy * 100:.2f}%")

    return  max_train_accuracy, max_test_accuracy, train_hist, cv_hist