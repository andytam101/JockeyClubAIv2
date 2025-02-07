import torch

import numpy as np

import argparse
import random
import os
import json
import matplotlib.pyplot as plt

import utils.config as config
from model import load_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str)
    parser.add_argument("model_dir", type=str)
    parser.add_argument("-d", "--data_path", type=str, required=True)
    parser.add_argument("-cv", "--cv_ratio", type=float, default=0.2)
    parser.add_argument("-e", "--epochs", type=int, default=100)
    return parser.parse_args()


def read_metadata(data_path):
    metadata_path = os.path.join(data_path, "metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    return metadata


def split_train_cv(x, cv_ratio=0.2):
    number_of_races = len(x)
    cv_size = int(number_of_races * cv_ratio)
    train_size = number_of_races - cv_size
    race_ids = list(x.keys())
    train_ids = set(random.sample(race_ids, train_size))
    all_ids = set(race_ids)
    cv_ids = all_ids - train_ids

    return list(train_ids), list(cv_ids)


def load_data(data_path):
    data_x_path = os.path.join(data_path, 'data_x.npz')
    data_y_path = os.path.join(data_path, 'data_y.npz')

    x = np.load(data_x_path)
    y = np.load(data_y_path)

    return x, y


def normalise_data(x, train_ids):
    total_size = 0
    feature_count = x[train_ids[0]].shape[1]
    for train_id in train_ids:
        total_size += x[train_id].shape[0]

    result = torch.zeros((total_size, feature_count), dtype=torch.float32, device=config.device)
    current_pointer = 0
    for train_id in train_ids:
        this_x = torch.tensor(x[train_id], dtype=torch.float32)
        result[current_pointer:this_x.size(0), :] = this_x

    return torch.mean(result, dim=0), torch.std(result, dim=0)


def transform_data(model, race_id, x, y, train_mean, train_std):
    this_x = torch.tensor(x[race_id], dtype=torch.float32, device=config.device)
    this_y = torch.tensor(y[race_id], dtype=torch.float32, device=config.device)
    this_x = (this_x - train_mean) / train_std
    this_y = model.format_y(this_y)
    return this_x, this_y


def calculate_cost(model, race_ids, x, y, train_mean, train_std):
    criterion = model.criterion()
    total_loss = 0
    for race_id in race_ids:
        this_x, this_y = transform_data(model, race_id, x, y, train_mean, train_std)
        prediction = model(this_x)
        loss = criterion(prediction, this_y)
        total_loss += loss.item()
    return total_loss / len(race_ids)


def train_model(model, x, y, train_mean, train_std, train_ids, cv_ids, epochs):
    model.eval()
    with torch.no_grad():
        train_cost = calculate_cost(model, train_ids, x, y, train_mean, train_std)
        cv_cost = calculate_cost(model, cv_ids, x, y, train_mean, train_std)
    print(f"Initial: train cost: {train_cost:.6f}, cv cost: {cv_cost:.6f}")

    model.train()
    optimizer = model.optimizer()
    criterion = model.criterion()
    train_history = [train_cost]
    cv_history = [cv_cost]
    for epoch in range(epochs):
        model.train()
        for race_id in train_ids:
            this_x, this_y = transform_data(model, race_id, x, y, train_mean, train_std)
            optimizer.zero_grad()
            prediction = model(this_x)
            train_loss = criterion(prediction, this_y)
            train_loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_cost = calculate_cost(model, train_ids, x, y, train_mean, train_std)
            cv_cost = calculate_cost(model, cv_ids, x, y, train_mean, train_std)
        train_history.append(train_cost)
        cv_history.append(cv_cost)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}: train cost: {train_cost:.6f}, cv cost: {cv_cost:.6f}")


    return train_history, cv_history


def save(model, model_dir, train_mean, train_std):
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "model_state_dict.pth")
    mean_path = os.path.join(model_dir, "train_mean.pth")
    std_path = os.path.join(model_dir, "train_std.pth")

    torch.save(model.state_dict(), model_path)
    torch.save(train_mean, mean_path)
    torch.save(train_std, std_path)

    print(f"Saved model to path {model_path}.")


def main():
    args = parse_args()

    data_path = args.data_path
    x, y = load_data(data_path)
    metadata = read_metadata(data_path)

    train_ids, cv_ids = split_train_cv(x, cv_ratio=args.cv_ratio)
    train_mean, train_std = normalise_data(x, train_ids)
    model = load_model(args.model_name, metadata).to(config.device)
    train_history, cv_history = train_model(model, x, y, train_mean, train_std, train_ids, cv_ids, args.epochs)

    save(model, args.model_dir, train_mean, train_std)

    # plotting graphs
    x_axis = list(range(len(train_history)))

    plt.plot(x_axis, train_history, label="train")
    plt.plot(x_axis, cv_history, label="cv")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
