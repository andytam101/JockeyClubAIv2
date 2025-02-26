import torch

import numpy as np

from argparse import ArgumentParser
import os
import json

from train_model import read_metadata, transform_data
from model import load_model
from utils.pools import *
import utils.config as config
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("model_name", type=str)
    parser.add_argument('model_path', type=str)
    parser.add_argument('-k', "--k", type=float, default=0)
    parser.add_argument("-d", "--data_path", type=str, required=True)
    return parser.parse_args()


def read_data_dir(data_path):
    data_x_path = os.path.join(data_path, "data_x.npz")
    result_path = os.path.join(data_path, "result.json")
    data_x = np.load(data_x_path)

    with open(result_path, "r") as f:
        result = json.load(f)

    return data_x, result


def read_model_dir(model_path):
    model_state_dict_path = os.path.join(model_path, "model_state_dict.pth")
    train_mean_path = os.path.join(model_path, "train_mean.pth")
    train_std_path = os.path.join(model_path, "train_std.pth")

    model_state_dict = torch.load(model_state_dict_path)
    train_mean = torch.load(train_mean_path)
    train_std = torch.load(train_std_path)

    return model_state_dict, train_mean, train_std


def evaluate_model(model, x, train_mean, train_std, results, k=0):
    model.eval()
    race_ids = list(x.keys())

    number_of_races_bet = {}
    total_bet_count = {}
    correct_bet_count = {}
    profits = {}

    for pool in ALL_POOLS:
        number_of_races_bet[pool] = 0
        total_bet_count[pool] = 0
        correct_bet_count[pool] = 0
        profits[pool] = 0

    for race_id in tqdm(race_ids, desc="Evaluating races"):
        has_bet = set()
        this_x = torch.tensor(x[race_id], dtype=torch.float32, device=config.device)
        horse_nums = this_x[:, 10].tolist().copy()
        this_x = (this_x - train_mean) / train_std
        bet = model.perform_bet(horse_nums, this_x, k=k)
        actual_result = results[race_id]

        for pool, comb in bet:
            has_bet.add(pool)
            if pool == WIN or pool == PLACE:
                for result in actual_result[pool]:
                    if int(comb) == int(result["combination"]):
                        correct_bet_count[pool] += 1
                        profits[pool] += result["amount"]
                        break
                total_bet_count[pool] += 1
                profits[pool] -= 10
            else:
                raise NotImplementedError

        for p in has_bet:
            number_of_races_bet[p] += 1

    return profits, correct_bet_count, total_bet_count, number_of_races_bet, len(x)


def display_results(profits, correct_bet_count, total_bet_count, number_of_races_bet, total_races):
    for pool in ALL_POOLS:
        if total_bet_count[pool] == 0:
            continue
        print(f"====== Statistics for pool {pool} ======")
        print(f"Overall profit: {profits[pool]:.2f}")
        print(f"Correct bet count: {correct_bet_count[pool]}")
        print(f"Total bet count: {total_bet_count[pool]}")
        print(f"Accuracy: {correct_bet_count[pool] / total_bet_count[pool] * 100:.2f}%")
        print(f"Total number of races bet: {number_of_races_bet[pool]}")
        print(f"Proportion of races bet: {number_of_races_bet[pool] / total_races * 100:.2f}%")

def main():
    args = parse_args()
    model_name = args.model_name
    model_path = args.model_path
    data_path = args.data_path

    metadata = read_metadata(data_path)
    data_x, result = read_data_dir(data_path)
    model_state_dict, train_mean, train_std = read_model_dir(model_path)

    model = load_model(model_name, metadata)
    model.load_state_dict(model_state_dict)

    profits, correct_bet_count, total_bet_count, number_of_races_bet, total_races = (
        evaluate_model(model, data_x, train_mean, train_std, result, k=args.k))
    display_results(profits, correct_bet_count, total_bet_count, number_of_races_bet, total_races)


if __name__ == '__main__':
    main()
