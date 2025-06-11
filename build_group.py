import argparse

import torch
import numpy as np

import os
from utils.config import device
from final_models import PWPlaceBinary, PWWinnerBinary, PWRankingScore, PWRelativeRanking

from tqdm import tqdm


def build_model_prediction(model_classes, model_paths, data_x, mean, std):
    result = {}
    race_keys = data_x.keys()
    for race_id in tqdm(race_keys, desc="Building grouped predictions"):
        this_race = torch.zeros((data_x[race_id].shape[0], len(model_classes)), dtype=torch.float64, device=device)
        counter = 0
        for model_class, model_path in zip(model_classes, model_paths):
            model = model_class().to(device).double()
            model.load_state_dict(torch.load(model_path, map_location=device))
            this_x = data_x[race_id]
            normalised_x = (torch.tensor(this_x, device=device, dtype=torch.float64) - mean) / std
            output = model(normalised_x).squeeze(-1)
            this_race[:, counter] = output
            counter += 1

        result[race_id] = this_race
    return result


def get_mean_std(data_x):
    total = np.zeros(64, dtype=np.float64)
    total_sq = np.zeros(64, dtype=np.float64)
    count = 0
    for race_id in data_x:
        total += np.sum(data_x[race_id], axis=0)
        total_sq += np.sum(np.square(data_x[race_id]), axis=0)
        count += data_x[race_id].shape[0]

    mean = total / count
    variance = (total_sq / count) - np.square(mean)

    return torch.tensor(mean, dtype=torch.float64, device=device), torch.sqrt(
        torch.tensor(variance, dtype=torch.float64, device=device))


def extract_group_result(data_y, k=4):
    result = {}
    for race_id in tqdm(data_y, desc="Building group "):
        this_race_y = data_y[race_id]
        size = this_race_y.shape[0]
        this_y = torch.zeros((size, 2))
        ranking = torch.tensor(this_race_y[:, 0], dtype=torch.float64, device=device)
        number_of_participants = torch.tensor(this_race_y[:, k], dtype=torch.float64, device=device)

        reciprocal_rank = number_of_participants / ranking
        reciprocal_rank = torch.softmax(reciprocal_rank, dim=0)

        win_value = torch.tensor(this_race_y[:, 0] == 1, dtype=torch.float64, device=device) * (torch.tensor(
            this_race_y[:, 3], dtype=torch.float64, device=device))

        this_y[:, 0] = reciprocal_rank
        this_y[:, 1] = win_value

        result[race_id] = this_y

    return result

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("data_path")
    parser.add_argument("model_path")
    parser.add_argument("output_path")

    return parser.parse_args()


def main():
    args = parse_args()
    data_path = args.data_path
    model_path = args.model_path
    output_path = args.output_path

    data_x = np.load(os.path.join(data_path, "train/data_x.npz"))
    # data_y = np.load(os.path.join(data_path, "train/data_y.npz"))
    # horse_nums = np.load("final_loaded_data/location_ST/weighed/train/horse_nums.npz")
    model_classes = [PWWinnerBinary, PWPlaceBinary, PWRankingScore, PWRelativeRanking]
    model_paths = list(map(lambda x: os.path.join(model_path, x),
                           ["Winner_Binary.pth", "Place_Binary.pth", "Ranking_Score.pth", "Relative_Ranking.pth"]))

    mean, std = get_mean_std(data_x)
    result = build_model_prediction(model_classes, model_paths, data_x, mean, std)
    # result_y = extract_group_result(data_y)
    os.makedirs(output_path, exist_ok=True)
    torch.save(result, os.path.join(output_path, "grouped_outputs.pt"))
    # torch.save(result_y, "final_grouped_outputs/grouped_results.pt")


if __name__ == "__main__":
    main()
