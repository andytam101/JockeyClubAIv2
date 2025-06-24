import argparse

import torch
import numpy as np

import os
from utils.config import device
from final_models import PWPlaceBinary, PWWinnerBinary, PWRankingScore, PWRelativeRanking

from tqdm import tqdm


def build_one_group(models, race_x, mean, std):
    this_race = torch.zeros((race_x.shape[0], len(models)), dtype=torch.float32, device=device)
    counter = 0
    for model in models:
        normalised_x = (torch.tensor(race_x, device=device, dtype=torch.float32) - mean) / std
        output = model(normalised_x).squeeze(-1)
        this_race[:, counter] = output
        counter += 1
    return this_race


def build_model_prediction(model_classes, model_paths, data_x, mean, std):
    result = {}
    race_keys = data_x.keys()

    models = []
    for model_class, model_path in zip(model_classes, model_paths):
        model = model_class().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        models.append(model)

    for race_id in tqdm(race_keys, desc="Building grouped predictions"):
        result[race_id] = build_one_group(models, data_x[race_id], mean, std)
    return result


def get_mean_std(data_x):
    total = np.zeros(64, dtype=np.float32)
    total_sq = np.zeros(64, dtype=np.float32)
    count = 0
    for race_id in data_x:
        total += np.sum(data_x[race_id], axis=0)
        total_sq += np.sum(np.square(data_x[race_id]), axis=0)
        count += data_x[race_id].shape[0]

    mean = total / count
    variance = (total_sq / count) - np.square(mean)

    return torch.tensor(mean, dtype=torch.float32, device=device), torch.sqrt(
        torch.tensor(variance, dtype=torch.float32, device=device))


def extract_group_result(data_y, k=4):
    result = {}
    for race_id in tqdm(data_y, desc="Building group "):
        this_race_y = data_y[race_id]
        size = this_race_y.shape[0]
        this_y = torch.zeros((size, 2))
        ranking = torch.tensor(this_race_y[:, 0], dtype=torch.float32, device=device)
        number_of_participants = torch.tensor(this_race_y[:, k], dtype=torch.float32, device=device)

        reciprocal_rank = number_of_participants / ranking
        reciprocal_rank = torch.softmax(reciprocal_rank, dim=0)

        win_value = torch.tensor(this_race_y[:, 0] == 1, dtype=torch.float32, device=device) * (torch.tensor(
            this_race_y[:, 3], dtype=torch.float32, device=device))

        this_y[:, 0] = reciprocal_rank
        this_y[:, 1] = win_value

        result[race_id] = this_y

    return result

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path_name")

    return parser.parse_args()


def main():
    args = parse_args()
    path_name = args.path_name

    data_path = f"final_loaded_data/{path_name}/weighed"
    model_path = f"final_trained_models/{path_name}"
    output_path = f"final_grouped_outputs/{path_name}"

    data_x = np.load(os.path.join(data_path, "train/data_x.npz"))
    test_x = np.load(os.path.join(data_path, "test/data_x.npz"))
    model_classes = [PWWinnerBinary, PWPlaceBinary, PWRankingScore, PWRelativeRanking]
    model_paths = list(map(lambda x: os.path.join(model_path, x),
                           ["Winner_Binary.pth", "Place_Binary.pth", "Ranking_Score.pth", "Relative_Ranking.pth"]))

    mean, std = get_mean_std(data_x)
    result = build_model_prediction(model_classes, model_paths, data_x, mean, std)
    test_result = build_model_prediction(model_classes, model_paths, test_x, mean, std)
    os.makedirs(output_path, exist_ok=True)
    torch.save(result, os.path.join(output_path, "grouped_outputs.pt"))
    torch.save(test_result, os.path.join(output_path, "test_grouped_outputs.pt"))


if __name__ == "__main__":
    main()
