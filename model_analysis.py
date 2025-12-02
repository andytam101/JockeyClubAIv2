import torch
import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm

from pw_models import PWRankingScore, PWRankingScore, PWPlaceBinary, PWRelativeRanking, PWWinnerBinary


from database import init_engine, get_session, Race


from utils.config import device


def load_model(f, path):
    model = f()
    model.to(device).double()
    model.load_state_dict(torch.load(path, weights_only=True, map_location=device))
    return model


def load_data(path):
    data_x = np.load(f"{path}/data_x.npz")
    data_y = np.load(f"{path}/data_y.npz")
    horse_nums = np.load(f"{path}/horse_nums.npz")
    wins = np.load(f"{path}/wins.npz")
    places = np.load(f"{path}/places.npz")
    return data_x, data_y, horse_nums, wins, places


def get_overall_mean_std(data_x):
    total_size = 0
    for key in data_x:
        total_size += data_x[key].shape[0]

    concatenated = np.zeros((total_size, 64), dtype=np.float64)

    counter = 0
    for key in tqdm(data_x):
        this_size = data_x[key].shape[0]
        concatenated[counter: counter + this_size] = data_x[key]
        counter += this_size

    mean = np.mean(concatenated, axis=0)
    std = np.std(concatenated, axis=0)
    std[std == 0] = 1

    return mean, std


def get_sorted_prediction(model, race_x, horse_nums, mean, std):
    model.eval()
    with torch.no_grad():
        normalized_x = torch.tensor((race_x - mean) / std, dtype=torch.float64, device=device)
        predictions = model(normalized_x)
        predictions = predictions.tolist()

    horse_nums = horse_nums.tolist()
    corresponding = list(zip(horse_nums, predictions))
    corresponding.sort(key=lambda x: x[1], reverse=(not model.reverse_points))

    return list(map(lambda x: x[0], corresponding))


def get_actual_result(race_y, horse_nums):
    horse_nums = horse_nums.tolist()
    rankings = race_y[:, 0].tolist()
    corresponding = list(zip(horse_nums, rankings))
    corresponding.sort(key=lambda x: x[1], reverse=False)
    return list(map(lambda x: x[0], corresponding))


def plot_results(counter_map, x_label):
    values = counter_map.keys()
    frequencies = counter_map.values()

    total_count = sum(frequencies)
    frequencies = [f / total_count for f in frequencies]

    plt.bar(values, frequencies)
    plt.xlabel(x_label)
    plt.show()


def display_cumulative_table(counter_map):
    values = list(counter_map.keys())
    values.sort()

    total = sum(counter_map.values())

    acc = 0
    for v in values:
        acc += counter_map[v]
        density = acc / total
        print(f"{v}: {density}")



def main():
    model = load_model(PWRankingScore, "final_trained_models/Ranking_Score.pth")
    data_x, _, _, _, _ = load_data("final_loaded_data/location_ST_1600/weighed/train")
    mean, std = get_overall_mean_std(data_x)
    test_x, test_y, test_h_nums, _, _ = load_data("final_loaded_data/combined/weighed/test")

    # what ranking actual winner was predicted as
    # key: predicted rank, value: count
    counter_predicted_ranking = {}
    counter_actual_ranking = {}

    init_engine()

    session = get_session()

    for race_id in tqdm(test_x, desc="Analysing"):

        if session.query(Race).filter(Race.id == race_id).one().location != "Sha Tin":
            continue

        sorted_prediction = get_sorted_prediction(model, test_x[race_id], test_h_nums[race_id], mean, std)
        actual_result = get_actual_result(test_y[race_id], test_h_nums[race_id])

        actual_winner = actual_result[0]
        predicted_ranking = sorted_prediction.index(actual_winner) + 1

        predicted_winner = sorted_prediction[0]
        actual_ranking = actual_result.index(predicted_winner) + 1

        if predicted_ranking not in counter_predicted_ranking:
            counter_predicted_ranking[predicted_ranking] = 0
        counter_predicted_ranking[predicted_ranking] += 1

        if actual_ranking not in counter_actual_ranking:
            counter_actual_ranking[actual_ranking] = 0
        counter_actual_ranking[actual_ranking] += 1

    print("Predicted ranking for actual winner")
    display_cumulative_table(counter_predicted_ranking)

    print("\nActual ranking for predicted winner")
    display_cumulative_table(counter_actual_ranking)


    plot_results(counter_predicted_ranking, "Predicted ranking for actual winner")
    plot_results(counter_actual_ranking, "Actual ranking for predicted winner")

if __name__ == "__main__":
    main()
