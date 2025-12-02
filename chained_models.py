import torch
import numpy as np

from pw_models import PWPlaceBinary, PWWinnerBinary, PWRankingScore, PWRelativeRanking, PairBinary
from model_analysis import load_model, get_overall_mean_std, load_data

from tqdm import tqdm


def get_group_pw_decision(pw_models, race_x, horse_nums):
    result = torch.zeros((race_x.size(0), 1), dtype=torch.float64, device=race_x.device)
    horse_nums = horse_nums.tolist()
    for model in pw_models:
        model.eval()
        this_prediction = model(race_x)
        result += this_prediction

    overall_pred = result.flatten().tolist()
    corresponding = list(zip(horse_nums, overall_pred))
    corresponding.sort(key=lambda x: x[1], reverse=True)
    top_4_horse_nums = list(map(lambda x: x[0], corresponding[:4]))
    return top_4_horse_nums


def format_pairwise(data):
    m = data.size(0)
    n = data.size(1)
    result = torch.zeros((m * (m - 1), n * 2), dtype=torch.float64, device=data.device)
    counter = 0
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            result[counter, :n] = data[i]
            result[counter, n:] = data[j]
            counter += 1
    return result


def get_pairwise_decision(model, top_4, horse_nums):
    assert top_4.size(0) == 4
    data_x = format_pairwise(top_4)

    model.eval()
    pairwise_prediction = model(data_x)

    result_matrix = torch.eye(4, dtype=torch.float64, device="cuda")

    counter = 0
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            result_matrix[i, j] = pairwise_prediction[counter]
            counter += 1

    print(result_matrix)
    for i in range(3):
        for j in range(i + 1, 4):
            total = result_matrix[i, j] + result_matrix[j, i]
            result_matrix[i, j] = result_matrix[i, j] / total
            result_matrix[j, i] = result_matrix[j, i] / total

    row_sum = torch.prod(result_matrix, dim=1)
    col_sum = torch.sum(result_matrix, dim=0)
    print(result_matrix)
    scores = row_sum
    corresponding = list(zip(horse_nums, scores))
    corresponding.sort(key=lambda x: x[1], reverse=True)

    print(corresponding[0][0])
    return corresponding[0][0]


def main():
    pw_models = [
        load_model(PWPlaceBinary, "final_trained_models/Place_Binary.pth"),
        load_model(PWWinnerBinary, "final_trained_models/Winner_Binary.pth"),
        load_model(PWRankingScore, "final_trained_models/Ranking_Score.pth"),
    ]

    pairwise_model = load_model(PairBinary, "final_trained_models/Pairwise_Binary.pth")

    data_x, _, _, _, _ = load_data("distance_1600/weighed/train")
    mean, std = get_overall_mean_std(data_x)
    test_x, test_y, test_h_nums, test_wins, test_places = load_data("combined/weighed/test")

    before_correct = 0
    before_total = 0

    correct = 0
    total = 0

    different = 0
    for key in tqdm(test_x):
        this_x = torch.tensor((test_x[key] - mean) / std, dtype=torch.float64, device="cuda")
        top_4_horses = get_group_pw_decision(pw_models, this_x, test_h_nums[key])

        before_winner = top_4_horses[0]
        if before_winner in test_wins[key]:
            before_correct += 1
        before_total += 1

        indices = np.where(np.isin(test_h_nums[key], top_4_horses))[0]
        pairwise_h_nums = test_h_nums[key][indices]
        indices = torch.from_numpy(indices)

        top_4_x = this_x[indices]
        print("=" * 100)
        print(top_4_horses)
        final_winner = get_pairwise_decision(pairwise_model, top_4_x, pairwise_h_nums)

        if final_winner in test_wins[key]:
            correct += 1
        total += 1

        if before_winner != final_winner:
            different += 1


    before_accuracy = before_correct / before_total
    accuracy = correct / total
    print(f"Before accuracy: {before_accuracy * 100:.2f}%")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Different proportion: {different / total * 100:.2f}%")


if __name__ == "__main__":
    main()
