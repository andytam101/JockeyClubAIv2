import torch

from final_dataloader import normalize_weights
from final_models import PWRankingScore, PWWinnerBinary, PWPlaceBinary, PWRelativeRanking
from final_model_analysis import load_model, load_data, get_sorted_prediction, get_overall_mean_std


def get_voted_winner(models, race_x, horse_nums, mean, std):
    predictions = {}
    for (idx, model) in enumerate(models):
        this_prediction = get_sorted_prediction(model, race_x, horse_nums, mean, std)[0]
        if this_prediction not in predictions:
            predictions[this_prediction] = 0
        predictions[this_prediction] += 1

    return max(predictions, key=predictions.get)


def main():
    data_x, _, _, _, _ = load_data("distance_1600/weighed/train")
    mean, std = get_overall_mean_std(data_x)
    test_x, test_y, test_h_nums, test_wins, test_places = load_data("combined/weighed/test")
    all_models = [
        load_model(PWWinnerBinary, "final_trained_models/Winner_Binary.pth"),
        load_model(PWPlaceBinary, "final_trained_models/Place_Binary.pth"),
        load_model(PWRankingScore, "final_trained_models/Ranking_Score.pth"),
        load_model(PWRelativeRanking, "final_trained_models/Relative_Ranking.pth"),
    ]

    correct = 0
    total = 0

    for key in test_x:
        voted_winner = get_voted_winner(all_models, test_x[key], test_h_nums[key], mean, std)
        if voted_winner in test_wins[key]:
            correct += 1

        total += 1

    acc = correct / total * 100
    print(f"WIN Accuracy: {acc:.2f}%")


if __name__ == "__main__":
    main()
