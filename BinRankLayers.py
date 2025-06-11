import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from math import factorial
from itertools import permutations

from utils.config import device
from tqdm import tqdm

from TopKInNModel import normalise_outputs, normalise_pw_outputs, to_tensor, DataPaths, shuffle_data, split_train_cv, \
    train_model


class BinaryClassifier(nn.Module):
    def __init__(self, n, layer_2_size):
        super(BinaryClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4 * n, layer_2_size),
            nn.ReLU(),
            nn.Linear(layer_2_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class Ranking(nn.Module):
    def __init__(self, n, layer_2_size):
        super(Ranking, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4 * n, layer_2_size),
            nn.ReLU(),
            nn.Linear(layer_2_size, n),
        )

    def forward(self, x):
        return self.model(x)


class BinRankLayers:
    def __init__(self, n, k, data_paths):
        self.n = n
        self.k = k
        self.layer_1 = BinaryClassifier(n, n)
        self.layer_2 = Ranking(n, 2 * n)

        self.race_outputs = np.load(data_paths.race_outputs)
        self.pw_outputs = torch.load(data_paths.pw_outputs, map_location=device)
        self.horse_nums = np.load(data_paths.horse_nums)

    def load_pw_output_in_tensor(self, tensor, idx, pw_outputs):
        normalised_pw_outputs = normalise_pw_outputs(pw_outputs)
        sum_normalised_outputs = torch.sum(normalised_pw_outputs, dim=1)
        _, top_k_idx = torch.topk(sum_normalised_outputs, k=self.n, dim=0)
        top_k_normalised_score = normalised_pw_outputs[top_k_idx]
        tensor[idx] = top_k_normalised_score.flatten()

        return top_k_idx

    def is_valid_race(self, tensor):
        return tensor.size(0) >= self.n

    def build_input_data(self):
        n = self.n

        pw_outputs = self.pw_outputs
        result = torch.zeros((pw_outputs.size(0), 4 * n), dtype=torch.float64, device=device)

        counter = 0
        for race_id in pw_outputs.keys():
            this_race_outputs = pw_outputs[race_id]
            normalised_pw_outputs = normalise_pw_outputs(this_race_outputs)
            sum_normalised_outputs = torch.sum(normalised_pw_outputs, dim=1)
            _, top_k_idx = torch.topk(sum_normalised_outputs, k=self.n, dim=0)
            top_k_normalised_score = normalised_pw_outputs[top_k_idx]

            result[counter] = top_k_normalised_score.flatten()
            counter += 1

        result = result[:counter]

        return result, counter
