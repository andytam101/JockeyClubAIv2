import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

import numpy as np

from utils.config import device


def get_races(path):
    races = np.load(path)
    result = []
    for race_id in races:
        result.append(torch.from_numpy(races[race_id]).to(device).double())

    return result


def get_reciprocal_ranks(path):
    race_ys = np.load(path)
    result = []
    for race_id in race_ys:
        race_y = torch.from_numpy(race_ys[race_id]).to(device).double()
        ranking = race_y[:, 0]
        number_of_horses = race_y[:, 4]

        result.append((number_of_horses - 1) / (ranking - 1))

    return result


def pad_races(races, targets):
    padded_inputs = pad_sequence(races, batch_first=True)  # [B, H, 64]
    padded_targets = pad_sequence(targets, batch_first=True)  # [B, H]
    mask = torch.zeros(padded_targets.shape, dtype=torch.bool)
    for i, r in enumerate(targets):
        mask[i, :r.shape[0]] = 1
    return padded_inputs, padded_targets, mask

def main():
    reciprocal_ranks = get_reciprocal_ranks("../final_loaded_data/location_ST_1600/weighed/train/data_y.npz")
    races = get_races("../final_loaded_data/location_ST_1600/weighed/train/data_x.npz")

    assert len(reciprocal_ranks) == len(races)

    inputs, targets, mask = pad_races(races, reciprocal_ranks)


if __name__ == '__main__':
    main()
