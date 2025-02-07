import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import os
import random
import utils.config as config

from dataloader import PointwiseLoader
from load_data import INPUT_FEATURES

from database import init_engine, get_session, Winnings


class LambdaRankModel(nn.Module):
    def __init__(self):
        super(LambdaRankModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(INPUT_FEATURES, 128),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(128, 64),
            # nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.model(x)


def swap_vector_matrix(vector):
  n = vector.size(0)
  i = torch.arange(n).view(n, 1, 1)
  j = torch.arange(n).view(1, n, 1)

  M = vector.repeat(n, n, 1)
  M[i, j, i], M[i, j, j] = M[i, j, j], M[i, j, i]

  return M


def ndcg(a_score, p_score, k=None):
    if k is None:
        k = a_score.size(0)

    # Ensure p_score is a 3D tensor (batch_size, num_vectors, vector_size)
    if p_score.dim() == 1:
        p_score = p_score.unsqueeze(0).unsqueeze(0)  # Convert to 3D if it's a single matrix

    batch_size, num_vectors, vector_size = p_score.shape

    # Get the top-k indices for each vector in p_score
    _, p_rankings = torch.topk(p_score, k, dim=2)  # Shape: (batch_size, num_vectors, k)

    # Positions for DCG calculation (same for all vectors)
    positions = torch.arange(1, k + 1, dtype=torch.float32, device=a_score.device)  # Shape: (k,)

    # Gather the real scores using p_rankings
    # Expand a_score to match the shape of p_rankings for advanced indexing
    a_score_expanded = a_score.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, vector_size)
    real_scores = a_score_expanded.expand(batch_size, num_vectors, -1).gather(2, p_rankings)  # Shape: (batch_size, num_vectors, k)

    # Calculate DCG for each vector
    dcg = torch.sum((2 ** real_scores - 1) / torch.log2(positions + 1), dim=2)  # Shape: (batch_size, num_vectors)

    # Calculate IDCG (same for all vectors)
    ideal_scores, _ = torch.sort(a_score, descending=True)
    ideal_scores = ideal_scores[:k]  # Shape: (k,)
    idcg = torch.sum((2 ** ideal_scores - 1) / torch.log2(positions + 1))  # Scalar

    # Compute NDCG for all vectors
    ndcg_scores = dcg / idcg if idcg > 0 else torch.zeros_like(dcg)  # Shape: (batch_size, num_vectors)

    return ndcg_scores


def lambda_rank_loss(p_score, a_score, k=None):
    s_ij = p_score.unsqueeze(1) - p_score.unsqueeze(0)
    a_ij = a_score.unsqueeze(1) - a_score.unsqueeze(0)
    sign = (a_ij > 0).float()

    # old_ndcg = ndcg(a_score, p_score, k=k)
    # swapped_p = swap_vector_matrix(p_score)
    # new_ndcg =  ndcg(a_score, swapped_p, k=k)
    # delta_ndcg = new_ndcg - old_ndcg

    loss = sign * torch.log(torch.sigmoid(s_ij) + 1e-10) + (1 - sign) * torch.log(1 - torch.sigmoid(s_ij) + 1e-10) * torch.abs(a_ij)
    loss = torch.nan_to_num(loss, nan=0)

    equal_mask = (a_ij != 0).float()
    diagonal_mask = torch.ones_like(loss).fill_diagonal_(False)
    loss = loss * equal_mask * diagonal_mask
    return -loss.mean()



def split_data_normalise(data_x, cv_ratio):
    race_ids = list(data_x.keys())
    n = len(race_ids)
    train_size = n - int(n * cv_ratio)
    train_ids = set(random.sample(race_ids, train_size))
    cv_ids = set(race_ids) - train_ids

    size = 0
    for train_id in train_ids:
        size += data_x[train_id].shape[0]

    normalised_data_x = torch.zeros((size, INPUT_FEATURES), dtype=torch.float32, device=config.device)
    for train_id in train_ids:
        this_data_x = torch.tensor(data_x[train_id], dtype=torch.float32, device=config.device)
        normalised_data_x[:this_data_x.size(0)] = this_data_x

    train_mean = torch.mean(normalised_data_x, dim=0)
    train_std = torch.std(normalised_data_x, dim=0)
    train_std[train_std == 0] = 1

    return train_ids, cv_ids, train_mean, train_std


def convert_ranking_to_relevance_score(x):
    return 1 / x


def train_model(data_x, data_y, train_ids, cv_ids, train_mean, train_std, epochs=1000, k=None):
    train_n = len(train_ids)
    cv_n = len(cv_ids)
    model = LambdaRankModel().to(config.device)

    train_ids = list(train_ids)
    cv_ids = list(cv_ids)
    criterion = lambda x, y: lambda_rank_loss(x, y, k=k)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.eval()
    train_loss = 0
    cv_loss = 0
    for train_id in train_ids[:train_n]:
        this_x = torch.tensor(data_x[train_id], dtype=torch.float32, device=config.device)
        this_x = (this_x - train_mean) / train_std
        this_y = torch.tensor(data_y[train_id][:, 0], dtype=torch.float32, device=config.device)
        this_y = convert_ranking_to_relevance_score(this_y)

        output = model(this_x)
        output_flattened = output.view(-1)
        loss = criterion(output_flattened, this_y)
        train_loss += loss.item()

    for cv_id in cv_ids[:cv_n]:
        this_x = torch.tensor(data_x[cv_id], dtype=torch.float32, device=config.device)
        this_x = (this_x - train_mean) / train_std
        this_y = torch.tensor(data_y[cv_id][:, 0], dtype=torch.float32, device=config.device)
        this_y = convert_ranking_to_relevance_score(this_y)

        output = model(this_x)
        output_flattened = output.view(-1)
        loss = criterion(output_flattened, this_y)
        cv_loss += loss.item()

    train_loss /= train_n
    cv_loss /= cv_n

    print(f"Initial loss: train loss = {train_loss:.6f}, cv loss = {cv_loss:.6f}")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for train_id in train_ids[:train_n]:
            this_x = torch.tensor(data_x[train_id], dtype=torch.float32, device=config.device)
            this_x = (this_x - train_mean) / train_std
            this_y = torch.tensor(data_y[train_id][:, 0], dtype=torch.float32, device=config.device)
            this_y = convert_ranking_to_relevance_score(this_y)

            optimizer.zero_grad()
            output = model(this_x)
            output_flattened = output.view(-1)
            loss = criterion(output_flattened, this_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= train_n

        cv_epoch_loss = 0
        model.eval()
        for cv_id in cv_ids[:cv_n]:
            this_x = torch.tensor(data_x[cv_id], dtype=torch.float32, device=config.device)
            this_x = (this_x - train_mean) / train_std
            this_y =  torch.tensor(data_y[cv_id][:, 0], dtype=torch.float32, device=config.device)
            this_y = convert_ranking_to_relevance_score(this_y)

            output = model(this_x)
            output_flattened = output.view(-1)
            cv_loss = criterion(output_flattened, this_y)
            cv_epoch_loss += cv_loss.item()

        cv_epoch_loss /= cv_n

        print(f"Epoch {epoch + 1}, train loss: {epoch_loss:.6f}, cv loss: {cv_epoch_loss:.6f}")

    return model, optimizer


def save_model(path, model, optimizer, train_mean, train_std):
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, "model_state_dict.pth"))
    torch.save(optimizer.state_dict(), os.path.join(path, "optimizer_state_dict.pth"))
    torch.save(train_mean, os.path.join(path, "train_mean.pth"))
    torch.save(train_std, os.path.join(path, "train_std.pth"))


def main():
    data_dir = "old/loaded_data/final_data_v2_2015_2020"
    x_path = os.path.join(data_dir, "data_x.npz")
    y_path = os.path.join(data_dir, "data_y.npz")
    data_x = np.load(x_path)
    data_y = np.load(y_path)
    train_ids, cv_ids, train_mean, train_std = split_data_normalise(data_x, cv_ratio=0.2)
    model, optimizer = train_model(data_x, data_y, train_ids, cv_ids, train_mean, train_std, epochs=50, k=4)
    save_model("old/trained_models/lambda_rank_epoch_50", model, optimizer, train_mean, train_std)


def evaluate():
    data_dir = "old/loaded_data/final_data_2015_2020"
    x_path = os.path.join(data_dir, "data_x.npz")
    y_path = os.path.join(data_dir, "data_y.npz")
    data_x = np.load(x_path)

    init_engine()
    session = get_session()

    model_state_dict = torch.load("old/trained_models/lambda_rank_epoch_100/model_state_dict.pth")
    train_mean = torch.load("old/trained_models/lambda_rank_epoch_100/train_mean.pth")
    train_std = torch.load("old/trained_models/lambda_rank_epoch_100/train_std.pth")
    model = LambdaRankModel().to(config.device)
    model.load_state_dict(model_state_dict)

    total = 0
    correct = 0
    race_ids = list(data_x.keys())
    for race_id in race_ids:
        this_x = torch.tensor(data_x[race_id], dtype=torch.float32, device=config.device)
        horse_num = this_x[:, 10].tolist()
        this_x = (this_x - train_mean) / train_std
        model.eval()
        output = model(this_x)
        output_flattened = output.view(-1).tolist()
        corresponding = list(zip(horse_num, output_flattened))
        winner = max(corresponding, key=lambda x: x[1])[0]

        winner_num = int(winner)
        winnings = session.query(Winnings).filter(Winnings.race_id == race_id).filter(Winnings.pool == "WIN").all()
        actual_winners = list(map(lambda x: int(x.combination), winnings))
        if winner_num in actual_winners:
            correct += 1
        total += 1

    session.close()
    accuracy = correct / total
    print(f"Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
