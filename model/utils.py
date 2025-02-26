import torch

def convert_ranking_to_score(ranking):
    t = torch.zeros_like(ranking, dtype=torch.float32)

    mask1 = ranking <= 3
    mask2 = (ranking >= 4) & (ranking <= 7)
    mask3 = ranking >= 8

    t[mask1] = 1.2 - 0.2 * ranking[mask1]
    t[mask2] = 1.05 - 0.15 * ranking[mask2]
    t[mask3] = 0

    return t
