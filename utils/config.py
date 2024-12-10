import torch


# GLOBAL DATABASE
DATABASE_PATH = "sqlite:///database.db"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
