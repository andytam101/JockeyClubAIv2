import torch

from dataloader.simple_loader import SimpleLoader
from model import load_model
import model.train as train
import utils.config as config

from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("output_state")
    parser.add_argument("-e", "--epochs", type=int, default=10000)
    parser.add_argument("-m", "--model_dir")
    parser.add_argument("-t", "--train_size", type=float, default=0.8)

    return parser.parse_args()


def train_model(
        model,
        x,
        y,
        epochs,
        train_size,
        model_save_path
):
    m = x.shape[0]
    x = torch.tensor(x, dtype=torch.float32, device=config.device)
    # TODO: unsqueeze(1) on y is not always true, depends on the data
    y = torch.tensor(y, dtype=torch.float32, device=config.device).unsqueeze(1)
    train_hist, cv_hist = train.train_model(model, x, y, train_size, epochs, model_save_path)
    train_cost, train_acc = train_hist[-1]
    cv_cost, cv_acc = cv_hist[-1]

    print("=" * 100)
    print(f"Stats for {model} after epoch {epochs}")
    print(f"Final train cost: {train_cost:.4f}")
    print(f"Final train acc: {train_acc:.4f}")
    print(f"Final cv_cost: {cv_cost:.4f}")
    print(f"Final cv_acc: {cv_acc:.4f}")


def main():
    args = parse_args()
    loader = SimpleLoader(0.2, "data/simple_loader_v1/")
    x, y, train_mean, train_std = loader.load()
    model = load_model(args.model, args.state_dict_path, input_dim=19)
    train_model(model, x, y, args.epochs, args.train_size, args.output_state)


if __name__ == "__main__":
    main()
