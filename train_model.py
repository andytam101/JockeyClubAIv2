import torch

from dataloader.simple_loader import SimpleLoader
from model import train
from model.top_3_LR import Top3LR
from model.top_3_NN import Top3NN
import utils.config as config

from argparse import ArgumentParser


model_dict = {
    "Top3LR": Top3LR,
    "Top3NN": Top3NN,
}


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("output_state")
    parser.add_argument("-e", "--epochs", type=int, default=10000)
    parser.add_argument("-s", "--state_dict_path")
    parser.add_argument("-t", "--train_size", type=float, default=0.8)

    return parser.parse_args()


def load_model(model_name, state_dict_path, **kwargs):
    try:
        model = model_dict[model_name](**kwargs)
    except KeyError:
        print("Invalid model name.")
        return None

    if state_dict_path:
        model.load_state_dict(torch.load(state_dict_path))
    return model


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
    x, y = SimpleLoader(0.2, "data/simple_loader_v1/").load()
    model = load_model(args.model, args.state_dict_path, input_dim=19)
    train_model(model, x, y, args.epochs, args.train_size, args.output_state)

if __name__ == "__main__":
    main()
