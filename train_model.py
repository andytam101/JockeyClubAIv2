from dataloader.simple_loader import SimpleLoader
import model.train as train
from model.top_3_LR import Top3LR
from model.top_3_NN import Top3NN
import torch
import utils.config as config


def train_model(model, x, y, epochs, model_save_path):
    # TODO: refactor this function to take in train_size or cv_size
    m = x.shape[0]
    x = torch.tensor(x, dtype=torch.float32, device=config.device)
    y = torch.tensor(y, dtype=torch.float32, device=config.device).unsqueeze(1)
    train_hist, cv_hist = train.train_model(model, x, y, 0.8, epochs, model_save_path)
    # print(train_hist)
    # print(cv_hist)
    train_cost, train_acc = train_hist[-1]
    cv_cost, cv_acc = cv_hist[-1]

    print("=" * 100)
    print(f"Stats for {model} after epoch {epochs}")
    print(f"Final train cost: {train_cost:.4f}")
    print(f"Final train acc: {train_acc:.4f}")
    print(f"Final cv_cost: {cv_cost:.4f}")
    print(f"Final cv_acc: {cv_acc:.4f}")


def main():
    x, y = SimpleLoader(0.2).load()
    model_1 = Top3LR(input_dim=19)
    model_2 = Top3NN(input_dim=19)
    train_model(model_1, x, y, 10000)
    train_model(model_2, x, y, 10000)

    print(model_1.predict(x[-1]))
    print(model_2.predict(x[-1]))

if __name__ == "__main__":
    main()
