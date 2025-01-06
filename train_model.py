from model import load_model
from model.model_trainer import ModelTrainer

from argparse import ArgumentParser

import matplotlib.pyplot as plt


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("data_dir")

    # model_dir will be where model is saved
    parser.add_argument("-m", "--model_dir", required=True)
    parser.add_argument("-e", "--epochs", type=int, default=10000)
    parser.add_argument("-cv", "--cv_size", type=float, default=0.2)

    return parser.parse_args()


def main():
    args = parse_args()
    model = load_model(args.model)
    model_trainer = ModelTrainer(model)
    train_hist, cv_hist = model_trainer.train_model(args.data_dir, epochs=args.epochs, cv_size=args.cv_size)
    model_trainer.save(args.model_dir)

    train_cost, train_acc = train_hist[-1]
    cv_cost, cv_acc = cv_hist[-1]

    print("=" * 100)
    print(f"Stats for {model} after epoch {args.epochs}")
    print(f"Final train cost: {train_cost:.4f}")
    print(f"Final train acc: {train_acc:.4f}")
    print(f"Final cv cost: {cv_cost:.4f}")
    print(f"Final cv acc: {cv_acc:.4f}")
    train_cost = list(map(lambda x: x[0], train_hist))
    cv_cost = list(map(lambda x: x[0], cv_hist))
    print(f"Minimum cv cost: {min(cv_cost):.4f}")

    plt.plot(range(len(train_cost)), train_cost, label="train")
    plt.plot(range(len(cv_cost)), cv_cost, label="cv")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
