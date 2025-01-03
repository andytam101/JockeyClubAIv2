from dataloader import load_dataloader
from argparse import ArgumentParser

from database import init_engine
from database import get_session


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("loader_name")
    parser.add_argument("-s", "--save_path", required=True)
    parser.add_argument("-db", "--db_path", default="database.db")
    parser.add_argument("-cv", "--cv_size", type=float, default=0.2)
    return parser.parse_args()


def main():
    args = parse_args()
    db_path = "sqlite:///" + args.db_path
    init_engine(db_path)
    session = get_session()
    dataloader = load_dataloader(args.loader_name)
    dataloader.load_train(session, args.save_path)
    session.close()


if __name__ == "__main__":
    main()
