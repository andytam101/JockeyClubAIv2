from dataloader import load_dataloader
from argparse import ArgumentParser

from database import init_engine
from database import get_session

from datetime import datetime


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("loader_name")
    parser.add_argument("-s", "--save_path", required=True)
    parser.add_argument("-db", "--db_path", default="database.db")
    parser.add_argument("-b", "--begin")
    parser.add_argument("-e", "--end")
    return parser.parse_args()


def main():
    args = parse_args()
    db_path = "sqlite:///" + args.db_path
    init_engine(db_path)

    start_date = None
    if args.begin:
        start_date = datetime.strptime(args.begin, "%Y/%m/%d").date()

    end_date = None
    if args.end:
        end_date = datetime.strptime(args.end, "%Y/%m/%d").date()

    session = get_session()
    dataloader = load_dataloader(args.loader_name)
    dataloader.load_train(session, args.save_path, start_date=start_date, end_date=end_date)
    session.close()


if __name__ == "__main__":
    main()
