from model import load_model
from model.model_prediction import ModelPrediction
from database import Race, get_session, init_engine

from evaluate_strategy import simulate_upcoming_race, is_solo
from utils.pools import *

from datetime import datetime
from argparse import ArgumentParser
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("model", help="model name")
    parser.add_argument("model_dir")
    parser.add_argument("-s", "--start", default="2023/01/01")
    parser.add_argument("-e", "--end", default=datetime.today().strftime("%Y/%m/%d"))
    return parser.parse_args()


def get_all_races(session, start_date, end_date):
    races = session.query(Race).filter(Race.date >= start_date).filter(Race.date <= end_date).all()
    return races


def get_correct_combination_of_race(race, pool=WIN):
    all_winnings = race.winnings
    relevant_winnings = list(filter(lambda x: x.pool == pool, all_winnings))
    if pool in {PLACE, Q_PLACE}:
        return relevant_winnings
    else:
        return relevant_winnings[0]


def main():
    args = parse_args()

    init_engine()
    model_name = args.model
    model_dir = args.model_dir
    start_date = datetime.strptime(args.start, "%Y/%m/%d")
    end_date = datetime.strptime(args.end, "%Y/%m/%d")

    model = load_model(model_name)

    predictor = ModelPrediction(model, model_dir)
    session = get_session()
    all_races = get_all_races(session, start_date, end_date)

    correct_dict = {
        WIN: 0,
        PLACE: 0
    }

    total_dict = {
        WIN: 0,
        PLACE: 0
    }

    for race in tqdm(all_races, desc="Evaluating races..."):
        data = simulate_upcoming_race(race)
        guesses = predictor.guess_outcome_of_race(session, data)

        for pool in guesses:
            if pool == "ALL":
                continue
            winning_comb = get_correct_combination_of_race(race, pool)
            if pool == WIN:
                if is_solo(winning_comb.combination, guesses[pool][0]):
                    correct_dict[pool] += 1
                total_dict[WIN] += 1
            # elif pool == PLACE:
            #     for guess in guesses[pool]:
            #         for comb in winning_comb:
            #             if is_solo(comb.combination, guess):
            #                 correct_dict[PLACE] += 1
            #                 break
            #         total_dict[PLACE] += 1
            else:
                raise NotImplementedError

    for pool in total_dict:
        if total_dict[pool] == 0:
            continue

        correct_count = correct_dict.get(pool, 0)
        accuracy = correct_count / total_dict[pool]
        print(f"Accuracy for {pool}: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
