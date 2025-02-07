from model import load_model
from model.model_prediction import ModelPrediction
from database import Race, get_session, init_engine

from evaluate_strategy import simulate_upcoming_race, is_solo, is_unordered, is_ordered
from utils.pools import *

from datetime import datetime
from argparse import ArgumentParser
from tqdm import tqdm

import numpy as np

from dataloader.utils import get_number_of_participants

import os
import logging


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("model", help="model name")
    parser.add_argument("model_dir")
    parser.add_argument("-r", "--races_dir", required=True)
    # parser.add_argument("-s", "--start", default="2023/01/01")
    # parser.add_argument("-e", "--end", default=datetime.today().strftime("%Y/%m/%d"))
    return parser.parse_args()


def get_all_races(session, start_date, end_date):
    races = session.query(Race).filter(Race.date >= start_date).filter(Race.date <= end_date).all()
    return races


def get_correct_combination_of_race(race, pool=WIN):
    all_winnings = race.winnings
    relevant_winnings = list(filter(lambda x: x.pool == pool.replace("_", " "), all_winnings))
    if pool in {PLACE, Q_PLACE}:
        return relevant_winnings
    else:
        if len(relevant_winnings) > 0:
            return relevant_winnings[0]
        else:
            return None


def read_races(races_dir):
    data_x_path = os.path.join(races_dir, "data_x.npz")
    data = np.load(data_x_path)

    return data


def main():
    args = parse_args()

    # logging.basicConfig(level=logging.INFO, filename="evaluate_model.log")
    # logger = logging.getLogger("Evaluator")

    init_engine()
    model_name = args.model
    model_dir = args.model_dir
    races_dir = args.races_dir

    model = load_model(model_name)

    predictor = ModelPrediction(model, model_dir)
    session = get_session()
    all_races = read_races(races_dir)

    correct_dict = {
        WIN: 0,
        PLACE: 0,
        QUINELLA: 0,
        Q_PLACE: 0,
        FORECAST: 0,
        TIERCE: 0,
        TRIO: 0,
        FIRST_4: 0,
        QUARTET: 0
    }

    total_dict = {
        WIN: 0,
        PLACE: 0,
        QUINELLA: 0,
        Q_PLACE: 0,
        FORECAST: 0,
        TIERCE: 0,
        TRIO: 0,
        FIRST_4: 0,
        QUARTET: 0
    }

    for race_id in tqdm(all_races, desc="Evaluating races..."):
        data = all_races[race_id]
        race = session.query(Race).filter(Race.id == race_id).one()

        if get_number_of_participants(race) < 4:
            continue

        guesses = predictor.guess_outcome_of_race(data)

        for pool in guesses:
            if pool == "ALL":
                continue
            winning_comb = get_correct_combination_of_race(race, pool)
            if winning_comb is None:
                continue

            if pool not in {PLACE, Q_PLACE} and winning_comb.combination == "-":
                continue

            if pool == WIN:
                if is_solo(winning_comb.combination, guesses[pool][0]):
                    correct_dict[WIN] += 1
                total_dict[WIN] += 1
            elif pool == PLACE:
                for guess in guesses[pool]:
                    for comb in winning_comb:
                        if comb.combination == "-":
                            continue
                        if is_solo(comb.combination, guess):
                            correct_dict[PLACE] += 1
                            break
                    total_dict[PLACE] += 1
            elif pool == QUINELLA:
                if is_unordered(winning_comb.combination, guesses[pool]):
                    correct_dict[QUINELLA] += 1
                total_dict[QUINELLA] += 1
            elif pool == FORECAST:
                if is_ordered(winning_comb.combination, guesses[pool]):
                    correct_dict[FORECAST] += 1
                total_dict[FORECAST] += 1
            elif pool == Q_PLACE:
                for guess in guesses[pool]:
                    for comb in winning_comb:
                        if comb.combination == "-":
                            continue
                        if is_unordered(comb.combination, guess):
                            correct_dict[Q_PLACE] += 1
                            break
                    total_dict[Q_PLACE] += 1
            elif pool == TRIO:
                if is_unordered(winning_comb.combination, guesses[pool]):
                    correct_dict[TRIO] += 1
                total_dict[TRIO] += 1
            elif pool == TIERCE:
                if is_ordered(winning_comb.combination, guesses[pool]):
                    correct_dict[TIERCE] += 1
                total_dict[TIERCE] += 1
            elif pool == FIRST_4:
                if is_unordered(winning_comb.combination, guesses[pool]):
                    correct_dict[FIRST_4] += 1
                total_dict[FIRST_4] += 1
            elif pool == QUARTET:
                if is_ordered(winning_comb.combination, guesses[pool]):
                    correct_dict[QUARTET] += 1
                total_dict[QUARTET] += 1
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
