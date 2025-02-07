import numpy as np

from database import init_engine, get_session, Horse, Race, Participation, Winnings

from dataloader.independent_loader import load_one_independent_participation, INDEPENDENT_FEATURES
from dataloader.utils import *

from utils.pools import ALL_POOLS

from datetime import datetime
from argparse import ArgumentParser
import os
from tqdm import tqdm
import json

# constants
RACE_FEATURES = 9
INPUT_FEATURES = RACE_FEATURES + INDEPENDENT_FEATURES
OUTPUT_FEATURES = 4


def convert_race_to_dict(race):
    return {
        "id": race.id,
        "date": race.date,
        "race_class": convert_race_class(race.race_class),
        "distance": race.distance,
        "location": race.location,
        "course": race.course,
        "condition": race.condition,
        "total_bet": race.total_bet,
        "number_of_participants": get_number_of_participants(race),
    }


def load_race_features(race):
    # race as a dictionary
    race_class = race["race_class"]
    distance = race["distance"]
    location = race["location"] == "Sha Tin"  # (i.e. 0 for Happy Valley, 1 for Sha Tin)
    width = utils.get_track_width(race["location"], race["course"])
    condition = race["condition"]
    total_bet = race["total_bet"]
    number_of_participants = race["number_of_participants"]
    try:
        race_upper_limit = utils.RACE_UPPER_LIMIT[int(race_class)]
        race_lower_limit = utils.RACE_LOWER_LIMIT[int(race_class)]
    except IndexError:
        race_upper_limit = 0
        race_lower_limit = 0
    return np.array([
        race_class,
        distance,
        location,
        encode_condition(condition),
        width,
        total_bet,
        number_of_participants,
        race_upper_limit,
        race_lower_limit,
    ], dtype=np.float32)


def load_from_db(session, start_date=None, end_date=None):
    races = get_races_between_dates(session, start_date, end_date)
    n = len(races)

    all_x = {}
    all_y = {}
    result = {}
    metadata = {}
    data_size = 0
    metadata["start_date"] = start_date.strftime("%Y/%m/%d")
    metadata["end_date"] = end_date.strftime("%Y/%m/%d")

    for idx in tqdm(range(n), desc="Loading data"):
        race = races[idx]
        race_dict = convert_race_to_dict(race)
        ps = utils.remove_unranked_participants(race.participations)
        p_count = len(ps)
        if p_count <= 4:
            continue

        ps.sort(key=lambda x: x.number)
        this_x = np.zeros((p_count, INPUT_FEATURES), dtype=np.float32)
        this_y = np.zeros((p_count, OUTPUT_FEATURES), dtype=np.float32)

        data_size += p_count
        for i, p in enumerate(ps):
            this_x[i, :RACE_FEATURES] = load_race_features(race_dict)
            this_x[i, RACE_FEATURES:RACE_FEATURES + INDEPENDENT_FEATURES] = (
                load_one_independent_participation(p, session, False))

            this_y[i, 0] = get_ranking_from_participation(p)
            this_y[i, 1] = get_ranking_from_participation(p) / get_number_of_participants(race)
            this_y[i, 2] = time_to_number_of_seconds(p.finish_time)
            this_y[i, 3] = p.win_odds

        all_x[race.id] = this_x
        all_y[race.id] = this_y

        this_result = {}
        winnings = race.winnings

        for pool in ALL_POOLS:
            this_result[pool] = [x.combination for x in winnings if x.pool == pool]
        result[race.id] = this_result

    metadata["input_features"]  = INPUT_FEATURES
    metadata["output_features"] = OUTPUT_FEATURES
    metadata["size"] = data_size
    return all_x, all_y, result, metadata


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('save_path', type=str)
    parser.add_argument("-s", "--start", type=str, default=None)
    parser.add_argument("-e", "--end", type=str, default=None)
    parser.add_argument("-db", "--database", type=str, default="database.db")
    return parser.parse_args()


def save(directory, x, y, result, metadata):
    os.makedirs(directory, exist_ok=True)
    x_path = os.path.join(directory, "data_x.npz")
    y_path = os.path.join(directory, "data_y.npz")
    result_path = os.path.join(directory, "result.json")
    metadata_path = os.path.join(directory, "metadata.json")
    np.savez(x_path, **x)
    np.savez(y_path, **y)

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)


def main():
    args = parse_args()
    db_path = "sqlite:///" + args.database
    save_path = args.save_path
    init_engine(db_path)

    start_date = args.start
    if start_date is not None:
        start_date = datetime.strptime(start_date, "%Y/%m/%d").date()
    end_date = args.end
    if end_date is not None:
        end_date = datetime.strptime(end_date, "%Y/%m/%d").date()

    session = get_session()
    x, y, result, metadata = load_from_db(session, start_date, end_date)
    session.close()

    # x and y should be dictionary
    save(save_path, x, y, result, metadata)


if __name__ == '__main__':
    main()
