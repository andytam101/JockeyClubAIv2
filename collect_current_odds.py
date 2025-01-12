from scraper import Scraper

from utils.pools import *

from argparse import ArgumentParser
import os
import csv

from datetime import datetime
import time

from tqdm import tqdm


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("race_num", type=int)
    parser.add_argument("-d", "--directory", default=f"recorded_odds/odds-{datetime.now().strftime('%Y-%m-%d')}")

    # collect data every 5 minutes by default
    parser.add_argument("-i", "--interval", default=300, type=int)
    return parser.parse_args()


def setup_target_dir(directory, races):
    for i in range(1, races+1):
        race_dir = os.path.join(directory, f"race-{i}/")
        os.makedirs(race_dir, exist_ok=True)


def initialize_csv(directory, pool, odds, timestamp):
    combination_odds = get_combination_odds(odds)
    combinations = list(map(lambda x: x[0], combination_odds))
    path = os.path.join(directory, f"{pool.lower()}-odds.csv")
    with open(path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp"] + combinations)
    add_data_to_csv(directory, pool, odds, timestamp)


def add_data_to_csv(directory, pool, odds, timestamp):
    combination_odds = get_combination_odds(odds)
    odds = list(map(lambda x: x[1], combination_odds))
    path = os.path.join(directory, f"{pool.lower()}-odds.csv")
    with open(path, "a") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp] + odds)


def collect_win_odds_with_time(scraper, url, pool):
    timestamp = datetime.now().time()
    odds = scraper.scrape_current_odds(url, pool)
    return timestamp, odds


def get_combination_odds(odds):
    c_odds = list(map(lambda x: (x["combination"], x["odds"]), odds))
    c_odds.sort(key=lambda x: x[0])
    return c_odds


def main():
    args = parse_args()
    file_dest = args.directory
    # interval = args.interval
    race_numbers = args.race_num

    scraper = Scraper()
    current_odds_base_url = "https://bet.hkjc.com/en/racing/wp/2025-01-08/HV/".lower()
    # last_time_stamp = time.time()

    # First time
    print(f"Reading races now at time: {datetime.now().time()}")
    for i in range(1, race_numbers + 1):
        race_dir = os.path.join(file_dest, f"race-{i}/")
        url = current_odds_base_url + f"{i}"
        os.makedirs(race_dir, exist_ok=True)

        wpl_time, wpl_odds = collect_win_odds_with_time(scraper, url, WIN)
        q_time, q_odds = collect_win_odds_with_time(scraper, url, QUINELLA)
        forecast_time, forecast_odds = collect_win_odds_with_time(scraper, url, FORECAST)
        win_odds, place_odds = wpl_odds
        qin_odds, qpl_odds = q_odds

        initialize_csv(race_dir, WIN, win_odds, wpl_time)
        initialize_csv(race_dir, PLACE, place_odds, wpl_time)
        initialize_csv(race_dir, QUINELLA, qin_odds, q_time)
        initialize_csv(race_dir, Q_PLACE, qpl_odds, q_time)
        initialize_csv(race_dir, FORECAST, forecast_odds, forecast_time)

    # Loop
    while True:
        # time_diff = last_time_stamp - time.time()
        # time_to_sleep = int(interval - time_diff)
        # time.sleep(time_to_sleep)
        print(f"Reading races now at time: {datetime.now().time().strftime('%H:%M:%S')}")
        # last_time_stamp = time.time()
        for i in range(1, race_numbers + 1):
            try:
                race_dir = os.path.join(file_dest, f"race-{i}/")
                url = current_odds_base_url + f"{i}"

                wpl_time, wpl_odds = collect_win_odds_with_time(scraper, url, WIN)
                q_time, q_odds = collect_win_odds_with_time(scraper, url, QUINELLA)
                forecast_time, forecast_odds = collect_win_odds_with_time(scraper, url, FORECAST)
                win_odds, place_odds = wpl_odds
                qin_odds, qpl_odds = q_odds

                add_data_to_csv(race_dir, WIN, win_odds, wpl_time)
                add_data_to_csv(race_dir, PLACE, place_odds, wpl_time)
                add_data_to_csv(race_dir, QUINELLA, qin_odds, q_time)
                add_data_to_csv(race_dir, Q_PLACE, qpl_odds, q_time)
                add_data_to_csv(race_dir, FORECAST, forecast_odds, forecast_time)
            except Exception as e:
                print(f"Failed to read current odds for race {i}: {e}")


if __name__ == "__main__":
    main()
