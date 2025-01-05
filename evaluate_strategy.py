from database import Participation, get_session, init_engine, Race
from database.fetch import Fetch
import dataloader.utils as dataloader

from strategy import *
from strategy.HighEVWinnerStrategy import HighEVWinnerStrategy
from strategy.HighPWinnerStrategy import HighPWinnerStrategy
from strategy.RandomWinnerStrategy import RandomWinnerStrategy
from strategy.PevWinnerStrategy import PevWinnerStrategy

from datetime import datetime
import warnings
from tqdm import tqdm

from utils import utils


def is_solo(target, bet):
    # for win and place
    return int(target) == int(bet) or target == "-"


def is_unordered(target, bet):
    # for quinella, quinella place, trio, first 4
    target_set = set(map(int, target.split(",")))
    return set(bet) == target_set or target == "-"


def is_ordered(target, bet):
    # for forecast, tierce and quartet
    bet_str = ",".join(bet)
    return target == bet_str or target == "-"


def calculate_one_pool_payout(validate_func, winnings, bets):
    # bets is a dictionary where key = horse number, value = amount
    target = winnings.combination
    res = 0
    number_won = 0
    for bet in bets:
        amount = bets[bet]
        if amount > 0 and validate_func(target, bet):
            res += winnings.amount * amount / 10
            number_won += 1

        res -= amount

    return res, number_won


def calculate_payout(
        fetch,
        race_id,
        win=None,
        place=None,
        quinella=None,
        quinella_place=None,
        forecast=None,
        tierce=None,
        trio=None,
        first_4=None,
        quartet=None
):
    payouts = fetch.fetch_winnings(race_id=race_id)
    get_one = lambda x: list(filter(lambda y: y.pool == x, payouts))[0]
    get_all = lambda x: list(filter(lambda y: y.pool == x, payouts))

    total = 0
    total_won_count = 0

    if win is not None:
        winning = get_one("WIN")
        p, won_count = calculate_one_pool_payout(is_solo, winning, win)
        total += p
        total_won_count += won_count

    if place is not None:
        winnings = get_all("PLACE")
        for winning in winnings:
            p, won_count = calculate_one_pool_payout(is_solo, winning, place)
            total += p
            total_won_count += won_count

    if quinella is not None:
        winning = get_one("QUINELLA")
        p, won_count = calculate_one_pool_payout(is_unordered, winning, quinella)
        total += p
        total_won_count += won_count

    if quinella_place is not None:
        winnings = get_all("QUINELLA PLACE")
        for winning in winnings:
            p, won_count = calculate_one_pool_payout(is_unordered, winning, quinella_place)
            total += p
            total_won_count += won_count

    if forecast is not None:
        winning = get_one("FORECAST")
        p, won_count = calculate_one_pool_payout(is_ordered, winning, forecast)
        total += p
        total_won_count += won_count

    if tierce is not None:
        winning = get_one("TIERCE")
        p, won_count = calculate_one_pool_payout(is_ordered, winning, tierce)
        total += p
        total_won_count += won_count

    if trio is not None:
        winning = get_one("TRIO")
        p, won_count = calculate_one_pool_payout(is_unordered, winning, trio)
        total += p
        total_won_count += won_count

    if first_4 is not None:
        winning = get_one("FIRST 4")
        p, won_count = calculate_one_pool_payout(is_unordered, winning, first_4)
        total += p
        total_won_count += won_count

    if quartet is not None:
        winning = get_one("QUARTET")
        p, won_count = calculate_one_pool_payout(is_ordered, winning, quartet)
        total += p
        total_won_count += won_count

    return {
        "profit": total,
        "number_won": total_won_count
    }


def simulate_upcoming_race(race):
    ps: [Participation] = race.participations
    ps = utils.remove_unranked_participants(ps)

    data = []
    for p in ps:
        this_p = {
            "horse_id": p.horse_id,
            "number": p.number,
            "jockey_id": p.jockey_id,
            "rating": p.rating,
            "lane": p.lane,
            "gear_weight": p.gear_weight,
            "horse_weight": p.horse_weight,

            # TODO: add variance to win odds to simulate uncertainty in betting
            "win_odds": p.win_odds,

            "date": race.date,
            "race_class": dataloader.convert_race_class(race.race_class),
            "distance": race.distance,
            "total_bet": race.total_bet,
            "trainer_id": p.horse.trainer_id,
        }
        data.append(this_p)

    return data


def count_bets(bets):
    result = 0
    for b in bets:
        for h in bets[b]:
            if bets[b][h] > 0:
                result += 1

    return result


def evaluate_strategy(strategy, start_date, end_date=datetime.today().date()):
    session = get_session()
    all_races = session.query(Race).filter(Race.date >= start_date).filter(Race.date <= end_date).all()
    all_bets = []
    for race in tqdm(all_races, desc=f"{strategy}"):
        data = simulate_upcoming_race(race)
        data.sort(key=lambda x: x["number"])
        all_bets.append({
            "race_id": race.id,
            "bets": strategy.bet(session, data)
        })
    session.close()

    profit = 0
    number_won = 0
    number_bets = 0

    fetch = Fetch()
    for bet in all_bets:
        this_result = calculate_payout(fetch, bet["race_id"], **bet["bets"])
        profit += this_result["profit"]
        number_won += this_result["number_won"]
        number_bets += count_bets(bet["bets"])


    return {
        "profit": profit,
        "number_of_races": len(all_races),
        "number_of_bets": number_bets,
        "number_won": number_won,
    }


def display_results(strategy, result):
    strategy_name = f" {strategy} "
    header = f"{strategy_name.center(75, "=")}"
    if result['number_of_bets'] > 0:
        accuracy = result["number_won"] / result['number_of_bets']
    else:
        accuracy = 0

    print(header)
    print(f"Profit: {result["profit"]}")
    print(f"Total number of races: {result['number_of_races']}")
    print(f"Total number of bets: {result['number_of_bets']}")
    print(f"Total number won: {result['number_won']}")
    print(f"Accuracy: {accuracy * 100:.3f}%")
    print()


def main():
    init_engine()

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    model_dir = "trained_models/winner_nn_timed"
    strategies = [
        RandomWinnerStrategy(),
        HighPWinnerStrategy(model_dir=model_dir, threshold=1.3, count=1),
        HighPWinnerStrategy(model_dir=model_dir, threshold=1.35, count=1),
        HighPWinnerStrategy(model_dir=model_dir, threshold=1.4, count=1),
        HighPWinnerStrategy(model_dir=model_dir, threshold=1.45, count=1),
        HighPWinnerStrategy(model_dir=model_dir, threshold=1.5, count=1),

        # HighEVWinnerStrategy(model_dir=model_dir),
        # PevWinnerStrategy(model_dir=model_dir, w=0.25),
        # PevWinnerStrategy(model_dir=model_dir, w=0.50),
        # PevWinnerStrategy(model_dir=model_dir, w=0.75),
    ]

    results = []

    for strategy in strategies:
        this_r = evaluate_strategy(strategy, start_date=datetime(2024, 9, 1))
        print()
        results.append(this_r)

    for i in range(len(results)):
        display_results(strategies[i], results[i])


if __name__ == '__main__':
    main()
