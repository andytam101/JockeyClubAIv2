from database import Participation, get_session, init_engine, Race, Winnings
from database.fetch import Fetch
import dataloader.utils as dataloader

from strategy import *
from strategy.HighEVWinnerStrategy import HighEVWinnerStrategy
from strategy.HighPWinnerAbsoluteStrategy import HighPWinnerAbsoluteStrategy
from strategy.HighPWinnerProportionStrategy import HighPWinnerProportionStrategy
from strategy.HighPWinnerProgressiveStrategy import HighPWinnerProgressiveStrategy
from strategy.RandomWinnerStrategy import RandomWinnerStrategy
from strategy.PevWinnerStrategy import PevWinnerStrategy

import matplotlib.pyplot as plt

from datetime import datetime
import warnings
from tqdm import tqdm

from utils import utils
from utils.pools import *

from tabulate import tabulate


def is_solo(target, bet):
    # for win and place
    return int(target) == int(bet) or target == "-"


def is_unordered(target, bet):
    # for quinella, quinella place, trio, first 4
    target_set = set(map(int, target.split(",")))
    return set(bet) == target_set or target == "-"


def is_ordered(target, bet):
    # for forecast, tierce and quartet
    bet_str = ",".join(list(map(str, bet)))
    return target == bet_str or target == "-"


def calculate_payout(validate_func, winning, bet_combination, amount):
    target_combination = winning.combination
    win_odds = winning.amount / 10
    if validate_func(target_combination, bet_combination):
        return float(win_odds) * amount - amount
    else:
        return -amount


def calculate_one_bet_payout(
        session,
        race_id,
        bet_combination,
        amount,
        pool: str,
):
    payouts = session.query(Winnings).filter(Winnings.race_id == race_id).all()
    get_one = lambda x: list(filter(lambda y: y.pool == x, payouts))[0]
    get_all = lambda x: list(filter(lambda y: y.pool == x, payouts))
    pool = pool.upper()

    if pool == WIN:
        winning = get_one(WIN)
        return calculate_payout(is_solo, winning, bet_combination, amount)
    elif pool == PLACE:
        raise NotImplementedError()
    elif pool == QUINELLA:
        winning = get_one(QUINELLA)
        return calculate_payout(is_unordered, winning, bet_combination, amount)
    elif pool == Q_PLACE:
        raise NotImplementedError()
    elif pool == FORECAST:
        winning = get_one(FORECAST)
        return calculate_payout(is_ordered, winning, bet_combination, amount)
    elif pool == TRIO:
        winning = get_one(TRIO)
        return calculate_payout(is_unordered, winning, bet_combination, amount)
    elif pool == TIERCE:
        winning = get_one(TIERCE)
        return calculate_payout(is_ordered, winning, bet_combination, amount)
    elif pool == FIRST_4:
        winning = get_one(FIRST_4)
        return calculate_payout(is_unordered, winning, bet_combination, amount)
    elif pool == QUARTET:
        winning = get_one(QUARTET)
        return calculate_payout(is_ordered, winning, bet_combination, amount)
    else:
        raise Exception("Invalid pool")


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
            "win_odds": utils.randomize_odds(float(p.win_odds)),

            "date": race.date,
            "race_class": dataloader.convert_race_class(race.race_class),
            "distance": race.distance,
            "total_bet": race.total_bet,
            "trainer_id": p.horse.trainer_id,
            "number_of_horses": len(ps),
            "opponents": []
        }

        opponents = utils.remove_unranked_participants(p.race.participations)
        opponents = list(filter(lambda x: x.horse_id != p.horse_id, opponents))

        for opponent in opponents:
            this_p["opponents"].append({
                "horse_id": opponent.horse_id,
                "jockey_id": opponent.jockey_id,
                "rating": opponent.rating,
                "gear_weight": opponent.gear_weight,
                "horse_weight": opponent.horse_weight,
                "win_odds": opponent.win_odds,
            })

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

    number_won = 0
    number_bets = 0
    number_of_races = 0

    balance_history = [strategy.init_balance]

    done = False
    for race in tqdm(all_races, desc=f"{strategy}"):
        data = simulate_upcoming_race(race)
        data.sort(key=lambda x: x["number"])
        bet = strategy.bet(session, data)

        race_amount_bet = 0
        for pool in bet:
            for combination in bet[pool]:
                amount = bet[pool][combination]
                if amount <= 0:
                    continue
                payout = calculate_one_bet_payout(session, race.id, combination, amount, pool)
                if payout > 0:
                   number_won += 1
                race_amount_bet += amount
                number_bets += 1
                strategy.update_balance(payout)
                if strategy.balance <= 0:
                    done = True
                    break
            if done:
                break
        if race_amount_bet > 0:
            number_of_races += 1
        balance_history.append(strategy.balance)
        if done:
            break

    session.close()

    return {
        "total_number_of_races": len(all_races),
        "number_of_races": number_of_races,
        "number_of_bets": number_bets,
        "number_won": number_won,
        "history": balance_history,
    }

def display_results(strategy, result):
    strategy_name = f" {strategy} "
    final_balance = strategy.balance
    initial_balance = strategy.init_balance
    profit = final_balance - initial_balance
    percentage = profit / initial_balance * 100

    header = f"{strategy_name.center(100, "=")}"
    if result['number_of_bets'] > 0:
        accuracy = result["number_won"] / result['number_of_bets']
    else:
        accuracy = 0

    print(header)
    print(f"Initial balance: {initial_balance:.2f}")
    print(f"Final Balance: {final_balance:.2f}")
    print(f"Profit: {profit:.2f}")
    print(f"Percentage change: {percentage:.2f}%")
    print(f"Total number of races: {result['total_number_of_races']}")
    print(f"Number of races bet: {result['number_of_races']}")
    print(f"Total number of bets: {result['number_of_bets']}")
    print(f"Total number of bets won: {result['number_won']}")
    print(f"Max balance reached: {max(result["history"])}")
    print(f"Accuracy: {accuracy * 100:.3f}%")
    print()

    balance_history = result["history"]
    plt.plot(range(len(balance_history)), balance_history)
    plt.show()


def main():
    init_engine()

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    model_dir = "trained_models/ranking_nn_long_epoch_1000"
    strategies = [
        # RandomWinnerStrategy(),
        # HighPWinnerAbsoluteStrategy(model_dir=model_dir, threshold=0, count=1),

        # HighPWinnerAbsoluteStrategy(model_dir=model_dir, threshold=0.9, count=1),
        # HighPWinnerAbsoluteStrategy(model_dir=model_dir, threshold=1.0, count=1),
        # HighPWinnerAbsoluteStrategy(model_dir=model_dir, threshold=1.1, count=1),
        # HighPWinnerAbsoluteStrategy(model_dir=model_dir, threshold=1.3, count=1),

        # HighPWinnerProgressiveStrategy(model_dir=model_dir, threshold=0.9, count=1),
        # HighPWinnerProgressiveStrategy(model_dir=model_dir, threshold=1.0, count=1),
        # HighPWinnerProgressiveStrategy(model_dir=model_dir, threshold=1.1, count=1),
        # # HighPWinnerProgressiveStrategy(model_dir=model_dir, threshold=1.3, count=1),
        #
        # HighPWinnerProportionStrategy(model_dir=model_dir, threshold=0.9, count=1),
        # HighPWinnerProportionStrategy(model_dir=model_dir, threshold=1.0, count=1),
        # HighPWinnerProportionStrategy(model_dir=model_dir, threshold=1.1, count=1),
        # HighPWinnerProportionStrategy(model_dir=model_dir, threshold=1.3, count=1),

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
