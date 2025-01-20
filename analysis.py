# from evaluate_strategy import simulate_upcoming_race, is_solo
# from model import load_model
# from model.model_prediction import ModelPrediction
# from database import init_engine, get_session, Race
# from datetime import datetime
#
# from utils.pools import *
# from tqdm import tqdm
#
# ranking_model = load_model("RankingNN")
# pairwise_model = load_model("PairBinary")
# ranking_predictor = ModelPrediction(ranking_model, "trained_models/ranking_nn_ultimate_epoch_10000")
# pairwise_predictor = ModelPrediction(pairwise_model, "trained_models/pairwise_binary_ultimate_epoch_25000")
#
# init_engine()
# session = get_session()
# races = session.query(Race).filter(datetime(2023, 9, 1).date() < Race.date).filter(Race.date < datetime(2024, 9, 1).date()).all()
#
# guess = {}
#
# for race in tqdm(races):
#     data = simulate_upcoming_race(race)
#     ranking_output = ranking_predictor.guess_outcome_of_race(session, data)[WIN]
#     pairwise_output = pairwise_predictor.guess_outcome_of_race(session, data)[WIN]
#
#     win_winning = list(filter(lambda x: x.pool == WIN, race.winnings))[0]
#
#     guess[race.id] = (ranking_output[0], pairwise_output[0], int(win_winning.combination))
#
# session.close()
# print(guess)

from database import init_engine, get_session, Horse, Participation
import matplotlib.pyplot as plt
import statistics
import utils.utils as utils

def get_seconds(time_obj):
    return time_obj.minute * 60 + time_obj.second + time_obj.microsecond * 1e-6

init_engine()
session = get_session()

horses = session.query(Horse).all()
speeds = {}
for horse in horses:
    if horse.origin not in speeds:
        speeds[horse.origin] = []
    ps = utils.remove_unranked_participants(horse.participations)
    for p in ps:
        time = get_seconds(p.finish_time)
        distance = p.race.distance
        speed = distance / time
        speeds[horse.origin].append(speed)

colours = []
average_speeds = []
for colour in speeds:
    colours.append(colour)
    average_speeds.append(statistics.mean(speeds[colour]))

total_speed = sum(average_speeds)
normalized_speeds = list(map(lambda x: x / total_speed, average_speeds))

plt.bar(colours, normalized_speeds)
plt.show()

session.close()
