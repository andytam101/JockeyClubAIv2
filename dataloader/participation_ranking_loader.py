from .loader import Loader
from .independent_loader import *

from tqdm import tqdm


import utils.utils as utils
import statistics

from utils.utils import remove_unranked_participants


# Stats for horses participation:
# - speed (mean, std)
# - ranking (median, interquartile range)
# - weights (mean, std)
# - win / place ratio
# - win odds (mean, std)
#
# Stats for jockeys participation:
# - speed (mean, std)
# - ranking (median, interquartile range)
# - win / place ratio
# - win odds (mean, std)
# - gear weight (mean)
#
# Stats for trainers participation:
# - speed (mean)
# - ranking (median)
# - win / place ratio
# - win odds (mean)
#
# Opponent horses:
# - win odds (harmonic mean)
# - weight (mean)
# - gear weight (mean)
#
# Opponent horses' participation:
# - speed (mean)
# - ranking (median)
# - win odds (mean)
# - win / place ratio
#
# Opponent jockey's participation:
# - speed (mean)
# - ranking (median)
# - win odds (mean)
# - win / place ratio


class ParticipationRankingLoader(Loader):
    def __init__(self):
        super().__init__()
        self.opponent_data_features = 4
        self.opponent_ps_summary_features = 16

    @property
    def input_features(self):
        return INDIVIDUAL_FEATURES

    def __repr__(self):
        return "Participation Ranking Loader"

    # @staticmethod
    # def get_opponent_participation_data(ps: [Participation]):
    #     win_odds = list(map(lambda p: p.win_odds, ps))
    #     harmonic_mean_odds = statistics.harmonic_mean(win_odds)
    #     mean_horse_weights, _ = get_group_horse_weights(ps)
    #     mean_gear_weights, _ = get_group_gear_weights(ps)
    #     rating = np.array(list(map(lambda x: x.rating, ps)), dtype=np.float32)
    #     rating = rating[~np.isnan(rating)]
    #
    #     return [
    #         harmonic_mean_odds,
    #         mean_horse_weights,
    #         mean_gear_weights,
    #         np.mean(rating)
    #     ]
    #
    # @staticmethod
    # def get_opponent_participations_summary(session, ps: [Participation]):
    #     race = ps[0].race
    #     h_speed_mean = []
    #     h_ranking_median = []
    #     h_rating_mean = []
    #     h_win_odds_mean = []
    #     h_top_1_ratio = []
    #     h_top_2_ratio = []
    #     h_top_3_ratio = []
    #     h_top_4_ratio = []
    #
    #     j_speed_mean = []
    #     j_ranking_median = []
    #     j_rating_mean = []
    #     j_win_odds_mean = []
    #     j_top_1_ratio = []
    #     j_top_2_ratio = []
    #     j_top_3_ratio = []
    #     j_top_4_ratio = []
    #
    #     for p in ps:
    #         horse_ps, jockey_ps, _ = get_relevant_participations(session, race.date, 90, p.horse_id, p.jockey_id, None)
    #         if len(horse_ps) == 0:
    #             continue
    #         h_speed_mean.append(get_group_speed(horse_ps)[0])
    #         h_ranking_median.append(get_group_ranking(horse_ps)[0])
    #         h_rating_mean.append(get_group_rating(horse_ps)[0])
    #         h_win_odds_mean.append(get_group_win_odds(horse_ps)[0])
    #         h_top_1_ratio.append(get_group_top_ratio(horse_ps, 1))
    #         h_top_2_ratio.append(get_group_top_ratio(horse_ps, 2))
    #         h_top_3_ratio.append(get_group_top_ratio(horse_ps, 3))
    #         h_top_4_ratio.append(get_group_top_ratio(horse_ps, 4))
    #
    #         j_speed_mean.append(get_group_speed(jockey_ps)[0])
    #         j_ranking_median.append(get_group_ranking(jockey_ps)[0])
    #         j_rating_mean.append(get_group_rating(jockey_ps)[0])
    #         j_win_odds_mean.append(get_group_win_odds(jockey_ps)[0])
    #         j_top_1_ratio.append(get_group_top_ratio(jockey_ps, 1))
    #         j_top_2_ratio.append(get_group_top_ratio(jockey_ps, 2))
    #         j_top_3_ratio.append(get_group_top_ratio(jockey_ps, 3))
    #         j_top_4_ratio.append(get_group_top_ratio(jockey_ps, 4))
    #
    #     return [
    #         np.mean(h_speed_mean),
    #         np.mean(h_ranking_median),
    #         np.mean(h_rating_mean),
    #         np.mean(h_win_odds_mean),
    #         np.mean(h_top_1_ratio),
    #         np.mean(h_top_2_ratio),
    #         np.mean(h_top_3_ratio),
    #         np.mean(h_top_4_ratio),
    #         np.mean(j_speed_mean),
    #         np.mean(j_ranking_median),
    #         np.mean(j_rating_mean),
    #         np.mean(j_win_odds_mean),
    #         np.mean(j_top_1_ratio),
    #         np.mean(j_top_2_ratio),
    #         np.mean(j_top_3_ratio),
    #         np.mean(j_top_4_ratio),
    #     ]
    #
    # def _load_from_db(self, session, start_date=None, end_date=None) -> (np.ndarray, np.ndarray):
    #     ps = get_training_participations(session, start_date, end_date)
    #     m = len(ps)
    #
    #     result_x = np.zeros((m, self.input_features), dtype=np.float32)
    #     result_y = np.array(list(map(get_ranking_from_participation, ps[:m])), dtype=np.float32)
    #
    #     for idx in tqdm(range(m), desc="Loading data"):
    #         p = ps[idx]
    #         res = load_individual_participation(session, p)
    #         opponent_ps = list(filter(lambda x: x.horse_id != p.horse_id, p.race.participations))
    #         opponent_ps = remove_unranked_participants(opponent_ps)
    #         opponent_data = self.get_opponent_participation_data(opponent_ps)
    #         opponent_ps_summary = self.get_opponent_participations_summary(session, opponent_ps)
    #         result_x[idx, :INDIVIDUAL_FEATURES] = res
    #         result_x[idx, INDIVIDUAL_FEATURES:self.input_features] = np.array(opponent_data + opponent_ps_summary)
    #
    #     return np.nan_to_num(result_x), np.nan_to_num(result_y)
    #
    # def load_predict(self, session, data):
    #     combinations = []
    #     result = np.zeros((len(data), self.input_features), dtype=np.float32)
    #     for idx, d in enumerate(data):
    #         combinations.append(d["number"])
    #         result[idx, :INDIVIDUAL_FEATURES] = np.copy(load_individual_predict(session, **d))
    #
    #         opponents = d["opponents"]
    #         horse_weights = []
    #         gear_weights = []
    #         win_odds = []
    #
    #         h_speed_mean = []
    #         h_rating_mean = []
    #         h_ranking_median = []
    #         h_win_odds_mean = []
    #         h_top_1_ratio = []
    #         h_top_2_ratio = []
    #         h_top_3_ratio = []
    #         h_top_4_ratio = []
    #
    #         j_speed_mean = []
    #         j_ranking_median = []
    #         j_rating_mean = []
    #         j_win_odds_mean = []
    #         j_top_1_ratio = []
    #         j_top_2_ratio = []
    #         j_top_3_ratio = []
    #         j_top_4_ratio = []
    #
    #         rating = np.array(list(map(lambda x: x["rating"], opponents)), dtype=np.float32)
    #         rating = rating[~np.isnan(rating)]
    #
    #         for o in opponents:
    #             horse_weights.append(o["horse_weight"])
    #             gear_weights.append(o["gear_weight"])
    #             win_odds.append(o["win_odds"])
    #
    #             horse_ps, jockey_ps, _ = get_relevant_participations(session, d["date"], 90, o["horse_id"],
    #                                                                  o["jockey_id"], None)
    #
    #             if len(horse_ps) == 0:
    #                 continue
    #
    #             h_speed_mean.append(get_group_speed(horse_ps)[0])
    #             h_ranking_median.append(get_group_ranking(horse_ps)[0])
    #             h_rating_mean.append(get_group_rating(horse_ps)[0])
    #             h_win_odds_mean.append(get_group_win_odds(horse_ps)[0])
    #             h_top_1_ratio.append(get_group_top_ratio(horse_ps, 1))
    #             h_top_2_ratio.append(get_group_top_ratio(horse_ps, 2))
    #             h_top_3_ratio.append(get_group_top_ratio(horse_ps, 3))
    #             h_top_4_ratio.append(get_group_top_ratio(horse_ps, 4))
    #
    #             j_speed_mean.append(get_group_speed(jockey_ps)[0])
    #             j_ranking_median.append(get_group_ranking(jockey_ps)[0])
    #             j_rating_mean.append(get_group_rating(jockey_ps)[0])
    #             j_win_odds_mean.append(get_group_win_odds(jockey_ps)[0])
    #             j_top_1_ratio.append(get_group_top_ratio(jockey_ps, 1))
    #             j_top_2_ratio.append(get_group_top_ratio(jockey_ps, 2))
    #             j_top_3_ratio.append(get_group_top_ratio(jockey_ps, 3))
    #             j_top_4_ratio.append(get_group_top_ratio(jockey_ps, 4))
    #
    #         opponent_participation_data = [
    #             statistics.harmonic_mean(win_odds),
    #             np.mean(horse_weights),
    #             np.mean(gear_weights),
    #             np.mean(rating),
    #             np.mean(h_speed_mean),
    #             np.mean(h_ranking_median),
    #             np.mean(h_rating_mean),
    #             np.mean(h_win_odds_mean),
    #             np.mean(h_top_1_ratio),
    #             np.mean(h_top_2_ratio),
    #             np.mean(h_top_3_ratio),
    #             np.mean(h_top_4_ratio),
    #             np.mean(j_speed_mean),
    #             np.mean(j_ranking_median),
    #             np.mean(j_rating_mean),
    #             np.mean(j_win_odds_mean),
    #             np.mean(j_top_1_ratio),
    #             np.mean(j_top_2_ratio),
    #             np.mean(j_top_3_ratio),
    #             np.mean(j_top_4_ratio),
    #         ]
    #
    #         result[idx, INDIVIDUAL_FEATURES:self.input_features] = np.array(opponent_participation_data, dtype=np.float32)
    #     return combinations, np.nan_to_num(result)

    def _load_from_db(self, session, start_date=None, end_date=None):
        ps = get_training_participations(session, start_date, end_date)
        m = len(ps)

        result = np.zeros((m, self.input_features))

    def load_predict(self, session, data):
        pass
