from strategy._Strategy import Strategy
import random


class RandomWinnerStrategy(Strategy):
    def __repr__(self):
        return "Random Winner Strategy"

    def _bet(self, session, data):
        horse_nums = list(map(lambda x: x['number'], data))
        horse_num = str(random.choice(horse_nums))
        return {
            "win": {
                horse_num: 10,
            }
        }
