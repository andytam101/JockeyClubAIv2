from strategy._Strategy import Strategy


class ExtensiveRankingStrategy(Strategy):
    def __init__(self, strategy_instance):
        super(ExtensiveRankingStrategy, self).__init__(strategy_instance)

    def __repr__(self):
        return "Extensive Ranking Strategy"