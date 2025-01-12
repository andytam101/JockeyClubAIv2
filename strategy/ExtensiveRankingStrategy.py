from strategy._Strategy import Strategy


class ExtensiveRankingStrategy(Strategy):
    """
    Still a static strategy, no Reinforcement Learning used.
    """
    def __init__(self, strategy_instance):
        super(ExtensiveRankingStrategy, self).__init__(strategy_instance)

    def __repr__(self):
        return "Extensive Ranking Strategy"