from . import Evaluator

class Top3ProbabilityEvaluator(Evaluator):
    def evaluate(self, horse_id, *args, **kwargs):
        """
        Bets 10% on top 3 probabilities.
        :param horse_id:
        :param args:
        :param kwargs:
        :return:
        """
        probabilities = kwargs['probabilities']
        mapping = list(zip(horse_id, probabilities))
        mapping.sort(key=lambda x: x[1], reverse=True)

        first_horse_id  = mapping[0][0]
        second_horse_id = mapping[1][0]
        third_horse_id  = mapping[2][0]

        return {
            first_horse_id: 10,
            second_horse_id: 10,
            third_horse_id: 10,
        }
