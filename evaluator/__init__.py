from abc import ABC, abstractmethod

class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, horse_id, *args, **kwargs):
        """
        Returns dictionary where key is the horse_id and value is the proportion of money spent on that horse
        :param horse_id:
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    def __call__(self, horse_id, *args, **kwargs):
        return self.evaluate(horse_id, *args, **kwargs)


def assess_evaluator():
    pass
