from abc import ABC, abstractmethod
from typing import override


class Strategy(ABC):
    def __init__(self, init_balance=250, **kwargs):
        self.init_balance = init_balance
        self.balance = init_balance

    @override
    def __repr__(self) -> str:
        raise NotImplementedError()

    def bet(self, session, data):
        if self.balance == 0:
            return {}
        else:
            return self._bet(session, data)

    @abstractmethod
    def _bet(self, session, data):
        raise NotImplementedError()

    def update_balance(self, profit):
        # profit can be negative
        self.balance += profit
