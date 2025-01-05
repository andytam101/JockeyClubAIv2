from abc import ABC, abstractmethod
from typing import override


class Strategy(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    @override
    def __repr__(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def bet(self, session, data):
        raise NotImplementedError()
