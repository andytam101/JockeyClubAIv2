"""
Provides an API for fetching data from the database
"""

from contextlib import contextmanager

from sqlalchemy.exc import NoResultFound

from . import get_session
from ._horse import Horse
from ._race import Race
from ._participation import Participation
from ._jockey import Jockey
from ._trainer import Trainer
from ._training import Training


class Fetch:
    def __init__(self, model):
        self._model = model

    @contextmanager
    def session_scope(self):
        """Provide a transactional scope around a series of operations."""
        session = get_session()
        try:
            yield session
        finally:
            session.close()

    def all(self):
        with self.session_scope() as session:
            return session.query(self._model).all()

    def one(self, **kwargs):
        """
        Gets the lone data entry using the id primary key. Cannot be used on the participation table.
        :return:
        """
        with self.session_scope() as session:
            query = session.query(self._model)
            for key, value in kwargs.items():
                query = query.filter(getattr(self._model, key) == value)
            try:
                return query.one()
            except NoResultFound as e:
                raise ValueError(f"No result found") from e


    def exist(self, **kwargs):
        try:
            self.one(**kwargs)
        except ValueError:
            return False
        return True


    def filter(self, **kwargs):
        """
        Flexible filter to allow for dynamic queries.
        """
        with self.session_scope() as session:
            return session.query(self._model).filter_by(**kwargs).all()


# Create specific instances of Fetch for each model
FetchHorse = Fetch(Horse)
FetchRace = Fetch(Race)
FetchParticipation = Fetch(Participation)
FetchJockey = Fetch(Jockey)
FetchTrainer = Fetch(Trainer)
FetchTraining = Fetch(Training)
