"""
Provides an API for fetching loaded_data from the database
"""

from contextlib import contextmanager

from sqlalchemy.exc import NoResultFound

from . import get_session, Winnings
from ._horse import Horse
from ._race import Race
from ._participation import Participation
from ._jockey import Jockey
from ._trainer import Trainer
from ._training import Training


class _FetchDB:
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

    def all_url(self):
        # does not work with participations and training
        return list(map(lambda x: x.url, self.all()))

    def one(self, **kwargs):
        """
        Gets the lone loaded_data entry using the id primary key. Cannot be used on the participation table.
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

    def __call__(self, **kwargs):
        # can modify a bit
        return self.filter(**kwargs)


class Fetch:
    def __init__(self):
        self.fetch_horse = _FetchDB(Horse)
        self.fetch_race = _FetchDB(Race)
        self.fetch_participation = _FetchDB(Participation)
        self.fetch_jockey = _FetchDB(Jockey)
        self.fetch_trainer = _FetchDB(Trainer)
        self.fetch_training = _FetchDB(Training)
        self.fetch_winnings = _FetchDB(Winnings)
        