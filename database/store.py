"""
Provides an API for storing loaded_data into the database. Should only be used for the control.
"""

from contextlib import contextmanager

from . import get_session
from ._horse import Horse
from ._race import Race
from ._participation import Participation
from ._jockey import Jockey
from ._trainer import Trainer
from ._training import Training

import utils.utils as utils


class Store:
    @contextmanager
    def _get_session(self):
        session = get_session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def store_horse(self, horse_data):
        with (self._get_session() as session):
            horse = session.query(Horse).filter(Horse.id == horse_data['id']).one_or_none()
            if horse:
                # horse_id won't change
                horse.name_eng = horse_data['name_eng']
                horse.name_chi = horse_data.get('name_chi', None)
                horse.age = horse_data['age']
                horse.retired = horse_data['retired']
                horse.age = horse_data['age']
                # horse.sex = horse_data["sex"]
                horse.retired = horse_data['retired']
                horse.origin = horse_data['origin']
                horse.colour = horse_data['colour']
                horse.trainer_id = horse_data['trainer_id']
            else:
                new_horse = Horse(
                    id=horse_data['id'],
                    name_chi=horse_data.get('name_chi', None),
                    name_eng=horse_data['name_eng'],
                    age=horse_data['age'],
                    sex=horse_data["sex"],
                    retired=horse_data['retired'],
                    origin=horse_data['origin'],
                    colour=horse_data['colour'],
                    trainer_id=horse_data['trainer_id'],
                    url=horse_data['url'],
                )
                session.add(new_horse)


    def store_race(self, race_data):
        # should be completely static, so should not have duplicate
        race_id = utils.build_race_id(race_data['season_id'], race_data['date'])

        with self._get_session() as session:
            # TODO: get rid of bad commit here
            if session.query(Race).filter(Race.id == race_id).one_or_none():
                return
            new_race = Race(
                id=race_id,
                season_id=race_data['season_id'],
                season=utils.calc_season(race_data["date"]),
                date=race_data["date"],
                race_class=race_data["race_class"],
                distance=race_data["distance"],
                location=race_data["location"],
                course=race_data["course"],
                condition=race_data["condition"],
                total_bet=race_data["total_bet"],
                url=race_data["url"],
            )
            session.add(new_race)


    def store_jockey(self, jockey_data):
        with self._get_session() as session:
            jockey = session.query(Jockey).filter(Jockey.name == jockey_data['name']).one_or_none()
            if jockey:
                jockey.age = jockey_data['age']
            else:
                new_jockey = Jockey(
                    name=jockey_data['name'],
                    age=jockey_data['age'],
                    url=jockey_data['url'],
                )
                session.add(new_jockey)


    def store_trainer(self, trainer_data):
        with self._get_session() as session:
            trainer = session.query(Trainer).filter(Trainer.name == trainer_data['name']).one_or_none()
            if trainer:
                trainer.age = trainer_data['age']
            else:
                new_trainer = Trainer(
                    name=trainer_data['name'],
                    age=trainer_data['age'],
                    url=trainer_data['url'],
                )
                session.add(new_trainer)


    def store_training(self, training_data):
        # should be completely static, so should not have duplicate
        with self._get_session() as session:
            new_training = Training(
                horse_id=training_data['horse_id'],
                date=training_data['date'],
                train_type=training_data['trainType'],
                location=training_data['location'],
                track=training_data['track'],
                description=training_data['description'],
            )
            session.add(new_training)


    def store_participation(self, participation_data):
        # should be completely static, so should not have duplicate
        with self._get_session() as session:
            p = session.query(Participation).filter(Participation.horse_id == participation_data['horse_id']).filter(
                    Participation.race_id == participation_data["race_id"]).one_or_none()
            if p is not None:
                p.lane = participation_data.get('lane', p.lane)
                p.rating = participation_data.get('rating', p.rating)
                p.gear_weight = participation_data.get('gear_weight', p.gear_weight)
                p.horse_weight = participation_data.get('horse_weight', p.horse_weight)
                p.win_odds = participation_data.get('win_odds', p.win_odds)
                p.ranking = participation_data.get('ranking', p.ranking)
                p.jockey_id = participation_data.get('jockey_id', p.jockey_id)
                p.finish_time = participation_data.get("finish_time", p.finish_time)
            else:
                new_participation = Participation(
                    horse_id=participation_data['horse_id'],
                    race_id=participation_data['race_id'],
                    lane=participation_data['lane'],
                    rating=participation_data['rating'],
                    gear_weight=participation_data['gear_weight'],
                    horse_weight=participation_data['horse_weight'],
                    win_odds=participation_data['win_odds'],
                    ranking=participation_data['ranking'],
                    jockey_id=participation_data.get('jockey_id'),
                    finish_time=participation_data.get("finish_time"),
                )
                session.add(new_participation)
