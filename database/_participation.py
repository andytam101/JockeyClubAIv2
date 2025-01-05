from sqlalchemy import Column, String, Integer, DECIMAL, ForeignKey, Time
from sqlalchemy.orm import relationship

from ._base import Base


class Participation(Base):
    __tablename__ = 'participation'

    horse_id     = Column(String, ForeignKey("horse.id"), primary_key=True, nullable=False)
    race_id      = Column(String, ForeignKey("race.id"), primary_key=True, nullable=False)
    ranking      = Column(String)    # can have special error codes: like DISQ for disqualified
    number       = Column(Integer, nullable=False)
    lane         = Column(Integer)
    rating       = Column(Integer)
    gear_weight  = Column(Integer)
    horse_weight = Column(Integer)
    win_odds     = Column(DECIMAL(3, 1))
    finish_time  = Column(Time)
    jockey_id    = Column(Integer, ForeignKey("jockey.id"))

    horse = relationship("Horse", back_populates="participations")
    race  = relationship("Race", back_populates="participations")
    jockey = relationship("Jockey", back_populates="participations")
