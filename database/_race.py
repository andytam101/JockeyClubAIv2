from sqlalchemy import Column, Integer, String, Date, Time
from sqlalchemy.orm import relationship

from ._base import Base


class Race(Base):
    __tablename__ = 'race'

    id         = Column(String, primary_key=True)
    season_id  = Column(Integer, nullable=False)
    season     = Column(Integer, nullable=False)
    date       = Column(Date)
    race_class = Column(String)  # stored as string for now, as there are subclasses in class 1
    distance   = Column(Integer)
    location   = Column(String)
    course     = Column(String)
    condition  = Column(String)
    total_bet  = Column(Integer)
    url        = Column(String, unique=True, nullable=False)

    participations = relationship("Participation", back_populates="race")
    winnings = relationship("Winnings", back_populates="race")
