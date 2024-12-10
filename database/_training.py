from sqlalchemy import Column, Integer, Date, String, ForeignKey
from sqlalchemy.orm import relationship

from ._base import Base


class Training(Base):
    __tablename__ = 'training'

    id          = Column(Integer, primary_key=True, autoincrement=True)
    horse_id    = Column(Integer, ForeignKey('horse.id'))
    date        = Column(Date)
    train_type  = Column(String)
    location    = Column(String)
    track       = Column(String)
    description = Column(String)
    gear        = Column(String)

    horse       = relationship('Horse', back_populates='training')
