from sqlalchemy import Column, String, Integer, Boolean, ForeignKey
from sqlalchemy.orm import relationship

from ._base import Base


class Horse(Base):
    __tablename__ = 'horse'

    id             = Column(String, primary_key=True)
    name_chi       = Column(String)
    name_eng       = Column(String)
    sex            = Column(String)
    age            = Column(Integer)
    retired        = Column(Boolean)
    origin         = Column(String)
    colour         = Column(String)
    trainer_id     = Column(Integer, ForeignKey('trainer.id'), nullable=False)
    url            = Column(String, unique=True, nullable=False)

    participations = relationship("Participation", back_populates="horse")
    training       = relationship("Training", back_populates="horse")
    trainer        = relationship("Trainer", back_populates="horses")
