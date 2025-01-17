from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship

from ._base import Base


class Trainer(Base):
    __tablename__ = 'trainer'

    id   = Column(String, primary_key=True)
    name = Column(String)
    age  = Column(Integer)
    url  = Column(String)

    horses = relationship("Horse", back_populates="trainer")
