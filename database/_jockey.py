from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship

from ._base import Base


class Jockey(Base):
    __tablename__ = 'jockey'

    id   = Column(String, primary_key=True)
    name = Column(String, unique=True)
    age  = Column(Integer)
    url  = Column(String)

    participations = relationship('Participation', back_populates='jockey')