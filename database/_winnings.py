from sqlalchemy import Column, String, DECIMAL, ForeignKey, Integer
from sqlalchemy.orm import relationship

from ._base import Base


class Winnings(Base):
    __tablename__ = "winnings"

    race_id = Column(String, ForeignKey("race.id"), nullable=False, primary_key=True)
    pool = Column(String, nullable=False, primary_key=True)
    combination = Column(String, nullable=False, primary_key=True)
    amount  = Column(DECIMAL(10, 2), nullable=False)

    race = relationship("Race", back_populates="winnings")
