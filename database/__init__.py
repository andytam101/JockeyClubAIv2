from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from utils import config
from ._base import Base
from ._horse import Horse
from ._race import Race
from ._participation import Participation
from ._jockey import Jockey
from ._trainer import Trainer
from ._training import Training


_engine = None
_Session: sessionmaker = None

def init_engine(db_path=config.DATABASE_PATH):
    global _engine, _Session
    _engine = create_engine(db_path)
    Base.metadata.create_all(_engine)
    _Session = sessionmaker(bind=_engine)


def get_session():
    if _Session is None:
        raise RuntimeError("Session factory is not initialized. Call init_engine() first.")
    return _Session()
