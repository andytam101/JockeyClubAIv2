from datetime import datetime
from wcwidth import wcswidth


def calc_season(day=datetime.now()):
    # TODO: get rid of magic number, consider using some dynamic form of way of storing turning date/month
    season = day.year + (day.month >= 9)
    return season


def parse_date(date_str):
    return datetime.strptime(date_str, '%d/%m/%Y').date()


def build_race_id(season_id, date):
    return f"{calc_season(date)}:{int(season_id)}"


def remove_unranked_participants(ps):
    return list(filter(lambda x: x.ranking.replace("DH", "").strip().isnumeric() and x.horse is not None, ps))


def pad_chinese(text, width):
    """Pad Chinese text to ensure it aligns properly."""
    current_width = wcswidth(text)
    return text + " " * (width - current_width)

