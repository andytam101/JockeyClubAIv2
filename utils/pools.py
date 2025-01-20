WIN = "WIN"
PLACE = "PLACE"
QUINELLA = "QUINELLA"
Q_PLACE = "QUINELLA PLACE"
FORECAST = "FORECAST"
TIERCE = "TIERCE"
TRIO = "TRIO"
FIRST_4 = "FIRST_4"
QUARTET = "QUARTET"


ALL_POOLS = [WIN, PLACE, QUINELLA, Q_PLACE, FORECAST, TIERCE, TRIO, FIRST_4, QUARTET]


def generate_empty_pool_dictionary(init_value=0):
    if type(init_value) == list:
        return {
            WIN: init_value.copy(),
            PLACE: init_value.copy(),
            QUINELLA: init_value.copy(),
            Q_PLACE: init_value.copy(),
            FORECAST: init_value.copy(),
            TIERCE: init_value.copy(),
            TRIO: init_value.copy(),
            FIRST_4: init_value.copy(),
            QUARTET: init_value.copy(),
        }
    else:
        return {
            WIN: init_value,
            PLACE: init_value,
            QUINELLA: init_value,
            Q_PLACE: init_value,
            FORECAST: init_value,
            TIERCE: init_value,
            TRIO: init_value,
            FIRST_4: init_value,
            QUARTET: init_value,
        }
