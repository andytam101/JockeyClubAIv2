def none_if_invalid(
        x,
        invalid_val,
        type_cast=lambda x: x  # default type_cast function is the identity function
):
    if x == invalid_val:
        return None
    else:
        return type_cast(x)


def extract_jockey_trainer_id_from_url(url: str):
    url = url.lower()
    args = url.split("?")[1]
    args = args.split("&")

    for arg in args:
        prompt, value = arg.split("=")
        if prompt == "trainerid" or prompt == "jockeyid":
            return value
