from model.top_3_NN import Top3NN
from model.top_3_LR import Top3LR


model_dict = {
    "Top3LR": Top3LR,
    "Top3NN": Top3NN,
}


def load_model(model_name, dataloader, model_dir=None, **kwargs):
    try:
        model = model_dict[model_name](dataloader, **kwargs)
    except KeyError:
        print("Invalid model name.")
        return None

    if model_dir:
        model.load(model_dir)
    return model
