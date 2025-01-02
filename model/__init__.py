from model.top_3_NN import Top3NN
from model.top_3_LR import Top3LR
from utils import config

model_dict = {
    "Top3LR": Top3LR,
    "Top3NN": Top3NN,
}


def load_model(model_name):
    model = model_dict[model_name]
    return model()
