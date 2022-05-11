import pandas
import torch


def get_features(dataframe:pandas.DataFrame)->torch.Tensor:
    return dataframe[["Theta", "Pitch", "Duty Cycle", "Fill Factor", "Mode", "Lambda"]].values