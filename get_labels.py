import pandas
import torch


def get_labels(dataframe:pandas.DataFrame)->torch.Tensor:
    return dataframe[['Transmission']].values