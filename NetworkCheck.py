import torch
from torch import nn
import numpy as np
import pandas
from torch.utils.data import Dataset, DataLoader
import time

pandas.set_option('precision', 20)


class Network(nn.Module):
        def __init__(self):
                super().__init__()

                # Layer sizes
                self.input = nn.Linear(6, 50)
                self.first_hidden = nn.Linear(50, 100)
                self.second_hidden = nn.Linear(100, 250)
                self.third_hidden = nn.Linear(250, 250)
                self.fourth_hidden = nn.Linear(250, 100)
                self.fifth_hidden = nn.Linear(100, 50)
                self.output = nn.Linear(50, 1)

                # Activation functions
                self.relu = nn.ReLU()
                self.sigmoid = nn.Sigmoid()

        def forward(self, x:[])->[]:
                x = self.input(x)
                x = self.relu(x)
                x = self.first_hidden(x)
                x = self.relu(x)
                x = self.second_hidden(x)
                x = self.relu(x)
                x = self.third_hidden(x)
                x = self.relu(x)
                x = self.fourth_hidden(x)
                x = self.relu(x)
                x = self.fifth_hidden(x)
                x = self.relu(x)            
                x = self.output(x)

                return x

def get_features(dataframe:pandas.DataFrame)->torch.Tensor:
    return dataframe[["Fill Factor", "Pitch", "Duty Cycle", "Theta", "Lambda", "Mode"]].values


def get_labels(dataframe:pandas.DataFrame)->torch.Tensor:
    return dataframe[['Transmission']].values

GratingCouplerNet = torch.load('GratingCouplerNetModel').eval()
dataset = pandas.read_csv('DATA_FILES/dataset_new_min_max_normalized.csv')
x = torch.tensor(get_features(dataset), dtype=torch.float32)
predicted_transmissions = np.power(10, GratingCouplerNet(x).detach().numpy())
dataset['Predicted Values'] = predicted_transmissions
dataset.to_csv('DATA_FILES/dataset_with_predicted_values.csv')
