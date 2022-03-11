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


GratingCouplerNet = torch.load('GratingCouplerNetModel')

dataframe = pandas.read_csv('DATA_FILES/dataset_new.csv')
print(dataframe.min(), dataframe.max())

fill_factor = (0.25 - dataframe.min()['Fill Factor'])/(dataframe.max()['Fill Factor'] - dataframe.min()['Fill Factor'])
pitch = (0.75*10**(-6) - dataframe.min()['Pitch'])/(dataframe.max()['Pitch'] - dataframe.min()['Pitch'])
duty_cycle = (0.65 - dataframe.min()['Duty Cycle'])/(dataframe.max()['Duty Cycle'] - dataframe.min()['Duty Cycle'])
theta = (14.375 - dataframe.min()['Theta'])/(dataframe.max()['Theta'] - dataframe.min()['Theta'])
wavelength = np.linspace(1.3*10**(-6), 1.7*10**(-6), 200)
mode = 0

transmission = []

for i in range(len(wavelength)):
	wavelength_normed = (wavelength[i] - min(wavelength))/(max(wavelength) - min(wavelength))
	x = torch.tensor([theta, pitch, duty_cycle, fill_factor, wavelength_normed, mode], dtype=torch.float32)
	val = np.power(10, GratingCouplerNet(x).detach().numpy())
	print(wavelength[i], val)
	transmission.append(val)
