import torch
from torch import nn
import numpy as np
import pandas


features_path = "features.csv"
labels_path = "labels.csv"


class Network(nn.Module):
	def __init__(self):
		super().__init__()

		# Inputs to hidden layer linear transformation
		self.input = nn.Linear(3, 50)
		self.first_hidden = nn.Linear(50, 50)
		self.second_hidden = nn.Linear(50, 50)
		self.third_hidden = nn.Linear(50, 50)
		self.output = nn.Linear(50, 50)

		self.relu = nn.ReLU()

	def forward(self, x:[])->[]:
		# 3 -> 5
		x = self.input(x)
		x = self.relu(x)
		# 5 -> 5
		x = self.first_hidden(x)
		x = self.relu(x)
		# 5 -> 5
		x = self.second_hidden(x)
		x = self.relu(x)
		# 5 -> 5
		x = self.third_hidden(x)
		x = self.relu(x)
		# 5 -> 5
		x = self.output(x)

		return x


class Dataset():
	def get_values(path_to_values:str)->torch.Tensor:
		return torch.Tensor(pandas.read_csv(path_to_values, header=None).values)


x = torch.nn.functional.normalize(Dataset.get_values(features_path))
Y = torch.nn.functional.normalize(Dataset.get_values((labels_path)))

gratingCouplerNet = Network()
optimizer = torch.optim.SGD(gratingCouplerNet.parameters(), lr=0.0001)
loss_function = torch.nn.MSELoss()

for epoch in range(100000):
	prediction = gratingCouplerNet(x)
	loss = loss_function(prediction, Y)

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	print("Epoch: {}, Loss: {:0.6f}".format(epoch, loss))


print(gratingCouplerNet(x[0]))
print(Y[0])
print(loss_function(gratingCouplerNet(x[0]), Y[0]))
print("Hello")

# grating_coupler = Network()
# print(grating_coupler)
# x = torch.tensor([0.0629*pow(10, -6), 0.0629*pow(10, -6), 0.0629*pow(10, -6)])
# print(x)

# print(grating_coupler(x))


