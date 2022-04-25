import torch
from torch import nn
import numpy as np
import pandas
import time
from torch.utils.data import Dataset, DataLoader


def get_features(dataframe:pandas.DataFrame)->torch.Tensor:
	return dataframe[["Theta", "Pitch", "Duty Cycle", "Fill Factor", "Lambda", "Mode"]].values


def get_labels(dataframe:pandas.DataFrame)->torch.Tensor:
	return dataframe[['Transmission']].values


def transform_labels(values):
	return np.log10(np.abs(values))


class MSELoss(nn.Module):
	def __init__(self):
		super().__init__()
		self.mse = nn.MSELoss()

	def forward(self, pred, actual):
		return self.mse(pred, actual)


class GratingCouplerDataset(torch.utils.data.Dataset):
	def __init__(self, x, y):
		self.x = torch.tensor(x, dtype=torch.float32)
		self.y = torch.tensor(y, dtype=torch.float32)
		self.length = self.x.shape[0]
		
		
	def __getitem__(self, idx):
		return self.x[idx], self.y[idx]
	
	
	def __len__(self):
		return self.length   


class Network(nn.Module):
		def __init__(self):
				super().__init__()

				# Layer sizes
				self.input = nn.Linear(6, 100)
				self.first_hidden = nn.Linear(100, 250)
				self.second_hidden = nn.Linear(250, 500)
				self.third_hidden = nn.Linear(500, 250)
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


if __name__=='__main__':

	training_losses_mse = []
	testing_losses_mse = []
	learning_rates = []

	start_time = time.time()

	# GET THE TRAINING SET
	training_set = pandas.read_csv('DATASETS/NOISE/TRAINING_MIN_MAX_NORMALIZED_23March2022.csv')
	training_set = training_set.sample(frac=1)

	x = get_features(training_set)
	y = transform_labels(get_labels(training_set))

	x_training_set = torch.tensor(x, dtype=torch.float32)
	y_training_set = torch.tensor(y, dtype=torch.float32)

	# # GET THE TESTING SET
	testing_set = pandas.read_csv('DATASETS/NOISE/TESTING.csv')

	x_test = torch.tensor(get_features(training_set), dtype=torch.float32)
	y_test = torch.tensor(transform_labels(get_labels(training_set)), dtype=torch.float32)

	# Dataloader
	Dataset = GratingCouplerDataset(x, y)
	dataloader = DataLoader(dataset = Dataset, batch_size=10000)

	# MODEL AND PARAMETERS
	GratingCouplerNet = Network()

	learning_rate = 0.001
	weight_decay = 0.0000
	optimizer = torch.optim.Adam(GratingCouplerNet.parameters(), lr=learning_rate, weight_decay=weight_decay)

	# Learning rate scehduler
	decayRate = 0.999
	learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

	mse_loss = MSELoss()

	# INITLIAZE EPOCH AND LOSSES
	epoch = 0
	max_epoch = 250

	for epoch in range(max_epoch):    
		for i, (x_train, y_train) in enumerate(dataloader):

			test_prediction = GratingCouplerNet(x_test)

			# Evaluate the training MSE loss from the Training set
			prediction = GratingCouplerNet(x_train)
			
			training_mse_error = mse_loss(prediction, y_train)
			# testing_mse_error = mse_loss(test_prediction, y_test)

			# Zero the gradients in the Network
			optimizer.zero_grad()

			# Update the weights and step the optimizer
			training_mse_error.backward()
			optimizer.step()

			print("Batch: {},"\
				  " Training Loss (MSE): {:0.6f},"\
				  " Training Loss (MSE): {:0.6f}".format(i, training_mse_error, 0))

		# GET THE MSE
		value1 = mse_loss(GratingCouplerNet(x_training_set), 
											y_training_set).detach().numpy()
		training_losses_mse.append(value1)

		# value2 = mse_loss(GratingCouplerNet(x_test), y_test).detach().numpy()
		# testing_losses_mse_losses_mse.append(value2)

		print("\nEpoch/Time: {}/{:0.6f}, "\
		  "lr: {:0.8f}, "\
		  "wd: {:0.8f}, "\
		  "Training Loss: {:0.6f},"\
		  "Testing Loss: {:0.6f}\n".format(epoch, 
										  (time.time()-start_time)/60,
										  learning_rate_scheduler.get_last_lr()[0],
										  weight_decay, 
										  value1,
										  0))

		learning_rates.append(learning_rate_scheduler.get_last_lr()[0])

	# Save the model
	torch.save(GratingCouplerNet, 'DATASETS/test_network')

	# Save the losses to a dataframe and csv file
	d = {'training_loss_mse': training_losses_mse,
		 'testing_loss_mse' : testing_losses_mse,
		 'learning_rate': learning_rates}
	dataframe = pandas.DataFrame(data=d)
	dataframe.to_csv('DATASETS/training_stats_23March2022.csv')
