import torch
from torch import nn
import numpy as np
import pandas
import time
from torch.utils.data import Dataset, DataLoader


features_path = "features.csv"
labels_path = "labels.csv"
dataset_path = "data.csv"
new_dataset = "output_frame.csv"


def get_features(dataframe:pandas.DataFrame)->torch.Tensor:
        return dataframe[["Fill Factor", "Pitch", "Duty Cycle", "Theta", "Lambda", "Mode"]].values


def get_labels(dataframe:pandas.DataFrame)->torch.Tensor:
        return dataframe[['Transmission']].values


def transform_labels(values):
    return np.log10(np.abs(values))


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
                self.first_hidden = nn.Linear(100, 150)
                self.second_hidden = nn.Linear(150, 150)
                self.third_hidden = nn.Linear(150, 100)
                self.output = nn.Linear(100, 1)

                # Activation functions
                self.relu = nn.ReLU()

        def forward(self, x:[])->[]:
                x = self.input(x)
                x = self.relu(x)
                x = self.first_hidden(x)
                x = self.relu(x)
                x = self.second_hidden(x)
                x = self.relu(x)
                x = self.third_hidden(x)
                x = self.relu(x)
                x = self.output(x)

                return x


# Get the start time
start_time = time.time()

# Load the dataset from saved CSV
training_set = pandas.read_csv('DATA_FILES/training_set_normalized.csv')
testing_set = pandas.read_csv('DATA_FILES/testing_set_normalized.csv')

# TRAINING SET
# Get the x, y values
x = get_features(training_set)
y = transform_labels(get_labels(training_set))
Dataset = GratingCouplerDataset(x, y)
dataloader = DataLoader(dataset = Dataset, batch_size=10000)

# TESTING SET
# Get the x, y values
x_test = torch.tensor(get_features(testing_set), dtype=torch.float32)
y_test = torch.tensor(transform_labels(get_labels(testing_set)), dtype=torch.float32)

# MODEL AND PARAMETERS
GratingCouplerNet = Network()
learning_rate = 0.0001
weight_decay = 0.0005
optimizer = torch.optim.Adam(GratingCouplerNet.parameters(), lr=learning_rate, weight_decay=weight_decay)
loss_function = torch.nn.MSELoss()

# INITLIAZE EPOCH AND LOSSES
epoch = 0
max_epoch = 10000

# Training loss
loss = 1000

# Valdiation loss
validation_loss = 100

training_losses = []
testing_losses = []

# Train GratingCouplerNet
for epoch in range(max_epoch):    
    for i, (x_train, y_train) in enumerate(dataloader):
        test_prediction = GratingCouplerNet(x_test)

        # Evaluate the training MSE loss from the Training set
        prediction = GratingCouplerNet(x_train)
        loss = loss_function(prediction, y_train.reshape(-1, 1))

        # EVALUATE THE TESTING LOSS FROM THE TESTING SET
        testing_loss = loss_function(test_prediction, y_test.reshape(-1, 1))

        # Zero the gradients in the Network
        optimizer.zero_grad()

        # Update the weights and step the optimizer
        loss.backward()
        optimizer.step()

        print("Batch: {}, Training Loss: {:0.6f}, Testing Loss: {:0.6f}".format(i, loss, testing_loss))

    print("\nEpoch/Time: {}/{:0.6f}, "\
          "lr: {:0.8f}, "\
          "wd: {:0.8f}, "\
          "Training Loss: {:0.6f}, "\
          "Testing Loss: {:0.6f}\n".format(epoch, (time.time()-start_time)/60, learning_rate, weight_decay, loss, testing_loss))

    # Appennd training and validation losses to lists 
    training_losses.append(loss.detach().numpy())
    testing_losses.append(testing_loss.detach().numpy())


    # Save the model
    torch.save(GratingCouplerNet, 'GratingCouplerNetModel')

    # Save the losses to a dataframe and csv file
    d = {'training_loss': training_losses, 'testing_loss': testing_losses}
    dataframe = pandas.DataFrame(data=d)
    dataframe.to_csv('training_losses.csv')

    if testing_loss < 0.05:
        break

    if (epoch > 0) and (epoch%1000 == 0):
        learning_rate = learning_rate/2
        optimizer = torch.optim.Adam(GratingCouplerNet.parameters(), lr=learning_rate, weight_decay=weight_decay)

print("Execution time: {}".format(time.time() - start_time))

