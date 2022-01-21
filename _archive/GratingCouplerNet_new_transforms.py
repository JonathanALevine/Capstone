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


def norm(tensor:torch.tensor)->torch.tensor:
        return torch.nn.functional.normalize(tensor).float()
        
        
def magnitude(vector:np.array)->float:
        return np.linalg.norm(vector)


def get_progress(first_num:float, second_num:float)->float:
    return (first_num - second_num)*100


def check_progress(first_num, second_num)->bool:
    if get_progress(first_num, second_num) < 2:
        return True


class GratingCouplerDataset(Dataset):
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
                self.celu = nn.CELU()

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
y = get_labels(training_set)
Dataset = GratingCouplerDataset(x, y)
dataloader = DataLoader(dataset = Dataset, batch_size=10000)

# TESTING SET
# Get the x, y values
x_test = torch.tensor(get_features(testing_set))
y_test = torch.tensor(get_labels(testing_set))

# MODEL AND PARAMETERS
GratingCouplerNet = Network()
learning_rate = 0.0001
weight_decay = 0.0005
optimizer = torch.optim.Adam(GratingCouplerNet.parameters(), lr=learning_rate, weight_decay=weight_decay)
loss_function = torch.nn.MSELoss()

# INITLIAZE EPOCH AND LOSSES
epoch = 0
max_epoch = 100

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
        prediction = GratingCouplerNet(norm(x_train))
        loss = loss_function(prediction, y_train.reshape(-1, 1))

        # EVALUATE THE TESTING LOSS FROM THE TESTING SET
        testing_loss = loss_function(test_prediction, y_test.reshape(-1, 1))

        # Zero the gradients in the Network
        optimizer.zero_grad()

        # Update the weights and step the optimizer
        loss.backward()
        optimizer.step()

        print("Batch: {}, Training Loss: {:0.6f}, Testing Loss: {:0.6f}".format(i, loss, testing_loss))

    print("\nEpoch: {}, "\
          "lr: {}, "\
          "wd: {}, "\
          "Training Loss: {:0.6f}, "\
          "Testing Loss: {:0.6f}\n".format(epoch, learning_rate, weight_decay, loss, testing_loss))

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

print("Execution time: {}".format(time.time() - start_time))

