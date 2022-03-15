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
    return dataframe[["Theta", "Pitch", "Duty Cycle", "Fill Factor", "Lambda", "Mode"]].values


def get_labels(dataframe:pandas.DataFrame)->torch.Tensor:
    return dataframe[['Transmission']].values


def transform_labels(values):
    return np.log10(np.abs(values))


def percent_error(array1, array2)->float:
    return 0


def loss_function(labels, target)->float:
    return 0


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


class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred+1), torch.log(actual+1)))


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        return self.mse(pred, actual)


class mean_absolute_error(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, actual):
        return torch.mean(torch.abs((pred - actual) / actual))


# Get the start time
start_time = time.time()

# Load the dataset from saved CSV
training_set = pandas.read_csv('DATA_FILES/training_set.csv')
testing_set = pandas.read_csv('DATA_FILES/testing_set.csv')

# TRAINING SET
# Get the x, y values
x = get_features(training_set)
y = transform_labels(get_labels(training_set))

# TESTING SET
# Get the x, y values
x_test = torch.tensor(get_features(testing_set), dtype=torch.float32)
y_test = torch.tensor(transform_labels(get_labels(testing_set)), dtype=torch.float32)

# Dataloader
Dataset = GratingCouplerDataset(x, y)
dataloader = DataLoader(dataset = Dataset, batch_size=10000)

# MODEL AND PARAMETERS
GratingCouplerNet = Network()

learning_rate = 0.001
weight_decay = 0.0005
optimizer = torch.optim.Adam(GratingCouplerNet.parameters(), lr=learning_rate, weight_decay=weight_decay)
mae_loss = mean_absolute_error()
mse_loss = MSELoss()

# Learning rate scehduler
decayRate = 0.999
learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

# INITLIAZE EPOCH AND LOSSES
epoch = 0
max_epoch = 250

# Training loss
loss = 1000

# Valdiation loss
validation_loss = 1000

training_losses = []
testing_losses = []
learning_rates = []

# Train GratingCouplerNet
for epoch in range(max_epoch):    
    for i, (x_train, y_train) in enumerate(dataloader):
        test_prediction = GratingCouplerNet(x_test)

        # Evaluate the training MSE loss from the Training set
        prediction = GratingCouplerNet(x_train)
        
        training_mse_error = mse_loss(prediction, y_train)

        # EVALUATE THE TESTING LOSS FROM THE TESTING SET
        testing_mse_error = mse_loss(test_prediction, y_test)


        # Zero the gradients in the Network
        optimizer.zero_grad()

        # Update the weights and step the optimizer
        training_mse_error.backward()
        optimizer.step()

        print("Batch: {},"\
              " Training Loss (MSE): {:0.6f},"\
              " Testing Loss (MSE): {:0.6f}".format(i, training_mse_error, testing_mse_error))

    # Step the learning rate scheduler
    # learning_rate_scheduler.step()

    print("\nEpoch/Time: {}/{:0.6f}, "\
          "lr: {:0.8f}, "\
          "wd: {:0.8f}, "\
          "Training Loss: {:0.6f}, "\
          "Testing Loss: {:0.6f}\n".format(epoch, 
                                          (time.time()-start_time)/60,
                                           learning_rate_scheduler.get_last_lr()[0],
                                           weight_decay, 
                                           training_mse_error, 
                                           testing_mse_error))

    # Append training and validation losses to lists 
    training_losses.append(training_mse_error.detach().numpy())
    testing_losses.append(testing_mse_error.detach().numpy())
    learning_rates.append(learning_rate_scheduler.get_last_lr()[0])

    # Save the model
    torch.save(GratingCouplerNet, 'GratingCouplerNetModel_weight_decay')

    # Save the losses to a dataframe and csv file
    d = {'training_loss': training_losses, 'testing_loss': testing_losses, 'learning_rate': learning_rates}
    dataframe = pandas.DataFrame(data=d)
    dataframe.to_csv('DATA_FILES/training_stats_weight_decay.csv')

print("Execution time: {}".format(time.time() - start_time))
