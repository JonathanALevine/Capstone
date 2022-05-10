import torch
from torch import nn
import numpy as np
import pandas
import time
from torch.utils.data import Dataset, DataLoader
import bz2
import pickle
import _pickle as cPickle

# Better accuracy is needed!
# torch.backends.cuda.matmul.allow_tf32 = False


# Use the GPU for training
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

device = torch.device(dev)


def get_features(dataframe:pandas.DataFrame)->torch.Tensor:
    return dataframe[["Theta", "Pitch", "Duty Cycle", "Fill Factor", "Lambda", "Mode"]].values


def get_labels(dataframe:pandas.DataFrame)->torch.Tensor:
    return dataframe[['Transmission']].values


def transform_labels(values):
    return 1/np.log10(np.abs(values))


# Pickle a file and then compress it into a file with extension 
def compressed_pickle(title, data):
    with bz2.BZ2File(title + '.pbz2', 'w') as f: 
        cPickle.dump(data, f)


# Load any compressed pickle file
def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)
    return data


class GratingCouplerDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x.clone().detach()
        self.y = y.clone().detach()
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


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        return self.mse(pred, actual)


# Get the start time
start_time = time.time()

# Load the dataset from saved CSV
training_set = decompress_pickle('datasets/training_set_normalized_features.pbz2')
training_set = training_set.sample(frac=1)

testing_set = decompress_pickle('datasets/testing_set_normalized_features.pbz2')
testing_set = testing_set.sample(frac=1)

# TRAINING SET
# Get the x, y values
x = torch.tensor(get_features(training_set), dtype=torch.float32).to(device)
y = torch.tensor(transform_labels(get_labels(training_set)), dtype=torch.float32).to(device)

# TESTING SET
# Get the x, y values
x_test = torch.tensor(get_features(testing_set), dtype=torch.float32).to(device)
y_test = torch.tensor(transform_labels(get_labels(testing_set)), dtype=torch.float32).to(device)

# Dataloader
Dataset = GratingCouplerDataset(x, y)
dataloader = DataLoader(dataset = Dataset, batch_size=10000)

# MODEL AND PARAMETERS
GratingCouplerNet = Network().to(device)

learning_rate = 0.0001
weight_decay = 0
optimizer = torch.optim.Adam(GratingCouplerNet.parameters(), lr=learning_rate)

mse_loss = MSELoss()

# Learning rate scehduler
decayRate = 0.999
learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

# INITLIAZE EPOCH AND LOSSES
epoch = 0
max_epoch = 250

# Training loss
loss = 1000

# Train GratingCouplerNet
for epoch in range(max_epoch):
    for i, (x_train, y_train) in enumerate(dataloader):
        test_prediction = GratingCouplerNet(x_test)

        # EVALUATE THE TRAINING LOSS
        prediction = GratingCouplerNet(x)
        training_mse_error = mse_loss(prediction, y)

        # EVALUATE THE TESTING LOSS
        testing_mse_error = mse_loss(test_prediction, y_test)

        # ZERO THE GRADIENTS IN THE NETWORK
        optimizer.zero_grad()

        # UPDATE THE WEIGHTS AND STEP THE OPTIMIZER
        training_mse_error.backward()
        optimizer.step()

        print("Batch: {},"\
              " Training Loss (MSE): {:0.6f},"\
              " Testing Loss (MSE): {:0.6f}".format(i, training_mse_error, testing_mse_error))

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

    # Save the model
    torch.save(GratingCouplerNet, 'models/GratingCouplerNetModel')

    torch.cuda.empty_cache()

print("Execution time: {}".format(time.time() - start_time))
