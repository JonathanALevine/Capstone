import torch
from torch import nn
import numpy as np
import pandas


features_path = "features.csv"
labels_path = "labels.csv"
dataset_path = "data.csv"
new_dataset = "output_frame.csv"


def get_features(dataframe:pandas.DataFrame)->torch.Tensor:
        return dataframe[["Fill Factor", "Pitch", "Duty Cycle", "Theta", "Lambda", "Mode"]]


def get_labels(dataframe:pandas.DataFrame)->torch.Tensor:
        return dataframe[['Transmission']]
        

def transform_features(DataFrame:pandas.DataFrame)->torch.tensor:
        tensor = torch.tensor(DataFrame.values)
        tensor[:,0:5] = np.log10(tensor[:,0:5])
        return tensor


def transform_labels(DataFrame:pandas.DataFrame)->torch.tensor:
        tensor = torch.tensor(DataFrame.values)
        tensor[:,0] = np.log10(np.abs(tensor[:,0]))
        return tensor


def norm(tensor:torch.tensor)->torch.tensor:
        return torch.nn.functional.normalize(tensor).float()
        
        
def magnitude(vector:np.array)->float:
        return np.linalg.norm(vector)


def get_progress(first_num:float, second_num:float)->float:
    return (first_num - second_num)*100


def check_progress(first_num, second_num)->bool:
    if get_progress(first_num, second_num) < 2:
        return True


class Network(nn.Module):
        def __init__(self):
                super().__init__()

                # Layer sizes
                self.input = nn.Linear(6, 50)
                self.first_hidden = nn.Linear(50, 100)
                self.second_hidden = nn.Linear(100, 200)
                self.third_hidden = nn.Linear(200, 200)
                self.fourth_hidden = nn.Linear(200, 100)
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
                x = self.fourth_hidden(x)
                x = self.relu(x)
                x = self.output(x)

                return x


# Get the training data
training_set = pandas.read_csv('DATA_FILES/training_set.csv')

# Get the testing data
testing_set = pandas.read_csv('DATA_FILES/testing_set.csv')

# GET THE TRAINING DATA
X = transform_features(get_features(training_set))
X_normed = norm(X)
y = transform_labels(get_labels(training_set))

# GET THE TESTING DATA
X_test = transform_features(get_features(testing_set))
X_test_normed = norm(X_test)
y_test = transform_labels(get_labels(testing_set))

# MODEL AND PARAMETERS
GratingCouplerNet = torch.load('GratingCouplerNetModel_nn_50_100_200_100_1').eval()
learning_rate = 0.00005
weight_decay = 0.0005
optimizer = torch.optim.Adam(GratingCouplerNet.parameters(), lr=learning_rate, weight_decay=weight_decay)
loss_function = torch.nn.MSELoss()

# INITLIAZE EPOCH AND LOSSES
epoch = 0
max_epoch = 3746

# Training loss
loss = 100

# Valdiation loss
validation_loss = 100

training_losses = []
validation_losses = []

# Train GratingCouplerNet
for epoch in range(max_epoch):

    # Evaluate the test MSE loss from the Test set
    test_prediction = GratingCouplerNet(X_test_normed)
    validation_loss = loss_function(test_prediction, y_test.float())
    
    # Zero the gradients in the Network
    optimizer.zero_grad()

    # Evaluate the training MSE loss from the Training set
    prediction = GratingCouplerNet(X_normed)
    loss = loss_function(prediction, y.float())

    # Update the weights and step the optimizer
    loss.backward()
    optimizer.step()

    print("Epoch: {}, "\
          "lr: {}, "\
          "wd: {}, "\
          "Training Loss: {:0.6f}, "\
          "Validation Loss: {:0.6f}".format(epoch, learning_rate, weight_decay, loss, validation_loss))

    # Appennd training and validation losses to lists 
    training_losses.append(loss.detach().numpy())
    validation_losses.append(validation_loss.detach().numpy())


    # Save the model
    torch.save(GratingCouplerNet, 'GratingCouplerNetModel_nn_50_100_200_100_1_second')

    # Save the losses to a dataframe and csv file
    d = {'training_loss': training_losses, 'validation_loss': validation_losses}
    dataframe = pandas.DataFrame(data=d)
    dataframe.to_csv('losses_nn_50_100_200_100_1_second.csv')

    if validation_loss < 0.25:
        break
