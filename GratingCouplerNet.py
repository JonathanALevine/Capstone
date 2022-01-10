import torch
from torch import nn
import numpy as np
import pandas


features_path = "features.csv"
labels_path = "labels.csv"
dataset_path = "data.csv"
new_dataset = "output_frame.csv"


def get_dataset(path_to_dataset:str)->pandas.DataFrame:
    return pandas.read_csv(path_to_dataset, header=None)


def get_dataset_no_zeroes(dataframe:pandas.DataFrame)->pandas.DataFrame:
    return dataframe[dataframe[0] != 0]


def get_features(dataframe:pandas.DataFrame)->torch.Tensor:
    return dataframe[["Fill Factor", "Pitch", "Duty Cycle", "Theta", "Lambda", "Mode"]]


def get_labels(dataframe:pandas.DataFrame)->torch.Tensor:
    return dataframe[['Transmission']]


def get_features_no_zeroes(dataframe:pandas.DataFrame)->torch.Tensor:
    return torch.tensor(dataframe[dataframe[0] != 0].iloc[:,[0,1,2]].values)


def get_labels_no_zeroes(dataframe:pandas.DataFrame)->torch.Tensor:
    return torch.tensor(dataframe[dataframe[0] != 0].iloc[:,[3, 4]].values)
    

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


class Network(nn.Module):
    def __init__(self):
        super().__init__()

        # Inputs to hidden layer linear transformation
        self.input = nn.Linear(6, 100)
        self.first_hidden = nn.Linear(100, 50)
        self.second_hidden = nn.Linear(50, 50)
        self.third_hidden = nn.Linear(50, 50)
        self.output = nn.Linear(50, 1)

        self.relu = nn.ReLU()

    def forward(self, x:[])->[]:
        # 3 -> 2
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
        # 2 -> 2
        x = self.output(x)

        return x


dataset = pandas.read_csv("DATA_FILES/dataset.csv")
dataset["Mode"] = dataset["Mode"].astype('category').cat.codes
dataset = dataset[dataset["Mode"]!=0]

training_set = dataset.sample(frac = 0.85)

testing_set = dataset.drop(training_set.index)

# GET THE TRAINING DATA
X = transform_features(get_features(training_set))
X_normed = norm(X)
y = transform_labels(get_labels(training_set))

# GET THE TESTING DATA
X_test = transform_features(get_features(testing_set))
X_test_normed = norm(X_test)
y_test = transform_labels(get_labels(testing_set))

# MODEL AND PARAMETERS
GratingCouplerNet = Network()
optimizer = torch.optim.Adam(GratingCouplerNet.parameters(), lr=0.0001)
loss_function = torch.nn.MSELoss()

# INITLIAZE EPOCH AND LOSSES
epoch = 0
loss = 100
validation_loss = 100

# MODEL AND PARAMETERS
GratingCouplerNet = Network()
optimizer = torch.optim.Adam(GratingCouplerNet.parameters(), lr=0.0001)
loss_function = torch.nn.MSELoss()

# INITLIAZE EPOCH AND LOSSES
epoch = 0

loss = 100
validation_loss = 100

# Train GratingCouplerNet
while validation_loss > 0.8:
    prediction = GratingCouplerNet(X_normed)
    loss = loss_function(prediction, y.float())
    test_prediction = GratingCouplerNet(X_test_normed)
    validation_loss = loss_function(test_prediction, y_test.float())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    epoch += 1
    print("Epoch: {}, Training Loss: {:0.6f}, Validation Loss: {:0.6f}".format(epoch, loss, validation_loss))


# Save the model
torch.save(GratingCouplerNet, 'GratingCouplerNetModel')