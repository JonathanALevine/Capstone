import time

import numpy as np
import pandas
import torch
from torch.utils.data import DataLoader

import pickle_functions
from Dataset import Dataset
from get_features import get_features
from get_labels import get_labels
from MSE import MSE
from Network import Network
from transform_labels import transform_labels


def train(num_epochs:int, dataloader:DataLoader, optimizer:torch.optim)->None:
    mse_loss = MSE()
    # Train GratingCouplerNet
    for epoch in range(num_epochs):
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


def get_device_for_training()->str:
    # Use the GPU for training
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    return dev


path_to_training_set = 'datasets/training_set/09May2022/training_set_normalized.pbz2'
training_set = pickle_functions.decompress_pickle(path_to_training_set).sample(frac=1)
path_to_testing_set = 'datasets/testing_set/testing_set_normalized.pbz2'
testing_set = pickle_functions.decompress_pickle(path_to_testing_set).sample(frac=1)


if __name__ == '__main__':
    device = get_device_for_training()
    print(device)

    # Get the start time
    start_time = time.time()

    # TRAINING SET
    # Get the x, y values
    x = torch.tensor(get_features(training_set), dtype=torch.float64).to(device)
    y = torch.tensor(get_labels(training_set), dtype=torch.float64).to(device)

    # TESTING SET
    # Get the x, y values
    x_test = torch.tensor(get_features(testing_set), dtype=torch.float64).to(device)
    y_test = torch.tensor(get_labels(testing_set), dtype=torch.float64).to(device)

    # dataloader for batching
    Dataset = Dataset(x, y)
    dataloader = DataLoader(dataset = Dataset, batch_size=100000)

    # MODEL AND PARAMETERS
    GratingCouplerNet = Network(num_features=6, num_labels=1, num_nodes=100).to(device)

    learning_rate = 0.001
    weight_decay = 0
    optimizer = torch.optim.Adam(GratingCouplerNet.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Learning rate scehduler
    decayRate = 0.999
    learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

    # TRAIN THE NETWORK
    train(num_epochs=10, dataloader=dataloader, optimizer=optimizer)

    print("Execution time: {}".format(time.time() - start_time))
