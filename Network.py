import torch
from torch import nn


class Network(nn.Module):
        def __init__(self, num_features:int, num_labels:int, num_nodes:int):
                super().__init__()
                # Layer sizes
                self.input = nn.Linear(num_features, num_nodes).double()
                self.first_hidden = nn.Linear(num_nodes, num_nodes).double()
                self.second_hidden = nn.Linear(num_nodes, num_nodes).double()
                self.third_hidden = nn.Linear(num_nodes, num_nodes).double()
                self.fourth_hidden = nn.Linear(num_nodes, num_nodes).double()
                self.fifth_hidden = nn.Linear(num_nodes, num_nodes).double()
                self.output = nn.Linear(num_nodes, num_labels).double()

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