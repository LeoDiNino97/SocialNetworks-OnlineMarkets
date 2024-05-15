from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.utils import dropout_edge, to_dense_adj

import torch
import torch.nn as nn
import torch.functional as F


class GCNNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_features, out_channels):
        # Call the initializer of the parent class (torch.nn.Module)
        super().__init__()

        # Define the first GCN layer with in_channels and hidden_channels
        self.conv1 = GCNConv(in_channels, hidden_channels, normalize=True, bias=True)

        # Define a skip connection layer
        self.skip = nn.Linear(in_channels, hidden_channels)

        # Define the second GCN layer with hidden_channels and out_channels
        self.conv2 = GCNConv(hidden_channels, out_features, normalize=True, bias = True)

        # Define a fully connected head
        self.linear = nn.Linear(out_features, out_channels)
        
        
    # Define the forward pass of the model
    def forward(self, x, edge_index):
        x_ = torch.clone(x)

        # Pass the input through the first GCN layer and apply the ReLU activation function
        edges, _ = dropout_edge(edge_index, p = 0.5)
        x = self.conv1(x, edges)

        #x += self.skip(x_)
        x = x.relu() 

        # Pass the result through the second GCN layer
        edges, _ = dropout_edge(edge_index, p = 0.5)
        x = self.conv2(x, edges).relu()

        # Return the final output
        return self.linear(x)
    
class SAGENet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_features, out_channels):
        # Call the initializer of the parent class (torch.nn.Module)
        super().__init__()

        # Define the first GCN layer with in_channels and hidden_channels
        self.conv1 = SAGEConv(in_channels, hidden_channels, normalize=True, bias=True)

        # Define a skip connection layer
        self.skip = torch.nn.Linear(in_channels, hidden_channels)

        # Define the second GCN layer with hidden_channels and out_channels
        self.conv2 = SAGEConv(hidden_channels, out_features, normalize=True, bias = True)

        # Define a fully connected head
        self.linear = torch.nn.Linear(out_features, out_channels)
        
        
    # Define the forward pass of the model
    def forward(self, x, edge_index):
        x_ = torch.clone(x)

        # Pass the input through the first GCN layer and apply the ReLU activation function
        x = self.conv1(x, edge_index)

        x += self.skip(x_)
        x = x.tanh() 

        # Pass the result through the second GCN layer
        edges, _ = dropout_edge(edge_index, p = 0.3)
        x = self.conv2(x, edges).tanh()

        # Return the final output
        return self.linear(x)
    
class GATNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_features, out_channels):
        # Call the initializer of the parent class (torch.nn.Module)
        super().__init__()

        # Define the first GCN layer with in_channels and hidden_channels
        self.conv1 = GATConv(in_channels, hidden_channels, bias=True)

        # Define a skip connection layer
        self.skip = torch.nn.Linear(in_channels, hidden_channels)

        # Define the second GCN layer with hidden_channels and out_channels
        self.conv2 = GATConv(hidden_channels, out_features, bias = True)

        # Define a fully connected head
        self.linear = torch.nn.Linear(out_features, out_channels)
        
        
    # Define the forward pass of the model
    def forward(self, x, edge_index):
        x_ = torch.clone(x)

        # Pass the input through the first GCN layer and apply the ReLU activation function
        x = self.conv1(x, edge_index)

        x += self.skip(x_)
        x = x.tanh() 

        # Pass the result through the second GCN layer
        edges, _ = dropout_edge(edge_index, p = 0.3)
        x = self.conv2(x, edges).tanh()

        # Return the final output
        return self.linear(x)
    