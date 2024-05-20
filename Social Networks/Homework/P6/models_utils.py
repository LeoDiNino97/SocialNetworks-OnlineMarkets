from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.utils import dropout_edge

import torch
import torch.nn as nn
import torch.nn.functional as F

# ___________________________________________NODE CLASSIFICATION CLASSES____________________________________________________

class GCNNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()

        # Define the first GCN layer with in_channels and hidden_channels
        self.conv1 = GCNConv(in_channels, hidden_channels)

        # Define the second GCN layer with hidden_channels and out_channels
        self.conv2 = GCNConv(hidden_channels, out_channels)

        
    # Define the forward pass of the model
    def forward(self, x, edge_index):
        # Initialize the skip connection
        #x_ = torch.clone(x)

        #x = F.dropout(x, p=0.6, training=self.training)

        # Pass the input through the first GCN layer and apply the ReLU activation function
        #edges, _ = dropout_edge(edge_index, p = 0.3)
        x = self.conv1(x, edge_index).relu() 

        x = F.dropout(x, p=0.5, training=self.training)

        # Pass the result through the second GCN layer
        #edges, _ = dropout_edge(edge_index, p = 0.3)
        
        x = self.conv2(x, edge_index)

        # Realize the skip connection 
        #x = torch.cat([x,x_], dim=1)

        # Fully connected layer
        #x = self.fc(x)

        return F.log_softmax(x, dim=1)
    
class SAGENet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()

        # Define the first GCN layer with in_channels and hidden_channels
        self.conv1 = SAGEConv(in_channels, hidden_channels)

        # Define the second GCN layer with hidden_channels and out_channels
        self.conv2 = SAGEConv(hidden_channels, out_channels)

        
    # Define the forward pass of the model
    def forward(self, x, edge_index):
        # Initialize the skip connection
        #x_ = torch.clone(x)

        #x = self.dropout(x)

        # Pass the input through the first GCN layer and apply the ReLU activation function
        #edges, _ = dropout_edge(edge_index, p = 0.3)
        x = self.conv1(x, edge_index).relu() 

        x = F.dropout(x, p=0.5, training=self.training)

        # Pass the result through the second GCN layer
        #edges, _ = dropout_edge(edge_index, p = 0.3)
        
        x = self.conv2(x, edge_index)

        # Realize the skip connection 
        #x = torch.cat([x,x_], dim=1)

        # Fully connected layer
        #x = self.fc(x)

        return F.log_softmax(x, dim=1)


class GATNet(torch.nn.Module):
    def __init__(self, heads, in_channels, hidden_channels, out_channels):
        super().__init__()

        # Define the first GCN layer with in_channels and hidden_channels
        self.conv1 = GATConv(in_channels, hidden_channels, heads = heads)

        # Define the second GCN layer with hidden_channels and out_channels
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads = 1)
        
        
    # Define the forward pass of the model
    def forward(self, x, edge_index):
        # Initialize the skip connection
        #x_ = torch.clone(x)

        x = F.dropout(x, p=0.5, training=self.training)

        # Pass the input through the first GCN layer and apply the ReLU activation function
        #edges, _ = dropout_edge(edge_index, p = 0.3)
        x = self.conv1(x, edge_index)

        x = F.elu(x)

        x = F.dropout(x, p=0.5, training=self.training)

        # Pass the result through the second GCN layer
        #edges, _ = dropout_edge(edge_index, p = 0.3)
        
        x = self.conv2(x, edge_index)

        # Realize the skip connection 
        #x = torch.cat([x,x_], dim=1)

        # Fully connected layer
        #x = self.fc(x)

        return F.log_softmax(x, dim=1)

# ___________________________________________LINK PREDICTION CLASSES____________________________________________________

class EncoderDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    # Define the encoder pass of the model
    def encode(self, x, edge_index):

        x = self.conv1(x, edge_index).relu() 
        x = self.conv2(x, edge_index)
        return x
    
    # Define the decoder pass of the model
    def decode(self, x_u, x_v):

        # Compute the dot product between the embeddings of  two nodes
        x = torch.sum(x_u * x_v, 
                      dim = -1, 
                      keepdim = True)

        # Wrap in a sigmoid function
        return torch.sigmoid(x)
