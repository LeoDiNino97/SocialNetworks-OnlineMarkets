from torch_geometric.nn import GCNConv, GATConv, SAGEConv
import torch.nn as nn
import torch


class GCNNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        # Call the initializer of the parent class (torch.nn.Module)
        super().__init__()

        # Define the first GCN layer with in_channels and hidden_channels
        self.conv1 = GCNConv(in_channels, hidden_channels, normalize=True, bias=True)

        # Define the second GCN layer with hidden_channels and out_channels
        self.conv2 = GCNConv(hidden_channels, out_channels, normalize=True, bias = True)

        
    # Define the forward pass of the model
    def forward(self, x, edge_index):

        # Pass the input through the first GCN layer and apply the ReLU activation function
        x = self.conv1(x, edge_index).relu() 

        # Pass the result through the second GCN layer
        x = self.conv2(x, edge_index) 

        # Return the final output
        return x
    
class SAGENet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        # Call the initializer of the parent class (torch.nn.Module)
        super().__init__()

        # Number of layers in the architecture
        self.num_layers = len(hidden_channels)

        # Definition of the layers
        self.layers = []

        for i in range(0,self.num_layers):
            if i == 0:
                self.layers.append(SAGEConv(in_channels,hidden_channels[i]))
            elif i > 0 and i < self.num_layers:
                self.layers.append(SAGEConv(hidden_channels[i-1],hidden_channels[i]))
            
        self.layers.append(SAGEConv(hidden_channels[self.num_layers-1], out_channels))
        
        # Torch wrapper for the layers list
        self.layers = nn.ModuleList(self.layers)
        
    # Define the forward pass of the model
    def forward(self, x, edge_index):

        for i in range(0,self.num_layers-1):
            x = self.layers[i](x, edge_index).relu()

        # Return the final output
        return self.layers[self.num_layers - 1](x, edge_index)
    
class GATNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        # Call the initializer of the parent class (torch.nn.Module)
        super().__init__()

        # Number of layers in the architecture
        self.num_layers = len(hidden_channels)

        # Definition of the layers
        self.layers = []

        for i in range(0,self.num_layers):
            if i == 0:
                self.layers.append(GATConv(in_channels,hidden_channels[i]))
            elif i > 0 and i < self.num_layers:
                self.layers.append(GATConv(hidden_channels[i-1],hidden_channels[i]))
            
        self.layers.append(GATConv(hidden_channels[self.num_layers-1], out_channels))
        
        # Torch wrapper for the layers list
        self.layers = nn.ModuleList(self.layers)
        
    # Define the forward pass of the model
    def forward(self, x, edge_index):

        for i in range(0,self.num_layers-1):
            x = self.layers[i](x, edge_index).relu()

        # Return the final output
        return self.layers[self.num_layers - 1](x, edge_index)
    