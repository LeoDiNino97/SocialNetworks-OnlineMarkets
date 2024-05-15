import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GCNConv, TAGConv, SAGEConv
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dropout_edge, to_dense_adj

def train(model, optimizer, data):
    # Set the model to training mode
    model.train()

    # Clear gradients
    optimizer.zero_grad()  

    # Forward pass
    out = model(data.x, data.edge_index) 

    # We put the softmax head direct into the training loop through the choice of the loss function
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])

    # Update step
    loss.backward() 
    optimizer.step() 

    return float(loss)

# Define the testing function
@torch.no_grad()  # Disable gradient computation for testing
def test(model, data):

    # Set the model to evaluation mode
    model.eval() 

    # Get predictions
    pred = model(data.x, data.edge_index).argmax(dim=-1)

    accs = []
    
    # Calculate accuracy for training, validation, and test sets
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs

def training_loop(model, optimizer, data, epochs, patience = False):
    # Initialize variables to keep track of the best validation accuracy and test accuracy
    best_val_acc = 0
    test_acc = 0
    train_loss = []
    val_accs = []

    # Training loop
    for epoch in range(1, epochs + 1):
        loss = train(model, optimizer, data)  # Train the model
        train_loss.append(loss)

        train_acc, val_acc, tmp_test_acc = test(model, data)  # Test the model
        val_accs.append(val_acc)

        # Update the best validation accuracy and corresponding test accuracy
        if patience:
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = tmp_test_acc
                best_epoch = epoch

                # Reset early stopping counter
                early_stop_counter = 0  
            else:

                # Increment early stopping counter if no improvement
                early_stop_counter += 1  

            # Log training progress
            # log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Test=test_acc)

            # Check for early stopping
            if early_stop_counter >= patience:
                print(f'Early stopping at epoch {epoch} with best validation accuracy: {best_val_acc:.4f}')
                print(f'Test accuracy at epoch {best_epoch}: {test_acc:.4f}')
                break
        else:
            test_acc = tmp_test_acc 

    return train_loss, val_accs, test_acc