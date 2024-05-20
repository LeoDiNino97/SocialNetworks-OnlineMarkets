import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

# _______________________________NODE CLASSIFICATION TASK______________________________

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

def node_classification_train_loop(model, optimizer, data, epochs, patience = False):
    # Initialize variables to keep track of the best validation accuracy and test accuracy
    best_val_acc = 0
    test_acc = 0
    train_loss = []
    val_accs = []

    # Training loop
    for epoch in range(1, epochs + 1):
        loss = train(model, optimizer, data)  # Train the model
        train_loss.append(loss)

        _, val_acc, tmp_test_acc = test(model, data)  # Test the model
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

# ______________________________________________LINK PREDICTION TASK________________________________________

def edge_splitting(edges, nodes, negative=False):

    SEED = 0
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # First sampling for train mask
    if negative:
        edges = negative_sampling(edges, nodes.shape[0], int(edges.shape[1]))

    edges_train_mask = torch.randperm(edges.shape[0])[:int(0.8*edges.shape[1])]

    edges_train = edges[edges_train_mask]
    edges_test = edges[~edges_train_mask]

    # Now we stratify the sampling to split in validation and test
    edges_test_mask = torch.randperm(edges_test.shape[0])[:int(0.5*edges_test.shape[1])]

    edges_test = edges_test[edges_test_mask]
    edges_val = edges_test[~edges_test_mask]

    return edges_train, edges_val, edges_test

def encoder_decoder_train_loop(epochs,
                               model,
                               optimizer,
                               criterion,
                               data,
                               pos_edges_train,
                               neg_edges_train,
                               pos_edges_val,
                               neg_edges_val):
    train_loss = []
    val_auc = []

    for epoch in range(epochs):
        # Set the model in training mode
        model.train()
        optimizer.zero_grad()
        
        # Embed nodes
        node_embeddings = model.encode(data.x, data.edge_index).to(torch.float64)
        
        # Edge predictions and their concatenation

        pos_preds = model.decode(node_embeddings[pos_edges_train[0]], 
                                node_embeddings[pos_edges_train[1]])
        
        neg_preds = model.decode(node_embeddings[neg_edges_train[0]], 
                                node_embeddings[neg_edges_train[1]])
        
        preds = torch.cat([pos_preds, neg_preds], dim=0)

        
        # Defining the tensors for the labels and their concatenation

        pos_labels = torch.ones(pos_preds.size(0), 1)

        neg_labels = torch.zeros(neg_preds.size(0), 1)
        
        labels = torch.cat([pos_labels, neg_labels], dim=0)

        
        # Compute loss and backpropagate
        
        loss = criterion(preds, labels)
        train_loss.append(loss.detach().numpy())
        
        loss.backward()
        optimizer.step()

        # Set the model in evaluation mode
        model.eval()

        with torch.no_grad():
            
            # Edge predictions and their concatenation

            pos_preds = model.decode(node_embeddings[pos_edges_val[0]], 
                                     node_embeddings[pos_edges_val[1]]).squeeze()
            
            neg_preds = model.decode(node_embeddings[neg_edges_val[0]], 
                                     node_embeddings[neg_edges_val[1]]).squeeze()
            
            val_preds = torch.cat([pos_preds, neg_preds], dim=0).detach().numpy()

            # Defining a tensor for the labels
            
            val_labels = torch.cat([torch.ones(pos_preds.size(0)), torch.zeros(neg_preds.size(0))], dim=0).detach().numpy()
            
            # Compute AUC-ROC score
            auc_score = roc_auc_score(val_labels, val_preds)
            val_auc.append(auc_score)

            # Show results
            if epoch % 5 == 0:
                print(f'Epoch: {epoch:<5} | Training Loss: {loss:.4f} | AUC-ROC Score: {auc_score:.4f}')
    
    return train_loss, val_auc

def test_encoder_decoder(model,
                         data,
                         pos_edges_test,
                         neg_edges_test):
    
    node_embeddings = model.encode(data.x, data.edge_index).to(torch.float64)

    # Edge predictions and their concatenation

    pos_preds = model.decode(node_embeddings[pos_edges_test[0]], 
                            node_embeddings[pos_edges_test[1]])

    neg_preds = model.decode(node_embeddings[neg_edges_test[0]], 
                            node_embeddings[neg_edges_test[1]])

    test_preds = torch.cat([pos_preds, neg_preds], dim=0).detach().numpy()

    test_labels = torch.cat([torch.ones(pos_preds.size(0)), torch.zeros(neg_preds.size(0))], dim=0).detach().numpy()
    auc_score = roc_auc_score(test_labels, test_preds)

    print(f' Test AUC-ROC Score: {auc_score:.4f}')
