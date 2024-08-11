import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import SubsetRandomSampler
from torch.utils.data.dataloader import default_collate
import os

FONT_DATASET_PATH = "./fonts_image_dataset"



def load_dataset(dataset_path, batch_size):

    # Convert the images to tensors and normalize them
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), transforms.Grayscale(num_output_channels=1)])
    gestures_dataset = torchvision.datasets.ImageFolder(root = dataset_path, transform=transform)
    
    num_classes = len(gestures_dataset.classes)
    
    # Create a list of indices for all the images in the dataset
    dataset_size = len(gestures_dataset)
    indices = list(range(dataset_size))
    np.random.seed(0)
    np.random.shuffle(indices)

    # Split the indices into 60% Training 20% Validation 20% Testing. We need most of the data for training the network, but we must also set aside a bit for validation to fine tune the network, and test the network at the very end.
    split1 = int(0.6 * dataset_size)
    split2 = int(0.8 * dataset_size)
    train_indices, val_indices, test_indices = indices[:split1], indices[split1:split2], indices[split2:]

    # Create a sampler for the training, validation, and testing sets
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    def custom_collate_fn(batch):
    
        # Use the default collate function to batch the data (images)
        batch = default_collate(batch)
        images, labels = batch
        
        # Apply one-hot encoding to the labels
        labels = F.one_hot(labels, num_classes=num_classes)

        return images, labels

    # Create the dataloaders for the training, validation, and testing sets
    train_loader = torch.utils.data.DataLoader(gestures_dataset, batch_size=batch_size,sampler=train_sampler,collate_fn=custom_collate_fn)
    val_loader = torch.utils.data.DataLoader(gestures_dataset, batch_size=batch_size,sampler=val_sampler,collate_fn=custom_collate_fn)
    test_loader = torch.utils.data.DataLoader(gestures_dataset, batch_size=batch_size,sampler=test_sampler,collate_fn=custom_collate_fn)

    print("Done Loading Data")

    return train_loader, val_loader, test_loader, gestures_dataset.classes



def total_error(outputs, labels):
    
    # Find the indices of the max values
    _, indices = torch.max(outputs, dim=1, keepdim=True)

    # Create a tensor of zeros with the same shape as x
    zeros = torch.zeros_like(outputs)

    # Set the max values to 1
    zeros.scatter_(1, indices, 1)
    
    return (zeros != labels).any(dim=1).float().sum()

def evaluate(net, loader, criterion):
    
    net.eval()
    
    total_loss = 0.0
    total_err = 0.0
    total_epoch = 0

    with torch.no_grad():

        for i, (inputs, labels) in enumerate(loader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
                
            # Forward pass
            outputs = net(inputs)
            
            # Calculate the statistics
            total_err += total_error(outputs, labels)
            total_loss += criterion(outputs, labels.float()).item()
            total_epoch += len(labels)

    err = float(total_err) / total_epoch
    loss = float(total_loss) / (i + 1)

    return err, loss
    

def evaluate_auto_encoder(net, loader, criterion):
    total_loss = 0.0
    total_epoch = 0

    net.eval()
    
    with torch.no_grad():

        for i, (inputs, labels) in enumerate(loader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
                
            # Forward pass
            outputs = net(inputs)
            
            # Calculate the statistics
            total_loss += criterion(outputs, inputs).item()
            total_epoch += len(labels)

    loss = float(total_loss) / (i + 1)
    
    return loss


def train_net(net, model_name, dataset_path = FONT_DATASET_PATH, batch_size=128, learning_rate=0.01, num_epochs=30, patience=None):

    torch.cuda.empty_cache()

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)


    # Create the directory to store model if it does not exist
    if not os.path.exists(model_name):
      os.makedirs(model_name)
    
    # Set the seed for reproducibility
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # Load the data
    train_loader, val_loader, test_loader, classes = load_dataset(dataset_path, batch_size)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=learning_rate,weight_decay=1e-3)

    if patience != None:
        num_epochs = 1000
    
    # Set up some numpy arrays to store the loss/error rate
    train_err = np.zeros(num_epochs)
    train_loss = np.zeros(num_epochs)
    val_err = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)
    
    min_validation_loss = 10000000
    stop_counter = 0
    
    print("Starting Training")
    
    # Train the network
    for epoch in range(num_epochs):
        
        total_train_loss = 0.0
        total_train_err = 0.0
        total_epoch = 0
        
        for i, (inputs, labels) in enumerate(train_loader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            net.train()
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass, backward pass, and optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            net.eval()
            
            # Calculate the statistics
            total_train_err += total_error(outputs, labels)
            total_train_loss += loss.item()
            total_epoch += len(labels)
        
        # Store the statistics in the numpy arrays
        train_err[epoch] = float(total_train_err) / total_epoch
        train_loss[epoch] = float(total_train_loss) / (i+1)
        val_err[epoch], val_loss[epoch] = evaluate(net, val_loader, criterion)
        
        # Print the statistics
        print(f"Epoch {epoch + 1}: Train err: {train_err[epoch]}, Train loss: {train_loss[epoch]} | Validation err: {val_err[epoch]}, Validation loss: {val_loss[epoch]}")
        
        # Write the loss/err into CSV file for plotting later
        np.savetxt(f"{model_name}/train_err.csv", train_err)
        np.savetxt(f"{model_name}/train_loss.csv", train_loss)
        np.savetxt(f"{model_name}/val_err.csv", val_err)
        np.savetxt(f"{model_name}/val_loss.csv", val_loss)
        
        # Save the best model
        if val_err[epoch] <= min_validation_loss:
            min_validation_loss = val_err[epoch]
            torch.save(net.state_dict(), f"{model_name}/best_model")
            stop_counter = 0
        
        if patience != None and stop_counter >= patience:
            break
        
        stop_counter += 1

    print('Finished Training')

    

def test_net(net, model_path, dataset_path = FONT_DATASET_PATH):

    
    # Load the data
    train_loader, val_loader, test_loader, classes = load_dataset(dataset_path, batch_size=128)
    
    # Load the model
    net.load_state_dict(torch.load(model_path))
    
    # Evaluate the model on the test set
    criterion = nn.CrossEntropyLoss()
    test_err, test_loss = evaluate(net, test_loader, criterion)
    
    print(f"Test error: {test_err}, Test loss: {test_loss}")
    
    
def train_auto_encoder(net, model_name, dataset_path = FONT_DATASET_PATH, batch_size=128, learning_rate=0.01, num_epochs=30, patience=None, device):

    torch.cuda.empty_cache()

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)


    
    # Create the directory to store model if it does not exist
    if not os.path.exists(model_name):
      os.makedirs(model_name)
    
    # Set the seed for reproducibility
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # Load the data
    train_loader, val_loader, test_loader, classes = load_dataset(dataset_path, batch_size)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    if patience != None:
        num_epochs = 1000

    # Set up some numpy arrays to store the loss/error rate
    train_loss = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)
    
    min_validation_loss= 10000000
    stop_counter = 0
    
    print("Starting Training")
    
    # Train the network
    for epoch in range(num_epochs):
        
        total_train_loss = 0.0
        total_epoch = 0
        
        for i, (inputs, labels) in enumerate(train_loader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            net.train()
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass, backward pass, and optimize
            outputs = net(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            net.eval()
            
            # Calculate the statistics
            total_train_loss += loss.item()
            total_epoch += len(labels)
        
        # Store the statistics in the numpy arrays
        train_loss[epoch] = float(total_train_loss) / (i+1)
        val_loss[epoch] = evaluate_auto_encoder(net, val_loader, criterion)

        # Save best model
        if val_loss[epoch] <= min_validation_loss:
            min_validation_loss = val_loss[epoch]
            torch.save(net.state_dict(), f"{model_name}/best_model")
            stop_counter = 0
        
        # Write the loss/err into CSV file for plotting later
        np.savetxt(f"{model_name}/train_loss.csv", train_loss)
        np.savetxt(f"{model_name}/val_loss.csv", val_loss)
        
        # Print the statistics
        print(f"Epoch {epoch + 1}: Train loss: {train_loss[epoch]} | Validation loss: {val_loss[epoch]}")
        
        if patience != None and stop_counter >= patience:
            break
        stop_counter += 1
    
    print('Finished Training')

    