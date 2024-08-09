import torch
import matplotlib.pyplot as plt
import numpy as np
from train import load_dataset, FONT_DATASET_PATH

def plot_training_curve(path):
    train_err = np.loadtxt("{}/train_err.csv".format(path))
    val_err = np.loadtxt("{}/val_err.csv".format(path))
    train_loss = np.loadtxt("{}/train_loss.csv".format(path))
    val_loss = np.loadtxt("{}/val_loss.csv".format(path))
    plt.title("Train vs Validation Error")
    num_epochs = len(train_err)
    plt.plot(range(1,num_epochs+1), train_err, label="Train")
    plt.plot(range(1,num_epochs+1), val_err, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend(loc='best')
    plt.show()
    plt.title("Train vs Validation Loss")
    plt.plot(range(1,num_epochs+1), train_loss, label="Train")
    plt.plot(range(1,num_epochs+1), val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()


def visualize_output(num_images, model_path, model_class, dataset_path = FONT_DATASET_PATH):

    # Load the data
    train_loader, val_loader, test_loader, classes = load_dataset(dataset_path, batch_size=num_images)
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    # Get ground truth labels
    ground_truth = [classes[np.argmax(labels[j], axis=0)] for j in range(num_images)]

    # Get model predictions
    net = model_class()
    net.load_state_dict(torch.load(model_path))
    outputs = net(images)
    outputs = np.argmax(outputs.detach().numpy(), axis=1)    
    predicted = [classes[outputs[j]] for j in range(num_images)]
    
    fig, axs = plt.subplots(1, num_images, figsize=(20, 20))

    # Print Images
    for i in range(num_images):
        img = images[i]
        img = img / 2 + 0.5
        npimg = img.numpy()

        axs[i].imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
        axs[i].set_yticklabels([])
        axs[i].set_xticklabels([])
        axs[i].set_xticks([])
        axs[i].set_yticks([])

        axs[i].set_title(f"Predicted: {predicted[i]} \n Ground Truth: {ground_truth[i]}")
    plt.show()
    
def visualize_autoencoder_output(num_images, model_path, model_class, dataset_path = FONT_DATASET_PATH):
    # Load the data
    train_loader, val_loader, test_loader, classes = load_dataset(dataset_path, batch_size=num_images)
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    # Get model predictions
    net = model_class()
    net.load_state_dict(torch.load(model_path))
    outputs=net(images)
    
    # Reformat
    outputs = outputs.cpu().detach().numpy()
    images = images.cpu().detach().numpy()
    
    
    fig, axs = plt.subplots(2, num_images, figsize=(20, 20))

    for i in range(num_images):
        
        for j in range(2):
            img = images[i] if j == 0 else outputs[i]
            img = img / 2 + 0.5

            axs[j,i].imshow(np.transpose(img, (1, 2, 0)), cmap='gray')
            axs[j,i].set_yticklabels([])
            axs[j,i].set_xticklabels([])
            axs[j,i].set_xticks([])
            axs[j,i].set_yticks([])
        

def generate_confusion_matrix(model_class, model_path, dataset_path=FONT_DATASET_PATH):

    net = model_class()

     # Load the data
    train_loader, val_loader, test_loader, classes = load_dataset(dataset_path, batch_size=128)
    
    # Load the model
    net.load_state_dict(torch.load(model_path))
    
    confusion_matrix = np.zeros((len(classes), len(classes)))

    with torch.no_grad():

        for i, (inputs, labels) in enumerate(test_loader, 0):
                
            # Forward pass
            outputs = net(inputs)
            
            # Find the indices of the max values
            _, indices = torch.max(outputs, dim=1, keepdim=True)
            
            for j in range(len(labels)):
                confusion_matrix[np.argmax(labels[j], axis=0), indices[j]] += 1
        
    plt.imshow(confusion_matrix, interpolation='nearest')
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
                
    return confusion_matrix