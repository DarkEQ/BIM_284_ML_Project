import torch

import matplotlib.pyplot as plt
import numpy as np

from tqdm.auto import tqdm

def validation(model, test_loader, loss_fn, device):

    loss_per_bacth = []
    
    # Put model into evaluation Mode
    model.eval()

    with torch.no_grad():
        with tqdm(test_loader, unit="batch", total=len(test_loader)) as tbatches:
            for test_data in tbatches:
                images, labels = test_data[0].to(device), test_data[1].to(device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                loss_per_bacth.append(loss.item())

    return loss_per_bacth

def test_accuracy(model, data_loader, device):
    model.to(device)
    correct = 0
    total = 0
    accuracy = 0

    # Put model into evaluation Mode
    model.eval()

    with torch.no_grad():
        with tqdm(data_loader, unit="batch", total=len(data_loader)) as tbatches:
            for test_data in tbatches:
                images, labels = test_data[0].cuda(), test_data[1].cuda()
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = (100 * correct / total)

    return accuracy

def train(model, loss_fn, optimizer, train_loader, test_loader, num_epochs, device, history):

    if device is not None:
        model.to(device)
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model.to(device)
    # Iterate through all Epochs
    for epoch in range(num_epochs):
        # Init train loss
        train_loss = []

        # Set model to train at the start of every epoch
        model.train()
        
        with tqdm(enumerate(train_loader), unit="batch", total=len(train_loader)) as tepoch:
            # Iterate through training dataset
            for i, data in tepoch:

                # Flatten images and load images/labels onto GPU
                images, labels = data[0].to(device), data[1].to(device)
                # Zero collected gradients at each step
                optimizer.zero_grad()
                # Forward Propagate
                outputs = model(images)
                # Calculate Loss
                loss = loss_fn(outputs, labels)
                # Store Loss
                train_loss.append(loss.item())
                # Back propagate
                loss.backward()
                # Update weights
                optimizer.step()

        # Print statistics on every epoch
        # Calculate validation loss for this epoch
        validation_loss = validation(model=model, test_loader=test_loader, loss_fn=loss_fn, device=device)
        # Calculate mean of validation loss
        mean_val_loss = np.mean(validation_loss)
        # Calculate mean of training loss
        mean_train_loss = np.mean(train_loss)
        # Store Validation Loss per epoch
        history['validation_loss'].append(mean_val_loss)
        # Store Training Loss per epoch
        history['training_loss'].append(mean_train_loss)
        
        # Validation Accuracy
        history['validation_accuracy'].append(test_accuracy(model=model, data_loader=test_loader, device=device))
        # Training Accuracy
        history['training_accuracy'].append(test_accuracy(model=model, data_loader=train_loader, device=device))

        print('Epoch [%d/%d] End, Training Loss: %.4f., Validation Loss: %.4f., Training Accuracy: %.2f %%, Validation Accuracy: %.2f %%'
                            %(epoch+1, num_epochs, 
                            loss.item(), mean_val_loss,
                            history['training_accuracy'][epoch],
                            history['validation_accuracy'][epoch]))
        print('--------------------------------------')
    return history

def plot_learning_curve(history):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(history['training_loss'], label='training loss')
    plt.plot(history['validation_loss'], label='validation loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    if history['momentum'] != None:
        plt.suptitle('Model: %s, Batch Size: %i, LR: %.5f, Mo: %.4f, Optimizer: %s' %(history['model'], history['batch_size'], history['learning_rate'], history['momentum'], history['optimizer']))
    else:
        plt.suptitle('Model: %s, Batch Size: %i, LR: %.5f, Mo: %.4f, Optimizer: %s' %(history['model'], history['batch_size'], history['learning_rate'], history['optimizer']))
    plt.title('Loss vs. Iteration')

    plt.subplot(2, 1, 2)
    plt.plot(history['training_accuracy'], label='training accuracy')
    plt.plot(history['validation_accuracy'], label='validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(axis='y')
    # plt.suptitle('Model: %s, Batch Size: %i, LR: %.4f, Optimizer: %s' %(history['model'], history['batch_size'], history['learning_rate'], history['optimizer']))
    plt.title('Accuracy vs. Epoch')

    plt.tight_layout()
    plt.show()

def predict(model, test_loader, device):
    model.to(device)

    # predictions = torch.zeros(0,dtype=torch.long, device='cpu')
    # test_labels = torch.zeros(0,dtype=torch.long, device='cpu')
    predictions = []
    test_labels = []

    # Put model into evaluation Mode
    model.eval()

    with torch.no_grad():
        with tqdm(test_loader, unit="batch", total=len(test_loader)) as tbatches:
            for test_data in tbatches:
                images, labels = test_data[0].cuda(), test_data[1].cuda()
                outputs = model(images)
                outputs = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
                predictions.extend(outputs)

                labels = labels.data.detach().cpu().numpy()
                test_labels.extend(labels)

    return predictions, test_labels