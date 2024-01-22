import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict

import matplotlib.pyplot as plt


def transformations(data_dir):
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(250),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(250),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder

    train_dataset = datasets.ImageFolder(train_dir, transform = train_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform = test_transforms)
    validation_dataset = datasets.ImageFolder(valid_dir, transform = validation_transforms)
    image_datasets = {'train_dataset': train_dataset, 'test_dataset': test_dataset, 'validation_dataset': validation_dataset}

    # TODO: Using the image datasets and the trainforms, define the dataloaders

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64)
    validationloader = torch.utils.data.DataLoader(validation_dataset, batch_size=64)
    dataloaders = {'trainloader': trainloader, 'testloader': testloader, 'validationloader': validationloader}
    
    return image_datasets, dataloaders

def create_model(hidden_units, arch):
    
    model = getattr(models, arch)(pretrained=True)
    for parameter in model.parameters():
        parameter.requires_grad = False
    classifier = nn.Sequential(OrderedDict([
    ('fc1' , nn.Linear(25088, hidden_units)),
    ('relu' , nn.ReLU()),
    ('fc2' , nn.Linear(hidden_units, 102)),
    ('output' , nn.LogSoftmax(dim=1))]))
    
    model.classifier = classifier
    return model

def train_model(model, device, n_epochs, learning_rate, dataloaders):
    model.to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    epochs = n_epochs

    train_losses, validation_losses = [],[]
    for epoch in range(epochs):
        training_loss = 0
        for images, labels in dataloaders['trainloader']:
            images, labels = images.to(device), labels.to(device)
            logps = model(images)
            loss = criterion(logps, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
        else:
            validation_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for images, labels in dataloaders['validationloader']:
                    images, labels = images.to(device), labels.to(device)
                    logps = model(images)
                    loss = criterion(logps, labels)
                    validation_loss += loss.item()
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {training_loss/len(dataloaders['trainloader']):.3f}.. "
                  f"Validation loss: {validation_loss/len(dataloaders['validationloader']):.3f}.. "
                  f"Validation accuracy: {accuracy/len(dataloaders['validationloader']):.3f}")
            train_losses.append(training_loss/len(dataloaders['trainloader']))
            validation_losses.append(validation_loss/len(dataloaders['validationloader']))
            model.train()
            
    
    test_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for images, labels in dataloaders['testloader']:
            images, labels = images.to(device), labels.to(device)
            logps = model(images)
            loss = criterion(logps, labels)
            test_loss += loss.item()
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    print(f"Test loss on trained model: {test_loss/len(dataloaders['testloader']):.3f}.. "
      f"Test accuracy on trained model: {accuracy/len(dataloaders['testloader']):.3f}")
    
    return model, optimizer

def create_checkpoint(save_dir, model, datasets, optimizer, n_epochs, learning_rate, arch):
    checkpoint = {
              'state_dict': model.state_dict(),
              'class_to_idx' : datasets['train_dataset'].class_to_idx,
              'classifier' : model.classifier,
        
              'epochs' : n_epochs,
              'optimizer' : optimizer.state_dict,
              'learning_rate' : learning_rate,
              'arch' : arch
    }

    torch.save(checkpoint, save_dir+'checkpoint.pth')

def load_checkpoint(filepath, device):
    checkpoint = torch.load(filepath)
    learning_rate = checkpoint['learning_rate']
    arch = checkpoint['arch']

    model = getattr(models, arch)(pretrained=True)
    for parameter in model.parameters():
      parameter.requires_grad = False

    model.classifier = checkpoint['classifier']
    model.load_state_dict = checkpoint['state_dict']
    model.class_to_idx = checkpoint['class_to_idx']
    model.to(device)

    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    optimizer.load_state_dict = checkpoint['optimizer']


    return model
    
    

 
    

    
    
    

