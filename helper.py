# PROGRAMMER: Stephen Roth
# DATE CREATED: 6/1/2020  
# FILE PURPOSE: Helper functions including data loading,
#               label mapping, and image processing
#
# Imports here
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms

from workspace_utils import active_session
from PIL import Image

#load in image data for training, validation, and testing sets
#portions of the following code are derived from the Udacity "intro-to-pytorch" exercises
def loadData(data_dir):
    
    #data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    #define transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    #load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    #using the image datasets and the trainforms, define the iterative dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=18, shuffle=True)
    
    #return all of the training, validating, and testing data and loaders to be used elsewhere in other functions
    return train_data, valid_data, test_data, trainloader, validloader, testloader
      
#flower label mapping from json file
def labelMapping(catNameFile):
    
    import json
    with open(catNameFile, 'r') as f:
        cat_to_name = json.load(f)   
    return cat_to_name

#process the image
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''    
    im = Image.open(image)
    #create the data transform
    data_transforms = transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    #update the image accordingly
    im = data_transforms(im)  
    return im
