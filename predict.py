# PROGRAMMER: Stephen Roth
# DATE CREATED: 6/1/2020  
# FILE PURPOSE: Predict the image using a pretrained model
#
#imports python modules
from time import time, sleep
import sys

#imports functions created for this program
from get_arguments import get_train_args
from get_arguments import get_predict_args

import helper
from workspace_utils import active_session
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import models
import matplotlib.ticker as tick
import numpy as np

#from os import listdir
import copy
import pandas as pd

# Main program function defined below
def main():
    #measures total program runtime by collecting start time
    start_time = time()
    #get the command line arguments
    in_arg = get_predict_args()
        
    #load it from the trained file save in the train.py class
    model = loadCheckpoint(in_arg.checkpoint_file+".pth")
    
    #predict and check it
    #calculate the probabilities by passing in the image filename, model, and other parameters
    predictions = predict(in_arg.image_file, model, in_arg.gpu, in_arg.top_k, in_arg.category_names)
    #print it out
    print(predictions)
    #what's the time?
    end_time = time()
    
    #computes overall runtime in seconds & prints it in hh:mm:ss format
    tot_time = end_time - start_time #calculate difference between end time and start time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )
   
    
#load it back into the model from the checkpoint file
def loadCheckpoint(filepath):
    
    checkpoint = torch.load(filepath)
    inputs = 25088
    #recreate the same pretrained network
    
    #depending upon the architecture, it affects some of the
    #parameters and settings    
    if (checkpoint['arch'] == 'alexnet'):
        model = models.alexnet(pretrained=True)
        for param in model.parameters():
            param.requires_grad=False
        inputs = 9216
    elif (checkpoint['arch'] == 'resnet18'):
        model = models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad=True
        inputs = 512
    else:
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad=False
        inputs = 25088
          
    model.class_to_idx = checkpoint['class_to_idx']
    #recreate the same classifier
    classifier = nn.Sequential(nn.Dropout(0.15),
                           nn.Linear(inputs, checkpoint['hidden_layers']),
                           nn.ReLU(),
                           nn.Linear(checkpoint['hidden_layers'], 120),
                           nn.ReLU(),
                           nn.Linear(120, 102),
                           nn.LogSoftmax(dim=1))  
    
    model.classifier = classifier
    #get the state dicts from the saved file
    model.load_state_dict(checkpoint['state_dict'])
    
    return model
  
#implement the code to predict the class from an image file

#portions of the following code are developed by looking at and getting advice from the GitHub repo of 
# https://github.com/paulstancliffe/Udacity-Image-Classifier and also the Udacity "intro-to-pytorch" exercises    
def predict(image_path, model, gpu, topk, catNameFile):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    #is cuda available?
    torch.cuda.is_available()
    #if so, take advantage of the GPU power
    if torch.cuda.is_available() and (gpu == True):
        model.cuda()
    
    #get the list of categories
    cat_to_name = helper.labelMapping(catNameFile)
    
    #implement the code to predict the class from an image file
    #process the image and turn it into a numpy array
    np_image = helper.process_image(image_path)
    
    #changing from numpy to pytorch tensor
    ptensor = torch.tensor(np_image).float().unsqueeze(0)

    #run model to make predictions
    model.eval()
    with torch.no_grad():
        logsoftmax = model.forward(ptensor.cuda())
    predictions = torch.exp(logsoftmax)
    
    #identify top predictions and top labels
    top_predictions, top_labels = predictions.topk(topk)
    
    #convert top predictions into a numpy list (get it back to local memory)
    top_predictions = top_predictions.detach().cpu().clone().numpy().tolist()
   
    #change top labels into a list
    top_labels = top_labels.tolist()
    
    #create a pandas dataframe joining class to flower names
    table = pd.DataFrame({'class':pd.Series(model.class_to_idx),'flower name':pd.Series(cat_to_name)})
    table = table.set_index('class')
    
    #limit the dataframe to top labels and add their predictions
    table = table.iloc[top_labels[0]]
    table['prediction'] = top_predictions[0]
    
    return table
    
    
# Call to main function to run the program
if __name__ == "__main__":
    main()