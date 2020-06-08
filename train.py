# PROGRAMMER: Stephen Roth
# DATE CREATED: 6/1/2020                                  
# FILE PURPOSE: Train the model
#
# Imports python modules and functions
from time import time, sleep
from get_arguments import get_train_args
from get_arguments import get_predict_args

import helper
from workspace_utils import active_session

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import models

# Main program function defined below
def main():
    
    #measures total program runtime by collecting start time
    start_time = time()
    
    #import the command line arguments
    in_arg = get_train_args()

    #load in the data and loaders from the helper function
    train_data, valid_data, test_data, trainloader, validloader, testloader = helper.loadData(in_arg.data_dir)
    
    #based on the architecture type, assign it to the model
    if (in_arg.arch == 'alexnet'):
        model = models.alexnet(pretrained=True)
    elif (in_arg.arch == 'resnet18'):
        model = models.resnet18(pretrained=True)
    else: # (in_arg.arch == 'vgg16'):
        model = models.vgg16(pretrained=True)
    
    #define the classifier, optimizer, criterion and store in variables 
    #to be passed into the training function
    model, optimizer, criterion = defineClassifier(model, in_arg.hidden_units, in_arg.learning_rate, in_arg.arch)
        
    #is cuda available?
    torch.cuda.is_available()
    #if so, take advantage of the GPU power
    if torch.cuda.is_available() and (in_arg.gpu == True):
        device = 'cuda'
    else:
        device = 'cpu'
    #device = torch.device('cuda' if ((torch.cuda.is_available() and (in_arg.gpu == True)) else 'cpu')
    #move to GPU
    model.to(device)                   
    #train the model
    model = trainIt(model, trainloader, validloader, optimizer, criterion, device, in_arg.epochs)
    #test the model against test data
    testIt(model, testloader, criterion, device)
    #save the model to a file for future use
    saveCheckpoint(model, train_data, in_arg.arch, in_arg.hidden_units)

    #how much time did it take?
    end_time = time()
    
    #computes overall runtime in seconds & prints it in hh:mm:ss format
    tot_time = end_time - start_time #calculate difference between end time and start time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )
   
#define the classifier, criterion, optimizer
def defineClassifier(model, hidden_units, learning_rate, arch):
    
    #freeze the parameters (for vgg and alexnet)
    if (arch == 'vgg16' or arch == 'alexnet'):
        for param in model.parameters():
            param.requires_grad = False
    #for resnet, do not freeze
    if (arch == 'resnet18'):
        #freeze parameters
        for param in model.parameters():
            param.requires_grad = True
    
    #get the # of inputs based upon the architecture selected
    inputs = 25088
    if (arch == 'vgg16'): 
        inputs = 25088
    elif (arch == 'alexnet'): 
        inputs = 9216
    elif (arch == 'resnet18'): 
        inputs = 512
            
    #define the classifier with an input layer, hidden layers, output layer and a dropout of 15%
    classifier = nn.Sequential(nn.Dropout(0.15),
                           nn.Linear(inputs, hidden_units),
                           nn.ReLU(),
                           nn.Linear(hidden_units, 120),
                           nn.ReLU(),
                           nn.Linear(120, 102),
                           nn.LogSoftmax(dim=1))
    
    model.classifier = classifier
    #define the criterion and optimizer (loss and gradiant descent)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
    #return the model, optimizer, and criterion that is used for training
    return model, optimizer, criterion

#train the model
#portions of the following code are derived from the Udacity "intro-to-pytorch" exercises
def trainIt(model, trainloader, validloader, optimizer, criterion, device, epochs):
    
    steps = 0
    running_loss = 0
    print_every = 10
    
    #make sure it does not time-out
    with active_session():
        #how many loops?
        for epoch in range(epochs):
            for inputs, labels in trainloader:
              steps += 1
              #move input and label tensors to the default device
              inputs, labels = inputs.to(device), labels.to(device)
        
              optimizer.zero_grad()
              #forward through the network
              logps = model.forward(inputs)
              loss = criterion(logps, labels)
              #now, backward
              loss.backward()
              #step it up
              optimizer.step()

              running_loss += loss.item()
              #for every 5 or 10 loops, let's compare it to validation data to make sure we are not overfitting too much
              if steps % print_every == 0:
                    valid_loss = 0
                    accuracy = 0
                    model.eval()
                    with torch.no_grad():
                       for inputs2, labels2 in validloader:
                            inputs2, labels2 = inputs2.to(device), labels2.to(device)
                            logps = model.forward(inputs2)
                            batch_loss = criterion(logps, labels2)
                    
                            valid_loss += batch_loss.item()
                    
                            #calculate accuracy
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels2.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                    print(f"Epoch {epoch+1}/{epochs}.. "
                        f"Train loss: {running_loss/print_every:.3f}.. "
                        f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                        f"Validation accuracy: {accuracy/len(validloader):.3f}")
                    running_loss = 0   
                    model.train()
                    
        return model
    
#validate the test set and see how close we are (hopefully > 70%)
#portions of the following code are derived from the Udacity "intro-to-pytorch" exercises
def testIt(model, testloader, criterion, device):

    test_loss = 0
    test_losses = []
    accuracy = 0
        
    #turn off gradients for validation, saves memory and computations
    with torch.no_grad():
        model.eval()
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            log_ps = model(images)
            test_loss += criterion(log_ps, labels)
                
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))
        
    model.train()
        
    test_losses.append(test_loss/len(testloader))

    print("Test Loss: {:.3f}.. ".format(test_losses[-1]),
              "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
    
#save the model to a file
def saveCheckpoint(model, train_data, arch, hidden_units):
    
    #save the checkpoint 
    model.cpu
    model.class_to_idx = train_data.class_to_idx
    
    #also store the architecture and hidden_layer count for use in the predict.py functions
    checkpoint = {              
             'state_dict': model.state_dict(),
             'class_to_idx': model.class_to_idx,
             'arch': arch,
             'hidden_layers': hidden_units 
             }

    torch.save(checkpoint, 'checkpoint.pth')    
         
        
# Call to main function to run the program
if __name__ == "__main__":
    main()