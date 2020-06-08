# PROGRAMMER: Stephen Roth
# DATE CREATED: 6/1/2020  
# FILE PURPOSE: Get the command line arguments for both the training and predicting functions
#
##
# Imports python modules
import argparse

def get_train_args():
    """
    Retrieves and parses the command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to created and defined these 3 command line arguments. If 
    the user fails to provide some or all of the 3 arguments, then the default 
    values are used for the missing arguments. 
    Command Line Arguments:
      1. Image Folder as --dir with default value 'flowers'
      2. Model Architecture as --arch with default value 'vgg'
      3. Text File with flower names as --flowerfile
      4. And more...
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
   # Creates parse 
    parser = argparse.ArgumentParser()

    # Creates command line arguments args.dir for path to images files,
    # args.arch which CNN model to use for classification, args.labels path to
    # text file with names of flowers and more.
    
    parser.add_argument('--data_dir', type=str, default='flowers/', help='path to folder of flower images')
    parser.add_argument('--save_dir', type=str, default='ImageClassifier/', help='path to save model classifier after training')
    parser.add_argument('--arch', default = 'vgg16' )
    parser.add_argument('--hidden_units', type=int, default = 500 )
    parser.add_argument('--learning_rate', type=float, default = 0.001 )
    parser.add_argument('--epochs', type=int, default = 1 )
    parser.add_argument('--gpu', type=bool, default = True ) 

    return parser.parse_args()

def get_predict_args():
    """
    Retrieves and parses the 3 command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to created and defined these 3 command line arguments. If 
    the user fails to provide some or all of the 3 arguments, then the default 
    values are used for the missing arguments. 
    """
   # Creates parse 
    parser = argparse.ArgumentParser()

    # Creates command line arguments args.dir for path to images files,
    # args.arch which CNN model to use for classification, args.labels path to
    # text file with names of flowers and more
    
    parser.add_argument('image_file', type=str)
    parser.add_argument('checkpoint_file', type=str, default='checkpoint')
    parser.add_argument('--data_dir', type=str, default='flowers/', help='path to folder of flower images')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--category_names', default = 'cat_to_name.json' )
    parser.add_argument('--gpu', type=bool, default = True ) 
 
    return parser.parse_args()