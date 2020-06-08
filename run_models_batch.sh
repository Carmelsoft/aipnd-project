# PROGRAMMER: Stephen Roth
# DATE CREATED: 6/1/2020  
# FILE PURPOSE: Sample command line commands and arguments
#

python train.py --arch alexnet
python predict.py flowers/test/65/image_03243.jpg checkpoint --top_k 5

python train.py --arch alexnet --hidden_units 500 --learning_rate 0.001 --epochs 5 --gpu True
python predict.py flowers/test/65/image_03243.jpg checkpoint --top_k 5  --gpu True
