# train

## args

The file contains all the parameters for running commands, with brief descriptions for each parameter included. It should be used in conjunction with the code location file in the handover document. The training hyperparameters are as follows:
nproc_per_node:  the number of graphics cards
use_env: the file of main function 
model: the model time
has_bootstrap: whether to multiply the training data volume by this coefficient, which is defaulted to 1.
extend_num: the parameters of bounding box
the parameters for testing： nproc_per_node=1, batch-size=1, ===--eval
pred_state: train, valid or test  
resume: the path of .pth file

## datasets

"datasets.py" is data preprocessing file.

## engine

"engine.py" contains three main functions: train_one_epoch，evaluate，evaluate_eval.
"main.py" is the pipeline file.

# infer

## main

"main.py" is the pipeline file for model inferring.

## predictor

"predictor.py" is the data preprocessing function and model framework for model inferring.

# requirements

The "requirements.txt" is the model training and infering environment with python and PyTorch.

