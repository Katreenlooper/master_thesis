from __future__ import division
import torchvision.transforms as transforms
import json

import torch
torch.cuda.current_device()
torch.autograd.set_detect_anomaly(True)

from utils import *

from train import train
from eval import eval
from baseline import *

MASTER_ROOT_DIR = ""
trainFile = open("TRAIN_PROGRESS.txt", 'w+')

#load hyperparameters
print("HYPERPARAMETERS:")
trainFile.write("HYPERPARAMETERS:" + "\n")
hype = convert_to_dict(os.path.join(MASTER_ROOT_DIR, "csv files", "hyperparameters.csv"))
for key, value in hype.items():
    print(key + " = " + str(value))
    trainFile.write(key + " = " + str(value) + "\n")
print("=============================================================")
trainFile.write("=============================================================" + "\n")

transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
flip_transform = transforms.Compose([transforms.RandomHorizontalFlip(1),
                                     transforms.Resize((224,224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#Load arguments
print("SCRIPT ARGUMENTS:")
trainFile.write("SCRIPT ARGUMENTS:" + "\n")
script_args = convert_to_dict(os.path.join(MASTER_ROOT_DIR, "csv files", "script_args.csv"))
for key, value in script_args.items():
    print(key + " = " + str(value))
    trainFile.write(key + " = " + str(value) + "\n")
print("=============================================================")
trainFile.write("=============================================================" + "\n")

json_file = open(os.path.join(MASTER_ROOT_DIR, "json files", script_args['jsonfileName']))
data = json.load(json_file)

#Load paths of all saved models
paths = convert_to_dict(os.path.join(MASTER_ROOT_DIR, "csv files", "models_dict.csv"))

if script_args['TRAIN']:
    train(device, transform, flip_transform, data, MASTER_ROOT_DIR, trainFile, hype, script_args, paths)
elif script_args['EVAL']:
    eval(MASTER_ROOT_DIR, script_args, hype, paths, transform, data, trainFile)
elif script_args['BASELINE'] in ['AVG', 'KALMAN']:
    use_baseline(script_args, hype, MASTER_ROOT_DIR)
elif script_args['SHOW_PREDICTIONS']:
    show_predictions(script_args, data, MASTER_ROOT_DIR, os.path.join(MASTER_ROOT_DIR, script_args['prediction_dir']))
elif script_args['SHOW_LOSSES']:
    loss_list = open(os.path.join(MASTER_ROOT_DIR, "losses.txt"), "rb")
    display_losses(loss_list, MASTER_ROOT_DIR)
