import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
import numpy as np
import json
import os

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import (
    EfficientNet_V2_S_Weights,
    VGG13_BN_Weights
)

class Trainer:
    model_choices = {
        'efficientnetv2': {
            'model_fn': models.efficientnet_v2_s,
            'weights_enum': EfficientNet_V2_S_Weights
        },
        'vgg13': {
            'model_fn': models.vgg13_bn,
            'weights_enum': VGG13_BN_Weights
        }
    }

    def __init__(self, training_config):
        self.training_config = training_config

        # Prepare training, validation, and testing datasets
        dataloader = self.prepare_dataloader()
        self.train_dataloader = dataloader['train_dataloader']
        self.valid_dataloader = dataloader['valid_dataloader']
        self.test_dataloader = dataloader['test_dataloader']

        # Define the model architecture and add additional hidden layers
        self.model = self.build_model()
        print(self.model)


    def prepare_dataloader(self):
        # Ensure that train, valid, and test folders exists in the data directory
        train_dir = os.path.join(self.training_config['data_dir'], 'train')
        valid_dir = os.path.join(self.training_config['data_dir'], 'train')
        test_dir = os.path.join(self.training_config['data_dir'], 'train')

        assert os.path.isdir(train_dir), "Missing 'train' directory in data folder"
        assert os.path.isdir(valid_dir), "Missing 'valid' directory in data folder"
        assert os.path.isdir(test_dir), "Missing 'test' directory in data folder"


        # Define your transforms for the training, validation, and testing sets
        train_transforms = transforms.Compose([
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.17), # Vary image pallete
            transforms.RandomRotation(30), # Rotate the image with a random angle bw [-30, 30]
            transforms.RandomResizedCrop(224), # Random crop and then resize
            transforms.RandomHorizontalFlip(), # Flip the images horizontally sometimes
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])

        valid_transforms = transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])

        test_transforms = transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])


        # Load the datasets with ImageFolder
        train_image_datasets = datasets.ImageFolder(train_dir, train_transforms)
        valid_image_datasets = datasets.ImageFolder(valid_dir, valid_transforms)
        test_image_datasets = datasets.ImageFolder(test_dir, test_transforms)

        # Using the image datasets and the trainforms, define the dataloaders
        batch_size = self.training_config['batch_size']

        dataloader = {
            'train_dataloader': DataLoader(train_image_datasets, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True),
            'valid_dataloader': DataLoader(valid_image_datasets, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True),
            'test_dataloader': DataLoader(test_image_datasets, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)
        }

        return dataloader

    def build_model(self):
        arch = self.training_config['arch']

        if arch not in Trainer.model_choices:
            raise Exception("Incompatible model architecture chosen for buliding model")
        
        # Retrive the arch and weights of pretrained model
        model_fn = Trainer.model_choices[arch]['model_fn']
        weights_enum = Trainer.model_choices[arch]['weights_enum']
        model = model_fn(weights=weights_enum.DEFAULT)

        # Freeze the non classifier layers weights
        for param in model.parameters():
            param.requires_grad = False

        # Retrieve the in_features for the first classifier layer
        in_features = model.classifier[1].in_features

        # Define the classifier layers
        classifier = nn.Sequential()
    
        for layer_units in self.training_config['hidden_units']:
            classifier.append(nn.Linear(in_features=in_features, out_features=layer_units, bias=True))
            classifier.append(nn.ReLU())
            classifier.append(nn.Dropout(p=self.training_config['dropout_prob']))
            
            # For the next layer, in_features becomes the out_features of previous layer
            in_features = layer_units

        # Add the last output layer with 102 classification output nodes
        classifier.append(nn.Linear(in_features=in_features, out_features=102, bias=True))
        classifier.append(nn.LogSoftmax(dim=1))

        model.classifier = classifier

        return model 
        