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

class Trainer:
    def __init__(self, training_config):
        self.training_config = training_config

    def prepare_dataloader(self, data_dir):
        # Ensure that train, valid, and test folders exists in the data directory
        assert(os.path.isdir(os.path.join(data_dir, 'train')), True)
        assert(os.path.isdir(os.path.join(data_dir, 'valid')), True)
        assert(os.path.isdir(os.path.join(data_dir, 'test')), True)

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
        train_image_datasets = datasets.ImageFolder(os.path.join(data_dir, 'train'), train_transforms)
        valid_image_datasets = datasets.ImageFolder(os.path.join(data_dir, 'valid'), valid_transforms)
        test_image_datasets = datasets.ImageFolder(os.path.join(data_dir, 'test'), test_transforms)

        # Using the image datasets and the trainforms, define the dataloaders
        batch_size = self.training_config['batch_size']
        train_dataloader = DataLoader(train_image_datasets, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)
        valid_dataloader = DataLoader(valid_image_datasets, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)
        test_dataloader = DataLoader(test_image_datasets, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)