from datetime import datetime, timezone
from pathlib import Path
import logging
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

        # Define model training specifications
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.training_config['learning_rate'],
                                    weight_decay=self.training_config["L2_lambda"])

        # Move the model to appropiate device
        self.device = self.training_config['device']
        self.model.to(self.device)
        logging.info("Model build successful")


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

        # NOTE: Save the class to idx variable as it will be necessary
        # when saving checkpoints and predicting class
        self.model_idx_to_class = {value: key for key, value in train_image_datasets.class_to_idx.items()}

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

    def train_model(self):
        # Train the model
        train_losses, valid_losses = [], []

        for epoch in range(1, self.training_config['epochs'] + 1):
            self.model.train() # Enable dropout layer
            epoch_train_loss = 0
            
            for images, labels in self.train_dataloader:
                # Move the inputs and labels to GPU if available
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Reset the gradiant values
                self.optimizer.zero_grad()

                # execute feed-forward
                log_ps = self.model(images)
                loss = self.criterion(log_ps, labels)
                epoch_train_loss += loss.item()
                
                # back-propogation step
                loss.backward()
                self.optimizer.step()
                
            self.model.eval() # Disable the dropout layer
            epoch_valid_loss = 0
            epoch_valid_correct = 0
            
            # Turn of gradient computation for effeciency
            with torch.no_grad():
                for images, labels in self.valid_dataloader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    log_ps = self.model(images)
                    loss = self.criterion(log_ps, labels)
                    epoch_valid_loss += loss.item()

                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    epoch_valid_correct += equals.sum().item()

                # Get average epoch train and validation losses
                avg_epoch_train_loss = epoch_train_loss / len(self.train_dataloader.dataset)
                avg_epoch_valid_loss = epoch_valid_loss / len(self.valid_dataloader.dataset)

                # Append the train and validation losses
                train_losses.append(avg_epoch_train_loss)
                valid_losses.append(avg_epoch_valid_loss)

                # Print metrics for each epoch
                print(f"Epoch: {epoch}/{self.training_config['epochs']}.. ",
                    f"Train Loss: {avg_epoch_train_loss:.3f}.. ",
                    f"Valid Loss: {avg_epoch_valid_loss:.3f}.. ",
                    f"Valid Acc: {epoch_valid_correct*100 / len(self.valid_dataloader.dataset):.1f}%")
        
        logging.info("Model training successful")
        return True
    
    def test_accuracy(self):
        self.model.eval() # Disable the dropout layer
        test_accuracy = 0
        for images, labels in self.test_dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            ps = torch.exp(self.model(images))

            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            test_accuracy += equals.sum().item()

        test_accuracy /= len(self.test_dataloader.dataset)

        print(f"Model accuracy on test dataset: {test_accuracy * 100:.2f}%")

    def save_checkpoint(self):
        self.model.idx_to_class = self.model_idx_to_class

        checkpoint = {
            "training_cofig": self.training_config,
            "idx_to_class": self.model.idx_to_class,
            "pre_trained_model": Trainer.model_choices[self.training_config['arch']]['model_fn'],
            "classifier": self.model.classifier,
            "model_state_dict": self.model.state_dict(),
            "optim_state_dict": self.optimizer.state_dict()
        }

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        save_path = Path(self.training_config["save_dir"]) / f"checkpoint_{timestamp}.pth"

        # Ensure directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the checkpoint
        torch.save(checkpoint, save_path)
