import warnings


import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
import numpy as np
import json

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models



import torch
from torchvision import datasets, transforms, models

from torchvision.models import (
    EfficientNet_V2_S_Weights,
    VGG13_BN_Weights
)

class Predictor:

    model_weights = {
        'efficientnetv2': EfficientNet_V2_S_Weights,
        'vgg13': VGG13_BN_Weights
    }

    def __init__(self, predict_config):
        # Unload all the config dict to variables for clarity
        self.device = predict_config['device']
        self.checkpoint_pth = predict_config['checkpoint_pth']
        self.top_k = predict_config['top_k']
        self.category_names_file = predict_config['category_names_file']

        # Load the model through the checkpoint
        self.model = self._load_model()
        self.model.to(self.device)

    def _load_model(self):
        # Load the checkpoint file
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            checkpoint = torch.load(self.checkpoint_pth)

        # Load the pre-trained arch with the weights
        arch_name = checkpoint['training_cofig']['arch']
        model = checkpoint['pre_trained_model'](weights=Predictor.model_weights[arch_name].DEFAULT)

        for param in model.parameters():
            param.requires_grad = False
            
        model.classifier = checkpoint['classifier']
        model.load_state_dict(checkpoint['model_state_dict'])
        model.idx_to_class = checkpoint['idx_to_class']

        return model

    def _process_image(self, image):
        # Resizing image to (256x256)
        image = image.resize((256, 256))

        # Center crop image of (224x224)
        width, height = image.size
        left = (width - 224)/2
        top = (height - 224)/2
        right = (width + 224)/2
        bottom = (height + 224)/2
        image = image.crop((left, top, right, bottom))

        # Convert PIL image to numpy array
        np_image = np.array(image) / 255.0

        # Normalize the color channels
        means = [0.485, 0.456, 0.406]
        stds = [0.229, 0.224, 0.225]
        np_image = (np_image - means) / stds
        
        # Reorder dimensions to have color channels first
        np_image = np_image.transpose(2, 0, 1)
        
        return torch.FloatTensor(np_image)

    def _imshow(self, image, ax=None, title=None):
        if ax is None:
            fig, ax = plt.subplots()
        
        # PyTorch tensors assume the color channel is the first dimension
        # but matplotlib assumes is the third dimension
        image = image.numpy().transpose((1, 2, 0))
        
        # Undo preprocessing
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        
        # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
        image = np.clip(image, 0, 1)
        
        ax.imshow(image)
        
        return ax

    def _get_class_probs(self, img_path):
        self.model.eval() # Just for safety (disable dropout layer)
        with Image.open(img_path) as image:
            processed_image = self._process_image(image).to(self.device)
            processed_image = processed_image.reshape(1, *processed_image.shape)
            
            ps = torch.exp(self.model(processed_image))
            top_p, top_idx = ps.topk(k=self.top_k, dim=1)
            top_p = top_p.view(-1).tolist()
            top_class = [self.model.idx_to_class[idx.item()] for idx in top_idx.view(-1)]

            return top_p, top_class

    def predict_class(self, img_path):
        with Image.open(img_path) as image:
            processed_image = self._process_image(image)
            fig, axs = plt.subplots(1, 2, figsize=(8, 4))
            
            # Call imshow with subplot axis
            self._imshow(processed_image, ax=axs[0], title="Pink Primrose")
            
            # Show probabilities of top 5 classes
            probs, classes = self._get_class_probs(img_path)

            # Open json file to convert class idx to names
            with open(self.category_names_file, 'r') as f:
                cat_to_name = json.load(f)

            class_names = [cat_to_name[cat] for cat in classes]
            axs[1].barh(y=class_names[::-1], width=probs[::-1])
            axs[1].set_title("Top 5 Predicted Classes")
            
            plt.tight_layout()

            # Also print the class names and its probability
            for class_name, class_prob in zip(class_names, probs):
                print(f"Predicted Class: {class_name}; Probability: {class_prob}")

            plt.show()