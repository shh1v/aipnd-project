import warnings
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
        self.model = self.load_model()

    def load_model(self):
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

    def predict_class(self):
        pass