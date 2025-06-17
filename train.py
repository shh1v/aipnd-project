import json
import logging
import torch
import argparse

from helper.train_helper import Trainer

logging.basicConfig(
    level=logging.INFO,  # or DEBUG for more verbosity
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def main(args):

    training_config = {
        "data_dir": args.data_directory,
        "save_dir":args.save_dir,
        "arch": args.arch,
        "device": torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu"),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "hidden_units": args.hidden_units,
        "dropout_prob": args.dropout_prob,
        "L2_lambda": 1e-4
    }

    # Print model configuration for info
    training_config_serializable = training_config.copy()
    training_config_serializable["device"] = str(training_config_serializable["device"])
    logging.info("Training Configuration:\n%s", json.dumps(training_config_serializable, indent=2))

    # Define Trainer object that defines the model upon initialization
    trainer = Trainer(training_config)

    # Train the model
    logging.info("Model training in progress")
    trainer.train_model()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model checkpoints and other specficiations
    parser.add_argument('data_directory', help='Path to data directory for training, validation, and testing (ImageFolder compatible)')
    parser.add_argument('--save_dir', default='checkpoints/', help='Path to directory for saving checkpoints')
    parser.add_argument('--arch', default='efficientnetv2', choices=list(Trainer.model_choices.keys()))
    parser.add_argument('--gpu', action='store_true', help='Enable using GPU if available (default is CPU)')

    # Model hyperparameters
    parser.add_argument('--batch_size', default=32, type=int, help='Set the batch size for model training')
    parser.add_argument('--epochs', default=30, type=int, help='Set the epochs for model training')
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='Set the learning rate for model training')
    parser.add_argument('--hidden_units', nargs='*', default=[640, 320], type=int, help='Set the hidden layer units for model training')
    parser.add_argument('--dropout_prob', default=0.3, type=float, help='Set the epochs for model training')

    # Parse the arguments and run main()
    args = parser.parse_args()
    main(args)