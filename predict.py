import json
import logging
import torch
import argparse

from helper.predict_helper import Predictor

logging.basicConfig(
    level=logging.INFO,  # or DEBUG for more verbosity
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def main(args):
    
    predict_config = {
        "device": torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu"),
        "checkpoint_pth": args.checkpoint,
        "top_k": args.top_k,
        "category_names_file": args.category_names
    }

    # Print model configuration for info
    predict_config_serializable = predict_config.copy()
    predict_config_serializable["device"] = str(predict_config_serializable["device"])
    logging.info("Model Prediction Configuration:\n%s", json.dumps(predict_config_serializable, indent=2))

    # Create predictor object
    predictor = Predictor(predict_config)

    # Predict the class of the provided image
    predictor.predict_class(img_path=args.img_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model prediction specifications
    parser.add_argument('img_path', help='Path to image as model input')
    parser.add_argument('checkpoint', help='Path to checkpoint file for loading the trained model')
    parser.add_argument('--top_k', type=int, default='3', help="Number of top predicted classses to show")
    parser.add_argument('--category_names', default='cat_to_name.json', help="Path to mapping files of categories to real names")
    parser.add_argument('--gpu', action='store_true', help='Enable using GPU if available (default is CPU)')

    # Parse the arguments and run main()
    args = parser.parse_args()
    main(args)