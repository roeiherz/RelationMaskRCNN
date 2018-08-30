import os
import sys
# import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)
import time

# Import Mask RCNN

ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)  # To find local version of the library
# Root directory of the project
from mrcnn import model as modellib
from samples.bdd100k.BDD100K import BDD100KDataset, BDD100KConfig
import argparse


# Directory to save logs and model checkpoints, if not provided through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
# Dataset path for the data
DATASET_DIR = "/home/roeiherzig/data/BDD/bdd100k/"

if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Mask R-CNN on MS COCO.')
    parser.add_argument('--local', help='input directory of videos', action='store', default=False)
    parser.add_argument('--dataset_dir',
                        default=DATASET_DIR,
                        metavar="/path/to/coco/",
                        help='Directory of the Nexars Incidents dataset')
    parser.add_argument('--model',
                        default="nexar",
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')
    parser.add_argument('--gpu', required=False,
                        default=0,
                        metavar="0, 1, ...",
                        help='GPU number ro run',
                        type=int)
    parser.add_argument('--workers', required=False,
                        default=0,
                        metavar="0, 1, ...",
                        help='Number of workers',
                        type=int)
    args = parser.parse_args()

    # Use Local params
    if args.local:
        args.dataset_dir = "/Users/roeiherzig/Datasets/BDD/bdd100k/"
        args.model = "bdd100k"

    print("Model: ", args.model)
    print("Dataset dir: ", args.dataset_dir)
    print("Logs: ", args.logs)
    print("GPU: ", args.gpu)
    print("Number of Workers: ", args.workers)

    # Define GPU training
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Configurations training
    config = BDD100KConfig()
    config.display()

    # Create model
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=args.logs)

    # Select weights file to load
    if args.model.lower() == "bdd100k":
        model_path = model.get_imagenet_weights()
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model

    # Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path, by_name=True)

    # # Save in a new locations
    # stmp = time.strftime("%c").replace(" ", "_")
    # model_path = os.path.join(MODEL_PATH, stmp)
    # create_folder(model_path)
    # model_path = os.path.join(model_path, stmp, "mask_rcnn.h5")

    # Training dataset. Use the training set and 35K from the validation set, as as in the Mask RCNN paper.
    dataset_train = BDD100KDataset()
    dataset_train.load_bdd100k(args.dataset_dir, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = BDD100KDataset()
    dataset_val.load_bdd100k(args.dataset_dir, "val")
    dataset_val.prepare()

    # Image Augmentation
    # Right/Left flip 50% of the time
    # augmentation = imgaug.augmenters.Fliplr(0.5)
    augmentation = None
    #
    # Training - Fine tune all layers
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=320,
                layers='all',
                augmentation=augmentation,
                workers_nb=config.WORKERS_NB,
                queue_size=config.QUEUE_SIZE)

