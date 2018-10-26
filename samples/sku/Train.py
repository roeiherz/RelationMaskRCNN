import os
import sys
# import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)
import time

# Import Mask RCNN

ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)  # To find local version of the library
# Root directory of the project
from mrcnn import model as modellib
from samples.sku.SKU import SKUDataset, SKUConfig
import argparse

# Directory to save logs and model checkpoints, if not provided through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
# Dataset path for the data
DATASET_DIR = "/data/SKU_dataset/"

if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Graph Detector on SKU dataset.')
    parser.add_argument('--local', help='input directory of videos', action='store', default=False, type=bool)
    parser.add_argument('--dataset_dir',
                        default=DATASET_DIR,
                        metavar="/path/to/coco/",
                        help='Directory of the Nexars Incidents dataset')
    parser.add_argument('--model',
                        default="trax",
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=None,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)',
                        type=int)
    parser.add_argument('--shuffle', required=False,
                        default=True,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)',
                        type=bool)
    parser.add_argument('--gpu', required=False,
                        default=0,
                        metavar="0, 1, ...",
                        help='GPU number ro run',
                        type=int)
    parser.add_argument('--workers', required=False,
                        default=5,
                        metavar="0, 1, ...",
                        help='Number of workers',
                        type=int)
    parser.add_argument('--queue_size', required=False,
                        default=200,
                        metavar="0, 1, ...",
                        help='Number of workers',
                        type=int)
    args = parser.parse_args()

    # Configurations training
    config = SKUConfig()
    config.display()

    # Use Local params
    if args.local:
        args.dataset_dir = "/Users/roeiherzig/Datasets/SKU_dataset/"
        # Resnet101 COCO Model
        args.model = "/Users/roeiherzig/RelationMaskRCNN/logs/Coco/mask_rcnn_coco.h5"
        args.workers = 0
        args.queue_size = 10

    print("Model: ", args.model)
    print("Dataset dir: ", args.dataset_dir)
    print("Logs: ", args.logs)
    print("GPU: ", args.gpu)
    print("Number of Workers: ", args.workers)
    print("Number of Queue Size: ", args.queue_size)
    print("Shuffle: ", args.shuffle)
    print("Local: ", args.local)
    print("Limit: ", args.limit)

    # Define GPU training
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Create model
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=args.logs)

    # Select weights file to load
    if args.model.lower() == "sku":
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
    model.load_weights(model_path, by_name=True, exclude=config.EXCLUDE_LAYERS)

    # Check eval map in training at the end of each epoch
    predicting_model = None
    if config.EVAL_MAP_IN_TRAINING:
        # Create model
        predicting_model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.logs)
        predicting_model.load_weights(model_path, by_name=True)

    # Training dataset. Use the training set and 35K from the validation set, as as in the Mask RCNN paper.
    dataset_train = SKUDataset()
    dataset_train.load_sku(args.dataset_dir, "train", load_images_flag=not args.local)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = SKUDataset()
    dataset_val.load_sku(args.dataset_dir, "val", limit=args.limit, shuffle=args.shuffle,
                         load_images_flag=not args.local)
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
                epochs=config.EPOCH,
                layers='all',
                augmentation=augmentation,
                workers_nb=args.workers,
                queue_size=args.queue_size,
                prediction_model=predicting_model)
