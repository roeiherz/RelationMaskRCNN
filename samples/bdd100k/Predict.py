import os
import sys
# import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)
import time
import random
import matplotlib.pyplot as plt
# Import Mask RCNN
from samples.bdd100k.BDD100K import BDD100KDataset, BDD100KConfig

ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)  # To find local version of the library

# Root directory of the project
from Utils import create_folder
from samples.coco.coco import CocoConfig, CocoDataset
from mrcnn import model as modellib
import argparse
from mrcnn import utils
from mrcnn import visualize

# Path to trained weights file
MODEL_PATH = os.path.join(ROOT_DIR, 'weights')

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_DATASET_YEAR = "2017"
# Dataset path for the data
DATASET_DIR = "/data/BDD/bdd100k/"


def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Predict Graph Detector on BDD.')
    parser.add_argument('--local', help='local debug', action='store', default=False)
    parser.add_argument('--dataset_dir',
                        default=DATASET_DIR,
                        metavar="/path/to/coco/",
                        help='Directory of the Nexars Incidents dataset')
    parser.add_argument('--model',
                        default="nexar",
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--save_path',
                        default=None,
                        metavar="/path/to/output_images",
                        help="Save images in path'")
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
    print("Model: ", args.model)
    print("Dataset dir: ", args.dataset_dir)
    print("Logs: ", args.logs)
    print("GPU: ", args.gpu)
    print("Number of Workers: ", args.workers)
    print("Save Path: ", args.save_path)

    # Define GPU training
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Use Local params
    if args.local:
        args.dataset_dir = "/Users/roeiherzig/Datasets/BDD/bdd100k/"
        # args.model = "/Users/roeiherzig/RelationMaskRCNN/logs/bdd100k20180831T1657/mask_rcnn_bdd100k_0029.h5"
        # args.model = "/Users/roeiherzig/RelationMaskRCNN/logs/bdd100k20180831T1657/mask_rcnn_bdd100k_0042.h5"
        # args.model = "/Users/roeiherzig/RelationMaskRCNN/logs/bdd100k20180831T1657/mask_rcnn_bdd100k_0114.h5"
        # args.model = "/Users/roeiherzig/RelationMaskRCNN/logs/bdd100k20180902T1624/mask_rcnn_bdd100k_0038.h5"
        args.model = "/Users/roeiherzig/RelationMaskRCNN/logs/bdd100k20180902T1624/mask_rcnn_bdd100k_0160.h5"
        args.model = "/Users/roeiherzig/RelationMaskRCNN/logs/bdd100k20180902T1624/mask_rcnn_bdd100k_0160.h5"
        args.save_path = "/Users/roeiherzig/RelationMaskRCNN/samples/bdd100k/7_160_resnet101.jpg"

    # Configurations
    class InferenceConfig(BDD100KConfig):
        # Set batch size to 1 since we'll be running inference on one image at a time.
        # Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        DETECTION_MIN_CONFIDENCE = 0
        POST_NMS_ROIS_INFERENCE = 50
        BACKBONE = "resnet50"


    config = InferenceConfig()
    config.display()

    # Create model
    model = modellib.MaskRCNN(mode="inference", config=config,
                              model_dir=args.logs)

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
    model.load_weights(model_path, by_name=True, mode='inference')

    # # Save in a new locations
    # stmp = time.strftime("%c").replace(" ", "_")
    # model_path = os.path.join(MODEL_PATH, stmp)
    # create_folder(model_path)
    # model_path = os.path.join(model_path, stmp, "mask_rcnn.h5")

    # Testing dataset
    dataset = BDD100KDataset()
    dataset.load_bdd100k(args.dataset_dir, "val", load_images_flag=False)
    dataset.prepare()

    # image_id = random.choice(dataset.image_ids)
    image_id = 1536
    image, _, gt_class_id, gt_bbox = modellib.load_image_gt(dataset, config, image_id)
    info = dataset.image_info[image_id]
    print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                           dataset.image_reference(image_id)))
    # Run object detection
    results = model.detect([image], verbose=1)

    # Display results
    ax = get_ax(1)
    r = results[0]
    image = dataset.load_image(image_id)
    visualize.save_instances(image, r['rois'], gt_bbox, r['class_ids'], gt_class_id, dataset.class_names, r['scores'],
                             ax=ax, title="Predictions_{}".format(info["id"]), path=args.save_path,
                             show_mask=False)
    print("gt_class_id", gt_class_id)
    print("gt_bbox", gt_bbox)
