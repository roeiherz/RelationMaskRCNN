import numpy as np
import os
import sys
# import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)
import time
import random
import matplotlib.pyplot as plt
# Import Mask RCNN
import skimage

from mrcnn.utils import compute_iou
from samples.bdd100k.BDD100K import BDD100KDataset, BDD100KConfig
from samples.coco.coco import CocoConfig

ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
# Root directory of the project
from mrcnn import model as modellib
import argparse
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


def get_ids_from_uuids(dataset, uuids):
    """
    This function get uuids and returns the ids
    :param dataset: dataset
    :param uuids: uuids from bdd100k
    :return:
    """
    ids = []
    i = 0
    for img in dataset.image_info:
        if img['id'] in uuids:
            ids.append(i)
        i += 1
    return ids


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
                        default=None,
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

    # Define GPU training
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Use Local params
    if args.local:
        args.dataset_dir = "/Users/roeiherzig/Datasets/BDD/bdd100k/"
        # Resnet101 COCO Model
        # args.model = "/Users/roeiherzig/RelationMaskRCNN/logs/Coco/mask_rcnn_coco.h5"
        # Resnet101 Pretrained COCO Model only rois fixed
        # args.model = "/Users/roeiherzig/RelationMaskRCNN/logs/bdd100k20180928T1743/mask_rcnn_bdd100k_0160.h5"
        # different loss
        # args.model = "/Users/roeiherzig/RelationMaskRCNN/logs/bdd100k20180928T1748/mask_rcnn_bdd100k_0023.h5"
        # Resnet101 Pretrained bdd100k20180928T1743 Model GPI only rois fixed
        args.model = "/Users/roeiherzig/RelationMaskRCNN/logs/bdd100k20180929T1156/mask_rcnn_bdd100k_0088.h5"
        # Resnet101 GPI Model pre trained from COCO
        # args.model = "/Users/roeiherzig/RelationMaskRCNN/logs/bdd100k20180926T1231/mask_rcnn_bdd100k_0009.h5"
        # Resnet50 pretrained on bdd without GPI
        # args.model = "/Users/roeiherzig/RelationMaskRCNN/logs/bdd100k20181018T2014/mask_rcnn_bdd100k_0149.h5"
        args.save_path = "/Users/roeiherzig/RelationMaskRCNN/samples/bdd100k"
        # args.save_path = "/Users/roeiherzig/RelationMaskRCNN/samples/bdd100k/7_160_resnet101.jpg"

    print("Model: ", args.model)
    print("Dataset dir: ", args.dataset_dir)
    print("Logs: ", args.logs)
    print("GPU: ", args.gpu)
    print("Number of Workers: ", args.workers)
    print("Save Path: ", args.save_path)
    print("Local: ", args.local)
    print("Limit: ", args.limit)


    # Configurations
    class InferenceConfig(BDD100KConfig):
        # Set batch size to 1 since we'll be running inference on one image at a time.
        # Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        DETECTION_MIN_CONFIDENCE = 0.0
        POST_NMS_ROIS_INFERENCE = 100


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

    # Testing dataset
    # dataset = BDD100KDataset()
    # dataset.load_bdd100k(args.dataset_dir, "val", load_images_flag=False)
    # dataset.prepare()

    ids = ["000090", "000091", "000093", "000094", "000095", "000096", "000097", "000098", "000099", "000100",
           "000101", "000102", "000103", "000104", "000105", "000106", "000107", "000108", "000109", "000110",
           "000111", "000112", "000113", "000114", "000115", "000116", "000117", "000119", "000120", "000121"]
    uuid = "ae006141-d434-4f38-91f3-4b2f390c9025"
    main_acc_box = [430, 606, 575, 1034]
    main_acc_box_area = float(main_acc_box[2] - main_acc_box[0]) * (main_acc_box[3] - main_acc_box[1])

    for image_id in ids:
        # Load image and mask
        image = skimage.io.imread("/Users/roeiherzig/Datasets/Accidents1K/Images/{}/{}.jpg".format(uuid, image_id))
        original_shape = image.shape

        # Run object detection
        results = model.detect([image], verbose=1, gpi_type=config.GPI_TYPE)

        # Display results
        ax = get_ax(1)
        r = results[0]
        gpi = "" if config.GPI_TYPE is None else "_gpi"
        visualize.save_instances(image, r['rois'], np.array([]), r['class_ids'], np.array([]), np.array([]),
                                 r['scores'],
                                 ax=ax, title="Predictions_{}_{}".format(uuid, image_id),
                                 path="{}/{}_{}_{}.jpg".format(args.save_path, args.model.split('/')[-2], uuid,
                                                               image_id),
                                 show_mask=False)

        # Get the main accidents box
        rois_area = (r['rois'][:, 2] - r['rois'][:, 0]) * (r['rois'][:, 3] - r['rois'][:, 1])
        ious = compute_iou(main_acc_box, r['rois'].astype("float"), main_acc_box_area, rois_area)
        # chosen_iou = np.array([r['rois'][np.argmax(ious)]])
        chosen_iou_ind = np.argmax(ious)

        if r['relation_attention'] is not None:
            visualize.draw_attention(r['rois'], r['relation_attention'], image, "{}_{}".format(uuid, image_id),
                                     chosen_iou_ind)

    print("End Graph Detector Prediction")
