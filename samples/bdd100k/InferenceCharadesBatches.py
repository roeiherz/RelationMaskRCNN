import csv

import numpy as np
import os
import sys
# import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)
import time
import random
import matplotlib.pyplot as plt
# Import Mask RCNN
import pandas as pd
import skimage


ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)  # To find local version of the library
from samples.bdd100k.BDD100K import BDD100KConfig, _open_for_csv, _read_classes
from Utils.utils import COCO_LABELS
# Root directory of the project
from mrcnn.utils import non_max_suppression
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
DATASET_DIR = "/ssd/data/Charades"
BATCH_SIZE = 15


def clean_name(name):
    """Returns a shorter version of object names for cleaner display."""
    return ",".join(name.split(",")[:1])


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
    args.dataset_dir = DATASET_DIR

    # Use Local params
    if args.local:
        args.dataset_dir = "/home/roei/Datasets/Charades/"
        # Resnet101 COCO Model
        args.model = "/home/roei/RelationMaskRCNN/logs/Coco/mask_rcnn_coco.h5"
        args.gpu = 0
        # Resnet101 Pretrained COCO Model only rois fixed
        # args.model = "/home/roei/RelationMaskRCNN/logs/bdd100k20180928T1743/mask_rcnn_bdd100k_0160.h5"
        # different loss
        # args.model = "/home/roei/RelationMaskRCNN/logs/bdd100k20180928T1748/mask_rcnn_bdd100k_0023.h5"
        # Resnet101 Pretrained bdd100k20180928T1743 Model GPI only rois fixed
        # args.model = "/home/roei/RelationMaskRCNN/logs/bdd100k20180929T1156/mask_rcnn_bdd100k_0088.h5"
        # Resnet50 pretrained on bdd 256 x 256 with GPI
        # args.model = "/home/roei/RelationMaskRCNN/logs/bdd100k20181027T1246/mask_rcnn_bdd100k_0033.h5"
        # Resnet101 GPI Model pre trained from COCO
        # args.model = "/home/roei/RelationMaskRCNN/logs/bdd100k20180926T1231/mask_rcnn_bdd100k_0009.h5"
        # Resnet50 pretrained on bdd without GPI
        # args.model = "/home/roei/RelationMaskRCNN/logs/bdd100k20181018T2014/mask_rcnn_bdd100k_0149.h5"
        args.save_path = "/home/roei/RelationMaskRCNN/samples/bdd100k"
        # args.save_path = "/home/roei/RelationMaskRCNN/samples/bdd100k/7_160_resnet101.jpg"

    # Define GPU training
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

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
        IMAGES_PER_GPU = BATCH_SIZE
        DETECTION_MIN_CONFIDENCE = 0.8
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

    # Paths
    input_path = os.path.join(args.dataset_dir, "Charades_Imgs")
    output_path = os.path.join(args.dataset_dir, "Charades_BBox")

    # Get UUIDs
    dirs = [dr for dr in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, dr))]
    dirs = dirs[1300:2600]
    print("Number of dirs: {}".format(len(dirs)))

    cnt = 0
    for uuid in dirs:
        csv_data_lst = []
        cnt += 1

        # Check whether there is a concrete
        if os.path.isfile(os.path.join(output_path, "{}.csv".format(uuid))):
            print("File {} already exists".format(uuid))
            continue

        # Sort out images
        imgs = [img for img in os.listdir(os.path.join(input_path, uuid)) if ".jpg" in img]

        # Get number of batches (one file is JSON)
        size = len(imgs)
        if size % BATCH_SIZE == 0:
            num_of_batches_per_epoch = size / BATCH_SIZE
        else:
            num_of_batches_per_epoch = size / BATCH_SIZE + 1

        print("Processing UUID: {} with number of images: {}, Index: {}".format(uuid, len(imgs), cnt))
        with open(os.path.join(output_path, "{}.csv".format(uuid)), 'wb') as fl_csv:

            for batch in range(num_of_batches_per_epoch):
                try:
                    images = []
                    image_ids = []
                    # Define number of samples per batch
                    if BATCH_SIZE * (batch + 1) >= size:
                        nof_samples_per_batch = size - BATCH_SIZE * batch
                    else:
                        nof_samples_per_batch = BATCH_SIZE

                    for current_index in range(nof_samples_per_batch):
                        # Get index from files
                        ind = batch * BATCH_SIZE + current_index
                        # Get file
                        image_id = imgs[ind]
                        # Load image and mask
                        image = skimage.io.imread(os.path.join(input_path, uuid, image_id))
                        images.append(image)
                        image_ids.append(image_id)

                    # Fill padding images for the end of the batch
                    if nof_samples_per_batch < BATCH_SIZE:
                        padding_num = BATCH_SIZE - nof_samples_per_batch
                        for padd in range(padding_num):
                            padd_img = np.zeros((config.IMAGE_MIN_DIM, config.IMAGE_MAX_DIM, 3))
                            images.append(padd_img)

                    # Run object detection
                    start = time.time()
                    results = model.detect(images, verbose=0, gpi_type=config.GPI_TYPE)
                    print("processing time on batch_num {0}: {1}".format(batch, time.time() - start))

                    # Go over results
                    current_index = 0
                    for r in results:

                        # The last batch and stop if we got to the padded zeros images
                        if nof_samples_per_batch < BATCH_SIZE:
                            if current_index == nof_samples_per_batch:
                                break

                        image = images[current_index]
                        image_id = image_ids[current_index]

                        boxes = r['rois']
                        scores = r['scores']
                        classes_ids = r['class_ids']

                        # # Sort NMS
                        # indices = non_max_suppression(boxes, scores, 0.2)
                        # boxes = boxes[indices]

                        # # Sort boxes
                        # x1, y1, y2, x2 = boxes[:, 1], boxes[:, 0], boxes[:, 2], boxes[:, 3]
                        # # x2 < 140 or x1 > 1140 or y1 < 50 or y2 > 600
                        # keep_boxes_ind = np.where((x2 > 140) & (x1 < 1140) & (y2 < 600) & (y1 > 50))[0]
                        # boxes = boxes[keep_boxes_ind]

                        # # Sort 1/6 boxes from the sides
                        # x1, y1, y2, x2 = boxes[:, 1], boxes[:, 0], boxes[:, 2], boxes[:, 3]
                        # sub_x1, obj_x1 = np.meshgrid(x1, x1)
                        # sub_y1, obj_y1 = np.meshgrid(y1, y1)
                        # sub_x2, obj_x2 = np.meshgrid(x2, x2)
                        # sub_y2, obj_y2 = np.meshgrid(y2, y2)

                        # # Take boxes indices
                        # scores = scores[keep_boxes_ind]
                        # classes_ids = classes_ids[keep_boxes_ind]

                        # # Stats detections
                        for box, score, label in zip(boxes, scores, classes_ids):

                            # if COCO_LABELS[label] == "traffic sign":
                            #     continue

                            # Append data to csv outputs
                            b = box.astype(int)
                            x1 = b[1]
                            y1 = b[0]
                            x2 = b[3]
                            y2 = b[2]

                            row = [os.path.join(uuid, image_id), x1, y1, x2, y2, score, COCO_LABELS[label]]
                            csv_data_lst.append(row)

                        # Counter
                        current_index += 1

                        # # Display results
                        # ax = get_ax(1)
                        # gpi = "" if config.GPI_TYPE is None else "_gpi"
                        # visualize.save_instances(image, boxes, np.array([]), classes_ids, np.array([]), COCO_LABELS,
                        #                          scores,
                        #                          ax=ax, title="Predictions_{}_{}".format(uuid, image_id),
                        #                          path="{}/{}_{}_{}.jpg".format(args.save_path, args.model.split('/')[-2], uuid,
                        #                                                        image_id),
                        #                          show_mask=False)

                except Exception as e:
                    print("Error: {}".format(str(e)))

            # Write file
            writer = csv.writer(fl_csv)
            writer.writerows(csv_data_lst)
    print("End BDD Prediction")
