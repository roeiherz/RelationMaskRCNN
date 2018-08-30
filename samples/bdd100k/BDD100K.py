"""
Mask R-CNN
Configurations and data loading code for MS COCO.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 coco.py train --dataset=/path/to/coco/ --model=coco

    # Train a new model starting from ImageNet weights. Also auto download COCO dataset
    python3 coco.py train --dataset=/path/to/coco/ --model=imagenet --download=True

    # Continue training a model that you had trained earlier
    python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 coco.py train --dataset=/path/to/coco/ --model=last

    # Run COCO evaluatoin on the last model you trained
    python3 coco.py evaluate --dataset=/path/to/coco/ --model=last
"""
import csv
import cv2
import os
import sys
import time
import numpy as np
import pandas as pd
from six import raise_from

# import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")



############################################################
#  Configurations
############################################################


class BDD100KConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "bdd100k"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of workers
    WORKERS_NB = 0
    # QUEUE Size
    QUEUE_SIZE = 10

    # Number of classes
    NUM_CLASSES = 2  # Incident/ No Incident

    # Train or not backbone weights
    TRAINABLE_BACKBONE = True
    TRAINABLE_FPN = True
    TRAINABLE_RPN = True

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 200

    # Size of the fully-connected layers in the classification graph
    FPN_CLASSIF_FC_LAYERS_SIZE = 1024

    # Percent of positive ROIs used to train classifier/mask heads
    ROI_POSITIVE_RATIO = 0.33

    # Max number of final detections (for inference)
    DETECTION_MAX_INSTANCES = 100

    # Minimum probability value to accept a detected instance ROIs below this threshold are skipped (for inference)
    DETECTION_MIN_CONFIDENCE = 0.7

    # Non-maximum suppression threshold for detection (for inference)
    DETECTION_NMS_THRESHOLD = 0.3


############################################################
#  Dataset
############################################################

def _open_for_csv(path):
    """ Open a file with flags suitable for csv.reader.

    This is different for python2 it means with mode 'rb',
    for python3 this means 'r' with "universal newlines".
    """
    if sys.version_info[0] < 3:
        return open(path, 'rb')
    else:
        return open(path, 'r', newline='')


def _parse(value, function, fmt):
    """
    Parse a string into a value, and format a nice ValueError if it fails.

    Returns `function(value)`.
    Any `ValueError` raised is catched and a new `ValueError` is raised
    with message `fmt.format(e)`, where `e` is the caught `ValueError`.
    """
    try:
        return function(value)
    except ValueError as e:
        raise_from(ValueError(fmt.format(e)), None)


def _read_classes(csv_reader):
    """
    Parse the classes file given by csv_reader.
    """
    result = {}
    for line, row in enumerate(csv_reader):
        line += 1

        try:

            class_id = int(float(row[1]))
            class_name = row[0]

        except ValueError:
            raise_from(ValueError('line {}: format should be \'class_name,class_id\''.format(line)), None)
        class_id = _parse(class_id, int, 'line {}: malformed class ID: {{}}'.format(line))

        result[class_name] = class_id
    return result


def _read_annotations(csv_reader, classes, start_dir_path):
    """
    Read annotations from the csv_reader.
    """
    result = {}
    for line, row in enumerate(csv_reader):
        line += 1

        try:
            img_file, x1, y1, x2, y2, class_name = row[:6]
            # Append root path
            img_file = os.path.join(start_dir_path, img_file)
            img = cv2.imread(img_file)
            if img is None:
                print("Image file {} is not existing".format(img_file))
                continue
            height, width, channels = img.shape
        except ValueError:
            raise_from(ValueError(
                'line {}: format should be \'img_file,x1,y1,x2,y2,class_name\' or \'img_file,,,,,\''.format(line)),
                None)

        if img_file not in result:
            result[img_file] = []

        # If a row contains only an image path, it's an image without annotations.
        if (x1, y1, x2, y2, class_name) == ('', '', '', '', ''):
            continue

        x1 = _parse(x1, int, 'line {}: malformed x1: {{}}'.format(line))
        y1 = _parse(y1, int, 'line {}: malformed y1: {{}}'.format(line))
        x2 = _parse(x2, int, 'line {}: malformed x2: {{}}'.format(line))
        y2 = _parse(y2, int, 'line {}: malformed y2: {{}}'.format(line))
        width = _parse(width, int, 'line {}: malformed y2: {{}}'.format(line))
        height = _parse(height, int, 'line {}: malformed y2: {{}}'.format(line))

        # Check that the bounding box is valid.
        if x2 <= x1:
            raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
        if y2 <= y1:
            raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))

        # check if the current class name is correctly present
        if class_name not in classes:
            raise ValueError('line {}: unknown class name: \'{}\' (classes: {})'.format(line, class_name, classes))

        result[img_file].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': class_name, 'width': width,
                                 'height': height})
    return result


class BDD100KDataset(utils.Dataset):

    def load_bdd100k(self, dataset_dir, subset):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the BDD100K dataset.
        subset: What to load - train, val
        """

        # Class mapping
        mappings_csv = os.path.join(dataset_dir, "bdd100k_class_mappings_{}.csv".format(subset))
        try:
            # parse the provided class file
            with _open_for_csv(mappings_csv) as file:
                class_ids = _read_classes(csv.reader(file, delimiter=','))

            # Add classes
            for index, label in enumerate(class_ids):
                self.add_class("bdd100k", index, label)

        except ValueError as e:
            raise_from(ValueError('invalid CSV class file: {}: {}'.format(mappings_csv, e)), None)

        # Images
        imgs_csv = os.path.join(dataset_dir, "bdd100k_annotations_{}.csv".format(subset))
        try:
            # Csv with img_file, distance, lane, lane_label
            with _open_for_csv(imgs_csv) as file:
                images_data = _read_annotations(csv.reader(file, delimiter=','), class_ids,
                                                start_dir_path=dataset_dir[:dataset_dir.find("BDD")])

            # Add images
            for index, img_path in enumerate(images_data):
                img_data = images_data[img_path]
                img_id = os.path.basename(img_path).split('.')[0]
                for detection in img_data:
                    box = [detection['x1'], detection['y1'], detection['x2'], detection['y2']]
                    label = detection['class']
                    width = detection['width']
                    height = detection['height']
                    self.add_image("bdd100k",
                                   image_id=img_id,
                                   path=img_path,
                                   width=width,
                                   height=height,
                                   box=box,
                                   label=label)

        except ValueError as e:
            raise_from(ValueError('invalid CSV annotations file: {}: {}'.format(mappings_csv, e)), None)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "balloon":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


############################################################
#  Nexar Evaluation
############################################################

def build_coco_results(dataset, image_ids, rois, class_ids, scores):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "coco"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score
            }
            results.append(result)
    return results


def evaluate_coco(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        # Cast masks to uint8 because COCO tools errors out on bool
        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"])
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    coco_results = coco.loadRes(results)

    # Evaluate
    cocoEval = COCOeval(coco, coco_results, eval_type)
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)
