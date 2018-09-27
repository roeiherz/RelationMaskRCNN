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
import random

import cv2
import os
import sys
import time
import numpy as np
import pandas as pd
from six import raise_from

# import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)

# Root directory of the project
from mrcnn import utils
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
from mrcnn import model as modellib, visualize
from mrcnn.utils import compute_overlap, _compute_ap


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
    BACKBONE = "resnet101"

    # Run eval of map at each end of epoch
    EVAL_MAP_IN_TRAINING = False

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes
    NUM_CLASSES = 10 + 1  # BDD100K 10 classes + 1 negative
    # NUM_CLASSES = 80 + 1  # MS-COCO 80 classes + 1 negative

    # Relation Networks or no Relation Networks at all
    # GPI_TYPE = "FeatureAttention"
    GPI_TYPE = None

    # Train or not backbone weights
    TRAINABLE_BACKBONE = True
    TRAINABLE_FPN = True
    TRAINABLE_RPN = True

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting the RPN NMS threshold.
    # TRAIN_ROIS_PER_IMAGE = 20
    TRAIN_ROIS_PER_IMAGE = 128

    # Size of the fully-connected layers in the classification graph
    # FPN_CLASSIF_FC_LAYERS_SIZE = 512
    FPN_CLASSIF_FC_LAYERS_SIZE = 1024

    # Exclude layers
    # EXCLUDE_LAYERS = ['mrcnn_bbox_fc', 'mrcnn_class_logits']
    EXCLUDE_LAYERS = None

    # Percent of positive ROIs used to train classifier/mask heads
    ROI_POSITIVE_RATIO = 0.33

    # Max number of final detections (for inference)
    DETECTION_MAX_INSTANCES = 100

    # Minimum probability value to accept a detected instance ROIs below this threshold are skipped (for inference)
    DETECTION_MIN_CONFIDENCE = 0.7

    # Non-maximum suppression threshold for detection (for inference)
    DETECTION_NMS_THRESHOLD = 0.3

    # Image meta data length
    # See compose_image_meta() for details
    IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + NUM_CLASSES

    STEPS_PER_EPOCH = 1000

    # Number of epochs
    EPOCH = 200


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


def _read_annotations(csv_reader, classes, start_dir_path, load_images_flag=True):
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
            height, width = -1, -1
            if load_images_flag:
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
            # raise ValueError('line {}: unknown class name: \'{}\' (classes: {})'.format(line, class_name, classes))
            print('line {}: unknown class name: \'{}\' (classes: {})'.format(line, class_name, classes))
            continue

        result[img_file].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': class_name, 'width': width,
                                 'height': height})
    return result


class BDD100KDataset(utils.Dataset):

    def load_bdd100k(self, dataset_dir, subset, load_images_flag=True, limit=None):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the BDD100K dataset.
        subset: What to load - train, val
        load_images_flag: Load images before read_annotations flag
        limit: int of limit of examples to validate
        """

        # Class mapping
        mappings_csv = os.path.join(dataset_dir, "bdd100k_class_mappings_{}.csv".format(subset))
        try:
            # parse the provided class file
            with _open_for_csv(mappings_csv) as file:
                class_ids = _read_classes(csv.reader(file, delimiter=','))
                # from pycocotools.coco import COCO
                # coco = COCO("/Users/roeiherzig/Datasets/MSCoco/annotations/instances_{}{}.json".format(subset, 2017))
                # class_ids = sorted(coco.getCatIds())
                # class_ids = [coco.loadCats(label)[0]["name"] for label in class_ids]

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
                                                start_dir_path=dataset_dir[:dataset_dir.find("BDD")],
                                                load_images_flag=load_images_flag)

                images_data = images_data.items()

                if limit is not None:
                    random.shuffle(images_data)
                    images_data = images_data[:limit]

            # Add images
            for index, data in enumerate(images_data):
                img_path = data[0]
                img_data = data[1]
                img_id = os.path.basename(img_path).split('.')[0]
                boxes = []
                labels = []
                widths = []
                heights = []
                for detection in img_data:
                    box = [detection['y1'], detection['x1'], detection['y2'], detection['x2']]
                    boxes.append(box)
                    label = detection['class']
                    labels.append(label)
                    width = detection['width']
                    widths.append(width)
                    height = detection['height']
                    heights.append(height)

                self.add_image("bdd100k",
                               image_id=img_id,
                               path=img_path,
                               widths=widths,
                               heights=heights,
                               boxes=boxes,
                               labels=labels)

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

def _get_detections_annotations(dataset, model, save_path=None, config=None, batch_size=1):
    """
    Get the detections from the model using the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]

    # Arguments
        generator       : The generator used to run images through the model.
        model           : The model to run on the images.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the data set.
    """
    all_detections = [[None for i in range(dataset.num_labels())] for j in range(dataset.size())]
    all_annotations = [[None for i in range(dataset.num_labels())] for j in range(dataset.size())]
    image_ids = dataset.image_ids

    size = len(image_ids)
    t_prediction = 0
    t_start = time.time()

    # Decide batches per epoch
    if size % batch_size == 0:
        num_of_batches_per_epoch = size / batch_size
    else:
        num_of_batches_per_epoch = size / batch_size + 1

    for batch in range(num_of_batches_per_epoch):
        # Define number of samples per batch
        if batch_size * (batch + 1) >= size:
            nof_samples_per_batch = size - batch_size * batch
        else:
            nof_samples_per_batch = batch_size

        image_lst = []
        gt_class_id_lst = []
        gt_bbox_lst = []
        for current_index in range(nof_samples_per_batch):
            # Get index from files
            ind = batch * batch_size + current_index
            image_id = image_ids[ind]
            # Get data
            image, _, gt_class_id, gt_bbox = modellib.load_image_gt(dataset, config, image_id)
            # Append
            image_lst.append(image)
            gt_class_id_lst.append(gt_class_id)
            gt_bbox_lst.append(gt_bbox)

        # Run detection
        t = time.time()
        r_lst = model.detect(image_lst, verbose=0)
        t_prediction += (time.time() - t)

        for current_index in range(nof_samples_per_batch):
            # Get index from files
            i = batch * batch_size + current_index
            # Get data
            image_id = image_ids[i]
            gt_class_id = gt_class_id_lst[current_index]
            gt_bbox = gt_bbox_lst[current_index]
            r = r_lst[current_index]

            image_boxes = r["rois"]
            image_labels = r["class_ids"]
            image_scores = r["scores"]
            id = dataset.image_info[image_id]['id']

            if save_path is not None:
                image = dataset.load_image(image_id)
                visualize.save_instances(image, r['rois'], gt_bbox, r['class_ids'], gt_class_id, dataset.class_names,
                                         r['scores'], ax=None, show_mask=False,
                                         path=os.path.join(save_path, "{}.jpg".format(id)),
                                         title="Predictions_{}".format(id))

            # select detections - [[num_boxes, y1, x1, y2, x2, score, class_id]]
            image_detections = np.concatenate(
                [image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

            # load the annotations - [[num_boxes, y1, x1, y2, x2, class_id]]
            annotations = np.concatenate([gt_bbox, np.expand_dims(gt_class_id, axis=1)], axis=1)

            # copy detections to all_detections
            for label in range(dataset.num_labels()):
                all_detections[i][label] = image_detections[image_detections[:, -1] == label, :-1]

            # copy detections to all_annotations
            for label in range(dataset.num_labels()):
                all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()

            # print('{}/{}'.format(i + 1, dataset.size()))

        print('Batch {}/{}'.format(batch + 1, num_of_batches_per_epoch))

    print("Prediction time: {}. Average {}/image".format(t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)
    return all_detections, all_annotations


def evaluate(
        dataset,
        model,
        iou_threshold=0.5,
        save_path=None,
        config=None,
        batch_size=1):
    """
    Evaluate a given dataset using a given model.

    # Arguments
        dataset       : The dataset that represents the dataset to evaluate.
        model           : The model to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
        batch size       : Effective batch size.
    # Returns
        A dict mapping class names to mAP scores.
    """
    # gather all detections and annotations
    all_detections, all_annotations = _get_detections_annotations(dataset, model, save_path=save_path, config=config,
                                                                  batch_size=batch_size)
    average_precisions = {}

    # process detections and annotations
    for label in range(dataset.num_labels()):
        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0

        for i in range(dataset.size()):
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]
            num_annotations += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    continue

                overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0
            continue

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = _compute_ap(recall, precision)
        average_precisions[label] = average_precision

    return average_precisions
