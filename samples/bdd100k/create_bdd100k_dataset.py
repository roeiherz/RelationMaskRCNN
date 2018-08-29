import argparse
import csv
import os
import re
import pandas as pd
import numpy as np
import json

DATASET_PATH = "/data/BDD/bdd100k/labels/100k/"


def get_data(dir_path, pixel_thr=20):
    """
    This function returns the csv data and labels formats:
    csv_data: list of [path/to/image.jpg, x1, y1, x2, y2, class_name]
    labels: list of [class_label, index number]
    :param dir_path: directory path
    :param pixel_thr: pixel threshold
    :return:
    """
    csv_data = []
    labels = set([])
    for mode in ['train', 'test']:
        print("Get data from {}".format(mode))
        for fl in os.listdir(os.path.join(dir_path, mode)):

            data_path = os.path.join(dir_path, mode, fl)
            data = json.load(open(data_path))

            for frame in data['frames']:
                for obj in frame['objects']:

                    # Not an object
                    if 'box2d' not in obj:
                        continue

                    # Get image path
                    img_path = "{0}/images/100k/{1}/{2}.jpg".format(dir_path, mode, fl.split('.')[0])
                    x1 = int(obj['box2d']['x1'])
                    y1 = int(obj['box2d']['y1'])
                    x2 = int(obj['box2d']['x2'])
                    y2 = int(obj['box2d']['y2'])
                    label = obj['category']

                    # Few pixels
                    if (int(obj['box2d']['x2']) - int(obj['box2d']['x1'])) < pixel_thr or \
                            (int(obj['box2d']['y2']) - int(obj['box2d']['y1'])) < pixel_thr:
                        continue

                    # row example to be saved in the csv: [path/to/image.jpg,x1,y1,x2,y2,class_name]
                    row = [img_path, x1, y1, x2, y2, label]
                    labels.add(obj['category'])
                    csv_data.append(row)

    print("Got annotations and labels data for BDD100K")
    return csv_data, labels


if __name__ == '__main__':

    # Get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--local', help='local variable for debugging', action='store', default=False)
    parser.add_argument('--dataset_dir', help='index csv file', action='store', default=DATASET_PATH)
    args = parser.parse_args()

    # Use Local params
    if args.local:
        args.dataset_dir = "/Users/roeiherzig/Datasets/BBD/bdd100k/labels/100k/"

    # Check directory exists
    if not os.path.exists(args.dataset_dir):
        print('Can not find data set BDD100K directory: {}'.format(args.dataset_dir))
        exit(-1)

    csv_data, labels = get_data(args.dataset_dir, pixel_thr=20)

    # Save annotations to csv file
    path_write_file = "bdd100k_annotations.csv"
    with open(path_write_file, 'wb') as fl:
        writer = csv.writer(fl)
        writer.writerows(csv_data)
    print("BDD100K annotations have been saved in {}".format(path_write_file))

    # Save class mapping file
    csv_labels = []
    ind = 0
    for label in labels:
        label = [label, ind]
        ind += 1
        csv_labels.append(label)

    path_write_file = "bdd100k_class_mappings.csv"
    with open(path_write_file, 'wb') as fl:
        writer = csv.writer(fl)
        writer.writerows(csv_labels)
    print("BDD100K class mappings have been saved in {}".format(path_write_file))
