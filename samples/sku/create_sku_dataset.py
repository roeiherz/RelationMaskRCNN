import csv
import json
import os

import sys

import pandas as pd

__author__ = 'roeiherz'

DATASET_PATH = "/mine_retinanet/object_detector_retinanet/data"
DATA_PATH = "/SKU_dataset/"
# DATASET_PATH = "/home/roeih/mine_retinanet/object_detector_retinanet/data"
# DATA_PATH = "/home/roeih/SKU_dataset/"


def _open_for_csv(path):
    """ Open a file with flags suitable for csv.reader.
    This is different for python2 it means with mode 'rb',
    for python3 this means 'r' with "universal newlines".
    """
    if sys.version_info[0] < 3:
        return open(path, 'rb')
    else:
        return open(path, 'r', newline='')


def get_data(csv_reader, mode='', pixel_thr=0):
    """
        This function returns the csv data and labels formats:
        csv_data: list of [path/to/image.jpg, x1, y1, x2, y2, class_name]
        labels: list of [class_label, index number]
    :param csv_reader: csv reader
    :param pixel_thr: pixel threshold
    :return:
    """
    csv_data = []
    labels = set([])

    df = pd.read_csv(os.path.join(DATASET_PATH, "{}_probes.csv".format(mode)), names=["probes"])
    mode_probes = list(df.probes)

    for line, row in enumerate(csv_reader):
        line += 1

        try:

            if line == 1:
                continue

            img_path = row[8]

            # if not os.path.exists(os.path.join(DATA_PATH, img_path)):
            #     print("Image not exits {}".format(os.path.join(DATA_PATH, img_path)))
            #     continue

            x1 = int(row[2])
            y1 = int(row[3])
            x2 = int(row[4])
            y2 = int(row[5])
            class_name = "object"
            width = int(row[-3])
            height = int(row[-2])

            # Not in the probes mode
            if img_path not in mode_probes:
                continue

            # Few pixels
            if (x2 - x1) < pixel_thr or (y2 - y1) < pixel_thr:
                continue

            # row example to be saved in the csv: [path/to/image.jpg,x1,y1,x2,y2,class_name, width, height]
            row = [img_path, x1, y1, x2, y2, class_name, width, height]
            labels.add(class_name)
            csv_data.append(row)

        except Exception as e:
            print("Error in {}".format(e))

    return csv_data, labels


def save_annotations_and_labels(csv_data_1, labels_1, mode, path):
    # Save annotations csv file
    path_write_file = os.path.join(path, "annotations_{}.csv".format(mode))
    with open(path_write_file, 'wb') as csv_writer:
        writer = csv.writer(csv_writer)
        writer.writerows(csv_data_1)
    print("Annotations saved in mode {}".format(mode))

    # Save class mapping file
    csv_labels = []
    ind = 0
    for label in labels_1:
        label = [label, ind]
        ind += 1
        csv_labels.append(label)
    path_write_file = os.path.join(path, "class_mappings_{0}.csv".format(mode))
    with open(path_write_file, 'wb') as csv_writer:
        writer = csv.writer(csv_writer)
        writer.writerows(csv_labels)
    print("Class mappings saved in mode {}".format(mode))


if __name__ == "__main__":

    input = os.path.join(DATASET_PATH, "SKU_dataset.csv")
    # Parse the provided class file
    with _open_for_csv(input) as file:
        csv_reader = csv.reader(file, delimiter=',')
        csv_data_train, labels_train = get_data(csv_reader, "train")

    with _open_for_csv(input) as file:
        csv_reader = csv.reader(file, delimiter=',')
        csv_data_val, labels_val = get_data(csv_reader, "val")

    with _open_for_csv(input) as file:
        csv_reader = csv.reader(file, delimiter=',')
        csv_data_test, labels_test = get_data(csv_reader, "test")

    save_annotations_and_labels(csv_data_train, labels_train, "train", DATASET_PATH)
    save_annotations_and_labels(csv_data_val, labels_val, "val", DATASET_PATH)
    save_annotations_and_labels(csv_data_test, labels_test, "test", DATASET_PATH)

    print("Finish")
