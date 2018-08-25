import cv2

import av
import shutil

from Utils import download_incidents, INPUT_ROOT, OUTPUT_ROOT, INDEX_FILE
import os
import pandas as pd

__author__ = 'roeiherz'

DATA_PATH = "/Users/roeiherzig/Datasets/Incidents/"


def download_nexar_data(input, output):
    """
    This function downloads the data - Nexar's incidents
    :return:
    """
    download_incidents(input_file=input, output_dir=output)


def get_data_split(index_path):
    """
    This function use index file and split it accordingly
    :return:
    """

    # Get dir path
    dir_path = os.path.dirname(index_path)
    # Get files from dir
    files = os.listdir(dir_path)
    # Get index data frame
    df = pd.read_csv(index_path)

    # # Split Train data
    df_train = df[df.Dataset == "Train"]
    train_links = [os.path.basename(ll) for ll in list(df_train.link)]

    for fl in files:
        if fl in train_links:
            src = os.path.join(dir_path, fl)
            dst = os.path.join(dir_path, "Train", fl)
            shutil.move(src, dst)

    print("Finished splitting Training data")
    # # Split Test data
    df_test = df[df.Dataset == "Test"]
    test_links = [os.path.basename(ll) for ll in list(df_test.link)]

    files = os.listdir(dir_path)
    for fl in files:
        if fl in test_links:
            src = os.path.join(dir_path, fl)
            dst = os.path.join(dir_path, "Test", fl)
            shutil.move(src, dst)

    print("Finished splitting Testing data")


if __name__ == '__main__':
    get_data_split(index_path=INDEX_FILE)
    exit()

    # download_nexar_data(input=os.path.join(INPUT_ROOT, "Find_Video S3 Path for Incident_2018_04_12.csv"), output=OUTPUT_ROOT)
