from Utils import download_incidents, INPUT_ROOT, OUTPUT_ROOT
import os

__author__ = 'roeiherz'


def download_data(input, output):
    """
    This function downloads the data - Nexar's incidents
    :return:
    """
    download_incidents(input_file=input, output_dir=output)


if __name__ == '__main__':
    download_data(input=os.path.join(INPUT_ROOT, "Find_Video S3 Path for Incident_2018_04_12.csv"), output=OUTPUT_ROOT)
