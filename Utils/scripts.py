from Utils import download_incidents, INPUT_ROOT, OUTPUT_ROOT

__author__ = 'roeiherz'


def download_data(input, output):
    """
    This function downloads the data - Nexar's incidents
    :return:
    """
    download_incidents(input_file=input, output_dir=output)


if __name__ == '__main__':
    download_data(input=INPUT_ROOT, output=OUTPUT_ROOT)
