from Utils import download_incidents

__author__ = 'roeiherz'


def download_data(input, output):
    """
    This function downloads the data - Nexar's incidents
    :return:
    """
    download_incidents(input_file=input, output_dir=output)


if __name__ == '__main__':
    download_data(input="/Users/roeiherzig/Downloads/Find_Video S3 Path for Incident_2018_04_10.csv",
                  output="/Users/roeiherzig/Incidents")
