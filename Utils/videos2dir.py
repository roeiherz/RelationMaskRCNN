import argparse
import os
import pandas as pd

from Utils.Utils import create_folder, video_to_frames
from Utils.scripts import download_data

__author__ = 'roeiherz'


def get_video_links(index_path):
    """
    This function returns the data
    :param output_dir:
    :return:
    """

    df_index = pd.read_csv(index_path)
    video_links = list(df_index['video link'].unique())

    return video_links


if __name__ == "__main__":
    """
    This Script downloads videos and using video2dir code to parse multiple videos to images 
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='input directory of videos', action='store')
    parser.add_argument('--index', help='index file path', action='store')
    parser.add_argument('--output', help='output directory of videos', action='store')
    args = parser.parse_args()

    download_data(input=args.index, output=args.input)

    # Check directory exists
    if not os.path.exists(args.input):
        print('Can not find videos directory: {}'.format(args.i))
        exit(-1)

    # Get files
    files = os.listdir(args.input)
    # files = ['412563fe-ce68-4c17-92ce-b8770d6fb140.mov']
    print('Number of files: {} from input directory'.format(len(files)))
    for base_name in files:
        try:
            if '.mov' not in base_name:
                continue

            in_dir = os.path.join(args.input, base_name)
            # Without .mov extension
            out_dir = os.path.join(args.output, os.path.splitext(base_name)[0], 'images')

            if os.path.exists(out_dir):
                print("Dir {} already exists".format(base_name))
                continue

            create_folder(out_dir)
            print('{} --> {}'.format(args.input, out_dir))
            video_to_frames(in_dir, out_dir)

        except Exception as e:
            print("Error in incident {} with {}".format(base_name, str(e)))
