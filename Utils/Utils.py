import numpy as np
import csv
import cv2
import av
import boto3
import os

FILE_EXISTS_ERROR = (17, 'File exists')

VIDEO_PATH = "/data/Incidents/Videos"
INDEX_PATH = "/data/Incidents/index.csv"
IMAGE_PATH = "/data/Incidents/Images"


# INDEX_FILE = "/Users/roeiherzig/Datasets/Incidents/index.csv"
# INPUT_ROOT = "/Users/roeiherzig/Downloads/"
# OUTPUT_ROOT = "/Users/roeiherzig/Incidents"


def create_folder(path):
    """
    Checks if the path exists, if not creates it.
    :param path: A valid path that might not exist
    :return: An indication if the folder was created
    """
    folder_missing = not os.path.exists(path)

    if folder_missing:
        # Using makedirs since the path hierarchy might not fully exist.
        try:
            os.makedirs(path)
        except OSError as e:
            if (e.errno, e.strerror) == FILE_EXISTS_ERROR:
                print(e)
            else:
                raise

        print('Created folder {0}'.format(path))

    return folder_missing


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


def video_to_frames(input_video, out_dir, refinment=1, jump=False, fps=None):
    video = av.open(input_video.encode("utf8"))
    rotation = int(video.streams[0].metadata.get('rotate', 0))
    vidcap = cv2.VideoCapture(input_video)

    # Jump using fps with its None
    if jump and fps is None:
        duration = float(video.streams[0].duration * video.streams[0].time_base)
        frames = video.streams[0].frames
        fps = int(round(frames / duration))
    elif fps is None:
        fps = 1

    count = 0
    image_files = []
    counter = 0
    index = 0
    while True:
        success, image = vidcap.read()
        if not success:
            print("Finished/Error in video: {}".format(input_video))
            break
        counter += 1
        if ((counter - 1) % refinment) > 0:
            continue

        image = rotate_bound(image, rotation)
        outpath = os.path.join(out_dir, "%.6d.jpg" % (index))

        if count % fps == 0:
            cv2.imwrite(outpath, image)
            image_files.append(outpath)
            index += 1
        count = count + 1


def download_incidents(input_file="", output_dir=""):
    """
    This function downloads incidents via index csv file
    :param input_file: input file path to index.csv which contains incident_id and video_link
    :param output_dir: output dir path to download the protobufs
    :return:
    """
    pb_links = []
    with open(os.path.join(input_file), 'rb') as mapping_csv:
        reader = csv.reader(mapping_csv)
        for ind, row in enumerate(reader):

            # First line is a header
            if ind == 0:
                continue

            # Negative example
            if row[3] != 'yes':
                continue

            link = row[1]
            parse = link[link.find("?") + 1:]
            parse = parse.split("&")
            download_link = "user/{}/ride/{}/artifacts/incident-{}.mp4".format(parse[0][parse[0].find("=") + 1:],
                                                                                parse[1][parse[1].find("=") + 1:],
                                                                                parse[2][parse[2].find("=") + 1:])
            # user/25da72a103f16be86fb50c1d457a3d87/ride/d4576d10bb0b58432f094bd24602411b/artifacts/incident-2319d2801dff97e4bb2dbda2c4f37fae.mp4

            pb_links.append(download_link)

    download_from_s3('nexar-upload', pb_links, output_dir)


def download_from_s3(bucket_name, list_of_objects_to_download, out_dir):
    s3_client = boto3.client('s3')
    print('Downloading {} objects from S3 to bucket "{}"'.format(len(list_of_objects_to_download), bucket_name))
    n_print = 10
    i = 1
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    errors = []
    for s3_obj in list_of_objects_to_download:
        local_obj = os.path.join(out_dir, os.path.basename(s3_obj))
        if not os.path.isfile(local_obj):
            try:
                s3_client.download_file(bucket_name, s3_obj, local_obj)
            except Exception as e:
                print('Failed to download object {} to {}'.format(s3_obj, local_obj))
                errors.append((s3_obj, local_obj))
        i += 1
        if i % n_print == 0:
            print('  --> {}  '.format(i))

    errors_1 = []
    if len(errors) > 0:
        print('Failed to download {} objects'.format(len(errors)))
        for s in errors:
            s3_obj = s[0]
            local_obj = s[1]
            try:
                s3_client.download_file(bucket_name, s3_obj, local_obj)
            except:
                errors_1.append((s3_obj, local_obj))

    print('Downloaded ended.')
    if len(errors_1) > 0:
        print('Failed to download {} objects'.format(len(errors_1)))
