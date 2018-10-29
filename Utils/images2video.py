import os
import cv2
import av
from os import listdir
from os.path import isfile, join
import numpy as np

__author__ = 'roeiherz'


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


def blow_video_to_images(input_video, out_dir):
    video = av.open(input_video.encode("utf8"))

    rotation = int(video.streams[0].metadata.get('rotate', 0))

    vidcap = cv2.VideoCapture(input_video)
    count = 0
    image_files = []
    while True:
        success, image = vidcap.read()
        if not success:
            print("BYEBYEVIDEO")
            break
        # if not(count % 15 == 0):
        #     count = count + 1
        #     continue
        image = rotate_bound(image, rotation)
        # if generator:
        # yield image
        # else:
        #
        # outpath = os.path.join(out_dir, "frame%d.jpg" % (count/15))
        outpath = os.path.join(out_dir, "frame%d.jpg" % (count))

        top = 0;
        bottom = 0;
        left = 0;
        right = 0;
        replicate = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REPLICATE)

        cv2.imwrite(outpath, replicate)
        print("Saved video frame to {}".format(outpath))
        image_files.append(outpath)
        count = count + 1


def images_to_video(dir_path):
    images = [fl for fl in os.listdir(dir_path) if ".jpg" in fl]
    # Sort images so the frames will be consecutive
    images.sort()
    print("Number of images {}".format(len(images)))

    output_video_path = os.path.join(dir_path, "video.avi")
    height, width, layers = cv2.imread(os.path.join(dir_path, images[0])).shape
    outvid = cv2.VideoWriter(output_video_path, -1, 5, (width, height))
    i = 0
    for img in images:
        I = cv2.imread(os.path.join(dir_path, img))
        outvid.write(I)
        i += 1

    # Cleanup and save video
    cv2.destroyAllWindows()
    outvid.release()

    return


if __name__ == "__main__":
    """
    This Script taking images and parse them to a video 
    """
    images_to_video(
        dir_path='/Users/roeiherzig/RelationMaskRCNN/samples/bdd100k/bruno_scene_understanding_video/attention_maps')
