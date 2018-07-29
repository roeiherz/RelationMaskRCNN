import cv2
import numpy as np
from mrcnn.Visualizer import VisualizerDrawer
from mrcnn.utils import denorm_boxes, norm_boxes, resize_image
from samples.coco.coco import CocoConfig


class InferenceConfig(CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0


__author__ = 'roeiherz'

if __name__ == '__main__':

    config = InferenceConfig()
    config.display()
    boxes = []

    img = cv2.imread("/home/roeih/RelationMaskRCNN/000000419759.jpg")
    original_image_shape = img.shape
    image, window, scale, padding, crop = resize_image(
        img,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)

    window = norm_boxes(window, image.shape[:2])
    wy1, wx1, wy2, wx2 = window
    shift = np.array([wy1, wx1, wy1, wx1])
    wh = wy2 - wy1  # window height
    ww = wx2 - wx1  # window width
    scale = np.array([wh, ww, wh, ww])
    # Convert boxes to normalized coordinates on the window
    boxes = np.divide(boxes - shift, scale)
    # Convert boxes to pixel coordinates on the original image
    boxes = denorm_boxes(boxes, original_image_shape[:2])

    # Filter out detections with zero area. Happens in early training when
    # network weights are still random
    exclude_ix = np.where(
        (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
    if exclude_ix.shape[0] > 0:
        boxes = np.delete(boxes, exclude_ix, axis=0)

    new_boxs = np.zeros((boxes.shape))
    new_boxs[:, 0] = boxes[:, 1]
    new_boxs[:, 1] = boxes[:, 0]
    new_boxs[:, 2] = boxes[:, 3]
    new_boxs[:, 3] = boxes[:, 2]
    new_boxs = new_boxs.astype(np.int32)

    VisualizerDrawer.draw_labeled_probe(img, new_boxs)
    cv2.imwrite("test.jpg", img)
