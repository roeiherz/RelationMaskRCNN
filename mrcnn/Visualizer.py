import cv2
import numpy

from mrcnn.utils import BOX

__author__ = 'roeih'

FONT_FACE = cv2.FONT_HERSHEY_PLAIN
FONT_SCALE = 1
FONT_THICKNESS = 2
LINE_THICKNESS = -1
FONT_COLOR = (0, 0, 0)
GREEN_COLOR = (0, 255, 0)
FILL_COLOR = (255, 255, 0)
PADDING = 1
SCALE_FACTOR_DEF = 500


class CvColor:
    def __init__(self):
        pass

    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    BLUE = (255, 0, 0)
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    ORANGE = (0, 128, 255)
    GRAY = (160, 160, 160)
    PURPLE = (102, 0, 102)
    YELLOW = (0, 255, 255)


class VisualizerDrawer(object):
    @staticmethod
    def draw_labeled_probe(image, boxes, labels=None, scale=None):
        for index, box in enumerate(boxes):
            if labels is not None:
                label = labels[index]
            else:
                label = None
            VisualizerDrawer.draw_labeled_box(image, box, label, scale=scale)

    @staticmethod
    def draw_labeled_box(image, box, label=None, rect_color=CvColor.GREEN, scale=None, text_color=None,
                         where="top_left", label2=None):
        # Drawing the rectangular.
        cv2.rectangle(image, (box[BOX.X1], box[BOX.Y1]), (box[BOX.X2], box[BOX.Y2]), rect_color, 2)

        if label is not None:
            label = label.encode('utf-8') if isinstance(label, basestring) else str(label)

            # Drawing the label.
            SCALE_FACTOR = SCALE_FACTOR_DEF if scale is None else scale
            scale = max(1, int(image.shape[1] / SCALE_FACTOR))
            font_scale = scale
            font_thickness = scale

            label_size = cv2.getTextSize(label, FONT_FACE, font_scale, FONT_THICKNESS)
            label_pixel_size = label_size[0]

            if where == "top_left":
                pt1 = (box[BOX.X1], box[BOX.Y1])
            if where == "center":
                pt1 = ((box[BOX.X1] + box[BOX.X2]) / 2, box[BOX.Y1])
            if where == "top_right":
                pt1 = (box[BOX.X2] - label_pixel_size[0], box[BOX.Y1])
            if where == "down_left":
                pt1 = (box[BOX.X1], box[BOX.Y2])

            pt2 = tuple(map(lambda x: numpy.float32(sum(x)), zip(pt1, label_pixel_size, (2 * PADDING, 2 * PADDING))))
            # Notice: We need the conversion to numpy.float32 since the sum method will convert to float64, which
            # cv.rectangle won't accept and the falsely require integers.
            cv2.rectangle(img=image, pt1=pt1, pt2=pt2, color=FILL_COLOR, thickness=LINE_THICKNESS)

            # Notice: cv2.rectangle uses the pt1 param as the top left point of the rect, while cv2.putText puts the
            # text at bottom left corner, thus we are required to add the label pixel height.
            label_pixel_height = label_pixel_size[1]
            text_org = tuple(map(lambda x: numpy.float32(sum(x)), zip(pt1, (0, label_pixel_height),
                                                                      (PADDING, 2 * PADDING))))
            if label2 is not None:
                label_size2 = cv2.getTextSize(label2, FONT_FACE, font_scale, FONT_THICKNESS)
                label_pixel_size2 = label_size2[0]
                label_pixel_height2 = label_pixel_size2[1]
                pt1 = (int(pt1[0]), int(pt2[1]))
                pt2 = tuple(map(lambda x: numpy.float32(sum(x)), zip(pt1, label_pixel_size2, (2 * PADDING, 2 * PADDING))))
                cv2.rectangle(img=image, pt1=pt1, pt2=pt2, color=FILL_COLOR, thickness=LINE_THICKNESS)
                text_org2 = tuple(map(lambda x: numpy.float32(sum(x)), zip(pt1, (0, label_pixel_height2), (PADDING, 2 * PADDING))))

            if text_color is None:
                text_color = FONT_COLOR
            cv2.putText(image, label, text_org, FONT_FACE, font_scale, text_color, font_thickness)
            if label2 is not None:
                cv2.putText(image, label2, text_org2, FONT_FACE, font_scale, text_color, font_thickness)
