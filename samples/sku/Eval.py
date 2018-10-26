import os
import sys
# import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)

# Import Mask RCNN

ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)  # To find local version of the library
# Root directory of the project
from samples.sku.SKU import SKUDataset, SKUConfig, evaluate
from mrcnn import model as modellib
import argparse

# Path to trained weights file
MODEL_PATH = os.path.join(ROOT_DIR, 'weights')

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
# Dataset path for the data
DATASET_DIR = "/data/BDD/bdd100k/"


def evaluate_sku(model, dataset, config, iou_threshold=0.5, save_path=None, batch_size=1, save_detections=False):
    """
    Evaluate a given dataset using a given model.

    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
        batch size       : Effective batch size.
    """

    # run evaluation
    average_precisions = evaluate(
        dataset,
        model,
        iou_threshold=iou_threshold,
        save_path=save_path,
        config=config,
        batch_size=batch_size,
        save_detections=save_detections
    )

    mean_ap = sum(average_precisions.values()) / len(average_precisions)

    for label, average_precision in average_precisions.items():
        print(dataset.class_names[label], '{:.4f}'.format(average_precision))
    print('mAP: {:.4f}'.format(mean_ap))


if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Eval Graph Detector on BDD.')
    parser.add_argument('--local', help='local debug', action='store', default=False, type=bool)
    parser.add_argument('--dataset_dir',
                        default=DATASET_DIR,
                        metavar="/path/to/coco/",
                        help='Directory of the Nexars Incidents dataset')
    parser.add_argument('--model',
                        default="nexar",
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--save_path',
                        default=None,
                        metavar="/path/to/output_images",
                        help="Save images in path'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=None,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)',
                        type=int)
    parser.add_argument('--shuffle', required=False,
                        default=False,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)',
                        type=bool)
    parser.add_argument('--gpu', required=False,
                        default=0,
                        metavar="0, 1, ...",
                        help='GPU number ro run',
                        type=int)
    parser.add_argument('--workers', required=False,
                        default=0,
                        metavar="0, 1, ...",
                        help='Number of workers',
                        type=int)
    args = parser.parse_args()

    # Define GPU training
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Use Local params
    if args.local:
        args.dataset_dir = "/Users/roeiherzig/Datasets/BDD/bdd100k/"
        # Resnet101 Pretrained COCO Model only rois fixed
        args.model = "/Users/roeiherzig/RelationMaskRCNN/logs/bdd100k20180928T1743/mask_rcnn_bdd100k_0160.h5"
        # different loss
        # args.model = "/Users/roeiherzig/RelationMaskRCNN/logs/bdd100k20180928T1748/mask_rcnn_bdd100k_0023.h5"
        # Resnet101 Pretrained bdd100k20180928T1743 Model GPI only rois fixed
        # args.model = "/Users/roeiherzig/RelationMaskRCNN/logs/bdd100k20180929T1156/mask_rcnn_bdd100k_0061.h5"
        args.save_path = "/Users/roeiherzig/RelationMaskRCNN/samples/bdd100k/"
        # args.save_path = None
        args.limit = 500

    print("Model: ", args.model)
    print("Dataset dir: ", args.dataset_dir)
    print("Logs: ", args.logs)
    print("GPU: ", args.gpu)
    print("Number of Workers: ", args.workers)
    print("Save Path: ", args.save_path)
    print("Shuffle: ", args.shuffle)
    print("Local: ", args.local)
    print("Limit: ", args.limit)

    # Configurations
    class InferenceConfig(SKUConfig):
        # Effective Batch Size: Set batch size to 1 since we'll be running inference on one image at a time.
        # Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        DETECTION_MIN_CONFIDENCE = 0
        POST_NMS_ROIS_INFERENCE = 100
        EVAL_BATCH_SIZE = 1

    config = InferenceConfig()
    config.display()

    # Our current GPI version does not support Multi Batch
    if config.GPI_TYPE is None:
        config.EVAL_MAP_IN_TRAINING = 1

    # Create model
    model = modellib.MaskRCNN(mode="inference", config=config,
                              model_dir=args.logs)

    # Select weights file to load
    if args.model.lower() == "coco":
        model_path = model.get_imagenet_weights()
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model

    # Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path, by_name=True)

    # Testing dataset
    dataset = SKUDataset()
    dataset.load_sku(args.dataset_dir, "val", load_images_flag=False)
    dataset.prepare()

    print("Running SKU evaluation on {} images.".format(dataset.size()))
    evaluate_sku(model, dataset, config, save_path=args.save_path, batch_size=config.EVAL_BATCH_SIZE,
                 save_detections=True)
