#!/usr/bin/python3.7
"""
\author Yubao Liu
\date Dec. 2020
"""
from __future__ import division
from __future__ import print_function

import sys

mrcnn_root = '/home/catkin_build_ws/src/RDS-SLAM/MaskRCNN_ROS/include/MaskRCNN'
sys.path.insert(0, mrcnn_root)
print(sys.version_info)

#from mrcnn import utils
from tensorflow.python.keras.models import load_model
#from tensorflow.python.keras.backend import set_session
from torchvision import  models
import torch
import torch.nn.functional as F

import mrcnn.model as modellib
from mrcnn import visualize

from maskrcnn_ros.msg import *
import maskrcnn_ros.msg
import actionlib
import PIL
# import matplotlib.pyplot as plt
# import matplotlib
# import skimage.io
# import time
import logging
import threading
# import copy
# import cv2
# from skimage.transform import resize
# import message_filters
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Image
import rospy
import math
import random
import sys

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf


class Config(object):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """
    # Name the configurations. For example, 'COCO', 'Experiment 3', ...etc.
    # Useful if your code needs to do things differently depending on which
    # experiment is running.
    NAME = None  # Override in sub-classes

    # NUMBER OF GPUs to use. When using only a CPU, this needs to be set to 1.
    GPU_COUNT = 1

    # Number of images to train with on each GPU. A 12GB GPU can typically
    # handle 2 images of 1024x1024px.
    # Adjust based on your GPU memory and image sizes. Use the highest
    # number that your GPU can handle for best performance.
    IMAGES_PER_GPU = 2

    # Number of training steps per epoch
    # This doesn't need to match the size of the training set. Tensorboard
    # updates are saved at the end of each epoch, so setting this to a
    # smaller number means getting more frequent TensorBoard updates.
    # Validation stats are also calculated at each epoch end and they
    # might take a while, so don't set this too small to avoid spending
    # a lot of time on validation stats.
    STEPS_PER_EPOCH = 1000

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 50

    # Backbone network architecture
    # Supported values are: resnet50, resnet101.
    # You can also provide a callable that should have the signature
    # of model.resnet_graph. If you do so, you need to supply a callable
    # to COMPUTE_BACKBONE_SHAPE as well
    BACKBONE = "resnet101"

    # Only useful if you supply a callable to BACKBONE. Should compute
    # the shape of each layer of the FPN Pyramid.
    # See model.compute_backbone_shapes
    COMPUTE_BACKBONE_SHAPE = None

    # The strides of each layer of the FPN Pyramid. These values
    # are based on a Resnet101 backbone.
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]

    # Size of the fully-connected layers in the classification graph
    FPN_CLASSIF_FC_LAYERS_SIZE = 1024

    # Size of the top-down layers used to build the feature pyramid
    TOP_DOWN_PYRAMID_SIZE = 256

    # Number of classification classes (including background)
    NUM_CLASSES = 1  # Override in sub-classes

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]

    # Anchor stride
    # If 1 then anchors are created for each cell in the backbone feature map.
    # If 2, then anchors are created for every other cell, and so on.
    RPN_ANCHOR_STRIDE = 1

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256
    
    # ROIs kept after tf.nn.top_k and before non-maximum suppression
    PRE_NMS_LIMIT = 6000

    # ROIs kept after non-maximum suppression (training and inference)
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Input image resizing
    # Generally, use the "square" resizing mode for training and predicting
    # and it should work well in most cases. In this mode, images are scaled
    # up such that the small side is = IMAGE_MIN_DIM, but ensuring that the
    # scaling doesn't make the long side > IMAGE_MAX_DIM. Then the image is
    # padded with zeros to make it a square so multiple images can be put
    # in one batch.
    # Available resizing modes:
    # none:   No resizing or padding. Return the image unchanged.
    # square: Resize and pad with zeros to get a square image
    #         of size [max_dim, max_dim].
    # pad64:  Pads width and height with zeros to make them multiples of 64.
    #         If IMAGE_MIN_DIM or IMAGE_MIN_SCALE are not None, then it scales
    #         up before padding. IMAGE_MAX_DIM is ignored in this mode.
    #         The multiple of 64 is needed to ensure smooth scaling of feature
    #         maps up and down the 6 levels of the FPN pyramid (2**6=64).
    # crop:   Picks random crops from the image. First, scales the image based
    #         on IMAGE_MIN_DIM and IMAGE_MIN_SCALE, then picks a random crop of
    #         size IMAGE_MIN_DIM x IMAGE_MIN_DIM. Can be used in training only.
    #         IMAGE_MAX_DIM is not used in this mode.
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128
    # Minimum scaling ratio. Checked after MIN_IMAGE_DIM and can force further
    # up scaling. For example, if set to 2 then images are scaled up to double
    # the width and height, or more, even if MIN_IMAGE_DIM doesn't require it.
    # However, in 'square' mode, it can be overruled by IMAGE_MAX_DIM.
    IMAGE_MIN_SCALE = 0
    # Number of color channels per image. RGB = 3, grayscale = 1, RGB-D = 4
    # Changing this requires other changes in the code. See the WIKI for more
    # details: https://github.com/matterport/Mask_RCNN/wiki
    IMAGE_CHANNEL_COUNT = 3

    # Image mean (RGB)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 200

    # Percent of positive ROIs used to train classifier/mask heads
    ROI_POSITIVE_RATIO = 0.33

    # Pooled ROIs
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14

    # Shape of output mask
    # To change this you also need to change the neural network mask branch
    MASK_SHAPE = [28, 28]

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 100

    # Bounding box refinement standard deviation for RPN and final detections.
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 100

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.7

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimizer
    # implementation.
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9

    # Weight decay regularization
    WEIGHT_DECAY = 0.0001
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }
    USE_RPN_ROIS = True
    TRAIN_BN = False  # Defaulting to False since batch size is often small

    # Gradient norm clipping
    GRADIENT_CLIP_NORM = 5.0

    def __init__(self):
        """Set values of computed attributes."""
        # Effective batch size
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

        # Input image size
        if self.IMAGE_RESIZE_MODE == "crop":
            self.IMAGE_SHAPE = np.array([self.IMAGE_MIN_DIM, self.IMAGE_MIN_DIM,
                self.IMAGE_CHANNEL_COUNT])
        else:
            self.IMAGE_SHAPE = np.array([self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM,
                self.IMAGE_CHANNEL_COUNT])

        # Image meta data length
        # See compose_image_meta() for details
        self.IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")


config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction=0.8
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


# Here should be the project name

# logging.basicConfig( level=logging.DEBUG, format='(%(threadName)-9s) %(message)s',)

logging.basicConfig(filename='maskrcnn_action_server.log',
                    format='%(asctime)s %(levelname)s:%(message)s',
                    level=logging.INFO,
                    datefmt='%m/%d/%Y %I:%M:%S %p')

# Maskrcnn
# Root directory of the project
# TODO: Set ROOT_DIR in launch file
ROOT_DIR = os.path.abspath(
    "/root/catkin_ws/src/MaskRCNN_ROS/include/MaskRCNN/")
# ROOT_DIR = os.path.abspath( "/root/catkin_ws/src/MaskRCNN_ROS/include/MaskRCNN/")
print("ROOT_DIR: ", ROOT_DIR)

# why cannot use the relative path
# ROOT_DIR = os.path.abspath("include/MaskRCNN/")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
# from mrcnn import utils
# Import COCO config
# To find local version
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))
#import coco
# from mrcnn.config import Config

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# Local path to trained weights file
# COCO_MODEL_PATH = os.path.join(ROOT_DIR, "model/mask_rcnn_coco.h5")
# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
# RESULT_DIR = os.path.join(ROOT_DIR, "results")

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = [
    'BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# Totoal time consuming
total_time = float(0.0)
# Total number of images
total_number = int(0)

# To sync semantic segmentation thread and action server thread
cn_task = threading.Condition()
cn_ready = threading.Condition()

batch_size = 2

class InferenceConfig(Config):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128
    NUM_CLASSES = 1 + 80  # COCO has 80 classes
    NAME = "coco"


class MaskRcnn(object):
    def __init__(self):
        print('Init MaskRCNN')
        # Init Mask rcnn
        # Initialize Maskrcnn
        config = InferenceConfig()
        config.display()

        # ORIGINAL = MASKRCNN using Tensorflow
        # Create model object in inference mode.
        self.model = modellib.MaskRCNN(mode="inference",
                                       model_dir=MODEL_DIR,
                                       config=config)
        self.coco_model_path_ = rospy.get_param(
            'model_path', '/root/cnnmodel/mask_rcnn_coco.h5')
        # Load weights trained on MS-COCO
        # self.model.load_weights(COCO_MODEL_PATH, by_name=True)
        self.model.load_weights(self.coco_model_path_, by_name=True)

        # if bPublish_result:
        self.masked_image__pub = rospy.Publisher("masked_image",
                                                 Image,
                                                 queue_size=10)

        print("=============Initialized MaskRCNN==============")

    def segment(self, image):
        timer_start = rospy.Time.now()

        results = self.model.detect(image, verbose=1)

        segment_time = (rospy.Time.now() - timer_start).to_sec() * 1000
        print("predict time: %f ms \n" % segment_time)

        timer_start = rospy.Time.now()
        r = results[0]
        # self.masked_image_ = visualize.ros_semantic_result(image[0],
        #                                                    r['rois'],
        #                                                    r['masks'],
        #                                                    r['class_ids'],
        #                                                    class_names,
        #                                                    r['scores'],
        #                                                    show_opencv=False)
        segment_time = (rospy.Time.now() - timer_start).to_sec() * 1000
        print("Visualize time: %f ms \n" % segment_time)

        return self.masked_image_, results
        # visualize.display_instances(color_image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

    def publish_result(self, image, stamp):
        if image is None or stamp is None:
            print("Image invalid")
            return
        # h, w, c = image.shape
        # assert c == 3
        msg_img = CvBridge().cv2_to_imgmsg(image, encoding='passthrough')
        msg_img.header.stamp = stamp
        self.masked_image__pub.publish(msg_img)


# Segmentation cannot putted into ROS action callback
def worker(maskrcnn, bPublish_result=False):
    # Control whether start maskrcnn thread
    global bStartMaskRCNN
    global color_image
    global stamp
    global masked_image
    global result

    maskrcnn = MaskRcnn()
    bStartMaskRCNN = True
    color_image = []
    stamp = []
    while bStartMaskRCNN:
        with cn_task:
            cn_task.wait()
            print("New task comming")
            timer_start = rospy.Time.now()
            masked_image, result = maskrcnn.segment(color_image)
            segment_time = (rospy.Time.now() - timer_start).to_sec() * 1000
            print("MaskRCNN segment time for cuurent image: %f ms \n" %
                  segment_time)
            with cn_ready:
                cn_ready.notifyAll()
                if bPublish_result:
                    maskrcnn.publish_result(masked_image, stamp)

    print("Exit MaskRCNN thread")


class SemanticActionServer(object):
    _feedback = maskrcnn_ros.msg.batchFeedback()
    _result = maskrcnn_ros.msg.batchResult()

    def __init__(self):
        print("Initialize Action Server")

        # Get action server name
        self._action_name = rospy.get_param('/semantic/action_name',
                                            '/maskrcnn_action_server')
        print("Action name: ", self._action_name)

        # Start Action server
        self._as = actionlib.SimpleActionServer(
            self._action_name,
            maskrcnn_ros.msg.batchAction,
            execute_cb=self.execute_cb,
            auto_start=False)
        self._as.start()

        # Semantic segentation result
        # self.label_topic = rospy.get_param('/semantic/semantic_label',
        #                                    '/semantic_label')
        # self.label_color_topic = rospy.get_param(
        #     '/semantic/semantic_label_color', '/semantic_label_color')

        # publish result
        # self.semantic_label_pub = rospy.Publisher(self.label_topic,
        #                                           Image,
        #                                           queue_size=10)
        # self.semantic_label_color_pub = rospy.Publisher(self.label_color_topic,
        #                                                 Image,
        #                                                 queue_size=10)

        # self.graph = tf.get_default_graph()
        # image = cv2.imread('/root/catkin_ws/src/MaskRCNN_ROS/include/MaskRCNN/images/tum_rgb.png')
        # results = self.model.detect([image], verbose=1)
        # r = results[0]
        # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

        print("MaskRCNN action server start...")

    # Notice: Inconvinient to add MaskRCNN detection here, because this callback running in a seperate thread,
    # The initializaiotn of MaskRCNN will not work here
    def execute_cb(self, goal):
        global total_time
        global total_number
        global color_image
        global stamp
        global masked_image
        global result
        # global bSaveResult

        # clear the old data
        color_image = []
        stamp = []

        if not self._as.is_active():
            logging.debug("[Error] MaskRCNN action server cannot active")
            return

        time_each_loop_start = rospy.Time.now()

        print("----------------------------------")
        print("ID: %d" % goal.id)
        # batch_size = goal.batch_size
        # print("batch size: %d" %  batch_size)

        # Receive source image from client
        for i in range(batch_size):
            try:
                color_image.append(
                    CvBridge().imgmsg_to_cv2(goal.image[i], 'bgr8'))
                stamp.append(goal.image[i].header.stamp)
                # logging.debug('Color: ', color_image.shape)
            except CvBridgeError as e:
                print(e)
                return

        # Test image communication
        # (rows, cols, channels) = color_image[0].shape
        # print("rows: ", rows, " cols: ", cols)
        # verify the image received
        # cv2.imshow("Image window", color_image)
        # cv2.waitKey(1)

        timer_start = rospy.Time.now()
        # perform segmenting
        if np.any(color_image[0]):
            with cn_task:
                print("Inform new task")
                cn_task.notifyAll()

            with cn_ready:
                cn_ready.wait()
                print("semantic result ready")

        # save_file = RESULT_DIR +'/'+ str(goal.id) + '.png'
        # cv2.imwrite(save_file, color_image)
        # skimage_image = skimage.io.imread(save_file)
        # print("Shape of skimage: ")

        # Visualize results
        # r = results[0]
        # visualize.display_instances(color_image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
        # # to ROS msg
        # msg_label_color = CvBridge().cv2_to_imgmsg(decoded, encoding="bgr8")
        # msg_label = CvBridge().cv2_to_imgmsg(label, encoding="mono8")

        # calculate time cost
        segment_time = (rospy.Time.now() - timer_start).to_sec() * 1000
        print("MaskRCNN segment time: %f ms \n" % segment_time)
        # calculate average time consuming
        total_time = float(total_time) + float(segment_time)
        total_number = total_number + 1
        if int(total_number) > 0:
            average = total_time / float(total_number)
            print("Average time: %f ms" % average)

        # object_num = []
        labelMsgs = []
        # scoreMsgs = []

        for i in range(batch_size):
            # Original results of MaskRCNN
            boxes = result[i]['rois']
            masks = result[i]['masks']
            class_ids = result[i]['class_ids']
            # scores = result[i]['scores']

            # Total number of objects
            N = boxes.shape[0]

            # Result of request
            label_img = np.zeros(masks.shape[0:2], dtype=np.uint8)
            #  score_img = np.zeros(masks.shape[0:2], dtype=np.float32)
            logging.info('shap of label image: %s', label_img.shape)

            # label_img_color = np.zeros((masks.shape[0], masks.shape[1], 3), dtype=np.uint8)
            # logging.info('shap of label_color image: %s', label_img_color.shape)
            for i in range(N):
                # masks is 0/1 two value image, extents to [0-255]
                mask = masks[:, :, i] * 255
                # print('type of mask: ', mask.dtype)
                # merge mask of each object to one mask image
                label_img += (masks[:, :, i] * class_ids[i]).astype(np.uint8)
                #  score_img += (masks[:, :, i] * scores[i]).astype(np.float32)

            # semantic label
            msg_label = CvBridge().cv2_to_imgmsg(label_img.astype(np.uint8), encoding='mono8')
            #  msg_score = CvBridge().cv2_to_imgmsg(
                #  score_img.astype(np.float32), encoding='passthrough')

            labelMsgs.append(msg_label)
            # scoreMsgs.append(msg_score)
            # object_num.append(N)

        # return the request result
        # Total number of objects
        # self._result.object_num = object_num
        # Return segment result
        self._result.id = goal.id
        self._result.label = labelMsgs
        #  self._result.score = scoreMsgs

        # feedback
        self._feedback.complete = True
        self._as.set_succeeded(self._result)
        self._as.publish_feedback(self._feedback)

        # calculate time consuming
        time_each_loop = (rospy.Time.now() -
                          time_each_loop_start).to_sec() * 1000
        print("Time of each request: %f ms \n" % time_each_loop)

        # Save results after sending back the results
        # if bSaveResult:
        # boxes = result['rois']
        # masks = result['masks']
        # class_ids = result['class_ids']
        # scores = result['scores']
        # N = boxes.shape[0]
        # for i in range(N):
        #     save_file = RESULT_DIR + '/mask/' + str(goal.id) + '_' + str(class_ids[i]) + '.png'
        #     m_img = (masks[:, :, i] * 255).astype(np.uint8)
        #     cv2.imwrite(save_file, m_img)
        #     # print('type of m_img: ', m_img.dtype)

        # # Save semantic label
        # save_file = RESULT_DIR + '/label/' + str(goal.id) + '.png'
        # cv2.imwrite(save_file, label_img)
        # print('type of label_img: ', label_img.dtype)

        # publish result
        # masked_image: just for visualization
        # msg_label_color = CvBridge().cv2_to_imgmsg(masked_image, encoding="bgr8")

        # if self.semantic_label_pub.get_num_connections() > 0:
        #     msg_label.header.stamp = goal.image.header.stamp
        #     self.semantic_label_pub.publish(msg_label)
        #
        # if self.semantic_label_color_pub.get_num_connections() > 0:
        #     msg_label_color.header.stamp = goal.image.header.stamp
        #     self.semantic_label_color_pub.publish(msg_label_color)


def main(args):
    global total_time
    global total_number
    # global bSaveResult

    # save result
    # bSaveResult = True
    # bSaveResult = False

    rospy.init_node('maskrcnn_server', anonymous=False)

    th_maskrcnn = threading.Thread(name='maskrcnn',
                                   target=worker,
                                   args=(
                                       MaskRcnn,
                                       False,
                                   ))
    th_maskrcnn.start()

    actionserver = SemanticActionServer()

    print('Setting up MaskRCNN Action Server...')

    logging.debug('Waiting for worker threads')
    bStartMaskRCNN = False
    main_thread = threading.currentThread()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

    for t in threading.enumerate():
        if t is not main_thread:
            t.join()


if __name__ == '__main__':
    main(sys.argv)
