#!/usr/bin/python3.7
"""
\author Yubao Liu
\date Dec. 2020
"""
from __future__ import division
from __future__ import print_function

import sys

# mrcnn_root = '/home/catkin_build_ws/src/RDS-SLAM/MaskRCNN_ROS/build/devel/lib'
# sys.path.insert(0, mrcnn_root + 'python3')
# print(sys.version_info)

# import mrcnn.model as modellib
# from mrcnn import visualize

from torchvision import models
from torchvision import transforms
import torch
import torch.nn.functional as F
from maskrcnn_ros.msg import *
import maskrcnn_ros.msg
import actionlib
import logging
import threading

import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Image
import PIL.Image as PIL_Image
import rospy
import os
import cv2

# Set cuda device and batch size
os.environ["CUDA_VISIBLE_DEVICES"]="0"
batch_size = 2

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
torch.cuda.Event(enable_timing=True)

# Here should be the project name
logging.basicConfig(filename='deeplab_action_server.log',
                    format='%(asctime)s %(levelname)s:%(message)s',
                    level=logging.INFO,
                    datefmt='%m/%d/%Y %I:%M:%S %p')

# Root directory of the project
ROOT_DIR = os.path.abspath(
    "/root/catkin_ws/src/MaskRCNN_ROS/include/MaskRCNN/")
print("ROOT_DIR: ", ROOT_DIR)
sys.path.append(ROOT_DIR)  # To find local version of the library
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

# Pascal VOC class names (for pretrained pytorch deeplabv3)
class_names = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
               'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
               'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

# Get total time consumed and total number of images
total_time = float(0.0)
num_images = int(0)

# To sync semantic segmentation thread and action server thread
cn_task = threading.Condition()
cn_ready = threading.Condition()

class DeepLab(object):
    def __init__(self):
        print('Init DeepLabV3')
        self.model = models.segmentation.lraspp_mobilenet_v3_large(pretrained=True)
        #self.model = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True)
        self.model.eval()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("DEVICE USED: " + str(self.device))
        self.model.to(self.device)

        # if bPublish_result:
        self.masked_image__pub = rospy.Publisher("masked_image",
                                                 Image,
                                                 queue_size=10)

        print("=============Initialized DeepLab==============")

    def segment(self, image):
        input_tensors = []
        for img in image:
            # convert to pytorch format of BCHW
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            im_pil = PIL_Image.fromarray(img)
            #image_np = np.array(image).transpose((0, 3, 1, 2))
            #transform = models.segmentation.LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1.transforms
            transform = transforms.Compose([
                transforms.Resize((520, 520)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            input_tensor = transform(im_pil) # Add batch dimension
            input_tensors.append(input_tensor)

        input_images = torch.stack(input_tensors)
        input_images = input_images.to(self.device)

        print("input shape: ")
        print(input_images.shape)

        output_tensor_pytorch = self.model(input_images)
        #timer_start = rospy.Time.now()
        starter.record()
        results = output_tensor_pytorch["out"].to('cpu')
        #segment_time = (rospy.Time.now() - timer_start).to_sec() * 1000
        ender.record()
        torch.cuda.synchronize()
        segment_time = starter.elapsed_time(ender)
        print("pytorch tensor output shape: ")
        print(results.shape)

        ## convert results to tensorflow tensor 
        # numpy_array = output_tensor_pytorch["out"].cpu().detach().numpy()
        # numpy_array_tf = numpy_array.transpose((0, 2, 3, 1))
        # results = tf.convert_to_tensor(numpy_array_tf)

        print("predict time: %f ms \n" % segment_time)
        self.masked_image_ = None # PLACEHOLDER before visualize function is implemented

        # IMPLEMENT VISUALIZE FUNCTION HERE
        # timer_start = rospy.Time.now()

        # r = results[0]
        # self.masked_image_ = visualize.ros_semantic_result(image[0],
        #                                                    r['rois'],
        #                                                    r['masks'],
        #                                                    r['class_ids'],
        #                                                    class_names,
        #                                                    r['scores'],
        #                                                    show_opencv=False)

        # segment_time = (rospy.Time.now() - timer_start).to_sec() * 1000
        # print("Visualize time: %f ms \n" % segment_time)

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


# Segmentation cannot be put into ROS action callback
def worker(deeplab, bPublish_result=False):
    global bStartDeepLab # control whether to start deeplab thread
    global color_image
    global masked_image
    global stamp
    global result

    deeplab = DeepLab()
    bStartDeepLab = True
    color_image = []
    stamp = []
    while bStartDeepLab:
        with cn_task:
            cn_task.wait()
            print("New task coming")
            timer_start = rospy.Time.now()
            masked_image, result = deeplab.segment(color_image)
            segment_time = (rospy.Time.now() - timer_start).to_sec() * 1000
            print("DeepLab segment time for current image: %f ms \n" %
                  segment_time)
            with cn_ready:
                cn_ready.notifyAll()
                if bPublish_result:
                    deeplab.publish_result(masked_image, stamp)

    print("Exit MaskRCNN thread")


class SemanticActionServer(object):
    _feedback = maskrcnn_ros.msg.batchFeedback()
    _result = maskrcnn_ros.msg.batchResult()

    def __init__(self):
        print("Initialize Action Server")

        # Get LUT image
        m_lut_file = "/root/semseg/LUT/pascal.png"
        self.m_lut_image = cv2.imread(m_lut_file, 1)

        # Get action server name
        self._action_name = rospy.get_param('/semantic/action_name',
                                            '/deeplab_action_server')
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
    # The initialization of MaskRCNN will not work here
    def execute_cb(self, goal):
        global total_time
        global num_images
        global color_image
        global stamp
        global masked_image
        global result
        # global bSaveResult

        # clear the old data
        color_image = []
        stamp = []

        if not self._as.is_active():
            logging.debug("[Error] MaskRCNN action server cannot activate")
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
                print("Semantic result ready")

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
        print("DeepLab segment time: %f ms \n" % segment_time)
        # calculate average time consumed
        total_time = float(total_time) + float(segment_time)
        num_images = num_images + 1
        if int(num_images) > 0:
            average = total_time / float(num_images)
            print("Average time: %f ms" % average)

        # object_num = []
        labelMsgs = []
        # scoreMsgs = []

        for i in range(batch_size):
            label_img = result[i].argmax(0)
            #print("class labels within image:")
            #print(np.unique(label_img.numpy()))
            label_img = torch.eq(label_img, 15).unsqueeze(0)
            print(label_img.shape)
            label_img = transforms.Resize(color_image[0].shape[:2])(label_img).squeeze(0)

            # converting to np and then to cv_mat
            #label_img_np = np.array(result[i][15].detach().numpy() * 255, dtype = np.uint8)
            # m_label_color = cv2.adaptiveThreshold(label_img_np, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)

            #label_color_tmp = cv2.cvtColor(label_img_np, cv2.COLOR_GRAY2BGR)
            #m_label_color = cv2.LUT(label_color_tmp, self.m_lut_image)  # LUT: Look up table
            # cv2.imshow("Result", label_img_np)
            # cv2.waitKey(1)

            print("output class map shape: ")
            print(label_img.shape)

            # semantic label
            msg_label = CvBridge().cv2_to_imgmsg(label_img.numpy().astype(np.uint8), encoding='mono8')

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
    global num_images
    # global bSaveResult

    # save result
    # bSaveResult = True
    # bSaveResult = False

    rospy.init_node('deeplab_server', anonymous=False)

    th_deeplab = threading.Thread(name='deeplab',
                                   target=worker,
                                   args=(
                                       DeepLab,
                                       False,
                                   ))
    th_deeplab.start()

    actionserver = SemanticActionServer()

    print('Setting up MaskRCNN Action Server...')

    logging.debug('Waiting for worker threads')
    bStartDeepLab = False
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
