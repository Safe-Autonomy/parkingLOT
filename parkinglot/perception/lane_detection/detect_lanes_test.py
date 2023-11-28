#!/usr/bin/env python3

import time
import cv2
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from skimage import morphology
from scipy import ndimage

from YOLOPv2.utils.utils import *

from utils.dataloader import LoadImage
from utils.line_fit import bird_viz, final_viz, fit_one_lane, fit_two_lane
from utils.misc import Line, perspective_transform

# from parkinglot.topics import SIM_CAMERA_TOPIC, GEM_CAMERA_TOPIC

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32MultiArray
from parkinglot.topics import SIM_CAMERA_TOPIC, GEM_CAMERA_TOPIC

class LaneDetector():
    def __init__(self, image_topic: str, device=None, enable_ros=True) -> None:

        # ------------- Rospy Init-------------
        if enable_ros:

            # init ros node
            self.cameraSub = rospy.Subscriber(image_topic, Image, self.image_handler, queue_size=1)
            self.laneDetPub = rospy.Publisher("lane_detection/image", Image, queue_size=1)
            self.bridge = CvBridge()
            self.centerlinePub = rospy.Publisher("lane_detection/centerline_points", Float32MultiArray, queue_size=1) 

        # ------------- YOLOPv2 Init-------------
        # Load model
        weights = './YOLOPv2/data/weights/yolopv2.pt'   # local relative path

        self.device = device
        self.model = torch.jit.load(weights).to(device=self.device)
        self.model.half() if torch.cuda.is_available() else self.model.float()
        self.model.eval() # set model to eval mode
        self.imgsz = 640
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once

        # ------------- Lane Detection Init-------------
        self.left_line = Line(n=5)
        self.right_line = Line(n=5)
        self.detected = False
        self.hist = True
        self.count = 0

    def image_handler(self, data):
        try:
            # Convert a ROS image message into an OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        # run detection
        combine_fit_img, bird_fit_img ,mid_line_pts= self.detect_lane_pipeline(cv_image.copy(),apply_obj_det=False)
        if combine_fit_img is not None:
            # Convert an OpenCV image into a ROS image message
            out_img_msg = self.bridge.cv2_to_imgmsg(combine_fit_img, 'bgr8')
            out_birdeye_msg = self.bridge.cv2_to_imgmsg(bird_fit_img,'bgr8')

            # Publish image message in ROS
            self.laneDetPub.publish(out_img_msg)
            self.laneDetPub.publish(out_birdeye_msg)
            self.centerlinePub.publish(mid_line_pts)

    # ------- These 2 functions belowed are modified from YOLOPv2 utils -------
    def lane_line_mask(self, ll = None):
        ll_seg_mask = torch.round(ll).squeeze(1)
        ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()
        return ll_seg_mask
    
    def detect_lanes_yolop(self, img, iou_thres=0.45,conf_thres=0.3,classes=0,agnostic_nms=False,apply_obj_det=False):

        # ------ initialize timer ------
        inf_time = AverageMeter()
        waste_time = AverageMeter()
        nms_time = AverageMeter()

        # ------ pre-processing -------
        data_img = LoadImage(img, img_size=640, stride=32)
        iter_data_img = iter(data_img)

        
        t0 = time.time()
        
        # load first image in data_img (NOTE: only one)
        img, im0s = next(iter_data_img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() 
        img /= 255.0  # normalize
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # -------- Inference ---------
        t1 = time_synchronized()
        with torch.no_grad():
            [pred,anchor_grid],seg,ll= self.model(img)
            torch.cuda.empty_cache()
        t2 = time_synchronized()

        # ------ pre-processing -------
        # waste time: the incompatibility of  torch.jit.trace causes extra time consumption in demo version 
        # but this problem will not appear in offical version 
        pred = split_for_trace_model(pred,anchor_grid)

        # Apply NMS
        t3 = time_synchronized()
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=0, agnostic=agnostic_nms)
        t4 = time_synchronized()

        ll_seg_mask = lane_line_mask(ll)

        if apply_obj_det:
            # Process detections
            # TODO implement post-processing for object detection
            pass 

        inf_time.update(t2-t1,img.size(0))
        nms_time.update(t4-t3,img.size(0))
        print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' % (inf_time.avg,nms_time.avg))
        print(f'Done. ({time.time() - t0:.3f}s)')

        return ll_seg_mask
        # return (ll_seg_mask*255).astype(np.uint8)
        
    def detect_lanes_color(self, img, thresh_l=(190, 255)):
        """
        Convert RGB to LUV and HSV color space and apply threshold for white lane
        """
        # White Line Detector : 
        # Convert to LUV color space and threshold
        white_lane = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)[:, :, 0]
        white_lane_mask = np.zeros_like(white_lane)
        white_lane_mask[(white_lane >= thresh_l[0]) & (white_lane <= thresh_l[1])] = 1

        return white_lane_mask
    
    def extract_waypoints(self, camera_img, lane_mask):

        # Perspective Transform
        img_birdeye, M, Minv = perspective_transform(lane_mask)
        # cv2.imwrite('./visualization/birdeye.png',img_birdeye*255)
        # find numer of lanes
        labeled_lanes, num_lanes = ndimage.label(img_birdeye)
        print("numer of lanes:", num_lanes)

        # check num of lanes
        if num_lanes == 1:
            mid_line_pts = fit_one_lane(labeled_lanes)
        elif num_lanes == 2:
            mid_line_pts = fit_two_lane(labeled_lanes)

        
        # print(mid_line_pts)
        birdeye_fit_img = bird_viz(img_birdeye, mid_line_pts)
        combine_fit_img = final_viz(camera_img, mid_line_pts, Minv)
        # cv2.imwrite('visualization/combine_fit_img.png',combine_fit_img)
        # cv2.imwrite('visualization/birdeye_fit_img.png',birdeye_fit_img)

        self.laneDetPub.publish(mid_line_pts) #probly wrong

        return combine_fit_img, birdeye_fit_img, mid_line_pts
        
    def detect_lane_pipeline(self, camera_img, apply_obj_det=False, visualize=False):

        # detect lanes
        ll_seg_mask = self.detect_lanes_yolop(camera_img,apply_obj_det=False)
        color_thresh_mask = self.detect_lanes_color(camera_img)
        
        # extract lane mask
        raw_lane_mask = np.zeros_like(ll_seg_mask)
        raw_lane_mask[(ll_seg_mask == 1) & (color_thresh_mask == 1)] = 1
        
        raw_lane_mask = morphology.remove_small_objects(raw_lane_mask.astype('bool'),min_size=400,connectivity=2)

        lane_mask = cv2.GaussianBlur(raw_lane_mask.astype(np.float32),(9,9),0)
        
        if visualize:
            cv2.imwrite('./visualization/lane_mask.png',lane_mask*255)

        # extract waypoints from lane mask
        combine_fit_img, bird_fit_img,mid_line_pts = self.extract_waypoints(camera_img,lane_mask)   # mid_line_pts (N,2)
        
        if visualize:
            cv2.imwrite('./visualization/combine_fit.png',combine_fit_img)
            cv2.imwrite('./visualization/bird_fit.png',bird_fit_img)

        return combine_fit_img, bird_fit_img, mid_line_pts
    

def detect_lanes():
    # init args
    rospy.init_node('detect_lanes', anonymous=True)

    # perception
    camera_topic = GEM_CAMERA_TOPIC
    LaneDetector(image_topic=camera_topic, device=torch.device('cuda:1'), enable_ros=True)
    
    while not rospy.core.is_shutdown():
        rospy.rostime.wallsleep(0.5)

# NOTE: Scripts below are for GEM testing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for lane following')
    # parser.add_argument('--gem', action=argparse.BooleanOptionalAction, help='Add this flag to run on vehicle')
    # args = parser.parse_args()

    try:
        detect_lanes()
    except rospy.exceptions.ROSInterruptException:
        rospy.loginfo("Shutting down")
    
# NOTE: Scripts below are for testing purpose only (will be removed after perception done)
if __name__ == '__main__':
    # rospy.init_node('lane_detector', anonymous=True)
    # image_topic = rospy.get_param('~image_topic', '/camera/color/image_raw')
    # device = rospy.get_param('~device', '0')
    # lane_detector = LaneDetector(image_topic, device)
    # rospy.spin()

    # img = cv2.imread('/home/gem/Documents/gem_01/src/parkingLOT/parkinglot/perception/YOLOPv2/data/samples/1.jpg')
    # img = cv2.imread('/home/ziruiw3/ece484fa23/parkingLOT/parkinglot/perception/lane_detection/data/50.png')
    img = cv2.imread('data/150.png')
    lane_detector = LaneDetector(image_topic=None, device=torch.device('cuda:0'), enable_ros = False)
    count = 1
    for i in range(count):
        lane_detector.detect_lane_pipeline(img, apply_obj_det=False, visualize=True)
        
    