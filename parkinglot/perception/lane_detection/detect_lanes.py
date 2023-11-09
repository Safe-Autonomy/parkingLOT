#!/usr/bin/env python3

import time
import cv2
import torch
import numpy as np
import argparse

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from YOLOPv2.utils.utils import (time_synchronized,
                                  select_device, 
                                  increment_path,
                                  scale_coords,
                                  xyxy2xywh,
                                  non_max_suppression,
                                  split_for_trace_model,
                                  driving_area_mask,
                                  lane_line_mask,
                                  plot_one_box,
                                  show_seg_result,
                                  AverageMeter)

from dataloader import LoadImage

from parkinglot.topics import SIM_CAMERA_TOPIC, GEM_CAMERA_TOPIC

class LaneDetector():
    def __init__(self, image_topic: str, device=None) -> None:
        # ------------- YOLOPv2 Init-------------
        # Load model
        weights = '/home/gem/Documents/gem_01/src/parkingLOT/parkinglot/perception/lane_detection/YOLOPv2/data/weights/yolopv2.pt' # gem absolute path
        # weights = '/home/ziruiw3/ece484fa23/parkingLOT/parkinglot/perception/YOLOPv2/data/weights/yolopv2.pt'   # local absolute path

        self.device = device
        self.model = torch.jit.load(weights).to(device=self.device)
        self.model.half() if torch.cuda.is_available() else self.model.float()
        self.model.eval() # set model to eval mode
        self.imgsz = 640

        # ------------- Rospy Init-------------
        self.cameraSub = rospy.Subscriber(image_topic, Image, self.image_handler, queue_size=1)
        self.laneDetPub = rospy.Publisher("lane_detection/image", Image, queue_size=1)
        self.bridge = CvBridge()

    def image_handler(self, data):
        try:
            # Convert a ROS image message into an OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        # run detection
        out_img = self.detect_lanes(cv_image.copy(),apply_obj_det=False)

        if out_img is not None:
            # Convert an OpenCV image into a ROS image message
            out_img_msg = self.bridge.cv2_to_imgmsg(out_img, 'mono8')

            # Publish image message in ROS
            self.laneDetPub.publish(out_img_msg)

    # ------- These 2 functions belowed are modified from YOLOPv2 utils -------
    def lane_line_mask(self, ll = None):
        ll_seg_mask = torch.round(ll).squeeze(1)
        ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()
        return ll_seg_mask
    
    def detect_lanes(self, img, iou_thres=0.45,conf_thres=0.3,classes=0,agnostic_nms=False,apply_obj_det=False):

        # ------ initialize timer ------
        inf_time = AverageMeter()
        waste_time = AverageMeter()
        nms_time = AverageMeter()

        # ------ pre-processing -------
        data_img = LoadImage(img, img_size=640, stride=32)
        iter_data_img = iter(data_img)

        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        t0 = time.time()
        
        # load first image in data_img (NOTE: only one)
        img, im0s = next(iter_data_img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() 
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # -------- Inference ---------
        t1 = time_synchronized()
        [pred,anchor_grid],seg,ll= self.model(img)
        t2 = time_synchronized()

        # waste time: the incompatibility of  torch.jit.trace causes extra time consumption in demo version 
        # but this problem will not appear in offical version 
        tw1 = time_synchronized()
        pred = split_for_trace_model(pred,anchor_grid)
        tw2 = time_synchronized()

        # Apply NMS
        t3 = time_synchronized()
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=0, agnostic=agnostic_nms)
        t4 = time_synchronized()

        ll_seg_mask = lane_line_mask(ll)
        cv2.imwrite('/home/gem/Documents/gem_01/src/parkingLOT/parkinglot/perception/lane_detection/ll_seg_mask.png', ll_seg_mask*255)

        if apply_obj_det:
            # Process detections
            # TODO implement post-processing for object detection
            pass 

        inf_time.update(t2-t1,img.size(0))
        nms_time.update(t4-t3,img.size(0))
        waste_time.update(tw2-tw1,img.size(0))
        print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' % (inf_time.avg,nms_time.avg))
        print(f'Done. ({time.time() - t0:.3f}s)')

        return (ll_seg_mask*255).astype(np.uint8)

def detect_lanes():
    # init args
    rospy.init_node('detect_lanes', anonymous=True)

    # perception
    camera_topic = GEM_CAMERA_TOPIC
    LaneDetector(image_topic=camera_topic, device=torch.device('cuda:0'))
    
    while not rospy.core.is_shutdown():
        rospy.rostime.wallsleep(0.5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for lane following')
    # parser.add_argument('--gem', action=argparse.BooleanOptionalAction, help='Add this flag to run on vehicle')
    # args = parser.parse_args()

    try:
        detect_lanes()
    except rospy.exceptions.ROSInterruptException:
        rospy.loginfo("Shutting down")
    
# NOTE: Scripts below are for testing purpose only (will be removed after perception done)
# if __name__ == '__main__':
#     # rospy.init_node('lane_detector', anonymous=True)
#     # image_topic = rospy.get_param('~image_topic', '/camera/color/image_raw')
#     # device = rospy.get_param('~device', '0')
#     # lane_detector = LaneDetector(image_topic, device)
#     # rospy.spin()

#     # img = cv2.imread('/home/gem/Documents/gem_01/src/parkingLOT/parkinglot/perception/YOLOPv2/data/samples/1.jpg')
#     img = cv2.imread('/home/ziruiw3/ece484fa23/parkingLOT/parkinglot/perception/YOLOPv2/data/images/tusimple/0.jpg')
#     lane_detector = LaneDetector(image_topic=None)
#     count = 200
#     for i in range(count):
#         ll_seg_mask = lane_detector.detect_lanes(img,apply_obj_det=False)
    