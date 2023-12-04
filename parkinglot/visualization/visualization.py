#!/usr/bin/env python3

import os
import numpy as np

import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError

from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Float32MultiArray
import scipy.cluster.hierarchy as hcluster

from parkinglot.slam.utils import *
from parkinglot.constants import *

class Visualizer(object):
	def __init__(self):
		self.rate = rospy.Rate(20)

		self.bridge = CvBridge()

		# map image
		self.viz_pub = rospy.Publisher("/visualization", Image, queue_size=1) 

		# gnss map
		image_path = '/home/gem/Documents/gem_01/src/parkingLOT/parkinglot/visualization/gnss_map.png'
		self.gnss_map = cv2.imread(image_path)

		# obstacles
		self.obstacles_sub = rospy.Subscriber("/obstacle", Float32MultiArray, self.obstacle_handler, queue_size=1) 
		self.obstacle_map = np.zeros((MAP_IMG_HEIGHT, MAP_IMG_WIDTH))

		# pose
		self.pose_sub = rospy.Subscriber("/pose", Pose, self.pose_handler, queue_size=1)
		self.x   = 0
		self.y   = 0
		self.yaw = 0
		
		# extrinsic
		self.gps_to_lidar_offset = 0

		# waypoints 
		self.waypoints_sub = rospy.Subscriber("/global_waypoints", Float32MultiArray, self.waypoints_handler, queue_size=1) 
		self.waypoints_x = None
		self.waypoints_y = None

	def waypoints_handler(self, data):
		assert isinstance(data, Float32MultiArray)
		waypoints = np.asarray(data.data).reshape(-1, 2)

		lat, lon = self.global_xy_to_filter_img(waypoints[:,0], waypoints[:,1])
		self.waypoints_x, self.waypoints_y = lon, lat

	def obstacle_handler(self, data):
		assert isinstance(data, Float32MultiArray)
		obstacles = np.asarray(data.data).reshape(-1, 2)

		mask = np.where(obstacles[:,1] > -15, 1, 0)
		obstacles = obstacles[mask == 1]

		thresh = 1
		clusters = hcluster.fclusterdata(obstacles, thresh, criterion="distance")

		print(len(obstacles))

		lat, lon = self.global_xy_to_filter_img(obstacles[:,0], obstacles[:,1])
		self.obstacle_map[lat, lon] = 1
		
	def global_xy_to_filter_img(self, x, y):
		lon, lat = global_xy_to_image(x.copy(), y.copy())

		# dont plots points out of bound
		mask = np.where(np.logical_and(lon < MAP_IMG_WIDTH, lon >= 0), 1, 0)
		mask &= np.where(np.logical_and(lat < MAP_IMG_HEIGHT, lat >= 0), 1, 0)

		lon = lon[mask==1]
		lat = lat[mask==1]

		return lat, lon

	def pose_handler(self, data):
		assert isinstance(data, Pose)
		
		curr_yaw = quaternion_to_yaw(data.orientation)
		curr_x = data.position.x + self.gps_to_lidar_offset * np.cos(curr_yaw) 
		curr_y = data.position.y + self.gps_to_lidar_offset * np.sin(curr_yaw)

		self.x, self.y, self.yaw = round(curr_x, 3), round(curr_y, 3), round(curr_yaw, 4)

	def create_viz_image(self):
		debug_img = self.gnss_map.copy()

		# obstacle 
		debug_img[self.obstacle_map == 1] = [0,0,255]

		# current location
		lon_x, lat_y = global_xy_to_image(self.x, self.y)
		cv2.circle(debug_img, (lon_x, lat_y), 12, (0,255,0), 2)

		# global waypoints
		if self.waypoints_x is not None:
			for i in range(len(self.waypoints_x)):
				cv2.circle(debug_img, (self.waypoints_x[i], self.waypoints_y[i]), 5, (255,255,255), -1)

		return debug_img

	def start(self):
		while not rospy.is_shutdown():

			# publish debug image
			try:
				# Convert OpenCV image to ROS image and publish
				debug_img = self.create_viz_image()

				self.viz_pub.publish(self.bridge.cv2_to_imgmsg(debug_img, "bgr8"))
			except CvBridgeError as e:
				rospy.logerr("CvBridge Error: {0}".format(e))

			self.rate.sleep()

def main():

		rospy.init_node('visualization', anonymous=True)
		viz = Visualizer()

		try:
			viz.start()
		except KeyboardInterrupt:
			print ("Shutting down gnss image node.")
			cv2.destroyAllWindows()

if __name__ == '__main__':
		main()


		
	
	
		
