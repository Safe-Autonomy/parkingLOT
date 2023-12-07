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
from scipy.spatial.transform import Rotation as Rot

from parkinglot.slam.utils import *
from parkinglot.constants import *
from parkinglot.planner.hybridastar.car import VRX, VRY

SCALE = 1 if not FLAG_GAZEBO else 4

class Visualizer(object):
	def __init__(self):
		self.rate = rospy.Rate(20)

		self.bridge = CvBridge()

		# map image
		self.viz_pub = rospy.Publisher("/visualization", Image, queue_size=1) 

		# gnss map
		if not FLAG_GAZEBO:
			curr_path = os.path.abspath(__file__) 
			image_path = os.path.dirname(curr_path) + '/gnss_map.png'
			self.gnss_map = cv2.imread(image_path)
		else:
			self.gnss_map = np.zeros((MAP_IMG_HEIGHT, MAP_IMG_WIDTH, 3)).astype('uint8')

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

		# metrics 
		self.frames = 0
		self.sum = 0

	def waypoints_handler(self, data):
		assert isinstance(data, Float32MultiArray)
		if self.waypoints_x is not None: return
		waypoints = np.asarray(data.data).reshape(-1, 3)

		lat, lon = self.global_xy_to_filter_img(waypoints[:,0], waypoints[:,1])
		self.waypoints_x, self.waypoints_y = lon, lat

	def obstacle_handler(self, data):
		assert isinstance(data, Float32MultiArray)
		obstacles = np.asarray(data.data).reshape(-1, 2)

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
		
	def create_car_contour(self, image):
		lanes_bitmask = np.zeros_like(image[:,:,0])
		lanes_bitmask[np.all(self.gnss_map > 200, axis=2)] = 255
		lanes_bitmask = lanes_bitmask > 0

		rot = Rot.from_euler('z', -self.yaw + np.pi/2).as_matrix()[0:2, 0:2]
		car_outline_x, car_outline_y = [], []
		for rx, ry in zip(VRX, VRY):
			converted_xy = np.stack([rx, ry]).T @ rot
			car_outline_x.append(converted_xy[0]+self.x)
			car_outline_y.append(converted_xy[1]+self.y)

		car_bitmask = np.zeros_like(image)
		car_x, car_y = global_xy_to_image(np.array(car_outline_x), np.array(car_outline_y))
		for i in range(4):
			cv2.line(car_bitmask, (car_x[i], car_y[i]), (car_x[(i+1)%4], car_y[(i+1)%4]), (255, 255, 255), thickness=1)
			cv2.line(image, (car_x[i], car_y[i]), (car_x[(i+1)%4], car_y[(i+1)%4]), (0, 255, 0), thickness=2)

		car_bitmask = np.all(car_bitmask.copy() > 0, axis=2)
		num_pixel_hits = np.sum(np.bitwise_and(lanes_bitmask, car_bitmask))
		self.frames += 1
		is_hit = num_pixel_hits > 5
		self.sum += is_hit
		print(f'Frame: {self.frames:4d} Hit: {str(is_hit):5s} Hit ratio: {self.sum / self.frames:.3f} ({self.sum:4d}/{self.frames:4d})')
		return image

	def create_viz_image(self):
		debug_img = self.gnss_map.copy()

		# obstacle 
		debug_img[self.obstacle_map == 1] = [0,0,255]

		# global waypoints
		if self.waypoints_x is not None:
			for i in range(len(self.waypoints_x)):
				cv2.circle(debug_img, (self.waypoints_x[i], self.waypoints_y[i]), 5 // SCALE, (255,0,0), -1)

		debug_img = self.create_car_contour(debug_img)

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


		
	
	
		
