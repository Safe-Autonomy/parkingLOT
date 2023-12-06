#!/usr/bin/env python3

import os
import numpy as np

import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError

from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Float32MultiArray
from sensor_msgs import point_cloud2

from parkinglot.slam.utils import *
from parkinglot.constants import *

class LidarMapper(object):
	def __init__(self):
		self.rate = rospy.Rate(20)

		self.bridge = CvBridge()

		# map image
		self.map_image = np.zeros((MAP_IMG_HEIGHT, MAP_IMG_WIDTH)).astype('uint8')
		self.current_point_image = np.zeros_like(self.map_image)
		self.map_image_pub = rospy.Publisher("/lidar_points", Image, queue_size=1) 

		# gnss map
		curr_path = os.path.abspath(__file__) 
		image_path = curr_path.split('mapping')[0] + 'image/gnss_map.png'
		self.gnss_map = cv2.imread(image_path)

		# obstacles
		self.obstacles_pub = rospy.Publisher("/obstacle", Float32MultiArray, queue_size=1) 

		# point cloud
		self.point_sub = rospy.Subscriber(LIDAR_TOPIC, PointCloud2, self.cloud_handler, queue_size=1)
		self.current_points = None

		# pose
		self.pose_sub = rospy.Subscriber("/pose", Pose, self.pose_handler, queue_size=1)
		self.x   = 0
		self.y   = 0
		self.yaw = 0
		
		# extrinsic
		self.gps_to_lidar_offset = 0

	def cloud_handler(self, data):
		assert isinstance(data, PointCloud2)

		# read + filter raw lidar frame in BEV
		gen = point_cloud2.readgen = point_cloud2.read_points(cloud=data, field_names=('x', 'y', 'z', 'ring'))
		points = []
		for p in gen:
			if abs(p[0]) < 100 and abs(p[0]) > 1 and abs(p[1]) < 100 and abs(p[1]) > 1 and p[2] > -0.5 and p[2] < 2:
				points.append([p[0], p[1], 0, 1])
		self.current_points = np.vstack(points)

		# convert lidar points lidar frame -> world frame
		self.current_points = self.transform_points_to_world(self.current_points)[:,:2]

		# publish obstacles
		array = Float32MultiArray()
		array.data = self.current_points.copy().flatten().tolist()
		self.obstacles_pub.publish(array)

		# create a debug map image
		current_point_map = self.get_current_points_image()
		self.map_image[current_point_map == 255] = 255

	def pose_handler(self, data):
		assert isinstance(data, Pose)
		
		curr_yaw = quaternion_to_yaw(data.orientation)
		curr_x = data.position.x + self.gps_to_lidar_offset * np.cos(curr_yaw) 
		curr_y = data.position.y + self.gps_to_lidar_offset * np.sin(curr_yaw)

		self.x, self.y, self.yaw = round(curr_x, 3), round(curr_y, 3), round(curr_yaw, 4)

		print(self.x, self.y, self.yaw)
	
	# lidar frame -> world frame
	def transform_points_to_world(self, points):
		return (trans_from_x_y_yaw(self.x, self.y, self.yaw) @ points.T).T

	def get_current_points_image(self):
		# self.
		# quit if there is no points
		if self.current_points is None:
			return self.image.copy()

		# convert lidar points world frame -> gnss frame 
		points = self.current_points.copy()
		lon, lat = global_xy_to_image(points[:,0], points[:,1])

		# dont plots points out of bound
		mask = np.where(np.logical_and(lon < MAP_IMG_WIDTH, lon >= 0), 1, 0)
		mask &= np.where(np.logical_and(lat < MAP_IMG_HEIGHT, lat >= 0), 1, 0)

		lon = lon[mask==1]
		lat = lat[mask==1]

		out_img = self.current_point_image.copy()
		out_img[lat, lon] = 255

		return out_img

	def start(self):
		while not rospy.is_shutdown():

			# # publish debug image
			# if self.map_image is not None:
			# 	try:
			# 		# Convert OpenCV image to ROS image and publish
			# 		debug_img = self.gnss_map.copy()
			# 		debug_img[self.map_image == 255] = [0,0,255]

			# 		lon_x, lat_y = global_xy_to_image(self.x, self.y)
			# 		cv2.circle(debug_img, (lon_x, lat_y), 12, (0,255,0), 2)

			# 		print(self.x, self.y, self.yaw)

			# 		self.map_image_pub.publish(self.bridge.cv2_to_imgmsg(debug_img, "bgr8"))
			# 	except CvBridgeError as e:
			# 		rospy.logerr("CvBridge Error: {0}".format(e))

			self.rate.sleep()

def main():

		rospy.init_node('occupancy_grid', anonymous=True)
		mapper = LidarMapper()

		try:
			mapper.start()
		except KeyboardInterrupt:
			print ("Shutting down gnss image node.")
			cv2.destroyAllWindows()

if __name__ == '__main__':
		main()


		
	
	
		
