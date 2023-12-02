#!/usr/bin/env python3

import os
import cv2 
import numpy as np

import rospy
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import parkinglot.slam.utils as slam
from parkinglot.constants import *

if not FLAG_GAZEBO:
	from novatel_gps_msgs.msg import Inspva
else:
	from gazebo_msgs.srv import GetModelState, GetModelStateResponse

class GNSSLocalizer(object):

	def __init__(self):

		self.rate = rospy.Rate(45)

		# pose 
		self.pose_pub = rospy.Publisher("/pose", Pose, queue_size=1) 
		self.pose_msg = Pose()

		# Read image in BGR format
		curr_path = os.path.abspath(__file__) 
		image_path = curr_path.split('localization')[0] + 'image/gnss_map.png'
		self.map_image = cv2.imread(image_path)

		# Create the cv_bridge object
		self.bridge  = CvBridge()
		self.map_image_pub = rospy.Publisher("/motion_image", Image, queue_size=1) 

		# Subscribe information from sensors
		self.lat      = 0
		self.lon      = 0
		self.heading  = 0
		self.gnss_sub = rospy.Subscriber("/novatel/inspva", Inspva, self.inspva_callback)

		self.lat_scale    = 0.00062    # 0.0007    
		self.lon_scale    = 0.00136    # 0.00131   

		self.arrow        = 40 
		self.img_width    = 2107
		self.img_height   = 1313

	def inspva_callback(self, inspva_msg):
		self.lat     = inspva_msg.latitude
		self.lon     = inspva_msg.longitude
		self.heading = inspva_msg.azimuth 

		# pose
		curr_x, curr_y = slam.gnss_to_global_xy(self.lon, self.lat)
		quat = slam.yaw_to_quaternion(self.heading_to_yaw_stanley(self.heading)) 
		self.pose_msg.position.x = round(curr_x, 3)
		self.pose_msg.position.y = round(curr_y, 3)
		self.pose_msg.orientation = quat

		print(self.lon, self.lat)

		self.pose_pub.publish(self.pose_msg)

	def image_heading(self, lon_x, lat_y, heading):
			
		if(heading >=0 and heading < 90):
			angle  = np.radians(90-heading)
			lon_xd = lon_x + int(self.arrow * np.cos(angle))
			lat_yd = lat_y - int(self.arrow * np.sin(angle))

		elif(heading >= 90 and heading < 180):
			angle  = np.radians(heading-90)
			lon_xd = lon_x + int(self.arrow * np.cos(angle))
			lat_yd = lat_y + int(self.arrow * np.sin(angle))  

		elif(heading >= 180 and heading < 270):
			angle = np.radians(270-heading)
			lon_xd = lon_x - int(self.arrow * np.cos(angle))
			lat_yd = lat_y + int(self.arrow * np.sin(angle))

		else:
			angle = np.radians(heading-270)
			lon_xd = lon_x - int(self.arrow * np.cos(angle))
			lat_yd = lat_y - int(self.arrow * np.sin(angle)) 

		return lon_xd, lat_yd         

	def create_gnss_image(self):
		lon_x = int(self.img_width*(self.lon-LONG_ORIGIN)/self.lon_scale)
		lat_y = int(self.img_height-self.img_height*(self.lat-LAT_ORIGIN)/self.lat_scale)
		lon_xd, lat_yd = self.image_heading(lon_x, lat_y, self.heading)

		pub_image = np.copy(self.map_image)
		cv2.arrowedLine(pub_image, (lon_x, lat_y), (lon_xd, lat_yd), (0, 0, 255), 2)
		cv2.circle(pub_image, (lon_x, lat_y), 12, (0,0,255), 2)

		return pub_image

	# Conversion of GNSS heading to vehicle heading
	def heading_to_yaw_stanley(self, heading_curr):
		if (heading_curr >= 0 and heading_curr < 90):
			yaw_curr = np.radians(-heading_curr-90)
		else:
			yaw_curr = np.radians(-heading_curr+270)

		return round(yaw_curr, 4)

	def start(self):
		
		while not rospy.is_shutdown():

			# rviz visualization image
			pub_image = self.create_gnss_image()
			try:
				# Convert OpenCV image to ROS image and publish
				self.map_image_pub.publish(self.bridge.cv2_to_imgmsg(pub_image, "bgr8"))
			except CvBridgeError as e:
				rospy.logerr("CvBridge Error: {0}".format(e))

			self.rate.sleep()

def main():

		rospy.init_node('gem_gnss_image_node', anonymous=True)

		localizer = GNSSLocalizer()

		try:
			localizer.start()
		except KeyboardInterrupt:
			print ("Shutting down gnss image node.")
			cv2.destroyAllWindows()

if __name__ == '__main__':
		main()

