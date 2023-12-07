#!/usr/bin/env python3

import os
import cv2 
import numpy as np

import rospy
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from parkinglot.slam.utils import *
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

		# Subscribe information from sensors
		self.lat      = 0
		self.lon      = 0
		self.heading  = 0
		# if not FLAG_GAZEBO:
		self.gnss_sub = rospy.Subscriber("/novatel/inspva", Inspva, self.inspva_callback)

		self.arrow = 40 

	def inspva_callback(self, inspva_msg):
		self.lat     = inspva_msg.latitude
		self.lon     = inspva_msg.longitude
		self.heading = inspva_msg.azimuth 

		# pose
		curr_x, curr_y = gnss_to_global_xy(self.lon, self.lat)
		quat = yaw_to_quaternion(self.heading_to_yaw_stanley(self.heading)) 
		self.pose_msg.position.x = round(curr_x, 3)
		self.pose_msg.position.y = round(curr_y, 3)
		self.pose_msg.orientation = quat

		print(curr_x, curr_y)

		self.pose_pub.publish(self.pose_msg)    

	# Conversion of GNSS heading to vehicle heading
	def heading_to_yaw_stanley(self, heading_curr):
		if (heading_curr >= 0 and heading_curr < 90):
			yaw_curr = np.radians(-heading_curr-90)
		else:
			yaw_curr = np.radians(-heading_curr+270)

		return round(yaw_curr, 4)

	def getModelState(self):
		rospy.wait_for_service('/gazebo/get_model_state')
		try:
			serviceResponse = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
			resp = serviceResponse(model_name='gem')
		except rospy.ServiceException as exc:
			rospy.loginfo("Service did not process request: "+str(exc))
			resp = GetModelStateResponse()
			resp.success = False
		return resp

	def start(self):
		
		while not rospy.is_shutdown():
			# for GAZEBO ONLY
			if FLAG_GAZEBO:
				current_pose = self.getModelState()

				if current_pose.success:
					self.pose_pub.publish(current_pose.pose)

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

