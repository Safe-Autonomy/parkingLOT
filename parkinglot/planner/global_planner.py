#!/usr/bin/env python3

import numpy as np

import rospy

from geometry_msgs.msg import Pose
from std_msgs.msg import Float32MultiArray

from parkinglot.slam.utils import *
from parkinglot.constants import *
from hybridastar.hybridastar import hybrid_a_star_planning

class Visualizer(object):
	def __init__(self):
		self.rate = rospy.Rate(2)

		# obstacles
		self.obstacles_sub = rospy.Subscriber("/obstacle", Float32MultiArray, self.obstacle_handler, queue_size=1) 
		self.obstacle_x = [0]
		self.obstacle_y = [0]

		# pose
		self.pose_sub = rospy.Subscriber("/pose", Pose, self.pose_handler, queue_size=1)
		self.x   = None
		self.y   = None
		self.yaw = None
		
		# extrinsic
		self.gps_to_lidar_offset = 0

		# waypoints 
		self.waypoints_pub = rospy.Publisher("/global_waypoints", Float32MultiArray, queue_size=1) 

		# start = [-23.203, -6.018, -1.95021]
		if not FLAG_GAZEBO:
			self.goal = [-56.741, -3.935, 2.207]
		else:
			self.goal = [-17.5, -30.95, -1.2]

		# roi 
		self.roi_x_min = -23
		self.roi_x_max = 23
		self.roi_y_min = -12
		self.roi_y_max = 2

	def obstacle_handler(self, data):
		assert isinstance(data, Float32MultiArray)
		obstacles = np.asarray(data.data).reshape(-1, 2)

		# mask = np.where(np.logical_and(obstacles[:,0] < self.roi_x_max, obstacles[:,0] >= self.roi_x_min), 1, 0)
		obstacles = np.round(obstacles, 0)
		obstacles = np.unique(obstacles, axis=1) 
		mask = np.where(obstacles[:,1] > self.roi_y_min, 1, 0)

		obstacles = obstacles[mask == 1]
  
		self.obstacle_x = obstacles[:,0].tolist()
		self.obstacle_y = obstacles[:,1].tolist()

	def pose_handler(self, data):
		assert isinstance(data, Pose)
		
		curr_yaw = quaternion_to_yaw(data.orientation)
		curr_x = data.position.x + self.gps_to_lidar_offset * np.cos(curr_yaw) 
		curr_y = data.position.y + self.gps_to_lidar_offset * np.sin(curr_yaw)

		self.x, self.y, self.yaw = round(curr_x, 3), round(curr_y, 3), round(curr_yaw, 4)
		
	def start(self):
		while not rospy.is_shutdown():

			if self.x is not None:
				start = [self.x, self.y, self.yaw]
				print(self.x, self.y, self.yaw)

				path = hybrid_a_star_planning(start=start,
											goal=self.goal,
											ox=[0],
											oy=[0],
											xy_resolution=0.05,
											yaw_resolution=np.deg2rad(1.0))
				
				if path is not None:
					xs = np.array(path.x_list)
					ys = np.array(path.y_list)
					yaws = np.array(path.yaw_list)
					waypoints = np.vstack((xs, ys, yaws)).T

					array = Float32MultiArray()
					array.data = waypoints.flatten().tolist()
					self.waypoints_pub.publish(array)

				self.rate.sleep()

def main():

		rospy.init_node('visualization', anonymous=True)
		viz = Visualizer()
		viz.start()

if __name__ == '__main__':
		main()


		
	
	
		
