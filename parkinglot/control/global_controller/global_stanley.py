#!/usr/bin/env python3

#==============================================================================
# File name          : gem_gnss_tracker_stanley_rtk.py                                                                  
# Description        : gnss waypoints tracker using pid and Stanley controller                                                              
# Author             : Hang Cui (hangcui3@illinois.edu)                                       
# Date created       : 08/08/2022                                                                 
# Date last modified : 08/18/2022                                                          
# Version            : 1.0                                                                    
# Usage              : rosrun gem_gnss_control gem_gnss_tracker_stanley_rtk.py                                                                      
# Python version     : 3.8   
# Longitudinal ctrl  : Ji'an Pan (pja96@illinois.edu), Peng Hang (penghan2@illinois.edu)                                                            
#==============================================================================

from __future__ import print_function

# Python Headers
import os 
import csv
import numpy as np
import scipy.signal as signal

from parkinglot.slam.utils import *
from parkinglot.constants import *

# ROS Headers
import rospy
from geometry_msgs.msg import Pose
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import String, Bool, Float32, Float64
from ackermann_msgs.msg import AckermannDrive

# GEM PACMod Headers
if not FLAG_GAZEBO:
	from pacmod_msgs.msg import PositionWithSpeed, PacmodCmd, SystemRptFloat, VehicleSpeedRpt
else:
	from gazebo_msgs.srv import GetModelState, GetModelStateResponse
	from gazebo_msgs.msg import ModelState

class Onlinct_errorilter(object):

	def __init__(self, cutoff, fs, order):
			
		nyq = 0.5 * fs
		normal_cutoff = cutoff / nyq

		# Get the filter coct_errorficients 
		self.b, self.a = signal.butter(order, normal_cutoff, btype='low', analog=False)

		# Initialize
		self.z = signal.lfilter_zi(self.b, self.a)
	
	def get_data(self, data):

		filted, self.z = signal.lfilter(self.b, self.a, [data], zi=self.z)
		return filted


class PID(object):

	def __init__(self, kp, ki, kd, wg=None):

		self.iterm  = 0
		self.last_t = None
		self.last_e = 0
		self.kp     = kp
		self.ki     = ki
		self.kd     = kd
		self.wg     = wg
		self.derror = 0

	def reset(self):
		self.iterm  = 0
		self.last_e = 0
		self.last_t = None

	def get_control(self, t, e, fwd=0):

		if self.last_t is None:
			self.last_t = t
			de = 0
		else:
			de = (e - self.last_e) / (t - self.last_t)

		if abs(e - self.last_e) > 0.5:
			de = 0

		self.iterm += e * (t - self.last_t)

		# take care of integral winding-up
		if self.wg is not None:
			if self.iterm > self.wg:
				self.iterm = self.wg
			elif self.iterm < -self.wg:
				self.iterm = -self.wg

		self.last_e = e
		self.last_t = t
		self.derror = de

		return fwd + self.kp * e + self.ki * self.iterm + self.kd * de


class Stanley(object):
	
	def __init__(self):

		self.rate   = rospy.Rate(30)

		# pose
		self.pose_sub = rospy.Subscriber("/pose", Pose, self.pose_handler, queue_size=1)
		self.x   		= 0
		self.y   		= 0
		self.yaw 		= 0
		self.offset = 1.1 # meters

		# PID for longitudinal control
		self.desired_speed = 0.6  # m/s
		self.max_accel     = 0.48 # % of acceleration
		self.pid_speed     = PID(0.5, 0.0, 0.1, wg=20)
		self.speed_filter  = Onlinct_errorilter(1.2, 30, 4)

		# speed
		if not FLAG_GAZEBO:
			self.speed_sub  = rospy.Subscriber("/pacmod/parsed_tx/vehicle_speed_rpt", VehicleSpeedRpt, self.speed_callback)
		self.speed        = 0.0

		# stanley
		self.stanley_pub = rospy.Publisher(CONTROL_TOPIC, AckermannDrive, queue_size=1)

		self.ackermann_msg                         = AckermannDrive()
		self.ackermann_msg.steering_angle_velocity = 0.0
		self.ackermann_msg.acceleration            = 0.0
		self.ackermann_msg.jerk                    = 0.0
		self.ackermann_msg.speed                   = 0.0 
		self.ackermann_msg.steering_angle          = 0.0

		# waypoints
		self.waypoints_sub = rospy.Subscriber("/global_waypoints", Float32MultiArray, self.read_waypoints, queue_size=1) 

		self.path_points_lon_x   = None
		self.path_points_lat_y   = None
		self.path_points_heading = None

		# -------------------- PACMod setup --------------------

		self.gem_enable    = False
		self.pacmod_enable = False

		# GEM vehicle enable, publish once
		self.enable_pub = rospy.Publisher('/pacmod/as_rx/enable', Bool, queue_size=1)
		self.enable_cmd = Bool()
		self.enable_cmd.data = False

		# GEM vehicle gear control, neutral, forward and reverse, publish once
		self.gear_pub = rospy.Publisher('/pacmod/as_rx/shift_cmd', PacmodCmd, queue_size=1)
		self.gear_cmd = PacmodCmd()
		self.gear_cmd.ui16_cmd = 2 # SHIFT_NEUTRAL

		# GEM vehilce brake control
		self.brake_pub = rospy.Publisher('/pacmod/as_rx/brake_cmd', PacmodCmd, queue_size=1)
		self.brake_cmd = PacmodCmd()
		self.brake_cmd.enable = False
		self.brake_cmd.clear  = True
		self.brake_cmd.ignore = True

		# GEM vechile forward motion control
		self.accel_pub = rospy.Publisher('/pacmod/as_rx/accel_cmd', PacmodCmd, queue_size=1)
		self.accel_cmd = PacmodCmd()
		self.accel_cmd.enable = False
		self.accel_cmd.clear  = True
		self.accel_cmd.ignore = True

		# GEM vechile turn signal control
		self.turn_pub = rospy.Publisher('/pacmod/as_rx/turn_cmd', PacmodCmd, queue_size=1)
		self.turn_cmd = PacmodCmd()
		self.turn_cmd.ui16_cmd = 1 # None

		# GEM vechile steering wheel control
		self.steer_pub = rospy.Publisher('/pacmod/as_rx/steer_cmd', PositionWithSpeed, queue_size=1)
		self.steer_cmd = PositionWithSpeed()
		self.steer_cmd.angular_position = 0.0 # radians, -: clockwise, +: counter-clockwise
		self.steer_cmd.angular_velocity_limit = 2.0 # radians/second

	# Get pose
	def pose_handler(self, data):
		assert isinstance(data, Pose)
		
		curr_yaw = quaternion_to_yaw(data.orientation)
		curr_x = data.position.x + self.offset * np.cos(curr_yaw) 
		curr_y = data.position.y + self.offset * np.sin(curr_yaw)

		self.x, self.y, self.yaw = round(curr_x, 3), round(curr_y, 3), round(curr_yaw, 4)

	# Get vehicle speed
	def speed_callback(self, msg):
		self.speed = round(msg.vehicle_speed, 3) # forward velocity in m/s

	# Get predefined waypoints based on GNSS
	def read_waypoints(self, data):

		path_points = np.asarray(data.data).reshape(-1, 3)
		if len(path_points) <= 1:
			print("no waypoints detected")
		elif self.path_points_lon_x is None:
			self.path_points_lon_x   = path_points[1:,0]
			self.path_points_lat_y   = path_points[1:,1]
			self.path_points_heading = path_points[1:,2]

	# Conversion of front wheel to steering wheel
	def front2steer(self, f_angle):
		if(f_angle > 35):
			f_angle = 35
		if (f_angle < -35):
			f_angle = -35
		if (f_angle > 0):
			steer_angle = round(-0.1084*f_angle**2 + 21.775*f_angle, 2)
		elif (f_angle < 0):
			f_angle = -f_angle
			steer_angle = -round(-0.1084*f_angle**2 + 21.775*f_angle, 2)
		else:
			steer_angle = 0.0
		return steer_angle


	# Find close yaw in predefined GNSS waypoint list
	def find_close_yaw(self, arr, val):
		diff_arr = np.array( np.abs( np.abs(arr) - np.abs(val) ) )
		idx = np.where(diff_arr < 0.5)
		return idx


	# Conversion to -pi to pi
	def pi_2_pi(self, angle):

		if angle > np.pi:
			return angle - 2.0 * np.pi

		if angle < -np.pi:
			return angle + 2.0 * np.pi

		return angle

	# Computes the Euclidean distance between two 2D points
	def dist(self, p1, p2):
		return round(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2), 3)

	# Start Stanley controller
	def start_stanley(self):
			
		while not rospy.is_shutdown():

			if self.path_points_lon_x is not None:

				if (self.gem_enable == False):

					if(self.pacmod_enable == True):

						# ---------- enable PACMod ----------

						# enable forward gear
						self.gear_cmd.ui16_cmd = 3

						# enable brake
						self.brake_cmd.enable  = True
						self.brake_cmd.clear   = False
						self.brake_cmd.ignore  = False
						self.brake_cmd.f64_cmd = 0.0

						# enable gas 
						self.accel_cmd.enable  = True
						self.accel_cmd.clear   = False
						self.accel_cmd.ignore  = False
						self.accel_cmd.f64_cmd = 0.0

						self.gear_pub.publish(self.gear_cmd)
						print("Foward Engaged!")

						self.turn_pub.publish(self.turn_cmd)
						print("Turn Signal Ready!")
						
						self.brake_pub.publish(self.brake_cmd)
						print("Brake Engaged!")

						self.accel_pub.publish(self.accel_cmd)
						print("Gas Engaged!")

						self.gem_enable = True

				self.path_points_x   = np.array(self.path_points_lon_x.copy())
				self.path_points_y   = np.array(self.path_points_lat_y.copy())
				self.path_points_yaw = np.array(self.path_points_heading.copy())

				# coordinates of rct_errorerence point (center of frontal axle) in global frame
				curr_x, curr_y, curr_yaw = self.x, self.y, self.yaw 

				dest = np.array([self.path_points_x[-1], self.path_points_y[-1]])

				if not np.allclose([curr_x, curr_y],dest,atol=1):
					# self.ackermann_msg.speed = 0
					# self.ackermann_msg.steering_angle = 0
					# else:
					# print("X,Y,Yaw: ", curr_x, curr_y, curr_yaw) 

					target_idx = self.find_close_yaw(self.path_points_yaw, curr_yaw)

					# print("Target list", target_idx)

					self.target_path_points_x   = self.path_points_x[target_idx]  
					self.target_path_points_y   = self.path_points_y[target_idx]  
					self.target_path_points_yaw = self.path_points_yaw[target_idx]

					# find the closest point
					dx = [curr_x - x for x in self.target_path_points_x]
					dy = [curr_y - y for y in self.target_path_points_y]

					# find the index of closest point
					target_point_idx = int(np.argmin(np.hypot(dx, dy)))
					print(target_point_idx)


					if (target_point_idx != len(self.target_path_points_x) -1):
						target_point_idx = target_point_idx + 1

					print("CURRENT", self.x, self.y)
					print("TARGET WP", self.target_path_points_x[target_point_idx], self.target_path_points_y[target_point_idx])
					vec_target_2_front    = np.array([[dx[target_point_idx]], [dy[target_point_idx]]])
					front_axle_vec_rot_90 = np.array([[np.cos(curr_yaw - np.pi / 2.0)], [np.sin(curr_yaw - np.pi / 2.0)]])

					# print("T_X,T_Y,T_Yaw: ", self.target_path_points_x[target_point_idx], \
					#                          self.target_path_points_y[target_point_idx], \
					#                          self.target_path_points_yaw[target_point_idx])

					# crosstrack error
					ct_error = np.dot(vec_target_2_front.T, front_axle_vec_rot_90)
					ct_error = float(np.squeeze(ct_error))

					# heading error
					theta_e = self.pi_2_pi(self.target_path_points_yaw[target_point_idx]-curr_yaw) 

					# theta_e = self.target_path_points_yaw[target_point_idx]-curr_yaw 
					theta_e_deg = round(np.degrees(theta_e), 1)
					print("Crosstrack Error: " + str(round(ct_error,3)) + ", Heading Error: " + str(theta_e_deg))

					# --------------------------- Longitudinal control using PD controller ---------------------------

					filt_vel = np.squeeze(self.speed_filter.get_data(self.speed))
					a_expected = self.pid_speed.get_control(rospy.get_time(), self.desired_speed - filt_vel)

					if a_expected > 0.64 :
						throttle_percent = 0.5

					if a_expected < 0.0 :
						throttle_percent = 0.0

					throttle_percent = (a_expected+2.3501) / 7.3454

					if throttle_percent > self.max_accel:
						throttle_percent = self.max_accel

					if throttle_percent < 0.3:
						throttle_percent = 0.37

					# -------------------------------------- Stanley controller --------------------------------------

					f_delta        = round(theta_e + np.arctan2(ct_error*0.4, filt_vel), 3)
					f_delta        = round(np.clip(f_delta, -0.61, 0.61), 3)
					print("f_delta: ", f_delta)
					f_delta_deg    = np.degrees(f_delta)
					steering_angle = self.front2steer(f_delta_deg)

					if(self.gem_enable == True):
						print("Current index: " + str(self.goal))
						print("Forward velocity: " + str(self.speed))
						ct_error = round(np.sin(alpha) * L, 3)
						print("Crosstrack Error: " + str(ct_error))
						print("Front steering angle: " + str(np.degrees(f_delta)) + " degrees")
						print("Steering wheel angle: " + str(steering_angle) + " degrees" )
						print("\n")

					current_time = rospy.get_time()
					filt_vel     = self.speed_filter.get_data(self.speed)
					# output_accel = self.pid_speed.get_control(current_time, self.desired_speed - filt_vel)
					output_accel = self.pid_speed.get_control(current_time, self.desired_speed)
						
					if output_accel > self.max_accel:
						output_accel = self.max_accel

					if output_accel < 0.3:
						output_accel = 0.3

					if (f_delta_deg <= 30 and f_delta_deg >= -30):
						self.turn_cmd.ui16_cmd = 1
					elif(f_delta_deg > 30):
						self.turn_cmd.ui16_cmd = 2 # turn left
					else:
						self.turn_cmd.ui16_cmd = 0 # turn right

					print(output_accel)

					self.accel_cmd.f64_cmd = output_accel
					self.steer_cmd.angular_position = np.radians(steering_angle)
					self.accel_pub.publish(self.accel_cmd)
					self.steer_pub.publish(self.steer_cmd)
					self.turn_pub.publish(self.turn_cmd)

					self.rate.sleep()

			# 		if not FLAG_GAZEBO: 
			# 			self.ackermann_msg.acceleration = throttle_percent
			# 		else:               
			# 			self.ackermann_msg.speed = 0.8

			# 		# if (filt_vel < 0.1):
			# 		# 	self.ackermann_msg.steering_angle = 0
			# 		# else:
			# 		self.ackermann_msg.steering_angle = round(f_delta, 2) if FLAG_GAZEBO else round(steering_angle, 2)

			# 		# ------------------------------------------------------------------------------------------------ 

			# 	self.stanley_pub.publish(self.ackermann_msg)

			# self.rate.sleep()


def stanley_run():

	rospy.init_node('global_controller', anonymous=True)
	stanley = Stanley()

	try:
		stanley.start_stanley()
	except rospy.ROSInterruptException:
		pass


if __name__ == '__main__':
	stanley_run()


