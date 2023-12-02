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
import numpy as np
import math
import scipy.signal as signal

# ROS Headers
import rospy
from ackermann_msgs.msg import AckermannDrive
from std_msgs.msg import Float32MultiArray

from parkinglot.constants import *

if not FLAG_GAZEBO:
    from pacmod_msgs.msg import VehicleSpeedRpt
else:
    from gazebo_msgs.srv import GetModelState, GetModelStateResponse
    from gazebo_msgs.msg import ModelState

class OnlineFilter(object):

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

class Local_Stanley(object):
    
    def __init__(self):

        self.rate   = rospy.Rate(30)

        # self.olat   = 40.0928232 
        # self.olon   = -88.2355788
        self.offset = 1.1 # meters

        # PID for longitudinal control
        self.desired_speed = 0.6  # m/s
        self.max_accel     = 0.48 # % of acceleration
        self.pid_speed     = PID(0.5, 0.0, 0.1, wg=20)
        self.speed_filter  = OnlineFilter(1.2, 30, 4)

        ######### SUBSCRIBERS ######### 
        if not FLAG_GAZEBO:
            # Speed from GEM
            self.speed_sub  = rospy.Subscriber("/pacmod/parsed_tx/vehicle_speed_rpt", VehicleSpeedRpt, self.speed_callback)
        self.speed      = 0.0

        # local waypoints from percception
        self.centerline_sub = rospy.Subscriber("lane_detection/centerline_points", Float32MultiArray, self.read_waypoints, queue_size=1)

        ######### PUBLISHERS ######### 
        # control 
        self.stanley_pub = rospy.Publisher(CONTROL_TOPIC, AckermannDrive, queue_size=1)

        self.ackermann_msg                         = AckermannDrive()
        self.ackermann_msg.steering_angle_velocity = 0.0
        self.ackermann_msg.acceleration            = 0.0
        self.ackermann_msg.jerk                    = 0.0
        self.ackermann_msg.speed                   = 0.0 
        self.ackermann_msg.steering_angle          = 0.0

        self.unprocessed_path_points_x = None
        self.unprocessed_path_points_y = None
        self.unprocessed_path_points_heading = None

    # Get vehicle speed
    def speed_callback(self, msg):
        self.speed = round(msg.vehicle_speed, 3) # forward velocity in m/s

    # Get value of steering wheel
    def steer_callback(self, msg):
        self.steer = round(np.degrees(msg.output),1)

    def planner_callback(self, msg):
        # TODO
        pass


    # Get predefined waypoints based on local planner(perception)
    def read_waypoints(self, path_points):
        path_points = np.asarray(path_points.data).reshape(-1, 2)
        if len(path_points) <= 1:
            print("no waypoints detected")
            self.unprocessed_path_points_x = np.array([0])
            self.unprocessed_path_points_y = np.array([10])
            self.unprocessed_path_points_heading = np.array([0])
        else:
            # x towards East and y towards North
            # self.path_points_x   = path_points[0,:] # longitude
            # self.path_points_y   = path_points[1,:] # latitude
            self.unprocessed_path_points_x   = path_points[:,0] # longitude
            self.unprocessed_path_points_y   = path_points[:,1] # latitude
            self.unprocessed_path_points_heading = np.arctan2(self.unprocessed_path_points_x, self.unprocessed_path_points_y)
            # print("point x: ", self.path_points_x)  
            # print("point y: ", self.path_points_y)  
            # print("point headings: ", self.path_points_heading)                                                                                                                                                                                                                                                                                                    # heading

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

    # for GAZEBO ONLY
    def get_gem_gazebo_speed(self):
        # Get the current state of the vehicle
        # Input: None
        # Output: ModelState, the state of the vehicle, contain the
        #   position, orientation, linear velocity, angular velocity
        #   of the vehicle
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            serviceResponse = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            resp = serviceResponse(model_name='gem')
        except rospy.ServiceException as exc:
            rospy.loginfo("Service did not process request: "+str(exc))
            resp = GetModelStateResponse()
            resp.success = False

        
        return resp

    # Get vehicle states: x, y, yaw
    def get_gem_state(self):

        # needs to define all of these shit in local frame
        if FLAG_GAZEBO:
            state = self.get_gem_gazebo_speed()

            if state.success:
                twist = state.twist
                self.speed = math.sqrt(twist.linear.x ** 2 + twist.linear.y ** 2)


        # vehicle gnss heading (yaw) in degrees
        # vehicle x, y position in fixed local frame, in meters
        # rct_errorerence point is located at the center of GNSS antennas
        # local_x_curr, local_y_curr = 0, 0 #TODO: get local way points from local planner(perception)

        # heading to yaw (degrees to radians)
        # heading is calculated from two GNSS antennas
        curr_yaw = 0 

        # rct_errorerence point is located at the center of front axle
        # curr_x = local_x_curr + self.offset * np.cos(curr_yaw)
        # curr_y = local_y_curr + self.offset * np.sin(curr_yaw)
        curr_x = 0
        curr_y = 0

        return round(curr_x, 3), round(curr_y, 3), round(curr_yaw, 4)

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
            if self.unprocessed_path_points_x is None:
                continue
            self.path_points_x   = np.array(self.unprocessed_path_points_x)
            self.path_points_y   = np.array(self.unprocessed_path_points_y)
            self.path_points_yaw = np.array(self.unprocessed_path_points_heading)

            # coordinates of rct_errorerence point (center of frontal axle) in global frame
            curr_x, curr_y, curr_yaw = self.get_gem_state()

            print("[DEBUG] GEM X,Y,Yaw: ", curr_x, curr_y, curr_yaw)

            target_idx = 0 #self.find_close_yaw(self.path_points_yaw, curr_yaw)

            # print("Target list", target_idx)

            close_ignore_thresh = 0

            self.target_path_points_x   = self.path_points_x[close_ignore_thresh:]
            self.target_path_points_y   = self.path_points_y[close_ignore_thresh:]
            self.target_path_points_yaw = self.path_points_yaw[close_ignore_thresh:]

            # find the closest point
            dx = [curr_x - x for x in self.target_path_points_x]
            dy = [curr_y - y for y in self.target_path_points_y]

            # find the index of closest point
            target_point_idx = int(np.argmin(np.hypot(dx, dy)))

            target_point_idx = int(np.argmin(np.hypot(dx, dy)))

            print("Target Point Index", target_point_idx + close_ignore_thresh)


            if (target_point_idx != len(self.target_path_points_x) -1):
                target_point_idx = target_point_idx + 1


            vec_target_2_front    = np.array([[dx[target_point_idx]], [dy[target_point_idx]]])
            # front_axle_vec_rot_90 = np.array([[np.cos(curr_yaw - np.pi / 2.0)], [np.sin(curr_yaw - np.pi / 2.0)]])
            front_axle_vec_rot_0 = np.array([[np.cos(curr_yaw)], [np.sin(curr_yaw)]])


            # print("T_X,T_Y,T_Yaw: ", self.target_path_points_x[target_point_idx], \
            #                          self.target_path_points_y[target_point_idx], \
            #                          self.target_path_points_yaw[target_point_idx])

            # crosstrack error
            # ct_error = np.dot(vec_target_2_front.T, front_axle_vec_rot_90)
            ct_error = np.dot(vec_target_2_front.T, front_axle_vec_rot_0)

            ct_error = float(np.squeeze(ct_error))

            print("Target Point Yaw", self.target_path_points_yaw[target_point_idx])
            print("Current Yaw", curr_yaw)

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

            f_delta        = round(theta_e + np.arctan2(ct_error*10, filt_vel), 3)
            print("f_delta: ", f_delta)
            f_delta        = round(np.clip(f_delta, -0.61, 0.61), 3)
            f_delta_deg    = np.degrees(f_delta)
            steering_angle = self.front2steer(f_delta_deg)
            print("steer agnle", steering_angle)

            # if (filt_vel < 0.2):
            #     self.ackermann_msg.acceleration   = throttle_percent
            #     self.ackermann_msg.steering_angle = 0
            #     print(self.ackermann_msg.acceleration, self.ackermann_msg.steering_angle)
            # else:
            #     self.ackermann_msg.acceleration   = throttle_percent
            #     self.ackermann_msg.steering_angle = round(steering_angle,1)
            #     print(self.ackermann_msg.acceleration, self.ackermann_msg.steering_angle)

            self.ackermann_msg.speed = 0.5
            print("VEL", filt_vel)
            # self.ackermann_msg.speed = 0
            
            self.ackermann_msg.steering_angle = round(steering_angle,5)
            # self.ackermann_msg.steering_angle = 0

            # ------------------------------------------------------------------------------------------------ 

            self.stanley_pub.publish(self.ackermann_msg)

            self.rate.sleep()


def stanley_run():

    rospy.init_node('local_planner', anonymous=True)
    stanley = Local_Stanley()

    try:
        stanley.start_stanley()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    stanley_run()

