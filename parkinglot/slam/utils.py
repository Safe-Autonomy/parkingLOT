from geometry_msgs.msg import Quaternion
import numpy as np
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import parkinglot.slam.alvinxy as axy 

from parkinglot.constants import *

# yaw to quaternion
def yaw_to_quaternion(yaw):
	assert isinstance(yaw, float)
	
	yaw = round(yaw, 4)
	quat = quaternion_from_euler(0, 0, yaw) 
	quat_msg = Quaternion(quat[1], quat[2], quat[3], quat[0])

	return quat_msg

# quaternion to yaw
def quaternion_to_yaw(quat):
	assert isinstance(quat, Quaternion)
	_, _, yaw = euler_from_quaternion([quat.w, quat.x, quat.y, quat.z])

	return yaw

# gnss long, lat to world x, y
def gnss_to_global_xy(lon, lat):
	x, y = axy.ll2xy(lat, lon, LAT_ORIGIN, LONG_ORIGIN)
	return -x, -y   

# gnss long, lat to world x, y
def global_xy_to_gnss(x, y):
	lat, lon = axy.xy2ll(-x, -y, LAT_ORIGIN, LONG_ORIGIN)
	return lon, lat

# x,y,yaw -> 4x4 trans matrix
def trans_from_x_y_yaw(x, y, yaw):
	return np.array([[np.cos(yaw), -np.sin(yaw), 0,    x],
									 [np.sin(yaw), np.cos(yaw) , 0,    y],
									 [0          , 0           , 1,    0],
									 [0          , 0           , 0,    1]])

# gnss to image
def gnss_to_image(lon, lat):
	x = MAP_IMG_WIDTH * (lon - LONG_ORIGIN) / MAP_IMG_LON_SCALE
	y = MAP_IMG_HEIGHT - MAP_IMG_HEIGHT * (lat - LAT_ORIGIN) / MAP_IMG_LAT_SCALE
	
	if isinstance(x, float):
		x, y = int(x), int(y)
	else:
		x, y = x.astype(int), y.astype(int)
	return x, y 

# global xy to image
def global_xy_to_image(x,y):
	lon, lat = global_xy_to_gnss(x, y)
	return gnss_to_image(lon, lat)