import os
import numpy as np

import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError

from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2

from parkinglot.slam.utils import *
from parkinglot.constants import *

class LidarMapper(object):
	def __init__(self):
		self.rate = rospy.Rate(20)

		self.bridge = CvBridge()

		# map image
		self.map_img_width      = 2107
		self.map_img_height     = 1313
		self.map_img_lat_scale  = 0.00062    # 0.0007    
		self.map_img_lon_scale  = 0.00136    # 0.00131   

		self.map_image = np.zeros((self.map_img_height, self.map_img_width)).astype('uint8')
		self.current_point_image = np.zeros_like(self.map_image)
		self.map_image_pub = rospy.Publisher("/grid_image", Image, queue_size=1) 

		# gnss map
		curr_path = os.path.abspath(__file__) 
		image_path = curr_path.split('mapping')[0] + 'image/gnss_map.png'
		self.gnss_map = cv2.imread(image_path)

		# occupancy grid 
		

		# point cloud
		self.point_sub = rospy.Subscriber("/lidar1/velodyne_points", PointCloud2, self.cloud_handler, queue_size=1)
		self.current_points = None

		# pose
		self.pose_sub = rospy.Subscriber("/pose", Pose, self.pose_handler, queue_size=1)
		self.x   = None
		self.y   = None
		self.yaw = None
		
		# extrinsic
		self.gps_to_lidar_offset = 0

	def cloud_handler(self, data):
		assert isinstance(data, PointCloud2)

		# read + filter raw lidar frame in BEV
		gen = point_cloud2.readgen = point_cloud2.read_points(cloud=data, field_names=('x', 'y', 'z', 'ring'))
		points = []
		for p in gen:
			points.append([p[0], p[1], 0, 1])
		self.current_points = np.vstack(points)

		# convert lidar points lidar frame -> world frame
		self.current_points = self.transform_points_to_world(self.current_points)[:,:2]

		# print(self.current_points[:10])

		# create a debug map image
		current_point_map = self.get_current_points_image()
		self.map_image[current_point_map == 255] = 255

	def pose_handler(self, data):
		assert isinstance(data, Pose)
		
		curr_yaw = quaternion_to_yaw(data.orientation)
		curr_x = data.position.x + self.gps_to_lidar_offset * np.cos(curr_yaw) 
		curr_y = data.position.y + self.gps_to_lidar_offset * np.sin(curr_yaw)

		self.x, self.y, self.yaw = round(curr_x, 3), round(curr_y, 3), round(curr_yaw, 4)
	
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
		lon, lat = global_xy_to_gnss(points[:,0], points[:,1])

		# convert lidar points world gnss frame -> imagge frame  
		lon = (self.map_img_width * (lon - LONG_ORIGIN) / self.map_img_lon_scale).astype(int)
		lat = (self.map_img_height - self.map_img_height * (lat - LAT_ORIGIN) / self.map_img_lat_scale).astype(int)

		# print(lon, lat)

		# dont plots points out of bound
		mask = np.where(np.logical_and(lon < self.map_img_width, lon >= 0), 1, 0)
		mask &= np.where(np.logical_and(lat < self.map_img_height, lat >= 0), 1, 0)

		lon = lon[mask==1]
		lat = lat[mask==1]

		out_img = self.current_point_image.copy()
		out_img[lat, lon] = 255

		return out_img

	def start(self):
		while not rospy.is_shutdown():

			# publish debug image
			if self.map_image is not None:
				try:
					# Convert OpenCV image to ROS image and publish
					debug_img = self.gnss_map.copy()
					debug_img[self.map_image == 255] = [0,0,255]
					self.map_image_pub.publish(self.bridge.cv2_to_imgmsg(debug_img, "bgr8"))
				except CvBridgeError as e:
					rospy.logerr("CvBridge Error: {0}".format(e))

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


		
	
	
		
