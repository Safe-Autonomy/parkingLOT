import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
# from combined_thresh import combined_thresh
# from perspective_transform import perspective_transform

# feel free to adjust the parameters in the code if necessary

def bird_viz(binary_warped, mid_line_pts, save_file=None):
	"""
	Visualize the predicted lane lines with margin, on binary warped image
	save_file is a string representing where to save the image (if None, then just display)
	"""

	mid_line_pts += (binary_warped.shape[0] // 2, 0)

	# Create an image to draw on and an image to show the selection window
	out_img = (np.dstack((binary_warped, binary_warped, binary_warped))*255).astype('uint8')
	window_img = np.zeros_like(out_img)

	# cv2.line(out_img, (out_img.shape[0]//2, 0), (out_img.shape[0]//2, out_img.shape[1] - 1), (0, 0, 255), thickness = 3)
	# cv2.line(out_img, (40, 0), (40, out_img.shape[1] - 1), (255, 0, 0), thickness = 4)
	# cv2.line(out_img, (out_img.shape[0] - 40, 0), (out_img.shape[0] - 40, out_img.shape[1] - 1), (255, 255, 0), thickness = 4)



	# Color the middle lane
	for i in range(len(mid_line_pts)):
		cv2.circle(out_img, (int(mid_line_pts[i][0]), int(mid_line_pts[i][1])), 5, (0,255,0), -1)

	# Draw the lane onto the warped blank image
	result = cv2.addWeighted(out_img, 1, window_img, 1, 0)

	return result

def final_viz(undist,mid_line_pts, m_inv):
	"""
	Final lane line prediction visualized and overlayed on top of original image
	"""
	# # Generate x and y values for plotting
	# ploty = np.linspace(0, undist.shape[0]-1, undist.shape[0])
	# left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	# right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	# Create an image to draw the lines on
	#warp_zero = np.zeros_like(warped).astype(np.uint8)
	#color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
	color_warp = np.zeros((720, 1280, 3), dtype='uint8')  # NOTE: Hard-coded image dimensions

	# # Recast the x and y points into usable format for cv2.fillPoly()
	# pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	# pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	# pts = np.hstack((pts_left, pts_right))

	# # Draw the lane onto the warped blank image
	# cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

	# draw waypoints as circles 
	for i in range(len(mid_line_pts)):
		cv2.circle(color_warp, (int(mid_line_pts[i][0]), int(mid_line_pts[i][1])), 5, (255,0,0), -1)

	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	newwarp = cv2.warpPerspective(color_warp, m_inv, (undist.shape[1], undist.shape[0]))
	# Combine the result with the original image
	# Convert arrays to 8 bit for later cv to ros image transfer
	undist = np.array(undist, dtype=np.uint8)
	newwarp = np.array(newwarp, dtype=np.uint8)
	result = cv2.addWeighted(undist, 1, newwarp, 1, 0)

	return result


def fit_one_lane(labeled_lanes,num_points=10):
	lane = np.zeros_like(labeled_lanes)
	lane[labeled_lanes == 1] = 1
	
	#if left or right #TODO: decide lane side
	# take only the bottom 1/5 of the image for ref
	height,width = labeled_lanes.shape[:2]
	left_view = np.zeros_like(labeled_lanes)
	right_view = np.zeros_like(labeled_lanes)

	left_view[int(4*height/5):,:int(2*width/4)] = lane[int(4*height/5):,:int(2*width/4)]
	right_view[int(4*height/5):,int(2*width/4):] = lane[int(4*height/5):,int(2*width/4):]  

	lane_is_right = np.sum(left_view) < np.sum(right_view)
	# for i in range(left_view.shape[0]-1, 0, -1):
	# 	if np.max(labeled_lanes[i]) == 0: continue

	# 	idx = np.argmax(labeled_lanes)
	# 	lane_is_right = ( idx >= (width / 2) )
	# 	break

	offset = np.array([200, 0])

	nonzero_lane = lane.nonzero()
	lane_y = np.array(nonzero_lane[0])
	lane_x = np.array(nonzero_lane[1])

	try:
		lane_fit = np.polyfit(lane_y, lane_x, 2)
	####
	except TypeError:
		print("Unable to detect lane")

	# find waypoints
	ploty = np.linspace(0, labeled_lanes.shape[0]-1, num_points)
	lane_fitx = lane_fit[0]*ploty**2 + lane_fit[1]*ploty + lane_fit[2]
	mid_fitx = lane_fitx - offset[0] if lane_is_right else lane_fitx + offset[0]
	mid_line_pts = np.array([np.transpose(np.vstack([mid_fitx, ploty]))])[0]

	return mid_line_pts

def fit_two_lane(labeled_lanes, num_points=10):
	unique, counts = np.unique(labeled_lanes, return_counts=True)

	lane1 = np.zeros_like(labeled_lanes)
	lane2 = np.zeros_like(labeled_lanes)
	lane1[labeled_lanes == 1] = 1
	lane2[labeled_lanes == 2] = 1

	if 0.5 * counts[1] > counts[2]:
		return fit_one_lane(lane1)
	elif 0.5 * counts[2] > counts[1]:
		return fit_one_lane(lane2)

	# find x and y coodiate for two lane
	nonzero_lane1 = lane1.nonzero()
	lane1_y = np.array(nonzero_lane1[0])
	lane1_x = np.array(nonzero_lane1[1])

	nonzero_lane2 = lane2.nonzero()
	lane2_y = np.array(nonzero_lane2[0])
	lane2_x = np.array(nonzero_lane2[1])

	try:
		lane1_fit = np.polyfit(lane1_y, lane1_x, 2)
	####
	except TypeError:
		print("Unable to detect lane 1")
	try:
		lane2_fit = np.polyfit(lane2_y, lane2_x, 2)
	####
	except TypeError:
		print("Unable to detect lanes 2")

	
	# find waypoints
	ploty = np.linspace(0, labeled_lanes.shape[0]-1, num_points)
	lane1_fitx = lane1_fit[0]*ploty**2 + lane1_fit[1]*ploty + lane1_fit[2]
	lane2_fitx = lane2_fit[0]*ploty**2 + lane2_fit[1]*ploty + lane2_fit[2]
	mid_fitx = (lane1_fitx + lane2_fitx)/2.
	mid_line_pts = np.array([np.transpose(np.vstack([mid_fitx, ploty]))])[0]

	return mid_line_pts

def fit_largest_lane(labeled_lanes, num_points=10):
	unique, counts = np.unique(labeled_lanes, return_counts=True)
	left_label = unique[np.argsort(counts)[-2]]
	right_label = unique[np.argsort(counts)[-3]]
	
		

	lane1 = np.zeros_like(labeled_lanes)
	lane2 = np.zeros_like(labeled_lanes)
	lane1[labeled_lanes == left_label] = 1
	lane2[labeled_lanes == right_label] = 1

	# find x and y coodiate for two lane
	nonzero_lane1 = lane1.nonzero()
	lane1_y = np.array(nonzero_lane1[0])
	lane1_x = np.array(nonzero_lane1[1])

	nonzero_lane2 = lane2.nonzero()
	lane2_y = np.array(nonzero_lane2[0])
	lane2_x = np.array(nonzero_lane2[1])

	try:
		lane1_fit = np.polyfit(lane1_y, lane1_x, 2)
	####
	except TypeError:
		print("Unable to detect lane 1")
	try:
		lane2_fit = np.polyfit(lane2_y, lane2_x, 2)
	####
	except TypeError:
		print("Unable to detect lanes 2")

	
	# find waypoints
	ploty = np.linspace(0, labeled_lanes.shape[0]-1, num_points)
	lane1_fitx = lane1_fit[0]*ploty**2 + lane1_fit[1]*ploty + lane1_fit[2]
	lane2_fitx = lane2_fit[0]*ploty**2 + lane2_fit[1]*ploty + lane2_fit[2]
	mid_fitx = (lane1_fitx + lane2_fitx)/2.
	mid_line_pts = np.array([np.transpose(np.vstack([mid_fitx, ploty]))])[0]

	return mid_line_pts