import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
# from combined_thresh import combined_thresh
# from perspective_transform import perspective_transform

# feel free to adjust the parameters in the code if necessary

def line_fit(binary_warped):
	"""
	Find and fit lane lines
	"""
	# Assuming you have created a warped binary image called "binary_warped"
	# Take a histogram of the bottom half of the image
	histogram = np.sum(binary_warped[2*binary_warped.shape[0]//3:,:], axis=0)

	# Create an output image to draw on and visualize the result
	out_img = (np.dstack((binary_warped, binary_warped, binary_warped))*255).astype('uint8')
	# Find the peak of the left and right halves of the histogram
	# These will be the starting point for the left and right lines
	midpoint = np.int64(histogram.shape[0]/2)
	leftx_base = np.argmax(histogram[100:midpoint]) + 100
	rightx_base = np.argmax(histogram[midpoint:-100]) + midpoint

	# print("left box base:", leftx_base)
	# print("right box base:",rightx_base)

	# Choose the number of sliding windows
	nwindows = 11
	# Set height of windows
	window_height = np.int64(binary_warped.shape[0]/nwindows)
	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	# Current positions to be updated for each window
	leftx_current = leftx_base
	rightx_current = rightx_base
	# Set the width of the windows +/- margin
	# NOTE: smaller window margin 
	margin = 50	# default: 100
	# Set minimum number of pixels found to recenter window
	minpix = 50
	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []

	# get the height and width of the binary_warped img
	height,width = binary_warped.shape


	# Step through the windows one by one
	for window in range(nwindows):
		# Identify window boundaries in x and y (and right and left)

		y_up = height - window*window_height		# high boundary in y axis
		y_bottom = height - (window+1)*window_height	# low boundary in y axis
		xleft_min = leftx_current - margin			# left boundary in x axis (left lane)
		xleft_max = leftx_current + margin			# right boundary in x axis (left lane)
		xright_min = rightx_current - margin		# left boundary in x axis (right lane)
		xright_max = rightx_current + margin		# right boundary in x axis (right lane)
		
		####
		# Draw the windows on the visualization image using cv2.rectangle()
		# left window
		cv2.rectangle(out_img, (xleft_min, y_up), (xleft_max, y_bottom), (0,255,0), 2)		# (top left, right bottom) green color, thickness 2
		# right window
		cv2.rectangle(out_img, (xright_min, y_up), (xright_max, y_bottom), (0,0,255), 2)			# (top left, right bottom) green color, thickness 2
		####
		# Identify the nonzero pixels in x and y within the window
		# left window
		left_window_inds = ((nonzerox >= xleft_min) & (nonzerox < xleft_max) & (nonzeroy >= y_bottom) & (nonzeroy < y_up)).nonzero()[0]
		# right window
		right_window_inds = ((nonzerox >= xright_min) & (nonzerox < xright_max) & (nonzeroy >= y_bottom) & (nonzeroy < y_up)).nonzero()[0]

		####
		# Append these indices to the lists
		left_lane_inds.append(left_window_inds)
		right_lane_inds.append(right_window_inds)

		####
		# If you found > minpix pixels, recenter next window on their mean position
		# left window
		if len(left_window_inds) > minpix:
			leftx_current = np.int64(np.mean(nonzerox[left_window_inds]))
		# right window
		if len(right_window_inds) > minpix:
			rightx_current = np.int64(np.mean(nonzerox[right_window_inds]))

		####

	# Concatenate the arrays of indices
	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)

	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds]
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]

	# Fit a second order polynomial to each using np.polyfit()
	# If there isn't a good fit, meaning any of leftx, lefty, rightx, and righty are empty,
	# the second order polynomial is unable to be solved.
	# Thus, it is unable to detect edges.
	try:
		left_fit = np.polyfit(lefty, leftx, 2)
	####
	except TypeError:
		print("Unable to detect left anes")
		return None
	try:
		right_fit = np.polyfit(righty, rightx, 2)
	####
	except TypeError:
		print("Unable to detect right lanes")
		return None
	
	# find waypoints
	ploty = np.linspace(0, binary_warped.shape[0]-1, 10)
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
	mid_fitx = (left_fitx + right_fitx)/2.
	mid_line_pts = np.array([np.transpose(np.vstack([mid_fitx, ploty]))])

	# Return a dict of relevant variables
	ret = {}
	ret['left_fit'] = left_fit
	ret['right_fit'] = right_fit
	ret['waypoints'] = mid_line_pts[0]
	ret['nonzerox'] = nonzerox
	ret['nonzeroy'] = nonzeroy
	ret['out_img'] = out_img
	ret['left_lane_inds'] = left_lane_inds
	ret['right_lane_inds'] = right_lane_inds

	return ret


def tune_fit(binary_warped, left_fit, right_fit):
	"""
	Given a previously fit line, quickly try to find the line based on previous lines
	"""
	# Assume you now have a new warped binary image
	# from the next frame of video (also called "binary_warped")
	# It's now much easier to find line pixels!
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	margin = 20
	left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
	right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

	# Again, extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds]
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]

	# If we don't find enough relevant points, return all None (this means error)
	min_inds = 10
	if lefty.shape[0] < min_inds or righty.shape[0] < min_inds:
		return None

	# Fit a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)
	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	# find waypoints
	mid_fity = np.linspace(0, binary_warped.shape[0]-1, 10)
	left_fitx = left_fit[0]*mid_fity**2 + left_fit[1]*mid_fity + left_fit[2]
	right_fitx = right_fit[0]*mid_fity**2 + right_fit[1]*mid_fity + right_fit[2]
	mid_fitx = (left_fitx + right_fitx)/2.
	mid_line_pts = np.array([np.transpose(np.vstack([mid_fitx, mid_fity]))])
	# Return a dict of relevant variables
	ret = {}
	ret['left_fit'] = left_fit
	ret['right_fit'] = right_fit
	ret['waypoints'] = mid_line_pts[0]
	ret['nonzerox'] = nonzerox
	ret['nonzeroy'] = nonzeroy
	ret['left_lane_inds'] = left_lane_inds
	ret['right_lane_inds'] = right_lane_inds

	return ret


def viz1(binary_warped, ret, save_file=None):
	"""
	Visualize each sliding window location and predicted lane lines, on binary warped image
	save_file is a string representing where to save the image (if None, then just display)
	"""
	# Grab variables from ret dictionary
	left_fit = ret['left_fit']
	right_fit = ret['right_fit']
	nonzerox = ret['nonzerox']
	nonzeroy = ret['nonzeroy']
	out_img = ret['out_img']
	left_lane_inds = ret['left_lane_inds']
	right_lane_inds = ret['right_lane_inds']

	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
	plt.imshow(out_img)
	plt.plot(left_fitx, ploty, color='yellow')
	plt.plot(right_fitx, ploty, color='yellow')
	plt.xlim(0, 1280)
	plt.ylim(720, 0)
	if save_file is None:
		plt.show()
	else:
		plt.savefig(save_file)
	plt.gcf().clear()


def bird_fit(binary_warped, ret, save_file=None):
	"""
	Visualize the predicted lane lines with margin, on binary warped image
	save_file is a string representing where to save the image (if None, then just display)
	"""
	# Grab variables from ret dictionary
	left_fit = ret['left_fit']
	right_fit = ret['right_fit']
	nonzerox = ret['nonzerox']
	nonzeroy = ret['nonzeroy']
	mid_line_pts = ret['waypoints']
	left_lane_inds = ret['left_lane_inds']
	right_lane_inds = ret['right_lane_inds']

	# Create an image to draw on and an image to show the selection window
	out_img = (np.dstack((binary_warped, binary_warped, binary_warped))*255).astype('uint8')
	window_img = np.zeros_like(out_img)
	# Color in left and right line pixels
	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]


	# Generate a polygon to illustrate the search window area
	# And recast the x and y points into usable format for cv2.fillPoly()
	margin = 50  # NOTE: Keep this in sync with *_fit()
	left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
	left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
	left_line_pts = np.hstack((left_line_window1, left_line_window2))
	right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
	right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
	right_line_pts = np.hstack((right_line_window1, right_line_window2))


	# Draw the lane onto the warped blank image
	cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
	cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
	cv2.fillPoly(window_img, np.int_([mid_line_pts]), (255,0, 0))
	result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

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
	lane[labeled_lanes != 0 ] = 1
	cv2.imwrite('visualization/one_lane.png',lane*255)



def fit_two_lane(labeled_lanes, num_points=10):
	lane1 = np.zeros_like(labeled_lanes)
	lane2 = np.zeros_like(labeled_lanes)
	lane1[labeled_lanes == 1] = 1
	lane2[labeled_lanes == 2] = 1

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



