import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle


def perspective_transform(img, verbose=False):
	"""
	Get bird's eye view from input image
	points for transformation in this format:
		[bottom-left, bottom-right, top-left, top-right]
	"""

	bottom_left = [0,700]
	bottom_right = [1280,700]
	top_left = [50,500]
	top_right = [1200,500]

	# Define 4 source points and 4 destination points
	src = np.float32([bottom_left,bottom_right,top_left,top_right])
	dst = np.float32([[0, 480], [640, 480], [0, 0], [640, 0]])

	#Get M, the transform matrix, and Minv, the inverse
	M = cv2.getPerspectiveTransform(src, dst)
	Minv = cv2.getPerspectiveTransform(dst, src)

	# Warp the image using OpenCV warpPerspective()
	warped_img = cv2.warpPerspective(np.float32(img), M, (640, 480), flags=cv2.INTER_LINEAR)

	return warped_img, M, Minv

# Define a class to receive the characteristics of each line detection
class Line():
	def __init__(self, n):
		"""
		n is the window size of the moving average
		"""
		self.n = n
		self.detected = False

		# Polynomial coefficients: x = A*y^2 + B*y + C
		# Each of A, B, C is a "list-queue" with max length n
		self.A = []
		self.B = []
		self.C = []
		# Average of above
		self.A_avg = 0.
		self.B_avg = 0.
		self.C_avg = 0.

	def get_fit(self):
		return (self.A_avg, self.B_avg, self.C_avg)

	def add_fit(self, fit_coeffs):
		"""
		Gets most recent line fit coefficients and updates internal smoothed coefficients
		fit_coeffs is a 3-element list of 2nd-order polynomial coefficients
		"""
		# Coefficient queue full?
		q_full = len(self.A) >= self.n

		# Append line fit coefficients
		self.A.append(fit_coeffs[0])
		self.B.append(fit_coeffs[1])
		self.C.append(fit_coeffs[2])

		# Pop from index 0 if full
		if q_full:
			_ = self.A.pop(0)
			_ = self.B.pop(0)
			_ = self.C.pop(0)

		# Simple average of line coefficients
		self.A_avg = np.mean(self.A)
		self.B_avg = np.mean(self.B)
		self.C_avg = np.mean(self.C)

		return (self.A_avg, self.B_avg, self.C_avg)