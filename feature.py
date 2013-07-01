

import cv
import numpy
from scipy.ndimage import filters, interpolation
from scipy.spatial import distance
import imgutil


def harris(image, sigmaD=1.0, sigmaI=1.5, count=512):
	'''
	Finds Harris corner features for an input image.  The input image
	should be a 2D numpy array.  The sigmaD and sigmaI parameters define
	the differentiation and integration scales (respectively) of the
	Harris feature detector---the defaults are reasonable for many images.
	
	Returns:
	a maxPoints x 3 array, where each row contains the (x, y) 
	position, and Harris score of a feature point.  The array is sorted
	by decreasing score.
	'''
	
	image = image.astype(numpy.float32)
	h, w = image.shape[0:2]
	
	# compute image derivatives
	Ix = filters.gaussian_filter1d(image, sigmaD, 0, 0)
	Ix = filters.gaussian_filter1d(Ix, sigmaD, 1, 1)
	Iy = filters.gaussian_filter1d(image, sigmaD, 1, 0)
	Iy = filters.gaussian_filter1d(Iy, sigmaD, 0, 1)
	
	# compute elements of the structure tensor
	Ixx = filters.gaussian_filter(Ix**2, sigmaI, 0)
	Iyy = filters.gaussian_filter(Iy**2, sigmaI, 0)
	Ixy = filters.gaussian_filter(Ix * Iy, sigmaI, 0)
	
	# compute Harris feature strength, avoiding divide by zero
	imgH = (Ixx * Iyy - Ixy**2) / (Ixx + Iyy + 1e-8)
		
	# exclude points near the image border
	imgH[:16, :] = 0
	imgH[-16:, :] = 0
	imgH[:, :16] = 0
	imgH[:, -16:] = 0
	
	# non-maximum suppression in 5x5 regions
	maxH = filters.maximum_filter(imgH, (5,5))
	imgH = imgH * (imgH == maxH)
	
	# sort points by strength and find their positions
	sortIdx = numpy.argsort(imgH.flatten())[::-1]
	sortIdx = sortIdx[:count]
	yy = sortIdx / w
	xx = sortIdx % w
		
	# concatenate positions and values
	xyv = numpy.vstack((xx, yy, imgH.flatten()[sortIdx])).transpose()
	
	return xyv
	
	
if __name__ == "__main__":
	img = cv.LoadImage("stained2.jpg")
	npimg = imgutil.cv2array(img)
	pts = harris(npimg[:,:,1])
	print pts
