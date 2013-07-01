'''
Dan Nelson
autoPanorama.py
'''

import numpy
import scipy
from scipy.spatial import*
import cv
import optparse
import glob
import os.path
import math

import feature
import imgutil
import transform


class autoPanorama:
	'''
	Creates a panorama by sticthing multiple images together and combines the images using gaussian
	blending. The process is done automatically.
	'''
	
	def __init__(self, path=None, autoCrop=0):
		'''
		Creates a new panorama object
		@path: a path containing images'
		@file: 
		@autoCrop:
		@return: None
		'''		
		self.files = []
		self.path = path		
		self.autoCrop = autoCrop
		
		self.importImages(self.path)
		
	
	def importImages(self, path):
		'''
		Grabs images from a directory and puts them into a list
		@path: a path containing images
		@return: True if import worked, false otherwise
		'''
		if self.path == None:
			return False
		
		fileList = set()
		
		# find image files in the directory
		exts = ["*.jpg", "*.JPG"]
		for ext in exts:
			fileList = fileList | set(glob.glob(os.path.join(path, ext)))
		
		self.files = list(fileList)
		self.files.sort()
		
		#print "\nFrom path: ", self.path
		#print "Importing...\n"
		#print "Files:  ", self.files
		return True
		
	
	def grayscale(self, npimg):
		'''
		Converts a cvimg to greyscale
		@img: an image to convert
		@return: a numpy array of dimension h x w x luminosity where luminosity
				 is calculate by 0.299R + 0.587G + 0.114B
		'''
		npimg = 0.114*npimg[...,0] + 0.587*npimg[...,1] + 0.299*npimg[...,2]
		return npimg
		
	
	def homography(self, p1, p2):
		'''
		Finds the homography that maps points from p1 to p2
		@p1: a Nx2 array of positions that correspond to p2, N >= 4
		@p2: a Nx2 array of positions that correspond to p1, N >= 4
		@return: a 3x3 matrix that maps the points from p1 to p2
		p2=Hp1
		'''
		#print "\n--------------------------------"
		#print "Computing homography"
		#print "--------------------------------\n"
		
		#print "p1: ", p1
		#print "p2: ", p2
		
		# check if there is at least 4 points
		if p1.shape[0] < 4 or p2.shape[0] < 4:
			raise ValueError("p1 and p2 must have at least 4 row")
			return
	
		# create matrix A
		A = numpy.zeros((p1.shape[0]*2, 8),dtype=float)
		A = numpy.matrix(A,dtype=float)
		
		# fill A
		for i in range(0, A.shape[0]):
			# if i is event
			if i % 2 == 0:
				A[i,0] = p1[i/2,0]
				A[i,1] = p1[i/2,1]
				A[i,2] = 1
				A[i,6] = -p2[i/2,0]*p1[i/2,0]
				A[i,7] = -p2[i/2,0]*p1[i/2,1]
			# if i is odd
			else:
				A[i,3] = p1[i/2,0]
				A[i,4] = p1[i/2,1]
				A[i,5] = 1
				A[i,6] = -p2[i/2,1]*p1[i/2,0]
				A[i,7] = -p2[i/2,1]*p1[i/2,1]
		
		#print "\nA\n", A
		
		# create vector b
		b = p2.flatten()
		b = b.reshape(b.shape[1],1)
		b = b.astype(float)
		
		#print "\nb:\n", b
		
		#calculate homography Ax=b
		if p1.shape[0] == 4:
			x = numpy.linalg.solve(A,b)
		else:
			x = numpy.linalg.lstsq(A,b)[0]
			
		# reshape x
		x = numpy.vstack((x,numpy.matrix(1)))
		x = x.reshape((3,3))
		
		#print "\nx:\n", x, "\n"
		
		return x
	
	
	def extract(self, img, harrisCorners, radius=4, scale=4):
		'''
		Extracts feature descriptors from an image
		@img: a numpy image to extract features from
		@harrisCorners: feature locations returned by the harris() corner detector
						in feature.py (a numpy array)
		@radius: distance from feature locations to extract
		@scale: equal to the sigma used in the guassian filter
				also step for pixel selection in the patch
		@return: a (w x w x N) numpy array
		'''
		
		#smooth the image with a gaussian filter
		npimg = scipy.ndimage.filters.gaussian_filter(img, sigma=scale)
		
		yy,xx = scale * numpy.mgrid[-radius:radius+1, -radius:radius+1]
		
		w = 2*radius + 1
		
		#create a numpy array to hold feature descriptors
		desc = numpy.zeros((w,w,harrisCorners.shape[0]),dtype=float)
				
		for i in range(harrisCorners.shape[0]):
		
			#get the position of the harris corner
			x = harrisCorners[i,0]
			y = harrisCorners[i,1]
			
			#create the patch
			patch = scipy.ndimage.interpolation.map_coordinates(img,[y+yy,x+xx],prefilter=False)
			
			#find mean and std of the patch
			mean = patch.mean()
			std = patch.std()
			
			#normalize the descriptor
			patch = (patch - mean) / std
			
			desc[...,i] = patch
		
		#return a numpy array of the descriptors
		return desc
		
		
	def montage(self, descriptors, numCols=32):
		'''
		Displays all of the descriptors in a montage.
		@descriptors: a numpy array of feature descriptors with size w x w x N
		@numCols: the number of descriptors to display
		@return: a numpy array (the montage)
		'''
		numPatches = descriptors.shape[2]
		numRows = numPatches/numCols
		sideLength = descriptors.shape[0]
		
		#resize descriptors into a montage
		montage = numpy.zeros((sideLength*numRows,sideLength*numCols))
		
		#place the descriptors on a canvas
		count = 0
		for i in range(numRows):
			for j in range(numCols):
				montage[i*sideLength:(i+1)*sideLength,j*sideLength:(j+1)*sideLength] = descriptors[...,count]
				count += 1
		
		#show the image
		imgutil.imageShow(montage, title="montage", norm=True)
		
		#cv.WaitKey(0)
		
		return montage
	
	
	def matching(self, desc1, desc2):
		'''
		Calculate the matching features between 2 sets of descriptors
		@desc1: descriptors for image 1
		@desc2: descriptors for image 2
		@return a n x 2 numpy array of matching feature indices
		'''
		h,w,n = desc1.shape[0:3]
		match1 = (desc1.reshape((w**2,n))).T
		match2 = (desc2.reshape((w**2,n))).T
			
		dists = distance.cdist(match1, match2)
		
		sortIdx = numpy.argsort(dists, 1)
		
		#find best idices and their distances
		bestIdx = sortIdx[:, 0]
		bestDist = dists[numpy.r_[0:n], bestIdx]
		
		#find second best indices and their distances
		secondBestIdx = sortIdx[:, 1]
		secondBestDist = dists[numpy.r_[0:n], secondBestIdx]
		
		#find the average of the second best distance
		mean = secondBestDist.mean()
		
		ratio = bestDist / mean
		
		#find the indices of the bestMatches for each descriptor
		desc1Match = numpy.argwhere(ratio < 0.5)
		desc2Match = bestIdx[desc1Match]
		
		#put the matches in a single array and return as type int
		matches = numpy.hstack([desc1Match,desc2Match])
		return matches.astype(int)


	def ransac(self, data, tolerance=0.5, maxIters=100, confidence=0.95):
		'''
		Finds the best homography that maps the features between 2 images
		
		Input
		@data: an array of N x 4 point correspondences where each row 
			   provides a point correspondence, (x1, y1, x2, y2)
		@tolerance: the error allowed for each data point to be an inlier
		@maxIters: the maximum number of times to generate a random model
		
		Output
		model: the best model (a homography)
		inliers: the indices of the inliers in data 
		'''
		#use a matrix to go along with the homography function
		data = numpy.matrix(data)
		
		iterations = 0
		bestModel = None
		bestCount = 0
		bestIndices = None
		
		#if we reached the maximum iteration
		while iterations < maxIters:
			
			#make two copies of the data
			tempData = numpy.matrix(numpy.copy(data))
			tempShuffle = numpy.copy(data)
			
			#shuffle the copied data and select 4 points
			numpy.random.shuffle(tempShuffle)
			tempShuffle = numpy.matrix(tempShuffle)[0:4]
						
			#build a homography
			homography = self.homography(tempShuffle[:,0:2],tempShuffle[:,2:])
		
			#grab the data for the appropriate image
			tempData1 = tempData[:,0:2].transpose()
			tempData1 = transform.homogeneous(tempData1)
			tempData2 = tempData[:,2:].transpose()
			
			#compute error for each point correspondence
			tformPts = (homography*tempData1)
			tformPts = transform.homogeneous(tformPts)[0:2,:]
			tformPts = numpy.array(tformPts)
			error = numpy.sqrt((numpy.array(tformPts - tempData2)**2).sum(0))
						
			inlierCount = (error < tolerance).sum()
			#if this homography is better than previous ones keep it
			if inlierCount > bestCount:
				bestModel = homography
				bestCount = inlierCount
				bestIndices = numpy.argwhere(error < tolerance)
				
				#recalculate maxIters
				p = float(inlierCount)/data.shape[0]
				maxIters = math.log(1-confidence)/math.log(1-(p**4))
		
			#increment iterations
			iterations += 1
		
		if bestModel == None:
			raise ValueError("computed error never less than threshold")
		else:
			return bestModel, bestIndices
		
		
	def showHarris(self, img, harrisCorners):
		'''
		Creates boxes around the harrisCorners of an image
		@img: a cvimg
		@harrisCorners: an array of harrisCorners
		'''
		for pxy in harrisCorners:
			cv.Rectangle(img, (int(pxy[0]-5), int(pxy[1]-5)), (int(pxy[0]+5), int(pxy[1]+5)), (0, 0, 255))
	
	
	def showMatches(self, img1, img2, matches1, matches2):
		'''
		Draws lines between matching features
		@img: a cvimg
		@harrisCorners: an array of matches
		'''		
		npimg1 = imgutil.cv2array(img1)
		npimg2 = imgutil.cv2array(img2)
		
		#create a new window
		combined = numpy.zeros((max(npimg1.shape[0],npimg2.shape[0]),npimg1.shape[1]+npimg2.shape[1],3))
		combined[0:npimg1.shape[0],0:npimg1.shape[1],...] = npimg1
		combined[0:npimg2.shape[0],npimg1.shape[1]:npimg1.shape[1]+npimg2.shape[1],...] = npimg2
		combined = imgutil.array2cv(combined)
		
		#draw lines
		for i in range(matches1.shape[0]):
			cv.Line(combined, (int(matches1[i,0]),int(matches1[i,1])), (int(matches2[i,0]+npimg1.shape[1]),int(matches2[i,1])), (0, 255, 0))
		combined = imgutil.cv2array(combined)
		
		#show the image
		imgutil.imageShow(combined, "combined")
		#cv.WaitKey(0)
		
		
	def panorama(self, sigma):
		'''
		Creates a panorama with alpha stitching and displays
		'''
		#print "\n--------------------------------"
		#print "Panorama "
		#print "--------------------------------\n"
		#list to hold the homographies
		inlierL = []
		homography = [numpy.matrix(numpy.identity(3))]
		
		# find the homography between each set of pictures
		for i in range(len(self.files)-1):
			#get everything for image 1
			img1 = cv.LoadImage(self.files[i])
			npimg1 = imgutil.cv2array(img1)
			npimg1 = self.grayscale(npimg1)
			pts1 = feature.harris(npimg1,count=512)
			desc1 = self.extract(npimg1, pts1)
			
			#get everything for image 2
			img2 = cv.LoadImage(self.files[i+1])
			npimg2 = imgutil.cv2array(img2)
			npimg2 = self.grayscale(npimg2)
			pts2 = feature.harris(npimg2,count=512)
			desc2 = self.extract(npimg2, pts2)
			
			matches = self.matching(desc1,desc2)
			self.showHarris(img1, pts1[matches[:,0]])
			self.showHarris(img2, pts2[matches[:,1]])
			
			"""
			montagePts = feature.harris(npimg1,count=20)
			montageDesc = self.extract(npimg1, montagePts)
			montage = self.montage(montageDesc, numCols=5)
			imgutil.imageShow(montage, "montage")
			"""
			
			imgutil.imageShow(img1,"image1")
			imgutil.imageShow(img2,"image2")
			#cv.WaitKey(0)
			
			matches1 = pts1[matches[:,0],0:2]
			matches2 = pts2[matches[:,1],0:2]
			data = numpy.hstack((matches1,matches2))
			
			h = self.ransac(data,0.5)
			self.showMatches(img1, img2, data[h[1]][:,0,0:2], data[h[1]][:,0,2:])
			
			homography.append(numpy.linalg.inv(h[0]))
			inlierL.append(h[1])
			
		#print "List of homographies: "
		#print homography
		
		midHomographyL = []
		#map all the homographies to image 1
		for i in range(1,len(homography)):
			homography[i] =  homography[i-1] * homography[i]
		
		middle = len(self.files)/2
		for i in range(len(homography)):
			#warp mid,  Him = Hm0^-1 * Hi0 where m is middle image
			inverse = numpy.linalg.inv(homography[middle])
			midHomography = inverse * homography[i]
			midHomographyL.append(midHomography)
		
		#find bounds of global extent and original picture
		warpedL = []
		output_range = self.corners(midHomographyL)[0]
		midCorners = self.corners(midHomographyL)[1]
		
		# warp the images
		for i in range(len(self.files)):
			#convert the file
			cvimg = cv.LoadImage(self.files[i])
			npimg = imgutil.cv2array(cvimg)
			
			#compute the gaussian weight
			h = npimg.shape[0]
			w = npimg.shape[1]
			yy,xx = numpy.mgrid[0:h,0:w]
			dist = (yy - h/2)**2 + (xx - w/2)**2
			gwt = numpy.exp(-dist/(2.0*sigma**2))
			
			#add the gaussian weight as the 4th channel
			npimg = numpy.dstack((npimg,gwt))
			
			#append the warped image to the list
			warpedImg = transform.transformImage(npimg,midHomographyL[i], output_range)
			warpedL.append(warpedImg)
			
			imgutil.imageShow(warpedImg, "test")
		
		#stich the images
		top = numpy.zeros(warpedL[0].shape,dtype=float)
		bot = numpy.zeros(warpedL[0].shape,dtype=float)
		bot[:,:,3]=1.0
		for i in range(len(warpedL)):
			top[:,:,0] += warpedL[i][:,:,3] * warpedL[i][:,:,0]
			top[:,:,1] += warpedL[i][:,:,3] * warpedL[i][:,:,1]
			top[:,:,2] += warpedL[i][:,:,3] * warpedL[i][:,:,2]
			top[:,:,3] += warpedL[i][:,:,3]
			bot[:,:,0] += warpedL[i][:,:,3]
			bot[:,:,1] += warpedL[i][:,:,3]
			bot[:,:,2] += warpedL[i][:,:,3]
		
		bot[bot == 0] = 1
	
		output = top/bot

		#autoCrop if it is on
		if self.autoCrop:
			output = self.crop(output, output_range, midCorners[0:2,...])
		
		#show the panorama
		print "showing panorama"
		imgutil.imageShow(output, "final")
		cv.WaitKey(0)
		
		
	def cropHeight(self):
		'''
		Returns the height of the middle image
		'''
		#grab the middle image
		img = cv.LoadImage(self.files[len(self.files)/2])
		npimg = imgutil.cv2array(img)
		height = npimg.shape[0]
		return height
		
	
	def crop(self, img, edges, orig):
		'''
		Crops an image so that it is a rectangle
		@npimg: an image to crop
		@return: the cropped image
		'''
		#find the extents in the y direction
		maxX = orig[0,...].max()
		maxY = orig[1,...].max()
		x1 = 0 - edges[0,0]
		x2 = maxX - edges[1,0]
		y1 = 0 - edges[0,1]
		y2 = -(maxY - edges [1,1])
		
		#slice the image in y direction
		img = img[y1+1:img.shape[0]-y2-1]
		
		#find the extents in the x direction
		cropTop = numpy.argwhere(img[0,:,3]!=0)
		cropBot = numpy.argwhere(img[-1,:,3]!=0)
		minT = cropTop.min()
		maxT = cropTop.max()
		minB = cropBot.min()
		maxB = cropBot.max()
		
		#grab the correct extents
		xMin = max(minT,minB)
		xMax = min(maxT,maxB)
		
		#slice the image in x direction
		img = img[:,xMin:xMax]
		
		return img

	
	def corners(self, homography):
		'''
		Finds the corners of the images
		@homography: a list of homographies
		@return: an array of corners of the global window
		'''
		#find the corners of all the images
		cornerL = []
		midCorners = None
		for i in range(len(self.files)):
			#convert the file
			cvimg = cv.LoadImage(self.files[i])
			npimg = imgutil.cv2array(cvimg)
			
			# set up the corners in an array
			h, w = npimg.shape[0:2]
			corners = numpy.array( [[ 0, w, w, 0],
									[ 0, 0, h, h]],dtype=float)
			corners = transform.homogeneous(corners)
			tform = homography[i]
			A = numpy.dot(tform, corners)
			A = transform.homogeneous(A)
			A = A.astype(int)
			cornerL.append(A)
			
			if i == len(self.files)/2:
				midCorners = A
		
		#  find the new corners of the image 
		w1L = []
		w2L = []
		h1L = []
		h2L = []
		for i in range(len(cornerL)):
			w1L.append(numpy.min(cornerL[i][0,:]))
			w2L.append(numpy.max(cornerL[i][0,:]))
			h1L.append(numpy.min(cornerL[i][1,:]))
			h2L.append(numpy.max(cornerL[i][1,:]))
		w1 = min(w1L)
		w2 = max(w2L)
		h1 = min(h1L)
		h2 = max(h2L)
		
		#set up array to return
		ndarray = numpy.array([(w1, h1), (w2, h2)])
		
		return ndarray,midCorners
		
		
		
		
if __name__ == "__main__":
	# parse command line parameters
	parser = optparse.OptionParser()
	parser.add_option("-d", "--dir", help="a directory containing pictures", default=".")
	parser.add_option("-f", "--file", help="a file for rectification", default=None)
	parser.add_option("-c", "--crop", help="turn auto cropping on or off", default=0)
	options, remain = parser.parse_args()

	# launch auto panorama
	pan = autoPanorama(options.dir,options.crop)
	pan.panorama(50.0)
