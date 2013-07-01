import numpy
from scipy.ndimage import interpolation


def homogeneous(xywIn):
	'''
	Converts a 2xN or 3xN set of points to homogeneous coordinates.
		- for 2xN arrays, adds a row of ones
		- for 3xN arrays, divides all rows by the third row
	'''
	if xywIn.shape[0] == 3:
		xywOut = numpy.zeros_like(xywIn)
		for i in range(3):
			xywOut[i, :] = xywIn[i, :] / xywIn[2, :]
	elif xywIn.shape[0] == 2:
		xywOut = numpy.vstack((xywIn, numpy.ones((1, xywIn.shape[1]), dtype=xywIn.dtype)))
	
	return xywOut


def transformImage(input, tform, output_range="same"):
	'''
	Transform the input image using a 3x3 transform.  The array tform should
	specify the 3x3 transform from input to output coordinates, a forward
	mapping.  This matrix needs to have an inverse.
	
	output_range should be one of:
		"same" - match the range from the input image
		"auto" - determine a range by applying the transform to the corners of the input image
		ndarray - a 2x2 numpy array like [(x1, y1), (x2, y2)]
	'''
	print "\n--------------------------------"
	print "Transforming Image"
	print "--------------------------------\n"
	
	h, w = input.shape[0:2]
	tform = numpy.array(tform)
	
	# determine pixel coordinates of output image
	if output_range == "auto":

		# set up the corners in an array
		corners = numpy.array( [[ 0, w, w, 0],
								[ 0, 0, h, h]],dtype=float)
								
		# transform the corners
		corners = homogeneous(corners)
		A = numpy.dot(tform, corners)
		A = homogeneous(A)
		A = A.astype(int)
				
		#  find the new corners of the image 
		w1 = min(A[0,:])
		w2 = max(A[0,:])
		h1 = min(A[1,:])
		h2 = max(A[1,:])

		#set up the new window size
		yy, xx = numpy.mgrid[h1:h2, w1:w2]
		h = h2 - h1
		w = w2 - w1
		
	elif type(output_range) == numpy.ndarray:
			
		#grab the corners
		w1 = output_range[0,0]
		w2 = output_range[1,0]
		h1 = output_range[0,1]
		h2 = output_range[1,1]
		
		#set up the new window size
		yy, xx = numpy.mgrid[h1:h2, w1:w2]
		h = h2 - h1
		w = w2 - w1
		
	else:
		yy, xx = numpy.mgrid[0:h, 0:w]
		
	print "transforming image...  ",	
	
	# transform output pixel coordinates into input image coordinates
	xywOut = numpy.vstack((xx.flatten(), yy.flatten()))
	xywOut = homogeneous(xywOut)
	xywIn = numpy.dot(numpy.linalg.inv(tform), xywOut)
	xywIn = homogeneous(xywIn)
	
	# reshape input image coordinates
	xxIn = xywIn[0,:].reshape((h, w))
	yyIn = xywIn[1,:].reshape((h, w))
	
	# sample output image
	if input.ndim == 3:
		output = numpy.zeros((h, w, input.shape[2]), dtype=input.dtype)
		for d in range(input.shape[2]):
			output[..., d] = interpolation.map_coordinates(input[..., d], [yyIn, xxIn])
	else:
		output = interpolation.map_coordinates(input, [yyIn, xxIn])
	
	print "done"
		
	return output


def makeCenteredRotation(angle, center=(0, 0)):
	'''
	Creates a 3x3 transform matrix for centered rotation.
	'''
	# center
	Cf = numpy.array([[1, 0, -center[0]],
					  [0, 1, -center[1]],
					  [0, 0, 1]])
	Cb = numpy.linalg.inv(Cf)
	
	# rotate
	cost = numpy.cos(numpy.deg2rad(angle))
	sint = numpy.sin(numpy.deg2rad(angle))
	R = numpy.array([[cost, -sint, 0],
					 [sint, cost, 0],
					 [0, 0, 1]])
	
	# compose
	A = numpy.dot(R, Cf)
	A = numpy.dot(Cb, A)
	return A


def testRotate():
	import cv
	import optparse
	import imgutil
	
	# handle command line arguments
	parser = optparse.OptionParser()
	parser.add_option("-f", "--filename", help="input image file")
	parser.add_option("-a", "--angle", help="rotation angle in degrees", default=0, type="float")
	options, remain = parser.parse_args()
	if options.filename is None:
		parser.print_help()
		exit(0)
	
	# load image
	cvimg = cv.LoadImage(options.filename)
	npimg = imgutil.cv2array(cvimg)
	
	# rotate image
	h,w = npimg.shape[0:2]
	print h, w
	A = makeCenteredRotation(options.angle, (w/2.0, h/2.0))
	nprot = transformImage(npimg, A, "auto")
	
	imgutil.imageShow(npimg, "original")
	imgutil.imageShow(nprot, "rotate")
	cv.WaitKey(0)
	
	
	
	
if __name__ == "__main__":
	testRotate()