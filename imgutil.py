'''
Created on Apr 4, 2010

@author: bseastwo
'''
import cv
import numpy

def imageInfo(image, title="image"):
    '''
    Print image information.
    '''
    print "{0}: {1} of {2}, {3} {4} {5}".format(
        title, image.shape, image.dtype,  image.min(), image.max(), image.mean())

def normalize(image, range=(0,255), dtype=numpy.uint8):
    '''
    Linearly remap values in input data into range (0-255, by default).  
    Returns the dtype result of the normalization (numpy.uint8 by default).
    '''
    # find input and output range of data
    if isinstance(range, (int, float, long)):
        minOut, maxOut = 0., float(range)
    else:
        minOut, maxOut = float(range[0]), float(range[1])
    minIn, maxIn = image.min(), image.max()
    ratio = (maxOut - minOut) / (maxIn - minIn)
    
    # remap data
    output = (image - minIn) * ratio + minOut
    
    return output.astype(dtype)

def ncc(img1, img2):
    '''
    Computes the normalized cross correlation for a pair of images.
    NCC is computed as follows: 
    \mu = \frac{1}{N} \sum_{x=1}^N I(x)
    ncc(I_1, I_2) = \frac{(I_1 - \mu_1)(I_2 - \mu_2)}{\sqrt{\sum (I_1 - \mu_1)^2 \sum (I_2 - \mu_2)^2}}
    
    where all sums are over the image plane, and the two images I_1 and I_2 
    have the same number of elements, N.
    
    If the supplied images have a different number of elements, returns -1.
    '''
    if (img1.size != img2.size):
        return -1
    
    I1 = img1 - img1.mean()
    I2 = img2 - img2.mean()
    
    correlation = (I1 * I2).sum()
    normalizer = numpy.sqrt((I1**2).sum() * (I2**2).sum())
    
    return correlation / normalizer
    
def equalize(image, alpha=1.0):
    '''
    Apply histogram equalization to an image.  Returns the uint8 result of
    the equalization.
    '''
    # build histogram and cumulative distribution function
    hist = numpy.histogram(image, 256, (0, 255))
    cdist = numpy.cumsum(hist[0])
    cdist = (255.0 / image.size) * cdist
    
    # apply distribution function to image
    output = alpha * cdist[image] + (1-alpha) * image
    return numpy.uint8(output)

def gaussian(sigma, order=0, radius=0, norm=True):
    '''
    Computes the values of a 1D Gaussian function with standard deviation
    sigma.  The number of values returned is 2*radius + 1.  If radius is 0, 
    an appropriate radius is chosen to include at least 98% of the Gaussian.  
    If norm is True, the Gaussian function values sum to 1.
    
    returns a (2*radius+1)-element numpy array
    '''
    sigma = float(sigma)
    
    # choose an appropriate radius if one is not given; 98% of Gaussian is
    # within 5 sigma of the center.
    if radius == 0:
        radius = numpy.floor(sigma * 5.0/2) 
        
    # compute Gaussian values
    xrange = numpy.arange(-radius, radius + 1)
    denom = 1 / (2 * (sigma ** 2))
    data = numpy.exp(-denom * (xrange ** 2))
    
    # derivatives of Gaussians are products of polynomials (Hermite polynomials)
    # from Front-End Vision, pg. 54
    if order == 1:
        data = -data * (xrange / (sigma ** 2))
    elif order == 2:
        data =  data * ((xrange ** 2 - sigma ** 2) / (sigma ** 4))
    elif order == 3:
        data = -data * ((xrange ** 3 - 3 * xrange * sigma ** 2) / (sigma ** 6))
    elif order == 4:
        data =  data * ((xrange ** 4 - 6 * xrange ** 2 * sigma ** 2 + 3 * sigma ** 4) / (sigma ** 8))
    
    # normalize
    if norm:
        scale = 1 / (sigma * numpy.sqrt(2 * numpy.pi))
        data = scale * data
        
    return (data, xrange)

# OpenCV / numpy data conversion functions.  
# These may be obsoleted by OpenCV 2.2.
# From source at http://opencv.willowgarage.com/wiki/PythonInterface

def cv2array(im):
    depth2dtype = {
        cv.IPL_DEPTH_8U: 'uint8',
        cv.IPL_DEPTH_8S: 'int8',
        cv.IPL_DEPTH_16U: 'uint16',
        cv.IPL_DEPTH_16S: 'int16',
        cv.IPL_DEPTH_32S: 'int32',
        cv.IPL_DEPTH_32F: 'float32',
        cv.IPL_DEPTH_64F: 'float64',
    }
  
    a = numpy.fromstring(
         im.tostring(),
         dtype=depth2dtype[im.depth],
         count=im.width*im.height*im.nChannels)
    a.shape = (im.height,im.width,im.nChannels)
    return a
    
def array2cv(a):
    dtype2depth = {
        'uint8':   cv.IPL_DEPTH_8U,
        'int8':    cv.IPL_DEPTH_8S,
        'uint16':  cv.IPL_DEPTH_16U,
        'int16':   cv.IPL_DEPTH_16S,
        'int32':   cv.IPL_DEPTH_32S,
        'float32': cv.IPL_DEPTH_32F,
        'float64': cv.IPL_DEPTH_64F,
    }
    try:
        nChannels = a.shape[2]
    except:
        nChannels = 1
    cv_im = cv.CreateImageHeader((a.shape[1],a.shape[0]), 
                                 dtype2depth[str(a.dtype)],
                                 nChannels)
    cv.SetData(cv_im, a.tostring(), 
               a.dtype.itemsize*nChannels*a.shape[1])
    return cv_im

def imageShow(image, title="image", norm=True):
    '''
    Display an image in a resizable cv window.  If the image is a numpy
    array, it will be converted to a cv image before display with an
    optional normalization of the image data to the range [0 ... 255].
    '''
    if type(image) != numpy.ndarray:
        cvimg = image
    elif norm:    
        cvimg = array2cv(normalize(image))
    else:
        cvimg = array2cv(image)
    
    cv.NamedWindow(title, cv.CV_WINDOW_NORMAL)
    cv.ShowImage(title, cvimg)

if __name__ == "__main__":
    print "testing ncc"
    
    size = 1024
    mag = 256
    I1 = mag * numpy.random.rand(size, size)
    I1n = I1 + 0.15 * mag * numpy.random.randn(size, size)
    I2 = mag * numpy.random.rand(size, size)
    
    print "ncc(I1, I1) =", ncc(I1, I1)
    print "ncc(I1, I2) =", ncc(I1, I2)
    print "ncc(I2, inv(I2)) =", ncc(I2, mag - I2)
    print "ncc(I1, I1 + N(0, .15) =", ncc(I1, I1n)
    print "ncc(I1, I1n + 4) =", ncc(I1, I1n + mag/2)
    print "ncc(I1, I1n * 4) =", ncc(I1, I1n * mag/2)
