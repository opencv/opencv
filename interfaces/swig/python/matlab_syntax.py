#########################################################################################
#
#  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
#
#  By downloading, copying, installing or using the software you agree to this license.
#  If you do not agree to this license, do not download, install,
#  copy or use the software.
#
#
#                        Intel License Agreement
#                For Open Source Computer Vision Library
#
# Copyright (C) 2000, Intel Corporation, all rights reserved.
# Third party copyrights are property of their respective owners.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#   * Redistribution's of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#   * Redistribution's in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#   * The name of Intel Corporation may not be used to endorse or promote products
#     derived from this software without specific prior written permission.
#
# This software is provided by the copyright holders and contributors "as is" and
# any express or implied warranties, including, but not limited to, the implied
# warranties of merchantability and fitness for a particular purpose are disclaimed.
# In no event shall the Intel Corporation or contributors be liable for any direct,
# indirect, incidental, special, exemplary, or consequential damages
# (including, but not limited to, procurement of substitute goods or services;
# loss of use, data, or profits; or business interruption) however caused
# and on any theory of liability, whether in contract, strict liability,
# or tort (including negligence or otherwise) arising in any way out of
# the use of this software, even if advised of the possibility of such damage.
#
#########################################################################################

"""Matlab syntax for OpenCV

For those who have switched from Matlab, this module offers similar syntax to the basic
Matlab commands. I.e. you can invoke 'imread' to load images, 'imshow', etc.
"""

from cv import *
from highgui import cvShowImage,cvNamedWindow,cvLoadImage,cvWaitKey 

#__all__ = ['imagesc', 'display', 'imread', 'imshow', 'saveimage', 'loadimage', 'pause',
#           'Image', 'Image8', 'Image8c3', 'Image32s', 'Image32f', 'Image64f']

def eye(*args):
    mat = array(*args)
    cvSetIdentity(mat);
    return mat

def ones(*args):
    mat = array(*args)
    cvSet(mat, cvScalarAll(1.0))
    return mat

def zeros(*args):
    mat = array(*args)
    cvSet(mat, cvScalarAll(0.0))
    return mat

def array(*args):
    m=1
    n=1
    c=1
    classname='single'
    nargs = len(args)
    # nasty argument parsing
    if nargs>0:
        if isinstance(args[0],tuple) or isinstance(args[0],list) and len(args[0]) > 1:
            m=args[0][0]
            n=args[0][1]
            if len(args[0])>2:
                c=args[0][2]
            if len(args)>1:
                classname = args[1]
        else:
            m=args[0]
            if nargs == 1:
                n=args[0]
            elif nargs > 1:
                # is the last argument the classname?
                if args[nargs-1].__class__ == str:
                    classname = args[nargs-1]
                    nargs-=1
                if nargs > 1:
                    n = args[1]
                if nargs > 2:
                    c = args[2]

    if(classname=='double'):
        depth=cv.CV_64F 
    elif(classname=='single'):
        depth=cv.CV_32F
    elif(classname=='int8'):
        depth=cv.CV_8S
    elif(classname=='uint8'):
        depth=cv.CV_8U
    elif(classname=='int16'):
        depth=cv.CV_16S
    elif(classname=='uint16'):
        depth=cv.CV_16U
    elif(classname=='int32' or classname=='uint32' or 
            classname=='int64' or classname=='uint64'):
        depth=cv.CV_32S
    else:
        depth=cv.CV_32F
    depth = CV_MAKETYPE(depth, c)
    return cvCreateMat(m,n,depth)
 
def size(X,dim=-1):
    # CvMat
    if hasattr(X, "type"):
        sz = (X.rows, X.cols, CV_MAT_CN(X.type))
    # IplImage
    elif hasattr(X, "nChannels"):
        sz = (X.height, X.width, X.nChannels)
    # CvMatNd
    else:
        sz = cvGetDims(X)

    if dim is -1:
        return sz
    return sz[dim]

def reshape(X, m, n=1, c=-1):
    '''
    reshape will produce different results in matlab and python due to the
    order of elements stored in the array (row-major vs. column major)
    '''
    if c==-1:
        c = CV_MAT_CN(X)
    return cvReshape(X, c, m)
     

def im2float(im):
    mat = cvGetMat(im);
    if CV_MAT_DEPTH(mat.type)==CV_32F:
        return mat
    
    im64f = array(size(im), 'float')
    cvConvertScale(im, im64f, 1.0, 0.0)
    return im64f

def im2double(im):
    mat = cvGetMat(im);
    if CV_MAT_DEPTH(mat.type)==CV_64F:
        return mat
    im64f = array(size(im), 'double')
    cvConvertScale(im, im64f, 1.0, 0.0)
    return im64f

def rgb2ntsc (rgb):
    trans = [ [0.299,  0.596,  0.211], [0.587, -0.274, -0.523], [0.114, -0.322,  0.312] ];
    return rgb * trans;
    
def rgb2gray(rgb):
    ntscmap = rgb2ntsc (rgb);
    graymap = ntscmap [:, 1] * ones (1, 3);
    return graymap

class cvImageViewer:
    """
    Wrapper class for some matlab/octave/scilab syntax image viewing functions
    """
    currentWindowName = ""
    currentWindow = -1
    maxWindow = -1

    def imagesc(self,im, clims=None):
        """
        Display a normalized version of the image
        """
        if(self.currentWindow==-1):
            self.display()

        # don't normalize multichannel image
        #if(im.nChannels>1):
        #    if(im.depth!=cv.IPL_DEPTH_8U):
        #        im2 = cvCreateImage( cvSize(im.width, im.height), cv.IPL_DEPTH_8U, im.nChannels)
        #        cvScale(im, im2)
        #        im = im2
        #    cvShowImage(self.currentWindowName, im)
        #    return self.currentWindow
        
        # normalize image
        if clims:
            [minv, maxv] = clims
        else:
            [minv,maxv] = cvMinMaxLoc(im)
        if maxv != minv:
            s = 255.0/(maxv-minv)
            shift =  255*(-minv)/(maxv-minv)
        else:
            s = 1.0
            shift = -maxv

        im2 = array( size(im), 'uint8' )
        cvConvertScale(im, im2, s, shift)
        
        cvShowImage(self.currentWindowName, im2)

    def image(self, im):
        """
        Display image as is -- probably not what you'd expect for FP or integer images
        """
        if(self.currentWindow==-1):
            self.display()

        cvShowImage(self.currentWindowName,im)
        return self.currentWindow
        
    
    def display(self, index=-1):
        """
        open a new window
        """
        if(index==-1):
            self.maxWindow = self.maxWindow+1;
            index= self.maxWindow;

        if(index > self.maxWindow):
            self.maxWindow = index;

        self.currentWindow = index;
        self.currentWindowName = "opencv-python window %d" % self.currentWindow
        cvNamedWindow(self.currentWindowName,0)
        return self.currentWindow

def drawnow():
    cvWaitKey(10)

def pause(delay=-1):
    if delay<0:
        cvWaitKey(-1)
    else:
        cvWaitKey(delay*1000)

c = cvImageViewer()
imagesc = c.imagesc
display = c.display
image = c.image
imshow = c.image

def imread(fname):
    return cvLoadImage(fname, -1)   
loadimage = imread
imload = imread

def imsave(im, fname, format):
    return cvSaveImage(fname, im)
saveimage = imsave

def gradient(F):
    F = im2float(F)
    Fx = array(size(F))
    Fy = array(size(F))
    
    # new images
    cvSobel(F, Fx, 1, 0, CV_SCHARR)
    cvSobel(F, Fy, 0, 1, CV_SCHARR)
    return (Fx, Fy)
