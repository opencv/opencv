Cookbook
========

.. highlight:: python


Here is a collection of code fragments demonstrating some features
of the OpenCV Python bindings.


Convert an image
----------------





.. doctest::


    
    >>> import cv
    >>> im = cv.LoadImageM("building.jpg")
    >>> print type(im)
    <type 'cv.cvmat'>
    >>> cv.SaveImage("foo.png", im)
    

..


Resize an image
---------------


To resize an image in OpenCV, create a destination image of the appropriate size, then call 
:ref:`Resize`
.




.. doctest::


    
    >>> import cv
    >>> original = cv.LoadImageM("building.jpg")
    >>> thumbnail = cv.CreateMat(original.rows / 10, original.cols / 10, cv.CV_8UC3)
    >>> cv.Resize(original, thumbnail)
    

..


Compute the Laplacian
---------------------





.. doctest::


    
    >>> import cv
    >>> im = cv.LoadImageM("building.jpg", 1)
    >>> dst = cv.CreateImage(cv.GetSize(im), cv.IPL_DEPTH_16S, 3)
    >>> laplace = cv.Laplace(im, dst)
    >>> cv.SaveImage("foo-laplace.png", dst)
    

..


Using GoodFeaturesToTrack
-------------------------


To find the 10 strongest corner features in an image, use 
:ref:`GoodFeaturesToTrack`
like this:




.. doctest::


    
    >>> import cv
    >>> img = cv.LoadImageM("building.jpg", cv.CV_LOAD_IMAGE_GRAYSCALE)
    >>> eig_image = cv.CreateMat(img.rows, img.cols, cv.CV_32FC1)
    >>> temp_image = cv.CreateMat(img.rows, img.cols, cv.CV_32FC1)
    >>> for (x,y) in cv.GoodFeaturesToTrack(img, eig_image, temp_image, 10, 0.04, 1.0, useHarris = True):
    ...    print "good feature at", x,y
    good feature at 198.0 514.0
    good feature at 791.0 260.0
    good feature at 370.0 467.0
    good feature at 374.0 469.0
    good feature at 490.0 520.0
    good feature at 262.0 278.0
    good feature at 781.0 134.0
    good feature at 3.0 247.0
    good feature at 667.0 321.0
    good feature at 764.0 304.0
    

..


Using GetSubRect
----------------


GetSubRect returns a rectangular part of another image.  It does this without copying any data.




.. doctest::


    
    >>> import cv
    >>> img = cv.LoadImageM("building.jpg")
    >>> sub = cv.GetSubRect(img, (60, 70, 32, 32))  # sub is 32x32 patch within img
    >>> cv.SetZero(sub)                             # clear sub to zero, which also clears 32x32 pixels in img
    

..


Using CreateMat, and accessing an element
-----------------------------------------





.. doctest::


    
    >>> import cv
    >>> mat = cv.CreateMat(5, 5, cv.CV_32FC1)
    >>> cv.Set(mat, 1.0)
    >>> mat[3,1] += 0.375
    >>> print mat[3,1]
    1.375
    >>> print [mat[3,i] for i in range(5)]
    [1.0, 1.375, 1.0, 1.0, 1.0]
    

..


ROS image message to OpenCV
---------------------------


See this tutorial: 
`Using CvBridge to convert between ROS images And OpenCV images <http://www.ros.org/wiki/cv_bridge/Tutorials/UsingCvBridgeToConvertBetweenROSImagesAndOpenCVImages>`_
.


PIL Image to OpenCV
-------------------


(For details on PIL see the 
`PIL handbook <http://www.pythonware.com/library/pil/handbook/image.htm>`_
.)




.. doctest::


    
    >>> import Image, cv
    >>> pi = Image.open('building.jpg')       # PIL image
    >>> cv_im = cv.CreateImageHeader(pi.size, cv.IPL_DEPTH_8U, 3)
    >>> cv.SetData(cv_im, pi.tostring())
    >>> print pi.size, cv.GetSize(cv_im)
    (868, 600) (868, 600)
    >>> print pi.tostring() == cv_im.tostring()
    True
    

..


OpenCV to PIL Image
-------------------





.. doctest::


    
    >>> import Image, cv
    >>> cv_im = cv.CreateImage((320,200), cv.IPL_DEPTH_8U, 1)
    >>> pi = Image.fromstring("L", cv.GetSize(cv_im), cv_im.tostring())
    >>> print pi.size
    (320, 200)
    

..


NumPy and OpenCV
----------------


Using the 
`array interface <http://docs.scipy.org/doc/numpy/reference/arrays.interface.html>`_
, to use an OpenCV CvMat in NumPy:




.. doctest::


    
    >>> import cv, numpy
    >>> mat = cv.CreateMat(3, 5, cv.CV_32FC1)
    >>> cv.Set(mat, 7)
    >>> a = numpy.asarray(mat)
    >>> print a
    [[ 7.  7.  7.  7.  7.]
     [ 7.  7.  7.  7.  7.]
     [ 7.  7.  7.  7.  7.]]
    

..

and to use a NumPy array in OpenCV:




.. doctest::


    
    >>> import cv, numpy
    >>> a = numpy.ones((480, 640))
    >>> mat = cv.fromarray(a)
    >>> print mat.rows
    480
    >>> print mat.cols
    640
    

..

also, most OpenCV functions can work on NumPy arrays directly, for example:




.. doctest::


    
    >>> picture = numpy.ones((640, 480))
    >>> cv.Smooth(picture, picture, cv.CV_GAUSSIAN, 15, 15)
    

..

Given a 2D array, 
the 
:ref:`fromarray`
function (or the implicit version shown above)
returns a single-channel 
:ref:`CvMat`
of the same size.
For a 3D array of size 
:math:`j \times k \times l`
, it returns a 
:ref:`CvMat`
sized 
:math:`j \times k`
with 
:math:`l`
channels.

Alternatively, use 
:ref:`fromarray`
with the 
``allowND``
option to always return a 
:ref:`cvMatND`
.


OpenCV to pygame
----------------


To convert an OpenCV image to a 
`pygame <http://www.pygame.org/>`_
surface:




.. doctest::


    
    >>> import pygame.image, cv
    >>> src = cv.LoadImage("lena.jpg")
    >>> src_rgb = cv.CreateMat(src.height, src.width, cv.CV_8UC3)
    >>> cv.CvtColor(src, src_rgb, cv.CV_BGR2RGB)
    >>> pg_img = pygame.image.frombuffer(src_rgb.tostring(), cv.GetSize(src_rgb), "RGB")
    >>> print pg_img
    <Surface(512x512x24 SW)>
    

..


OpenCV and OpenEXR
------------------


Using 
`OpenEXR's Python bindings <http://www.excamera.com/sphinx/articles-openexr.html>`_
you can make a simple
image viewer:




::


    
    import OpenEXR, Imath, cv
    filename = "GoldenGate.exr"
    exrimage = OpenEXR.InputFile(filename)
    
    dw = exrimage.header()['dataWindow']
    (width, height) = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    
    def fromstr(s):
        mat = cv.CreateMat(height, width, cv.CV_32FC1)
        cv.SetData(mat, s)
        return mat
    
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    (r, g, b) = [fromstr(s) for s in exrimage.channels("RGB", pt)]
    
    bgr = cv.CreateMat(height, width, cv.CV_32FC3)
    cv.Merge(b, g, r, None, bgr)
    
    cv.ShowImage(filename, bgr)
    cv.WaitKey()
    

..

