Introduction
============

Cookbook
--------

Here is a small collection of code fragments demonstrating some features
of the OpenCV Python bindings.

Convert an image from png to jpg
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    import cv
    cv.SaveImage("foo.png", cv.LoadImage("foo.jpg"))

Compute the Laplacian
^^^^^^^^^^^^^^^^^^^^^

::

    im = cv.LoadImage("foo.png", 1)
    dst = cv.CreateImage(cv.GetSize(im), cv.IPL_DEPTH_16S, 3);
    laplace = cv.Laplace(im, dst)
    cv.SaveImage("foo-laplace.png", dst)


Using cvGoodFeaturesToTrack
^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    img = cv.LoadImage("foo.jpg")
    eig_image = cv.CreateImage(cv.GetSize(img), cv.IPL_DEPTH_32F, 1)
    temp_image = cv.CreateImage(cv.GetSize(img), cv.IPL_DEPTH_32F, 1)
    # Find up to 300 corners using Harris
    for (x,y) in cv.GoodFeaturesToTrack(img, eig_image, temp_image, 300, None, 1.0, use_harris = True):
        print "good feature at", x,y

Using GetSubRect
^^^^^^^^^^^^^^^^

GetSubRect returns a rectangular part of another image.  It does this without copying any data.

::

    img = cv.LoadImage("foo.jpg")
    sub = cv.GetSubRect(img, (0, 0, 32, 32))  # sub is 32x32 patch from img top-left
    cv.SetZero(sub)                           # clear sub to zero, which also clears 32x32 pixels in img

Using CreateMat, and accessing an element
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    mat = cv.CreateMat(5, 5, cv.CV_32FC1)
    mat[3,2] += 0.787


ROS image message to OpenCV
^^^^^^^^^^^^^^^^^^^^^^^^^^^

See this tutorial: http://www.ros.org/wiki/cv_bridge/Tutorials/UsingCvBridgeToConvertBetweenROSImagesAndOpenCVImages

PIL Image to OpenCV
^^^^^^^^^^^^^^^^^^^

(For details on PIL see the `PIL manual <http://www.pythonware.com/library/pil/handbook/image.htm>`_).

::

    import Image
    import cv
    pi = Image.open('foo.png')       # PIL image
    cv_im = cv.CreateImageHeader(pi.size, cv.IPL_DEPTH_8U, 1)
    cv.SetData(cv_im, pi.tostring())

OpenCV to PIL Image
^^^^^^^^^^^^^^^^^^^

::

    cv_im = cv.CreateImage((320,200), cv.IPL_DEPTH_8U, 1)
    pi = Image.fromstring("L", cv.GetSize(cv_im), cv_im.tostring())

NumPy and OpenCV
^^^^^^^^^^^^^^^^

Using the `array interface <http://docs.scipy.org/doc/numpy/reference/arrays.interface.html>`_, to use an OpenCV CvMat in NumPy::

    import cv
    import numpy
    mat = cv.CreateMat(5, 5, cv.CV_32FC1)
    a = numpy.asarray(mat)

and to use a NumPy array in OpenCV::

    a = numpy.ones((640, 480))
    mat = cv.fromarray(a)

even easier, most OpenCV functions can work on NumPy arrays directly, for example::

    picture = numpy.ones((640, 480))
    cv.Smooth(picture, picture, cv.CV_GAUSSIAN, 15, 15)

Given a 2D array, 
the fromarray function (or the implicit version shown above)
returns a single-channel CvMat of the same size.
For a 3D array of size :math:`j \times k \times l`, it returns a 
CvMat sized :math:`j \times k` with :math:`l` channels.

Alternatively, use fromarray with the allowND option to always return a cvMatND.
