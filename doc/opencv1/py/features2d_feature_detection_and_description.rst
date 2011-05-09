Feature detection and description
=================================

.. highlight:: python




    
    * **image** The image. Keypoints (corners) will be detected on this. 
    
    
    * **keypoints** Keypoints detected on the image. 
    
    
    * **threshold** Threshold on difference between intensity of center pixel and 
                pixels on circle around this pixel. See description of the algorithm. 
    
    
    * **nonmaxSupression** If it is true then non-maximum supression will be applied to detected corners (keypoints).  
    
    
    

.. index:: CvSURFPoint

.. _CvSURFPoint:

CvSURFPoint
-----------

`id=0.785092904945 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/py/features2d/CvSURFPoint>`__

.. class:: CvSURFPoint



A SURF keypoint, represented as a tuple 
``((x, y), laplacian, size, dir, hessian)``
.



    
    
    .. attribute:: x
    
    
    
        x-coordinate of the feature within the image 
    
    
    
    .. attribute:: y
    
    
    
        y-coordinate of the feature within the image 
    
    
    
    .. attribute:: laplacian
    
    
    
        -1, 0 or +1. sign of the laplacian at the point.  Can be used to speedup feature comparison since features with laplacians of different signs can not match 
    
    
    
    .. attribute:: size
    
    
    
        size of the feature 
    
    
    
    .. attribute:: dir
    
    
    
        orientation of the feature: 0..360 degrees 
    
    
    
    .. attribute:: hessian
    
    
    
        value of the hessian (can be used to approximately estimate the feature strengths; see also params.hessianThreshold) 
    
    
    

.. index:: ExtractSURF

.. _ExtractSURF:

ExtractSURF
-----------

`id=0.999928834286 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/py/features2d/ExtractSURF>`__


.. function:: ExtractSURF(image,mask,storage,params)-> (keypoints,descriptors)

    Extracts Speeded Up Robust Features from an image.





    
    :param image: The input 8-bit grayscale image 
    
    :type image: :class:`CvArr`
    
    
    :param mask: The optional input 8-bit mask. The features are only found in the areas that contain more than 50 %  of non-zero mask pixels 
    
    :type mask: :class:`CvArr`
    
    
    :param keypoints: sequence of keypoints. 
    
    :type keypoints: :class:`CvSeq` of :class:`CvSURFPoint`
    
    
    :param descriptors: sequence of descriptors.  Each SURF descriptor is a list of floats, of length 64 or 128. 
    
    :type descriptors: :class:`CvSeq` of list of float
    
    
    :param storage: Memory storage where keypoints and descriptors will be stored 
    
    :type storage: :class:`CvMemStorage`
    
    
    :param params: Various algorithm parameters in a tuple  ``(extended, hessianThreshold, nOctaves, nOctaveLayers)`` : 
         
            * **extended** 0 means basic descriptors (64 elements each), 1 means extended descriptors (128 elements each) 
            
            * **hessianThreshold** only features with hessian larger than that are extracted.  good default value is ~300-500 (can depend on the average local contrast and sharpness of the image).  user can further filter out some features based on their hessian values and other characteristics. 
            
            * **nOctaves** the number of octaves to be used for extraction.  With each next octave the feature size is doubled (3 by default) 
            
            * **nOctaveLayers** The number of layers within each octave (4 by default) 
            
            
    
    :type params: :class:`CvSURFParams`
    
    
    
The function cvExtractSURF finds robust features in the image, as
described in 
Bay06
. For each feature it returns its location, size,
orientation and optionally the descriptor, basic or extended. The function
can be used for object tracking and localization, image stitching etc.

To extract strong SURF features from an image




.. doctest::


    
    >>> import cv
    >>> im = cv.LoadImageM("building.jpg", cv.CV_LOAD_IMAGE_GRAYSCALE)
    >>> (keypoints, descriptors) = cv.ExtractSURF(im, None, cv.CreateMemStorage(), (0, 30000, 3, 1))
    >>> print len(keypoints), len(descriptors)
    6 6
    >>> for ((x, y), laplacian, size, dir, hessian) in keypoints:
    ...     print "x=%d y=%d laplacian=%d size=%d dir=%f hessian=%f" % (x, y, laplacian, size, dir, hessian)
    x=30 y=27 laplacian=-1 size=31 dir=69.778503 hessian=36979.789062
    x=296 y=197 laplacian=1 size=33 dir=111.081039 hessian=31514.349609
    x=296 y=266 laplacian=1 size=32 dir=107.092300 hessian=31477.908203
    x=254 y=284 laplacian=1 size=31 dir=279.137360 hessian=34169.800781
    x=498 y=525 laplacian=-1 size=33 dir=278.006592 hessian=31002.759766
    x=777 y=281 laplacian=1 size=70 dir=167.940964 hessian=35538.363281
    

..


.. index:: GetStarKeypoints

.. _GetStarKeypoints:

GetStarKeypoints
----------------

`id=0.373658080009 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/py/features2d/GetStarKeypoints>`__


.. function:: GetStarKeypoints(image,storage,params)-> keypoints

    Retrieves keypoints using the StarDetector algorithm.





    
    :param image: The input 8-bit grayscale image 
    
    :type image: :class:`CvArr`
    
    
    :param storage: Memory storage where the keypoints will be stored 
    
    :type storage: :class:`CvMemStorage`
    
    
    :param params: Various algorithm parameters in a tuple  ``(maxSize, responseThreshold, lineThresholdProjected, lineThresholdBinarized, suppressNonmaxSize)`` : 
         
            * **maxSize** maximal size of the features detected. The following values of the parameter are supported: 4, 6, 8, 11, 12, 16, 22, 23, 32, 45, 46, 64, 90, 128 
            
            * **responseThreshold** threshold for the approximatd laplacian, used to eliminate weak features 
            
            * **lineThresholdProjected** another threshold for laplacian to eliminate edges 
            
            * **lineThresholdBinarized** another threshold for the feature scale to eliminate edges 
            
            * **suppressNonmaxSize** linear size of a pixel neighborhood for non-maxima suppression 
            
            
    
    :type params: :class:`CvStarDetectorParams`
    
    
    
The function GetStarKeypoints extracts keypoints that are local
scale-space extremas. The scale-space is constructed by computing
approximate values of laplacians with different sigma's at each
pixel. Instead of using pyramids, a popular approach to save computing
time, all of the laplacians are computed at each pixel of the original
high-resolution image. But each approximate laplacian value is computed
in O(1) time regardless of the sigma, thanks to the use of integral
images. The algorithm is based on the paper 
Agrawal08
, but instead
of a square, hexagon or octagon it uses an 8-end star shape, hence the name,
consisting of overlapping upright and tilted squares.

Each keypoint is represented by a tuple 
``((x, y), size, response)``
:


    
    * **x, y** Screen coordinates of the keypoint 
    
    
    * **size** feature size, up to  ``maxSize`` 
    
    
    * **response** approximated laplacian value for the keypoint 
    
    
    
