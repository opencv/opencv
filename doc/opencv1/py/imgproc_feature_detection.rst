Feature Detection
=================

.. highlight:: python



.. index:: Canny

.. _Canny:

Canny
-----

`id=0.573160740956 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/py/imgproc/Canny>`__


.. function:: Canny(image,edges,threshold1,threshold2,aperture_size=3)-> None

    Implements the Canny algorithm for edge detection.





    
    :param image: Single-channel input image 
    
    :type image: :class:`CvArr`
    
    
    :param edges: Single-channel image to store the edges found by the function 
    
    :type edges: :class:`CvArr`
    
    
    :param threshold1: The first threshold 
    
    :type threshold1: float
    
    
    :param threshold2: The second threshold 
    
    :type threshold2: float
    
    
    :param aperture_size: Aperture parameter for the Sobel operator (see  :ref:`Sobel` ) 
    
    :type aperture_size: int
    
    
    
The function finds the edges on the input image 
``image``
and marks them in the output image 
``edges``
using the Canny algorithm. The smallest value between 
``threshold1``
and 
``threshold2``
is used for edge linking, the largest value is used to find the initial segments of strong edges.


.. index:: CornerEigenValsAndVecs

.. _CornerEigenValsAndVecs:

CornerEigenValsAndVecs
----------------------

`id=0.769586068428 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/py/imgproc/CornerEigenValsAndVecs>`__


.. function:: CornerEigenValsAndVecs(image,eigenvv,blockSize,aperture_size=3)-> None

    Calculates eigenvalues and eigenvectors of image blocks for corner detection.





    
    :param image: Input image 
    
    :type image: :class:`CvArr`
    
    
    :param eigenvv: Image to store the results. It must be 6 times wider than the input image 
    
    :type eigenvv: :class:`CvArr`
    
    
    :param blockSize: Neighborhood size (see discussion) 
    
    :type blockSize: int
    
    
    :param aperture_size: Aperture parameter for the Sobel operator (see  :ref:`Sobel` ) 
    
    :type aperture_size: int
    
    
    
For every pixel, the function 
``cvCornerEigenValsAndVecs``
considers a 
:math:`\texttt{blockSize} \times \texttt{blockSize}`
neigborhood S(p). It calcualtes the covariation matrix of derivatives over the neigborhood as:



.. math::

    M =  \begin{bmatrix} \sum _{S(p)}(dI/dx)^2 &  \sum _{S(p)}(dI/dx  \cdot dI/dy)^2  \\ \sum _{S(p)}(dI/dx  \cdot dI/dy)^2 &  \sum _{S(p)}(dI/dy)^2 \end{bmatrix} 


After that it finds eigenvectors and eigenvalues of the matrix and stores them into destination image in form
:math:`(\lambda_1, \lambda_2, x_1, y_1, x_2, y_2)`
where


    

* :math:`\lambda_1, \lambda_2`
    are the eigenvalues of 
    :math:`M`
    ; not sorted
    

* :math:`x_1, y_1`
    are the eigenvectors corresponding to 
    :math:`\lambda_1`
    

* :math:`x_2, y_2`
    are the eigenvectors corresponding to 
    :math:`\lambda_2`
    
    

.. index:: CornerHarris

.. _CornerHarris:

CornerHarris
------------

`id=0.619256620171 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/py/imgproc/CornerHarris>`__


.. function:: CornerHarris(image,harris_dst,blockSize,aperture_size=3,k=0.04)-> None

    Harris edge detector.





    
    :param image: Input image 
    
    :type image: :class:`CvArr`
    
    
    :param harris_dst: Image to store the Harris detector responses. Should have the same size as  ``image`` 
    
    :type harris_dst: :class:`CvArr`
    
    
    :param blockSize: Neighborhood size (see the discussion of  :ref:`CornerEigenValsAndVecs` ) 
    
    :type blockSize: int
    
    
    :param aperture_size: Aperture parameter for the Sobel operator (see  :ref:`Sobel` ). 
    
    :type aperture_size: int
    
    
    :param k: Harris detector free parameter. See the formula below 
    
    :type k: float
    
    
    
The function runs the Harris edge detector on the image. Similarly to 
:ref:`CornerMinEigenVal`
and 
:ref:`CornerEigenValsAndVecs`
, for each pixel it calculates a 
:math:`2\times2`
gradient covariation matrix 
:math:`M`
over a 
:math:`\texttt{blockSize} \times \texttt{blockSize}`
neighborhood. Then, it stores



.. math::

    det(M) - k  \, trace(M)^2 


to the destination image. Corners in the image can be found as the local maxima of the destination image.


.. index:: CornerMinEigenVal

.. _CornerMinEigenVal:

CornerMinEigenVal
-----------------

`id=0.523904183834 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/py/imgproc/CornerMinEigenVal>`__


.. function:: CornerMinEigenVal(image,eigenval,blockSize,aperture_size=3)-> None

    Calculates the minimal eigenvalue of gradient matrices for corner detection.





    
    :param image: Input image 
    
    :type image: :class:`CvArr`
    
    
    :param eigenval: Image to store the minimal eigenvalues. Should have the same size as  ``image`` 
    
    :type eigenval: :class:`CvArr`
    
    
    :param blockSize: Neighborhood size (see the discussion of  :ref:`CornerEigenValsAndVecs` ) 
    
    :type blockSize: int
    
    
    :param aperture_size: Aperture parameter for the Sobel operator (see  :ref:`Sobel` ). 
    
    :type aperture_size: int
    
    
    
The function is similar to 
:ref:`CornerEigenValsAndVecs`
but it calculates and stores only the minimal eigen value of derivative covariation matrix for every pixel, i.e. 
:math:`min(\lambda_1, \lambda_2)`
in terms of the previous function.


.. index:: FindCornerSubPix

.. _FindCornerSubPix:

FindCornerSubPix
----------------

`id=0.448453276565 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/py/imgproc/FindCornerSubPix>`__


.. function:: FindCornerSubPix(image,corners,win,zero_zone,criteria)-> corners

    Refines the corner locations.





    
    :param image: Input image 
    
    :type image: :class:`CvArr`
    
    
    :param corners: Initial coordinates of the input corners as a list of (x, y) pairs 
    
    :type corners: sequence of (float, float)
    
    
    :param win: Half of the side length of the search window. For example, if  ``win`` =(5,5), then a  :math:`5*2+1 \times 5*2+1 = 11 \times 11`  search window would be used 
    
    :type win: :class:`CvSize`
    
    
    :param zero_zone: Half of the size of the dead region in the middle of the search zone over which the summation in the formula below is not done. It is used sometimes to avoid possible singularities of the autocorrelation matrix. The value of (-1,-1) indicates that there is no such size 
    
    :type zero_zone: :class:`CvSize`
    
    
    :param criteria: Criteria for termination of the iterative process of corner refinement. That is, the process of corner position refinement stops either after a certain number of iterations or when a required accuracy is achieved. The  ``criteria``  may specify either of or both the maximum number of iteration and the required accuracy 
    
    :type criteria: :class:`CvTermCriteria`
    
    
    
The function iterates to find the sub-pixel accurate location of corners, or radial saddle points, as shown in on the picture below.
It returns the refined coordinates as a list of (x, y) pairs.


.. image:: ../pics/cornersubpix.png



Sub-pixel accurate corner locator is based on the observation that every vector from the center 
:math:`q`
to a point 
:math:`p`
located within a neighborhood of 
:math:`q`
is orthogonal to the image gradient at 
:math:`p`
subject to image and measurement noise. Consider the expression:



.. math::

    \epsilon _i = {DI_{p_i}}^T  \cdot (q - p_i) 


where 
:math:`{DI_{p_i}}`
is the image gradient at the one of the points 
:math:`p_i`
in a neighborhood of 
:math:`q`
. The value of 
:math:`q`
is to be found such that 
:math:`\epsilon_i`
is minimized. A system of equations may be set up with 
:math:`\epsilon_i`
set to zero:



.. math::

    \sum _i(DI_{p_i}  \cdot {DI_{p_i}}^T) q =  \sum _i(DI_{p_i}  \cdot {DI_{p_i}}^T  \cdot p_i) 


where the gradients are summed within a neighborhood ("search window") of 
:math:`q`
. Calling the first gradient term 
:math:`G`
and the second gradient term 
:math:`b`
gives:



.. math::

    q = G^{-1}  \cdot b 


The algorithm sets the center of the neighborhood window at this new center 
:math:`q`
and then iterates until the center keeps within a set threshold.


.. index:: GoodFeaturesToTrack

.. _GoodFeaturesToTrack:

GoodFeaturesToTrack
-------------------

`id=0.0875265840344 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/py/imgproc/GoodFeaturesToTrack>`__


.. function:: GoodFeaturesToTrack(image,eigImage,tempImage,cornerCount,qualityLevel,minDistance,mask=NULL,blockSize=3,useHarris=0,k=0.04)-> corners

    Determines strong corners on an image.





    
    :param image: The source 8-bit or floating-point 32-bit, single-channel image 
    
    :type image: :class:`CvArr`
    
    
    :param eigImage: Temporary floating-point 32-bit image, the same size as  ``image`` 
    
    :type eigImage: :class:`CvArr`
    
    
    :param tempImage: Another temporary image, the same size and format as  ``eigImage`` 
    
    :type tempImage: :class:`CvArr`
    
    
    :param cornerCount: number of corners to detect 
    
    :type cornerCount: int
    
    
    :param qualityLevel: Multiplier for the max/min eigenvalue; specifies the minimal accepted quality of image corners 
    
    :type qualityLevel: float
    
    
    :param minDistance: Limit, specifying the minimum possible distance between the returned corners; Euclidian distance is used 
    
    :type minDistance: float
    
    
    :param mask: Region of interest. The function selects points either in the specified region or in the whole image if the mask is NULL 
    
    :type mask: :class:`CvArr`
    
    
    :param blockSize: Size of the averaging block, passed to the underlying  :ref:`CornerMinEigenVal`  or  :ref:`CornerHarris`  used by the function 
    
    :type blockSize: int
    
    
    :param useHarris: If nonzero, Harris operator ( :ref:`CornerHarris` ) is used instead of default  :ref:`CornerMinEigenVal` 
    
    :type useHarris: int
    
    
    :param k: Free parameter of Harris detector; used only if ( :math:`\texttt{useHarris} != 0` ) 
    
    :type k: float
    
    
    
The function finds the corners with big eigenvalues in the image. The function first calculates the minimal
eigenvalue for every source image pixel using the 
:ref:`CornerMinEigenVal`
function and stores them in 
``eigImage``
. Then it performs
non-maxima suppression (only the local maxima in 
:math:`3\times 3`
neighborhood
are retained). The next step rejects the corners with the minimal
eigenvalue less than 
:math:`\texttt{qualityLevel} \cdot max(\texttt{eigImage}(x,y))`
.
Finally, the function ensures that the distance between any two corners is not smaller than 
``minDistance``
. The weaker corners (with a smaller min eigenvalue) that are too close to the stronger corners are rejected.

Note that the if the function is called with different values 
``A``
and 
``B``
of the parameter 
``qualityLevel``
, and 
``A``
> {B}, the array of returned corners with 
``qualityLevel=A``
will be the prefix of the output corners array with 
``qualityLevel=B``
.


.. index:: HoughLines2

.. _HoughLines2:

HoughLines2
-----------

`id=0.925466467327 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/py/imgproc/HoughLines2>`__


.. function:: HoughLines2(image,storage,method,rho,theta,threshold,param1=0,param2=0)-> lines

    Finds lines in a binary image using a Hough transform.





    
    :param image: The 8-bit, single-channel, binary source image. In the case of a probabilistic method, the image is modified by the function 
    
    :type image: :class:`CvArr`
    
    
    :param storage: The storage for the lines that are detected. It can
        be a memory storage (in this case a sequence of lines is created in
        the storage and returned by the function) or single row/single column
        matrix (CvMat*) of a particular type (see below) to which the lines'
        parameters are written. The matrix header is modified by the function
        so its  ``cols``  or  ``rows``  will contain the number of lines
        detected. If  ``storage``  is a matrix and the actual number
        of lines exceeds the matrix size, the maximum possible number of lines
        is returned (in the case of standard hough transform the lines are sorted
        by the accumulator value) 
    
    :type storage: :class:`CvMemStorage`
    
    
    :param method: The Hough transform variant, one of the following: 
        
                
            * **CV_HOUGH_STANDARD** classical or standard Hough transform. Every line is represented by two floating-point numbers  :math:`(\rho, \theta)` , where  :math:`\rho`  is a distance between (0,0) point and the line, and  :math:`\theta`  is the angle between x-axis and the normal to the line. Thus, the matrix must be (the created sequence will be) of  ``CV_32FC2``  type 
            
               
            * **CV_HOUGH_PROBABILISTIC** probabilistic Hough transform (more efficient in case if picture contains a few long linear segments). It returns line segments rather than the whole line. Each segment is represented by starting and ending points, and the matrix must be (the created sequence will be) of  ``CV_32SC4``  type 
            
               
            * **CV_HOUGH_MULTI_SCALE** multi-scale variant of the classical Hough transform. The lines are encoded the same way as  ``CV_HOUGH_STANDARD`` 
            
            
    
    :type method: int
    
    
    :param rho: Distance resolution in pixel-related units 
    
    :type rho: float
    
    
    :param theta: Angle resolution measured in radians 
    
    :type theta: float
    
    
    :param threshold: Threshold parameter. A line is returned by the function if the corresponding accumulator value is greater than  ``threshold`` 
    
    :type threshold: int
    
    
    :param param1: The first method-dependent parameter: 
        
               
        
        *  For the classical Hough transform it is not used (0).
               
        
        *  For the probabilistic Hough transform it is the minimum line length.
               
        
        *  For the multi-scale Hough transform it is the divisor for the distance resolution  :math:`\rho` . (The coarse distance resolution will be  :math:`\rho`  and the accurate resolution will be  :math:`(\rho / \texttt{param1})` ). 
            
    
    :type param1: float
    
    
    :param param2: The second method-dependent parameter: 
        
               
        
        *  For the classical Hough transform it is not used (0).
               
        
        *  For the probabilistic Hough transform it is the maximum gap between line segments lying on the same line to treat them as a single line segment (i.e. to join them).
               
        
        *  For the multi-scale Hough transform it is the divisor for the angle resolution  :math:`\theta` . (The coarse angle resolution will be  :math:`\theta`  and the accurate resolution will be  :math:`(\theta / \texttt{param2})` ). 
            
    
    :type param2: float
    
    
    
The function implements a few variants of the Hough transform for line detection.


.. index:: PreCornerDetect

.. _PreCornerDetect:

PreCornerDetect
---------------

`id=0.420590326716 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/py/imgproc/PreCornerDetect>`__


.. function:: PreCornerDetect(image,corners,apertureSize=3)-> None

    Calculates the feature map for corner detection.





    
    :param image: Input image 
    
    :type image: :class:`CvArr`
    
    
    :param corners: Image to store the corner candidates 
    
    :type corners: :class:`CvArr`
    
    
    :param apertureSize: Aperture parameter for the Sobel operator (see  :ref:`Sobel` ) 
    
    :type apertureSize: int
    
    
    
The function calculates the function



.. math::

    D_x^2 D_{yy} + D_y^2 D_{xx} - 2 D_x D_y D_{xy} 


where 
:math:`D_?`
denotes one of the first image derivatives and 
:math:`D_{??}`
denotes a second image derivative.

The corners can be found as local maximums of the function below:

.. include:: /Users/vp/Projects/ocv/opencv/doc/python_fragments/precornerdetect.py
    :literal:


