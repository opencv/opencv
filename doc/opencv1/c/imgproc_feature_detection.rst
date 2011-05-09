Feature Detection
=================

.. highlight:: c



.. index:: Canny

.. _Canny:

Canny
-----

`id=0.99528666363 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/Canny>`__




.. cfunction:: void cvCanny( const CvArr* image,  CvArr* edges,  double threshold1,  double threshold2,  int aperture_size=3 )

    Implements the Canny algorithm for edge detection.





    
    :param image: Single-channel input image 
    
    
    :param edges: Single-channel image to store the edges found by the function 
    
    
    :param threshold1: The first threshold 
    
    
    :param threshold2: The second threshold 
    
    
    :param aperture_size: Aperture parameter for the Sobel operator (see  :ref:`Sobel` ) 
    
    
    
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

`id=0.880597486716 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/CornerEigenValsAndVecs>`__




.. cfunction:: void cvCornerEigenValsAndVecs(  const CvArr* image, CvArr* eigenvv, int blockSize, int aperture_size=3 )

    Calculates eigenvalues and eigenvectors of image blocks for corner detection.





    
    :param image: Input image 
    
    
    :param eigenvv: Image to store the results. It must be 6 times wider than the input image 
    
    
    :param blockSize: Neighborhood size (see discussion) 
    
    
    :param aperture_size: Aperture parameter for the Sobel operator (see  :ref:`Sobel` ) 
    
    
    
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

`id=0.765194293954 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/CornerHarris>`__




.. cfunction:: void cvCornerHarris(  const CvArr* image,  CvArr* harris_dst,  int blockSize,  int aperture_size=3,  double k=0.04 )

    Harris edge detector.





    
    :param image: Input image 
    
    
    :param harris_dst: Image to store the Harris detector responses. Should have the same size as  ``image`` 
    
    
    :param blockSize: Neighborhood size (see the discussion of  :ref:`CornerEigenValsAndVecs` ) 
    
    
    :param aperture_size: Aperture parameter for the Sobel operator (see  :ref:`Sobel` ). 
    
    
    :param k: Harris detector free parameter. See the formula below 
    
    
    
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

`id=0.956867089452 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/CornerMinEigenVal>`__




.. cfunction:: void cvCornerMinEigenVal(  const CvArr* image,  CvArr* eigenval,  int blockSize,  int aperture_size=3 )

    Calculates the minimal eigenvalue of gradient matrices for corner detection.





    
    :param image: Input image 
    
    
    :param eigenval: Image to store the minimal eigenvalues. Should have the same size as  ``image`` 
    
    
    :param blockSize: Neighborhood size (see the discussion of  :ref:`CornerEigenValsAndVecs` ) 
    
    
    :param aperture_size: Aperture parameter for the Sobel operator (see  :ref:`Sobel` ). 
    
    
    
The function is similar to 
:ref:`CornerEigenValsAndVecs`
but it calculates and stores only the minimal eigen value of derivative covariation matrix for every pixel, i.e. 
:math:`min(\lambda_1, \lambda_2)`
in terms of the previous function.


.. index:: FindCornerSubPix

.. _FindCornerSubPix:

FindCornerSubPix
----------------

`id=0.941466183497 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/FindCornerSubPix>`__




.. cfunction:: void cvFindCornerSubPix(  const CvArr* image,  CvPoint2D32f* corners,  int count,  CvSize win,  CvSize zero_zone,  CvTermCriteria criteria )

    Refines the corner locations.





    
    :param image: Input image 
    
    
    :param corners: Initial coordinates of the input corners; refined coordinates on output 
    
    
    :param count: Number of corners 
    
    
    :param win: Half of the side length of the search window. For example, if  ``win`` =(5,5), then a  :math:`5*2+1 \times 5*2+1 = 11 \times 11`  search window would be used 
    
    
    :param zero_zone: Half of the size of the dead region in the middle of the search zone over which the summation in the formula below is not done. It is used sometimes to avoid possible singularities of the autocorrelation matrix. The value of (-1,-1) indicates that there is no such size 
    
    
    :param criteria: Criteria for termination of the iterative process of corner refinement. That is, the process of corner position refinement stops either after a certain number of iterations or when a required accuracy is achieved. The  ``criteria``  may specify either of or both the maximum number of iteration and the required accuracy 
    
    
    
The function iterates to find the sub-pixel accurate location of corners, or radial saddle points, as shown in on the picture below.


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

`id=0.0876392134647 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/GoodFeaturesToTrack>`__




.. cfunction:: void cvGoodFeaturesToTrack(  const CvArr* image  CvArr* eigImage, CvArr* tempImage  CvPoint2D32f* corners  int* cornerCount  double qualityLevel  double minDistance  const CvArr* mask=NULL  int blockSize=3  int useHarris=0  double k=0.04 )

    Determines strong corners on an image.





    
    :param image: The source 8-bit or floating-point 32-bit, single-channel image 
    
    
    :param eigImage: Temporary floating-point 32-bit image, the same size as  ``image`` 
    
    
    :param tempImage: Another temporary image, the same size and format as  ``eigImage`` 
    
    
    :param corners: Output parameter; detected corners 
    
    
    :param cornerCount: Output parameter; number of detected corners 
    
    
    :param qualityLevel: Multiplier for the max/min eigenvalue; specifies the minimal accepted quality of image corners 
    
    
    :param minDistance: Limit, specifying the minimum possible distance between the returned corners; Euclidian distance is used 
    
    
    :param mask: Region of interest. The function selects points either in the specified region or in the whole image if the mask is NULL 
    
    
    :param blockSize: Size of the averaging block, passed to the underlying  :ref:`CornerMinEigenVal`  or  :ref:`CornerHarris`  used by the function 
    
    
    :param useHarris: If nonzero, Harris operator ( :ref:`CornerHarris` ) is used instead of default  :ref:`CornerMinEigenVal` 
    
    
    :param k: Free parameter of Harris detector; used only if ( :math:`\texttt{useHarris} != 0` ) 
    
    
    
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

`id=0.689753287363 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/HoughLines2>`__




.. cfunction:: CvSeq* cvHoughLines2(  CvArr* image, void* storage, int method, double rho, double theta, int threshold, double param1=0, double param2=0 )

    Finds lines in a binary image using a Hough transform.





    
    :param image: The 8-bit, single-channel, binary source image. In the case of a probabilistic method, the image is modified by the function 
    
    
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
    
    
    :param method: The Hough transform variant, one of the following: 
        
                
            * **CV_HOUGH_STANDARD** classical or standard Hough transform. Every line is represented by two floating-point numbers  :math:`(\rho, \theta)` , where  :math:`\rho`  is a distance between (0,0) point and the line, and  :math:`\theta`  is the angle between x-axis and the normal to the line. Thus, the matrix must be (the created sequence will be) of  ``CV_32FC2``  type 
            
               
            * **CV_HOUGH_PROBABILISTIC** probabilistic Hough transform (more efficient in case if picture contains a few long linear segments). It returns line segments rather than the whole line. Each segment is represented by starting and ending points, and the matrix must be (the created sequence will be) of  ``CV_32SC4``  type 
            
               
            * **CV_HOUGH_MULTI_SCALE** multi-scale variant of the classical Hough transform. The lines are encoded the same way as  ``CV_HOUGH_STANDARD`` 
            
            
    
    
    :param rho: Distance resolution in pixel-related units 
    
    
    :param theta: Angle resolution measured in radians 
    
    
    :param threshold: Threshold parameter. A line is returned by the function if the corresponding accumulator value is greater than  ``threshold`` 
    
    
    :param param1: The first method-dependent parameter: 
        
               
        
        *  For the classical Hough transform it is not used (0).
               
        
        *  For the probabilistic Hough transform it is the minimum line length.
               
        
        *  For the multi-scale Hough transform it is the divisor for the distance resolution  :math:`\rho` . (The coarse distance resolution will be  :math:`\rho`  and the accurate resolution will be  :math:`(\rho / \texttt{param1})` ). 
            
    
    
    :param param2: The second method-dependent parameter: 
        
               
        
        *  For the classical Hough transform it is not used (0).
               
        
        *  For the probabilistic Hough transform it is the maximum gap between line segments lying on the same line to treat them as a single line segment (i.e. to join them).
               
        
        *  For the multi-scale Hough transform it is the divisor for the angle resolution  :math:`\theta` . (The coarse angle resolution will be  :math:`\theta`  and the accurate resolution will be  :math:`(\theta / \texttt{param2})` ). 
            
    
    
    
The function implements a few variants of the Hough transform for line detection.

**Example. Detecting lines with Hough transform.**



::


    
    /* This is a standalone program. Pass an image name as a first parameter
    of the program.  Switch between standard and probabilistic Hough transform
    by changing "#if 1" to "#if 0" and back */
    #include <cv.h>
    #include <highgui.h>
    #include <math.h>
    
    int main(int argc, char** argv)
    {
        IplImage* src;
        if( argc == 2 && (src=cvLoadImage(argv[1], 0))!= 0)
        {
            IplImage* dst = cvCreateImage( cvGetSize(src), 8, 1 );
            IplImage* color_dst = cvCreateImage( cvGetSize(src), 8, 3 );
            CvMemStorage* storage = cvCreateMemStorage(0);
            CvSeq* lines = 0;
            int i;
            cvCanny( src, dst, 50, 200, 3 );
            cvCvtColor( dst, color_dst, CV_GRAY2BGR );
    #if 1
            lines = cvHoughLines2( dst,
                                   storage,
                                   CV_HOUGH_STANDARD,
                                   1,
                                   CV_PI/180,
                                   100,
                                   0,
                                   0 );
    
            for( i = 0; i < MIN(lines->total,100); i++ )
            {
                float* line = (float*)cvGetSeqElem(lines,i);
                float rho = line[0];
                float theta = line[1];
                CvPoint pt1, pt2;
                double a = cos(theta), b = sin(theta);
                double x0 = a*rho, y0 = b*rho;
                pt1.x = cvRound(x0 + 1000*(-b));
                pt1.y = cvRound(y0 + 1000*(a));
                pt2.x = cvRound(x0 - 1000*(-b));
                pt2.y = cvRound(y0 - 1000*(a));
                cvLine( color_dst, pt1, pt2, CV_RGB(255,0,0), 3, 8 );
            }
    #else
            lines = cvHoughLines2( dst,
                                   storage,
                                   CV_HOUGH_PROBABILISTIC,
                                   1,
                                   CV_PI/180,
                                   80,
                                   30,
                                   10 );
            for( i = 0; i < lines->total; i++ )
            {
                CvPoint* line = (CvPoint*)cvGetSeqElem(lines,i);
                cvLine( color_dst, line[0], line[1], CV_RGB(255,0,0), 3, 8 );
            }
    #endif
            cvNamedWindow( "Source", 1 );
            cvShowImage( "Source", src );
    
            cvNamedWindow( "Hough", 1 );
            cvShowImage( "Hough", color_dst );
    
            cvWaitKey(0);
        }
    }
    

..

This is the sample picture the function parameters have been tuned for:



.. image:: ../pics/building.jpg



And this is the output of the above program in the case of probabilistic Hough transform (
``#if 0``
case):



.. image:: ../pics/houghp.png




.. index:: PreCornerDetect

.. _PreCornerDetect:

PreCornerDetect
---------------

`id=0.671562199289 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/PreCornerDetect>`__




.. cfunction:: void cvPreCornerDetect(  const CvArr* image,  CvArr* corners,  int apertureSize=3 )

    Calculates the feature map for corner detection.





    
    :param image: Input image 
    
    
    :param corners: Image to store the corner candidates 
    
    
    :param apertureSize: Aperture parameter for the Sobel operator (see  :ref:`Sobel` ) 
    
    
    
The function calculates the function



.. math::

    D_x^2 D_{yy} + D_y^2 D_{xx} - 2 D_x D_y D_{xy} 


where 
:math:`D_?`
denotes one of the first image derivatives and 
:math:`D_{??}`
denotes a second image derivative.

The corners can be found as local maximums of the function below:




::


    
    // assume that the image is floating-point
    IplImage* corners = cvCloneImage(image);
    IplImage* dilated_corners = cvCloneImage(image);
    IplImage* corner_mask = cvCreateImage( cvGetSize(image), 8, 1 );
    cvPreCornerDetect( image, corners, 3 );
    cvDilate( corners, dilated_corners, 0, 1 );
    cvSubS( corners, dilated_corners, corners );
    cvCmpS( corners, 0, corner_mask, CV_CMP_GE );
    cvReleaseImage( &corners );
    cvReleaseImage( &dilated_corners );
    

..


.. index:: SampleLine

.. _SampleLine:

SampleLine
----------

`id=0.852353847021 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/SampleLine>`__




.. cfunction:: int cvSampleLine(  const CvArr* image  CvPoint pt1  CvPoint pt2  void* buffer  int connectivity=8 )

    Reads the raster line to the buffer.





    
    :param image: Image to sample the line from 
    
    
    :param pt1: Starting line point 
    
    
    :param pt2: Ending line point 
    
    
    :param buffer: Buffer to store the line points; must have enough size to store :math:`max( |\texttt{pt2.x} - \texttt{pt1.x}|+1, |\texttt{pt2.y} - \texttt{pt1.y}|+1 )` 
        points in the case of an 8-connected line and :math:`(|\texttt{pt2.x}-\texttt{pt1.x}|+|\texttt{pt2.y}-\texttt{pt1.y}|+1)` 
        in the case of a 4-connected line 
    
    
    :param connectivity: The line connectivity, 4 or 8 
    
    
    
The function implements a particular application of line iterators. The function reads all of the image points lying on the line between 
``pt1``
and 
``pt2``
, including the end points, and stores them into the buffer.

