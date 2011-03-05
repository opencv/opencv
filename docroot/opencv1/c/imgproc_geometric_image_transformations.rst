Geometric Image Transformations
===============================

.. highlight:: c


The functions in this section perform various geometrical transformations of 2D images. That is, they do not change the image content, but deform the pixel grid, and map this deformed grid to the destination image. In fact, to avoid sampling artifacts, the mapping is done in the reverse order, from destination to the source. That is, for each pixel 
:math:`(x, y)`
of the destination image, the functions compute coordinates of the corresponding "donor" pixel in the source image and copy the pixel value, that is:



.. math::

    \texttt{dst} (x,y)= \texttt{src} (f_x(x,y), f_y(x,y)) 


In the case when the user specifies the forward mapping: 
:math:`\left<g_x, g_y\right>: \texttt{src} \rightarrow \texttt{dst}`
, the OpenCV functions first compute the corresponding inverse mapping: 
:math:`\left<f_x, f_y\right>: \texttt{dst} \rightarrow \texttt{src}`
and then use the above formula.

The actual implementations of the geometrical transformations, from the most generic 
:ref:`Remap`
and to the simplest and the fastest 
:ref:`Resize`
, need to solve the 2 main problems with the above formula:


    

#.
    extrapolation of non-existing pixels. Similarly to the filtering functions, described in the previous section, for some 
    :math:`(x,y)`
    one of 
    :math:`f_x(x,y)`
    or 
    :math:`f_y(x,y)`
    , or they both, may fall outside of the image, in which case some extrapolation method needs to be used. OpenCV provides the same selection of the extrapolation methods as in the filtering functions, but also an additional method 
    ``BORDER_TRANSPARENT``
    , which means that the corresponding pixels in the destination image will not be modified at all.
        
    

#.
    interpolation of pixel values. Usually 
    :math:`f_x(x,y)`
    and 
    :math:`f_y(x,y)`
    are floating-point numbers (i.e. 
    :math:`\left<f_x, f_y\right>`
    can be an affine or perspective transformation, or radial lens distortion correction etc.), so a pixel values at fractional coordinates needs to be retrieved. In the simplest case the coordinates can be just rounded to the nearest integer coordinates and the corresponding pixel used, which is called nearest-neighbor interpolation. However, a better result can be achieved by using more sophisticated 
    `interpolation methods <http://en.wikipedia.org/wiki/Multivariate_interpolation>`_
    , where a polynomial function is fit into some neighborhood of the computed pixel 
    :math:`(f_x(x,y), f_y(x,y))`
    and then the value of the polynomial at 
    :math:`(f_x(x,y), f_y(x,y))`
    is taken as the interpolated pixel value. In OpenCV you can choose between several interpolation methods, see 
    :ref:`Resize`
    . 
    
    

.. index:: GetRotationMatrix2D

.. _GetRotationMatrix2D:

GetRotationMatrix2D
-------------------

`id=0.623450579574 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/GetRotationMatrix2D>`__




.. cfunction:: CvMat* cv2DRotationMatrix(  CvPoint2D32f center,  double angle,  double scale,  CvMat* mapMatrix )

    Calculates the affine matrix of 2d rotation.





    
    :param center: Center of the rotation in the source image 
    
    
    :param angle: The rotation angle in degrees. Positive values mean counter-clockwise rotation (the coordinate origin is assumed to be the top-left corner) 
    
    
    :param scale: Isotropic scale factor 
    
    
    :param mapMatrix: Pointer to the destination  :math:`2\times 3`  matrix 
    
    
    
The function 
``cv2DRotationMatrix``
calculates the following matrix:



.. math::

    \begin{bmatrix} \alpha &  \beta & (1- \alpha )  \cdot \texttt{center.x} -  \beta \cdot \texttt{center.y} \\ - \beta &  \alpha &  \beta \cdot \texttt{center.x} - (1- \alpha )  \cdot \texttt{center.y} \end{bmatrix} 


where



.. math::

    \alpha =  \texttt{scale} \cdot cos( \texttt{angle} ),  \beta =  \texttt{scale} \cdot sin( \texttt{angle} ) 


The transformation maps the rotation center to itself. If this is not the purpose, the shift should be adjusted.


.. index:: GetAffineTransform

.. _GetAffineTransform:

GetAffineTransform
------------------

`id=0.933805421933 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/GetAffineTransform>`__




.. cfunction:: CvMat* cvGetAffineTransform(  const CvPoint2D32f* src,  const CvPoint2D32f* dst,   CvMat* mapMatrix )

    Calculates the affine transform from 3 corresponding points.





    
    :param src:  Coordinates of 3 triangle vertices in the source image 
    
    
    :param dst:  Coordinates of the 3 corresponding triangle vertices in the destination image 
    
    
    :param mapMatrix:  Pointer to the destination  :math:`2 \times 3`  matrix 
    
    
    
The function cvGetAffineTransform calculates the matrix of an affine transform such that:



.. math::

    \begin{bmatrix} x'_i \\ y'_i \end{bmatrix} = \texttt{mapMatrix} \cdot \begin{bmatrix} x_i \\ y_i \\ 1 \end{bmatrix} 


where



.. math::

    dst(i)=(x'_i,y'_i),
    src(i)=(x_i, y_i),
    i=0,1,2 



.. index:: GetPerspectiveTransform

.. _GetPerspectiveTransform:

GetPerspectiveTransform
-----------------------

`id=0.709057737517 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/GetPerspectiveTransform>`__




.. cfunction:: CvMat* cvGetPerspectiveTransform(  const CvPoint2D32f* src,  const CvPoint2D32f* dst,  CvMat* mapMatrix )

    Calculates the perspective transform from 4 corresponding points.





    
    :param src: Coordinates of 4 quadrangle vertices in the source image 
    
    
    :param dst: Coordinates of the 4 corresponding quadrangle vertices in the destination image 
    
    
    :param mapMatrix: Pointer to the destination  :math:`3\times 3`  matrix 
    
    
    
The function 
``cvGetPerspectiveTransform``
calculates a matrix of perspective transforms such that:



.. math::

    \begin{bmatrix} x'_i \\ y'_i \end{bmatrix} = \texttt{mapMatrix} \cdot \begin{bmatrix} x_i \\ y_i \\ 1 \end{bmatrix} 


where



.. math::

    dst(i)=(x'_i,y'_i),
    src(i)=(x_i, y_i),
    i=0,1,2,3 



.. index:: GetQuadrangleSubPix

.. _GetQuadrangleSubPix:

GetQuadrangleSubPix
-------------------

`id=0.480550634961 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/GetQuadrangleSubPix>`__




.. cfunction:: void cvGetQuadrangleSubPix(  const CvArr* src,  CvArr* dst,  const CvMat* mapMatrix )

    Retrieves the pixel quadrangle from an image with sub-pixel accuracy.





    
    :param src: Source image 
    
    
    :param dst: Extracted quadrangle 
    
    
    :param mapMatrix: The transformation  :math:`2 \times 3`  matrix  :math:`[A|b]`  (see the discussion) 
    
    
    
The function 
``cvGetQuadrangleSubPix``
extracts pixels from 
``src``
at sub-pixel accuracy and stores them to 
``dst``
as follows:



.. math::

    dst(x, y)= src( A_{11} x' + A_{12} y' + b_1, A_{21} x' + A_{22} y' + b_2) 


where



.. math::

    x'=x- \frac{(width(dst)-1)}{2} , 
    y'=y- \frac{(height(dst)-1)}{2} 


and



.. math::

    \texttt{mapMatrix} =  \begin{bmatrix} A_{11} & A_{12} & b_1 \\ A_{21} & A_{22} & b_2 \end{bmatrix} 


The values of pixels at non-integer coordinates are retrieved using bilinear interpolation. When the function needs pixels outside of the image, it uses replication border mode to reconstruct the values. Every channel of multiple-channel images is processed independently.



.. index:: GetRectSubPix

.. _GetRectSubPix:

GetRectSubPix
-------------

`id=0.37305758361 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/GetRectSubPix>`__




.. cfunction:: void cvGetRectSubPix(  const CvArr* src,  CvArr* dst,  CvPoint2D32f center )

    Retrieves the pixel rectangle from an image with sub-pixel accuracy.
 




    
    :param src: Source image 
    
    
    :param dst: Extracted rectangle 
    
    
    :param center: Floating point coordinates of the extracted rectangle center within the source image. The center must be inside the image 
    
    
    
The function 
``cvGetRectSubPix``
extracts pixels from 
``src``
:



.. math::

    dst(x, y) = src(x +  \texttt{center.x} - (width( \texttt{dst} )-1)*0.5, y +  \texttt{center.y} - (height( \texttt{dst} )-1)*0.5) 


where the values of the pixels at non-integer coordinates are retrieved
using bilinear interpolation. Every channel of multiple-channel
images is processed independently. While the rectangle center
must be inside the image, parts of the rectangle may be
outside. In this case, the replication border mode is used to get
pixel values beyond the image boundaries.



.. index:: LogPolar

.. _LogPolar:

LogPolar
--------

`id=0.0887380164552 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/LogPolar>`__




.. cfunction:: void cvLogPolar(  const CvArr* src,  CvArr* dst,  CvPoint2D32f center,  double M,  int flags=CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS )

    Remaps an image to log-polar space.





    
    :param src: Source image 
    
    
    :param dst: Destination image 
    
    
    :param center: The transformation center; where the output precision is maximal 
    
    
    :param M: Magnitude scale parameter. See below 
    
    
    :param flags: A combination of interpolation methods and the following optional flags: 
        
                
            * **CV_WARP_FILL_OUTLIERS** fills all of the destination image pixels. If some of them correspond to outliers in the source image, they are set to zero 
            
               
            * **CV_WARP_INVERSE_MAP** See below 
            
            
    
    
    
The function 
``cvLogPolar``
transforms the source image using the following transformation:

Forward transformation (
``CV_WARP_INVERSE_MAP``
is not set):



.. math::

    dst( \phi , \rho ) = src(x,y) 


Inverse transformation (
``CV_WARP_INVERSE_MAP``
is set):



.. math::

    dst(x,y) = src( \phi , \rho ) 


where



.. math::

    \rho = M  \cdot \log{\sqrt{x^2 + y^2}} , \phi =atan(y/x) 


The function emulates the human "foveal" vision and can be used for fast scale and rotation-invariant template matching, for object tracking and so forth.
The function can not operate in-place.




::


    
    #include <cv.h>
    #include <highgui.h>
    
    int main(int argc, char** argv)
    {
        IplImage* src;
    
        if( argc == 2 && (src=cvLoadImage(argv[1],1) != 0 )
        {
            IplImage* dst = cvCreateImage( cvSize(256,256), 8, 3 );
            IplImage* src2 = cvCreateImage( cvGetSize(src), 8, 3 );
            cvLogPolar( src, dst, cvPoint2D32f(src->width/2,src->height/2), 40, 
            CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS );
            cvLogPolar( dst, src2, cvPoint2D32f(src->width/2,src->height/2), 40, 
            CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS+CV_WARP_INVERSE_MAP );
            cvNamedWindow( "log-polar", 1 );
            cvShowImage( "log-polar", dst );
            cvNamedWindow( "inverse log-polar", 1 );
            cvShowImage( "inverse log-polar", src2 );
            cvWaitKey();
        }
        return 0;
    }
    

..

And this is what the program displays when 
``opencv/samples/c/fruits.jpg``
is passed to it


.. image:: ../pics/logpolar.jpg





.. image:: ../pics/inv_logpolar.jpg




.. index:: Remap

.. _Remap:

Remap
-----

`id=0.485916549227 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/Remap>`__




.. cfunction:: void cvRemap(  const CvArr* src,  CvArr* dst,  const CvArr* mapx,  const CvArr* mapy,  int flags=CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS,  CvScalar fillval=cvScalarAll(0) )

    Applies a generic geometrical transformation to the image.





    
    :param src: Source image 
    
    
    :param dst: Destination image 
    
    
    :param mapx: The map of x-coordinates (CV _ 32FC1 image) 
    
    
    :param mapy: The map of y-coordinates (CV _ 32FC1 image) 
    
    
    :param flags: A combination of interpolation method and the following optional flag(s): 
        
                
            * **CV_WARP_FILL_OUTLIERS** fills all of the destination image pixels. If some of them correspond to outliers in the source image, they are set to  ``fillval`` 
            
            
    
    
    :param fillval: A value used to fill outliers 
    
    
    
The function 
``cvRemap``
transforms the source image using the specified map:



.. math::

    \texttt{dst} (x,y) =  \texttt{src} ( \texttt{mapx} (x,y), \texttt{mapy} (x,y)) 


Similar to other geometrical transformations, some interpolation method (specified by user) is used to extract pixels with non-integer coordinates.
Note that the function can not operate in-place.


.. index:: Resize

.. _Resize:

Resize
------

`id=0.249690626324 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/Resize>`__




.. cfunction:: void cvResize(  const CvArr* src,  CvArr* dst,  int interpolation=CV_INTER_LINEAR )

    Resizes an image.





    
    :param src: Source image 
    
    
    :param dst: Destination image 
    
    
    :param interpolation: Interpolation method: 
         
            * **CV_INTER_NN** nearest-neigbor interpolation 
            
            * **CV_INTER_LINEAR** bilinear interpolation (used by default) 
            
            * **CV_INTER_AREA** resampling using pixel area relation. It is the preferred method for image decimation that gives moire-free results. In terms of zooming it is similar to the  ``CV_INTER_NN``  method 
            
            * **CV_INTER_CUBIC** bicubic interpolation 
            
            
    
    
    
The function 
``cvResize``
resizes an image 
``src``
so that it fits exactly into 
``dst``
. If ROI is set, the function considers the ROI as supported.



.. index:: WarpAffine

.. _WarpAffine:

WarpAffine
----------

`id=0.0915967317176 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/WarpAffine>`__




.. cfunction:: void cvWarpAffine(  const CvArr* src,  CvArr* dst,  const CvMat* mapMatrix,  int flags=CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS,  CvScalar fillval=cvScalarAll(0) )

    Applies an affine transformation to an image.





    
    :param src: Source image 
    
    
    :param dst: Destination image 
    
    
    :param mapMatrix: :math:`2\times 3`  transformation matrix 
    
    
    :param flags: A combination of interpolation methods and the following optional flags: 
        
                
            * **CV_WARP_FILL_OUTLIERS** fills all of the destination image pixels; if some of them correspond to outliers in the source image, they are set to  ``fillval`` 
            
               
            * **CV_WARP_INVERSE_MAP** indicates that  ``matrix``  is inversely
                  transformed from the destination image to the source and, thus, can be used
                  directly for pixel interpolation. Otherwise, the function finds
                  the inverse transform from  ``mapMatrix`` 
            
        
        
        
    
    :param fillval: A value used to fill outliers 
    
    
    
The function 
``cvWarpAffine``
transforms the source image using the specified matrix:



.. math::

    dst(x',y') = src(x,y) 


where



.. math::

    \begin{matrix} \begin{bmatrix} x' \\ y' \end{bmatrix} =  \texttt{mapMatrix} \cdot \begin{bmatrix} x \\ y \\ 1 \end{bmatrix} &  \mbox{if CV\_WARP\_INVERSE\_MAP is not set} \\ \begin{bmatrix} x \\ y \end{bmatrix} =  \texttt{mapMatrix} \cdot \begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} &  \mbox{otherwise} \end{matrix} 


The function is similar to 
:ref:`GetQuadrangleSubPix`
but they are not exactly the same. 
:ref:`WarpAffine`
requires input and output image have the same data type, has larger overhead (so it is not quite suitable for small images) and can leave part of destination image unchanged. While 
:ref:`GetQuadrangleSubPix`
may extract quadrangles from 8-bit images into floating-point buffer, has smaller overhead and always changes the whole destination image content.
Note that the function can not operate in-place.

To transform a sparse set of points, use the 
:ref:`Transform`
function from cxcore.


.. index:: WarpPerspective

.. _WarpPerspective:

WarpPerspective
---------------

`id=0.647385091755 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/WarpPerspective>`__




.. cfunction:: void cvWarpPerspective(  const CvArr* src,  CvArr* dst,  const CvMat* mapMatrix,  int flags=CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS,  CvScalar fillval=cvScalarAll(0) )

    Applies a perspective transformation to an image.





    
    :param src: Source image 
    
    
    :param dst: Destination image 
    
    
    :param mapMatrix: :math:`3\times 3`  transformation matrix 
    
    
    :param flags: A combination of interpolation methods and the following optional flags: 
        
                
            * **CV_WARP_FILL_OUTLIERS** fills all of the destination image pixels; if some of them correspond to outliers in the source image, they are set to  ``fillval`` 
            
               
            * **CV_WARP_INVERSE_MAP** indicates that  ``matrix``  is inversely transformed from the destination image to the source and, thus, can be used directly for pixel interpolation. Otherwise, the function finds the inverse transform from  ``mapMatrix`` 
            
            
    
    
    :param fillval: A value used to fill outliers 
    
    
    
The function 
``cvWarpPerspective``
transforms the source image using the specified matrix:



.. math::

    \begin{matrix} \begin{bmatrix} x' \\ y' \end{bmatrix} =  \texttt{mapMatrix} \cdot \begin{bmatrix} x \\ y \\ 1 \end{bmatrix} &  \mbox{if CV\_WARP\_INVERSE\_MAP is not set} \\ \begin{bmatrix} x \\ y \end{bmatrix} =  \texttt{mapMatrix} \cdot \begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} &  \mbox{otherwise} \end{matrix} 


Note that the function can not operate in-place.
For a sparse set of points use the 
:ref:`PerspectiveTransform`
function from CxCore.

