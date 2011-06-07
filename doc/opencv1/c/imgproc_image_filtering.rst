Image Filtering
===============

.. highlight:: c


Functions and classes described in this section are used to perform various linear or non-linear filtering operations on 2D images (represented as 
:cpp:func:`Mat`
's), that is, for each pixel location 
:math:`(x,y)`
in the source image some its (normally rectangular) neighborhood is considered and used to compute the response. In case of a linear filter it is a weighted sum of pixel values, in case of morphological operations it is the minimum or maximum etc. The computed response is stored to the destination image at the same location 
:math:`(x,y)`
. It means, that the output image will be of the same size as the input image. Normally, the functions supports multi-channel arrays, in which case every channel is processed independently, therefore the output image will also have the same number of channels as the input one.

Another common feature of the functions and classes described in this section is that, unlike simple arithmetic functions, they need to extrapolate values of some non-existing pixels. For example, if we want to smooth an image using a Gaussian 
:math:`3 \times 3`
filter, then during the processing of the left-most pixels in each row we need pixels to the left of them, i.e. outside of the image. We can let those pixels be the same as the left-most image pixels (i.e. use "replicated border" extrapolation method), or assume that all the non-existing pixels are zeros ("contant border" extrapolation method) etc. 

.. index:: IplConvKernel

.. _IplConvKernel:

IplConvKernel
-------------



.. ctype:: IplConvKernel



An IplConvKernel is a rectangular convolution kernel, created by function 
:ref:`CreateStructuringElementEx`
.


.. index:: CopyMakeBorder

.. _CopyMakeBorder:

CopyMakeBorder
--------------






.. cfunction:: void cvCopyMakeBorder(  const CvArr* src,  CvArr* dst,  CvPoint offset,  int bordertype,  CvScalar value=cvScalarAll(0) )

    Copies an image and makes a border around it.





    
    :param src: The source image 
    
    
    :param dst: The destination image 
    
    
    :param offset: Coordinates of the top-left corner (or bottom-left in the case of images with bottom-left origin) of the destination image rectangle where the source image (or its ROI) is copied. Size of the rectanlge matches the source image size/ROI size 
    
    
    :param bordertype: Type of the border to create around the copied source image rectangle; types include: 
         
            * **IPL_BORDER_CONSTANT** border is filled with the fixed value, passed as last parameter of the function. 
            
            * **IPL_BORDER_REPLICATE** the pixels from the top and bottom rows, the left-most and right-most columns are replicated to fill the border. 
            
            
        (The other two border types from IPL,  ``IPL_BORDER_REFLECT``  and  ``IPL_BORDER_WRAP`` , are currently unsupported) 
    
    
    :param value: Value of the border pixels if  ``bordertype``  is  ``IPL_BORDER_CONSTANT`` 
    
    
    
The function copies the source 2D array into the interior of the destination array and makes a border of the specified type around the copied area. The function is useful when one needs to emulate border type that is different from the one embedded into a specific algorithm implementation. For example, morphological functions, as well as most of other filtering functions in OpenCV, internally use replication border type, while the user may need a zero border or a border, filled with 1's or 255's.


.. index:: CreateStructuringElementEx

.. _CreateStructuringElementEx:

CreateStructuringElementEx
--------------------------






.. cfunction:: IplConvKernel* cvCreateStructuringElementEx( int cols,   int rows,  int anchorX,  int anchorY,  int shape,  int* values=NULL )

    Creates a structuring element.





    
    :param cols: Number of columns in the structuring element 
    
    
    :param rows: Number of rows in the structuring element 
    
    
    :param anchorX: Relative horizontal offset of the anchor point 
    
    
    :param anchorY: Relative vertical offset of the anchor point 
    
    
    :param shape: Shape of the structuring element; may have the following values: 
        
                
            * **CV_SHAPE_RECT** a rectangular element 
            
               
            * **CV_SHAPE_CROSS** a cross-shaped element 
            
               
            * **CV_SHAPE_ELLIPSE** an elliptic element 
            
               
            * **CV_SHAPE_CUSTOM** a user-defined element. In this case the parameter  ``values``  specifies the mask, that is, which neighbors of the pixel must be considered 
            
            
    
    
    :param values: Pointer to the structuring element data, a plane array, representing row-by-row scanning of the element matrix. Non-zero values indicate points that belong to the element. If the pointer is  ``NULL`` , then all values are considered non-zero, that is, the element is of a rectangular shape. This parameter is considered only if the shape is  ``CV_SHAPE_CUSTOM``   
    
    
    
The function CreateStructuringElementEx allocates and fills the structure 
``IplConvKernel``
, which can be used as a structuring element in the morphological operations.


.. index:: Dilate

.. _Dilate:

Dilate
------


.. cfunction:: void cvDilate( const CvArr* src, CvArr* dst, IplConvKernel* element=NULL, int iterations=1 )

    Dilates an image by using a specific structuring element.
    
    :param src: Source image 
    
    :param dst: Destination image 
    
    :param element: Structuring element used for dilation. If it is ``NULL``,  a ``3 x 3``  rectangular structuring element is used 
    
    :param iterations: Number of times dilation is applied 
    
    
The function dilates the source image using the specified structuring element that determines the shape of a pixel neighborhood over which the maximum is taken:


.. math::

    \max _{(x',y')  \, in  \, \texttt{element} }src(x+x',y+y') 


The function supports the in-place mode. Dilation can be applied several (``iterations``) times. For color images, each channel is processed independently.


.. index:: Erode

.. _Erode:

Erode
-----






.. cfunction:: void cvErode( const CvArr* src,  CvArr* dst,  IplConvKernel* element=NULL,  int iterations=1)

    Erodes an image by using a specific structuring element.

    
    :param src: Source image 
    
    :param dst: Destination image 
    
    :param element: Structuring element used for erosion. If it is ``NULL`` , a  ``3 x 3`` rectangular structuring element is used 
    
    :param iterations: Number of times erosion is applied
    
The function erodes the source image using the specified structuring element that determines the shape of a pixel neighborhood over which the minimum is taken:


.. math::

    \min _{(x',y')  \, in  \, \texttt{element} }src(x+x',y+y') 


The function supports the in-place mode. Erosion can be applied several (``iterations``) times. For color images, each channel is processed independently.


.. index:: Filter2D

.. _Filter2D:

Filter2D
--------






.. cfunction:: void cvFilter2D(  const CvArr* src,  CvArr* dst,  const CvMat* kernel,  CvPoint anchor=cvPoint(-1,-1))

    Convolves an image with the kernel.





    
    :param src: The source image 
    
    
    :param dst: The destination image 
    
    
    :param kernel: Convolution kernel, a single-channel floating point matrix. If you want to apply different kernels to different channels, split the image into separate color planes using  :ref:`Split`  and process them individually 
    
    
    :param anchor: The anchor of the kernel that indicates the relative position of a filtered point within the kernel. The anchor shoud lie within the kernel. The special default value (-1,-1) means that it is at the kernel center 
    
    
    
The function applies an arbitrary linear filter to the image. In-place operation is supported. When the aperture is partially outside the image, the function interpolates outlier pixel values from the nearest pixels that are inside the image.


.. index:: Laplace

.. _Laplace:

Laplace
-------






.. cfunction:: void cvLaplace( const CvArr* src,  CvArr* dst,  int apertureSize=3)

    Calculates the Laplacian of an image.





    
    :param src: Source image 
    
    
    :param dst: Destination image 
    
    
    :param apertureSize: Aperture size (it has the same meaning as  :ref:`Sobel` ) 
    
    
    
The function calculates the Laplacian of the source image by adding up the second x and y derivatives calculated using the Sobel operator:



.. math::

    \texttt{dst} (x,y) =  \frac{d^2 \texttt{src}}{dx^2} +  \frac{d^2 \texttt{src}}{dy^2} 


Setting 
``apertureSize``
= 1 gives the fastest variant that is equal to convolving the image with the following kernel:



.. math::

    \vecthreethree {0}{1}{0}{1}{-4}{1}{0}{1}{0}  


Similar to the 
:ref:`Sobel`
function, no scaling is done and the same combinations of input and output formats are supported.


.. index:: MorphologyEx

.. _MorphologyEx:

MorphologyEx
------------






.. cfunction:: void cvMorphologyEx(  const CvArr* src,  CvArr* dst,  CvArr* temp,  IplConvKernel* element,  int operation,  int iterations=1 )

    Performs advanced morphological transformations.





    
    :param src: Source image 
    
    
    :param dst: Destination image 
    
    
    :param temp: Temporary image, required in some cases 
    
    
    :param element: Structuring element 
    
    
    :param operation: Type of morphological operation, one of the following: 
         
            * **CV_MOP_OPEN** opening 
            
            * **CV_MOP_CLOSE** closing 
            
            * **CV_MOP_GRADIENT** morphological gradient 
            
            * **CV_MOP_TOPHAT** "top hat" 
            
            * **CV_MOP_BLACKHAT** "black hat" 
            
            
    
    
    :param iterations: Number of times erosion and dilation are applied 
    
    
    
The function can perform advanced morphological transformations using erosion and dilation as basic operations.

Opening:



.. math::

    dst=open(src,element)=dilate(erode(src,element),element) 


Closing:



.. math::

    dst=close(src,element)=erode(dilate(src,element),element) 


Morphological gradient:



.. math::

    dst=morph \_ grad(src,element)=dilate(src,element)-erode(src,element) 


"Top hat":



.. math::

    dst=tophat(src,element)=src-open(src,element) 


"Black hat":



.. math::

    dst=blackhat(src,element)=close(src,element)-src 


The temporary image 
``temp``
is required for a morphological gradient and, in the case of in-place operation, for "top hat" and "black hat".


.. index:: PyrDown

.. _PyrDown:

PyrDown
-------






.. cfunction:: void cvPyrDown( const CvArr* src, CvArr* dst, int filter=CV_GAUSSIAN_5x5 )

    Downsamples an image.





    
    :param src: The source image 
    
    
    :param dst: The destination image, should have a half as large width and height than the source 
    
    
    :param filter: Type of the filter used for convolution; only  ``CV_GAUSSIAN_5x5``  is currently supported 
    
    
    
The function performs the downsampling step of the Gaussian pyramid decomposition. First it convolves the source image with the specified filter and then downsamples the image by rejecting even rows and columns.


.. index:: ReleaseStructuringElement

.. _ReleaseStructuringElement:

ReleaseStructuringElement
-------------------------






.. cfunction:: void cvReleaseStructuringElement( IplConvKernel** element )

    Deletes a structuring element.





    
    :param element: Pointer to the deleted structuring element 
    
    
    
The function releases the structure 
``IplConvKernel``
that is no longer needed. If 
``*element``
is 
``NULL``
, the function has no effect.

.. index:: Smooth

.. _Smooth:

Smooth
------






.. cfunction:: void cvSmooth( const CvArr* src,  CvArr* dst,  int smoothtype=CV_GAUSSIAN,  int param1=3,  int param2=0,  double param3=0,  double param4=0)

    Smooths the image in one of several ways.





    
    :param src: The source image 
    
    
    :param dst: The destination image 
    
    
    :param smoothtype: Type of the smoothing: 
        
                
            * **CV_BLUR_NO_SCALE** linear convolution with  :math:`\texttt{param1}\times\texttt{param2}`  box kernel (all 1's). If you want to smooth different pixels with different-size box kernels, you can use the integral image that is computed using  :ref:`Integral` 
            
               
            * **CV_BLUR** linear convolution with  :math:`\texttt{param1}\times\texttt{param2}`  box kernel (all 1's) with subsequent scaling by  :math:`1/(\texttt{param1}\cdot\texttt{param2})` 
            
               
            * **CV_GAUSSIAN** linear convolution with a  :math:`\texttt{param1}\times\texttt{param2}`  Gaussian kernel 
            
               
            * **CV_MEDIAN** median filter with a  :math:`\texttt{param1}\times\texttt{param1}`  square aperture 
            
               
            * **CV_BILATERAL** bilateral filter with a  :math:`\texttt{param1}\times\texttt{param1}`  square aperture, color sigma= ``param3``  and spatial sigma= ``param4`` . If  ``param1=0`` , the aperture square side is set to  ``cvRound(param4*1.5)*2+1`` . Information about bilateral filtering can be found at  http://www.dai.ed.ac.uk/CVonline/LOCAL\_COPIES/MANDUCHI1/Bilateral\_Filtering.html 
            
            
    
    
    :param param1: The first parameter of the smoothing operation, the aperture width. Must be a positive odd number (1, 3, 5, ...) 
    
    
    :param param2: The second parameter of the smoothing operation, the aperture height. Ignored by  ``CV_MEDIAN``  and  ``CV_BILATERAL``  methods. In the case of simple scaled/non-scaled and Gaussian blur if  ``param2``  is zero, it is set to  ``param1`` . Otherwise it must be a positive odd number. 
    
    
    :param param3: In the case of a Gaussian parameter this parameter may specify Gaussian  :math:`\sigma`  (standard deviation). If it is zero, it is calculated from the kernel size:  
        
        .. math::
        
            \sigma  = 0.3 (n/2 - 1) + 0.8  \quad   \text{where}   \quad  n= \begin{array}{l l} \mbox{\texttt{param1} for horizontal kernel} \\ \mbox{\texttt{param2} for vertical kernel} \end{array} 
        
        Using standard sigma for small kernels ( :math:`3\times 3`  to  :math:`7\times 7` ) gives better speed. If  ``param3``  is not zero, while  ``param1``  and  ``param2``  are zeros, the kernel size is calculated from the sigma (to provide accurate enough operation). 
    
    
    
The function smooths an image using one of several methods. Every of the methods has some features and restrictions listed below

Blur with no scaling works with single-channel images only and supports accumulation of 8-bit to 16-bit format (similar to 
:ref:`Sobel`
and 
:ref:`Laplace`
) and 32-bit floating point to 32-bit floating-point format.

Simple blur and Gaussian blur support 1- or 3-channel, 8-bit and 32-bit floating point images. These two methods can process images in-place.

Median and bilateral filters work with 1- or 3-channel 8-bit images and can not process images in-place.


.. index:: Sobel

.. _Sobel:

Sobel
-----






.. cfunction:: void cvSobel( const CvArr* src,  CvArr* dst,  int xorder,  int yorder,  int apertureSize=3 )

    Calculates the first, second, third or mixed image derivatives using an extended Sobel operator.





    
    :param src: Source image of type CvArr* 
    
    
    :param dst: Destination image 
    
    
    :param xorder: Order of the derivative x 
    
    
    :param yorder: Order of the derivative y 
    
    
    :param apertureSize: Size of the extended Sobel kernel, must be 1, 3, 5 or 7 
    
    
    
In all cases except 1, an 
:math:`\texttt{apertureSize} \times
\texttt{apertureSize}`
separable kernel will be used to calculate the
derivative. For 
:math:`\texttt{apertureSize} = 1`
a 
:math:`3 \times 1`
or 
:math:`1 \times 3`
a kernel is used (Gaussian smoothing is not done). There is also the special
value 
``CV_SCHARR``
(-1) that corresponds to a 
:math:`3\times3`
Scharr
filter that may give more accurate results than a 
:math:`3\times3`
Sobel. Scharr
aperture is



.. math::

    \vecthreethree{-3}{0}{3}{-10}{0}{10}{-3}{0}{3} 


for the x-derivative or transposed for the y-derivative.

The function calculates the image derivative by convolving the image with the appropriate kernel:



.. math::

    \texttt{dst} (x,y) =  \frac{d^{xorder+yorder} \texttt{src}}{dx^{xorder} \cdot dy^{yorder}} 


The Sobel operators combine Gaussian smoothing and differentiation
so the result is more or less resistant to the noise. Most often,
the function is called with (
``xorder``
= 1, 
``yorder``
= 0,
``apertureSize``
= 3) or (
``xorder``
= 0, 
``yorder``
= 1,
``apertureSize``
= 3) to calculate the first x- or y- image
derivative. The first case corresponds to a kernel of:



.. math::

    \vecthreethree{-1}{0}{1}{-2}{0}{2}{-1}{0}{1} 


and the second one corresponds to a kernel of:


.. math::

    \vecthreethree{-1}{-2}{-1}{0}{0}{0}{1}{2}{1} 


or a kernel of:


.. math::

    \vecthreethree{1}{2}{1}{0}{0}{0}{-1}{2}{-1} 


depending on the image origin (
``origin``
field of
``IplImage``
structure). No scaling is done, so the destination image
usually has larger numbers (in absolute values) than the source image does. To
avoid overflow, the function requires a 16-bit destination image if the
source image is 8-bit. The result can be converted back to 8-bit using the
:ref:`ConvertScale`
or the 
:ref:`ConvertScaleAbs`
function. Besides 8-bit images
the function can process 32-bit floating-point images. Both the source and the 
destination must be single-channel images of equal size or equal ROI size.

