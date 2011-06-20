Miscellaneous Image Transformations
===================================

.. highlight:: python



.. index:: AdaptiveThreshold

.. _AdaptiveThreshold:

AdaptiveThreshold
-----------------




.. function:: AdaptiveThreshold(src,dst,maxValue, adaptive_method=CV_ADAPTIVE_THRESH_MEAN_C, thresholdType=CV_THRESH_BINARY,blockSize=3,param1=5)-> None

    Applies an adaptive threshold to an array.





    
    :param src: Source image 
    
    :type src: :class:`CvArr`
    
    
    :param dst: Destination image 
    
    :type dst: :class:`CvArr`
    
    
    :param maxValue: Maximum value that is used with  ``CV_THRESH_BINARY``  and  ``CV_THRESH_BINARY_INV`` 
    
    :type maxValue: float
    
    
    :param adaptive_method: Adaptive thresholding algorithm to use:  ``CV_ADAPTIVE_THRESH_MEAN_C``  or  ``CV_ADAPTIVE_THRESH_GAUSSIAN_C``  (see the discussion) 
    
    :type adaptive_method: int
    
    
    :param thresholdType: Thresholding type; must be one of 
         
            * **CV_THRESH_BINARY** xxx 
            
            * **CV_THRESH_BINARY_INV** xxx 
            
            
    
    :type thresholdType: int
    
    
    :param blockSize: The size of a pixel neighborhood that is used to calculate a threshold value for the pixel: 3, 5, 7, and so on 
    
    :type blockSize: int
    
    
    :param param1: The method-dependent parameter. For the methods  ``CV_ADAPTIVE_THRESH_MEAN_C``  and  ``CV_ADAPTIVE_THRESH_GAUSSIAN_C``  it is a constant subtracted from the mean or weighted mean (see the discussion), though it may be negative 
    
    :type param1: float
    
    
    
The function transforms a grayscale image to a binary image according to the formulas:



    
    * **CV_THRESH_BINARY**  
        
        .. math::
        
             dst(x,y) =  \fork{\texttt{maxValue}}{if $src(x,y) > T(x,y)$}{0}{otherwise}   
        
        
    
    
    * **CV_THRESH_BINARY_INV**  
        
        .. math::
        
             dst(x,y) =  \fork{0}{if $src(x,y) > T(x,y)$}{\texttt{maxValue}}{otherwise}   
        
        
    
    
    
where 
:math:`T(x,y)`
is a threshold calculated individually for each pixel.

For the method 
``CV_ADAPTIVE_THRESH_MEAN_C``
it is the mean of a 
:math:`\texttt{blockSize} \times \texttt{blockSize}`
pixel neighborhood, minus 
``param1``
.

For the method 
``CV_ADAPTIVE_THRESH_GAUSSIAN_C``
it is the weighted sum (gaussian) of a 
:math:`\texttt{blockSize} \times \texttt{blockSize}`
pixel neighborhood, minus 
``param1``
.


.. index:: CvtColor

.. _CvtColor:

CvtColor
--------




.. function:: CvtColor(src,dst,code)-> None

    Converts an image from one color space to another.





    
    :param src: The source 8-bit (8u), 16-bit (16u) or single-precision floating-point (32f) image 
    
    :type src: :class:`CvArr`
    
    
    :param dst: The destination image of the same data type as the source. The number of channels may be different 
    
    :type dst: :class:`CvArr`
    
    
    :param code: Color conversion operation that can be specifed using  ``CV_ *src_color_space* 2 *dst_color_space*``  constants (see below) 
    
    :type code: int
    
    
    
The function converts the input image from one color
space to another. The function ignores the 
``colorModel``
and
``channelSeq``
fields of the 
``IplImage``
header, so the
source image color space should be specified correctly (including
order of the channels in the case of RGB space. For example, BGR means 24-bit
format with 
:math:`B_0, G_0, R_0, B_1, G_1, R_1, ...`
layout
whereas RGB means 24-format with 
:math:`R_0, G_0, B_0, R_1, G_1, B_1, ...`
layout).

The conventional range for R,G,B channel values is:



    

*
    0 to 255 for 8-bit images
    

*
    0 to 65535 for 16-bit images and
    

*
    0 to 1 for floating-point images.
    
    
Of course, in the case of linear transformations the range can be
specific, but in order to get correct results in the case of non-linear
transformations, the input image should be scaled.

The function can do the following transformations:



    

*
    Transformations within RGB space like adding/removing the alpha channel, reversing the channel order, conversion to/from 16-bit RGB color (R5:G6:B5 or R5:G5:B5), as well as conversion to/from grayscale using:
    
    
    .. math::
    
        \text{RGB[A] to Gray:} Y  \leftarrow 0.299  \cdot R + 0.587  \cdot G + 0.114  \cdot B 
    
    
    and
    
    
    .. math::
    
        \text{Gray to RGB[A]:} R  \leftarrow Y, G  \leftarrow Y, B  \leftarrow Y, A  \leftarrow 0 
    
    
    The conversion from a RGB image to gray is done with:
    
    
    
    ::
    
    
        
        cvCvtColor(src ,bwsrc, CV_RGB2GRAY)
        
    
    ..
    
    

*
    RGB 
    :math:`\leftrightarrow`
    CIE XYZ.Rec 709 with D65 white point (
    ``CV_BGR2XYZ, CV_RGB2XYZ, CV_XYZ2BGR, CV_XYZ2RGB``
    ):
    
    
    .. math::
    
        \begin{bmatrix} X  \\ Y  \\ Z \end{bmatrix} \leftarrow \begin{bmatrix} 0.412453 & 0.357580 & 0.180423 \\ 0.212671 & 0.715160 & 0.072169 \\ 0.019334 & 0.119193 & 0.950227 \end{bmatrix} \cdot \begin{bmatrix} R  \\ G  \\ B \end{bmatrix} 
    
    
    
    
    .. math::
    
        \begin{bmatrix} R  \\ G  \\ B \end{bmatrix} \leftarrow \begin{bmatrix} 3.240479 & -1.53715 & -0.498535 \\ -0.969256 &  1.875991 & 0.041556 \\ 0.055648 & -0.204043 & 1.057311 \end{bmatrix} \cdot \begin{bmatrix} X  \\ Y  \\ Z \end{bmatrix} 
    
    
    :math:`X`
    , 
    :math:`Y`
    and 
    :math:`Z`
    cover the whole value range (in the case of floating-point images 
    :math:`Z`
    may exceed 1).
    
    

*
    RGB 
    :math:`\leftrightarrow`
    YCrCb JPEG (a.k.a. YCC) (
    ``CV_BGR2YCrCb, CV_RGB2YCrCb, CV_YCrCb2BGR, CV_YCrCb2RGB``
    )
    
    
    .. math::
    
        Y  \leftarrow 0.299  \cdot R + 0.587  \cdot G + 0.114  \cdot B  
    
    
    
    
    .. math::
    
        Cr  \leftarrow (R-Y)  \cdot 0.713 + delta  
    
    
    
    
    .. math::
    
        Cb  \leftarrow (B-Y)  \cdot 0.564 + delta  
    
    
    
    
    .. math::
    
        R  \leftarrow Y + 1.403  \cdot (Cr - delta)  
    
    
    
    
    .. math::
    
        G  \leftarrow Y - 0.344  \cdot (Cr - delta) - 0.714  \cdot (Cb - delta)  
    
    
    
    
    .. math::
    
        B  \leftarrow Y + 1.773  \cdot (Cb - delta)  
    
    
    where
    
    
    .. math::
    
        delta =  \left \{ \begin{array}{l l} 128 &  \mbox{for 8-bit images} \\ 32768 &  \mbox{for 16-bit images} \\ 0.5 &  \mbox{for floating-point images} \end{array} \right . 
    
    
    Y, Cr and Cb cover the whole value range.
    
    

*
    RGB 
    :math:`\leftrightarrow`
    HSV (
    ``CV_BGR2HSV, CV_RGB2HSV, CV_HSV2BGR, CV_HSV2RGB``
    )
    in the case of 8-bit and 16-bit images
    R, G and B are converted to floating-point format and scaled to fit the 0 to 1 range
    
    
    .. math::
    
        V  \leftarrow max(R,G,B)  
    
    
    
    
    .. math::
    
        S  \leftarrow \fork{\frac{V-min(R,G,B)}{V}}{if $V \neq 0$}{0}{otherwise} 
    
    
    
    
    .. math::
    
        H  \leftarrow \forkthree{{60(G - B)}/{S}}{if $V=R$}{{120+60(B - R)}/{S}}{if $V=G$}{{240+60(R - G)}/{S}}{if $V=B$} 
    
    
    if 
    :math:`H<0`
    then 
    :math:`H \leftarrow H+360`
    On output 
    :math:`0 \leq V \leq 1`
    , 
    :math:`0 \leq S \leq 1`
    , 
    :math:`0 \leq H \leq 360`
    .
    
    The values are then converted to the destination data type:
    
    
        
    
    * 8-bit images
        
        
        .. math::
        
            V  \leftarrow 255 V, S  \leftarrow 255 S, H  \leftarrow H/2  \text{(to fit to 0 to 255)} 
        
        
        
    
    * 16-bit images (currently not supported)
        
        
        .. math::
        
            V <- 65535 V, S <- 65535 S, H <- H  
        
        
        
    
    * 32-bit images
        H, S, V are left as is
        
        
    

*
    RGB 
    :math:`\leftrightarrow`
    HLS (
    ``CV_BGR2HLS, CV_RGB2HLS, CV_HLS2BGR, CV_HLS2RGB``
    ).
    in the case of 8-bit and 16-bit images
    R, G and B are converted to floating-point format and scaled to fit the 0 to 1 range.
    
    
    .. math::
    
        V_{max}  \leftarrow {max}(R,G,B)  
    
    
    
    
    .. math::
    
        V_{min}  \leftarrow {min}(R,G,B)  
    
    
    
    
    .. math::
    
        L  \leftarrow \frac{V_{max} + V_{min}}{2} 
    
    
    
    
    .. math::
    
        S  \leftarrow \fork{\frac{V_{max} - V_{min}}{V_{max} + V_{min}}}{if $L < 0.5$}{\frac{V_{max} - V_{min}}{2 - (V_{max} + V_{min})}}{if $L \ge 0.5$} 
    
    
    
    
    .. math::
    
        H  \leftarrow \forkthree{{60(G - B)}/{S}}{if $V_{max}=R$}{{120+60(B - R)}/{S}}{if $V_{max}=G$}{{240+60(R - G)}/{S}}{if $V_{max}=B$} 
    
    
    if 
    :math:`H<0`
    then 
    :math:`H \leftarrow H+360`
    On output 
    :math:`0 \leq L \leq 1`
    , 
    :math:`0 \leq S \leq 1`
    , 
    :math:`0 \leq H \leq 360`
    .
    
    The values are then converted to the destination data type:
    
    
        
    
    * 8-bit images
        
        
        .. math::
        
            V  \leftarrow 255 V, S  \leftarrow 255 S, H  \leftarrow H/2  \text{(to fit to 0 to 255)} 
        
        
        
    
    * 16-bit images (currently not supported)
        
        
        .. math::
        
            V <- 65535 V, S <- 65535 S, H <- H  
        
        
        
    
    * 32-bit images
        H, S, V are left as is
        
        
    

*
    RGB 
    :math:`\leftrightarrow`
    CIE L*a*b* (
    ``CV_BGR2Lab, CV_RGB2Lab, CV_Lab2BGR, CV_Lab2RGB``
    )
    in the case of 8-bit and 16-bit images
    R, G and B are converted to floating-point format and scaled to fit the 0 to 1 range
    
    
    .. math::
    
        \vecthree{X}{Y}{Z} \leftarrow \vecthreethree{0.412453}{0.357580}{0.180423}{0.212671}{0.715160}{0.072169}{0.019334}{0.119193}{0.950227} \cdot \vecthree{R}{G}{B} 
    
    
    
    
    .. math::
    
        X  \leftarrow X/X_n,  \text{where} X_n = 0.950456  
    
    
    
    
    .. math::
    
        Z  \leftarrow Z/Z_n,  \text{where} Z_n = 1.088754  
    
    
    
    
    .. math::
    
        L  \leftarrow \fork{116*Y^{1/3}-16}{for $Y>0.008856$}{903.3*Y}{for $Y \le 0.008856$} 
    
    
    
    
    .. math::
    
        a  \leftarrow 500 (f(X)-f(Y)) + delta  
    
    
    
    
    .. math::
    
        b  \leftarrow 200 (f(Y)-f(Z)) + delta  
    
    
    where
    
    
    .. math::
    
        f(t)= \fork{t^{1/3}}{for $t>0.008856$}{7.787 t+16/116}{for $t<=0.008856$} 
    
    
    and
    
    
    .. math::
    
        delta =  \fork{128}{for 8-bit images}{0}{for floating-point images} 
    
    
    On output 
    :math:`0 \leq L \leq 100`
    , 
    :math:`-127 \leq a \leq 127`
    , 
    :math:`-127 \leq b \leq 127`
    The values are then converted to the destination data type:
    
    
        
    
    * 8-bit images
        
        
        .. math::
        
            L  \leftarrow L*255/100, a  \leftarrow a + 128, b  \leftarrow b + 128 
        
        
        
    
    * 16-bit images
        currently not supported
        
    
    * 32-bit images
        L, a, b are left as is
        
        
    

*
    RGB 
    :math:`\leftrightarrow`
    CIE L*u*v* (
    ``CV_BGR2Luv, CV_RGB2Luv, CV_Luv2BGR, CV_Luv2RGB``
    )
    in the case of 8-bit and 16-bit images
    R, G and B are converted to floating-point format and scaled to fit 0 to 1 range
    
    
    .. math::
    
        \vecthree{X}{Y}{Z} \leftarrow \vecthreethree{0.412453}{0.357580}{0.180423}{0.212671}{0.715160}{0.072169}{0.019334}{0.119193}{0.950227} \cdot \vecthree{R}{G}{B} 
    
    
    
    
    .. math::
    
        L  \leftarrow \fork{116 Y^{1/3}}{for $Y>0.008856$}{903.3 Y}{for $Y<=0.008856$} 
    
    
    
    
    .. math::
    
        u'  \leftarrow 4*X/(X + 15*Y + 3 Z)  
    
    
    
    
    .. math::
    
        v'  \leftarrow 9*Y/(X + 15*Y + 3 Z)  
    
    
    
    
    .. math::
    
        u  \leftarrow 13*L*(u' - u_n)  \quad \text{where} \quad u_n=0.19793943  
    
    
    
    
    .. math::
    
        v  \leftarrow 13*L*(v' - v_n)  \quad \text{where} \quad v_n=0.46831096  
    
    
    On output 
    :math:`0 \leq L \leq 100`
    , 
    :math:`-134 \leq u \leq 220`
    , 
    :math:`-140 \leq v \leq 122`
    .
    
    The values are then converted to the destination data type:
    
    
        
    
    * 8-bit images
        
        
        .. math::
        
            L  \leftarrow 255/100 L, u  \leftarrow 255/354 (u + 134), v  \leftarrow 255/256 (v + 140)  
        
        
        
    
    * 16-bit images
        currently not supported
        
    
    * 32-bit images
        L, u, v are left as is
        
        
    The above formulas for converting RGB to/from various color spaces have been taken from multiple sources on Web, primarily from
    the Ford98
    at the Charles Poynton site.
    
    

*
    Bayer 
    :math:`\rightarrow`
    RGB (
    ``CV_BayerBG2BGR, CV_BayerGB2BGR, CV_BayerRG2BGR, CV_BayerGR2BGR, CV_BayerBG2RGB, CV_BayerGB2RGB, CV_BayerRG2RGB, CV_BayerGR2RGB``
    ) The Bayer pattern is widely used in CCD and CMOS cameras. It allows one to get color pictures from a single plane where R,G and B pixels (sensors of a particular component) are interleaved like this:
    
    
    
    
    
    .. math::
    
        \newcommand{\Rcell}{\color{red}R} \newcommand{\Gcell}{\color{green}G} \newcommand{\Bcell}{\color{blue}B} \definecolor{BackGray}{rgb}{0.8,0.8,0.8} \begin{array}{ c c c c c } \Rcell & \Gcell & \Rcell & \Gcell & \Rcell \\ \Gcell & \colorbox{BackGray}{\Bcell} & \colorbox{BackGray}{\Gcell} & \Bcell & \Gcell \\ \Rcell & \Gcell & \Rcell & \Gcell & \Rcell \\ \Gcell & \Bcell & \Gcell & \Bcell & \Gcell \\ \Rcell & \Gcell & \Rcell & \Gcell & \Rcell \end{array} 
    
    
    The output RGB components of a pixel are interpolated from 1, 2 or
    4 neighbors of the pixel having the same color. There are several
    modifications of the above pattern that can be achieved by shifting
    the pattern one pixel left and/or one pixel up. The two letters
    :math:`C_1`
    and 
    :math:`C_2`
    in the conversion constants
    ``CV_Bayer``
    :math:`C_1 C_2`
    ``2BGR``
    and
    ``CV_Bayer``
    :math:`C_1 C_2`
    ``2RGB``
    indicate the particular pattern
    type - these are components from the second row, second and third
    columns, respectively. For example, the above pattern has very
    popular "BG" type.
    
    

.. index:: DistTransform

.. _DistTransform:

DistTransform
-------------




.. function:: DistTransform(src,dst,distance_type=CV_DIST_L2,mask_size=3,mask=None,labels=NULL)-> None

    Calculates the distance to the closest zero pixel for all non-zero pixels of the source image.





    
    :param src: 8-bit, single-channel (binary) source image 
    
    :type src: :class:`CvArr`
    
    
    :param dst: Output image with calculated distances (32-bit floating-point, single-channel) 
    
    :type dst: :class:`CvArr`
    
    
    :param distance_type: Type of distance; can be  ``CV_DIST_L1, CV_DIST_L2, CV_DIST_C``  or  ``CV_DIST_USER`` 
    
    :type distance_type: int
    
    
    :param mask_size: Size of the distance transform mask; can be 3 or 5. in the case of  ``CV_DIST_L1``  or  ``CV_DIST_C``  the parameter is forced to 3, because a  :math:`3\times 3`  mask gives the same result as a  :math:`5\times 5`  yet it is faster 
    
    :type mask_size: int
    
    
    :param mask: User-defined mask in the case of a user-defined distance, it consists of 2 numbers (horizontal/vertical shift cost, diagonal shift cost) in the case ofa   :math:`3\times 3`  mask and 3 numbers (horizontal/vertical shift cost, diagonal shift cost, knight's move cost) in the case of a  :math:`5\times 5`  mask 
    
    :type mask: sequence of float
    
    
    :param labels: The optional output 2d array of integer type labels, the same size as  ``src``  and  ``dst`` 
    
    :type labels: :class:`CvArr`
    
    
    
The function calculates the approximated
distance from every binary image pixel to the nearest zero pixel.
For zero pixels the function sets the zero distance, for others it
finds the shortest path consisting of basic shifts: horizontal,
vertical, diagonal or knight's move (the latest is available for a
:math:`5\times 5`
mask). The overall distance is calculated as a sum of these
basic distances. Because the distance function should be symmetric,
all of the horizontal and vertical shifts must have the same cost (that
is denoted as 
``a``
), all the diagonal shifts must have the
same cost (denoted 
``b``
), and all knight's moves must have
the same cost (denoted 
``c``
). For 
``CV_DIST_C``
and
``CV_DIST_L1``
types the distance is calculated precisely,
whereas for 
``CV_DIST_L2``
(Euclidian distance) the distance
can be calculated only with some relative error (a 
:math:`5\times 5`
mask
gives more accurate results), OpenCV uses the values suggested in
Borgefors86
:



.. table::

    ==============  ===================  ======================
    ``CV_DIST_C``   :math:`(3\times 3)`  a = 1, b = 1 \        
    ==============  ===================  ======================
    ``CV_DIST_L1``  :math:`(3\times 3)`  a = 1, b = 2 \        
    ``CV_DIST_L2``  :math:`(3\times 3)`  a=0.955, b=1.3693 \   
    ``CV_DIST_L2``  :math:`(5\times 5)`  a=1, b=1.4, c=2.1969 \
    ==============  ===================  ======================

And below are samples of the distance field (black (0) pixel is in the middle of white square) in the case of a user-defined distance:

User-defined 
:math:`3 \times 3`
mask (a=1, b=1.5)


.. table::

    ===  ===  ===  =  ===  ===  =====
    4.5  4    3.5  3  3.5  4    4.5 \
    ===  ===  ===  =  ===  ===  =====
    4    3    2.5  2  2.5  3    4 \  
    3.5  2.5  1.5  1  1.5  2.5  3.5 \
    3    2    1       1    2    3 \  
    3.5  2.5  1.5  1  1.5  2.5  3.5 \
    4    3    2.5  2  2.5  3    4 \  
    4.5  4    3.5  3  3.5  4    4.5 \
    ===  ===  ===  =  ===  ===  =====

User-defined 
:math:`5 \times 5`
mask (a=1, b=1.5, c=2)


.. table::

    ===  ===  ===  =  ===  ===  =====
    4.5  3.5  3    3  3    3.5  4.5 \
    ===  ===  ===  =  ===  ===  =====
    3.5  3    2    2  2    3    3.5 \
    3    2    1.5  1  1.5  2    3 \  
    3    2    1       1    2    3 \  
    3    2    1.5  1  1.5  2    3 \  
    3.5  3    2    2  2    3    3.5 \
    4    3.5  3    3  3    3.5  4 \  
    ===  ===  ===  =  ===  ===  =====

Typically, for a fast, coarse distance estimation 
``CV_DIST_L2``
,
a 
:math:`3\times 3`
mask is used, and for a more accurate distance estimation
``CV_DIST_L2``
, a 
:math:`5\times 5`
mask is used.

When the output parameter 
``labels``
is not 
``NULL``
, for
every non-zero pixel the function also finds the nearest connected
component consisting of zero pixels. The connected components
themselves are found as contours in the beginning of the function.

In this mode the processing time is still O(N), where N is the number of
pixels. Thus, the function provides a very fast way to compute approximate
Voronoi diagram for the binary image.


.. index:: CvConnectedComp

.. _CvConnectedComp:

CvConnectedComp
---------------



.. class:: CvConnectedComp



Connected component, represented as a tuple (area, value, rect), where
area is the area of the component as a float, value is the average color
as a 
:ref:`CvScalar`
, and rect is the ROI of the component, as a 
:ref:`CvRect`
.

.. index:: FloodFill

.. _FloodFill:

FloodFill
---------




.. function:: FloodFill(image,seed_point,new_val,lo_diff=(0,0,0,0),up_diff=(0,0,0,0),flags=4,mask=NULL)-> comp

    Fills a connected component with the given color.





    
    :param image: Input 1- or 3-channel, 8-bit or floating-point image. It is modified by the function unless the  ``CV_FLOODFILL_MASK_ONLY``  flag is set (see below) 
    
    :type image: :class:`CvArr`
    
    
    :param seed_point: The starting point 
    
    :type seed_point: :class:`CvPoint`
    
    
    :param new_val: New value of the repainted domain pixels 
    
    :type new_val: :class:`CvScalar`
    
    
    :param lo_diff: Maximal lower brightness/color difference between the currently observed pixel and one of its neighbors belonging to the component, or a seed pixel being added to the component. In the case of 8-bit color images it is a packed value 
    
    :type lo_diff: :class:`CvScalar`
    
    
    :param up_diff: Maximal upper brightness/color difference between the currently observed pixel and one of its neighbors belonging to the component, or a seed pixel being added to the component. In the case of 8-bit color images it is a packed value 
    
    :type up_diff: :class:`CvScalar`
    
    
    :param comp: Returned connected component for the repainted domain. Note that the function does not fill  ``comp->contour``  field. The boundary of the filled component can be retrieved from the output mask image using  :ref:`FindContours` 
    
    :type comp: :class:`CvConnectedComp`
    
    
    :param flags: The operation flags. Lower bits contain connectivity value, 4 (by default) or 8, used within the function. Connectivity determines which neighbors of a pixel are considered. Upper bits can be 0 or a combination of the following flags: 
        
                
            * **CV_FLOODFILL_FIXED_RANGE** if set, the difference between the current pixel and seed pixel is considered, otherwise the difference between neighbor pixels is considered (the range is floating) 
            
               
            * **CV_FLOODFILL_MASK_ONLY** if set, the function does not fill the image ( ``new_val``  is ignored), but fills the mask (that must be non-NULL in this case) 
            
            
    
    :type flags: int
    
    
    :param mask: Operation mask, should be a single-channel 8-bit image, 2 pixels wider and 2 pixels taller than  ``image`` . If not NULL, the function uses and updates the mask, so the user takes responsibility of initializing the  ``mask``  content. Floodfilling can't go across non-zero pixels in the mask, for example, an edge detector output can be used as a mask to stop filling at edges. It is possible to use the same mask in multiple calls to the function to make sure the filled area do not overlap.  **Note** : because the mask is larger than the filled image, a pixel in  ``mask``  that corresponds to  :math:`(x,y)`  pixel in  ``image``  will have coordinates  :math:`(x+1,y+1)`   
    
    :type mask: :class:`CvArr`
    
    
    
The function fills a connected component starting from the seed point with the specified color. The connectivity is determined by the closeness of pixel values. The pixel at 
:math:`(x,y)`
is considered to belong to the repainted domain if:



    

* grayscale image, floating range
    
    
    .. math::
    
        src(x',y')- \texttt{lo\_diff} <= src(x,y) <= src(x',y')+ \texttt{up\_diff} 
    
    
    

* grayscale image, fixed range
    
    
    .. math::
    
        src(seed.x,seed.y)- \texttt{lo\_diff} <=src(x,y)<=src(seed.x,seed.y)+ \texttt{up\_diff} 
    
    
    

* color image, floating range
    
    
    .. math::
    
        src(x',y')_r- \texttt{lo\_diff} _r<=src(x,y)_r<=src(x',y')_r+ \texttt{up\_diff} _r  
    
    
    
    
    .. math::
    
        src(x',y')_g- \texttt{lo\_diff} _g<=src(x,y)_g<=src(x',y')_g+ \texttt{up\_diff} _g  
    
    
    
    
    .. math::
    
        src(x',y')_b- \texttt{lo\_diff} _b<=src(x,y)_b<=src(x',y')_b+ \texttt{up\_diff} _b  
    
    
    

* color image, fixed range
    
    
    .. math::
    
        src(seed.x,seed.y)_r- \texttt{lo\_diff} _r<=src(x,y)_r<=src(seed.x,seed.y)_r+ \texttt{up\_diff} _r  
    
    
    
    
    .. math::
    
        src(seed.x,seed.y)_g- \texttt{lo\_diff} _g<=src(x,y)_g<=src(seed.x,seed.y)_g+ \texttt{up\_diff} _g  
    
    
    
    
    .. math::
    
        src(seed.x,seed.y)_b- \texttt{lo\_diff} _b<=src(x,y)_b<=src(seed.x,seed.y)_b+ \texttt{up\_diff} _b  
    
    
    
    
where 
:math:`src(x',y')`
is the value of one of pixel neighbors. That is, to be added to the connected component, a pixel's color/brightness should be close enough to the:


    

*
    color/brightness of one of its neighbors that are already referred to the connected component in the case of floating range
      
    

*
    color/brightness of the seed point in the case of fixed range.
    
    

.. index:: Inpaint

.. _Inpaint:

Inpaint
-------




.. function:: Inpaint(src,mask,dst,inpaintRadius,flags) -> None

    Inpaints the selected region in the image.





    
    :param src: The input 8-bit 1-channel or 3-channel image. 
    
    :type src: :class:`CvArr`
    
    
    :param mask: The inpainting mask, 8-bit 1-channel image. Non-zero pixels indicate the area that needs to be inpainted. 
    
    :type mask: :class:`CvArr`
    
    
    :param dst: The output image of the same format and the same size as input. 
    
    :type dst: :class:`CvArr`
    
    
    :param inpaintRadius: The radius of circlular neighborhood of each point inpainted that is considered by the algorithm. 
    
    :type inpaintRadius: float
    
    
    :param flags: The inpainting method, one of the following: 
         
            * **CV_INPAINT_NS** Navier-Stokes based method. 
            
            * **CV_INPAINT_TELEA** The method by Alexandru Telea  Telea04 
            
            
    
    :type flags: int
    
    
    
The function reconstructs the selected image area from the pixel near the area boundary. The function may be used to remove dust and scratches from a scanned photo, or to remove undesirable objects from still images or video.


.. index:: Integral

.. _Integral:

Integral
--------




.. function:: Integral(image,sum,sqsum=NULL,tiltedSum=NULL)-> None

    Calculates the integral of an image.





    
    :param image: The source image,  :math:`W\times H` , 8-bit or floating-point (32f or 64f) 
    
    :type image: :class:`CvArr`
    
    
    :param sum: The integral image,  :math:`(W+1)\times (H+1)` , 32-bit integer or double precision floating-point (64f) 
    
    :type sum: :class:`CvArr`
    
    
    :param sqsum: The integral image for squared pixel values,  :math:`(W+1)\times (H+1)` , double precision floating-point (64f) 
    
    :type sqsum: :class:`CvArr`
    
    
    :param tiltedSum: The integral for the image rotated by 45 degrees,  :math:`(W+1)\times (H+1)` , the same data type as  ``sum`` 
    
    :type tiltedSum: :class:`CvArr`
    
    
    
The function calculates one or more integral images for the source image as following:



.. math::

    \texttt{sum} (X,Y) =  \sum _{x<X,y<Y}  \texttt{image} (x,y) 




.. math::

    \texttt{sqsum} (X,Y) =  \sum _{x<X,y<Y}  \texttt{image} (x,y)^2 




.. math::

    \texttt{tiltedSum} (X,Y) =  \sum _{y<Y,abs(x-X+1) \leq Y-y-1}  \texttt{image} (x,y) 


Using these integral images, one may calculate sum, mean and standard deviation over a specific up-right or rotated rectangular region of the image in a constant time, for example:



.. math::

    \sum _{x_1<=x<x_2,  \, y_1<=y<y_2} =  \texttt{sum} (x_2,y_2)- \texttt{sum} (x_1,y_2)- \texttt{sum} (x_2,y_1)+ \texttt{sum} (x_1,x_1) 


It makes possible to do a fast blurring or fast block correlation with variable window size, for example. In the case of multi-channel images, sums for each channel are accumulated independently.



.. index:: PyrMeanShiftFiltering

.. _PyrMeanShiftFiltering:

PyrMeanShiftFiltering
---------------------




.. function:: PyrMeanShiftFiltering(src,dst,sp,sr,max_level=1,termcrit=(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,5,1))-> None

    Does meanshift image segmentation





    
    :param src: The source 8-bit, 3-channel image. 
    
    :type src: :class:`CvArr`
    
    
    :param dst: The destination image of the same format and the same size as the source. 
    
    :type dst: :class:`CvArr`
    
    
    :param sp: The spatial window radius. 
    
    :type sp: float
    
    
    :param sr: The color window radius. 
    
    :type sr: float
    
    
    :param max_level: Maximum level of the pyramid for the segmentation. 
    
    :type max_level: int
    
    
    :param termcrit: Termination criteria: when to stop meanshift iterations. 
    
    :type termcrit: :class:`CvTermCriteria`
    
    
    
The function implements the filtering
stage of meanshift segmentation, that is, the output of the function is
the filtered "posterized" image with color gradients and fine-grain
texture flattened. At every pixel 
:math:`(X,Y)`
of the input image (or
down-sized input image, see below) the function executes meanshift
iterations, that is, the pixel 
:math:`(X,Y)`
neighborhood in the joint
space-color hyperspace is considered:



.. math::

    (x,y): X- \texttt{sp} \le x  \le X+ \texttt{sp} , Y- \texttt{sp} \le y  \le Y+ \texttt{sp} , ||(R,G,B)-(r,g,b)||   \le \texttt{sr} 


where 
``(R,G,B)``
and 
``(r,g,b)``
are the vectors of color components at 
``(X,Y)``
and 
``(x,y)``
, respectively (though, the algorithm does not depend on the color space used, so any 3-component color space can be used instead). Over the neighborhood the average spatial value 
``(X',Y')``
and average color vector 
``(R',G',B')``
are found and they act as the neighborhood center on the next iteration: 

:math:`(X,Y)~(X',Y'), (R,G,B)~(R',G',B').`
After the iterations over, the color components of the initial pixel (that is, the pixel from where the iterations started) are set to the final value (average color at the last iteration): 

:math:`I(X,Y) <- (R*,G*,B*)`
Then 
:math:`\texttt{max\_level}>0`
, the gaussian pyramid of
:math:`\texttt{max\_level}+1`
levels is built, and the above procedure is run
on the smallest layer. After that, the results are propagated to the
larger layer and the iterations are run again only on those pixels where
the layer colors differ much ( 
:math:`>\texttt{sr}`
) from the lower-resolution
layer, that is, the boundaries of the color regions are clarified. Note,
that the results will be actually different from the ones obtained by
running the meanshift procedure on the whole original image (i.e. when
:math:`\texttt{max\_level}==0`
).


.. index:: PyrSegmentation

.. _PyrSegmentation:

PyrSegmentation
---------------




.. function:: PyrSegmentation(src,dst,storage,level,threshold1,threshold2)-> comp

    Implements image segmentation by pyramids.





    
    :param src: The source image 
    
    :type src: :class:`IplImage`
    
    
    :param dst: The destination image 
    
    :type dst: :class:`IplImage`
    
    
    :param storage: Storage; stores the resulting sequence of connected components 
    
    :type storage: :class:`CvMemStorage`
    
    
    :param comp: Pointer to the output sequence of the segmented components 
    
    :type comp: :class:`CvSeq`
    
    
    :param level: Maximum level of the pyramid for the segmentation 
    
    :type level: int
    
    
    :param threshold1: Error threshold for establishing the links 
    
    :type threshold1: float
    
    
    :param threshold2: Error threshold for the segments clustering 
    
    :type threshold2: float
    
    
    
The function implements image segmentation by pyramids. The pyramid builds up to the level 
``level``
. The links between any pixel 
``a``
on level 
``i``
and its candidate father pixel 
``b``
on the adjacent level are established if
:math:`p(c(a),c(b))<threshold1`
.
After the connected components are defined, they are joined into several clusters.
Any two segments A and B belong to the same cluster, if 
:math:`p(c(A),c(B))<threshold2`
.
If the input image has only one channel, then 
:math:`p(c^1,c^2)=|c^1-c^2|`
.
If the input image has three channels (red, green and blue), then


.. math::

    p(c^1,c^2) = 0.30 (c^1_r - c^2_r) +
                   0.59 (c^1_g - c^2_g) +
                   0.11 (c^1_b - c^2_b). 


There may be more than one connected component per a cluster. The images 
``src``
and 
``dst``
should be 8-bit single-channel or 3-channel images or equal size.


.. index:: Threshold

.. _Threshold:

Threshold
---------




.. function:: Threshold(src,dst,threshold,maxValue,thresholdType)-> None

    Applies a fixed-level threshold to array elements.





    
    :param src: Source array (single-channel, 8-bit or 32-bit floating point) 
    
    :type src: :class:`CvArr`
    
    
    :param dst: Destination array; must be either the same type as  ``src``  or 8-bit 
    
    :type dst: :class:`CvArr`
    
    
    :param threshold: Threshold value 
    
    :type threshold: float
    
    
    :param maxValue: Maximum value to use with  ``CV_THRESH_BINARY``  and  ``CV_THRESH_BINARY_INV``  thresholding types 
    
    :type maxValue: float
    
    
    :param thresholdType: Thresholding type (see the discussion) 
    
    :type thresholdType: int
    
    
    
The function applies fixed-level thresholding
to a single-channel array. The function is typically used to get a
bi-level (binary) image out of a grayscale image (
:ref:`CmpS`
could
be also used for this purpose) or for removing a noise, i.e. filtering
out pixels with too small or too large values. There are several
types of thresholding that the function supports that are determined by
``thresholdType``
:



    
    * **CV_THRESH_BINARY**  
        
        .. math::
        
              \texttt{dst} (x,y) =  \fork{\texttt{maxValue}}{if $\texttt{src}(x,y) > \texttt{threshold}$}{0}{otherwise}   
        
        
    
    
    * **CV_THRESH_BINARY_INV**  
        
        .. math::
        
              \texttt{dst} (x,y) =  \fork{0}{if $\texttt{src}(x,y) > \texttt{threshold}$}{\texttt{maxValue}}{otherwise}   
        
        
    
    
    * **CV_THRESH_TRUNC**  
        
        .. math::
        
              \texttt{dst} (x,y) =  \fork{\texttt{threshold}}{if $\texttt{src}(x,y) > \texttt{threshold}$}{\texttt{src}(x,y)}{otherwise}   
        
        
    
    
    * **CV_THRESH_TOZERO**  
        
        .. math::
        
              \texttt{dst} (x,y) =  \fork{\texttt{src}(x,y)}{if $\texttt{src}(x,y) > \texttt{threshold}$}{0}{otherwise}   
        
        
    
    
    * **CV_THRESH_TOZERO_INV**  
        
        .. math::
        
              \texttt{dst} (x,y) =  \fork{0}{if $\texttt{src}(x,y) > \texttt{threshold}$}{\texttt{src}(x,y)}{otherwise}   
        
        
    
    
    
Also, the special value 
``CV_THRESH_OTSU``
may be combined with
one of the above values. In this case the function determines the optimal threshold
value using Otsu's algorithm and uses it instead of the specified 
``thresh``
.
The function returns the computed threshold value.
Currently, Otsu's method is implemented only for 8-bit images.







