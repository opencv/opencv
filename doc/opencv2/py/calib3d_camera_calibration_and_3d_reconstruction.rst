Camera Calibration and 3d Reconstruction
========================================

.. highlight:: python


The functions in this section use the so-called pinhole camera model. That
is, a scene view is formed by projecting 3D points into the image plane
using a perspective transformation.



.. math::

    s  \; m' = A [R|t] M' 


or



.. math::

    s  \vecthree{u}{v}{1} =  \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1} \begin{bmatrix} r_{11} & r_{12} & r_{13} & t_1  \\ r_{21} & r_{22} & r_{23} & t_2  \\ r_{31} & r_{32} & r_{33} & t_3 \end{bmatrix} \begin{bmatrix} X \\ Y \\ Z \\ 1  \end{bmatrix} 


Where 
:math:`(X, Y, Z)`
are the coordinates of a 3D point in the world
coordinate space, 
:math:`(u, v)`
are the coordinates of the projection point
in pixels. 
:math:`A`
is called a camera matrix, or a matrix of
intrinsic parameters. 
:math:`(cx, cy)`
is a principal point (that is
usually at the image center), and 
:math:`fx, fy`
are the focal lengths
expressed in pixel-related units. Thus, if an image from camera is
scaled by some factor, all of these parameters should
be scaled (multiplied/divided, respectively) by the same factor. The
matrix of intrinsic parameters does not depend on the scene viewed and,
once estimated, can be re-used (as long as the focal length is fixed (in
case of zoom lens)). The joint rotation-translation matrix 
:math:`[R|t]`
is called a matrix of extrinsic parameters. It is used to describe the
camera motion around a static scene, or vice versa, rigid motion of an
object in front of still camera. That is, 
:math:`[R|t]`
translates
coordinates of a point 
:math:`(X, Y, Z)`
to some coordinate system,
fixed with respect to the camera. The transformation above is equivalent
to the following (when 
:math:`z \ne 0`
):



.. math::

    \begin{array}{l} \vecthree{x}{y}{z} = R  \vecthree{X}{Y}{Z} + t \\ x' = x/z \\ y' = y/z \\ u = f_x*x' + c_x \\ v = f_y*y' + c_y \end{array} 


Real lenses usually have some distortion, mostly
radial distortion and slight tangential distortion. So, the above model
is extended as:



.. math::

    \begin{array}{l} \vecthree{x}{y}{z} = R  \vecthree{X}{Y}{Z} + t \\ x' = x/z \\ y' = y/z \\ x'' = x'  \frac{1 + k_1 r^2 + k_2 r^4 + k_3 r^6}{1 + k_4 r^2 + k_5 r^4 + k_6 r^6} + 2 p_1 x' y' + p_2(r^2 + 2 x'^2)  \\ y'' = y'  \frac{1 + k_1 r^2 + k_2 r^4 + k_3 r^6}{1 + k_4 r^2 + k_5 r^4 + k_6 r^6} + p_1 (r^2 + 2 y'^2) + 2 p_2 x' y'  \\ \text{where} \quad r^2 = x'^2 + y'^2  \\ u = f_x*x'' + c_x \\ v = f_y*y'' + c_y \end{array} 


:math:`k_1`
, 
:math:`k_2`
, 
:math:`k_3`
, 
:math:`k_4`
, 
:math:`k_5`
, 
:math:`k_6`
are radial distortion coefficients, 
:math:`p_1`
, 
:math:`p_2`
are tangential distortion coefficients.
Higher-order coefficients are not considered in OpenCV. In the functions below the coefficients are passed or returned as


.. math::

    (k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6]])  


vector. That is, if the vector contains 4 elements, it means that 
:math:`k_3=0`
.
The distortion coefficients do not depend on the scene viewed, thus they also belong to the intrinsic camera parameters.
*And they remain the same regardless of the captured image resolution.*
That is, if, for example, a camera has been calibrated on images of 
:math:`320
\times 240`
resolution, absolutely the same distortion coefficients can
be used for images of 
:math:`640 \times 480`
resolution from the same camera (while 
:math:`f_x`
,
:math:`f_y`
, 
:math:`c_x`
and 
:math:`c_y`
need to be scaled appropriately).

The functions below use the above model to



    

*
    Project 3D points to the image plane given intrinsic and extrinsic parameters
     
    

*
    Compute extrinsic parameters given intrinsic parameters, a few 3D points and their projections.
     
    

*
    Estimate intrinsic and extrinsic camera parameters from several views of a known calibration pattern (i.e. every view is described by several 3D-2D point correspondences).
     
    

*
    Estimate the relative position and orientation of the stereo camera "heads" and compute the 
    *rectification*
    transformation that makes the camera optical axes parallel.
    
    

.. index:: CalibrateCamera2

.. _CalibrateCamera2:

CalibrateCamera2
----------------




.. function:: CalibrateCamera2(objectPoints,imagePoints,pointCounts,imageSize,cameraMatrix,distCoeffs,rvecs,tvecs,flags=0)-> None

    Finds the camera intrinsic and extrinsic parameters from several views of a calibration pattern.





    
    :param objectPoints: The joint matrix of object points - calibration pattern features in the model coordinate space. It is floating-point 3xN or Nx3 1-channel, or 1xN or Nx1 3-channel array, where N is the total number of points in all views. 
    
    :type objectPoints: :class:`CvMat`
    
    
    :param imagePoints: The joint matrix of object points projections in the camera views. It is floating-point 2xN or Nx2 1-channel, or 1xN or Nx1 2-channel array, where N is the total number of points in all views 
    
    :type imagePoints: :class:`CvMat`
    
    
    :param pointCounts: Integer 1xM or Mx1 vector (where M is the number of calibration pattern views) containing the number of points in each particular view. The sum of vector elements must match the size of  ``objectPoints``  and  ``imagePoints``  (=N). 
    
    :type pointCounts: :class:`CvMat`
    
    
    :param imageSize: Size of the image, used only to initialize the intrinsic camera matrix 
    
    :type imageSize: :class:`CvSize`
    
    
    :param cameraMatrix: The output 3x3 floating-point camera matrix  :math:`A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}` . If  ``CV_CALIB_USE_INTRINSIC_GUESS``  and/or  ``CV_CALIB_FIX_ASPECT_RATIO``  are specified, some or all of  ``fx, fy, cx, cy``  must be initialized before calling the function 
    
    :type cameraMatrix: :class:`CvMat`
    
    
    :param distCoeffs: The output vector of distortion coefficients  :math:`(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6]])`  of 4, 5 or 8 elements 
    
    :type distCoeffs: :class:`CvMat`
    
    
    :param rvecs: The output  3x *M*  or  *M* x3 1-channel, or 1x *M*  or  *M* x1 3-channel array   of rotation vectors (see  :ref:`Rodrigues2` ), estimated for each pattern view. That is, each k-th rotation vector together with the corresponding k-th translation vector (see the next output parameter description) brings the calibration pattern from the model coordinate space (in which object points are specified) to the world coordinate space, i.e. real position of the calibration pattern in the k-th pattern view (k=0.. *M* -1) 
    
    :type rvecs: :class:`CvMat`
    
    
    :param tvecs: The output  3x *M*  or  *M* x3 1-channel, or 1x *M*  or  *M* x1 3-channel array   of translation vectors, estimated for each pattern view. 
    
    :type tvecs: :class:`CvMat`
    
    
    :param flags: Different flags, may be 0 or combination of the following values: 
         
            * **CV_CALIB_USE_INTRINSIC_GUESS** ``cameraMatrix``  contains the valid initial values of  ``fx, fy, cx, cy``  that are optimized further. Otherwise,  ``(cx, cy)``  is initially set to the image center ( ``imageSize``  is used here), and focal distances are computed in some least-squares fashion. Note, that if intrinsic parameters are known, there is no need to use this function just to estimate the extrinsic parameters. Use  :ref:`FindExtrinsicCameraParams2`  instead. 
            
            * **CV_CALIB_FIX_PRINCIPAL_POINT** The principal point is not changed during the global optimization, it stays at the center or at the other location specified when    ``CV_CALIB_USE_INTRINSIC_GUESS``  is set too. 
            
            * **CV_CALIB_FIX_ASPECT_RATIO** The functions considers only  ``fy``  as a free parameter, the ratio  ``fx/fy``  stays the same as in the input  ``cameraMatrix`` .   When  ``CV_CALIB_USE_INTRINSIC_GUESS``  is not set, the actual input values of  ``fx``  and  ``fy``  are ignored, only their ratio is computed and used further. 
            
            * **CV_CALIB_ZERO_TANGENT_DIST** Tangential distortion coefficients  :math:`(p_1, p_2)`  will be set to zeros and stay zero. 
            
        
        :type flags: int
        
        
        * **CV_CALIB_FIX_K1,...,CV_CALIB_FIX_K6** Do not change the corresponding radial distortion coefficient during the optimization. If  ``CV_CALIB_USE_INTRINSIC_GUESS``  is set, the coefficient from the supplied  ``distCoeffs``  matrix is used, otherwise it is set to 0. 
        
        
        * **CV_CALIB_RATIONAL_MODEL** Enable coefficients k4, k5 and k6. To provide the backward compatibility, this extra flag should be explicitly specified to make the calibration function use the rational model and return 8 coefficients. If the flag is not set, the function will compute   only 5 distortion coefficients. 
        
        
        
    
    
The function estimates the intrinsic camera
parameters and extrinsic parameters for each of the views. The
coordinates of 3D object points and their correspondent 2D projections
in each view must be specified. That may be achieved by using an
object with known geometry and easily detectable feature points.
Such an object is called a calibration rig or calibration pattern,
and OpenCV has built-in support for a chessboard as a calibration
rig (see 
:ref:`FindChessboardCorners`
). Currently, initialization
of intrinsic parameters (when 
``CV_CALIB_USE_INTRINSIC_GUESS``
is not set) is only implemented for planar calibration patterns
(where z-coordinates of the object points must be all 0's). 3D
calibration rigs can also be used as long as initial 
``cameraMatrix``
is provided.

The algorithm does the following:


    

#.
    First, it computes the initial intrinsic parameters (the option only available for planar calibration patterns) or reads them from the input parameters. The distortion coefficients are all set to zeros initially (unless some of 
    ``CV_CALIB_FIX_K?``
    are specified).
        
    

#.
    The initial camera pose is estimated as if the intrinsic parameters have been already known. This is done using 
    :ref:`FindExtrinsicCameraParams2`
    

#.
    After that the global Levenberg-Marquardt optimization algorithm is run to minimize the reprojection error, i.e. the total sum of squared distances between the observed feature points 
    ``imagePoints``
    and the projected (using the current estimates for camera parameters and the poses) object points 
    ``objectPoints``
    ; see 
    :ref:`ProjectPoints2`
    .
    
    
Note: if you're using a non-square (=non-NxN) grid and
:cpp:func:`findChessboardCorners`
for calibration, and 
``calibrateCamera``
returns
bad values (i.e. zero distortion coefficients, an image center very far from
:math:`(w/2-0.5,h/2-0.5)`
, and / or large differences between 
:math:`f_x`
and 
:math:`f_y`
(ratios of
10:1 or more)), then you've probably used 
``patternSize=cvSize(rows,cols)``
,
but should use 
``patternSize=cvSize(cols,rows)``
in 
:ref:`FindChessboardCorners`
.

See also: 
:ref:`FindChessboardCorners`
, 
:ref:`FindExtrinsicCameraParams2`
, 
:cpp:func:`initCameraMatrix2D`
, 
:ref:`StereoCalibrate`
, 
:ref:`Undistort2`

.. index:: ComputeCorrespondEpilines

.. _ComputeCorrespondEpilines:

ComputeCorrespondEpilines
-------------------------




.. function:: ComputeCorrespondEpilines(points, whichImage, F, lines) -> None

    For points in one image of a stereo pair, computes the corresponding epilines in the other image.





    
    :param points: The input points.  ``2xN, Nx2, 3xN``  or  ``Nx3``  array (where  ``N``  number of points). Multi-channel  ``1xN``  or  ``Nx1``  array is also acceptable 
    
    :type points: :class:`CvMat`
    
    
    :param whichImage: Index of the image (1 or 2) that contains the  ``points`` 
    
    :type whichImage: int
    
    
    :param F: The fundamental matrix that can be estimated using  :ref:`FindFundamentalMat` 
        or  :ref:`StereoRectify` . 
    
    :type F: :class:`CvMat`
    
    
    :param lines: The output epilines, a  ``3xN``  or  ``Nx3``  array.   Each line  :math:`ax + by + c=0`  is encoded by 3 numbers  :math:`(a, b, c)` 
    
    :type lines: :class:`CvMat`
    
    
    
For every point in one of the two images of a stereo-pair the function finds the equation of the
corresponding epipolar line in the other image.

From the fundamental matrix definition (see 
:ref:`FindFundamentalMat`
),
line 
:math:`l^{(2)}_i`
in the second image for the point 
:math:`p^{(1)}_i`
in the first image (i.e. when 
``whichImage=1``
) is computed as:



.. math::

    l^{(2)}_i = F p^{(1)}_i  


and, vice versa, when 
``whichImage=2``
, 
:math:`l^{(1)}_i`
is computed from 
:math:`p^{(2)}_i`
as:



.. math::

    l^{(1)}_i = F^T p^{(2)}_i  


Line coefficients are defined up to a scale. They are normalized, such that 
:math:`a_i^2+b_i^2=1`
.


.. index:: ConvertPointsHomogeneous

.. _ConvertPointsHomogeneous:

ConvertPointsHomogeneous
------------------------




.. function:: ConvertPointsHomogeneous( src, dst ) -> None

    Convert points to/from homogeneous coordinates.





    
    :param src: The input array or vector of 2D, 3D or 4D points 
    
    :type src: :class:`CvMat`
    
    
    :param dst: The output vector of 2D or 2D points 
    
    :type dst: :class:`CvMat`
    
    
    
The 
2D or 3D points from/to homogeneous coordinates, or simply 
the array. If the input array dimensionality is larger than the output, each coordinate is divided by the last coordinate:



.. math::

    \begin{array}{l} (x,y[,z],w) -> (x',y'[,z']) \\ \text{where} \\ x' = x/w  \\ y' = y/w  \\ z' = z/w  \quad \text{(if output is 3D)} \end{array} 


If the output array dimensionality is larger, an extra 1 is appended to each point.  Otherwise, the input array is simply copied (with optional transposition) to the output.


.. index:: CreatePOSITObject

.. _CreatePOSITObject:

CreatePOSITObject
-----------------




.. function:: CreatePOSITObject(points)-> POSITObject

    Initializes a structure containing object information.





    
    :param points: List of 3D points 
    
    :type points: :class:`CvPoint3D32fs`
    
    
    
The function allocates memory for the object structure and computes the object inverse matrix.

The preprocessed object data is stored in the structure 
:ref:`CvPOSITObject`
, internal for OpenCV, which means that the user cannot directly access the structure data. The user may only create this structure and pass its pointer to the function.

An object is defined as a set of points given in a coordinate system. The function 
:ref:`POSIT`
computes a vector that begins at a camera-related coordinate system center and ends at the 
``points[0]``
of the object.

Once the work with a given object is finished, the function 
:ref:`ReleasePOSITObject`
must be called to free memory.


.. index:: CreateStereoBMState

.. _CreateStereoBMState:

CreateStereoBMState
-------------------




.. function:: CreateStereoBMState(preset=CV_STEREO_BM_BASIC,numberOfDisparities=0)-> StereoBMState

    Creates block matching stereo correspondence structure.





    
    :param preset: ID of one of the pre-defined parameter sets. Any of the parameters can be overridden after creating the structure.  Values are 
         
            * **CV_STEREO_BM_BASIC** Parameters suitable for general cameras 
            
            * **CV_STEREO_BM_FISH_EYE** Parameters suitable for wide-angle cameras 
            
            * **CV_STEREO_BM_NARROW** Parameters suitable for narrow-angle cameras 
            
            
    
    :type preset: int
    
    
    :param numberOfDisparities: The number of disparities. If the parameter is 0, it is taken from the preset, otherwise the supplied value overrides the one from preset. 
    
    :type numberOfDisparities: int
    
    
    
The function creates the stereo correspondence structure and initializes
it. It is possible to override any of the parameters at any time between
the calls to 
:ref:`FindStereoCorrespondenceBM`
.


.. index:: CreateStereoGCState

.. _CreateStereoGCState:

CreateStereoGCState
-------------------




.. function:: CreateStereoGCState(numberOfDisparities,maxIters)-> StereoGCState

    Creates the state of graph cut-based stereo correspondence algorithm.





    
    :param numberOfDisparities: The number of disparities. The disparity search range will be  :math:`\texttt{state->minDisparity} \le disparity < \texttt{state->minDisparity} + \texttt{state->numberOfDisparities}` 
    
    :type numberOfDisparities: int
    
    
    :param maxIters: Maximum number of iterations. On each iteration all possible (or reasonable) alpha-expansions are tried. The algorithm may terminate earlier if it could not find an alpha-expansion that decreases the overall cost function value. See  Kolmogorov03   for details.  
    
    :type maxIters: int
    
    
    
The function creates the stereo correspondence structure and initializes it. It is possible to override any of the parameters at any time between the calls to 
:ref:`FindStereoCorrespondenceGC`
.


.. index:: CvStereoBMState

.. _CvStereoBMState:

CvStereoBMState
---------------



.. class:: CvStereoBMState



The structure for block matching stereo correspondence algorithm.



    
    
    .. attribute:: preFilterType
    
    
    
        type of the prefilter,  ``CV_STEREO_BM_NORMALIZED_RESPONSE``  or the default and the recommended  ``CV_STEREO_BM_XSOBEL`` , int 
    
    
    
    .. attribute:: preFilterSize
    
    
    
        ~5x5..21x21, int 
    
    
    
    .. attribute:: preFilterCap
    
    
    
        up to ~31, int 
    
    
    
    .. attribute:: SADWindowSize
    
    
    
        Could be 5x5..21x21 or higher, but with 21x21 or smaller windows the processing speed is much higher, int 
    
    
    
    .. attribute:: minDisparity
    
    
    
        minimum disparity (=0), int 
    
    
    
    .. attribute:: numberOfDisparities
    
    
    
        maximum disparity - minimum disparity, int 
    
    
    
    .. attribute:: textureThreshold
    
    
    
        the textureness threshold. That is, if the sum of absolute values of x-derivatives computed over  ``SADWindowSize``  by  ``SADWindowSize``  pixel neighborhood is smaller than the parameter, no disparity is computed at the pixel, int 
    
    
    
    .. attribute:: uniquenessRatio
    
    
    
        the minimum margin in percents between the best (minimum) cost function value and the second best value to accept the computed disparity, int 
    
    
    
    .. attribute:: speckleWindowSize
    
    
    
        the maximum area of speckles to remove (set to 0 to disable speckle filtering), int 
    
    
    
    .. attribute:: speckleRange
    
    
    
        acceptable range of disparity variation in each connected component, int 
    
    
    
    .. attribute:: trySmallerWindows
    
    
    
        not used currently (0), int 
    
    
    
    .. attribute:: roi1, roi2
    
    
    
        These are the clipping ROIs for the left and the right images. The function  :ref:`StereoRectify`  returns the largest rectangles in the left and right images where after the rectification all the pixels are valid. If you copy those rectangles to the  ``CvStereoBMState``  structure, the stereo correspondence function will automatically clear out the pixels outside of the "valid" disparity rectangle computed by  :ref:`GetValidDisparityROI` . Thus you will get more "invalid disparity" pixels than usual, but the remaining pixels are more probable to be valid. 
    
    
    
    .. attribute:: disp12MaxDiff
    
    
    
        The maximum allowed difference between the explicitly computed left-to-right disparity map and the implicitly (by  :ref:`ValidateDisparity` ) computed right-to-left disparity. If for some pixel the difference is larger than the specified threshold, the disparity at the pixel is invalidated. By default this parameter is set to (-1), which means that the left-right check is not performed. 
    
    
    
The block matching stereo correspondence algorithm, by Kurt Konolige, is very fast single-pass stereo matching algorithm that uses sliding sums of absolute differences between pixels in the left image and the pixels in the right image, shifted by some varying amount of pixels (from 
``minDisparity``
to 
``minDisparity+numberOfDisparities``
). On a pair of images WxH the algorithm computes disparity in 
``O(W*H*numberOfDisparities)``
time. In order to improve quality and readability of the disparity map, the algorithm includes pre-filtering and post-filtering procedures.

Note that the algorithm searches for the corresponding blocks in x direction only. It means that the supplied stereo pair should be rectified. Vertical stereo layout is not directly supported, but in such a case the images could be transposed by user.


.. index:: CvStereoGCState

.. _CvStereoGCState:

CvStereoGCState
---------------



.. class:: CvStereoGCState



The structure for graph cuts-based stereo correspondence algorithm



    
    
    .. attribute:: Ithreshold
    
    
    
        threshold for piece-wise linear data cost function (5 by default) 
    
    
    
    .. attribute:: interactionRadius
    
    
    
        radius for smoothness cost function (1 by default; means Potts model) 
    
    
    
    .. attribute:: K, lambda, lambda1, lambda2
    
    
    
        parameters for the cost function (usually computed adaptively from the input data) 
    
    
    
    .. attribute:: occlusionCost
    
    
    
        10000 by default 
    
    
    
    .. attribute:: minDisparity
    
    
    
        0 by default; see  :ref:`CvStereoBMState` 
    
    
    
    .. attribute:: numberOfDisparities
    
    
    
        defined by user; see  :ref:`CvStereoBMState` 
    
    
    
    .. attribute:: maxIters
    
    
    
        number of iterations; defined by user. 
    
    
    
The graph cuts stereo correspondence algorithm, described in 
Kolmogorov03
(as 
**KZ1**
), is non-realtime stereo correspondence algorithm that usually gives very accurate depth map with well-defined object boundaries. The algorithm represents stereo problem as a sequence of binary optimization problems, each of those is solved using maximum graph flow algorithm. The state structure above should not be allocated and initialized manually; instead, use 
:ref:`CreateStereoGCState`
and then override necessary parameters if needed.


.. index:: DecomposeProjectionMatrix

.. _DecomposeProjectionMatrix:

DecomposeProjectionMatrix
-------------------------




.. function:: DecomposeProjectionMatrix(projMatrix, cameraMatrix, rotMatrix, transVect, rotMatrX = None, rotMatrY = None, rotMatrZ = None) -> eulerAngles

    Decomposes the projection matrix into a rotation matrix and a camera matrix.





    
    :param projMatrix: The 3x4 input projection matrix P 
    
    :type projMatrix: :class:`CvMat`
    
    
    :param cameraMatrix: The output 3x3 camera matrix K 
    
    :type cameraMatrix: :class:`CvMat`
    
    
    :param rotMatrix: The output 3x3 external rotation matrix R 
    
    :type rotMatrix: :class:`CvMat`
    
    
    :param transVect: The output 4x1 translation vector T 
    
    :type transVect: :class:`CvMat`
    
    
    :param rotMatrX: Optional 3x3 rotation matrix around x-axis 
    
    :type rotMatrX: :class:`CvMat`
    
    
    :param rotMatrY: Optional 3x3 rotation matrix around y-axis 
    
    :type rotMatrY: :class:`CvMat`
    
    
    :param rotMatrZ: Optional 3x3 rotation matrix around z-axis 
    
    :type rotMatrZ: :class:`CvMat`
    
    
    :param eulerAngles: Optional 3 points containing the three Euler angles of rotation 
    
    :type eulerAngles: :class:`CvPoint3D64f`
    
    
    
The function computes a decomposition of a projection matrix into a calibration and a rotation matrix and the position of the camera.

It optionally returns three rotation matrices, one for each axis, and the three Euler angles that could be used in OpenGL.

The function is based on 
:ref:`RQDecomp3x3`
.


.. index:: DrawChessboardCorners

.. _DrawChessboardCorners:

DrawChessboardCorners
---------------------




.. function:: DrawChessboardCorners(image,patternSize,corners,patternWasFound)-> None

    Renders the detected chessboard corners.





    
    :param image: The destination image; it must be an 8-bit color image 
    
    :type image: :class:`CvArr`
    
    
    :param patternSize: The number of inner corners per chessboard row and column. (patternSize = cv::Size(points _ per _ row,points _ per _ column) = cv::Size(rows,columns) ) 
    
    :type patternSize: :class:`CvSize`
    
    
    :param corners: The array of corners detected, this should be the output from findChessboardCorners wrapped in a cv::Mat(). 
    
    :type corners: sequence of (float, float)
    
    
    :param patternWasFound: Indicates whether the complete board was found  :math:`(\ne 0)`   or not  :math:`(=0)`  . One may just pass the return value  :ref:`FindChessboardCorners`  here 
    
    :type patternWasFound: int
    
    
    
The function draws the individual chessboard corners detected as red circles if the board was not found or as colored corners connected with lines if the board was found.


.. index:: FindChessboardCorners

.. _FindChessboardCorners:

FindChessboardCorners
---------------------




.. function:: FindChessboardCorners(image, patternSize, flags=CV_CALIB_CB_ADAPTIVE_THRESH) -> corners

    Finds the positions of the internal corners of the chessboard.





    
    :param image: Source chessboard view; it must be an 8-bit grayscale or color image 
    
    :type image: :class:`CvArr`
    
    
    :param patternSize: The number of inner corners per chessboard row and column
        ( patternSize = cvSize(points _ per _ row,points _ per _ colum) = cvSize(columns,rows) ) 
    
    :type patternSize: :class:`CvSize`
    
    
    :param corners: The output array of corners detected 
    
    :type corners: sequence of (float, float)
    
    
    :param flags: Various operation flags, can be 0 or a combination of the following values: 
        
               
            * **CV_CALIB_CB_ADAPTIVE_THRESH** use adaptive thresholding to convert the image to black and white, rather than a fixed threshold level (computed from the average image brightness). 
            
              
            * **CV_CALIB_CB_NORMALIZE_IMAGE** normalize the image gamma with  :ref:`EqualizeHist`  before applying fixed or adaptive thresholding. 
            
              
            * **CV_CALIB_CB_FILTER_QUADS** use additional criteria (like contour area, perimeter, square-like shape) to filter out false quads that are extracted at the contour retrieval stage. 
            
              
            * **CALIB_CB_FAST_CHECK** Runs a fast check on the image that looks for chessboard corners, and shortcuts the call if none are found. This can drastically speed up the call in the degenerate condition when
                 no chessboard is observed. 
            
            
    
    :type flags: int
    
    
    
The function attempts to determine
whether the input image is a view of the chessboard pattern and
locate the internal chessboard corners. The function returns a non-zero
value if all of the corners have been found and they have been placed
in a certain order (row by row, left to right in every row),
otherwise, if the function fails to find all the corners or reorder
them, it returns 0. For example, a regular chessboard has 8 x 8
squares and 7 x 7 internal corners, that is, points, where the black
squares touch each other. The coordinates detected are approximate,
and to determine their position more accurately, the user may use
the function 
:ref:`FindCornerSubPix`
.

Sample usage of detecting and drawing chessboard corners:



::


    
    Size patternsize(8,6); //interior number of corners
    Mat gray = ....; //source image
    vector<Point2f> corners; //this will be filled by the detected corners
    
    //CALIB_CB_FAST_CHECK saves a lot of time on images 
    //that don't contain any chessboard corners
    bool patternfound = findChessboardCorners(gray, patternsize, corners, 
            CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE 
            + CALIB_CB_FAST_CHECK);
    
    if(patternfound)
      cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1), 
        TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
        
    drawChessboardCorners(img, patternsize, Mat(corners), patternfound);
    

..

**Note:**
the function requires some white space (like a square-thick border, the wider the better) around the board to make the detection more robust in various environment (otherwise if there is no border and the background is dark, the outer black squares could not be segmented properly and so the square grouping and ordering algorithm will fail).


.. index:: FindExtrinsicCameraParams2

.. _FindExtrinsicCameraParams2:

FindExtrinsicCameraParams2
--------------------------




.. function:: FindExtrinsicCameraParams2(objectPoints,imagePoints,cameraMatrix,distCoeffs,rvec,tvec,useExtrinsicGuess=0)-> None

    Finds the object pose from the 3D-2D point correspondences





    
    :param objectPoints: The array of object points in the object coordinate space, 3xN or Nx3 1-channel, or 1xN or Nx1 3-channel, where N is the number of points.  
    
    :type objectPoints: :class:`CvMat`
    
    
    :param imagePoints: The array of corresponding image points, 2xN or Nx2 1-channel or 1xN or Nx1 2-channel, where N is the number of points.  
    
    :type imagePoints: :class:`CvMat`
    
    
    :param cameraMatrix: The input camera matrix  :math:`A = \vecthreethree{fx}{0}{cx}{0}{fy}{cy}{0}{0}{1}` 
    
    :type cameraMatrix: :class:`CvMat`
    
    
    :param distCoeffs: The input vector of distortion coefficients  :math:`(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6]])`  of 4, 5 or 8 elements. If the vector is NULL/empty, the zero distortion coefficients are assumed. 
    
    :type distCoeffs: :class:`CvMat`
    
    
    :param rvec: The output rotation vector (see  :ref:`Rodrigues2` ) that (together with  ``tvec`` ) brings points from the model coordinate system to the camera coordinate system 
    
    :type rvec: :class:`CvMat`
    
    
    :param tvec: The output translation vector 
    
    :type tvec: :class:`CvMat`
    
    
    :param useExtrinsicGuess: If true (1), the function will use the provided  ``rvec``  and  ``tvec``  as the initial approximations of the rotation and translation vectors, respectively, and will further optimize them. 
    
    :type useExtrinsicGuess: int
    
    
    
The function estimates the object pose given a set of object points, their corresponding image projections, as well as the camera matrix and the distortion coefficients. This function finds such a pose that minimizes reprojection error, i.e. the sum of squared distances between the observed projections 
``imagePoints``
and the projected (using 
:ref:`ProjectPoints2`
) 
``objectPoints``
.


The function's counterpart in the C++ API is 

.. index:: FindFundamentalMat

.. _FindFundamentalMat:

FindFundamentalMat
------------------




.. function:: FindFundamentalMat(points1, points2, fundamentalMatrix, method=CV_FM_RANSAC, param1=1., param2=0.99, status = None) -> None

    Calculates the fundamental matrix from the corresponding points in two images.





    
    :param points1: Array of  ``N``  points from the first image. It can be  ``2xN, Nx2, 3xN``  or  ``Nx3``  1-channel array or   ``1xN``  or  ``Nx1``  2- or 3-channel array  . The point coordinates should be floating-point (single or double precision) 
    
    :type points1: :class:`CvMat`
    
    
    :param points2: Array of the second image points of the same size and format as  ``points1`` 
    
    :type points2: :class:`CvMat`
    
    
    :param fundamentalMatrix: The output fundamental matrix or matrices. The size should be 3x3 or 9x3 (7-point method may return up to 3 matrices) 
    
    :type fundamentalMatrix: :class:`CvMat`
    
    
    :param method: Method for computing the fundamental matrix 
        
                
            * **CV_FM_7POINT** for a 7-point algorithm.  :math:`N = 7` 
            
               
            * **CV_FM_8POINT** for an 8-point algorithm.  :math:`N \ge 8` 
            
               
            * **CV_FM_RANSAC** for the RANSAC algorithm.  :math:`N \ge 8` 
            
               
            * **CV_FM_LMEDS** for the LMedS algorithm.  :math:`N \ge 8` 
            
            
    
    :type method: int
    
    
    :param param1: The parameter is used for RANSAC. It is the maximum distance from point to epipolar line in pixels, beyond which the point is considered an outlier and is not used for computing the final fundamental matrix. It can be set to something like 1-3, depending on the accuracy of the point localization, image resolution and the image noise 
    
    :type param1: float
    
    
    :param param2: The parameter is used for RANSAC or LMedS methods only. It specifies the desirable level of confidence (probability) that the estimated matrix is correct 
    
    :type param2: float
    
    
    :param status: The  optional   output array of N elements, every element of which is set to 0 for outliers and to 1 for the other points. The array is computed only in RANSAC and LMedS methods. For other methods it is set to all 1's 
    
    :type status: :class:`CvMat`
    
    
    
The epipolar geometry is described by the following equation:



.. math::

    [p_2; 1]^T F [p_1; 1] = 0  


where 
:math:`F`
is fundamental matrix, 
:math:`p_1`
and 
:math:`p_2`
are corresponding points in the first and the second images, respectively.

The function calculates the fundamental matrix using one of four methods listed above and returns 
the number of fundamental matrices found (1 or 3) and 0, if no matrix is found 
. Normally just 1 matrix is found, but in the case of 7-point algorithm the function may return up to 3 solutions (
:math:`9 \times 3`
matrix that stores all 3 matrices sequentially).

The calculated fundamental matrix may be passed further to
:ref:`ComputeCorrespondEpilines`
that finds the epipolar lines
corresponding to the specified points. It can also be passed to 
:ref:`StereoRectifyUncalibrated`
to compute the rectification transformation.


.. index:: FindHomography

.. _FindHomography:

FindHomography
--------------




.. function:: FindHomography(srcPoints,dstPoints,H,method,ransacReprojThreshold=3.0, status=None)-> None

    Finds the perspective transformation between two planes.





    
    :param srcPoints: Coordinates of the points in the original plane, 2xN, Nx2, 3xN or Nx3 1-channel array (the latter two are for representation in homogeneous coordinates), where N is the number of points. 1xN or Nx1 2- or 3-channel array can also be passed. 
    
    :type srcPoints: :class:`CvMat`
    
    :param dstPoints: Point coordinates in the destination plane, 2xN, Nx2, 3xN or Nx3 1-channel, or 1xN or Nx1 2- or 3-channel array. 
    
    :type dstPoints: :class:`CvMat`
    
    
    :param H: The output 3x3 homography matrix 
    
    :type H: :class:`CvMat`
    
    
    :param method:  The method used to computed homography matrix; one of the following: 
         
            * **0** a regular method using all the points 
            
            * **CV_RANSAC** RANSAC-based robust method 
            
            * **CV_LMEDS** Least-Median robust method 
            
            
    
    :type method: int
    
    
    :param ransacReprojThreshold: The maximum allowed reprojection error to treat a point pair as an inlier (used in the RANSAC method only). That is, if  
        
        .. math::
        
            \| \texttt{dstPoints} _i -  \texttt{convertPointsHomogeneous} ( \texttt{H}   \texttt{srcPoints} _i) \|  >  \texttt{ransacReprojThreshold} 
        
        then the point  :math:`i`  is considered an outlier. If  ``srcPoints``  and  ``dstPoints``  are measured in pixels, it usually makes sense to set this parameter somewhere in the range 1 to 10. 
    
    :type ransacReprojThreshold: float
    
    
    :param status: The optional output mask set by a robust method ( ``CV_RANSAC``  or  ``CV_LMEDS`` ).  *Note that the input mask values are ignored.* 
    
    :type status: :class:`CvMat`
    
    
    
The 
function finds 
the perspective transformation 
:math:`H`
between the source and the destination planes:



.. math::

    s_i  \vecthree{x'_i}{y'_i}{1} \sim H  \vecthree{x_i}{y_i}{1} 


So that the back-projection error



.. math::

    \sum _i \left ( x'_i- \frac{h_{11} x_i + h_{12} y_i + h_{13}}{h_{31} x_i + h_{32} y_i + h_{33}} \right )^2+ \left ( y'_i- \frac{h_{21} x_i + h_{22} y_i + h_{23}}{h_{31} x_i + h_{32} y_i + h_{33}} \right )^2 


is minimized. If the parameter 
``method``
is set to the default value 0, the function
uses all the point pairs to compute the initial homography estimate with a simple least-squares scheme.

However, if not all of the point pairs (
:math:`srcPoints_i`
,
:math:`dstPoints_i`
) fit the rigid perspective transformation (i.e. there
are some outliers), this initial estimate will be poor.
In this case one can use one of the 2 robust methods. Both methods,
``RANSAC``
and 
``LMeDS``
, try many different random subsets
of the corresponding point pairs (of 4 pairs each), estimate
the homography matrix using this subset and a simple least-square
algorithm and then compute the quality/goodness of the computed homography
(which is the number of inliers for RANSAC or the median re-projection
error for LMeDs). The best subset is then used to produce the initial
estimate of the homography matrix and the mask of inliers/outliers.

Regardless of the method, robust or not, the computed homography
matrix is refined further (using inliers only in the case of a robust
method) with the Levenberg-Marquardt method in order to reduce the
re-projection error even more.

The method 
``RANSAC``
can handle practically any ratio of outliers,
but it needs the threshold to distinguish inliers from outliers.
The method 
``LMeDS``
does not need any threshold, but it works
correctly only when there are more than 50
%
of inliers. Finally,
if you are sure in the computed features, where can be only some
small noise present, but no outliers, the default method could be the best
choice.

The function is used to find initial intrinsic and extrinsic matrices.
Homography matrix is determined up to a scale, thus it is normalized so that
:math:`h_{33}=1`
.

See also: 
:ref:`GetAffineTransform`
, 
:ref:`GetPerspectiveTransform`
, 
:ref:`EstimateRigidMotion`
, 
:ref:`WarpPerspective`
, 
:ref:`PerspectiveTransform`

.. index:: FindStereoCorrespondenceBM

.. _FindStereoCorrespondenceBM:

FindStereoCorrespondenceBM
--------------------------




.. function:: FindStereoCorrespondenceBM(left,right,disparity,state)-> None

    Computes the disparity map using block matching algorithm.





    
    :param left: The left single-channel, 8-bit image. 
    
    :type left: :class:`CvArr`
    
    
    :param right: The right image of the same size and the same type. 
    
    :type right: :class:`CvArr`
    
    
    :param disparity: The output single-channel 16-bit signed, or 32-bit floating-point disparity map of the same size as input images. In the first case the computed disparities are represented as fixed-point numbers with 4 fractional bits (i.e. the computed disparity values are multiplied by 16 and rounded to integers). 
    
    :type disparity: :class:`CvArr`
    
    
    :param state: Stereo correspondence structure. 
    
    :type state: :class:`CvStereoBMState`
    
    
    
The function cvFindStereoCorrespondenceBM computes disparity map for the input rectified stereo pair. Invalid pixels (for which disparity can not be computed) are set to 
``state->minDisparity - 1``
(or to 
``(state->minDisparity-1)*16``
in the case of 16-bit fixed-point disparity map)


.. index:: FindStereoCorrespondenceGC

.. _FindStereoCorrespondenceGC:

FindStereoCorrespondenceGC
--------------------------




.. function:: FindStereoCorrespondenceGC( left, right, dispLeft, dispRight, state, useDisparityGuess=(0))-> None

    Computes the disparity map using graph cut-based algorithm.





    
    :param left: The left single-channel, 8-bit image. 
    
    :type left: :class:`CvArr`
    
    
    :param right: The right image of the same size and the same type. 
    
    :type right: :class:`CvArr`
    
    
    :param dispLeft: The optional output single-channel 16-bit signed left disparity map of the same size as input images. 
    
    :type dispLeft: :class:`CvArr`
    
    
    :param dispRight: The optional output single-channel 16-bit signed right disparity map of the same size as input images. 
    
    :type dispRight: :class:`CvArr`
    
    
    :param state: Stereo correspondence structure. 
    
    :type state: :class:`CvStereoGCState`
    
    
    :param useDisparityGuess: If the parameter is not zero, the algorithm will start with pre-defined disparity maps. Both dispLeft and dispRight should be valid disparity maps. Otherwise, the function starts with blank disparity maps (all pixels are marked as occlusions). 
    
    :type useDisparityGuess: int
    
    
    
The function computes disparity maps for the input rectified stereo pair. Note that the left disparity image will contain values in the following range: 



.. math::

    - \texttt{state->numberOfDisparities} - \texttt{state->minDisparity} < dispLeft(x,y)  \le - \texttt{state->minDisparity} , 


or


.. math::

    dispLeft(x,y) ==  \texttt{CV\_STEREO\_GC\_OCCLUSION} 


and for the right disparity image the following will be true: 



.. math::

    \texttt{state->minDisparity} \le dispRight(x,y) 
    <  \texttt{state->minDisparity} +  \texttt{state->numberOfDisparities} 


or



.. math::

    dispRight(x,y) ==  \texttt{CV\_STEREO\_GC\_OCCLUSION} 


that is, the range for the left disparity image will be inversed,
and the pixels for which no good match has been found, will be marked
as occlusions.

Here is how the function can be used:

.. include:: ../../python_fragments/findstereocorrespondence.py
    :literal:


and this is the output left disparity image computed from the well-known
Tsukuba stereo pair and multiplied by -16 (because the values in the
left disparity images are usually negative):








.. index:: GetOptimalNewCameraMatrix

.. _GetOptimalNewCameraMatrix:

GetOptimalNewCameraMatrix
-------------------------




.. function:: GetOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, alpha, newCameraMatrix, newImageSize=(0,0), validPixROI=0) -> None

    Returns the new camera matrix based on the free scaling parameter





    
    :param cameraMatrix: The input camera matrix 
    
    :type cameraMatrix: :class:`CvMat`
    
    
    :param distCoeffs: The input vector of distortion coefficients  :math:`(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6]])`  of 4, 5 or 8 elements. If the vector is NULL/empty, the zero distortion coefficients are assumed. 
    
    :type distCoeffs: :class:`CvMat`
    
    
    :param imageSize: The original image size 
    
    :type imageSize: :class:`CvSize`
    
    
    :param alpha: The free scaling parameter between 0 (when all the pixels in the undistorted image will be valid) and 1 (when all the source image pixels will be retained in the undistorted image); see  :ref:`StereoRectify` 
    
    :type alpha: float
    
    
    :param newCameraMatrix: The output new camera matrix. 
    
    :type newCameraMatrix: :class:`CvMat`
    
    
    :param newImageSize: The image size after rectification. By default it will be set to  ``imageSize`` . 
    
    :type newImageSize: :class:`CvSize`
    
    
    :param validPixROI: The optional output rectangle that will outline all-good-pixels region in the undistorted image. See  ``roi1, roi2``  description in  :ref:`StereoRectify` 
    
    :type validPixROI: :class:`CvRect`
    
    
    
The function computes 
the optimal new camera matrix based on the free scaling parameter. By varying  this parameter the user may retrieve only sensible pixels 
``alpha=0``
, keep all the original image pixels if there is valuable information in the corners 
``alpha=1``
, or get something in between. When 
``alpha>0``
, the undistortion result will likely have some black pixels corresponding to "virtual" pixels outside of the captured distorted image. The original camera matrix, distortion coefficients, the computed new camera matrix and the 
``newImageSize``
should be passed to 
:ref:`InitUndistortRectifyMap`
to produce the maps for 
:ref:`Remap`
.


.. index:: InitIntrinsicParams2D

.. _InitIntrinsicParams2D:

InitIntrinsicParams2D
---------------------




.. function:: InitIntrinsicParams2D(objectPoints, imagePoints, npoints, imageSize, cameraMatrix, aspectRatio=1.) -> None

    Finds the initial camera matrix from the 3D-2D point correspondences





    
    :param objectPoints: The joint array of object points; see  :ref:`CalibrateCamera2` 
    
    :type objectPoints: :class:`CvMat`
    
    
    :param imagePoints: The joint array of object point projections; see  :ref:`CalibrateCamera2` 
    
    :type imagePoints: :class:`CvMat`
    
    
    :param npoints: The array of point counts; see  :ref:`CalibrateCamera2` 
    
    :type npoints: :class:`CvMat`
    
    
    :param imageSize: The image size in pixels; used to initialize the principal point 
    
    :type imageSize: :class:`CvSize`
    
    
    :param cameraMatrix: The output camera matrix  :math:`\vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}` 
    
    :type cameraMatrix: :class:`CvMat`
    
    
    :param aspectRatio: If it is zero or negative, both  :math:`f_x`  and  :math:`f_y`  are estimated independently. Otherwise  :math:`f_x = f_y * \texttt{aspectRatio}` 
    
    :type aspectRatio: float
    
    
    
The function estimates and returns the initial camera matrix for camera calibration process.
Currently, the function only supports planar calibration patterns, i.e. patterns where each object point has z-coordinate =0.


.. index:: InitUndistortMap

.. _InitUndistortMap:

InitUndistortMap
----------------




.. function:: InitUndistortMap(cameraMatrix,distCoeffs,map1,map2)-> None

    Computes an undistortion map.





    
    :param cameraMatrix: The input camera matrix  :math:`A = \vecthreethree{fx}{0}{cx}{0}{fy}{cy}{0}{0}{1}` 
    
    :type cameraMatrix: :class:`CvMat`
    
    
    :param distCoeffs: The input vector of distortion coefficients  :math:`(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6]])`  of 4, 5 or 8 elements. If the vector is NULL/empty, the zero distortion coefficients are assumed. 
    
    :type distCoeffs: :class:`CvMat`
    
    
    :param map1: The first output map  of type  ``CV_32FC1``  or  ``CV_16SC2``  - the second variant is more efficient  
    
    :type map1: :class:`CvArr`
    
    
    :param map2: The second output map  of type  ``CV_32FC1``  or  ``CV_16UC1``  - the second variant is more efficient  
    
    :type map2: :class:`CvArr`
    
    
    
The function is a simplified variant of 
:ref:`InitUndistortRectifyMap`
where the rectification transformation 
``R``
is identity matrix and 
``newCameraMatrix=cameraMatrix``
.


.. index:: InitUndistortRectifyMap

.. _InitUndistortRectifyMap:

InitUndistortRectifyMap
-----------------------




.. function:: InitUndistortRectifyMap(cameraMatrix,distCoeffs,R,newCameraMatrix,map1,map2)-> None

    Computes the undistortion and rectification transformation map.





    
    :param cameraMatrix: The input camera matrix  :math:`A=\vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}` 
    
    :type cameraMatrix: :class:`CvMat`
    
    
    :param distCoeffs: The input vector of distortion coefficients  :math:`(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6]])`  of 4, 5 or 8 elements. If the vector is NULL/empty, the zero distortion coefficients are assumed. 
    
    :type distCoeffs: :class:`CvMat`
    
    
    :param R: The optional rectification transformation in object space (3x3 matrix).  ``R1``  or  ``R2`` , computed by  :ref:`StereoRectify`  can be passed here. If the matrix is  NULL  , the identity transformation is assumed 
    
    :type R: :class:`CvMat`
    
    
    :param newCameraMatrix: The new camera matrix  :math:`A'=\vecthreethree{f_x'}{0}{c_x'}{0}{f_y'}{c_y'}{0}{0}{1}` 
    
    :type newCameraMatrix: :class:`CvMat`
    
    
    :param map1: The first output map  of type  ``CV_32FC1``  or  ``CV_16SC2``  - the second variant is more efficient  
    
    :type map1: :class:`CvArr`
    
    
    :param map2: The second output map  of type  ``CV_32FC1``  or  ``CV_16UC1``  - the second variant is more efficient  
    
    :type map2: :class:`CvArr`
    
    
    
The function computes the joint undistortion+rectification transformation and represents the result in the form of maps for 
:ref:`Remap`
. The undistorted image will look like the original, as if it was captured with a camera with camera matrix 
``=newCameraMatrix``
and zero distortion. In the case of monocular camera 
``newCameraMatrix``
is usually equal to 
``cameraMatrix``
, or it can be computed by 
:ref:`GetOptimalNewCameraMatrix`
for a better control over scaling. In the case of stereo camera 
``newCameraMatrix``
is normally set to 
``P1``
or 
``P2``
computed by 
:ref:`StereoRectify`
.

Also, this new camera will be oriented differently in the coordinate space, according to 
``R``
. That, for example, helps to align two heads of a stereo camera so that the epipolar lines on both images become horizontal and have the same y- coordinate (in the case of horizontally aligned stereo camera).

The function actually builds the maps for the inverse mapping algorithm that is used by 
:ref:`Remap`
. That is, for each pixel 
:math:`(u, v)`
in the destination (corrected and rectified) image the function computes the corresponding coordinates in the source image (i.e. in the original image from camera). The process is the following:



.. math::

    \begin{array}{l} x  \leftarrow (u - {c'}_x)/{f'}_x  \\ y  \leftarrow (v - {c'}_y)/{f'}_y  \\{[X\,Y\,W]} ^T  \leftarrow R^{-1}*[x \, y \, 1]^T  \\ x'  \leftarrow X/W  \\ y'  \leftarrow Y/W  \\ x"  \leftarrow x' (1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + 2p_1 x' y' + p_2(r^2 + 2 x'^2)  \\ y"  \leftarrow y' (1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + p_1 (r^2 + 2 y'^2) + 2 p_2 x' y'  \\ map_x(u,v)  \leftarrow x" f_x + c_x  \\ map_y(u,v)  \leftarrow y" f_y + c_y \end{array} 


where 
:math:`(k_1, k_2, p_1, p_2[, k_3])`
are the distortion coefficients. 
 
In the case of a stereo camera this function is called twice, once for each camera head, after 
:ref:`StereoRectify`
, which in its turn is called after 
:ref:`StereoCalibrate`
. But if the stereo camera was not calibrated, it is still possible to compute the rectification transformations directly from the fundamental matrix using 
:ref:`StereoRectifyUncalibrated`
. For each camera the function computes homography 
``H``
as the rectification transformation in pixel domain, not a rotation matrix 
``R``
in 3D space. The 
``R``
can be computed from 
``H``
as 



.. math::

    \texttt{R} =  \texttt{cameraMatrix} ^{-1}  \cdot \texttt{H} \cdot \texttt{cameraMatrix} 


where the 
``cameraMatrix``
can be chosen arbitrarily.


.. index:: POSIT

.. _POSIT:

POSIT
-----




.. function:: POSIT(posit_object,imagePoints,focal_length,criteria)-> (rotationMatrix,translation_vector)

    Implements the POSIT algorithm.





    
    :param posit_object: Pointer to the object structure 
    
    :type posit_object: :class:`CvPOSITObject`
    
    
    :param imagePoints: Pointer to the object points projections on the 2D image plane 
    
    :type imagePoints: :class:`CvPoint2D32f`
    
    
    :param focal_length: Focal length of the camera used 
    
    :type focal_length: float
    
    
    :param criteria: Termination criteria of the iterative POSIT algorithm 
    
    :type criteria: :class:`CvTermCriteria`
    
    
    :param rotationMatrix: Matrix of rotations 
    
    :type rotationMatrix: :class:`CvMatr32f_i`
    
    
    :param translation_vector: Translation vector 
    
    :type translation_vector: :class:`CvVect32f_i`
    
    
    
The function implements the POSIT algorithm. Image coordinates are given in a camera-related coordinate system. The focal length may be retrieved using the camera calibration functions. At every iteration of the algorithm a new perspective projection of the estimated pose is computed.

Difference norm between two projections is the maximal distance between corresponding points. The parameter 
``criteria.epsilon``
serves to stop the algorithm if the difference is small.


.. index:: ProjectPoints2

.. _ProjectPoints2:

ProjectPoints2
--------------




.. function:: ProjectPoints2(objectPoints,rvec,tvec,cameraMatrix,distCoeffs, imagePoints,dpdrot=NULL,dpdt=NULL,dpdf=NULL,dpdc=NULL,dpddist=NULL)-> None

    Project 3D points on to an image plane.





    
    :param objectPoints: The array of object points, 3xN or Nx3 1-channel or 1xN or Nx1 3-channel  , where N is the number of points in the view 
    
    :type objectPoints: :class:`CvMat`
    
    
    :param rvec: The rotation vector, see  :ref:`Rodrigues2` 
    
    :type rvec: :class:`CvMat`
    
    
    :param tvec: The translation vector 
    
    :type tvec: :class:`CvMat`
    
    
    :param cameraMatrix: The camera matrix  :math:`A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{_1}` 
    
    :type cameraMatrix: :class:`CvMat`
    
    
    :param distCoeffs: The input vector of distortion coefficients  :math:`(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6]])`  of 4, 5 or 8 elements. If the vector is NULL/empty, the zero distortion coefficients are assumed. 
    
    :type distCoeffs: :class:`CvMat`
    
    
    :param imagePoints: The output array of image points, 2xN or Nx2 1-channel or 1xN or Nx1 2-channel  
    
    :type imagePoints: :class:`CvMat`
    
    
    :param dpdrot: Optional 2Nx3 matrix of derivatives of image points with respect to components of the rotation vector 
    
    :type dpdrot: :class:`CvMat`
    
    
    :param dpdt: Optional 2Nx3 matrix of derivatives of image points with respect to components of the translation vector 
    
    :type dpdt: :class:`CvMat`
    
    
    :param dpdf: Optional 2Nx2 matrix of derivatives of image points with respect to  :math:`f_x`  and  :math:`f_y` 
    
    :type dpdf: :class:`CvMat`
    
    
    :param dpdc: Optional 2Nx2 matrix of derivatives of image points with respect to  :math:`c_x`  and  :math:`c_y` 
    
    :type dpdc: :class:`CvMat`
    
    
    :param dpddist: Optional 2Nx4 matrix of derivatives of image points with respect to distortion coefficients 
    
    :type dpddist: :class:`CvMat`
    
    
    
The function computes projections of 3D
points to the image plane given intrinsic and extrinsic camera
parameters. Optionally, the function computes jacobians - matrices
of partial derivatives of image points coordinates (as functions of all the
input parameters) with respect to the particular parameters, intrinsic and/or
extrinsic. The jacobians are used during the global optimization
in 
:ref:`CalibrateCamera2`
,
:ref:`FindExtrinsicCameraParams2`
and 
:ref:`StereoCalibrate`
. The
function itself can also used to compute re-projection error given the
current intrinsic and extrinsic parameters.

Note, that by setting 
``rvec=tvec=(0,0,0)``
, or by setting 
``cameraMatrix``
to 3x3 identity matrix, or by passing zero distortion coefficients, you can get various useful partial cases of the function, i.e. you can compute the distorted coordinates for a sparse set of points, or apply a perspective transformation (and also compute the derivatives) in the ideal zero-distortion setup etc.



.. index:: ReprojectImageTo3D

.. _ReprojectImageTo3D:

ReprojectImageTo3D
------------------




.. function:: ReprojectImageTo3D(disparity, _3dImage, Q, handleMissingValues=0) -> None

    Reprojects disparity image to 3D space.





    
    :param disparity: The input single-channel 16-bit signed or 32-bit floating-point disparity image 
    
    :type disparity: :class:`CvArr`
    
    
    :param _3dImage: The output 3-channel floating-point image of the same size as  ``disparity`` .
         Each element of  ``_3dImage(x,y)``  will contain the 3D coordinates of the point  ``(x,y)`` , computed from the disparity map. 
    
    :type _3dImage: :class:`CvArr`
    
    
    :param Q: The  :math:`4 \times 4`  perspective transformation matrix that can be obtained with  :ref:`StereoRectify` 
    
    :type Q: :class:`CvMat`
    
    
    :param handleMissingValues: If true, when the pixels with the minimal disparity (that corresponds to the outliers; see  :ref:`FindStereoCorrespondenceBM` ) will be transformed to 3D points with some very large Z value (currently set to 10000) 
    
    :type handleMissingValues: int
    
    
    
The function transforms 1-channel disparity map to 3-channel image representing a 3D surface. That is, for each pixel 
``(x,y)``
and the corresponding disparity 
``d=disparity(x,y)``
it computes: 



.. math::

    \begin{array}{l} [X \; Y \; Z \; W]^T =  \texttt{Q} *[x \; y \; \texttt{disparity} (x,y) \; 1]^T  \\ \texttt{\_3dImage} (x,y) = (X/W, \; Y/W, \; Z/W) \end{array} 


The matrix 
``Q``
can be arbitrary 
:math:`4 \times 4`
matrix, e.g. the one computed by 
:ref:`StereoRectify`
. To reproject a sparse set of points {(x,y,d),...} to 3D space, use 
:ref:`PerspectiveTransform`
.


.. index:: RQDecomp3x3

.. _RQDecomp3x3:

RQDecomp3x3
-----------




.. function:: RQDecomp3x3(M, R, Q, Qx = None, Qy = None, Qz = None) -> eulerAngles

    Computes the 'RQ' decomposition of 3x3 matrices.





    
    :param M: The 3x3 input matrix 
    
    :type M: :class:`CvMat`
    
    
    :param R: The output 3x3 upper-triangular matrix 
    
    :type R: :class:`CvMat`
    
    
    :param Q: The output 3x3 orthogonal matrix 
    
    :type Q: :class:`CvMat`
    
    
    :param Qx: Optional 3x3 rotation matrix around x-axis 
    
    :type Qx: :class:`CvMat`
    
    
    :param Qy: Optional 3x3 rotation matrix around y-axis 
    
    :type Qy: :class:`CvMat`
    
    
    :param Qz: Optional 3x3 rotation matrix around z-axis 
    
    :type Qz: :class:`CvMat`
    
    
    :param eulerAngles: Optional three Euler angles of rotation 
    
    :type eulerAngles: :class:`CvPoint3D64f`
    
    
    
The function computes a RQ decomposition using the given rotations. This function is used in 
:ref:`DecomposeProjectionMatrix`
to decompose the left 3x3 submatrix of a projection matrix into a camera and a rotation matrix.

It optionally returns three rotation matrices, one for each axis, and the three Euler angles 
that could be used in OpenGL.


.. index:: Rodrigues2

.. _Rodrigues2:

Rodrigues2
----------




.. function:: Rodrigues2(src,dst,jacobian=0)-> None

    Converts a rotation matrix to a rotation vector or vice versa.





    
    :param src: The input rotation vector (3x1 or 1x3) or rotation matrix (3x3) 
    
    :type src: :class:`CvMat`
    
    
    :param dst: The output rotation matrix (3x3) or rotation vector (3x1 or 1x3), respectively 
    
    :type dst: :class:`CvMat`
    
    
    :param jacobian: Optional output Jacobian matrix, 3x9 or 9x3 - partial derivatives of the output array components with respect to the input array components 
    
    :type jacobian: :class:`CvMat`
    
    
    


.. math::

    \begin{array}{l} \theta \leftarrow norm(r) \\ r  \leftarrow r/ \theta \\ R =  \cos{\theta} I + (1- \cos{\theta} ) r r^T +  \sin{\theta} \vecthreethree{0}{-r_z}{r_y}{r_z}{0}{-r_x}{-r_y}{r_x}{0} \end{array} 


Inverse transformation can also be done easily, since



.. math::

    \sin ( \theta ) \vecthreethree{0}{-r_z}{r_y}{r_z}{0}{-r_x}{-r_y}{r_x}{0} = \frac{R - R^T}{2} 


A rotation vector is a convenient and most-compact representation of a rotation matrix
(since any rotation matrix has just 3 degrees of freedom). The representation is
used in the global 3D geometry optimization procedures like 
:ref:`CalibrateCamera2`
,
:ref:`StereoCalibrate`
or 
:ref:`FindExtrinsicCameraParams2`
.



.. index:: StereoCalibrate

.. _StereoCalibrate:

StereoCalibrate
---------------




.. function:: StereoCalibrate( objectPoints, imagePoints1, imagePoints2, pointCounts, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize, R, T, E=NULL, F=NULL, term_crit=(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,30,1e-6), flags=CV_CALIB_FIX_INTRINSIC)-> None

    Calibrates stereo camera.





    
    :param objectPoints: The joint matrix of object points - calibration pattern features in the model coordinate space. It is floating-point 3xN or Nx3 1-channel, or 1xN or Nx1 3-channel array, where N is the total number of points in all views. 
    
    :type objectPoints: :class:`CvMat`
    
    
    :param imagePoints1: The joint matrix of object points projections in the first camera views. It is floating-point 2xN or Nx2 1-channel, or 1xN or Nx1 2-channel array, where N is the total number of points in all views 
    
    :type imagePoints1: :class:`CvMat`
    
    
    :param imagePoints2: The joint matrix of object points projections in the second camera views. It is floating-point 2xN or Nx2 1-channel, or 1xN or Nx1 2-channel array, where N is the total number of points in all views 
    
    :type imagePoints2: :class:`CvMat`
    
    
    :param pointCounts: Integer 1xM or Mx1 vector (where M is the number of calibration pattern views) containing the number of points in each particular view. The sum of vector elements must match the size of  ``objectPoints``  and  ``imagePoints*``  (=N). 
    
    :type pointCounts: :class:`CvMat`
    
    
    :param cameraMatrix1: The input/output first camera matrix:  :math:`\vecthreethree{f_x^{(j)}}{0}{c_x^{(j)}}{0}{f_y^{(j)}}{c_y^{(j)}}{0}{0}{1}` ,  :math:`j = 0,\, 1` . If any of  ``CV_CALIB_USE_INTRINSIC_GUESS`` ,    ``CV_CALIB_FIX_ASPECT_RATIO`` ,  ``CV_CALIB_FIX_INTRINSIC``  or  ``CV_CALIB_FIX_FOCAL_LENGTH``  are specified, some or all of the matrices' components must be initialized; see the flags description 
    
    :type cameraMatrix1: :class:`CvMat`
    
    
    :param distCoeffs: The input/output vector of distortion coefficients  :math:`(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6]])`  of 4, 5 or 8 elements.  
    
    
    :param cameraMatrix2: The input/output second camera matrix, as cameraMatrix1. 
    
    :type cameraMatrix2: :class:`CvMat`
    
    
    :param distCoeffs2: The input/output lens distortion coefficients for the second camera, as  ``distCoeffs1`` . 
    
    :type distCoeffs2: :class:`CvMat`
    
    
    :param imageSize: Size of the image, used only to initialize intrinsic camera matrix. 
    
    :type imageSize: :class:`CvSize`
    
    
    :param R: The output rotation matrix between the 1st and the 2nd cameras' coordinate systems. 
    
    :type R: :class:`CvMat`
    
    
    :param T: The output translation vector between the cameras' coordinate systems. 
    
    :type T: :class:`CvMat`
    
    
    :param E: The  optional   output essential matrix. 
    
    :type E: :class:`CvMat`
    
    
    :param F: The  optional   output fundamental matrix. 
    
    :type F: :class:`CvMat`
    
    
    :param term_crit: The termination criteria for the iterative optimization algorithm. 
    
    :type term_crit: :class:`CvTermCriteria`
    
    
    :param flags: Different flags, may be 0 or combination of the following values: 
         
            * **CV_CALIB_FIX_INTRINSIC** If it is set,  ``cameraMatrix?`` , as well as  ``distCoeffs?``  are fixed, so that only  ``R, T, E``  and  ``F``  are estimated. 
            
            * **CV_CALIB_USE_INTRINSIC_GUESS** The flag allows the function to optimize some or all of the intrinsic parameters, depending on the other flags, but the initial values are provided by the user. 
            
            * **CV_CALIB_FIX_PRINCIPAL_POINT** The principal points are fixed during the optimization. 
            
            * **CV_CALIB_FIX_FOCAL_LENGTH** :math:`f^{(j)}_x`  and  :math:`f^{(j)}_y`  are fixed. 
            
            * **CV_CALIB_FIX_ASPECT_RATIO** :math:`f^{(j)}_y`  is optimized, but the ratio  :math:`f^{(j)}_x/f^{(j)}_y`  is fixed. 
            
            * **CV_CALIB_SAME_FOCAL_LENGTH** Enforces  :math:`f^{(0)}_x=f^{(1)}_x`  and  :math:`f^{(0)}_y=f^{(1)}_y` 
              
            * **CV_CALIB_ZERO_TANGENT_DIST** Tangential distortion coefficients for each camera are set to zeros and fixed there. 
            
            * **CV_CALIB_FIX_K1,...,CV_CALIB_FIX_K6** Do not change the corresponding radial distortion coefficient during the optimization. If  ``CV_CALIB_USE_INTRINSIC_GUESS``  is set, the coefficient from the supplied  ``distCoeffs``  matrix is used, otherwise it is set to 0. 
            
            * **CV_CALIB_RATIONAL_MODEL** Enable coefficients k4, k5 and k6. To provide the backward compatibility, this extra flag should be explicitly specified to make the calibration function use the rational model and return 8 coefficients. If the flag is not set, the function will compute   only 5 distortion coefficients. 
            
            
    
    :type flags: int
    
    
    
The function estimates transformation between the 2 cameras making a stereo pair. If we have a stereo camera, where the relative position and orientation of the 2 cameras is fixed, and if we computed poses of an object relative to the fist camera and to the second camera, (R1, T1) and (R2, T2), respectively (that can be done with 
:ref:`FindExtrinsicCameraParams2`
), obviously, those poses will relate to each other, i.e. given (
:math:`R_1`
, 
:math:`T_1`
) it should be possible to compute (
:math:`R_2`
, 
:math:`T_2`
) - we only need to know the position and orientation of the 2nd camera relative to the 1st camera. That's what the described function does. It computes (
:math:`R`
, 
:math:`T`
) such that:



.. math::

    R_2=R*R_1
    T_2=R*T_1 + T, 


Optionally, it computes the essential matrix E:



.. math::

    E= \vecthreethree{0}{-T_2}{T_1}{T_2}{0}{-T_0}{-T_1}{T_0}{0} *R 


where 
:math:`T_i`
are components of the translation vector 
:math:`T`
: 
:math:`T=[T_0, T_1, T_2]^T`
. And also the function can compute the fundamental matrix F:



.. math::

    F = cameraMatrix2^{-T} E cameraMatrix1^{-1} 


Besides the stereo-related information, the function can also perform full calibration of each of the 2 cameras. However, because of the high dimensionality of the parameter space and noise in the input data the function can diverge from the correct solution. Thus, if intrinsic parameters can be estimated with high accuracy for each of the cameras individually (e.g. using 
:ref:`CalibrateCamera2`
), it is recommended to do so and then pass 
``CV_CALIB_FIX_INTRINSIC``
flag to the function along with the computed intrinsic parameters. Otherwise, if all the parameters are estimated at once, it makes sense to restrict some parameters, e.g. pass 
``CV_CALIB_SAME_FOCAL_LENGTH``
and 
``CV_CALIB_ZERO_TANGENT_DIST``
flags, which are usually reasonable assumptions.

Similarly to 
:ref:`CalibrateCamera2`
, the function minimizes the total re-projection error for all the points in all the available views from both cameras.

.. index:: StereoRectify

.. _StereoRectify:

StereoRectify
-------------




.. function:: StereoRectify( cameraMatrix1, cameraMatrix2, distCoeffs1, distCoeffs2, imageSize, R, T, R1, R2, P1, P2, Q=NULL, flags=CV_CALIB_ZERO_DISPARITY, alpha=-1, newImageSize=(0,0))-> (roi1, roi2)

    Computes rectification transforms for each head of a calibrated stereo camera.





    
    :param cameraMatrix1, cameraMatrix2: The camera matrices  :math:`\vecthreethree{f_x^{(j)}}{0}{c_x^{(j)}}{0}{f_y^{(j)}}{c_y^{(j)}}{0}{0}{1}` . 
    
    
    :param distCoeffs: The input vectors of distortion coefficients  :math:`(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6]])`  of 4, 5 or 8 elements each. If the vectors are NULL/empty, the zero distortion coefficients are assumed. 
    
    
    :param imageSize: Size of the image used for stereo calibration. 
    
    :type imageSize: :class:`CvSize`
    
    
    :param R: The rotation matrix between the 1st and the 2nd cameras' coordinate systems. 
    
    :type R: :class:`CvMat`
    
    
    :param T: The translation vector between the cameras' coordinate systems. 
    
    :type T: :class:`CvMat`
    
    
    :param R1, R2: The output  :math:`3 \times 3`  rectification transforms (rotation matrices) for the first and the second cameras, respectively. 
    
    
    :param P1, P2: The output  :math:`3 \times 4`  projection matrices in the new (rectified) coordinate systems. 
    
    
    :param Q: The output  :math:`4 \times 4`  disparity-to-depth mapping matrix, see  :cpp:func:`reprojectImageTo3D` . 
    
    :type Q: :class:`CvMat`
    
    
    :param flags: The operation flags; may be 0 or  ``CV_CALIB_ZERO_DISPARITY`` . If the flag is set, the function makes the principal points of each camera have the same pixel coordinates in the rectified views. And if the flag is not set, the function may still shift the images in horizontal or vertical direction (depending on the orientation of epipolar lines) in order to maximize the useful image area. 
    
    :type flags: int
    
    
    :param alpha: The free scaling parameter. If it is -1 , the functions performs some default scaling. Otherwise the parameter should be between 0 and 1.  ``alpha=0``  means that the rectified images will be zoomed and shifted so that only valid pixels are visible (i.e. there will be no black areas after rectification).  ``alpha=1``  means that the rectified image will be decimated and shifted so that all the pixels from the original images from the cameras are retained in the rectified images, i.e. no source image pixels are lost. Obviously, any intermediate value yields some intermediate result between those two extreme cases. 
    
    :type alpha: float
    
    
    :param newImageSize: The new image resolution after rectification. The same size should be passed to  :ref:`InitUndistortRectifyMap` , see the  ``stereo_calib.cpp``  sample in OpenCV samples directory. By default, i.e. when (0,0) is passed, it is set to the original  ``imageSize`` . Setting it to larger value can help you to preserve details in the original image, especially when there is big radial distortion. 
    
    :type newImageSize: :class:`CvSize`
    
    
    :param roi1, roi2: The optional output rectangles inside the rectified images where all the pixels are valid. If  ``alpha=0`` , the ROIs will cover the whole images, otherwise they likely be smaller, see the picture below 
    
    
    
The function computes the rotation matrices for each camera that (virtually) make both camera image planes the same plane. Consequently, that makes all the epipolar lines parallel and thus simplifies the dense stereo correspondence problem. On input the function takes the matrices computed by 
:cpp:func:`stereoCalibrate`
and on output it gives 2 rotation matrices and also 2 projection matrices in the new coordinates. The 2 cases are distinguished by the function are: 



    

#.
    Horizontal stereo, when 1st and 2nd camera views are shifted relative to each other mainly along the x axis (with possible small vertical shift). Then in the rectified images the corresponding epipolar lines in left and right cameras will be horizontal and have the same y-coordinate. P1 and P2 will look as: 
    
    
    
    .. math::
    
        \texttt{P1} = \begin{bmatrix} f & 0 & cx_1 & 0 \\ 0 & f & cy & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix} 
    
    
    
    
    .. math::
    
        \texttt{P2} = \begin{bmatrix} f & 0 & cx_2 & T_x*f \\ 0 & f & cy & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix} , 
    
    
    where 
    :math:`T_x`
    is horizontal shift between the cameras and 
    :math:`cx_1=cx_2`
    if 
    ``CV_CALIB_ZERO_DISPARITY``
    is set.
    

#.
    Vertical stereo, when 1st and 2nd camera views are shifted relative to each other mainly in vertical direction (and probably a bit in the horizontal direction too). Then the epipolar lines in the rectified images will be vertical and have the same x coordinate. P2 and P2 will look as:
    
    
    
    .. math::
    
        \texttt{P1} = \begin{bmatrix} f & 0 & cx & 0 \\ 0 & f & cy_1 & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix} 
    
    
    
    
    .. math::
    
        \texttt{P2} = \begin{bmatrix} f & 0 & cx & 0 \\ 0 & f & cy_2 & T_y*f \\ 0 & 0 & 1 & 0 \end{bmatrix} , 
    
    
    where 
    :math:`T_y`
    is vertical shift between the cameras and 
    :math:`cy_1=cy_2`
    if 
    ``CALIB_ZERO_DISPARITY``
    is set.
    
    
As you can see, the first 3 columns of 
``P1``
and 
``P2``
will effectively be the new "rectified" camera matrices. 
The matrices, together with 
``R1``
and 
``R2``
, can then be passed to 
:ref:`InitUndistortRectifyMap`
to initialize the rectification map for each camera.

Below is the screenshot from 
``stereo_calib.cpp``
sample. Some red horizontal lines, as you can see, pass through the corresponding image regions, i.e. the images are well rectified (which is what most stereo correspondence algorithms rely on). The green rectangles are 
``roi1``
and 
``roi2``
- indeed, their interior are all valid pixels.








.. index:: StereoRectifyUncalibrated

.. _StereoRectifyUncalibrated:

StereoRectifyUncalibrated
-------------------------




.. function:: StereoRectifyUncalibrated(points1,points2,F,imageSize,H1,H2,threshold=5)-> None

    Computes rectification transform for uncalibrated stereo camera.





    
    :param points1, points2: The 2 arrays of corresponding 2D points. The same formats as in  :ref:`FindFundamentalMat`  are supported 
    
    
    :param F: The input fundamental matrix. It can be computed from the same set of point pairs using  :ref:`FindFundamentalMat` . 
    
    :type F: :class:`CvMat`
    
    
    :param imageSize: Size of the image. 
    
    :type imageSize: :class:`CvSize`
    
    
    :param H1, H2: The output rectification homography matrices for the first and for the second images. 
    
    
    :param threshold: The optional threshold used to filter out the outliers. If the parameter is greater than zero, then all the point pairs that do not comply the epipolar geometry well enough (that is, the points for which  :math:`|\texttt{points2[i]}^T*\texttt{F}*\texttt{points1[i]}|>\texttt{threshold}` ) are rejected prior to computing the homographies.
        Otherwise all the points are considered inliers. 
    
    :type threshold: float
    
    
    
The function computes the rectification transformations without knowing intrinsic parameters of the cameras and their relative position in space, hence the suffix "Uncalibrated". Another related difference from 
:ref:`StereoRectify`
is that the function outputs not the rectification transformations in the object (3D) space, but the planar perspective transformations, encoded by the homography matrices 
``H1``
and 
``H2``
. The function implements the algorithm 
Hartley99
. 

Note that while the algorithm does not need to know the intrinsic parameters of the cameras, it heavily depends on the epipolar geometry. Therefore, if the camera lenses have significant distortion, it would better be corrected before computing the fundamental matrix and calling this function. For example, distortion coefficients can be estimated for each head of stereo camera separately by using 
:ref:`CalibrateCamera2`
and then the images can be corrected using 
:ref:`Undistort2`
, or just the point coordinates can be corrected with 
:ref:`UndistortPoints`
.



.. index:: Undistort2

.. _Undistort2:

Undistort2
----------




.. function:: Undistort2(src,dst,cameraMatrix,distCoeffs)-> None

    Transforms an image to compensate for lens distortion.





    
    :param src: The input (distorted) image 
    
    :type src: :class:`CvArr`
    
    
    :param dst: The output (corrected) image; will have the same size and the same type as  ``src`` 
    
    :type dst: :class:`CvArr`
    
    
    :param cameraMatrix: The input camera matrix  :math:`A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}` 
    
    :type cameraMatrix: :class:`CvMat`
    
    
    :param distCoeffs: The input vector of distortion coefficients  :math:`(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6]])`  of 4, 5 or 8 elements. If the vector is NULL/empty, the zero distortion coefficients are assumed. 
    
    :type distCoeffs: :class:`CvMat`
    
    
    
The function transforms the image to compensate radial and tangential lens distortion.

The function is simply a combination of 
:ref:`InitUndistortRectifyMap`
(with unity 
``R``
) and 
:ref:`Remap`
(with bilinear interpolation). See the former function for details of the transformation being performed.

Those pixels in the destination image, for which there is no correspondent pixels in the source image, are filled with 0's (black color).

The particular subset of the source image that will be visible in the corrected image can be regulated by 
``newCameraMatrix``
. You can use 
:ref:`GetOptimalNewCameraMatrix`
to compute the appropriate 
``newCameraMatrix``
, depending on your requirements.

The camera matrix and the distortion parameters can be determined using
:ref:`CalibrateCamera2`
. If the resolution of images is different from the used at the calibration stage, 
:math:`f_x, f_y, c_x`
and 
:math:`c_y`
need to be scaled accordingly, while the distortion coefficients remain the same.



.. index:: UndistortPoints

.. _UndistortPoints:

UndistortPoints
---------------




.. function:: UndistortPoints(src,dst,cameraMatrix,distCoeffs,R=NULL,P=NULL)-> None

    Computes the ideal point coordinates from the observed point coordinates.





    
    :param src: The observed point coordinates, 1xN or Nx1 2-channel (CV _ 32FC2 or CV _ 64FC2). 
    
    :type src: :class:`CvMat`
    
    
    :param dst: The output ideal point coordinates, after undistortion and reverse perspective transformation , same format as  ``src``  . 
    
    :type dst: :class:`CvMat`
    
    
    :param cameraMatrix: The camera matrix  :math:`\vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}` 
    
    :type cameraMatrix: :class:`CvMat`
    
    
    :param distCoeffs: The input vector of distortion coefficients  :math:`(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6]])`  of 4, 5 or 8 elements. If the vector is NULL/empty, the zero distortion coefficients are assumed. 
    
    :type distCoeffs: :class:`CvMat`
    
    
    :param R: The rectification transformation in object space (3x3 matrix).  ``R1``  or  ``R2`` , computed by  :cpp:func:`StereoRectify`  can be passed here. If the matrix is empty, the identity transformation is used 
    
    :type R: :class:`CvMat`
    
    
    :param P: The new camera matrix (3x3) or the new projection matrix (3x4).  ``P1``  or  ``P2`` , computed by  :cpp:func:`StereoRectify`  can be passed here. If the matrix is empty, the identity new camera matrix is used 
    
    :type P: :class:`CvMat`
    
    
    
The function is similar to 
:ref:`Undistort2`
and 
:ref:`InitUndistortRectifyMap`
, but it operates on a sparse set of points instead of a raster image. Also the function does some kind of reverse transformation to 
:ref:`ProjectPoints2`
(in the case of 3D object it will not reconstruct its 3D coordinates, of course; but for a planar object it will, up to a translation vector, if the proper 
``R``
is specified).




::


    
    // (u,v) is the input point, (u', v') is the output point
    // camera_matrix=[fx 0 cx; 0 fy cy; 0 0 1]
    // P=[fx' 0 cx' tx; 0 fy' cy' ty; 0 0 1 tz]
    x" = (u - cx)/fx
    y" = (v - cy)/fy
    (x',y') = undistort(x",y",dist_coeffs)
    [X,Y,W]T = R*[x' y' 1]T
    x = X/W, y = Y/W
    u' = x*fx' + cx'
    v' = y*fy' + cy',
    

..

where undistort() is approximate iterative algorithm that estimates the normalized original point coordinates out of the normalized distorted point coordinates ("normalized" means that the coordinates do not depend on the camera matrix).

The function can be used both for a stereo camera head or for monocular camera (when R is 
None 
).
