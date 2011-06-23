Camera Calibration and 3D Reconstruction
========================================

The functions in this section use a so-called pinhole camera model. In this model, a scene view is formed by projecting 3D points into the image plane
using a perspective transformation.

.. math::

    s  \; m' = A [R|t] M'

or

.. math::

    s  \vecthree{u}{v}{1} = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}
    \begin{bmatrix}
    r_{11} & r_{12} & r_{13} & t_1  \\
    r_{21} & r_{22} & r_{23} & t_2  \\
    r_{31} & r_{32} & r_{33} & t_3
    \end{bmatrix}
    \begin{bmatrix}
    X \\
    Y \\
    Z \\
    1
    \end{bmatrix}

where:

* :math:`(X, Y, Z)` are the coordinates of a 3D point in the world coordinate space
* :math:`(u, v)` are the coordinates of the projection point in pixels
* :math:`A` is a camera matrix, or a matrix of intrinsic parameters
* :math:`(cx, cy)` is a principal point that is usually at the image center
* :math:`fx, fy` are the focal lengths expressed in pixel-related units
Thus, if an image from the camera is
scaled by a factor, all of these parameters should
be scaled (multiplied/divided, respectively) by the same factor. The
matrix of intrinsic parameters does not depend on the scene viewed. So,
once estimated, it can be re-used as long as the focal length is fixed (in
case of zoom lens). The joint rotation-translation matrix
:math:`[R|t]` is called a matrix of extrinsic parameters. It is used to describe the
camera motion around a static scene, or vice versa, rigid motion of an
object in front of a still camera. That is,
:math:`[R|t]` translates
coordinates of a point
:math:`(X, Y, Z)` to a coordinate system,
fixed with respect to the camera. The transformation above is equivalent
to the following (when
:math:`z \ne 0` ):

.. math::

    \begin{array}{l}
    \vecthree{x}{y}{z} = R  \vecthree{X}{Y}{Z} + t \\
    x' = x/z \\
    y' = y/z \\
    u = f_x*x' + c_x \\
    v = f_y*y' + c_y
    \end{array}

Real lenses usually have some distortion, mostly
radial distortion and slight tangential distortion. So, the above model
is extended as:

.. math::

    \begin{array}{l} \vecthree{x}{y}{z} = R  \vecthree{X}{Y}{Z} + t \\ x' = x/z \\ y' = y/z \\ x'' = x'  \frac{1 + k_1 r^2 + k_2 r^4 + k_3 r^6}{1 + k_4 r^2 + k_5 r^4 + k_6 r^6} + 2 p_1 x' y' + p_2(r^2 + 2 x'^2)  \\ y'' = y'  \frac{1 + k_1 r^2 + k_2 r^4 + k_3 r^6}{1 + k_4 r^2 + k_5 r^4 + k_6 r^6} + p_1 (r^2 + 2 y'^2) + 2 p_2 x' y'  \\ \text{where} \quad r^2 = x'^2 + y'^2  \\ u = f_x*x'' + c_x \\ v = f_y*y'' + c_y \end{array}

:math:`k_1`,
:math:`k_2`,
:math:`k_3`,
:math:`k_4`,
:math:`k_5`, and
:math:`k_6` are radial distortion coefficients.
:math:`p_1` and
:math:`p_2` are tangential distortion coefficients.
Higher-order coefficients are not considered in OpenCV. In the functions below the coefficients are passed or returned as

.. math::

    (k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6]])

vector. That is, if the vector contains four elements, it means that
:math:`k_3=0` .
The distortion coefficients do not depend on the scene viewed. Thus, they also belong to the intrinsic camera parameters. And they remain the same regardless of the captured image resolution.
If, for example, a camera has been calibrated on images of
``320 x 240`` resolution, absolutely the same distortion coefficients can
be used for ``640 x 480`` images from the same camera while
:math:`f_x`,
:math:`f_y`,
:math:`c_x`, and
:math:`c_y` need to be scaled appropriately.

The functions below use the above model to do the following:

 * Project 3D points to the image plane given intrinsic and extrinsic parameters.

 * Compute extrinsic parameters given intrinsic parameters, a few 3D points, and their projections.

 * Estimate intrinsic and extrinsic camera parameters from several views of a known calibration pattern (every view is described by several 3D-2D point correspondences).

 * Estimate the relative position and orientation of the stereo camera "heads" and compute the *rectification* transformation that makes the camera optical axes parallel.



calibrateCamera
---------------

.. ocv:function:: double calibrateCamera( InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints, Size imageSize, InputOutputArray cameraMatrix, InputOutputArray distCoeffs, OutputArrayOfArrays rvecs, OutputArrayOfArrays tvecs, int flags=0 )

    Finds the camera intrinsic and extrinsic parameters from several views of a calibration pattern.

    :param objectPoints: Vector of vectors of calibration pattern points in the calibration pattern coordinate space. The outer vector contains as many elements as the number of the pattern views. If the same calibration pattern is shown in each view and it is fully visible, all the vectors will be the same. Although, it is possible to use partially occluded patterns, or even different patterns in different views. Then, the vectors will be different. The points are 3D, but since they are in a pattern coordinate system, then, if the rig is planar, it may make sense to put the model to a XY coordinate plane so that Z-coordinate of each input object point is 0.

    :param imagePoints: Vector of vectors of the projections of calibration pattern points. ``imagePoints.size()`` and ``objectPoints.size()`` and ``imagePoints[i].size()`` must be equal to ``objectPoints[i].size()`` for each ``i``.

    :param imageSize: Size of the image used only to initialize the intrinsic camera matrix.

    :param cameraMatrix: Output 3x3 floating-point camera matrix  :math:`A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}` . If  ``CV_CALIB_USE_INTRINSIC_GUESS``  and/or  ``CV_CALIB_FIX_ASPECT_RATIO``  are specified, some or all of  ``fx, fy, cx, cy``  must be initialized before calling the function.

    :param distCoeffs: Output vector of distortion coefficients  :math:`(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6]])`  of 4, 5, or 8 elements.

    :param rvecs: Output  vector of rotation vectors (see  :ref:`Rodrigues` ) estimated for each pattern view. That is, each k-th rotation vector together with the corresponding k-th translation vector (see the next output parameter description) brings the calibration pattern from the model coordinate space (in which object points are specified) to the world coordinate space, that is, a real position of the calibration pattern in the k-th pattern view (k=0.. *M* -1).

    :param tvecs: Output vector of translation vectors estimated for each pattern view.

    :param flags: Different flags that may be zero or a combination of the following values:

            * **CV_CALIB_USE_INTRINSIC_GUESS** ``cameraMatrix``  contains valid initial values of  ``fx, fy, cx, cy``  that are optimized further. Otherwise, ``(cx, cy)``  is initially set to the image center ( ``imageSize``  is used), and focal distances are computed in a least-squares fashion. Note, that if intrinsic parameters are known, there is no need to use this function just to estimate extrinsic parameters. Use  :ref:`solvePnP`  instead.

            * **CV_CALIB_FIX_PRINCIPAL_POINT** The principal point is not changed during the global optimization. It stays at the center or at a different location specified when    ``CV_CALIB_USE_INTRINSIC_GUESS``  is set too.

            * **CV_CALIB_FIX_ASPECT_RATIO** The functions considers only  ``fy``  as a free parameter. The ratio  ``fx/fy``  stays the same as in the input  ``cameraMatrix`` .   When  ``CV_CALIB_USE_INTRINSIC_GUESS``  is not set, the actual input values of  ``fx``  and  ``fy``  are ignored, only their ratio is computed and used further.

            * **CV_CALIB_ZERO_TANGENT_DIST** Tangential distortion coefficients  :math:`(p_1, p_2)`  are set to zeros and stay zero.

        * **CV_CALIB_FIX_K1,...,CV_CALIB_FIX_K6** The corresponding radial distortion coefficient is not changed during the optimization. If  ``CV_CALIB_USE_INTRINSIC_GUESS``  is set, the coefficient from the supplied  ``distCoeffs``  matrix is used. Otherwise, it is set to 0.

        * **CV_CALIB_RATIONAL_MODEL** Coefficients k4, k5, and k6 are enabled. To provide the backward compatibility, this extra flag should be explicitly specified to make the calibration function use the rational model and return 8 coefficients. If the flag is not set, the function computes  and returns  only 5 distortion coefficients.

The function estimates the intrinsic camera
parameters and extrinsic parameters for each of the views. The
coordinates of 3D object points and their corresponding 2D projections
in each view must be specified. That may be achieved by using an
object with a known geometry and easily detectable feature points.
Such an object is called a calibration rig or calibration pattern,
and OpenCV has built-in support for a chessboard as a calibration
rig (see
:ref:`findChessboardCorners` ). Currently, initialization
of intrinsic parameters (when ``CV_CALIB_USE_INTRINSIC_GUESS`` is not set) is only implemented for planar calibration patterns
(where Z-coordinates of the object points must be all zeros). 3D
calibration rigs can also be used as long as initial ``cameraMatrix`` is provided.

The algorithm performs the following steps:

#.
    Compute the initial intrinsic parameters (the option only available for planar calibration patterns) or read them from the input parameters. The distortion coefficients are all set to zeros initially unless some of ``CV_CALIB_FIX_K?``     are specified.

#.
    Estimate the initial camera pose as if the intrinsic parameters have been already known. This is done using
    :ref:`solvePnP` .
#.
    Run the global Levenberg-Marquardt optimization algorithm to minimize the reprojection error, that is, the total sum of squared distances between the observed feature points ``imagePoints``     and the projected (using the current estimates for camera parameters and the poses) object points ``objectPoints``. See :ref:`projectPoints` for details.

The function returns the final re-projection error.

**Note:**

If you use a non-square (=non-NxN) grid and
:ref:`findChessboardCorners` for calibration, and ``calibrateCamera`` returns
bad values (zero distortion coefficients, an image center very far from
:math:`(w/2-0.5,h/2-0.5)` , and/or large differences between
:math:`f_x` and
:math:`f_y` (ratios of
10:1 or more)), then you have probably used ``patternSize=cvSize(rows,cols)`` instead of using ``patternSize=cvSize(cols,rows)`` in
:ref:`FindChessboardCorners` .

See Also:
:ref:`FindChessboardCorners`,
:ref:`solvePnP`,
:ref:`initCameraMatrix2D`, 
:ref:`stereoCalibrate`,
:ref:`undistort`



calibrationMatrixValues
-----------------------
.. ocv:function:: void calibrationMatrixValues( InputArray cameraMatrix, Size imageSize, double apertureWidth, double apertureHeight, double& fovx, double& fovy, double& focalLength, Point2d& principalPoint, double& aspectRatio )

    Computes useful camera characteristics from the camera matrix.

    :param cameraMatrix: Input camera matrix that can be estimated by  :ref:`calibrateCamera`  or  :ref:`stereoCalibrate` .
    
    :param imageSize: Input image size in pixels.

    :param apertureWidth: Physical width of the sensor.

    :param apertureHeight: Physical height of the sensor.

    :param fovx: Output field of view in degrees along the horizontal sensor axis.

    :param fovy: Output field of view in degrees along the vertical sensor axis.

    :param focalLength: Focal length of the lens in mm.

    :param principalPoint: Principal point in pixels.

    :param aspectRatio: :math:`f_y/f_x`
    
The function computes various useful camera characteristics from the previously estimated camera matrix.



composeRT
-------------

.. ocv:function:: void composeRT( InputArray rvec1, InputArray tvec1, InputArray rvec2, InputArray tvec2, OutputArray rvec3, OutputArray tvec3, OutputArray dr3dr1=noArray(), OutputArray dr3dt1=noArray(), OutputArray dr3dr2=noArray(), OutputArray dr3dt2=noArray(), OutputArray dt3dr1=noArray(), OutputArray dt3dt1=noArray(), OutputArray dt3dr2=noArray(), OutputArray dt3dt2=noArray() )

    Combines two rotation-and-shift transformations.

    :param rvec1: The first rotation vector.

    :param tvec1: The first translation vector.

    :param rvec2: The second rotation vector.

    :param tvec2: The second translation vector.

    :param rvec3: Output rotation vector of the superposition.

    :param tvec3: Output translation vector of the superposition.

    :param d*d*: Optional output derivatives of  ``rvec3``  or  ``tvec3``  with regard to  ``rvec1``, ``rvec2``, ``tvec1`` and ``tvec2``, respectively.

The functions compute:

.. math::

    \begin{array}{l} \texttt{rvec3} =  \mathrm{rodrigues} ^{-1} \left ( \mathrm{rodrigues} ( \texttt{rvec2} )  \cdot \mathrm{rodrigues} ( \texttt{rvec1} ) \right )  \\ \texttt{tvec3} =  \mathrm{rodrigues} ( \texttt{rvec2} )  \cdot \texttt{tvec1} +  \texttt{tvec2} \end{array} ,

where :math:`\mathrm{rodrigues}` denotes a rotation vector to a rotation matrix transformation, and
:math:`\mathrm{rodrigues}^{-1}` denotes the inverse transformation. See :ref:`Rodrigues` for details.

Also, the functions can compute the derivatives of the output vectors with regards to the input vectors (see :ref:`matMulDeriv` ).
The functions are used inside :ref:`stereoCalibrate` but can also be used in your own code where Levenberg-Marquardt or another gradient-based solver is used to optimize a function that contains a matrix multiplication.



computeCorrespondEpilines
-----------------------------
.. ocv:function:: void computeCorrespondEpilines( InputArray points, int whichImage, InputArray F, OutputArray lines )

    For points in an image of a stereo pair, computes the corresponding epilines in the other image.

    :param points: Input points.  :math:`N \times 1`  or  :math:`1 \times N`  matrix of type  ``CV_32FC2``  or  ``vector<Point2f>`` .
    
    :param whichImage: Index of the image (1 or 2) that contains the  ``points`` .
    
    :param F: Fundamental matrix that can be estimated using  :ref:`findFundamentalMat`         or  :ref:`StereoRectify` .

    :param lines: Output vector of the epipolar lines corresponding to the points in the other image. Each line :math:`ax + by + c=0`  is encoded by 3 numbers  :math:`(a, b, c)` .
    
For every point in one of the two images of a stereo pair, the function finds the equation of the
corresponding epipolar line in the other image.

From the fundamental matrix definition (see
:ref:`findFundamentalMat` ),
line
:math:`l^{(2)}_i` in the second image for the point
:math:`p^{(1)}_i` in the first image (when ``whichImage=1`` ) is computed as:

.. math::

    l^{(2)}_i = F p^{(1)}_i

And vice versa, when ``whichImage=2``,
:math:`l^{(1)}_i` is computed from
:math:`p^{(2)}_i` as:

.. math::

    l^{(1)}_i = F^T p^{(2)}_i

Line coefficients are defined up to a scale. They are normalized so that
:math:`a_i^2+b_i^2=1` .




convertPointsToHomogeneous
------------------------

.. ocv:function:: void convertPointsToHomogeneous( InputArray src, OutputArray dst )

    Converts points from Euclidean to homogeneous space.

    :param src: Input vector of ``N``-dimensional points.

    :param dst: Output vector of ``N+1``-dimensional points.

The function converts points from Euclidean to homogeneous space by appending 1's to the tuple of point coordinates. That is, each point ``(x1, x2, ..., xn)`` is converted to ``(x1, x2, ..., xn, 1)``.



convertPointsFromHomogeneous
------------------------

.. ocv:function:: void convertPointsFromHomogeneous( InputArray src, OutputArray dst )

    Converts points from homogeneous to Euclidean space.

    :param src: Input vector of ``N``-dimensional points.

    :param dst: Output vector of ``N-1``-dimensional points.

The function converts points homogeneous to Euclidean space using perspective projection. That is, each point ``(x1, x2, ... x(n-1), xn)`` is converted to ``(x1/xn, x2/xn, ..., x(n-1)/xn)``. When ``xn=0``, the output point coordinates will be ``(0,0,0,...)``.



convertPointsHomogeneous
------------------------

.. ocv:function:: void convertPointsHomogeneous( InputArray src, OutputArray dst )

    Converts points to/from homogeneous coordinates.

    :param src: Input array or vector of 2D, 3D, or 4D points.

    :param dst: Output vector of 2D, 3D or 4D points.

The function converts 2D or 3D points from/to homogeneous coordinates by calling either :ocv:func:`convertPointsToHomogeneous` or :ocv:func:`convertPointsFromHomogeneous`. The function is obsolete; use one of the previous two instead.



.. _decomposeProjectionMatrix:

decomposeProjectionMatrix
-----------------------------
.. ocv:function:: void decomposeProjectionMatrix( InputArray projMatrix, OutputArray cameraMatrix, OutputArray rotMatrix, OutputArray transVect, OutputArray rotMatrixX=noArray(), OutputArray rotMatrixY=noArray(), OutputArray rotMatrixZ=noArray(), OutputArray eulerAngles=noArray() )

    Decomposes a projection matrix into a rotation matrix and a camera matrix.

    :param projMatrix: 3x4 input projection matrix P.

    :param cameraMatrix: The output 3x3 camera matrix K

    :param rotMatrix: Output 3x3 external rotation matrix R.

    :param transVect: Output 4x1 translation vector T.

    :param rotMatrX: Optional 3x3 rotation matrix around x-axis.

    :param rotMatrY: Optional 3x3 rotation matrix around y-axis.

    :param rotMatrZ: Optional 3x3 rotation matrix around z-axis.

    :param eulerAngles: Optional 3-element vector containing the three Euler angles of rotation.

The function computes a decomposition of a projection matrix into a calibration and a rotation matrix and the position of a camera.

It optionally returns three rotation matrices, one for each axis, and the three Euler angles that could be used in OpenGL.

The function is based on
:ref:`RQDecomp3x3` .



drawChessboardCorners
-------------------------
.. ocv:function:: void drawChessboardCorners( InputOutputArray image, Size patternSize, InputArray corners, bool patternWasFound )

    Renders the detected chessboard corners.

    :param image: Destination image. It must be an 8-bit color image.

    :param patternSize: Number of inner corners per a chessboard row and column ``(patternSize = cv::Size(points_per_row,points_per_column))``

    :param corners: Array of detected corners, the output of ``findChessboardCorners``.

    :param patternWasFound: Parameter indicating whether the complete board was found or not. The return value of :ocv:func:`findChessboardCorners` should be passed here.

The function draws individual chessboard corners detected either as red circles if the board was not found, or as colored corners connected with lines if the board was found.



findChessboardCorners
-------------------------
.. ocv:function:: bool findChessboardCorners( InputArray image, Size patternSize, OutputArray corners, int flags=CV_CALIB_CB_ADAPTIVE_THRESH+CV_CALIB_CB_NORMALIZE_IMAGE )

    Finds the positions of internal corners of the chessboard.

    :param image: Source chessboard view. It must be an 8-bit grayscale or color image.

    :param patternSize: Number of inner corners per a chessboard row and column. ``( patternSize = cvSize(points_per_row,points_per_colum) = cvSize(columns,rows) )``

    :param corners: Output array of detected corners. 

    :param flags: Various operation flags that can be zero or a combination of the following values:

            * **CV_CALIB_CB_ADAPTIVE_THRESH** Use adaptive thresholding to convert the image to black and white, rather than a fixed threshold level (computed from the average image brightness).

            * **CV_CALIB_CB_NORMALIZE_IMAGE** Normalize the image gamma with  :ref:`EqualizeHist`  before applying fixed or adaptive thresholding.

            * **CV_CALIB_CB_FILTER_QUADS** Use additional criteria (like contour area, perimeter, square-like shape) to filter out false quads that are extracted at the contour retrieval stage.

            * **CALIB_CB_FAST_CHECK** Run a fast check on the image that looks for chessboard corners, and shortcut the call if none is found. This can drastically speed up the call in the degenerate condition when no chessboard is observed.

The function attempts to determine
whether the input image is a view of the chessboard pattern and
locate the internal chessboard corners. The function returns a non-zero
value if all of the corners are found and they are placed
in a certain order (row by row, left to right in every row). Otherwise, if the function fails to find all the corners or reorder
them, it returns 0. For example, a regular chessboard has 8 x 8
squares and 7 x 7 internal corners, that is, points where the black
squares touch each other. The detected coordinates are approximate so the function calls :ref:`cornerSubPix` internally to determine their position more accurately.
You also may use the function :ref:`cornerSubPix` with different parameters if returned coordinates are not accurate enough.

Sample usage of detecting and drawing chessboard corners: ::

    Size patternsize(8,6); //interior number of corners
    Mat gray = ....; //source image
    vector<Point2f> corners; //this will be filled by the detected corners

    //CALIB_CB_FAST_CHECK saves a lot of time on images
    //that do not contain any chessboard corners
    bool patternfound = findChessboardCorners(gray, patternsize, corners,
            CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE
            + CALIB_CB_FAST_CHECK);

    if(patternfound)
      cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1),
        TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));

    drawChessboardCorners(img, patternsize, Mat(corners), patternfound);

**Note:**

The function requires white space (like a square-thick border, the wider the better) around the board to make the detection more robust in various environments. Otherwise, if there is no border and the background is dark, the outer black squares cannot be segmented properly and so the square grouping and ordering algorithm fails.



findCirclesGrid
-------------------
.. ocv:function:: bool findCirclesGrid( InputArray image, Size patternSize, OutputArray centers, int flags=CALIB_CB_SYMMETRIC_GRID, const Ptr<FeatureDetector> &blobDetector = new SimpleBlobDetector() )

    Finds the centers in the grid of circles.

    :param image: Grid view of source circles. It must be an 8-bit grayscale or color image.

    :param patternSize: Number of circles per a grid row and column ``( patternSize = Size(points_per_row, points_per_colum) )`` .

    :param centers: Output array of detected centers. 

    :param flags: Various operation flags that can be one of the following values:

            * **CALIB_CB_SYMMETRIC_GRID** Use symmetric pattern of circles.

            * **CALIB_CB_ASYMMETRIC_GRID** Use asymmetric pattern of circles.
            
            * **CALIB_CB_CLUSTERING** Use a special algorithm for grid detection. It is more robust to perspective distortions but much more sensitive to background clutter.

    :param blobDetector: FeatureDetector that finds blobs like dark circles on light background


The function attempts to determine
whether the input image contains a grid of circles. If it is, the function locates centers of the circles. The function returns a
non-zero value if all of the centers have been found and they have been placed
in a certain order (row by row, left to right in every row). Otherwise, if the function fails to find all the corners or reorder
them, it returns 0.

Sample usage of detecting and drawing the centers of circles: ::

    Size patternsize(7,7); //number of centers
    Mat gray = ....; //source image
    vector<Point2f> centers; //this will be filled by the detected centers

    bool patternfound = findCirclesGrid(gray, patternsize, centers);

    drawChessboardCorners(img, patternsize, Mat(centers), patternfound);

**Note:**

The function requires white space (like a square-thick border, the wider the better) around the board to make the detection more robust in various environments.



solvePnP
------------
.. ocv:function:: void solvePnP( InputArray objectPoints, InputArray imagePoints, InputArray cameraMatrix, InputArray distCoeffs, OutputArray rvec, OutputArray tvec, bool useExtrinsicGuess=false )

    Finds an object pose from 3D-2D point correspondences.

    :param objectPoints: Array of object points in the object coordinate space, 3xN/Nx3 1-channel or 1xN/Nx1 3-channel, where N is the number of points.  ``vector<Point3f>``  can be also passed here.

    :param imagePoints: Array of corresponding image points, 2xN/Nx2 1-channel or 1xN/Nx1 2-channel, where N is the number of points.  ``vector<Point2f>``  can be also passed here.

    :param cameraMatrix: Input camera matrix  :math:`A = \vecthreethree{fx}{0}{cx}{0}{fy}{cy}{0}{0}{1}` .
	
    :param distCoeffs: Input vector of distortion coefficients  :math:`(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6]])`  of 4, 5, or 8 elements. If the vector is NULL/empty, the zero distortion coefficients are assumed.

    :param rvec: Output rotation vector (see  :ref:`Rodrigues` ) that, together with  ``tvec`` , brings points from the model coordinate system to the camera coordinate system.

    :param tvec: Output translation vector.

    :param useExtrinsicGuess: If true (1), the function uses the provided  ``rvec``  and  ``tvec``  values as initial approximations of the rotation and translation vectors, respectively, and further optimizes them.

The function estimates the object pose given a set of object points, their corresponding image projections, as well as the camera matrix and the distortion coefficients. This function finds such a pose that minimizes reprojection error, that is, the sum of squared distances between the observed projections ``imagePoints`` and the projected (using
:ref:`projectPoints` ) ``objectPoints`` .



solvePnPRansac
------------------

.. ocv:function:: void solvePnPRansac( InputArray objectPoints, InputArray imagePoints, InputArray cameraMatrix, InputArray distCoeffs, OutputArray rvec, OutputArray tvec, bool useExtrinsicGuess=false, int iterationsCount = 100, float reprojectionError = 8.0, int minInliersCount = 100, OutputArray inliers = noArray()  )

    Finds an object pose from 3D-2D point correspondences using the RANSAC scheme.

    :param objectPoints: Array of object points in the object coordinate space, 3xN/Nx3 1-channel or 1xN/Nx1 3-channel, where N is the number of points.   ``vector<Point3f>``  can be also passed here.

    :param imagePoints: Array of corresponding image points, 2xN/Nx2 1-channel or 1xN/Nx1 2-channel, where N is the number of points.  ``vector<Point2f>``  can be also passed here.

    :param cameraMatrix: Input camera matrix  :math:`A = \vecthreethree{fx}{0}{cx}{0}{fy}{cy}{0}{0}{1}` .
    
    :param distCoeffs: Input vector of distortion coefficients  :math:`(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6]])`  of 4, 5, or 8 elements. If the vector is NULL/empty, the zero distortion coefficients are assumed.

    :param rvec: Output rotation vector (see  :ref:`Rodrigues` ) that, together with  ``tvec`` , brings points from the model coordinate system to the camera coordinate system.

    :param tvec: Output translation vector.

    :param useExtrinsicGuess: If true (1), the function uses the provided  ``rvec``  and  ``tvec`` values as initial approximations of the rotation and translation vectors, respectively, and further optimizes them.

    :param iterationsCount: Number of iterations. 
    
    :param reprojectionError: The inlier threshold value used by the RANSAC procedure. That is, the parameter value is the maximum allowed distance between the observed and computed point projections to consider it an inlier.
   
    :param minInliersCount: If the algorithm at some stage finds more inliers than ``minInliersCount`` , it finishs.
    
    :param inliers: Output vector that contains indices of inliers in ``objectPoints`` and ``imagePoints`` .

The function estimates an object pose given a set of object points, their corresponding image projections, as well as the camera matrix and the distortion coefficients. This function finds such a pose that minimizes reprojection error, that is, the sum of squared distances between the observed projections ``imagePoints`` and the projected (using
:ref:`projectPoints` ) ``objectPoints``. The use of RANSAC makes the function resistant to outliers.



findFundamentalMat
----------------------
.. ocv:function:: Mat findFundamentalMat( InputArray points1, InputArray points2, int method=FM_RANSAC, double param1=3., double param2=0.99, OutputArray mask=noArray() )

    Calculates a fundamental matrix from the corresponding points in two images.

    :param points1: Array of  ``N``  points from the first image. The point coordinates should be floating-point (single or double precision).

    :param points2: Array of the second image points of the same size and format as  ``points1`` .
	
    :param method: Method for computing a fundamental matrix.

            * **CV_FM_7POINT** for a 7-point algorithm.  :math:`N = 7`
            * **CV_FM_8POINT** for an 8-point algorithm.  :math:`N \ge 8`
            * **CV_FM_RANSAC** for the RANSAC algorithm.  :math:`N \ge 8`
            * **CV_FM_LMEDS** for the LMedS algorithm.  :math:`N \ge 8`
    
	:param param1: Parameter used for RANSAC. It is the maximum distance from a point to an epipolar line in pixels, beyond which the point is considered an outlier and is not used for computing the final fundamental matrix. It can be set to something like 1-3, depending on the accuracy of the point localization, image resolution, and the image noise.

    :param param2: Parameter used for the RANSAC or LMedS methods only. It specifies a desirable level of confidence (probability) that the estimated matrix is correct.

    :param status: Output array of N elements, every element of which is set to 0 for outliers and to 1 for the other points. The array is computed only in the RANSAC and LMedS methods. For other methods, it is set to all 1's.

The epipolar geometry is described by the following equation:

.. math::

    [p_2; 1]^T F [p_1; 1] = 0

where
:math:`F` is a fundamental matrix,
:math:`p_1` and
:math:`p_2` are corresponding points in the first and the second images, respectively.

The function calculates the fundamental matrix using one of four methods listed above and returns
the found fundamental matrix. Normally just one matrix is found. But in case of the 7-point algorithm, the function may return up to 3 solutions (
:math:`9 \times 3` matrix that stores all 3 matrices sequentially).

The calculated fundamental matrix may be passed further to
:ref:`ComputeCorrespondEpilines` that finds the epipolar lines
corresponding to the specified points. It can also be passed to
:ref:`StereoRectifyUncalibrated` to compute the rectification transformation. ::

    // Example. Estimation of fundamental matrix using the RANSAC algorithm
    int point_count = 100;
    vector<Point2f> points1(point_count);
    vector<Point2f> points2(point_count);

    // initialize the points here ... */
    for( int i = 0; i < point_count; i++ )
    {
        points1[i] = ...;
        points2[i] = ...;
    }

    Mat fundamental_matrix =
     findFundamentalMat(points1, points2, FM_RANSAC, 3, 0.99);



findHomography
------------------
.. ocv:function:: Mat findHomography( InputArray srcPoints, InputArray dstPoints, int method=0, double ransacReprojThreshold=3, OutputArray mask=noArray() )

    Finds a perspective transformation between two planes.

    :param srcPoints: Coordinates of the points in the original plane, a matrix of the type  ``CV_32FC2``  or ``vector<Point2f>`` .

    :param dstPoints: Coordinates of the points in the target plane, a matrix of the type  ``CV_32FC2``  or a  ``vector<Point2f>`` .

    :param method:  Method used to computed a homography matrix. The following methods are possible:

            * **0** - a regular method using all the points

            * **CV_RANSAC** - RANSAC-based robust method

            * **CV_LMEDS** - Least-Median robust method

    :param ransacReprojThreshold: Maximum allowed reprojection error to treat a point pair as an inlier (used in the RANSAC method only). That is, if

        .. math::

            \| \texttt{dstPoints} _i -  \texttt{convertPointsHomogeneous} ( \texttt{H} * \texttt{srcPoints} _i) \|  >  \texttt{ransacReprojThreshold}

        then the point  :math:`i`  is considered an outlier. If  ``srcPoints``  and  ``dstPoints``  are measured in pixels, it usually makes sense to set this parameter somewhere in the range of 1 to 10.

    :param status: Optional output mask set by a robust method ( ``CV_RANSAC``  or  ``CV_LMEDS`` ).  Note that the input mask values are ignored.

The functions find and return the perspective transformation :math:`H` between the source and the destination planes:

.. math::

    s_i  \vecthree{x'_i}{y'_i}{1} \sim H  \vecthree{x_i}{y_i}{1}

so that the back-projection error

.. math::

    \sum _i \left ( x'_i- \frac{h_{11} x_i + h_{12} y_i + h_{13}}{h_{31} x_i + h_{32} y_i + h_{33}} \right )^2+ \left ( y'_i- \frac{h_{21} x_i + h_{22} y_i + h_{23}}{h_{31} x_i + h_{32} y_i + h_{33}} \right )^2

is minimized. If the parameter ``method`` is set to the default value 0, the function
uses all the point pairs to compute an initial homography estimate with a simple least-squares scheme.

However, if not all of the point pairs (
:math:`srcPoints_i`,:math:`dstPoints_i` ) fit the rigid perspective transformation (that is, there
are some outliers), this initial estimate will be poor.
In this case, you can use one of the two robust methods. Both methods, ``RANSAC`` and ``LMeDS`` , try many different random subsets
of the corresponding point pairs (of four pairs each), estimate
the homography matrix using this subset and a simple least-square
algorithm, and then compute the quality/goodness of the computed homography
(which is the number of inliers for RANSAC or the median re-projection
error for LMeDs). The best subset is then used to produce the initial
estimate of the homography matrix and the mask of inliers/outliers.

Regardless of the method, robust or not, the computed homography
matrix is refined further (using inliers only in case of a robust
method) with the Levenberg-Marquardt method to reduce the
re-projection error even more.

The method ``RANSAC`` can handle practically any ratio of outliers
but it needs a threshold to distinguish inliers from outliers.
The method ``LMeDS`` does not need any threshold but it works
correctly only when there are more than 50% of inliers. Finally,
if there are no outliers and the noise is rather small, use the default method (``method=0``).

The function is used to find initial intrinsic and extrinsic matrices.
Homography matrix is determined up to a scale. Thus, it is normalized so that
:math:`h_{33}=1` .

See Also:
:ref:`GetAffineTransform`,
:ref:`GetPerspectiveTransform`,
:ref:`EstimateRigidMotion`,
:ref:`WarpPerspective`,
:ref:`PerspectiveTransform`


estimateAffine3D
--------------------
.. ocv:function:: int estimateAffine3D(InputArray srcpt, InputArray dstpt, OutputArray out,                     OutputArray inliers, double ransacThreshold = 3.0, double confidence = 0.99)

    Computes an optimal affine transformation between two 3D point sets.

    :param srcpt: The first input 3D point set.

    :param dstpt: The second input 3D point set.

    :param out: Output 3D affine transformation matrix  :math:`3 \times 4` .

    :param inliers: Output vector indicating which points are inliers.

    :param ransacThreshold: Maximum reprojection error in the RANSAC algorithm to consider a point as an inlier.

    :param confidence: The confidence level, between 0 and 1, that the estimated transformation will have. Anything between 0.95 and 0.99 is usually good enough. Too close to 1 values can slow down the estimation too much, lower than 0.8-0.9 confidence values can result in an incorrectly estimated transformation.

The function estimates an optimal 3D affine transformation between two 3D point sets using the RANSAC algorithm.




getOptimalNewCameraMatrix
-----------------------------
.. ocv:function:: Mat getOptimalNewCameraMatrix( InputArray cameraMatrix, InputArray distCoeffs, Size imageSize, double alpha, Size newImageSize=Size(), Rect* validPixROI=0)

    Returns the new camera matrix based on the free scaling parameter.

    :param cameraMatrix: Input camera matrix.

    :param distCoeffs: Input vector of distortion coefficients  :math:`(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6]])`  of 4, 5, or 8 elements. If the vector is NULL/empty, the zero distortion coefficients are assumed.

    :param imageSize: Original image size.

    :param alpha: Free scaling parameter between 0 (when all the pixels in the undistorted image are valid) and 1 (when all the source image pixels are retained in the undistorted image). See  :ref:`StereoRectify` for details.
	
    :param newCameraMatrix: Output new camera matrix.

    :param newImageSize: Image size after rectification. By default,it is set to  ``imageSize`` .

    :param validPixROI: Optional output rectangle that outlines all-good-pixels region in the undistorted image. See  ``roi1, roi2``  description in  :ref:`StereoRectify` .
    
The function computes and returns
the optimal new camera matrix based on the free scaling parameter. By varying  this parameter, you may retrieve only sensible pixels ``alpha=0`` , keep all the original image pixels if there is valuable information in the corners ``alpha=1`` , or get something in between. When ``alpha>0`` , the undistortion result is likely to have some black pixels corresponding to "virtual" pixels outside of the captured distorted image. The original camera matrix, distortion coefficients, the computed new camera matrix, and ``newImageSize`` should be passed to
:ref:`InitUndistortRectifyMap` to produce the maps for
:ref:`Remap` .



initCameraMatrix2D
----------------------
.. ocv:function:: Mat initCameraMatrix2D( InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints, Size imageSize, double aspectRatio=1.)

    Finds an initial camera matrix from 3D-2D point correspondences.

    :param objectPoints: Vector of vectors of the calibration pattern points in the calibration pattern coordinate space. See  :ref:`calibrateCamera` for details.
    
    :param imagePoints: Vector of vectors of the projections of the calibration pattern points.
    
    :param imageSize: Image size in pixels used to initialize the principal point.

    :param aspectRatio: If it is zero or negative, both  :math:`f_x`  and  :math:`f_y`  are estimated independently. Otherwise,  :math:`f_x = f_y * \texttt{aspectRatio}` .
    
The function estimates and returns an initial camera matrix for the camera calibration process.
Currently, the function only supports planar calibration patterns, which are patterns where each object point has z-coordinate =0.



matMulDeriv
---------------

.. ocv:function:: void matMulDeriv( InputArray A, InputArray B, OutputArray dABdA, OutputArray dABdB )

    Computes partial derivatives of the matrix product for each multiplied matrix.

    :param A: The first multiplied matrix.

    :param B: The second multiplied matrix.

    :param dABdA: The first output derivative matrix  ``d(A*B)/dA``  of size  :math:`\texttt{A.rows*B.cols} \times {A.rows*A.cols}` .
    
    :param dABdA: The second output derivative matrix  ``d(A*B)/dB``  of size  :math:`\texttt{A.rows*B.cols} \times {B.rows*B.cols}` .

The function computes partial derivatives of the elements of the matrix product
:math:`A*B` with regard to the elements of each of the two input matrices. The function is used to compute the Jacobian matrices in
:ref:`stereoCalibrate`  but can also be used in any other similar optimization function.



projectPoints
-----------------

.. ocv:function:: void projectPoints( InputArray objectPoints, InputArray rvec, InputArray tvec, InputArray cameraMatrix, InputArray distCoeffs, OutputArray imagePoints, OutputArray jacobian=noArray(), double aspectRatio=0 )

    Projects 3D points to an image plane.

    :param objectPoints: Array of object points, 3xN/Nx3 1-channel or 1xN/Nx1 3-channel  (or  ``vector<Point3f>`` ), where N is the number of points in the view.

    :param rvec: Rotation vector. See  :ref:`Rodrigues` for details.
    
    :param tvec: Translation vector.

    :param cameraMatrix: Camera matrix  :math:`A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{_1}` .
    
    :param distCoeffs: Input vector of distortion coefficients  :math:`(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6]])`  of 4, 5, or 8 elements. If the vector is NULL/empty, the zero distortion coefficients are assumed.

    :param imagePoints: Output array of image points, 2xN/Nx2 1-channel or 1xN/Nx1 2-channel, or  ``vector<Point2f>`` .

    :param jacobian: Optional output 2Nx(10+<numDistCoeffs>) jacobian matrix of derivatives of image points with respect to components of the rotation vector, translation vector, focal lengths, coordinates of the principal point and the distortion coefficients.

    :param aspectRatio: Optional "fixed aspect ratio" parameter. If the parameter is not 0, the function assumes that the aspect ratio (*fx/fy*) is fixed and correspondingly adjusts the jacobian matrix.

The function computes projections of 3D
points to the image plane given intrinsic and extrinsic camera
parameters. Optionally, the function computes Jacobians - matrices
of partial derivatives of image points coordinates (as functions of all the
input parameters) with respect to the particular parameters, intrinsic and/or
extrinsic. The Jacobians are used during the global optimization
in
:ref:`calibrateCamera`, 
:ref:`solvePnP`, and 
:ref:`stereoCalibrate` . The
function itself can also be used to compute a re-projection error given the
current intrinsic and extrinsic parameters.

**Note:**

By setting ``rvec=tvec=(0,0,0)``  or by setting ``cameraMatrix`` to a 3x3 identity matrix, or by passing zero distortion coefficients, you can get various useful partial cases of the function. This means that you can compute the distorted coordinates for a sparse set of points or apply a perspective transformation (and also compute the derivatives) in the ideal zero-distortion setup.



reprojectImageTo3D
----------------------

.. ocv:function:: void reprojectImageTo3D( InputArray disparity, OutputArray _3dImage, InputArray Q, bool handleMissingValues=false, int depth=-1 )

    Reprojects a disparity image to 3D space.

    :param disparity: Input single-channel 16-bit signed or 32-bit floating-point disparity image.

    :param _3dImage: Output 3-channel floating-point image of the same size as  ``disparity`` . Each element of  ``_3dImage(x,y)``  contains 3D coordinates of the point  ``(x,y)``  computed from the disparity map.

    :param Q: :math:`4 \times 4`  perspective transformation matrix that can be obtained with  :ref:`StereoRectify` .
    
    :param handleMissingValues: Indicates, whether the function should handle missing values (i.e. points where the disparity was not computed). If ``handleMissingValues=true``, then pixels with the minimal disparity that corresponds to the outliers (see  :ref:`StereoBM::operator ()` ) are transformed to 3D points with a very large Z value (currently set to 10000).

    :param ddepth: The optional output array depth. If it is ``-1``, the output image will have ``CV_32F`` depth. ``ddepth`` can also be set to ``CV_16S``, ``CV_32S`` or ``CV_32F``.
    
The function transforms a single-channel disparity map to a 3-channel image representing a 3D surface. That is, for each pixel ``(x,y)`` andthe  corresponding disparity ``d=disparity(x,y)`` , it computes:

.. math::

    \begin{array}{l} [X \; Y \; Z \; W]^T =  \texttt{Q} *[x \; y \; \texttt{disparity} (x,y) \; 1]^T  \\ \texttt{\_3dImage} (x,y) = (X/W, \; Y/W, \; Z/W) \end{array}

The matrix ``Q`` can be an arbitrary
:math:`4 \times 4` matrix (for example, the one computed by
:ref:`StereoRectify`). To reproject a sparse set of points {(x,y,d),...} to 3D space, use
:ref:`PerspectiveTransform` .



RQDecomp3x3
---------------

.. ocv:function:: Vec3d RQDecomp3x3( InputArray M, OutputArray R, OutputArray Q, OutputArray Qx=noArray(), OutputArray Qy=noArray(), OutputArray Qz=noArray() )

    Computes an RQ decomposition of 3x3 matrices.

    :param M: 3x3 input matrix.

    :param R: Output 3x3 upper-triangular matrix.

    :param Q: Output 3x3 orthogonal matrix.

    :param Qx: Optional output 3x3 rotation matrix around x-axis.

    :param Qy: Optional output 3x3 rotation matrix around y-axis.

    :param Qz: Optional output 3x3 rotation matrix around z-axis.

The function computes a RQ decomposition using the given rotations. This function is used in
:ref:`DecomposeProjectionMatrix` to decompose the left 3x3 submatrix of a projection matrix into a camera and a rotation matrix.

It optionally returns three rotation matrices, one for each axis, and the three Euler angles
(as the return value)
that could be used in OpenGL.



Rodrigues
-------------
.. ocv:function:: void Rodrigues(InputArray src, OutputArray dst, OutputArray jacobian=noArray())

    Converts a rotation matrix to a rotation vector or vice versa.

    :param src: Input rotation vector (3x1 or 1x3) or rotation matrix (3x3).

    :param dst: Output rotation matrix (3x3) or rotation vector (3x1 or 1x3), respectively.

    :param jacobian: Optional output Jacobian matrix, 3x9 or 9x3, which is a matrix of partial derivatives of the output array components with respect to the input array components.

.. math::

    \begin{array}{l} \theta \leftarrow norm(r) \\ r  \leftarrow r/ \theta \\ R =  \cos{\theta} I + (1- \cos{\theta} ) r r^T +  \sin{\theta} \vecthreethree{0}{-r_z}{r_y}{r_z}{0}{-r_x}{-r_y}{r_x}{0} \end{array}

Inverse transformation can be also done easily, since

.. math::

    \sin ( \theta ) \vecthreethree{0}{-r_z}{r_y}{r_z}{0}{-r_x}{-r_y}{r_x}{0} = \frac{R - R^T}{2}

A rotation vector is a convenient and most compact representation of a rotation matrix
(since any rotation matrix has just 3 degrees of freedom). The representation is
used in the global 3D geometry optimization procedures like
:ref:`calibrateCamera`,
:ref:`stereoCalibrate`, or
:ref:`solvePnP` .



StereoBM
--------
.. c:type:: StereoBM

Class for computing stereo correspondence using the block matching algorithm ::

    // Block matching stereo correspondence algorithm class StereoBM
    {
        enum { NORMALIZED_RESPONSE = CV_STEREO_BM_NORMALIZED_RESPONSE,
            BASIC_PRESET=CV_STEREO_BM_BASIC,
            FISH_EYE_PRESET=CV_STEREO_BM_FISH_EYE,
            NARROW_PRESET=CV_STEREO_BM_NARROW };

        StereoBM();
        // the preset is one of ..._PRESET above.
        // ndisparities is the size of disparity range,
        // in which the optimal disparity at each pixel is searched for.
        // SADWindowSize is the size of averaging window used to match pixel blocks
        //    (larger values mean better robustness to noise, but yield blurry disparity maps)
        StereoBM(int preset, int ndisparities=0, int SADWindowSize=21);
        // separate initialization function
        void init(int preset, int ndisparities=0, int SADWindowSize=21);
        // computes the disparity for the two rectified 8-bit single-channel images.
        // the disparity will be 16-bit signed (fixed-point) or 32-bit floating-point image of the same size as left.
        void operator()( InputArray left, InputArray right, OutputArray disparity, int disptype=CV_16S );

        Ptr<CvStereoBMState> state;
    };

The class is a C++ wrapper for the associated functions. In particular, ``StereoBM::operator ()`` is the wrapper for
:ref:`StereoBM::operator ()`. 




StereoBM::operator ()
-----------------------

.. ocv:function:: void StereoBM::operator()(InputArray left, InputArray right, OutputArray disp, int disptype=CV_16S )

    Computes disparity using the BM algorithm for a rectified stereo pair.

    :param left: Left 8-bit single-channel or 3-channel image.

    :param right: Right image of the same size and the same type as the left one.

    :param disp: Output disparity map. It has the same size as the input images. When ``disptype==CV_16S``, the map is a 16-bit signed single-channel image, containing disparity values scaled by 16. To get the true disparity values from such fixed-point representation, you will need to divide each  ``disp`` element by 16. If ``disptype==CV_32F``, the disparity map will already contain the real disparity values on output.
    
    :param disptype: Type of the output disparity map, ``CV_16S`` (default) or ``CV_32F``.

The method executes the BM algorithm on a rectified stereo pair. See the ``stereo_match.cpp`` OpenCV sample on how to prepare images and call the method. Note that the method is not constant, thus you should not use the same ``StereoBM`` instance from within different threads simultaneously.




StereoSGBM
----------

.. c:type:: StereoSGBM

Class for computing stereo correspondence using the semi-global block matching algorithm ::

    class StereoSGBM
    {
        StereoSGBM();
        StereoSGBM(int minDisparity, int numDisparities, int SADWindowSize,
                   int P1=0, int P2=0, int disp12MaxDiff=0,
                   int preFilterCap=0, int uniquenessRatio=0,
                   int speckleWindowSize=0, int speckleRange=0,
                   bool fullDP=false);
        virtual ~StereoSGBM();

        virtual void operator()(InputArray left, InputArray right, OutputArray disp);

        int minDisparity;
        int numberOfDisparities;
        int SADWindowSize;
        int preFilterCap;
        int uniquenessRatio;
        int P1, P2;
        int speckleWindowSize;
        int speckleRange;
        int disp12MaxDiff;
        bool fullDP;

        ...
    };

The class implements the modified H. Hirschmuller algorithm HH08 that differs from the original one as follows:

 * By default, the algorithm is single-pass, which means that you consider only 5 directions instead of 8. Set ``fullDP=true`` to run the full variant of the algorithm but beware that it may consume a lot of memory.

 * The algorithm matches blocks, not individual pixels. Though, setting ``SADWindowSize=1`` reduces the blocks to single pixels.

 * Mutual information cost function is not implemented. Instead, a simpler Birchfield-Tomasi sub-pixel metric from BT96 is used. Though, the color images are supported as well.

 * Some pre- and post- processing steps from K. Konolige algorithm :ref:`StereoBM::operator ()`  are included, for example: pre-filtering (``CV_STEREO_BM_XSOBEL`` type) and post-filtering (uniqueness check, quadratic interpolation and speckle filtering).



StereoSGBM::StereoSGBM
--------------------------
.. ocv:function:: StereoSGBM::StereoSGBM()

.. ocv:function:: StereoSGBM::StereoSGBM( int minDisparity, int numDisparities, int SADWindowSize, int P1=0, int P2=0, int disp12MaxDiff=0, int preFilterCap=0, int uniquenessRatio=0, int speckleWindowSize=0, int speckleRange=0, bool fullDP=false)

    The constructor.

    :param minDisparity: Minimum possible disparity value. Normally, it is zero but sometimes rectification algorithms can shift images, so this parameter needs to be adjusted accordingly.

    :param numDisparities: Maximum disparity minus minimum disparity. The value is always greater than zero. In the current implementation, this parameter must be divisible by 16.

    :param SADWindowSize: Matched block size. It must be an odd number  ``>=1`` . Normally, it should be somewhere in  the ``3..11``  range.

    :param P1, P2: Parameters that control disparity smoothness. The larger the values are, the smoother the disparity is.  ``P1``  is the penalty on the disparity change by plus or minus 1 between neighbor pixels.  ``P2``  is the penalty on the disparity change by more than 1 between neighbor pixels. The algorithm requires  ``P2 > P1`` . See  ``stereo_match.cpp``  sample where some reasonably good  ``P1``  and  ``P2``  values are shown (like  ``8*number_of_image_channels*SADWindowSize*SADWindowSize``  and  ``32*number_of_image_channels*SADWindowSize*SADWindowSize`` , respectively).

    :param disp12MaxDiff: Maximum allowed difference (in integer pixel units) in the left-right disparity check. Set it to a non-positive value to disable the check.

    :param preFilterCap: Truncation value for the prefiltered image pixels. The algorithm first computes x-derivative at each pixel and clips its value by  ``[-preFilterCap, preFilterCap]``  interval. The result values are passed to the Birchfield-Tomasi pixel cost function.

    :param uniquenessRatio: Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct. Normally, a value within the 5-15 range is good enough.

    :param speckleWindowSize: Maximum size of smooth disparity regions to consider their noise speckles and invalidate. Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.

    :param speckleRange: Maximum disparity variation within each connected component. If you do speckle filtering, set the parameter to a positive value, multiple of 16. Normally, 16 or 32 is good enough.

    :param fullDP: Set it to  ``true``  to run the full-scale two-pass dynamic programming algorithm. It will consume O(W*H*numDisparities) bytes, which is large for 640x480 stereo and huge for HD-size pictures. By default, it is set to ``false`` .

The first constructor initializes ``StereoSGBM`` with all the default parameters. So, you only have to set ``StereoSGBM::numberOfDisparities`` at minimum. The second constructor enables you to set each parameter to a custom value.



StereoSGBM::operator ()
-----------------------

.. ocv:function:: void StereoSGBM::operator()(InputArray left, InputArray right, OutputArray disp)

    Computes disparity using the SGBM algorithm for a rectified stereo pair.

    :param left: Left 8-bit single-channel or 3-channel image.

    :param right: Right image of the same size and the same type as the left one.

    :param disp: Output disparity map. It is a 16-bit signed single-channel image of the same size as the input image. It contains disparity values  scaled by 16. So, to get the floating-point disparity map, you need to divide each  ``disp``  element by 16.

The method executes the SGBM algorithm on a rectified stereo pair. See ``stereo_match.cpp`` OpenCV sample on how to prepare images and call the method. 

**Note**:

The method is not constant, so you should not use the same ``StereoSGBM`` instance from different threads simultaneously.



stereoCalibrate
-------------------
.. ocv:function:: double stereoCalibrate( InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints1, InputArrayOfArrays imagePoints2, InputOutputArray cameraMatrix1, InputOutputArray distCoeffs1, InputOutputArray cameraMatrix2, InputOutputArray distCoeffs2, Size imageSize, OutputArray R, OutputArray T, OutputArray E, OutputArray F, TermCriteria term_crit = TermCriteria(TermCriteria::COUNT+                         TermCriteria::EPS, 30, 1e-6), int flags=CALIB_FIX_INTRINSIC )

    Calibrates the stereo camera.

    :param objectPoints: Vector of vectors of the calibration pattern points.

    :param imagePoints1: Vector of vectors of the projections of the calibration pattern points, observed by the first camera.

    :param imagePoints2: Vector of vectors of the projections of the calibration pattern points, observed by the second camera.

    :param cameraMatrix1: Input/output first camera matrix:  :math:`\vecthreethree{f_x^{(j)}}{0}{c_x^{(j)}}{0}{f_y^{(j)}}{c_y^{(j)}}{0}{0}{1}` , 
	:math:`j = 0,\, 1` . If any of  ``CV_CALIB_USE_INTRINSIC_GUESS`` , ``CV_CALIB_FIX_ASPECT_RATIO`` , ``CV_CALIB_FIX_INTRINSIC`` , or  ``CV_CALIB_FIX_FOCAL_LENGTH``  are specified, some or all of the matrix components must be initialized. See the flags description for details.

    :param distCoeffs1: Input/output vector of distortion coefficients  :math:`(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6]])`  of 4, 5, or 8 elements. The output vector length depends on the flags.

    :param cameraMatrix2: Input/output second camera matrix. The parameter is similar to ``cameraMatrix1`` .

    :param distCoeffs2: Input/output lens distortion coefficients for the second camera. The parameter is similar to  ``distCoeffs1`` .

    :param imageSize: Size of the image used only to initialize intrinsic camera matrix.

    :param R: Output rotation matrix between the 1st and the 2nd camera coordinate systems.

    :param T: Output translation vector between the coordinate systems of the cameras.

    :param E: Output essential matrix.

    :param F: Output fundamental matrix.

    :param term_crit: Termination criteria for the iterative optimization algorithm.

    :param flags: Different flags that may be zero or a combination of the following values:

            * **CV_CALIB_FIX_INTRINSIC** Fix ``cameraMatrix?`` and  ``distCoeffs?``  so that only  ``R, T, E`` ,  and  ``F`` matrices are estimated.

            * **CV_CALIB_USE_INTRINSIC_GUESS** Optimize some or all of the intrinsic parameters according to the specified flags. Initial values are provided by the user.

            * **CV_CALIB_FIX_PRINCIPAL_POINT** Fix the principal points during the optimization.

            * **CV_CALIB_FIX_FOCAL_LENGTH** Fix :math:`f^{(j)}_x`  and  :math:`f^{(j)}_y` .

            * **CV_CALIB_FIX_ASPECT_RATIO** Optimize :math:`f^{(j)}_y` . Fix the ratio  :math:`f^{(j)}_x/f^{(j)}_y` .

            * **CV_CALIB_SAME_FOCAL_LENGTH** Enforce  :math:`f^{(0)}_x=f^{(1)}_x`  and  :math:`f^{(0)}_y=f^{(1)}_y` .
			
            * **CV_CALIB_ZERO_TANGENT_DIST** Set tangential distortion coefficients for each camera to zeros and fix there.

            * **CV_CALIB_FIX_K1,...,CV_CALIB_FIX_K6** Do not change the corresponding radial distortion coefficient during the optimization. If  ``CV_CALIB_USE_INTRINSIC_GUESS``  is set, the coefficient from the supplied  ``distCoeffs``  matrix is used. Otherwise, it is set to 0.

            * **CV_CALIB_RATIONAL_MODEL** Enable coefficients k4, k5, and k6. To provide the backward compatibility, this extra flag should be explicitly specified to make the calibration function use the rational model and return 8 coefficients. If the flag is not set, the function computes  and returns only 5 distortion coefficients.

The function estimates transformation between two cameras making a stereo pair. If you have a stereo camera where the relative position and orientation of two cameras is fixed, and if you computed poses of an object relative to the first camera and to the second camera, (R1, T1) and (R2, T2), respectively (this can be done with
:ref:`solvePnP` ), then those poses definitely relate to each other. This means that, given (
:math:`R_1`,:math:`T_1` ), it should be possible to compute (
:math:`R_2`,:math:`T_2` ). You only need to know the position and orientation of the second camera relative to the first camera. This is what the described function does. It computes (
:math:`R`,:math:`T` ) so that:

.. math::

    R_2=R*R_1
    T_2=R*T_1 + T,

Optionally, it computes the essential matrix E:

.. math::

    E= \vecthreethree{0}{-T_2}{T_1}{T_2}{0}{-T_0}{-T_1}{T_0}{0} *R

where
:math:`T_i` are components of the translation vector
:math:`T` :
:math:`T=[T_0, T_1, T_2]^T` . And the function can also compute the fundamental matrix F:

.. math::

    F = cameraMatrix2^{-T} E cameraMatrix1^{-1}

Besides the stereo-related information, the function can also perform a full calibration of each of two cameras. However, due to the high dimensionality of the parameter space and noise in the input data, the function can diverge from the correct solution. If the intrinsic parameters can be estimated with high accuracy for each of the cameras individually (for example, using
:ref:`calibrateCamera` ), you are recommended to do so and then pass ``CV_CALIB_FIX_INTRINSIC`` flag to the function along with the computed intrinsic parameters. Otherwise, if all the parameters are estimated at once, it makes sense to restrict some parameters, for example, pass ``CV_CALIB_SAME_FOCAL_LENGTH`` and ``CV_CALIB_ZERO_TANGENT_DIST`` flags, which is usually a reasonable assumption.

Similarly to :ref:`calibrateCamera` , the function minimizes the total re-projection error for all the points in all the available views from both cameras. The function returns the final value of the re-projection error.



stereoRectify
-----------------

.. ocv:function:: void stereoRectify( InputArray cameraMatrix1, InputArray distCoeffs1, InputArray cameraMatrix2, InputArray distCoeffs2, Size imageSize, InputArray R, InputArray T, OutputArray R1, OutputArray R2, OutputArray P1, OutputArray P2, OutputArray Q, int flags=CALIB_ZERO_DISPARITY, double alpha, Size newImageSize=Size(), Rect* roi1=0, Rect* roi2=0 )

    Computes rectification transforms for each head of a calibrated stereo camera.

    :param cameraMatrix1: The first camera matrix.
    
    :param cameraMatrix2: The second camera matrix.

    :param distCoeffs1: The first camera distortion parameters.
    
    :param distCoeffs2: The second camera distortion parameters.

    :param imageSize: Size of the image used for stereo calibration.

    :param R: Rotation matrix between the coordinate systems of the first and the second cameras.

    :param T: Translation vector between coordinate systems of the cameras.

    :param R1, R2: Output  :math:`3 \times 3`  rectification transforms (rotation matrices) for the first and the second cameras, respectively.

    :param P1, P2: Output  :math:`3 \times 4`  projection matrices in the new (rectified) coordinate systems.

    :param Q: Output  :math:`4 \times 4`  disparity-to-depth mapping matrix (see  :ref:`reprojectImageTo3D` ).

    :param flags: Operation flags that may be zero or  ``CV_CALIB_ZERO_DISPARITY`` . If the flag is set, the function makes the principal points of each camera have the same pixel coordinates in the rectified views. And if the flag is not set, the function may still shift the images in the horizontal or vertical direction (depending on the orientation of epipolar lines) to maximize the useful image area.

    :param alpha: Free scaling parameter. If it is -1  or absent, the function performs the default scaling. Otherwise, the parameter should be between 0 and 1.  ``alpha=0``  means that the rectified images are zoomed and shifted so that only valid pixels are visible (no black areas after rectification).  ``alpha=1``  means that the rectified image is decimated and shifted so that all the pixels from the original images from the cameras are retained in the rectified images (no source image pixels are lost). Obviously, any intermediate value yields an intermediate result between those two extreme cases.

    :param newImageSize: New image resolution after rectification. The same size should be passed to  :ref:`InitUndistortRectifyMap` (see the  ``stereo_calib.cpp``  sample in OpenCV samples directory). When (0,0) is passed (default), it is set to the original  ``imageSize`` . Setting it to larger value can help you preserve details in the original image, especially when there is a big radial distortion.

    :param roi1, roi2: Optional output rectangles inside the rectified images where all the pixels are valid. If  ``alpha=0`` , the ROIs cover the whole images. Otherwise, they are likely to be smaller (see the picture below).

The function computes the rotation matrices for each camera that (virtually) make both camera image planes the same plane. Consequently, this makes all the epipolar lines parallel and thus simplifies the dense stereo correspondence problem. The function takes the matrices computed by
:ref:`stereoCalibrate` as input. As output, it provides two rotation matrices and also two projection matrices in the new coordinates. The function distinguishes the following two cases:

#.
    **Horizontal stereo**: the first and the second camera views are shifted relative to each other mainly along the x axis (with possible small vertical shift). In the rectified images, the corresponding epipolar lines in the left and right cameras are horizontal and have the same y-coordinate. P1 and P2 look like:

    .. math::

        \texttt{P1} = \begin{bmatrix} f & 0 & cx_1 & 0 \\ 0 & f & cy & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix}

    .. math::

        \texttt{P2} = \begin{bmatrix} f & 0 & cx_2 & T_x*f \\ 0 & f & cy & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix} ,

    where
    :math:`T_x`     is a horizontal shift between the cameras and
    :math:`cx_1=cx_2`     if ``CV_CALIB_ZERO_DISPARITY``     is set.

#.
    **Vertical stereo**: the first and the second camera views are shifted relative to each other mainly in vertical direction (and probably a bit in the horizontal direction too). The epipolar lines in the rectified images are vertical and have the same x-coordinate. P1 and P2 look like:

    .. math::

        \texttt{P1} = \begin{bmatrix} f & 0 & cx & 0 \\ 0 & f & cy_1 & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix}

    .. math::

        \texttt{P2} = \begin{bmatrix} f & 0 & cx & 0 \\ 0 & f & cy_2 & T_y*f \\ 0 & 0 & 1 & 0 \end{bmatrix} ,

    where
    :math:`T_y`     is a vertical shift between the cameras and
    :math:`cy_1=cy_2`     if ``CALIB_ZERO_DISPARITY``     is set.

As you can see, the first three columns of ``P1`` and ``P2`` will effectively be the new "rectified" camera matrices.
The matrices, together with ``R1`` and ``R2`` , can then be passed to
:ref:`InitUndistortRectifyMap` to initialize the rectification map for each camera.

See below the screenshot from the ``stereo_calib.cpp`` sample. Some red horizontal lines pass through the corresponding image regions. This means that the images are well rectified, which is what most stereo correspondence algorithms rely on. The green rectangles are ``roi1`` and ``roi2`` . You see that their interiors are all valid pixels.

.. image:: pics/stereo_undistort.jpg



stereoRectifyUncalibrated
-----------------------------
.. ocv:function:: bool stereoRectifyUncalibrated( InputArray points1, InputArray points2, InputArray F, Size imgSize, OutputArray H1, OutputArray H2, double threshold=5 )

    Computes a rectification transform for an uncalibrated stereo camera.

    :param points1, points2: Two arrays of corresponding 2D points. The same formats as in  :ref:`findFundamentalMat`  are supported.

    :param F: Input fundamental matrix. It can be computed from the same set of point pairs using  :ref:`findFundamentalMat` .

    :param imageSize: Size of the image.

    :param H1, H2: Output rectification homography matrices for the first and for the second images.

    :param threshold: Optional threshold used to filter out the outliers. If the parameter is greater than zero, all the point pairs that do not comply with the epipolar geometry (that is, the points for which  :math:`|\texttt{points2[i]}^T*\texttt{F}*\texttt{points1[i]}|>\texttt{threshold}` ) are rejected prior to computing the homographies. Otherwise,all the points are considered inliers.

The function computes the rectification transformations without knowing intrinsic parameters of the cameras and their relative position in the space, which explains the suffix "uncalibrated". Another related difference from
:ref:`StereoRectify` is that the function outputs not the rectification transformations in the object (3D) space, but the planar perspective transformations encoded by the homography matrices ``H1`` and ``H2`` . The function implements the algorithm
Hartley99
.

**Note**:

While the algorithm does not need to know the intrinsic parameters of the cameras, it heavily depends on the epipolar geometry. Therefore, if the camera lenses have a significant distortion, it would be better to correct it before computing the fundamental matrix and calling this function. For example, distortion coefficients can be estimated for each head of stereo camera separately by using
:ref:`calibrateCamera` . Then, the images can be corrected using
:ref:`undistort` , or just the point coordinates can be corrected with
:ref:`undistortPoints` .
