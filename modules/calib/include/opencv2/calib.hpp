// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef OPENCV_CALIB_HPP
#define OPENCV_CALIB_HPP

#include "opencv2/core.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/features.hpp"
#include "opencv2/core/affine.hpp"

/**
  @defgroup calib Camera Calibration

The functions in this section use a so-called pinhole camera model. The view of a scene
is obtained by projecting a scene's 3D point \f$P_w\f$ into the image plane using a perspective
transformation which forms the corresponding pixel \f$p\f$. Both \f$P_w\f$ and \f$p\f$ are
represented in homogeneous coordinates, i.e. as 3D and 2D homogeneous vector respectively. You will
find a brief introduction to projective geometry, homogeneous vectors and homogeneous
transformations at the end of this section's introduction. For more succinct notation, we often drop
the 'homogeneous' and say vector instead of homogeneous vector.

The distortion-free projective transformation given by a  pinhole camera model is shown below.

\f[s \; p = A \begin{bmatrix} R|t \end{bmatrix} P_w,\f]

where \f$P_w\f$ is a 3D point expressed with respect to the world coordinate system,
\f$p\f$ is a 2D pixel in the image plane, \f$A\f$ is the camera intrinsic matrix,
\f$R\f$ and \f$t\f$ are the rotation and translation that describe the change of coordinates from
world to camera coordinate systems (or camera frame) and \f$s\f$ is the projective transformation's
arbitrary scaling and not part of the camera model.

The camera intrinsic matrix \f$A\f$ (notation used as in @cite Zhang2000 and also generally notated
as \f$K\f$) projects 3D points given in the camera coordinate system to 2D pixel coordinates, i.e.

\f[p = A P_c.\f]

The camera intrinsic matrix \f$A\f$ is composed of the focal lengths \f$f_x\f$ and \f$f_y\f$, which are
expressed in pixel units, and the principal point \f$(c_x, c_y)\f$, that is usually close to the
image center:

\f[A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1},\f]

and thus

\f[s \vecthree{u}{v}{1} = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1} \vecthree{X_c}{Y_c}{Z_c}.\f]

The matrix of intrinsic parameters does not depend on the scene viewed. So, once estimated, it can
be re-used as long as the focal length is fixed (in case of a zoom lens). Thus, if an image from the
camera is scaled by a factor, all of these parameters need to be scaled (multiplied/divided,
respectively) by the same factor.

The joint rotation-translation matrix \f$[R|t]\f$ is the matrix product of a projective
transformation and a homogeneous transformation. The 3-by-4 projective transformation maps 3D points
represented in camera coordinates to 2D points in the image plane and represented in normalized
camera coordinates \f$x' = X_c / Z_c\f$ and \f$y' = Y_c / Z_c\f$:

\f[Z_c \begin{bmatrix}
x' \\
y' \\
1
\end{bmatrix} = \begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0
\end{bmatrix}
\begin{bmatrix}
X_c \\
Y_c \\
Z_c \\
1
\end{bmatrix}.\f]

The homogeneous transformation is encoded by the extrinsic parameters \f$R\f$ and \f$t\f$ and
represents the change of basis from world coordinate system \f$w\f$ to the camera coordinate sytem
\f$c\f$. Thus, given the representation of the point \f$P\f$ in world coordinates, \f$P_w\f$, we
obtain \f$P\f$'s representation in the camera coordinate system, \f$P_c\f$, by

\f[P_c = \begin{bmatrix}
R & t \\
0 & 1
\end{bmatrix} P_w,\f]

This homogeneous transformation is composed out of \f$R\f$, a 3-by-3 rotation matrix, and \f$t\f$, a
3-by-1 translation vector:

\f[\begin{bmatrix}
R & t \\
0 & 1
\end{bmatrix} = \begin{bmatrix}
r_{11} & r_{12} & r_{13} & t_x \\
r_{21} & r_{22} & r_{23} & t_y \\
r_{31} & r_{32} & r_{33} & t_z \\
0 & 0 & 0 & 1
\end{bmatrix},
\f]

and therefore

\f[\begin{bmatrix}
X_c \\
Y_c \\
Z_c \\
1
\end{bmatrix} = \begin{bmatrix}
r_{11} & r_{12} & r_{13} & t_x \\
r_{21} & r_{22} & r_{23} & t_y \\
r_{31} & r_{32} & r_{33} & t_z \\
0 & 0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
X_w \\
Y_w \\
Z_w \\
1
\end{bmatrix}.\f]

Combining the projective transformation and the homogeneous transformation, we obtain the projective
transformation that maps 3D points in world coordinates into 2D points in the image plane and in
normalized camera coordinates:

\f[Z_c \begin{bmatrix}
x' \\
y' \\
1
\end{bmatrix} = \begin{bmatrix} R|t \end{bmatrix} \begin{bmatrix}
X_w \\
Y_w \\
Z_w \\
1
\end{bmatrix} = \begin{bmatrix}
r_{11} & r_{12} & r_{13} & t_x \\
r_{21} & r_{22} & r_{23} & t_y \\
r_{31} & r_{32} & r_{33} & t_z
\end{bmatrix}
\begin{bmatrix}
X_w \\
Y_w \\
Z_w \\
1
\end{bmatrix},\f]

with \f$x' = X_c / Z_c\f$ and \f$y' = Y_c / Z_c\f$. Putting the equations for instrincs and extrinsics together, we can write out
\f$s \; p = A \begin{bmatrix} R|t \end{bmatrix} P_w\f$ as

\f[s \vecthree{u}{v}{1} = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}
\begin{bmatrix}
r_{11} & r_{12} & r_{13} & t_x \\
r_{21} & r_{22} & r_{23} & t_y \\
r_{31} & r_{32} & r_{33} & t_z
\end{bmatrix}
\begin{bmatrix}
X_w \\
Y_w \\
Z_w \\
1
\end{bmatrix}.\f]

If \f$Z_c \ne 0\f$, the transformation above is equivalent to the following,

\f[\begin{bmatrix}
u \\
v
\end{bmatrix} = \begin{bmatrix}
f_x X_c/Z_c + c_x \\
f_y Y_c/Z_c + c_y
\end{bmatrix}\f]

with

\f[\vecthree{X_c}{Y_c}{Z_c} = \begin{bmatrix}
R|t
\end{bmatrix} \begin{bmatrix}
X_w \\
Y_w \\
Z_w \\
1
\end{bmatrix}.\f]

The following figure illustrates the pinhole camera model.

![Pinhole camera model](pics/pinhole_camera_model.png)

Real lenses usually have some distortion, mostly radial distortion, and slight tangential distortion.
So, the above model is extended as:

\f[\begin{bmatrix}
u \\
v
\end{bmatrix} = \begin{bmatrix}
f_x x'' + c_x \\
f_y y'' + c_y
\end{bmatrix}\f]

where

\f[\begin{bmatrix}
x'' \\
y''
\end{bmatrix} = \begin{bmatrix}
x' \frac{1 + k_1 r^2 + k_2 r^4 + k_3 r^6}{1 + k_4 r^2 + k_5 r^4 + k_6 r^6} + 2 p_1 x' y' + p_2(r^2 + 2 x'^2) + s_1 r^2 + s_2 r^4 \\
y' \frac{1 + k_1 r^2 + k_2 r^4 + k_3 r^6}{1 + k_4 r^2 + k_5 r^4 + k_6 r^6} + p_1 (r^2 + 2 y'^2) + 2 p_2 x' y' + s_3 r^2 + s_4 r^4 \\
\end{bmatrix}\f]

with

\f[r^2 = x'^2 + y'^2\f]

and

\f[\begin{bmatrix}
x'\\
y'
\end{bmatrix} = \begin{bmatrix}
X_c/Z_c \\
Y_c/Z_c
\end{bmatrix},\f]

if \f$Z_c \ne 0\f$.

The distortion parameters are the radial coefficients \f$k_1\f$, \f$k_2\f$, \f$k_3\f$, \f$k_4\f$, \f$k_5\f$, and \f$k_6\f$
,\f$p_1\f$ and \f$p_2\f$ are the tangential distortion coefficients, and \f$s_1\f$, \f$s_2\f$, \f$s_3\f$, and \f$s_4\f$,
are the thin prism distortion coefficients. Higher-order coefficients are not considered in OpenCV.

The next figures show two common types of radial distortion: barrel distortion
(\f$ 1 + k_1 r^2 + k_2 r^4 + k_3 r^6 \f$ monotonically decreasing)
and pincushion distortion (\f$ 1 + k_1 r^2 + k_2 r^4 + k_3 r^6 \f$ monotonically increasing).
Radial distortion is always monotonic for real lenses,
and if the estimator produces a non-monotonic result,
this should be considered a calibration failure.
More generally, radial distortion must be monotonic and the distortion function must be bijective.
A failed estimation result may look deceptively good near the image center
but will work poorly in e.g. AR/SFM applications.
The optimization method used in OpenCV camera calibration does not include these constraints as
the framework does not support the required integer programming and polynomial inequalities.
See [issue #15992](https://github.com/opencv/opencv/issues/15992) for additional information.

![](pics/distortion_examples.png)
![](pics/distortion_examples2.png)

In some cases, the image sensor may be tilted in order to focus an oblique plane in front of the
camera (Scheimpflug principle). This can be useful for particle image velocimetry (PIV) or
triangulation with a laser fan. The tilt causes a perspective distortion of \f$x''\f$ and
\f$y''\f$. This distortion can be modeled in the following way, see e.g. @cite Louhichi07.

\f[\begin{bmatrix}
u \\
v
\end{bmatrix} = \begin{bmatrix}
f_x x''' + c_x \\
f_y y''' + c_y
\end{bmatrix},\f]

where

\f[s\vecthree{x'''}{y'''}{1} =
\vecthreethree{R_{33}(\tau_x, \tau_y)}{0}{-R_{13}(\tau_x, \tau_y)}
{0}{R_{33}(\tau_x, \tau_y)}{-R_{23}(\tau_x, \tau_y)}
{0}{0}{1} R(\tau_x, \tau_y) \vecthree{x''}{y''}{1}\f]

and the matrix \f$R(\tau_x, \tau_y)\f$ is defined by two rotations with angular parameter
\f$\tau_x\f$ and \f$\tau_y\f$, respectively,

\f[
R(\tau_x, \tau_y) =
\vecthreethree{\cos(\tau_y)}{0}{-\sin(\tau_y)}{0}{1}{0}{\sin(\tau_y)}{0}{\cos(\tau_y)}
\vecthreethree{1}{0}{0}{0}{\cos(\tau_x)}{\sin(\tau_x)}{0}{-\sin(\tau_x)}{\cos(\tau_x)} =
\vecthreethree{\cos(\tau_y)}{\sin(\tau_y)\sin(\tau_x)}{-\sin(\tau_y)\cos(\tau_x)}
{0}{\cos(\tau_x)}{\sin(\tau_x)}
{\sin(\tau_y)}{-\cos(\tau_y)\sin(\tau_x)}{\cos(\tau_y)\cos(\tau_x)}.
\f]

In the functions below the coefficients are passed or returned as

\f[(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\f]

vector. That is, if the vector contains four elements, it means that \f$k_3=0\f$ . The distortion
coefficients do not depend on the scene viewed. Thus, they also belong to the intrinsic camera
parameters. And they remain the same regardless of the captured image resolution. If, for example, a
camera has been calibrated on images of 320 x 240 resolution, absolutely the same distortion
coefficients can be used for 640 x 480 images from the same camera while \f$f_x\f$, \f$f_y\f$,
\f$c_x\f$, and \f$c_y\f$ need to be scaled appropriately.

The functions below use the above model to do the following:

-   Project 3D points to the image plane given intrinsic and extrinsic parameters.
-   Compute extrinsic parameters given intrinsic parameters, a few 3D points, and their
projections.
-   Estimate intrinsic and extrinsic camera parameters from several views of a known calibration
pattern (every view is described by several 3D-2D point correspondences).
-   Estimate the relative position and orientation of the stereo camera "heads" and compute the
*rectification* transformation that makes the camera optical axes parallel.

<B> Homogeneous Coordinates </B><br>
Homogeneous Coordinates are a system of coordinates that are used in projective geometry. Their use
allows to represent points at infinity by finite coordinates and simplifies formulas when compared
to the cartesian counterparts, e.g. they have the advantage that affine transformations can be
expressed as linear homogeneous transformation.

One obtains the homogeneous vector \f$P_h\f$ by appending a 1 along an n-dimensional cartesian
vector \f$P\f$ e.g. for a 3D cartesian vector the mapping \f$P \rightarrow P_h\f$ is:

\f[\begin{bmatrix}
X \\
Y \\
Z
\end{bmatrix} \rightarrow \begin{bmatrix}
X \\
Y \\
Z \\
1
\end{bmatrix}.\f]

For the inverse mapping \f$P_h \rightarrow P\f$, one divides all elements of the homogeneous vector
by its last element, e.g. for a 3D homogeneous vector one gets its 2D cartesian counterpart by:

\f[\begin{bmatrix}
X \\
Y \\
W
\end{bmatrix} \rightarrow \begin{bmatrix}
X / W \\
Y / W
\end{bmatrix},\f]

if \f$W \ne 0\f$.

Due to this mapping, all multiples \f$k P_h\f$, for \f$k \ne 0\f$, of a homogeneous point represent
the same point \f$P_h\f$. An intuitive understanding of this property is that under a projective
transformation, all multiples of \f$P_h\f$ are mapped to the same point. This is the physical
observation one does for pinhole cameras, as all points along a ray through the camera's pinhole are
projected to the same image point, e.g. all points along the red ray in the image of the pinhole
camera model above would be mapped to the same image coordinate. This property is also the source
for the scale ambiguity s in the equation of the pinhole camera model.

As mentioned, by using homogeneous coordinates we can express any change of basis parameterized by
\f$R\f$ and \f$t\f$ as a linear transformation, e.g. for the change of basis from coordinate system
0 to coordinate system 1 becomes:

\f[P_1 = R P_0 + t \rightarrow P_{h_1} = \begin{bmatrix}
R & t \\
0 & 1
\end{bmatrix} P_{h_0}.\f]

@note
    -   Many functions in this module take a camera intrinsic matrix as an input parameter. Although all
        functions assume the same structure of this parameter, they may name it differently. The
        parameter's description, however, will be clear in that a camera intrinsic matrix with the structure
        shown above is required.
    -   A calibration sample for 3 cameras in a horizontal position can be found at
        opencv_source_code/samples/cpp/3calibration.cpp
    -   A calibration sample based on a sequence of images can be found at
        opencv_source_code/samples/cpp/calibration.cpp
    -   A calibration sample in order to do 3D reconstruction can be found at
        opencv_source_code/samples/cpp/build3dmodel.cpp
    -   A calibration example on stereo calibration can be found at
        opencv_source_code/samples/cpp/stereo_calib.cpp
    -   A calibration example on stereo matching can be found at
        opencv_source_code/samples/cpp/stereo_match.cpp
    -   (Python) A camera calibration sample can be found at
        opencv_source_code/samples/python/calibrate.py

  @{
    @defgroup calib3d_fisheye Fisheye camera model

    Definitions: Let P be a point in 3D of coordinates X in the world reference frame (stored in the
    matrix X) The coordinate vector of P in the camera reference frame is:

    \f[Xc = R X + T\f]

    where R is the rotation matrix corresponding to the rotation vector om: R = rodrigues(om); call x, y
    and z the 3 coordinates of Xc:

    \f[\begin{array}{l} x = Xc_1 \\ y = Xc_2 \\ z = Xc_3 \end{array} \f]

    The pinhole projection coordinates of P is [a; b] where

    \f[\begin{array}{l} a = x / z \ and \ b = y / z \\ r^2 = a^2 + b^2 \\ \theta = atan(r) \end{array} \f]

    Fisheye distortion:

    \f[\theta_d = \theta (1 + k_1 \theta^2 + k_2 \theta^4 + k_3 \theta^6 + k_4 \theta^8)\f]

    The distorted point coordinates are [x'; y'] where

    \f[\begin{array}{l} x' = (\theta_d / r) a \\ y' = (\theta_d / r) b \end{array} \f]

    Finally, conversion into pixel coordinates: The final pixel coordinates vector [u; v] where:

    \f[\begin{array}{l} u = f_x (x' + \alpha y') + c_x \\
    v = f_y y' + c_y \end{array} \f]

    Summary:
    Generic camera model @cite Kannala2006 with perspective projection and without distortion correction

  @}
 */

namespace cv {

//! @addtogroup calib
//! @{

enum { CALIB_CB_ADAPTIVE_THRESH = 1,
       CALIB_CB_NORMALIZE_IMAGE = 2,
       CALIB_CB_FILTER_QUADS    = 4,
       CALIB_CB_FAST_CHECK      = 8,
       CALIB_CB_EXHAUSTIVE      = 16,
       CALIB_CB_ACCURACY        = 32,
       CALIB_CB_LARGER          = 64,
       CALIB_CB_MARKER          = 128,
       CALIB_CB_PLAIN           = 256
     };

enum { CALIB_CB_SYMMETRIC_GRID  = 1,
       CALIB_CB_ASYMMETRIC_GRID = 2,
       CALIB_CB_CLUSTERING      = 4
     };

#define CALIB_NINTRINSIC 18 //!< Maximal size of camera internal parameters (initrinsics) vector

enum CameraModel {
    CALIB_MODEL_PINHOLE = 0, //!< Pinhole camera model
    CALIB_MODEL_FISHEYE = 1, //!< Fisheye camera model
};

enum { CALIB_USE_INTRINSIC_GUESS = 0x00001, //!< Use user provided intrinsics as initial point for optimization.
       CALIB_FIX_ASPECT_RATIO    = 0x00002, //!< Use with CALIB_USE_INTRINSIC_GUESS. The ratio fx/fy stays the same as in the input cameraMatrix.
       CALIB_FIX_PRINCIPAL_POINT = 0x00004, //!< The principal point (cx, cy) stays the same as in the input camera matrix. Image center is used as principal point, if CALIB_USE_INTRINSIC_GUESS is not set.
       CALIB_ZERO_TANGENT_DIST   = 0x00008, //!< For pinhole model only. Tangential distortion coefficients \f$(p_1, p_2)\f$ are set to zeros and stay zero.
       CALIB_FIX_FOCAL_LENGTH    = 0x00010, //!< Use with CALIB_USE_INTRINSIC_GUESS. The focal length (fx, fy) stays the same as in the input cameraMatrix.
       CALIB_FIX_K1              = 0x00020, //!< The corresponding distortion coefficient is not changed during the optimization. 0 value is used, if CALIB_USE_INTRINSIC_GUESS is not set.
       CALIB_FIX_K2              = 0x00040, //!< The corresponding distortion coefficient is not changed during the optimization. 0 value is used, if CALIB_USE_INTRINSIC_GUESS is not set.
       CALIB_FIX_K3              = 0x00080, //!< The corresponding distortion coefficient is not changed during the optimization. 0 value is used, if CALIB_USE_INTRINSIC_GUESS is not set.
       CALIB_FIX_K4              = 0x00800, //!< The corresponding distortion coefficient is not changed during the optimization. 0 value is used, if CALIB_USE_INTRINSIC_GUESS is not set.
       CALIB_FIX_K5              = 0x01000, //!< For pinhole model only. The corresponding distortion coefficient is not changed during the optimization. 0 value is used, if CALIB_USE_INTRINSIC_GUESS is not set.
       CALIB_FIX_K6              = 0x02000, //!< For pinhole model only. The corresponding distortion coefficient is not changed during the optimization. 0 value is used, if CALIB_USE_INTRINSIC_GUESS is not set.
       CALIB_RATIONAL_MODEL      = 0x04000, //!< For pinhole model only. Use rational distortion model with coefficients k4..k6.
       CALIB_THIN_PRISM_MODEL    = 0x08000, //!< For pinhole model only. Use thin prism distortion model with coefficients s1..s4.
       CALIB_FIX_S1_S2_S3_S4     = 0x10000, //!< For pinhole model only. The thin prism distortion coefficients are not changed during the optimization. 0 value is used, if CALIB_USE_INTRINSIC_GUESS is not set.
       CALIB_TILTED_MODEL        = 0x40000, //!< For pinhole model only. Coefficients tauX and tauY are enabled in camera matrix.
       CALIB_FIX_TAUX_TAUY       = 0x80000, //!< For pinhole model only. The tauX and tauY coefficients are not changed during the optimization. 0 value is used, if CALIB_USE_INTRINSIC_GUESS is not set.
       CALIB_USE_QR              = 0x100000, //!< Use QR instead of SVD decomposition for solving. Faster but potentially less precise
       CALIB_FIX_TANGENT_DIST    = 0x200000, //!< For pinhole model only. Tangential distortion coefficients (p1,p2) are set to zeros and stay zero.
       // only for stereo
       CALIB_FIX_INTRINSIC       = 0x00100, //!< For stereo and milti-camera calibration only. Do not optimize cameras intrinsics
       CALIB_SAME_FOCAL_LENGTH   = 0x00200, //!< For stereo calibration only. Use the same focal length for cameras in pair.
       // for stereo rectification
       CALIB_ZERO_DISPARITY      = 0x00400, //!< Deprecated synonim of @ref STEREO_ZERO_DISPARITY. See @ref stereoRectify.
       CALIB_USE_LU              = (1 << 17), //!< use LU instead of SVD decomposition for solving. much faster but potentially less precise
       CALIB_USE_EXTRINSIC_GUESS = (1 << 22), //!< For stereo and multi-view calibration. Use user provided extrinsics (R, T) as initial point for optimization
       // fisheye only flags
       CALIB_RECOMPUTE_EXTRINSIC = (1 << 23), //!< For fisheye model only. Recompute board position on each calibration iteration
       CALIB_CHECK_COND          = (1 << 24), //!< For fisheye model only. Check SVD decomposition quality for each frame during extrinsics estimation
       CALIB_FIX_SKEW            = (1 << 25), //!< For fisheye model only. Skew coefficient (alpha) is set to zero and stay zero.
       CALIB_STEREO_REGISTRATION = (1 << 26)  //!< For multiview calibration only. Use stereo correspondence approach for initial extrinsics guess. Limitation: all cameras should have the same type.
     };

enum HandEyeCalibrationMethod
{
    CALIB_HAND_EYE_TSAI         = 0, //!< A New Technique for Fully Autonomous and Efficient 3D Robotics Hand/Eye Calibration @cite Tsai89
    CALIB_HAND_EYE_PARK         = 1, //!< Robot Sensor Calibration: Solving AX = XB on the Euclidean Group @cite Park94
    CALIB_HAND_EYE_HORAUD       = 2, //!< Hand-eye Calibration @cite Horaud95
    CALIB_HAND_EYE_ANDREFF      = 3, //!< On-line Hand-Eye Calibration @cite Andreff99
    CALIB_HAND_EYE_DANIILIDIS   = 4  //!< Hand-Eye Calibration Using Dual Quaternions @cite Daniilidis98
};

enum RobotWorldHandEyeCalibrationMethod
{
    CALIB_ROBOT_WORLD_HAND_EYE_SHAH = 0, //!< Solving the robot-world/hand-eye calibration problem using the kronecker product @cite Shah2013SolvingTR
    CALIB_ROBOT_WORLD_HAND_EYE_LI   = 1  //!< Simultaneous robot-world and hand-eye calibration using dual-quaternions and kronecker product @cite Li2010SimultaneousRA
};

/** @brief Finds an initial camera intrinsic matrix from 3D-2D point correspondences.

@param objectPoints Vector of vectors of the calibration pattern points in the calibration pattern
coordinate space. In the old interface all the per-view vectors are concatenated. See
#calibrateCamera for details.
@param imagePoints Vector of vectors of the projections of the calibration pattern points. In the
old interface all the per-view vectors are concatenated.
@param imageSize Image size in pixels used to initialize the principal point.
@param aspectRatio If it is zero or negative, both \f$f_x\f$ and \f$f_y\f$ are estimated independently.
Otherwise, \f$f_x = f_y \cdot \texttt{aspectRatio}\f$ .

The function estimates and returns an initial camera intrinsic matrix for the camera calibration process.
Currently, the function only supports planar calibration patterns, which are patterns where each
object point has z-coordinate =0.
 */
CV_EXPORTS_W Mat initCameraMatrix2D( InputArrayOfArrays objectPoints,
                                     InputArrayOfArrays imagePoints,
                                     Size imageSize, double aspectRatio = 1.0 );

/** @brief Finds the positions of internal corners of the chessboard.

@param image Source chessboard view. It must be an 8-bit grayscale or color image.
@param patternSize Number of inner corners per a chessboard row and column
( patternSize = cv::Size(points_per_row,points_per_colum) = cv::Size(columns,rows) ).
@param corners Output array of detected corners.
@param flags Various operation flags that can be zero or a combination of the following values:
-   @ref CALIB_CB_ADAPTIVE_THRESH Use adaptive thresholding to convert the image to black
and white, rather than a fixed threshold level (computed from the average image brightness).
-   @ref CALIB_CB_NORMALIZE_IMAGE Normalize the image gamma with equalizeHist before
applying fixed or adaptive thresholding.
-   @ref CALIB_CB_FILTER_QUADS Use additional criteria (like contour area, perimeter,
square-like shape) to filter out false quads extracted at the contour retrieval stage.
-   @ref CALIB_CB_FAST_CHECK Run a fast check on the image that looks for chessboard corners,
and shortcut the call if none is found. This can drastically speed up the call in the
degenerate condition when no chessboard is observed.
-   @ref CALIB_CB_PLAIN All other flags are ignored. The input image is taken as is.
No image processing is done to improve to find the checkerboard. This has the effect of speeding up the
execution of the function but could lead to not recognizing the checkerboard if the image
is not previously binarized in the appropriate manner.

The function attempts to determine whether the input image is a view of the chessboard pattern and
locate the internal chessboard corners. The function returns a non-zero value if all of the corners
are found and they are placed in a certain order (row by row, left to right in every row).
Otherwise, if the function fails to find all the corners or reorder them, it returns 0. For example,
a regular chessboard has 8 x 8 squares and 7 x 7 internal corners, that is, points where the black
squares touch each other. The detected coordinates are approximate, and to determine their positions
more accurately, the function calls #cornerSubPix. You also may use the function #cornerSubPix with
different parameters if returned coordinates are not accurate enough.

Sample usage of detecting and drawing chessboard corners: :
@code
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
@endcode
@note The function requires white space (like a square-thick border, the wider the better) around
the board to make the detection more robust in various environments. Otherwise, if there is no
border and the background is dark, the outer black squares cannot be segmented properly and so the
square grouping and ordering algorithm fails.

Use gen_pattern.py (@ref tutorial_camera_calibration_pattern) to create checkerboard.
 */
CV_EXPORTS_W bool findChessboardCorners( InputArray image, Size patternSize, OutputArray corners,
                                         int flags = CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE );

/*
   Checks whether the image contains chessboard of the specific size or not.
   If yes, nonzero value is returned.
*/
CV_EXPORTS_W bool checkChessboard(InputArray img, Size size);

/** @brief Finds the positions of internal corners of the chessboard using a sector based approach.

@param image Source chessboard view. It must be an 8-bit grayscale or color image.
@param patternSize Number of inner corners per a chessboard row and column
( patternSize = cv::Size(points_per_row,points_per_colum) = cv::Size(columns,rows) ).
@param corners Output array of detected corners.
@param flags Various operation flags that can be zero or a combination of the following values:
-   @ref CALIB_CB_NORMALIZE_IMAGE Normalize the image gamma with equalizeHist before detection.
-   @ref CALIB_CB_EXHAUSTIVE Run an exhaustive search to improve detection rate.
-   @ref CALIB_CB_ACCURACY Up sample input image to improve sub-pixel accuracy due to aliasing effects.
-   @ref CALIB_CB_LARGER The detected pattern is allowed to be larger than patternSize (see description).
-   @ref CALIB_CB_MARKER The detected pattern must have a marker (see description).
This should be used if an accurate camera calibration is required.
@param meta Optional output arrray of detected corners (CV_8UC1 and size = cv::Size(columns,rows)).
Each entry stands for one corner of the pattern and can have one of the following values:
-   0 = no meta data attached
-   1 = left-top corner of a black cell
-   2 = left-top corner of a white cell
-   3 = left-top corner of a black cell with a white marker dot
-   4 = left-top corner of a white cell with a black marker dot (pattern origin in case of markers otherwise first corner)

The function is analog to #findChessboardCorners but uses a localized radon
transformation approximated by box filters being more robust to all sort of
noise, faster on larger images and is able to directly return the sub-pixel
position of the internal chessboard corners. The Method is based on the paper
@cite duda2018 "Accurate Detection and Localization of Checkerboard Corners for
Calibration" demonstrating that the returned sub-pixel positions are more
accurate than the one returned by cornerSubPix allowing a precise camera
calibration for demanding applications.

In the case, the flags @ref CALIB_CB_LARGER or @ref CALIB_CB_MARKER are given,
the result can be recovered from the optional meta array. Both flags are
helpful to use calibration patterns exceeding the field of view of the camera.
These oversized patterns allow more accurate calibrations as corners can be
utilized, which are as close as possible to the image borders.  For a
consistent coordinate system across all images, the optional marker (see image
below) can be used to move the origin of the board to the location where the
black circle is located.

@note The function requires a white boarder with roughly the same width as one
of the checkerboard fields around the whole board to improve the detection in
various environments. In addition, because of the localized radon
transformation it is beneficial to use round corners for the field corners
which are located on the outside of the board. The following figure illustrates
a sample checkerboard optimized for the detection. However, any other checkerboard
can be used as well.

Use gen_pattern.py (@ref tutorial_camera_calibration_pattern) to create checkerboard.
![Checkerboard](pics/checkerboard_radon.png)
 */
CV_EXPORTS_AS(findChessboardCornersSBWithMeta)
bool findChessboardCornersSB(InputArray image,Size patternSize, OutputArray corners,
                             int flags,OutputArray meta);
/** @overload */
CV_EXPORTS_W inline
bool findChessboardCornersSB(InputArray image, Size patternSize, OutputArray corners,
                             int flags = 0)
{
    return findChessboardCornersSB(image, patternSize, corners, flags, noArray());
}

/** @brief Estimates the sharpness of a detected chessboard.

Image sharpness, as well as brightness, are a critical parameter for accuracte
camera calibration. For accessing these parameters for filtering out
problematic calibraiton images, this method calculates edge profiles by traveling from
black to white chessboard cell centers. Based on this, the number of pixels is
calculated required to transit from black to white. This width of the
transition area is a good indication of how sharp the chessboard is imaged
and should be below ~3.0 pixels.

@param image Gray image used to find chessboard corners
@param patternSize Size of a found chessboard pattern
@param corners Corners found by #findChessboardCornersSB
@param rise_distance Rise distance 0.8 means 10% ... 90% of the final signal strength
@param vertical By default edge responses for horizontal lines are calculated
@param sharpness Optional output array with a sharpness value for calculated edge responses (see description)

The optional sharpness array is of type CV_32FC1 and has for each calculated
profile one row with the following five entries:
* 0 = x coordinate of the underlying edge in the image
* 1 = y coordinate of the underlying edge in the image
* 2 = width of the transition area (sharpness)
* 3 = signal strength in the black cell (min brightness)
* 4 = signal strength in the white cell (max brightness)

@return Scalar(average sharpness, average min brightness, average max brightness,0)
*/
CV_EXPORTS_W Scalar estimateChessboardSharpness(InputArray image, Size patternSize, InputArray corners,
                                                float rise_distance=0.8F,bool vertical=false,
                                                OutputArray sharpness=noArray());


//! finds subpixel-accurate positions of the chessboard corners
CV_EXPORTS_W bool find4QuadCornerSubpix( InputArray img, InputOutputArray corners, Size region_size );

/** @brief Renders the detected chessboard corners.

@param image Destination image. It must be an 8-bit color image.
@param patternSize Number of inner corners per a chessboard row and column
(patternSize = cv::Size(points_per_row,points_per_column)).
@param corners Array of detected corners, the output of #findChessboardCorners.
@param patternWasFound Parameter indicating whether the complete board was found or not. The
return value of #findChessboardCorners should be passed here.

The function draws individual chessboard corners detected either as red circles if the board was not
found, or as colored corners connected with lines if the board was found.
 */
CV_EXPORTS_W void drawChessboardCorners( InputOutputArray image, Size patternSize,
                                         InputArray corners, bool patternWasFound );

struct CV_EXPORTS_W_SIMPLE CirclesGridFinderParameters
{
    CV_WRAP CirclesGridFinderParameters();
    CV_PROP_RW cv::Size2f densityNeighborhoodSize;
    CV_PROP_RW float minDensity;
    CV_PROP_RW int kmeansAttempts;
    CV_PROP_RW int minDistanceToAddKeypoint;
    CV_PROP_RW int keypointScale;
    CV_PROP_RW float minGraphConfidence;
    CV_PROP_RW float vertexGain;
    CV_PROP_RW float vertexPenalty;
    CV_PROP_RW float existingVertexGain;
    CV_PROP_RW float edgeGain;
    CV_PROP_RW float edgePenalty;
    CV_PROP_RW float convexHullFactor;
    CV_PROP_RW float minRNGEdgeSwitchDist;

    enum GridType
    {
      SYMMETRIC_GRID, ASYMMETRIC_GRID
    };
    GridType gridType;

    CV_PROP_RW float squareSize; //!< Distance between two adjacent points. Used by CALIB_CB_CLUSTERING.
    CV_PROP_RW float maxRectifiedDistance; //!< Max deviation from prediction. Used by CALIB_CB_CLUSTERING.
};

#ifndef DISABLE_OPENCV_3_COMPATIBILITY
typedef CirclesGridFinderParameters CirclesGridFinderParameters2;
#endif

/** @brief Finds centers in the grid of circles.

@param image grid view of input circles; it must be an 8-bit grayscale or color image.
@param patternSize number of circles per row and column
( patternSize = Size(points_per_row, points_per_colum) ).
@param centers output array of detected centers.
@param flags various operation flags that can be one of the following values:
-   @ref CALIB_CB_SYMMETRIC_GRID uses symmetric pattern of circles.
-   @ref CALIB_CB_ASYMMETRIC_GRID uses asymmetric pattern of circles.
-   @ref CALIB_CB_CLUSTERING uses a special algorithm for grid detection. It is more robust to
perspective distortions but much more sensitive to background clutter.
@param blobDetector feature detector that finds blobs like dark circles on light background.
                    If `blobDetector` is NULL then `image` represents Point2f array of candidates.
@param parameters struct for finding circles in a grid pattern.

The function attempts to determine whether the input image contains a grid of circles. If it is, the
function locates centers of the circles. The function returns a non-zero value if all of the centers
have been found and they have been placed in a certain order (row by row, left to right in every
row). Otherwise, if the function fails to find all the corners or reorder them, it returns 0.

Sample usage of detecting and drawing the centers of circles: :
@code
    Size patternsize(7,7); //number of centers
    Mat gray = ...; //source image
    vector<Point2f> centers; //this will be filled by the detected centers

    bool patternfound = findCirclesGrid(gray, patternsize, centers);

    drawChessboardCorners(img, patternsize, Mat(centers), patternfound);
@endcode
@note The function requires white space (like a square-thick border, the wider the better) around
the board to make the detection more robust in various environments.
 */
CV_EXPORTS_W bool findCirclesGrid( InputArray image, Size patternSize,
                                   OutputArray centers, int flags,
                                   const Ptr<FeatureDetector> &blobDetector,
                                   const CirclesGridFinderParameters& parameters);

/** @overload */
CV_EXPORTS_W bool findCirclesGrid( InputArray image, Size patternSize,
                                   OutputArray centers, int flags = CALIB_CB_SYMMETRIC_GRID,
                                   const Ptr<FeatureDetector> &blobDetector = SimpleBlobDetector::create());

/** @brief Finds the camera intrinsic and extrinsic parameters from several views of a calibration
pattern.

@param objectPoints In the new interface it is a vector of vectors of calibration pattern points in
the calibration pattern coordinate space (e.g. std::vector<std::vector<cv::Vec3f>>). The outer
vector contains as many elements as the number of pattern views. If the same calibration pattern
is shown in each view and it is fully visible, all the vectors will be the same. Although, it is
possible to use partially occluded patterns or even different patterns in different views. Then,
the vectors will be different. Although the points are 3D, they all lie in the calibration pattern's
XY coordinate plane (thus 0 in the Z-coordinate), if the used calibration pattern is a planar rig.
In the old interface all the vectors of object points from different views are concatenated
together.
@param imagePoints In the new interface it is a vector of vectors of the projections of calibration
pattern points (e.g. std::vector<std::vector<cv::Vec2f>>). imagePoints.size() and
objectPoints.size(), and imagePoints[i].size() and objectPoints[i].size() for each i, must be equal,
respectively. In the old interface all the vectors of object points from different views are
concatenated together.
@param imageSize Size of the image used only to initialize the camera intrinsic matrix.
@param cameraMatrix Input/output 3x3 floating-point camera intrinsic matrix
\f$\cameramatrix{A}\f$ . If @ref CALIB_USE_INTRINSIC_GUESS
and/or @ref CALIB_FIX_ASPECT_RATIO, @ref CALIB_FIX_PRINCIPAL_POINT or @ref CALIB_FIX_FOCAL_LENGTH
are specified, some or all of fx, fy, cx, cy must be initialized before calling the function.
@param distCoeffs Input/output vector of distortion coefficients
\f$\distcoeffs\f$.
@param rvecs Output vector of rotation vectors (@ref Rodrigues ) estimated for each pattern view
(e.g. std::vector<cv::Mat>>). That is, each i-th rotation vector together with the corresponding
i-th translation vector (see the next output parameter description) brings the calibration pattern
from the object coordinate space (in which object points are specified) to the camera coordinate
space. In more technical terms, the tuple of the i-th rotation and translation vector performs
a change of basis from object coordinate space to camera coordinate space. Due to its duality, this
tuple is equivalent to the position of the calibration pattern with respect to the camera coordinate
space.
@param tvecs Output vector of translation vectors estimated for each pattern view, see parameter
description above.
@param stdDeviationsIntrinsics Output vector of standard deviations estimated for intrinsic
parameters. Order of deviations values:
\f$(f_x, f_y, c_x, c_y, k_1, k_2, p_1, p_2, k_3, k_4, k_5, k_6 , s_1, s_2, s_3,
 s_4, \tau_x, \tau_y)\f$ If one of parameters is not estimated, it's deviation is equals to zero.
@param stdDeviationsExtrinsics Output vector of standard deviations estimated for extrinsic
parameters. Order of deviations values: \f$(R_0, T_0, \dotsc , R_{M - 1}, T_{M - 1})\f$ where M is
the number of pattern views. \f$R_i, T_i\f$ are concatenated 1x3 vectors.
 @param perViewErrors Output vector of the RMS re-projection error estimated for each pattern view.
@param flags Different flags that may be zero or a combination of the following values:
-   @ref CALIB_USE_INTRINSIC_GUESS cameraMatrix contains valid initial values of
fx, fy, cx, cy that are optimized further. Otherwise, (cx, cy) is initially set to the image
center ( imageSize is used), and focal distances are computed in a least-squares fashion.
Note, that if intrinsic parameters are known, there is no need to use this function just to
estimate extrinsic parameters. Use @ref solvePnP instead.
-   @ref CALIB_FIX_PRINCIPAL_POINT The principal point is not changed during the global
optimization. It stays at the center or at a different location specified when
 @ref CALIB_USE_INTRINSIC_GUESS is set too.
-   @ref CALIB_FIX_ASPECT_RATIO The functions consider only fy as a free parameter. The
ratio fx/fy stays the same as in the input cameraMatrix . When
 @ref CALIB_USE_INTRINSIC_GUESS is not set, the actual input values of fx and fy are
ignored, only their ratio is computed and used further.
-   @ref CALIB_ZERO_TANGENT_DIST Tangential distortion coefficients \f$(p_1, p_2)\f$ are set
to zeros and stay zero.
-   @ref CALIB_FIX_FOCAL_LENGTH The focal length is not changed during the global optimization if
 @ref CALIB_USE_INTRINSIC_GUESS is set.
-   @ref CALIB_FIX_K1,..., @ref CALIB_FIX_K6 The corresponding radial distortion
coefficient is not changed during the optimization. If @ref CALIB_USE_INTRINSIC_GUESS is
set, the coefficient from the supplied distCoeffs matrix is used. Otherwise, it is set to 0.
-   @ref CALIB_RATIONAL_MODEL Coefficients k4, k5, and k6 are enabled. To provide the
backward compatibility, this extra flag should be explicitly specified to make the
calibration function use the rational model and return 8 coefficients or more.
-   @ref CALIB_THIN_PRISM_MODEL Coefficients s1, s2, s3 and s4 are enabled. To provide the
backward compatibility, this extra flag should be explicitly specified to make the
calibration function use the thin prism model and return 12 coefficients or more.
-   @ref CALIB_FIX_S1_S2_S3_S4 The thin prism distortion coefficients are not changed during
the optimization. If @ref CALIB_USE_INTRINSIC_GUESS is set, the coefficient from the
supplied distCoeffs matrix is used. Otherwise, it is set to 0.
-   @ref CALIB_TILTED_MODEL Coefficients tauX and tauY are enabled. To provide the
backward compatibility, this extra flag should be explicitly specified to make the
calibration function use the tilted sensor model and return 14 coefficients.
-   @ref CALIB_FIX_TAUX_TAUY The coefficients of the tilted sensor model are not changed during
the optimization. If @ref CALIB_USE_INTRINSIC_GUESS is set, the coefficient from the
supplied distCoeffs matrix is used. Otherwise, it is set to 0.
@param criteria Termination criteria for the iterative optimization algorithm.

@return the overall RMS re-projection error.

The function estimates the intrinsic camera parameters and extrinsic parameters for each of the
views. The algorithm is based on @cite Zhang2000 and @cite BouguetMCT . The coordinates of 3D object
points and their corresponding 2D projections in each view must be specified. That may be achieved
by using an object with known geometry and easily detectable feature points. Such an object is
called a calibration rig or calibration pattern, and OpenCV has built-in support for a chessboard as
a calibration rig (see @ref findChessboardCorners). Currently, initialization of intrinsic
parameters (when @ref CALIB_USE_INTRINSIC_GUESS is not set) is only implemented for planar calibration
patterns (where Z-coordinates of the object points must be all zeros). 3D calibration rigs can also
be used as long as initial cameraMatrix is provided.

The algorithm performs the following steps:

-   Compute the initial intrinsic parameters (the option only available for planar calibration
    patterns) or read them from the input parameters. The distortion coefficients are all set to
    zeros initially unless some of CALIB_FIX_K? are specified.

-   Estimate the initial camera pose as if the intrinsic parameters have been already known. This is
    done using @ref solvePnP .

-   Run the global Levenberg-Marquardt optimization algorithm to minimize the reprojection error,
    that is, the total sum of squared distances between the observed feature points imagePoints and
    the projected (using the current estimates for camera parameters and the poses) object points
    objectPoints. See @ref projectPoints for details.

@note
    If you use a non-square (i.e. non-N-by-N) grid and @ref findChessboardCorners for calibration,
    and @ref calibrateCamera returns bad values (zero distortion coefficients, \f$c_x\f$ and
    \f$c_y\f$ very far from the image center, and/or large differences between \f$f_x\f$ and
    \f$f_y\f$ (ratios of 10:1 or more)), then you are probably using patternSize=cvSize(rows,cols)
    instead of using patternSize=cvSize(cols,rows) in @ref findChessboardCorners.

@note
    The function may throw exceptions, if unsupported combination of parameters is provided or
    the system is underconstrained.

@sa
   calibrateCameraRO, findChessboardCorners, solvePnP, initCameraMatrix2D, stereoCalibrate,
   undistort
 */
CV_EXPORTS_AS(calibrateCameraExtended) double calibrateCamera( InputArrayOfArrays objectPoints,
                                     InputArrayOfArrays imagePoints, Size imageSize,
                                     InputOutputArray cameraMatrix, InputOutputArray distCoeffs,
                                     OutputArrayOfArrays rvecs, OutputArrayOfArrays tvecs,
                                     OutputArray stdDeviationsIntrinsics,
                                     OutputArray stdDeviationsExtrinsics,
                                     OutputArray perViewErrors,
                                     int flags = 0, TermCriteria criteria = TermCriteria(
                                        TermCriteria::COUNT + TermCriteria::EPS, 100, DBL_EPSILON) );

/** @overload */
CV_EXPORTS_W double calibrateCamera( InputArrayOfArrays objectPoints,
                                     InputArrayOfArrays imagePoints, Size imageSize,
                                     InputOutputArray cameraMatrix, InputOutputArray distCoeffs,
                                     OutputArrayOfArrays rvecs, OutputArrayOfArrays tvecs,
                                     int flags = 0, TermCriteria criteria = TermCriteria(
                                        TermCriteria::COUNT + TermCriteria::EPS, 100, DBL_EPSILON) );

/** @brief Finds the camera intrinsic and extrinsic parameters from several views of a calibration pattern.

This function is an extension of #calibrateCamera with the method of releasing object which was
proposed in @cite strobl2011iccv. In many common cases with inaccurate, unmeasured, roughly planar
targets (calibration plates), this method can dramatically improve the precision of the estimated
camera parameters. Both the object-releasing method and standard method are supported by this
function. Use the parameter **iFixedPoint** for method selection. In the internal implementation,
#calibrateCamera is a wrapper for this function.

@param objectPoints Vector of vectors of calibration pattern points in the calibration pattern
coordinate space. See #calibrateCamera for details. If the method of releasing object to be used,
the identical calibration board must be used in each view and it must be fully visible, and all
objectPoints[i] must be the same and all points should be roughly close to a plane. **The calibration
target has to be rigid, or at least static if the camera (rather than the calibration target) is
shifted for grabbing images.**
@param imagePoints Vector of vectors of the projections of calibration pattern points. See
#calibrateCamera for details.
@param imageSize Size of the image used only to initialize the intrinsic camera matrix.
@param iFixedPoint The index of the 3D object point in objectPoints[0] to be fixed. It also acts as
a switch for calibration method selection. If object-releasing method to be used, pass in the
parameter in the range of [1, objectPoints[0].size()-2], otherwise a value out of this range will
make standard calibration method selected. Usually the top-right corner point of the calibration
board grid is recommended to be fixed when object-releasing method being utilized. According to
\cite strobl2011iccv, two other points are also fixed. In this implementation, objectPoints[0].front
and objectPoints[0].back.z are used. With object-releasing method, accurate rvecs, tvecs and
newObjPoints are only possible if coordinates of these three fixed points are accurate enough.
@param cameraMatrix Output 3x3 floating-point camera matrix. See #calibrateCamera for details.
@param distCoeffs Output vector of distortion coefficients. See #calibrateCamera for details.
@param rvecs Output vector of rotation vectors estimated for each pattern view. See #calibrateCamera
for details.
@param tvecs Output vector of translation vectors estimated for each pattern view.
@param newObjPoints The updated output vector of calibration pattern points. The coordinates might
be scaled based on three fixed points. The returned coordinates are accurate only if the above
mentioned three fixed points are accurate. If not needed, noArray() can be passed in. This parameter
is ignored with standard calibration method.
@param stdDeviationsIntrinsics Output vector of standard deviations estimated for intrinsic parameters.
See #calibrateCamera for details.
@param stdDeviationsExtrinsics Output vector of standard deviations estimated for extrinsic parameters.
See #calibrateCamera for details.
@param stdDeviationsObjPoints Output vector of standard deviations estimated for refined coordinates
of calibration pattern points. It has the same size and order as objectPoints[0] vector. This
parameter is ignored with standard calibration method.
 @param perViewErrors Output vector of the RMS re-projection error estimated for each pattern view.
@param flags Different flags that may be zero or a combination of some predefined values. See
#calibrateCamera for details. If the method of releasing object is used, the calibration time may
be much longer. CALIB_USE_QR or CALIB_USE_LU could be used for faster calibration with potentially
less precise and less stable in some rare cases.
@param criteria Termination criteria for the iterative optimization algorithm.

@return the overall RMS re-projection error.

The function estimates the intrinsic camera parameters and extrinsic parameters for each of the
views. The algorithm is based on @cite Zhang2000, @cite BouguetMCT and @cite strobl2011iccv. See
#calibrateCamera for other detailed explanations.
@sa
   calibrateCamera, findChessboardCorners, solvePnP, initCameraMatrix2D, stereoCalibrate, undistort
 */
CV_EXPORTS_AS(calibrateCameraROExtended) double calibrateCameraRO( InputArrayOfArrays objectPoints,
                                     InputArrayOfArrays imagePoints, Size imageSize, int iFixedPoint,
                                     InputOutputArray cameraMatrix, InputOutputArray distCoeffs,
                                     OutputArrayOfArrays rvecs, OutputArrayOfArrays tvecs,
                                     OutputArray newObjPoints,
                                     OutputArray stdDeviationsIntrinsics,
                                     OutputArray stdDeviationsExtrinsics,
                                     OutputArray stdDeviationsObjPoints,
                                     OutputArray perViewErrors,
                                     int flags = 0, TermCriteria criteria = TermCriteria(
                                        TermCriteria::COUNT + TermCriteria::EPS, 100, DBL_EPSILON) );

/** @overload */
CV_EXPORTS_W double calibrateCameraRO( InputArrayOfArrays objectPoints,
                                     InputArrayOfArrays imagePoints, Size imageSize, int iFixedPoint,
                                     InputOutputArray cameraMatrix, InputOutputArray distCoeffs,
                                     OutputArrayOfArrays rvecs, OutputArrayOfArrays tvecs,
                                     OutputArray newObjPoints,
                                     int flags = 0, TermCriteria criteria = TermCriteria(
                                        TermCriteria::COUNT + TermCriteria::EPS, 100, DBL_EPSILON) );

/** @brief Computes useful camera characteristics from the camera intrinsic matrix.

@param cameraMatrix Input camera intrinsic matrix that can be estimated by #calibrateCamera or
#stereoCalibrate .
@param imageSize Input image size in pixels.
@param apertureWidth Physical width in mm of the sensor.
@param apertureHeight Physical height in mm of the sensor.
@param fovx Output field of view in degrees along the horizontal sensor axis.
@param fovy Output field of view in degrees along the vertical sensor axis.
@param focalLength Focal length of the lens in mm.
@param principalPoint Principal point in mm.
@param aspectRatio \f$f_y/f_x\f$

The function computes various useful camera characteristics from the previously estimated camera
matrix.

@note
   Do keep in mind that the unity measure 'mm' stands for whatever unit of measure one chooses for
    the chessboard pitch (it can thus be any value).
 */
CV_EXPORTS_W void calibrationMatrixValues( InputArray cameraMatrix, Size imageSize,
                                           double apertureWidth, double apertureHeight,
                                           CV_OUT double& fovx, CV_OUT double& fovy,
                                           CV_OUT double& focalLength, CV_OUT Point2d& principalPoint,
                                           CV_OUT double& aspectRatio );

/** @brief Calibrates a stereo camera set up. This function finds the intrinsic parameters
for each of the two cameras and the extrinsic parameters between the two cameras.

@param objectPoints Vector of vectors of the calibration pattern points. The same structure as
in @ref calibrateCamera. For each pattern view, both cameras need to see the same object
points. Therefore, objectPoints.size(), imagePoints1.size(), and imagePoints2.size() need to be
equal as well as objectPoints[i].size(), imagePoints1[i].size(), and imagePoints2[i].size() need to
be equal for each i.
@param imagePoints1 Vector of vectors of the projections of the calibration pattern points,
observed by the first camera. The same structure as in @ref calibrateCamera.
@param imagePoints2 Vector of vectors of the projections of the calibration pattern points,
observed by the second camera. The same structure as in @ref calibrateCamera.
@param cameraMatrix1 Input/output camera intrinsic matrix for the first camera, the same as in
@ref calibrateCamera. Furthermore, for the stereo case, additional flags may be used, see below.
@param distCoeffs1 Input/output vector of distortion coefficients, the same as in
@ref calibrateCamera.
@param cameraMatrix2 Input/output second camera intrinsic matrix for the second camera. See description for
cameraMatrix1.
@param distCoeffs2 Input/output lens distortion coefficients for the second camera. See
description for distCoeffs1.
@param imageSize Size of the image used only to initialize the camera intrinsic matrices.
@param R Output rotation matrix. Together with the translation vector T, this matrix brings
points given in the first camera's coordinate system to points in the second camera's
coordinate system. In more technical terms, the tuple of R and T performs a change of basis
from the first camera's coordinate system to the second camera's coordinate system. Due to its
duality, this tuple is equivalent to the position of the first camera with respect to the
second camera coordinate system.
@param T Output translation vector, see description above.
@param E Output essential matrix.
@param F Output fundamental matrix.
@param rvecs Output vector of rotation vectors ( @ref Rodrigues ) estimated for each pattern view in the
coordinate system of the first camera of the stereo pair (e.g. std::vector<cv::Mat>). More in detail, each
i-th rotation vector together with the corresponding i-th translation vector (see the next output parameter
description) brings the calibration pattern from the object coordinate space (in which object points are
specified) to the camera coordinate space of the first camera of the stereo pair. In more technical terms,
the tuple of the i-th rotation and translation vector performs a change of basis from object coordinate space
to camera coordinate space of the first camera of the stereo pair.
@param tvecs Output vector of translation vectors estimated for each pattern view, see parameter description
of previous output parameter ( rvecs ).
@param perViewErrors Output vector of the RMS re-projection error estimated for each pattern view.
@param flags Different flags that may be zero or a combination of the following values:
-   @ref CALIB_FIX_INTRINSIC Fix cameraMatrix? and distCoeffs? so that only R, T, E, and F
matrices are estimated.
-   @ref CALIB_USE_INTRINSIC_GUESS Optimize some or all of the intrinsic parameters
according to the specified flags. Initial values are provided by the user.
-   @ref CALIB_USE_EXTRINSIC_GUESS R and T contain valid initial values that are optimized further.
Otherwise R and T are initialized to the median value of the pattern views (each dimension separately).
-   @ref CALIB_FIX_PRINCIPAL_POINT Fix the principal points during the optimization.
-   @ref CALIB_FIX_FOCAL_LENGTH Fix \f$f^{(j)}_x\f$ and \f$f^{(j)}_y\f$ .
-   @ref CALIB_FIX_ASPECT_RATIO Optimize \f$f^{(j)}_y\f$ . Fix the ratio \f$f^{(j)}_x/f^{(j)}_y\f$
.
-   @ref CALIB_SAME_FOCAL_LENGTH Enforce \f$f^{(0)}_x=f^{(1)}_x\f$ and \f$f^{(0)}_y=f^{(1)}_y\f$ .
-   @ref CALIB_ZERO_TANGENT_DIST Set tangential distortion coefficients for each camera to
zeros and fix there.
-   @ref CALIB_FIX_K1,..., @ref CALIB_FIX_K6 Do not change the corresponding radial
distortion coefficient during the optimization. If @ref CALIB_USE_INTRINSIC_GUESS is set,
the coefficient from the supplied distCoeffs matrix is used. Otherwise, it is set to 0.
-   @ref CALIB_RATIONAL_MODEL Enable coefficients k4, k5, and k6. To provide the backward
compatibility, this extra flag should be explicitly specified to make the calibration
function use the rational model and return 8 coefficients. If the flag is not set, the
function computes and returns only 5 distortion coefficients.
-   @ref CALIB_THIN_PRISM_MODEL Coefficients s1, s2, s3 and s4 are enabled. To provide the
backward compatibility, this extra flag should be explicitly specified to make the
calibration function use the thin prism model and return 12 coefficients. If the flag is not
set, the function computes and returns only 5 distortion coefficients.
-   @ref CALIB_FIX_S1_S2_S3_S4 The thin prism distortion coefficients are not changed during
the optimization. If @ref CALIB_USE_INTRINSIC_GUESS is set, the coefficient from the
supplied distCoeffs matrix is used. Otherwise, it is set to 0.
-   @ref CALIB_TILTED_MODEL Coefficients tauX and tauY are enabled. To provide the
backward compatibility, this extra flag should be explicitly specified to make the
calibration function use the tilted sensor model and return 14 coefficients. If the flag is not
set, the function computes and returns only 5 distortion coefficients.
-   @ref CALIB_FIX_TAUX_TAUY The coefficients of the tilted sensor model are not changed during
the optimization. If @ref CALIB_USE_INTRINSIC_GUESS is set, the coefficient from the
supplied distCoeffs matrix is used. Otherwise, it is set to 0.
@param criteria Termination criteria for the iterative optimization algorithm.

The function estimates the transformation between two cameras making a stereo pair. If one computes
the poses of an object relative to the first camera and to the second camera,
( \f$R_1\f$,\f$T_1\f$ ) and (\f$R_2\f$,\f$T_2\f$), respectively, for a stereo camera where the
relative position and orientation between the two cameras are fixed, then those poses definitely
relate to each other. This means, if the relative position and orientation (\f$R\f$,\f$T\f$) of the
two cameras is known, it is possible to compute (\f$R_2\f$,\f$T_2\f$) when (\f$R_1\f$,\f$T_1\f$) is
given. This is what the described function does. It computes (\f$R\f$,\f$T\f$) such that:

\f[R_2=R R_1\f]
\f[T_2=R T_1 + T.\f]

Therefore, one can compute the coordinate representation of a 3D point for the second camera's
coordinate system when given the point's coordinate representation in the first camera's coordinate
system:

\f[\begin{bmatrix}
X_2 \\
Y_2 \\
Z_2 \\
1
\end{bmatrix} = \begin{bmatrix}
R & T \\
0 & 1
\end{bmatrix} \begin{bmatrix}
X_1 \\
Y_1 \\
Z_1 \\
1
\end{bmatrix}.\f]


Optionally, it computes the essential matrix E:

\f[E= \vecthreethree{0}{-T_2}{T_1}{T_2}{0}{-T_0}{-T_1}{T_0}{0} R\f]

where \f$T_i\f$ are components of the translation vector \f$T\f$ : \f$T=[T_0, T_1, T_2]^T\f$ .
And the function can also compute the fundamental matrix F:

\f[F = cameraMatrix2^{-T}\cdot E \cdot cameraMatrix1^{-1}\f]

Besides the stereo-related information, the function can also perform a full calibration of each of
the two cameras. However, due to the high dimensionality of the parameter space and noise in the
input data, the function can diverge from the correct solution. If the intrinsic parameters can be
estimated with high accuracy for each of the cameras individually (for example, using
#calibrateCamera ), you are recommended to do so and then pass @ref CALIB_FIX_INTRINSIC flag to the
function along with the computed intrinsic parameters. Otherwise, if all the parameters are
estimated at once, it makes sense to restrict some parameters, for example, pass
 @ref CALIB_SAME_FOCAL_LENGTH and @ref CALIB_ZERO_TANGENT_DIST flags, which is usually a
reasonable assumption.

Similarly to #calibrateCamera, the function minimizes the total re-projection error for all the
points in all the available views from both cameras. The function returns the final value of the
re-projection error.
 */
CV_EXPORTS_AS(stereoCalibrateExtended) double stereoCalibrate( InputArrayOfArrays objectPoints,
                                     InputArrayOfArrays imagePoints1, InputArrayOfArrays imagePoints2,
                                     InputOutputArray cameraMatrix1, InputOutputArray distCoeffs1,
                                     InputOutputArray cameraMatrix2, InputOutputArray distCoeffs2,
                                     Size imageSize, InputOutputArray R, InputOutputArray T, OutputArray E, OutputArray F,
                                     OutputArrayOfArrays rvecs, OutputArrayOfArrays tvecs,
                                     OutputArray perViewErrors, int flags = CALIB_FIX_INTRINSIC,
                                     TermCriteria criteria = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 100, 1e-6) );

/// @overload
CV_EXPORTS_W double stereoCalibrate( InputArrayOfArrays objectPoints,
                                     InputArrayOfArrays imagePoints1, InputArrayOfArrays imagePoints2,
                                     InputOutputArray cameraMatrix1, InputOutputArray distCoeffs1,
                                     InputOutputArray cameraMatrix2, InputOutputArray distCoeffs2,
                                     Size imageSize, OutputArray R,OutputArray T, OutputArray E, OutputArray F,
                                     int flags = CALIB_FIX_INTRINSIC,
                                     TermCriteria criteria = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 100, 1e-6) );

/// @overload
CV_EXPORTS_W double stereoCalibrate( InputArrayOfArrays objectPoints,
                                     InputArrayOfArrays imagePoints1, InputArrayOfArrays imagePoints2,
                                     InputOutputArray cameraMatrix1, InputOutputArray distCoeffs1,
                                     InputOutputArray cameraMatrix2, InputOutputArray distCoeffs2,
                                     Size imageSize, InputOutputArray R, InputOutputArray T, OutputArray E, OutputArray F,
                                     OutputArray perViewErrors, int flags = CALIB_FIX_INTRINSIC,
                                     TermCriteria criteria = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 1e-6) );

/** @brief Calibrates a camera pair set up. This function finds the extrinsic parameters between the two cameras.

@param objectPoints1 Vector of vectors of the calibration pattern points for camera 1.
A similar structure as objectPoints in @ref calibrateCamera and for each pattern view,
both cameras do not need to see the same object points. objectPoints1.size(), imagePoints1.size()
nees to be equal,as well as objectPoints1[i].size(), imagePoints1[i].size() need to be equal for each i.
@param objectPoints2 Vector of vectors of the calibration pattern points for camera 2.
A similar structure as objectPoints1. objectPoints2.size(), and imagePoints2.size() nees to be equal,
as well as objectPoints2[i].size(), imagePoints2[i].size() need to be equal for each i.
However, objectPoints1[i].size() and objectPoints2[i].size() are not required to be equal.
@param imagePoints1 Vector of vectors of the projections of the calibration pattern points,
observed by the first camera. The same structure as in @ref calibrateCamera.
@param imagePoints2 Vector of vectors of the projections of the calibration pattern points,
observed by the second camera. The same structure as in @ref calibrateCamera.
@param cameraMatrix1 Input/output camera intrinsic matrix for the first camera, the same as in
@ref calibrateCamera. Furthermore, for the stereo case, additional flags may be used, see below.
@param distCoeffs1 Input/output vector of distortion coefficients, the same as in
@ref calibrateCamera.
@param cameraModel1 Flag reflecting the type of model for camera 1 (pinhole / fisheye):
- @ref CALIB_MODEL_PINHOLE pinhole camera model
- @ref CALIB_MODEL_FISHEYE fisheye camera model
@param cameraMatrix2 Input/output second camera intrinsic matrix for the second camera.
See description for cameraMatrix1.
@param distCoeffs2 Input/output lens distortion coefficients for the second camera. See
description for distCoeffs1.
@param cameraModel2 Flag reflecting the type of model for camera 2 (pinhole / fisheye).
See description for cameraModel1.
@param R Output rotation matrix. Together with the translation vector T, this matrix brings
points given in the first camera's coordinate system to points in the second camera's
coordinate system. In more technical terms, the tuple of R and T performs a change of basis
from the first camera's coordinate system to the second camera's coordinate system. Due to its
duality, this tuple is equivalent to the position of the first camera with respect to the
second camera coordinate system.
@param T Output translation vector, see description above.
@param E Output essential matrix.
@param F Output fundamental matrix.
@param rvecs Output vector of rotation vectors ( @ref Rodrigues ) estimated for each pattern view in the
coordinate system of the first camera of the stereo pair (e.g. std::vector<cv::Mat>). More in detail, each
i-th rotation vector together with the corresponding i-th translation vector (see the next output parameter
description) brings the calibration pattern from the object coordinate space (in which object points are
specified) to the camera coordinate space of the first camera of the stereo pair. In more technical terms,
the tuple of the i-th rotation and translation vector performs a change of basis from object coordinate space
to the camera coordinate space of the first camera of the stereo pair.
@param tvecs Output vector of translation vectors estimated for each pattern view, see parameter description
of previous output parameter ( rvecs ).
@param perViewErrors Output vector of the RMS re-projection error estimated for each pattern view.
@param flags Different flags that may be zero or a combination of the following values:
-   @ref CALIB_USE_EXTRINSIC_GUESS R and T contain valid initial values that are optimized further.
@param criteria Termination criteria for the iterative optimization algorithm.

The function estimates the transformation between two cameras similar to stereo pair calibration.
The principle follows closely to @ref stereoCalibrate. To understand the problem of estimating the
relative pose between a camera pair, please refer to the description there. The difference for
this function is that, camera intrinsics are not optimized and two cameras are not required
to have overlapping fields of view as long as they are observing the same calibration target
and the absolute positions of each object point are known.
![](pics/register_pair.png)
The above illustration shows an example where such a case may become relevant.
Additionally, it supports a camera pair with the mixed model (pinhole / fisheye).
Similarly to #calibrateCamera, the function minimizes the total re-projection error for all the
points in all the available views from both cameras.
@return the final value of the re-projection error.

@sa calibrateCamera, stereoCalibrate
 */
CV_EXPORTS_AS(registerCamerasExtended) double registerCameras( InputArrayOfArrays objectPoints1,
                                     InputArrayOfArrays objectPoints2,
                                     InputArrayOfArrays imagePoints1, InputArrayOfArrays imagePoints2,
                                     InputArray cameraMatrix1, InputArray distCoeffs1,
                                     CameraModel cameraModel1,
                                     InputArray cameraMatrix2, InputArray distCoeffs2,
                                     CameraModel cameraModel2,
                                     InputOutputArray R, InputOutputArray T, OutputArray E, OutputArray F,
                                     OutputArrayOfArrays rvecs, OutputArrayOfArrays tvecs,
                                     OutputArray perViewErrors,
                                     int flags = 0,
                                     TermCriteria criteria = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 100, 1e-6) );

/// @overload
CV_EXPORTS_W double registerCameras( InputArrayOfArrays objectPoints1,
                                     InputArrayOfArrays objectPoints2,
                                     InputArrayOfArrays imagePoints1, InputArrayOfArrays imagePoints2,
                                     InputArray cameraMatrix1, InputArray distCoeffs1,
                                     CameraModel cameraModel1,
                                     InputArray cameraMatrix2, InputArray distCoeffs2,
                                     CameraModel cameraModel2,
                                     InputOutputArray R, InputOutputArray T, OutputArray E, OutputArray F,
                                     OutputArray perViewErrors,
                                     int flags = 0,
                                     TermCriteria criteria = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 100, 1e-6) );

/** @brief Estimates intrinsics and extrinsics (camera pose) for multi-camera system a.k.a multiview calibration.

@param[in] objPoints Calibration pattern object points. Expected shape: NUM_FRAMES x NUM_POINTS x 3. Supported data type: CV_32F.
@param[in] imagePoints Detected pattern points on camera images. Expected shape: NUM_CAMERAS x NUM_FRAMES x NUM_POINTS x 2.
This function supports partial observation of the calibration pattern.
To enable this, set the unobserved image points to be invalid points (eg. (-1., -1.)).
@param[in] imageSize Images resolution array for each camera.
@param[in] detectionMask Pattern detection mask. Each value defines if i-camera observes the calibration pattern in j-th frame.
Expected size: NUM_CAMERAS x NUM_FRAMES. Expected type: CV_8U.
@param[in] models indicates camera models for each camera: cv::CALIB_MODEL_PINHOLE or cv::CALIB_MODEL_PINHOLE.
Current implementation does not support mix of different camera models. Expected type: CV_8U.
@param[in] flagsForIntrinsics Flags used for each camera intrinsics calibration.
Use per-camera call and the `useIntrinsicsGuess` flag to get custom intrinsics calibration for each camera.
@param[in] flags Common multiview calibration flags. cv::CALIB_USE_INTRINSIC_GUESS and cv::CALIB_USE_EXTRINSIC_GUESS are supported.
See @ref CALIB_USE_INTRINSIC_GUESS and other `CALIB_` constants. Expected shape: NUM_CAMERAS x 1. Supported data type: CV_32S.
@param[out] Rs Rotation vectors relative to camera 0, where Rs[0] = 0. Output size: NUM_CAMERAS x 3 x 3.
@param[out] Ts Estimated translation vectors relative to camera 0, where Ts[0] = 0. Output size: NUM_CAMERAS x 3 x 1.
@param[out] rvecs0 Estimated rotation vectors for camera 0. Output size: NUM_FRAMES x 3 x 1 (may contain null Mat, if the frame is not valid). See @ref Rodrigues.
@param[out] tvecs0 Translation vectors for camera 0. Output size: NUM_FRAMES x 3 x 1. (may contain null Mat, if the frame is not valid).
@param[out] Ks Estimated floating-point camera intrinsic matrix. Output size: NUM_CAMERAS x 3 x 3.
@param[out] distortions Distortion coefficients. Output size: NUM_CAMERAS x NUM_PARAMS.
@param[out] perFrameErrors RMSE value for each visible frame, (-1 for non-visible). Output size: NUM_CAMERAS x NUM_FRAMES.
@param[out] initializationPairs Pairs with camera indices that were used for initial pairwise stereo calibration.

Output size: (NUM_CAMERAS-1) x 2.

@ref tutorial_multiview_camera_calibration provides a detailed tutorial of using this function. Please refer to it for more information.

Multiview calibration usually requires several cameras to observe the same calibration pattern simultaneously.
The fundamental assumption is that relative camera poses are fixed,
and then for each frame, only the absolute camera pose for a single camera is needed to fix the camera pose for the multiple cameras

![multiview calibration](pics/multiview_calib.png)
The above illustration shows an example setting for multiview camera calibration.

For each frame, suppose the absolute camera pose for camera \f$i\f$ is \f$R_i, t_i\f$,
and the relative camera pose between camera \f$i\f$ and camera \f$j\f$ is \f$R_{ij}, t_{ij}\f$.
Suppose \f$R_1, t_1\f$, and \f$R_{1i}\f$ for any \f$i\not=1\f$ are known, then its pose can be calculated by
\f[ R_i = R_{1i} R_1\f]
\f[ t_i = R_{1i} t_1 + t_{1i}\f]

Since the relative pose between two cameras can be calculated by
\f[ R_{ij} = R_j R_i^\top \f]
\f[ t_{ij} = -R_{ij} t_i + R_j \f]

This implies that any other relative pose of the form \f$R_{ij}, i\not=1\f$ is redundant.
Given this, the total number of poses to determine is (NUM_CAMERAS-1) and NUM_FRAMES.
This serves as the foundation of this function.

Similarly to #calibrateCamera, the function minimizes the total re-projection error for all the
points in all the available views from all cameras.

@return Overall RMS re-projection error over detectionMask.

@sa findChessboardCorners, findCirclesGrid, calibrateCamera, fisheye::calibrate, registerCameras
*/

CV_EXPORTS_AS(calibrateMultiviewExtended) double calibrateMultiview (
        InputArrayOfArrays objPoints, const std::vector<std::vector<Mat>> &imagePoints,
        const std::vector<cv::Size>& imageSize, InputArray detectionMask, InputArray models,
        InputOutputArrayOfArrays Ks, InputOutputArrayOfArrays distortions,
        InputOutputArrayOfArrays Rs, InputOutputArrayOfArrays Ts,
        OutputArray initializationPairs, OutputArrayOfArrays rvecs0,
        OutputArrayOfArrays tvecs0, OutputArray perFrameErrors,
        InputArray flagsForIntrinsics=noArray(), int flags = 0);

/// @overload
CV_EXPORTS_W double calibrateMultiview (
        InputArrayOfArrays objPoints, const std::vector<std::vector<Mat>> &imagePoints,
        const std::vector<cv::Size>& imageSize, InputArray detectionMask, InputArray models,
        InputOutputArrayOfArrays Ks, InputOutputArrayOfArrays distortions,
        InputOutputArrayOfArrays Rs, InputOutputArrayOfArrays Ts,
        InputArray flagsForIntrinsics=noArray(), int flags = 0);


/** @brief Computes Hand-Eye calibration: \f$_{}^{g}\textrm{T}_c\f$

@param[in] R_gripper2base Rotation part extracted from the homogeneous matrix that transforms a point
expressed in the gripper frame to the robot base frame (\f$_{}^{b}\textrm{T}_g\f$).
This is a vector (`vector<Mat>`) that contains the rotation, `(3x3)` rotation matrices or `(3x1)` rotation vectors,
for all the transformations from gripper frame to robot base frame.
@param[in] t_gripper2base Translation part extracted from the homogeneous matrix that transforms a point
expressed in the gripper frame to the robot base frame (\f$_{}^{b}\textrm{T}_g\f$).
This is a vector (`vector<Mat>`) that contains the `(3x1)` translation vectors for all the transformations
from gripper frame to robot base frame.
@param[in] R_target2cam Rotation part extracted from the homogeneous matrix that transforms a point
expressed in the target frame to the camera frame (\f$_{}^{c}\textrm{T}_t\f$).
This is a vector (`vector<Mat>`) that contains the rotation, `(3x3)` rotation matrices or `(3x1)` rotation vectors,
for all the transformations from calibration target frame to camera frame.
@param[in] t_target2cam Rotation part extracted from the homogeneous matrix that transforms a point
expressed in the target frame to the camera frame (\f$_{}^{c}\textrm{T}_t\f$).
This is a vector (`vector<Mat>`) that contains the `(3x1)` translation vectors for all the transformations
from calibration target frame to camera frame.
@param[out] R_cam2gripper Estimated `(3x3)` rotation part extracted from the homogeneous matrix that transforms a point
expressed in the camera frame to the gripper frame (\f$_{}^{g}\textrm{T}_c\f$).
@param[out] t_cam2gripper Estimated `(3x1)` translation part extracted from the homogeneous matrix that transforms a point
expressed in the camera frame to the gripper frame (\f$_{}^{g}\textrm{T}_c\f$).
@param[in] method One of the implemented Hand-Eye calibration method, see cv::HandEyeCalibrationMethod

The function performs the Hand-Eye calibration using various methods. One approach consists in estimating the
rotation then the translation (separable solutions) and the following methods are implemented:
  - R. Tsai, R. Lenz A New Technique for Fully Autonomous and Efficient 3D Robotics Hand/EyeCalibration \cite Tsai89
  - F. Park, B. Martin Robot Sensor Calibration: Solving AX = XB on the Euclidean Group \cite Park94
  - R. Horaud, F. Dornaika Hand-Eye Calibration \cite Horaud95

Another approach consists in estimating simultaneously the rotation and the translation (simultaneous solutions),
with the following implemented methods:
  - N. Andreff, R. Horaud, B. Espiau On-line Hand-Eye Calibration \cite Andreff99
  - K. Daniilidis Hand-Eye Calibration Using Dual Quaternions \cite Daniilidis98

The following picture describes the Hand-Eye calibration problem where the transformation between a camera ("eye")
mounted on a robot gripper ("hand") has to be estimated. This configuration is called eye-in-hand.

The eye-to-hand configuration consists in a static camera observing a calibration pattern mounted on the robot
end-effector. The transformation from the camera to the robot base frame can then be estimated by inputting
the suitable transformations to the function, see below.

![](pics/hand-eye_figure.png)

The calibration procedure is the following:
  - a static calibration pattern is used to estimate the transformation between the target frame
  and the camera frame
  - the robot gripper is moved in order to acquire several poses
  - for each pose, the homogeneous transformation between the gripper frame and the robot base frame is recorded using for
  instance the robot kinematics
\f[
    \begin{bmatrix}
    X_b\\
    Y_b\\
    Z_b\\
    1
    \end{bmatrix}
    =
    \begin{bmatrix}
    _{}^{b}\textrm{R}_g & _{}^{b}\textrm{t}_g \\
    0_{1 \times 3} & 1
    \end{bmatrix}
    \begin{bmatrix}
    X_g\\
    Y_g\\
    Z_g\\
    1
    \end{bmatrix}
\f]
  - for each pose, the homogeneous transformation between the calibration target frame and the camera frame is recorded using
  for instance a pose estimation method (PnP) from 2D-3D point correspondences
\f[
    \begin{bmatrix}
    X_c\\
    Y_c\\
    Z_c\\
    1
    \end{bmatrix}
    =
    \begin{bmatrix}
    _{}^{c}\textrm{R}_t & _{}^{c}\textrm{t}_t \\
    0_{1 \times 3} & 1
    \end{bmatrix}
    \begin{bmatrix}
    X_t\\
    Y_t\\
    Z_t\\
    1
    \end{bmatrix}
\f]

The Hand-Eye calibration procedure returns the following homogeneous transformation
\f[
    \begin{bmatrix}
    X_g\\
    Y_g\\
    Z_g\\
    1
    \end{bmatrix}
    =
    \begin{bmatrix}
    _{}^{g}\textrm{R}_c & _{}^{g}\textrm{t}_c \\
    0_{1 \times 3} & 1
    \end{bmatrix}
    \begin{bmatrix}
    X_c\\
    Y_c\\
    Z_c\\
    1
    \end{bmatrix}
\f]

This problem is also known as solving the \f$\mathbf{A}\mathbf{X}=\mathbf{X}\mathbf{B}\f$ equation:
  - for an eye-in-hand configuration
\f[
    \begin{align*}
    ^{b}{\textrm{T}_g}^{(1)} \hspace{0.2em} ^{g}\textrm{T}_c \hspace{0.2em} ^{c}{\textrm{T}_t}^{(1)} &=
    \hspace{0.1em} ^{b}{\textrm{T}_g}^{(2)} \hspace{0.2em} ^{g}\textrm{T}_c \hspace{0.2em} ^{c}{\textrm{T}_t}^{(2)} \\

    (^{b}{\textrm{T}_g}^{(2)})^{-1} \hspace{0.2em} ^{b}{\textrm{T}_g}^{(1)} \hspace{0.2em} ^{g}\textrm{T}_c &=
    \hspace{0.1em} ^{g}\textrm{T}_c \hspace{0.2em} ^{c}{\textrm{T}_t}^{(2)} (^{c}{\textrm{T}_t}^{(1)})^{-1} \\

    \textrm{A}_i \textrm{X} &= \textrm{X} \textrm{B}_i \\
    \end{align*}
\f]

  - for an eye-to-hand configuration
\f[
    \begin{align*}
    ^{g}{\textrm{T}_b}^{(1)} \hspace{0.2em} ^{b}\textrm{T}_c \hspace{0.2em} ^{c}{\textrm{T}_t}^{(1)} &=
    \hspace{0.1em} ^{g}{\textrm{T}_b}^{(2)} \hspace{0.2em} ^{b}\textrm{T}_c \hspace{0.2em} ^{c}{\textrm{T}_t}^{(2)} \\

    (^{g}{\textrm{T}_b}^{(2)})^{-1} \hspace{0.2em} ^{g}{\textrm{T}_b}^{(1)} \hspace{0.2em} ^{b}\textrm{T}_c &=
    \hspace{0.1em} ^{b}\textrm{T}_c \hspace{0.2em} ^{c}{\textrm{T}_t}^{(2)} (^{c}{\textrm{T}_t}^{(1)})^{-1} \\

    \textrm{A}_i \textrm{X} &= \textrm{X} \textrm{B}_i \\
    \end{align*}
\f]

\note
Additional information can be found on this [website](http://campar.in.tum.de/Chair/HandEyeCalibration).
\note
A minimum of 2 motions with non parallel rotation axes are necessary to determine the hand-eye transformation.
So at least 3 different poses are required, but it is strongly recommended to use many more poses.

 */
CV_EXPORTS void calibrateHandEye( InputArrayOfArrays R_gripper2base, InputArrayOfArrays t_gripper2base,
                                    InputArrayOfArrays R_target2cam, InputArrayOfArrays t_target2cam,
                                    OutputArray R_cam2gripper, OutputArray t_cam2gripper,
                                    HandEyeCalibrationMethod method=CALIB_HAND_EYE_TSAI );

/** @brief Computes Robot-World/Hand-Eye calibration: \f$_{}^{w}\textrm{T}_b\f$ and \f$_{}^{c}\textrm{T}_g\f$

@param[in] R_world2cam Rotation part extracted from the homogeneous matrix that transforms a point
expressed in the world frame to the camera frame (\f$_{}^{c}\textrm{T}_w\f$).
This is a vector (`vector<Mat>`) that contains the rotation, `(3x3)` rotation matrices or `(3x1)` rotation vectors,
for all the transformations from world frame to the camera frame.
@param[in] t_world2cam Translation part extracted from the homogeneous matrix that transforms a point
expressed in the world frame to the camera frame (\f$_{}^{c}\textrm{T}_w\f$).
This is a vector (`vector<Mat>`) that contains the `(3x1)` translation vectors for all the transformations
from world frame to the camera frame.
@param[in] R_base2gripper Rotation part extracted from the homogeneous matrix that transforms a point
expressed in the robot base frame to the gripper frame (\f$_{}^{g}\textrm{T}_b\f$).
This is a vector (`vector<Mat>`) that contains the rotation, `(3x3)` rotation matrices or `(3x1)` rotation vectors,
for all the transformations from robot base frame to the gripper frame.
@param[in] t_base2gripper Rotation part extracted from the homogeneous matrix that transforms a point
expressed in the robot base frame to the gripper frame (\f$_{}^{g}\textrm{T}_b\f$).
This is a vector (`vector<Mat>`) that contains the `(3x1)` translation vectors for all the transformations
from robot base frame to the gripper frame.
@param[out] R_base2world Estimated `(3x3)` rotation part extracted from the homogeneous matrix that transforms a point
expressed in the robot base frame to the world frame (\f$_{}^{w}\textrm{T}_b\f$).
@param[out] t_base2world Estimated `(3x1)` translation part extracted from the homogeneous matrix that transforms a point
expressed in the robot base frame to the world frame (\f$_{}^{w}\textrm{T}_b\f$).
@param[out] R_gripper2cam Estimated `(3x3)` rotation part extracted from the homogeneous matrix that transforms a point
expressed in the gripper frame to the camera frame (\f$_{}^{c}\textrm{T}_g\f$).
@param[out] t_gripper2cam Estimated `(3x1)` translation part extracted from the homogeneous matrix that transforms a point
expressed in the gripper frame to the camera frame (\f$_{}^{c}\textrm{T}_g\f$).
@param[in] method One of the implemented Robot-World/Hand-Eye calibration method, see cv::RobotWorldHandEyeCalibrationMethod

The function performs the Robot-World/Hand-Eye calibration using various methods. One approach consists in estimating the
rotation then the translation (separable solutions):
  - M. Shah, Solving the robot-world/hand-eye calibration problem using the kronecker product \cite Shah2013SolvingTR

Another approach consists in estimating simultaneously the rotation and the translation (simultaneous solutions),
with the following implemented method:
  - A. Li, L. Wang, and D. Wu, Simultaneous robot-world and hand-eye calibration using dual-quaternions and kronecker product \cite Li2010SimultaneousRA

The following picture describes the Robot-World/Hand-Eye calibration problem where the transformations between a robot and a world frame
and between a robot gripper ("hand") and a camera ("eye") mounted at the robot end-effector have to be estimated.

![](pics/robot-world_hand-eye_figure.png)

The calibration procedure is the following:
  - a static calibration pattern is used to estimate the transformation between the target frame
  and the camera frame
  - the robot gripper is moved in order to acquire several poses
  - for each pose, the homogeneous transformation between the gripper frame and the robot base frame is recorded using for
  instance the robot kinematics
\f[
    \begin{bmatrix}
    X_g\\
    Y_g\\
    Z_g\\
    1
    \end{bmatrix}
    =
    \begin{bmatrix}
    _{}^{g}\textrm{R}_b & _{}^{g}\textrm{t}_b \\
    0_{1 \times 3} & 1
    \end{bmatrix}
    \begin{bmatrix}
    X_b\\
    Y_b\\
    Z_b\\
    1
    \end{bmatrix}
\f]
  - for each pose, the homogeneous transformation between the calibration target frame (the world frame) and the camera frame is recorded using
  for instance a pose estimation method (PnP) from 2D-3D point correspondences
\f[
    \begin{bmatrix}
    X_c\\
    Y_c\\
    Z_c\\
    1
    \end{bmatrix}
    =
    \begin{bmatrix}
    _{}^{c}\textrm{R}_w & _{}^{c}\textrm{t}_w \\
    0_{1 \times 3} & 1
    \end{bmatrix}
    \begin{bmatrix}
    X_w\\
    Y_w\\
    Z_w\\
    1
    \end{bmatrix}
\f]

The Robot-World/Hand-Eye calibration procedure returns the following homogeneous transformations
\f[
    \begin{bmatrix}
    X_w\\
    Y_w\\
    Z_w\\
    1
    \end{bmatrix}
    =
    \begin{bmatrix}
    _{}^{w}\textrm{R}_b & _{}^{w}\textrm{t}_b \\
    0_{1 \times 3} & 1
    \end{bmatrix}
    \begin{bmatrix}
    X_b\\
    Y_b\\
    Z_b\\
    1
    \end{bmatrix}
\f]
\f[
    \begin{bmatrix}
    X_c\\
    Y_c\\
    Z_c\\
    1
    \end{bmatrix}
    =
    \begin{bmatrix}
    _{}^{c}\textrm{R}_g & _{}^{c}\textrm{t}_g \\
    0_{1 \times 3} & 1
    \end{bmatrix}
    \begin{bmatrix}
    X_g\\
    Y_g\\
    Z_g\\
    1
    \end{bmatrix}
\f]

This problem is also known as solving the \f$\mathbf{A}\mathbf{X}=\mathbf{Z}\mathbf{B}\f$ equation, with:
  - \f$\mathbf{A} \Leftrightarrow \hspace{0.1em} _{}^{c}\textrm{T}_w\f$
  - \f$\mathbf{X} \Leftrightarrow \hspace{0.1em} _{}^{w}\textrm{T}_b\f$
  - \f$\mathbf{Z} \Leftrightarrow \hspace{0.1em} _{}^{c}\textrm{T}_g\f$
  - \f$\mathbf{B} \Leftrightarrow \hspace{0.1em} _{}^{g}\textrm{T}_b\f$

\note
At least 3 measurements are required (input vectors size must be greater or equal to 3).

 */
CV_EXPORTS void calibrateRobotWorldHandEye( InputArrayOfArrays R_world2cam, InputArrayOfArrays t_world2cam,
                                              InputArrayOfArrays R_base2gripper, InputArrayOfArrays t_base2gripper,
                                              OutputArray R_base2world, OutputArray t_base2world,
                                              OutputArray R_gripper2cam, OutputArray t_gripper2cam,
                                              RobotWorldHandEyeCalibrationMethod method=CALIB_ROBOT_WORLD_HAND_EYE_SHAH );

/** @brief The methods in this namespace use a so-called fisheye camera model.
  @ingroup calib3d_fisheye
*/
namespace fisheye
{
//! @addtogroup calib3d_fisheye
//! @{

// For backward compatibility only!
// Use unified Pinhole and Fisheye model calibration flags
using cv::CALIB_USE_INTRINSIC_GUESS;
using cv::CALIB_RECOMPUTE_EXTRINSIC;
using cv::CALIB_CHECK_COND;
using cv::CALIB_FIX_SKEW;
using cv::CALIB_FIX_K1;
using cv::CALIB_FIX_K2;
using cv::CALIB_FIX_K3;
using cv::CALIB_FIX_K4;
using cv::CALIB_FIX_INTRINSIC;
using cv::CALIB_FIX_PRINCIPAL_POINT;
using cv::CALIB_ZERO_DISPARITY;
using cv::CALIB_FIX_FOCAL_LENGTH;

/** @brief Performs camera calibration

@param objectPoints vector of vectors of calibration pattern points in the calibration pattern
coordinate space.
@param imagePoints vector of vectors of the projections of calibration pattern points.
imagePoints.size() and objectPoints.size() and imagePoints[i].size() must be equal to
objectPoints[i].size() for each i.
@param image_size Size of the image used only to initialize the camera intrinsic matrix.
@param K Output 3x3 floating-point camera intrinsic matrix
\f$\cameramatrix{A}\f$ . If
@ref cv::CALIB_USE_INTRINSIC_GUESS is specified, some or all of fx, fy, cx, cy must be
initialized before calling the function.
@param D Output vector of distortion coefficients \f$\distcoeffsfisheye\f$.
@param rvecs Output vector of rotation vectors (see Rodrigues ) estimated for each pattern view.
That is, each k-th rotation vector together with the corresponding k-th translation vector (see
the next output parameter description) brings the calibration pattern from the model coordinate
space (in which object points are specified) to the world coordinate space, that is, a real
position of the calibration pattern in the k-th pattern view (k=0.. *M* -1).
@param tvecs Output vector of translation vectors estimated for each pattern view.
@param flags Different flags that may be zero or a combination of the following values:
-   @ref cv::CALIB_USE_INTRINSIC_GUESS  cameraMatrix contains valid initial values of
fx, fy, cx, cy that are optimized further. Otherwise, (cx, cy) is initially set to the image
center ( imageSize is used), and focal distances are computed in a least-squares fashion.
-   @ref cv::CALIB_RECOMPUTE_EXTRINSIC  Extrinsic will be recomputed after each iteration
of intrinsic optimization.
-   @ref cv::CALIB_CHECK_COND  The functions will check validity of condition number.
-   @ref cv::CALIB_FIX_SKEW  Skew coefficient (alpha) is set to zero and stay zero.
-   @ref cv::CALIB_FIX_K1,..., @ref cv::CALIB_FIX_K4 Selected distortion coefficients
are set to zeros and stay zero.
-   @ref cv::CALIB_FIX_PRINCIPAL_POINT  The principal point is not changed during the global
optimization. It stays at the center or at a different location specified when @ref cv::CALIB_USE_INTRINSIC_GUESS is set too.
-   @ref cv::CALIB_FIX_FOCAL_LENGTH The focal length is not changed during the global
optimization. It is the \f$max(width,height)/\pi\f$ or the provided \f$f_x\f$, \f$f_y\f$ when @ref cv::CALIB_USE_INTRINSIC_GUESS is set too.
@param criteria Termination criteria for the iterative optimization algorithm.
 */
CV_EXPORTS_W double calibrate(InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints, const Size& image_size,
    InputOutputArray K, InputOutputArray D, OutputArrayOfArrays rvecs, OutputArrayOfArrays tvecs, int flags = 0,
        TermCriteria criteria = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, DBL_EPSILON));

/** @brief Performs stereo calibration

@param objectPoints Vector of vectors of the calibration pattern points.
@param imagePoints1 Vector of vectors of the projections of the calibration pattern points,
observed by the first camera.
@param imagePoints2 Vector of vectors of the projections of the calibration pattern points,
observed by the second camera.
@param K1 Input/output first camera intrinsic matrix:
\f$\vecthreethree{f_x^{(j)}}{0}{c_x^{(j)}}{0}{f_y^{(j)}}{c_y^{(j)}}{0}{0}{1}\f$ , \f$j = 0,\, 1\f$ . If
any of @ref cv::CALIB_USE_INTRINSIC_GUESS , @ref cv::CALIB_FIX_INTRINSIC are specified,
some or all of the matrix components must be initialized.
@param D1 Input/output vector of distortion coefficients \f$\distcoeffsfisheye\f$ of 4 elements.
@param K2 Input/output second camera intrinsic matrix. The parameter is similar to K1 .
@param D2 Input/output lens distortion coefficients for the second camera. The parameter is
similar to D1 .
@param imageSize Size of the image used only to initialize camera intrinsic matrix.
@param R Output rotation matrix between the 1st and the 2nd camera coordinate systems.
@param T Output translation vector between the coordinate systems of the cameras.
@param rvecs Output vector of rotation vectors ( @ref Rodrigues ) estimated for each pattern view in the
coordinate system of the first camera of the stereo pair (e.g. std::vector<cv::Mat>). More in detail, each
i-th rotation vector together with the corresponding i-th translation vector (see the next output parameter
description) brings the calibration pattern from the object coordinate space (in which object points are
specified) to the camera coordinate space of the first camera of the stereo pair. In more technical terms,
the tuple of the i-th rotation and translation vector performs a change of basis from object coordinate space
to camera coordinate space of the first camera of the stereo pair.
@param tvecs Output vector of translation vectors estimated for each pattern view, see parameter description
of previous output parameter ( rvecs ).
@param flags Different flags that may be zero or a combination of the following values:
-   @ref cv::CALIB_FIX_INTRINSIC  Fix K1, K2? and D1, D2? so that only R, T matrices
are estimated.
-   @ref cv::CALIB_USE_INTRINSIC_GUESS  K1, K2 contains valid initial values of
fx, fy, cx, cy that are optimized further. Otherwise, (cx, cy) is initially set to the image
center (imageSize is used), and focal distances are computed in a least-squares fashion.
-   @ref cv::CALIB_RECOMPUTE_EXTRINSIC  Extrinsic will be recomputed after each iteration
of intrinsic optimization.
-   @ref cv::CALIB_CHECK_COND  The functions will check validity of condition number.
-   @ref cv::CALIB_FIX_SKEW  Skew coefficient (alpha) is set to zero and stay zero.
-   @ref cv::CALIB_FIX_K1,..., @ref cv::CALIB_FIX_K4 Selected distortion coefficients are set to zeros and stay
zero.
@param criteria Termination criteria for the iterative optimization algorithm.
 */
CV_EXPORTS_W double stereoCalibrate(InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints1, InputArrayOfArrays imagePoints2,
                                    InputOutputArray K1, InputOutputArray D1, InputOutputArray K2, InputOutputArray D2, Size imageSize,
                                    OutputArray R, OutputArray T, OutputArrayOfArrays rvecs, OutputArrayOfArrays tvecs, int flags = CALIB_FIX_INTRINSIC,
                                    TermCriteria criteria = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, DBL_EPSILON));

/// @overload
CV_EXPORTS_W double stereoCalibrate(InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints1, InputArrayOfArrays imagePoints2,
                                    InputOutputArray K1, InputOutputArray D1, InputOutputArray K2, InputOutputArray D2, Size imageSize,
                                    OutputArray R, OutputArray T, int flags = CALIB_FIX_INTRINSIC,
                                    TermCriteria criteria = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, DBL_EPSILON));

//! @} calib3d_fisheye
} //end namespace fisheye
} //end namespace cv

#endif
