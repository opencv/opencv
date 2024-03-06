// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef OPENCV_STEREO_HPP
#define OPENCV_STEREO_HPP

#include "opencv2/core.hpp"

/**
  @defgroup stereo Stereo Correspondence

 */

namespace cv {

enum
{
    STEREO_ZERO_DISPARITY=0x00400
};

//! @addtogroup stereo
//! @{

/** @brief Computes rectification transforms for each head of a calibrated stereo camera.

@param cameraMatrix1 First camera intrinsic matrix.
@param distCoeffs1 First camera distortion parameters.
@param cameraMatrix2 Second camera intrinsic matrix.
@param distCoeffs2 Second camera distortion parameters.
@param imageSize Size of the image used for stereo calibration.
@param R Rotation matrix from the coordinate system of the first camera to the second camera,
see @ref stereoCalibrate.
@param T Translation vector from the coordinate system of the first camera to the second camera,
see @ref stereoCalibrate.
@param R1 Output 3x3 rectification transform (rotation matrix) for the first camera. This matrix
brings points given in the unrectified first camera's coordinate system to points in the rectified
first camera's coordinate system. In more technical terms, it performs a change of basis from the
unrectified first camera's coordinate system to the rectified first camera's coordinate system.
@param R2 Output 3x3 rectification transform (rotation matrix) for the second camera. This matrix
brings points given in the unrectified second camera's coordinate system to points in the rectified
second camera's coordinate system. In more technical terms, it performs a change of basis from the
unrectified second camera's coordinate system to the rectified second camera's coordinate system.
@param P1 Output 3x4 projection matrix in the new (rectified) coordinate systems for the first
camera, i.e. it projects points given in the rectified first camera coordinate system into the
rectified first camera's image.
@param P2 Output 3x4 projection matrix in the new (rectified) coordinate systems for the second
camera, i.e. it projects points given in the rectified first camera coordinate system into the
rectified second camera's image.
@param Q Output \f$4 \times 4\f$ disparity-to-depth mapping matrix (see @ref reprojectImageTo3D).
@param flags Operation flags that may be zero or @ref STEREO_ZERO_DISPARITY . If the flag is set,
the function makes the principal points of each camera have the same pixel coordinates in the
rectified views. And if the flag is not set, the function may still shift the images in the
horizontal or vertical direction (depending on the orientation of epipolar lines) to maximize the
useful image area.
@param alpha Free scaling parameter. If it is -1 or absent, the function performs the default
scaling. Otherwise, the parameter should be between 0 and 1. alpha=0 means that the rectified
images are zoomed and shifted so that only valid pixels are visible (no black areas after
rectification). alpha=1 means that the rectified image is decimated and shifted so that all the
pixels from the original images from the cameras are retained in the rectified images (no source
image pixels are lost). Any intermediate value yields an intermediate result between
those two extreme cases.
@param newImageSize New image resolution after rectification. The same size should be passed to
#initUndistortRectifyMap (see the stereo_calib.cpp sample in OpenCV samples directory). When (0,0)
is passed (default), it is set to the original imageSize . Setting it to a larger value can help you
preserve details in the original image, especially when there is a big radial distortion.
@param validPixROI1 Optional output rectangles inside the rectified images where all the pixels
are valid. If alpha=0 , the ROIs cover the whole images. Otherwise, they are likely to be smaller
(see the picture below).
@param validPixROI2 Optional output rectangles inside the rectified images where all the pixels
are valid. If alpha=0 , the ROIs cover the whole images. Otherwise, they are likely to be smaller
(see the picture below).

The function computes the rotation matrices for each camera that (virtually) make both camera image
planes the same plane. Consequently, this makes all the epipolar lines parallel and thus simplifies
the dense stereo correspondence problem. The function takes the matrices computed by #stereoCalibrate
as input. As output, it provides two rotation matrices and also two projection matrices in the new
coordinates. The function distinguishes the following two cases:

-   **Horizontal stereo**: the first and the second camera views are shifted relative to each other
    mainly along the x-axis (with possible small vertical shift). In the rectified images, the
    corresponding epipolar lines in the left and right cameras are horizontal and have the same
    y-coordinate. P1 and P2 look like:

    \f[\texttt{P1} = \begin{bmatrix}
                        f & 0 & cx_1 & 0 \\
                        0 & f & cy & 0 \\
                        0 & 0 & 1 & 0
                     \end{bmatrix}\f]

    \f[\texttt{P2} = \begin{bmatrix}
                        f & 0 & cx_2 & T_x \cdot f \\
                        0 & f & cy & 0 \\
                        0 & 0 & 1 & 0
                     \end{bmatrix} ,\f]

    \f[\texttt{Q} = \begin{bmatrix}
                        1 & 0 & 0 & -cx_1 \\
                        0 & 1 & 0 & -cy \\
                        0 & 0 & 0 & f \\
                        0 & 0 & -\frac{1}{T_x} & \frac{cx_1 - cx_2}{T_x}
                    \end{bmatrix} \f]

    where \f$T_x\f$ is a horizontal shift between the cameras and \f$cx_1=cx_2\f$ if
    @ref STEREO_ZERO_DISPARITY is set.

-   **Vertical stereo**: the first and the second camera views are shifted relative to each other
    mainly in the vertical direction (and probably a bit in the horizontal direction too). The epipolar
    lines in the rectified images are vertical and have the same x-coordinate. P1 and P2 look like:

    \f[\texttt{P1} = \begin{bmatrix}
                        f & 0 & cx & 0 \\
                        0 & f & cy_1 & 0 \\
                        0 & 0 & 1 & 0
                     \end{bmatrix}\f]

    \f[\texttt{P2} = \begin{bmatrix}
                        f & 0 & cx & 0 \\
                        0 & f & cy_2 & T_y \cdot f \\
                        0 & 0 & 1 & 0
                     \end{bmatrix},\f]

    \f[\texttt{Q} = \begin{bmatrix}
                        1 & 0 & 0 & -cx \\
                        0 & 1 & 0 & -cy_1 \\
                        0 & 0 & 0 & f \\
                        0 & 0 & -\frac{1}{T_y} & \frac{cy_1 - cy_2}{T_y}
                    \end{bmatrix} \f]

    where \f$T_y\f$ is a vertical shift between the cameras and \f$cy_1=cy_2\f$ if
    @ref STEREO_ZERO_DISPARITY is set.

As you can see, the first three columns of P1 and P2 will effectively be the new "rectified" camera
matrices. The matrices, together with R1 and R2 , can then be passed to #initUndistortRectifyMap to
initialize the rectification map for each camera.

See below the screenshot from the stereo_calib.cpp sample. Some red horizontal lines pass through
the corresponding image regions. This means that the images are well rectified, which is what most
stereo correspondence algorithms rely on. The green rectangles are roi1 and roi2 . You see that
their interiors are all valid pixels.

![image](pics/stereo_undistort.jpg)
 */
CV_EXPORTS_W void stereoRectify( InputArray cameraMatrix1, InputArray distCoeffs1,
                                 InputArray cameraMatrix2, InputArray distCoeffs2,
                                 Size imageSize, InputArray R, InputArray T,
                                 OutputArray R1, OutputArray R2,
                                 OutputArray P1, OutputArray P2,
                                 OutputArray Q, int flags = STEREO_ZERO_DISPARITY,
                                 double alpha = -1, Size newImageSize = Size(),
                                 CV_OUT Rect* validPixROI1 = 0, CV_OUT Rect* validPixROI2 = 0 );

/** @brief Computes a rectification transform for an uncalibrated stereo camera.

@param points1 Array of feature points in the first image.
@param points2 The corresponding points in the second image. The same formats as in
#findFundamentalMat are supported.
@param F Input fundamental matrix. It can be computed from the same set of point pairs using
#findFundamentalMat .
@param imgSize Size of the image.
@param H1 Output rectification homography matrix for the first image.
@param H2 Output rectification homography matrix for the second image.
@param threshold Optional threshold used to filter out the outliers. If the parameter is greater
than zero, all the point pairs that do not comply with the epipolar geometry (that is, the points
for which \f$|\texttt{points2[i]}^T \cdot \texttt{F} \cdot \texttt{points1[i]}|>\texttt{threshold}\f$ )
are rejected prior to computing the homographies. Otherwise, all the points are considered inliers.

The function computes the rectification transformations without knowing intrinsic parameters of the
cameras and their relative position in the space, which explains the suffix "uncalibrated". Another
related difference from #stereoRectify is that the function outputs not the rectification
transformations in the object (3D) space, but the planar perspective transformations encoded by the
homography matrices H1 and H2 . The function implements the algorithm @cite Hartley99 .

@note
   While the algorithm does not need to know the intrinsic parameters of the cameras, it heavily
    depends on the epipolar geometry. Therefore, if the camera lenses have a significant distortion,
    it would be better to correct it before computing the fundamental matrix and calling this
    function. For example, distortion coefficients can be estimated for each head of stereo camera
    separately by using #calibrateCamera . Then, the images can be corrected using #undistort , or
    just the point coordinates can be corrected with #undistortPoints .
 */
CV_EXPORTS_W bool stereoRectifyUncalibrated( InputArray points1, InputArray points2,
                                             InputArray F, Size imgSize,
                                             OutputArray H1, OutputArray H2,
                                             double threshold = 5 );


CV_EXPORTS float rectify3Collinear( InputArray _cameraMatrix1, InputArray _distCoeffs1,
                   InputArray _cameraMatrix2, InputArray _distCoeffs2,
                   InputArray _cameraMatrix3, InputArray _distCoeffs3,
                   InputArrayOfArrays _imgpt1,
                   InputArrayOfArrays _imgpt3,
                   Size imageSize, InputArray _Rmat12, InputArray _Tmat12,
                   InputArray _Rmat13, InputArray _Tmat13,
                   OutputArray _Rmat1, OutputArray _Rmat2, OutputArray _Rmat3,
                   OutputArray _Pmat1, OutputArray _Pmat2, OutputArray _Pmat3,
                   OutputArray _Qmat,
                   double alpha, Size newImgSize,
                   Rect* roi1, Rect* roi2, int flags );

namespace fisheye {

/** @brief Stereo rectification for fisheye camera model

@param K1 First camera intrinsic matrix.
@param D1 First camera distortion parameters.
@param K2 Second camera intrinsic matrix.
@param D2 Second camera distortion parameters.
@param imageSize Size of the image used for stereo calibration.
@param R Rotation matrix between the coordinate systems of the first and the second
cameras.
@param tvec Translation vector between coordinate systems of the cameras.
@param R1 Output 3x3 rectification transform (rotation matrix) for the first camera.
@param R2 Output 3x3 rectification transform (rotation matrix) for the second camera.
@param P1 Output 3x4 projection matrix in the new (rectified) coordinate systems for the first
camera.
@param P2 Output 3x4 projection matrix in the new (rectified) coordinate systems for the second
camera.
@param Q Output \f$4 \times 4\f$ disparity-to-depth mapping matrix (see reprojectImageTo3D ).
@param flags Operation flags that may be zero or @ref cv::CALIB_ZERO_DISPARITY . If the flag is set,
the function makes the principal points of each camera have the same pixel coordinates in the
rectified views. And if the flag is not set, the function may still shift the images in the
horizontal or vertical direction (depending on the orientation of epipolar lines) to maximize the
useful image area.
@param newImageSize New image resolution after rectification. The same size should be passed to
#initUndistortRectifyMap (see the stereo_calib.cpp sample in OpenCV samples directory). When (0,0)
is passed (default), it is set to the original imageSize . Setting it to larger value can help you
preserve details in the original image, especially when there is a big radial distortion.
@param balance Sets the new focal length in range between the min focal length and the max focal
length. Balance is in range of [0, 1].
@param fov_scale Divisor for new focal length.
 */
CV_EXPORTS_W void stereoRectify(InputArray K1, InputArray D1, InputArray K2, InputArray D2, const Size &imageSize, InputArray R, InputArray tvec,
    OutputArray R1, OutputArray R2, OutputArray P1, OutputArray P2, OutputArray Q, int flags, const Size &newImageSize = Size(),
    double balance = 0.0, double fov_scale = 1.0);

} // namespace fisheye

/** @brief The base class for stereo correspondence algorithms.
 */
class CV_EXPORTS_W StereoMatcher : public Algorithm
{
public:
    enum { DISP_SHIFT = 4,
           DISP_SCALE = (1 << DISP_SHIFT)
         };

    /** @brief Computes disparity map for the specified stereo pair

    @param left Left 8-bit single-channel image.
    @param right Right image of the same size and the same type as the left one.
    @param disparity Output disparity map. It has the same size as the input images. Some algorithms,
    like StereoBM or StereoSGBM compute 16-bit fixed-point disparity map (where each disparity value
    has 4 fractional bits), whereas other algorithms output 32-bit floating-point disparity map.
     */
    CV_WRAP virtual void compute( InputArray left, InputArray right,
                                  OutputArray disparity ) = 0;

    CV_WRAP virtual int getMinDisparity() const = 0;
    CV_WRAP virtual void setMinDisparity(int minDisparity) = 0;

    CV_WRAP virtual int getNumDisparities() const = 0;
    CV_WRAP virtual void setNumDisparities(int numDisparities) = 0;

    CV_WRAP virtual int getBlockSize() const = 0;
    CV_WRAP virtual void setBlockSize(int blockSize) = 0;

    CV_WRAP virtual int getSpeckleWindowSize() const = 0;
    CV_WRAP virtual void setSpeckleWindowSize(int speckleWindowSize) = 0;

    CV_WRAP virtual int getSpeckleRange() const = 0;
    CV_WRAP virtual void setSpeckleRange(int speckleRange) = 0;

    CV_WRAP virtual int getDisp12MaxDiff() const = 0;
    CV_WRAP virtual void setDisp12MaxDiff(int disp12MaxDiff) = 0;
};


/** @brief Class for computing stereo correspondence using the block matching algorithm, introduced and
contributed to OpenCV by K. Konolige.
 */
class CV_EXPORTS_W StereoBM : public StereoMatcher
{
public:
    enum { PREFILTER_NORMALIZED_RESPONSE = 0,
           PREFILTER_XSOBEL              = 1
         };

    CV_WRAP virtual int getPreFilterType() const = 0;
    CV_WRAP virtual void setPreFilterType(int preFilterType) = 0;

    CV_WRAP virtual int getPreFilterSize() const = 0;
    CV_WRAP virtual void setPreFilterSize(int preFilterSize) = 0;

    CV_WRAP virtual int getPreFilterCap() const = 0;
    CV_WRAP virtual void setPreFilterCap(int preFilterCap) = 0;

    CV_WRAP virtual int getTextureThreshold() const = 0;
    CV_WRAP virtual void setTextureThreshold(int textureThreshold) = 0;

    CV_WRAP virtual int getUniquenessRatio() const = 0;
    CV_WRAP virtual void setUniquenessRatio(int uniquenessRatio) = 0;

    CV_WRAP virtual int getSmallerBlockSize() const = 0;
    CV_WRAP virtual void setSmallerBlockSize(int blockSize) = 0;

    CV_WRAP virtual Rect getROI1() const = 0;
    CV_WRAP virtual void setROI1(Rect roi1) = 0;

    CV_WRAP virtual Rect getROI2() const = 0;
    CV_WRAP virtual void setROI2(Rect roi2) = 0;

    /** @brief Creates StereoBM object

    @param numDisparities the disparity search range. For each pixel algorithm will find the best
    disparity from 0 (default minimum disparity) to numDisparities. The search range can then be
    shifted by changing the minimum disparity.
    @param blockSize the linear size of the blocks compared by the algorithm. The size should be odd
    (as the block is centered at the current pixel). Larger block size implies smoother, though less
    accurate disparity map. Smaller block size gives more detailed disparity map, but there is higher
    chance for algorithm to find a wrong correspondence.

    The function create StereoBM object. You can then call StereoBM::compute() to compute disparity for
    a specific stereo pair.
     */
    CV_WRAP static Ptr<StereoBM> create(int numDisparities = 0, int blockSize = 21);
};

/** @brief The class implements the modified H. Hirschmuller algorithm @cite HH08 that differs from the original
one as follows:

-   By default, the algorithm is single-pass, which means that you consider only 5 directions
instead of 8. Set mode=StereoSGBM::MODE_HH in createStereoSGBM to run the full variant of the
algorithm but beware that it may consume a lot of memory.
-   The algorithm matches blocks, not individual pixels. Though, setting blockSize=1 reduces the
blocks to single pixels.
-   Mutual information cost function is not implemented. Instead, a simpler Birchfield-Tomasi
sub-pixel metric from @cite BT98 is used. Though, the color images are supported as well.
-   Some pre- and post- processing steps from K. Konolige algorithm StereoBM are included, for
example: pre-filtering (StereoBM::PREFILTER_XSOBEL type) and post-filtering (uniqueness
check, quadratic interpolation and speckle filtering).

@note
   -   (Python) An example illustrating the use of the StereoSGBM matching algorithm can be found
        at opencv_source_code/samples/python/stereo_match.py
 */
class CV_EXPORTS_W StereoSGBM : public StereoMatcher
{
public:
    enum
    {
        MODE_SGBM = 0,
        MODE_HH   = 1,
        MODE_SGBM_3WAY = 2,
        MODE_HH4  = 3
    };

    CV_WRAP virtual int getPreFilterCap() const = 0;
    CV_WRAP virtual void setPreFilterCap(int preFilterCap) = 0;

    CV_WRAP virtual int getUniquenessRatio() const = 0;
    CV_WRAP virtual void setUniquenessRatio(int uniquenessRatio) = 0;

    CV_WRAP virtual int getP1() const = 0;
    CV_WRAP virtual void setP1(int P1) = 0;

    CV_WRAP virtual int getP2() const = 0;
    CV_WRAP virtual void setP2(int P2) = 0;

    CV_WRAP virtual int getMode() const = 0;
    CV_WRAP virtual void setMode(int mode) = 0;

    /** @brief Creates StereoSGBM object

    @param minDisparity Minimum possible disparity value. Normally, it is zero but sometimes
    rectification algorithms can shift images, so this parameter needs to be adjusted accordingly.
    @param numDisparities Maximum disparity minus minimum disparity. The value is always greater than
    zero. In the current implementation, this parameter must be divisible by 16.
    @param blockSize Matched block size. It must be an odd number \>=1 . Normally, it should be
    somewhere in the 3..11 range.
    @param P1 The first parameter controlling the disparity smoothness. See below.
    @param P2 The second parameter controlling the disparity smoothness. The larger the values are,
    the smoother the disparity is. P1 is the penalty on the disparity change by plus or minus 1
    between neighbor pixels. P2 is the penalty on the disparity change by more than 1 between neighbor
    pixels. The algorithm requires P2 \> P1 . See stereo_match.cpp sample where some reasonably good
    P1 and P2 values are shown (like 8\*number_of_image_channels\*blockSize\*blockSize and
    32\*number_of_image_channels\*blockSize\*blockSize , respectively).
    @param disp12MaxDiff Maximum allowed difference (in integer pixel units) in the left-right
    disparity check. Set it to a non-positive value to disable the check.
    @param preFilterCap Truncation value for the prefiltered image pixels. The algorithm first
    computes x-derivative at each pixel and clips its value by [-preFilterCap, preFilterCap] interval.
    The result values are passed to the Birchfield-Tomasi pixel cost function.
    @param uniquenessRatio Margin in percentage by which the best (minimum) computed cost function
    value should "win" the second best value to consider the found match correct. Normally, a value
    within the 5-15 range is good enough.
    @param speckleWindowSize Maximum size of smooth disparity regions to consider their noise speckles
    and invalidate. Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the
    50-200 range.
    @param speckleRange Maximum disparity variation within each connected component. If you do speckle
    filtering, set the parameter to a positive value, it will be implicitly multiplied by 16.
    Normally, 1 or 2 is good enough.
    @param mode Set it to StereoSGBM::MODE_HH to run the full-scale two-pass dynamic programming
    algorithm. It will consume O(W\*H\*numDisparities) bytes, which is large for 640x480 stereo and
    huge for HD-size pictures. By default, it is set to false .

    The first constructor initializes StereoSGBM with all the default parameters. So, you only have to
    set StereoSGBM::numDisparities at minimum. The second constructor enables you to set each parameter
    to a custom value.
     */
    CV_WRAP static Ptr<StereoSGBM> create(int minDisparity = 0, int numDisparities = 16, int blockSize = 3,
                                          int P1 = 0, int P2 = 0, int disp12MaxDiff = 0,
                                          int preFilterCap = 0, int uniquenessRatio = 0,
                                          int speckleWindowSize = 0, int speckleRange = 0,
                                          int mode = StereoSGBM::MODE_SGBM);
};

/** @brief Filters off small noise blobs (speckles) in the disparity map

@param img The input 16-bit signed disparity image
@param newVal The disparity value used to paint-off the speckles
@param maxSpeckleSize The maximum speckle size to consider it a speckle. Larger blobs are not
affected by the algorithm
@param maxDiff Maximum difference between neighbor disparity pixels to put them into the same
blob. Note that since StereoBM, StereoSGBM and may be other algorithms return a fixed-point
disparity map, where disparity values are multiplied by 16, this scale factor should be taken into
account when specifying this parameter value.
@param buf The optional temporary buffer to avoid memory allocation within the function.
 */
CV_EXPORTS_W void filterSpeckles( InputOutputArray img, double newVal,
                                  int maxSpeckleSize, double maxDiff,
                                  InputOutputArray buf = noArray() );

//! computes valid disparity ROI from the valid ROIs of the rectified images (that are returned by #stereoRectify)
CV_EXPORTS_W Rect getValidDisparityROI( Rect roi1, Rect roi2,
                                        int minDisparity, int numberOfDisparities,
                                        int blockSize );

//! validates disparity using the left-right check. The matrix "cost" should be computed by the stereo correspondence algorithm
CV_EXPORTS_W void validateDisparity( InputOutputArray disparity, InputArray cost,
                                     int minDisparity, int numberOfDisparities,
                                     int disp12MaxDisp = 1 );

/** @brief Reprojects a disparity image to 3D space.

@param disparity Input single-channel 8-bit unsigned, 16-bit signed, 32-bit signed or 32-bit
floating-point disparity image. The values of 8-bit / 16-bit signed formats are assumed to have no
fractional bits. If the disparity is 16-bit signed format, as computed by @ref StereoBM or
@ref StereoSGBM and maybe other algorithms, it should be divided by 16 (and scaled to float) before
being used here.
@param _3dImage Output 3-channel floating-point image of the same size as disparity. Each element of
_3dImage(x,y) contains 3D coordinates of the point (x,y) computed from the disparity map. If one
uses Q obtained by @ref stereoRectify, then the returned points are represented in the first
camera's rectified coordinate system.
@param Q \f$4 \times 4\f$ perspective transformation matrix that can be obtained with
@ref stereoRectify.
@param handleMissingValues Indicates, whether the function should handle missing values (i.e.
points where the disparity was not computed). If handleMissingValues=true, then pixels with the
minimal disparity that corresponds to the outliers (see StereoMatcher::compute ) are transformed
to 3D points with a very large Z value (currently set to 10000).
@param ddepth The optional output array depth. If it is -1, the output image will have CV_32F
depth. ddepth can also be set to CV_16S, CV_32S or CV_32F.

The function transforms a single-channel disparity map to a 3-channel image representing a 3D
surface. That is, for each pixel (x,y) and the corresponding disparity d=disparity(x,y) , it
computes:

\f[\begin{bmatrix}
X \\
Y \\
Z \\
W
\end{bmatrix} = Q \begin{bmatrix}
x \\
y \\
\texttt{disparity} (x,y) \\
1
\end{bmatrix}.\f]

@sa
   To reproject a sparse set of points {(x,y,d),...} to 3D space, use perspectiveTransform.
 */
CV_EXPORTS_W void reprojectImageTo3D( InputArray disparity,
                                      OutputArray _3dImage, InputArray Q,
                                      bool handleMissingValues = false,
                                      int ddepth = -1 );

} // namespace cv

#endif
