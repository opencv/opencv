// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef OPENCV_3D_DEPTH_HPP
#define OPENCV_3D_DEPTH_HPP

#include <opencv2/core.hpp>
#include <limits>

namespace cv
{
//! @addtogroup rgbd
//! @{

/** Object that can compute the normals in an image.
 * It is an object as it can cache data for speed efficiency
 * The implemented methods are either:
 * - FALS (the fastest) and SRI from
 * ``Fast and Accurate Computation of Surface Normals from Range Images``
 * by H. Badino, D. Huber, Y. Park and T. Kanade
 * - the normals with bilateral filtering on a depth image from
 * ``Gradient Response Maps for Real-Time Detection of Texture-Less Objects``
 * by S. Hinterstoisser, C. Cagniart, S. Ilic, P. Sturm, N. Navab, P. Fua, and V. Lepetit
 */
class CV_EXPORTS_W RgbdNormals
{
public:
    enum RgbdNormalsMethod
    {
      RGBD_NORMALS_METHOD_FALS = 0,
      RGBD_NORMALS_METHOD_LINEMOD = 1,
      RGBD_NORMALS_METHOD_SRI = 2,
      RGBD_NORMALS_METHOD_CROSS_PRODUCT = 3
    };

    RgbdNormals() { }
    virtual ~RgbdNormals() { }

    /** Creates new RgbdNormals object
     * @param rows the number of rows of the depth image normals will be computed on
     * @param cols the number of cols of the depth image normals will be computed on
     * @param depth the depth of the normals (only CV_32F or CV_64F)
     * @param K the calibration matrix to use
     * @param window_size the window size to compute the normals: can only be 1,3,5 or 7
     * @param diff_threshold threshold in depth difference, used in LINEMOD algirithm
     * @param method one of the methods to use: RGBD_NORMALS_METHOD_SRI, RGBD_NORMALS_METHOD_FALS
     */
    CV_WRAP static Ptr<RgbdNormals> create(int rows = 0, int cols = 0, int depth = 0, InputArray K = noArray(), int window_size = 5,
                                           float diff_threshold = 50.f,
                                           RgbdNormals::RgbdNormalsMethod method = RgbdNormals::RgbdNormalsMethod::RGBD_NORMALS_METHOD_FALS);

    /** Given a set of 3d points in a depth image, compute the normals at each point.
     * @param points a rows x cols x 3 matrix of CV_32F/CV64F or a rows x cols x 1 CV_U16S
     * @param normals a rows x cols x 3 matrix
     */
    CV_WRAP virtual void apply(InputArray points, OutputArray normals) const = 0;

    /** Prepares cached data required for calculation
    * If not called by user, called automatically at first calculation
    */
    CV_WRAP virtual void cache() const = 0;

    CV_WRAP virtual int getRows() const = 0;
    CV_WRAP virtual void setRows(int val) = 0;
    CV_WRAP virtual int getCols() const = 0;
    CV_WRAP virtual void setCols(int val) = 0;
    CV_WRAP virtual int getWindowSize() const = 0;
    CV_WRAP virtual void setWindowSize(int val) = 0;
    CV_WRAP virtual int getDepth() const = 0;
    CV_WRAP virtual void getK(OutputArray val) const = 0;
    CV_WRAP virtual void setK(InputArray val) = 0;
    CV_WRAP virtual RgbdNormals::RgbdNormalsMethod getMethod() const = 0;
};


/** Registers depth data to an external camera
 * Registration is performed by creating a depth cloud, transforming the cloud by
 * the rigid body transformation between the cameras, and then projecting the
 * transformed points into the RGB camera.
 *
 * uv_rgb = K_rgb * [R | t] * z * inv(K_ir) * uv_ir
 *
 * Currently does not check for negative depth values.
 *
 * @param unregisteredCameraMatrix the camera matrix of the depth camera
 * @param registeredCameraMatrix the camera matrix of the external camera
 * @param registeredDistCoeffs the distortion coefficients of the external camera
 * @param Rt the rigid body transform between the cameras. Transforms points from depth camera frame to external camera frame.
 * @param unregisteredDepth the input depth data
 * @param outputImagePlaneSize the image plane dimensions of the external camera (width, height)
 * @param registeredDepth the result of transforming the depth into the external camera
 * @param depthDilation whether or not the depth is dilated to avoid holes and occlusion errors (optional)
 */
CV_EXPORTS_W void registerDepth(InputArray unregisteredCameraMatrix, InputArray registeredCameraMatrix, InputArray registeredDistCoeffs,
                                InputArray Rt, InputArray unregisteredDepth, const Size& outputImagePlaneSize,
                                OutputArray registeredDepth, bool depthDilation=false);

/**
 * @param depth the depth image
 * @param in_K
 * @param in_points the list of xy coordinates
 * @param points3d the resulting 3d points (point is represented by 4 chanels value [x, y, z, 0])
 */
CV_EXPORTS_W void depthTo3dSparse(InputArray depth, InputArray in_K, InputArray in_points, OutputArray points3d);

/** Converts a depth image to 3d points. If the mask is empty then the resulting array has the same dimensions as `depth`,
 * otherwise it is 1d vector containing mask-enabled values only.
 * The coordinate system is x pointing left, y down and z away from the camera
 * @param depth the depth image (if given as short int CV_U, it is assumed to be the depth in millimeters
 *              (as done with the Microsoft Kinect), otherwise, if given as CV_32F or CV_64F, it is assumed in meters)
 * @param K The calibration matrix
 * @param points3d the resulting 3d points (point is represented by 4 channels value [x, y, z, 0]). They are of the same depth as `depth` if it is CV_32F or CV_64F, and the
 *        depth of `K` if `depth` is of depth CV_16U or CV_16S
 * @param mask the mask of the points to consider (can be empty)
 */
CV_EXPORTS_W void depthTo3d(InputArray depth, InputArray K, OutputArray points3d, InputArray mask = noArray());

/** If the input image is of type CV_16UC1 (like the Kinect one), the image is converted to floats, divided
 * by depth_factor to get a depth in meters, and the values 0 are converted to std::numeric_limits<float>::quiet_NaN()
 * Otherwise, the image is simply converted to floats
 * @param in the depth image (if given as short int CV_U, it is assumed to be the depth in millimeters
 *              (as done with the Microsoft Kinect), it is assumed in meters)
 * @param type the desired output depth (CV_32F or CV_64F)
 * @param out The rescaled float depth image
 * @param depth_factor (optional) factor by which depth is converted to distance (by default = 1000.0 for Kinect sensor)
 */
CV_EXPORTS_W void rescaleDepth(InputArray in, int type, OutputArray out, double depth_factor = 1000.0);

/** Warps depth or RGB-D image by reprojecting it in 3d, applying Rt transformation
 * and then projecting it back onto the image plane.
 * This function can be used to visualize the results of the Odometry algorithm.
 * @param depth Depth data, should be 1-channel CV_16U, CV_16S, CV_32F or CV_64F
 * @param image RGB image (optional), should be 1-, 3- or 4-channel CV_8U
 * @param mask Mask of used pixels (optional), should be CV_8UC1
 * @param Rt Rotation+translation matrix (3x4 or 4x4) to be applied to depth points
 * @param cameraMatrix Camera intrinsics matrix (3x3)
 * @param warpedDepth The warped depth data (optional)
 * @param warpedImage The warped RGB image (optional)
 * @param warpedMask The mask of valid pixels in warped image (optional)
 */
CV_EXPORTS_W void warpFrame(InputArray depth, InputArray image, InputArray mask, InputArray Rt, InputArray cameraMatrix,
                            OutputArray warpedDepth = noArray(), OutputArray warpedImage = noArray(), OutputArray warpedMask = noArray());

enum RgbdPlaneMethod
{
    RGBD_PLANE_METHOD_DEFAULT
};

/** Find the planes in a depth image
 * @param points3d the 3d points organized like the depth image: rows x cols with 3 channels
 * @param normals the normals for every point in the depth image; optional, can be empty
 * @param mask An image where each pixel is labeled with the plane it belongs to
 *        and 255 if it does not belong to any plane
 * @param plane_coefficients the coefficients of the corresponding planes (a,b,c,d) such that ax+by+cz+d=0, norm(a,b,c)=1
 *        and c < 0 (so that the normal points towards the camera)
 * @param block_size The size of the blocks to look at for a stable MSE
 * @param min_size The minimum size of a cluster to be considered a plane
 * @param threshold The maximum distance of a point from a plane to belong to it (in meters)
 * @param sensor_error_a coefficient of the sensor error. 0 by default, use 0.0075 for a Kinect
 * @param sensor_error_b coefficient of the sensor error. 0 by default
 * @param sensor_error_c coefficient of the sensor error. 0 by default
 * @param method The method to use to compute the planes.
 */
CV_EXPORTS_W void findPlanes(InputArray points3d, InputArray normals, OutputArray mask, OutputArray plane_coefficients,
                             int block_size = 40, int min_size = 40*40, double threshold = 0.01,
                             double sensor_error_a = 0, double sensor_error_b = 0,
                             double sensor_error_c = 0,
                             RgbdPlaneMethod method = RGBD_PLANE_METHOD_DEFAULT);



// TODO Depth interpolation
// Curvature
// Get rescaleDepth return dubles if asked for

//! @}

} /* namespace cv */

#endif // include guard
