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
      RGBD_NORMALS_METHOD_SRI = 2
    };

    RgbdNormals() { }
    virtual ~RgbdNormals() { }

    /** Creates new RgbdNormals object
     * @param rows the number of rows of the depth image normals will be computed on
     * @param cols the number of cols of the depth image normals will be computed on
     * @param depth the depth of the normals (only CV_32F or CV_64F)
     * @param K the calibration matrix to use
     * @param window_size the window size to compute the normals: can only be 1,3,5 or 7
     * @param method one of the methods to use: RGBD_NORMALS_METHOD_SRI, RGBD_NORMALS_METHOD_FALS
     */
    CV_WRAP static Ptr<RgbdNormals> create(int rows = 0, int cols = 0, int depth = 0, InputArray K = noArray(), int window_size = 5,
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
    CV_WRAP virtual void setMethod(RgbdNormals::RgbdNormalsMethod val) = 0;
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
 * @param points3d the resulting 3d points
 */
CV_EXPORTS_W void depthTo3dSparse(InputArray depth, InputArray in_K, InputArray in_points, OutputArray points3d);

/** Converts a depth image to an organized set of 3d points.
 * The coordinate system is x pointing left, y down and z away from the camera
 * @param depth the depth image (if given as short int CV_U, it is assumed to be the depth in millimeters
 *              (as done with the Microsoft Kinect), otherwise, if given as CV_32F or CV_64F, it is assumed in meters)
 * @param K The calibration matrix
 * @param points3d the resulting 3d points. They are of depth the same as `depth` if it is CV_32F or CV_64F, and the
 *        depth of `K` if `depth` is of depth CV_U
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

/** Warp the image: compute 3d points from the depth, transform them using given transformation,
 * then project color point cloud to an image plane.
 * This function can be used to visualize results of the Odometry algorithm.
 * @param image The image (of CV_8UC1 or CV_8UC3 type)
 * @param depth The depth (of type used in depthTo3d fuction)
 * @param mask The mask of used pixels (of CV_8UC1), it can be empty
 * @param Rt The transformation that will be applied to the 3d points computed from the depth
 * @param cameraMatrix Camera matrix
 * @param distCoeff Distortion coefficients
 * @param warpedImage The warped image.
 * @param warpedDepth The warped depth.
 * @param warpedMask The warped mask.
 */
CV_EXPORTS_W void warpFrame(InputArray image, InputArray depth, InputArray mask, InputArray Rt, InputArray cameraMatrix, InputArray distCoeff,
                            OutputArray warpedImage, OutputArray warpedDepth = noArray(), OutputArray warpedMask = noArray());

enum RgbdPlaneMethod
{
    RGBD_PLANE_METHOD_DEFAULT
};

/** Find the planes in a depth image
 * @param points3d the 3d points organized like the depth image: rows x cols with 3 channels
 * @param normals the normals for every point in the depth image
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

/** Object that contains a frame data that is possibly needed for the Odometry.
 * It's used for the efficiency (to pass precomputed/cached data of the frame that participates
 * in the Odometry processing several times).
 */
struct CV_EXPORTS_W OdometryFrame
{
public:
    /** These constants are used to set a type of cache which has to be prepared depending on the frame role:
     * srcFrame or dstFrame (see compute method of the Odometry class). For the srcFrame and dstFrame different cache data may be required,
     * some part of a cache may be common for both frame roles.
     * @param CACHE_SRC The cache data for the srcFrame will be prepared.
     * @param CACHE_DST The cache data for the dstFrame will be prepared.
     * @param CACHE_ALL The cache data for both srcFrame and dstFrame roles will be computed.
     * @param CACHE_DEPTH The frame will be generated from depth image
     * @param CACHE_PTS The frame will be built from point cloud
     */
    enum OdometryFrameCacheType
    {
      CACHE_SRC = 1, CACHE_DST = 2, CACHE_ALL = CACHE_SRC + CACHE_DST
    };

    /** Indicates what pyramid is to access using get/setPyramid... methods:
    * @param PYR_IMAGE The pyramid of RGB images
    * @param PYR_DEPTH The pyramid of depth images
    * @param PYR_MASK  The pyramid of masks
    * @param PYR_CLOUD The pyramid of point clouds, produced from the pyramid of depths
    * @param PYR_DIX   The pyramid of dI/dx derivative images
    * @param PYR_DIY   The pyramid of dI/dy derivative images
    * @param PYR_TEXMASK The pyramid of textured masks
    * @param PYR_NORM  The pyramid of normals
    * @param PYR_NORMMASK The pyramid of normals masks
    */
    enum OdometryFramePyramidType
    {
        PYR_IMAGE = 0, PYR_DEPTH = 1, PYR_MASK = 2, PYR_CLOUD = 3, PYR_DIX = 4, PYR_DIY = 5, PYR_TEXMASK = 6, PYR_NORM = 7, PYR_NORMMASK = 8,
        N_PYRAMIDS
    };

    OdometryFrame() : ID(-1) { }
    virtual ~OdometryFrame() { }

    CV_WRAP static Ptr<OdometryFrame> create(InputArray image = noArray(), InputArray depth = noArray(),
                                             InputArray  mask = noArray(), InputArray normals = noArray(), int ID = -1);

    CV_WRAP virtual void setImage(InputArray  image) = 0;
    CV_WRAP virtual void getImage(OutputArray image) = 0;
    CV_WRAP virtual void setDepth(InputArray  depth) = 0;
    CV_WRAP virtual void getDepth(OutputArray depth) = 0;
    CV_WRAP virtual void setMask(InputArray  mask) = 0;
    CV_WRAP virtual void getMask(OutputArray mask) = 0;
    CV_WRAP virtual void setNormals(InputArray  normals) = 0;
    CV_WRAP virtual void getNormals(OutputArray normals) = 0;

    CV_WRAP virtual void setPyramidLevels(size_t nLevels) = 0;
    CV_WRAP virtual size_t getPyramidLevels(OdometryFrame::OdometryFramePyramidType pyrType) = 0;

    CV_WRAP virtual void setPyramidAt(InputArray  pyrImage, OdometryFrame::OdometryFramePyramidType pyrType, size_t level) = 0;
    CV_WRAP virtual void getPyramidAt(OutputArray pyrImage, OdometryFrame::OdometryFramePyramidType pyrType, size_t level) = 0;

    CV_PROP int ID;
};


/** Base class for computation of odometry.
 */
class CV_EXPORTS_W Odometry
{
public:

    /** A class of transformation*/
    enum OdometryTransformType
    {
      ROTATION = 1, TRANSLATION = 2, RIGID_BODY_MOTION = 4
    };

    virtual ~Odometry() { }

    /** Create a new Odometry object based on its name. Currently supported names are:
    * "RgbdOdometry", "ICPOdometry", "RgbdICPOdometry", "FastICPOdometry".
    * @param odometryType algorithm's name
    */
    CV_WRAP static Ptr<Odometry> createFromName(const std::string& odometryType);

    /** Method to compute a transformation from the source frame to the destination one.
     * Some odometry algorithms do not used some data of frames (eg. ICP does not use images).
     * In such case corresponding arguments can be set as empty Mat.
     * The method returns true if all internal computations were possible (e.g. there were enough correspondences,
     * system of equations has a solution, etc) and resulting transformation satisfies some test if it's provided
     * by the Odometry inheritor implementation (e.g. thresholds for maximum translation and rotation).
     * @param srcImage Image data of the source frame (CV_8UC1)
     * @param srcDepth Depth data of the source frame (CV_32FC1, in meters)
     * @param srcMask Mask that sets which pixels have to be used from the source frame (CV_8UC1)
     * @param dstImage Image data of the destination frame (CV_8UC1)
     * @param dstDepth Depth data of the destination frame (CV_32FC1, in meters)
     * @param dstMask Mask that sets which pixels have to be used from the destination frame (CV_8UC1)
     * @param Rt Resulting transformation from the source frame to the destination one (rigid body motion):
     dst_p = Rt * src_p, where dst_p is a homogeneous point in the destination frame and src_p is
     homogeneous point in the source frame,
     Rt is 4x4 matrix of CV_64FC1 type.
     * @param initRt Initial transformation from the source frame to the destination one (optional)
     */
    CV_WRAP virtual bool compute(InputArray srcImage, InputArray srcDepth, InputArray srcMask, InputArray dstImage, InputArray dstDepth,
                                 InputArray dstMask, OutputArray Rt, InputArray initRt = noArray()) const = 0;

    /** One more method to compute a transformation from the source frame to the destination one.
     * It is designed to save on computing the frame data (image pyramids, normals, etc.).
     */
    CV_WRAP_AS(compute2) virtual bool compute(const Ptr<OdometryFrame>& srcFrame, const Ptr<OdometryFrame>& dstFrame,
                                              OutputArray Rt, InputArray initRt = noArray()) const = 0;

    /** Prepare a cache for the frame. The function checks the precomputed/passed data (throws the error if this data
     * does not satisfy) and computes all remaining cache data needed for the frame. Returned size is a resolution
     * of the prepared frame.
     * @param frame The odometry which will process the frame.
     * @param cacheType The cache type: CACHE_SRC, CACHE_DST or CACHE_ALL.
     */
    CV_WRAP virtual Size prepareFrameCache(Ptr<OdometryFrame> frame, OdometryFrame::OdometryFrameCacheType cacheType) const = 0;

    /** Create odometry frame for current Odometry implementation
     * @param image Image data of the frame (CV_8UC1)
     * @param depth Depth data of the frame (CV_32FC1, in meters)
     * @param mask  Mask that sets which pixels have to be used from the frame (CV_8UC1)
    */
    CV_WRAP virtual Ptr<OdometryFrame> makeOdometryFrame(InputArray image, InputArray depth, InputArray mask) const = 0;

    CV_WRAP virtual void getCameraMatrix(OutputArray val) const = 0;
    CV_WRAP virtual void setCameraMatrix(InputArray val) = 0;
    CV_WRAP virtual Odometry::OdometryTransformType getTransformType() const = 0;
    CV_WRAP virtual void setTransformType(Odometry::OdometryTransformType val) = 0;
    CV_WRAP virtual void getIterationCounts(OutputArray val) const = 0;
    CV_WRAP virtual void setIterationCounts(InputArray val) = 0;
    /** For each pyramid level the pixels will be filtered out if they have gradient magnitude less than minGradientMagnitudes[level].
    * Makes sense for RGB-based algorithms only.
    */
    CV_WRAP virtual void getMinGradientMagnitudes(OutputArray val) const = 0;
    CV_WRAP virtual void setMinGradientMagnitudes(InputArray val) = 0;

    /** Get max allowed translation in meters, default is 0.15 meters
    Found delta transform is considered successful only if the translation is in given limits. */
    CV_WRAP virtual double getMaxTranslation() const = 0;
    /** Set max allowed translation in meters.
    * Found delta transform is considered successful only if the translation is in given limits. */
    CV_WRAP virtual void setMaxTranslation(double val) = 0;
    /** Get max allowed rotation in degrees, default is 15 degrees.
    * Found delta transform is considered successful only if the rotation is in given limits. */
    CV_WRAP virtual double getMaxRotation() const = 0;
    /** Set max allowed rotation in degrees.
    * Found delta transform is considered successful only if the rotation is in given limits. */
    CV_WRAP virtual void setMaxRotation(double val) = 0;
};

/** Odometry based on the paper "Real-Time Visual Odometry from Dense RGB-D Images",
 * F. Steinbucker, J. Strum, D. Cremers, ICCV, 2011.
 */
class CV_EXPORTS_W RgbdOdometry: public Odometry
{
public:
    RgbdOdometry() { }

    /** Creates RgbdOdometry object
     * @param cameraMatrix Camera matrix
     * @param minDepth Pixels with depth less than minDepth will not be used (in meters)
     * @param maxDepth Pixels with depth larger than maxDepth will not be used (in meters)
     * @param maxDepthDiff Correspondences between pixels of two given frames will be filtered out
     *                     if their depth difference is larger than maxDepthDiff (in meters)
     * @param iterCounts Count of iterations on each pyramid level, defaults are [7, 7, 7, 10]
     * @param minGradientMagnitudes For each pyramid level the pixels will be filtered out, default is 10 per each level
     *                              if they have gradient magnitude less than minGradientMagnitudes[level].
     * @param maxPointsPart The method uses a random pixels subset of size frameWidth x frameHeight x pointsPart (from 0 to 1)
     * @param transformType Class of transformation
     */
    CV_WRAP static Ptr<RgbdOdometry> create(InputArray cameraMatrix = noArray(),
                                            float minDepth = 0.f,
                                            float maxDepth = 4.f,
                                            float maxDepthDiff = 0.07f,
                                            InputArray iterCounts = noArray(),
                                            InputArray minGradientMagnitudes = noArray(),
                                            float maxPointsPart = 0.07f,
                                            Odometry::OdometryTransformType transformType = Odometry::RIGID_BODY_MOTION);

    CV_WRAP virtual double getMinDepth() const = 0;
    CV_WRAP virtual void   setMinDepth(double val) = 0;
    CV_WRAP virtual double getMaxDepth() const = 0;
    CV_WRAP virtual void   setMaxDepth(double val) = 0;
    CV_WRAP virtual double getMaxDepthDiff() const = 0;
    CV_WRAP virtual void   setMaxDepthDiff(double val) = 0;
    CV_WRAP virtual double getMaxPointsPart() const = 0;
    CV_WRAP virtual void   setMaxPointsPart(double val) = 0;

};

/** Odometry based on the paper "KinectFusion: Real-Time Dense Surface Mapping and Tracking",
 * Richard A. Newcombe, Andrew Fitzgibbon, at al, SIGGRAPH, 2011.
 */
class CV_EXPORTS_W ICPOdometry: public Odometry
{
public:
    ICPOdometry() { }

    /** Creates new ICPOdometry object
     * @param cameraMatrix Camera matrix
     * @param minDepth Pixels with depth less than minDepth will not be used (in meters)
     * @param maxDepth Pixels with depth larger than maxDepth will not be used (in meters)
     * @param maxDepthDiff Correspondences between pixels of two given frames will be filtered out
     *                     if their depth difference is larger than maxDepthDiff (in meters)
     * @param maxPointsPart The method uses a random pixels subset of size frameWidth x frameHeight x pointsPart (from 0 to 1)
     * @param iterCounts Count of iterations on each pyramid level, defaults are [7, 7, 7, 10]
     * @param transformType Class of trasformation
     */
    CV_WRAP static Ptr<ICPOdometry> create(InputArray cameraMatrix = noArray(),
                                           float minDepth = 0.f,
                                           float maxDepth = 4.f,
                                           float maxDepthDiff = 0.07f,
                                           float maxPointsPart = 0.07f,
                                           InputArray iterCounts = noArray(),
                                           Odometry::OdometryTransformType transformType = Odometry::RIGID_BODY_MOTION);

    CV_WRAP virtual double getMinDepth() const = 0;
    CV_WRAP virtual void   setMinDepth(double val) = 0;
    CV_WRAP virtual double getMaxDepth() const = 0;
    CV_WRAP virtual void   setMaxDepth(double val) = 0;
    CV_WRAP virtual double getMaxDepthDiff() const = 0;
    CV_WRAP virtual void   setMaxDepthDiff(double val) = 0;
    CV_WRAP virtual double getMaxPointsPart() const = 0;
    CV_WRAP virtual void   setMaxPointsPart(double val) = 0;
    CV_WRAP virtual Ptr<RgbdNormals> getNormalsComputer() const = 0;
};

/** Odometry that merges RgbdOdometry and ICPOdometry by minimize sum of their energy functions.
 */
class CV_EXPORTS_W RgbdICPOdometry: public Odometry
{
public:
    RgbdICPOdometry() { }

    /** Creates RgbdICPOdometry object
     * @param cameraMatrix Camera matrix
     * @param minDepth Pixels with depth less than minDepth will not be used (in meters)
     * @param maxDepth Pixels with depth larger than maxDepth will not be used (in meters)
     * @param maxDepthDiff Correspondences between pixels of two given frames will be filtered out
     *                     if their depth difference is larger than maxDepthDiff (in meters)
     * @param maxPointsPart The method uses a random pixels subset of size frameWidth x frameHeight x pointsPart (from 0 to 1)
     * @param iterCounts Count of iterations on each pyramid level, defaults are [7, 7, 7, 10]
     * @param minGradientMagnitudes For each pyramid level the pixels will be filtered out
     *                              if they have gradient magnitude less than minGradientMagnitudes[level], default is 10 per each level
     * @param transformType Class of trasformation
     */
    CV_WRAP static Ptr<RgbdICPOdometry> create(InputArray cameraMatrix = noArray(),
                                               float minDepth = 0.f,
                                               float maxDepth = 4.f,
                                               float maxDepthDiff = 0.07f,
                                               float maxPointsPart = 0.07f,
                                               InputArray iterCounts = noArray(),
                                               InputArray minGradientMagnitudes = noArray(),
                                               Odometry::OdometryTransformType transformType = Odometry::RIGID_BODY_MOTION);

    CV_WRAP virtual double getMinDepth() const = 0;
    CV_WRAP virtual void   setMinDepth(double val) = 0;
    CV_WRAP virtual double getMaxDepth() const = 0;
    CV_WRAP virtual void   setMaxDepth(double val) = 0;
    CV_WRAP virtual double getMaxDepthDiff() const = 0;
    CV_WRAP virtual void   setMaxDepthDiff(double val) = 0;
    CV_WRAP virtual double getMaxPointsPart() const = 0;
    CV_WRAP virtual void   setMaxPointsPart(double val) = 0;

    CV_WRAP virtual Ptr<RgbdNormals> getNormalsComputer() const = 0;
};

/** A faster version of ICPOdometry which is used in KinectFusion implementation
 * Partial list of differences:
 * - UMats are processed using OpenCL
 * - CPU version is written in universal intrinsics
 * - Filters points by angle
 * - Interpolates points and normals
 * - Doesn't use masks or min/max depth filtering
 * - Doesn't use random subsets of points
 * - Supports only Rt transform type
 * - Initial transform is not supported and always ignored
 * - Supports only 4-float vectors as input type
 */
class CV_EXPORTS_W FastICPOdometry: public Odometry
{
public:
    FastICPOdometry() { }

    /** Creates FastICPOdometry object
     * @param cameraMatrix Camera matrix
     * @param maxDistDiff Correspondences between pixels of two given frames will be filtered out
     *                    if their depth difference is larger than maxDepthDiff (in meters)
     * @param angleThreshold Correspondence will be filtered out
     *                     if an angle between their normals is bigger than threshold
     * @param sigmaDepth Depth sigma in meters for bilateral smooth
     * @param sigmaSpatial Spatial sigma in pixels for bilateral smooth
     * @param kernelSize Kernel size in pixels for bilateral smooth
     * @param iterCounts Count of iterations on each pyramid level, defaults are [7, 7, 7, 10]
     * @param depthFactor pre-scale per 1 meter for input values
     * @param truncateThreshold Threshold for depth truncation in meters
     *        All depth values beyond this threshold will be set to zero
     */
    CV_WRAP static Ptr<FastICPOdometry> create(InputArray cameraMatrix = noArray(),
                                               float maxDistDiff = 0.07f,
                                               float angleThreshold = (float)(30. * CV_PI / 180.),
                                               float sigmaDepth = 0.04f,
                                               float sigmaSpatial = 4.5f,
                                               int kernelSize = 7,
                                               InputArray iterCounts = noArray(),
                                               float depthFactor = 1.f,
                                               float truncateThreshold = 0.f);

    CV_WRAP virtual double getMaxDistDiff() const = 0;
    CV_WRAP virtual void   setMaxDistDiff(float val) = 0;
    CV_WRAP virtual float  getAngleThreshold() const = 0;
    CV_WRAP virtual void   setAngleThreshold(float f) = 0;
    CV_WRAP virtual float  getSigmaDepth() const = 0;
    CV_WRAP virtual void   setSigmaDepth(float f) = 0;
    CV_WRAP virtual float  getSigmaSpatial() const = 0;
    CV_WRAP virtual void   setSigmaSpatial(float f) = 0;
    CV_WRAP virtual int    getKernelSize() const = 0;
    CV_WRAP virtual void   setKernelSize(int f) = 0;
    CV_WRAP virtual float  getDepthFactor() const = 0;
    CV_WRAP virtual void   setDepthFactor(float depthFactor) = 0;
    CV_WRAP virtual float  getTruncateThreshold() const = 0;
    CV_WRAP virtual void   setTruncateThreshold(float truncateThreshold) = 0;

};

// TODO Depth interpolation
// Curvature
// Get rescaleDepth return dubles if asked for

//! @}

} /* namespace cv */

#endif // include guard
