// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_WillowGarage.md file found in this module's directory

#ifndef __OPENCV_RGBD_DEPTH_HPP__
#define __OPENCV_RGBD_DEPTH_HPP__

#include <opencv2/core.hpp>
#include <limits>

namespace cv
{
namespace rgbd
{

//! @addtogroup rgbd
//! @{

  /** Checks if the value is a valid depth. For CV_16U or CV_16S, the convention is to be invalid if it is
   * a limit. For a float/double, we just check if it is a NaN
   * @param depth the depth to check for validity
   */
  CV_EXPORTS
  inline bool
  isValidDepth(const float & depth)
  {
    return !cvIsNaN(depth);
  }
  CV_EXPORTS
  inline bool
  isValidDepth(const double & depth)
  {
    return !cvIsNaN(depth);
  }
  CV_EXPORTS
  inline bool
  isValidDepth(const short int & depth)
  {
    return (depth != std::numeric_limits<short int>::min()) && (depth != std::numeric_limits<short int>::max());
  }
  CV_EXPORTS
  inline bool
  isValidDepth(const unsigned short int & depth)
  {
    return (depth != std::numeric_limits<unsigned short int>::min())
        && (depth != std::numeric_limits<unsigned short int>::max());
  }
  CV_EXPORTS
  inline bool
  isValidDepth(const int & depth)
  {
    return (depth != std::numeric_limits<int>::min()) && (depth != std::numeric_limits<int>::max());
  }
  CV_EXPORTS
  inline bool
  isValidDepth(const unsigned int & depth)
  {
    return (depth != std::numeric_limits<unsigned int>::min()) && (depth != std::numeric_limits<unsigned int>::max());
  }

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
  class CV_EXPORTS_W RgbdNormals: public Algorithm
  {
  public:
    enum RGBD_NORMALS_METHOD
    {
      RGBD_NORMALS_METHOD_FALS = 0,
      RGBD_NORMALS_METHOD_LINEMOD = 1,
      RGBD_NORMALS_METHOD_SRI = 2
    };

    RgbdNormals()
        :
          rows_(0),
          cols_(0),
          depth_(0),
          K_(Mat()),
          window_size_(0),
          method_(RGBD_NORMALS_METHOD_FALS),
          rgbd_normals_impl_(0)
    {
    }

    /** Constructor
     * @param rows the number of rows of the depth image normals will be computed on
     * @param cols the number of cols of the depth image normals will be computed on
     * @param depth the depth of the normals (only CV_32F or CV_64F)
     * @param K the calibration matrix to use
     * @param window_size the window size to compute the normals: can only be 1,3,5 or 7
     * @param method one of the methods to use: RGBD_NORMALS_METHOD_SRI, RGBD_NORMALS_METHOD_FALS
     */
    RgbdNormals(int rows, int cols, int depth, InputArray K, int window_size = 5, int method =
        RgbdNormals::RGBD_NORMALS_METHOD_FALS);

    ~RgbdNormals();

    CV_WRAP static Ptr<RgbdNormals> create(int rows, int cols, int depth, InputArray K, int window_size = 5, int method =
        RgbdNormals::RGBD_NORMALS_METHOD_FALS);

    /** Given a set of 3d points in a depth image, compute the normals at each point.
     * @param points a rows x cols x 3 matrix of CV_32F/CV64F or a rows x cols x 1 CV_U16S
     * @param normals a rows x cols x 3 matrix
     */
    CV_WRAP_AS(apply) void
    operator()(InputArray points, OutputArray normals) const;

    /** Initializes some data that is cached for later computation
     * If that function is not called, it will be called the first time normals are computed
     */
    CV_WRAP void
    initialize() const;

    CV_WRAP int getRows() const
    {
        return rows_;
    }
    CV_WRAP void setRows(int val)
    {
        rows_ = val;
    }
    CV_WRAP int getCols() const
    {
        return cols_;
    }
    CV_WRAP void setCols(int val)
    {
        cols_ = val;
    }
    CV_WRAP int getWindowSize() const
    {
        return window_size_;
    }
    CV_WRAP void setWindowSize(int val)
    {
        window_size_ = val;
    }
    CV_WRAP int getDepth() const
    {
        return depth_;
    }
    CV_WRAP void setDepth(int val)
    {
        depth_ = val;
    }
    CV_WRAP cv::Mat getK() const
    {
        return K_;
    }
    CV_WRAP void setK(const cv::Mat &val)
    {
        K_ = val;
    }
    CV_WRAP int getMethod() const
    {
        return method_;
    }
    CV_WRAP void setMethod(int val)
    {
        method_ = val;
    }

  protected:
    void
    initialize_normals_impl(int rows, int cols, int depth, const Mat & K, int window_size, int method) const;

    int rows_, cols_, depth_;
    Mat K_;
    int window_size_;
    int method_;
    mutable void* rgbd_normals_impl_;
  };

  /** Object that can clean a noisy depth image
   */
  class CV_EXPORTS_W DepthCleaner: public Algorithm
  {
  public:
    /** NIL method is from
     * ``Modeling Kinect Sensor Noise for Improved 3d Reconstruction and Tracking``
     * by C. Nguyen, S. Izadi, D. Lovel
     */
    enum DEPTH_CLEANER_METHOD
    {
      DEPTH_CLEANER_NIL
    };

    DepthCleaner()
        :
          depth_(0),
          window_size_(0),
          method_(DEPTH_CLEANER_NIL),
          depth_cleaner_impl_(0)
    {
    }

    /** Constructor
     * @param depth the depth of the normals (only CV_32F or CV_64F)
     * @param window_size the window size to compute the normals: can only be 1,3,5 or 7
     * @param method one of the methods to use: RGBD_NORMALS_METHOD_SRI, RGBD_NORMALS_METHOD_FALS
     */
    DepthCleaner(int depth, int window_size = 5, int method = DepthCleaner::DEPTH_CLEANER_NIL);

    ~DepthCleaner();

    CV_WRAP static Ptr<DepthCleaner> create(int depth, int window_size = 5, int method = DepthCleaner::DEPTH_CLEANER_NIL);

    /** Given a set of 3d points in a depth image, compute the normals at each point.
     * @param points a rows x cols x 3 matrix of CV_32F/CV64F or a rows x cols x 1 CV_U16S
     * @param depth a rows x cols matrix of the cleaned up depth
     */
    CV_WRAP_AS(apply) void
    operator()(InputArray points, OutputArray depth) const;

    /** Initializes some data that is cached for later computation
     * If that function is not called, it will be called the first time normals are computed
     */
    CV_WRAP void
    initialize() const;

    CV_WRAP int getWindowSize() const
    {
        return window_size_;
    }
    CV_WRAP void setWindowSize(int val)
    {
        window_size_ = val;
    }
    CV_WRAP int getDepth() const
    {
        return depth_;
    }
    CV_WRAP void setDepth(int val)
    {
        depth_ = val;
    }
    CV_WRAP int getMethod() const
    {
        return method_;
    }
    CV_WRAP void setMethod(int val)
    {
        method_ = val;
    }

  protected:
    void
    initialize_cleaner_impl() const;

    int depth_;
    int window_size_;
    int method_;
    mutable void* depth_cleaner_impl_;
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
  CV_EXPORTS_W
  void
  registerDepth(InputArray unregisteredCameraMatrix, InputArray registeredCameraMatrix, InputArray registeredDistCoeffs,
                InputArray Rt, InputArray unregisteredDepth, const Size& outputImagePlaneSize,
                OutputArray registeredDepth, bool depthDilation=false);

  /**
   * @param depth the depth image
   * @param in_K
   * @param in_points the list of xy coordinates
   * @param points3d the resulting 3d points
   */
  CV_EXPORTS_W
  void
  depthTo3dSparse(InputArray depth, InputArray in_K, InputArray in_points, OutputArray points3d);

  /** Converts a depth image to an organized set of 3d points.
   * The coordinate system is x pointing left, y down and z away from the camera
   * @param depth the depth image (if given as short int CV_U, it is assumed to be the depth in millimeters
   *              (as done with the Microsoft Kinect), otherwise, if given as CV_32F or CV_64F, it is assumed in meters)
   * @param K The calibration matrix
   * @param points3d the resulting 3d points. They are of depth the same as `depth` if it is CV_32F or CV_64F, and the
   *        depth of `K` if `depth` is of depth CV_U
   * @param mask the mask of the points to consider (can be empty)
   */
  CV_EXPORTS_W
  void
  depthTo3d(InputArray depth, InputArray K, OutputArray points3d, InputArray mask = noArray());

  /** If the input image is of type CV_16UC1 (like the Kinect one), the image is converted to floats, divided
   * by depth_factor to get a depth in meters, and the values 0 are converted to std::numeric_limits<float>::quiet_NaN()
   * Otherwise, the image is simply converted to floats
   * @param in the depth image (if given as short int CV_U, it is assumed to be the depth in millimeters
   *              (as done with the Microsoft Kinect), it is assumed in meters)
   * @param depth the desired output depth (floats or double)
   * @param out The rescaled float depth image
   * @param depth_factor (optional) factor by which depth is converted to distance (by default = 1000.0 for Kinect sensor)
   */
  CV_EXPORTS_W
  void
  rescaleDepth(InputArray in, int depth, OutputArray out, double depth_factor = 1000.0);

  /** Object that can compute planes in an image
   */
  class CV_EXPORTS_W RgbdPlane: public Algorithm
  {
  public:
    enum RGBD_PLANE_METHOD
    {
      RGBD_PLANE_METHOD_DEFAULT
    };

      RgbdPlane(int method = RgbdPlane::RGBD_PLANE_METHOD_DEFAULT)
        :
          method_(method),
          block_size_(40),
          min_size_(block_size_*block_size_),
          threshold_(0.01),
          sensor_error_a_(0),
          sensor_error_b_(0),
          sensor_error_c_(0)
    {
    }

    /** Constructor
     * @param block_size The size of the blocks to look at for a stable MSE
     * @param min_size The minimum size of a cluster to be considered a plane
     * @param threshold The maximum distance of a point from a plane to belong to it (in meters)
     * @param sensor_error_a coefficient of the sensor error. 0 by default, 0.0075 for a Kinect
     * @param sensor_error_b coefficient of the sensor error. 0 by default
     * @param sensor_error_c coefficient of the sensor error. 0 by default
     * @param method The method to use to compute the planes.
     */
    RgbdPlane(int method, int block_size,
              int min_size, double threshold, double sensor_error_a = 0,
              double sensor_error_b = 0, double sensor_error_c = 0);

    ~RgbdPlane();

    CV_WRAP static Ptr<RgbdPlane> create(int method, int block_size, int min_size, double threshold,
                                         double sensor_error_a = 0, double sensor_error_b = 0,
                                         double sensor_error_c = 0);

    /** Find The planes in a depth image
     * @param points3d the 3d points organized like the depth image: rows x cols with 3 channels
     * @param normals the normals for every point in the depth image
     * @param mask An image where each pixel is labeled with the plane it belongs to
     *        and 255 if it does not belong to any plane
     * @param plane_coefficients the coefficients of the corresponding planes (a,b,c,d) such that ax+by+cz+d=0, norm(a,b,c)=1
     *        and c < 0 (so that the normal points towards the camera)
     */
    CV_WRAP_AS(apply) void
    operator()(InputArray points3d, InputArray normals, OutputArray mask,
               OutputArray plane_coefficients);

    /** Find The planes in a depth image but without doing a normal check, which is faster but less accurate
     * @param points3d the 3d points organized like the depth image: rows x cols with 3 channels
     * @param mask An image where each pixel is labeled with the plane it belongs to
     *        and 255 if it does not belong to any plane
     * @param plane_coefficients the coefficients of the corresponding planes (a,b,c,d) such that ax+by+cz+d=0
     */
    CV_WRAP_AS(apply) void
    operator()(InputArray points3d, OutputArray mask, OutputArray plane_coefficients);

    CV_WRAP int getBlockSize() const
    {
        return block_size_;
    }
    CV_WRAP void setBlockSize(int val)
    {
        block_size_ = val;
    }
    CV_WRAP int getMinSize() const
    {
        return min_size_;
    }
    CV_WRAP void setMinSize(int val)
    {
        min_size_ = val;
    }
    CV_WRAP int getMethod() const
    {
        return method_;
    }
    CV_WRAP void setMethod(int val)
    {
        method_ = val;
    }
    CV_WRAP double getThreshold() const
    {
        return threshold_;
    }
    CV_WRAP void setThreshold(double val)
    {
        threshold_ = val;
    }
    CV_WRAP double getSensorErrorA() const
    {
        return sensor_error_a_;
    }
    CV_WRAP void setSensorErrorA(double val)
    {
        sensor_error_a_ = val;
    }
    CV_WRAP double getSensorErrorB() const
    {
        return sensor_error_b_;
    }
    CV_WRAP void setSensorErrorB(double val)
    {
        sensor_error_b_ = val;
    }
    CV_WRAP double getSensorErrorC() const
    {
        return sensor_error_c_;
    }
    CV_WRAP void setSensorErrorC(double val)
    {
        sensor_error_c_ = val;
    }

  private:
    /** The method to use to compute the planes */
    int method_;
    /** The size of the blocks to look at for a stable MSE */
    int block_size_;
    /** The minimum size of a cluster to be considered a plane */
    int min_size_;
    /** How far a point can be from a plane to belong to it (in meters) */
    double threshold_;
    /** coefficient of the sensor error with respect to the. All 0 by default but you want a=0.0075 for a Kinect */
    double sensor_error_a_, sensor_error_b_, sensor_error_c_;
  };

  /** Object that contains a frame data.
   */
  struct CV_EXPORTS_W RgbdFrame
  {
      RgbdFrame();
      RgbdFrame(const Mat& image, const Mat& depth, const Mat& mask=Mat(), const Mat& normals=Mat(), int ID=-1);
      virtual ~RgbdFrame();

      CV_WRAP static Ptr<RgbdFrame> create(const Mat& image=Mat(), const Mat& depth=Mat(), const Mat& mask=Mat(), const Mat& normals=Mat(), int ID=-1);

      CV_WRAP virtual void
      release();

      CV_PROP int ID;
      CV_PROP Mat image;
      CV_PROP Mat depth;
      CV_PROP Mat mask;
      CV_PROP Mat normals;
  };

  /** Object that contains a frame data that is possibly needed for the Odometry.
   * It's used for the efficiency (to pass precomputed/cached data of the frame that participates
   * in the Odometry processing several times).
   */
  struct CV_EXPORTS_W OdometryFrame : public RgbdFrame
  {
    /** These constants are used to set a type of cache which has to be prepared depending on the frame role:
     * srcFrame or dstFrame (see compute method of the Odometry class). For the srcFrame and dstFrame different cache data may be required,
     * some part of a cache may be common for both frame roles.
     * @param CACHE_SRC The cache data for the srcFrame will be prepared.
     * @param CACHE_DST The cache data for the dstFrame will be prepared.
     * @param CACHE_ALL The cache data for both srcFrame and dstFrame roles will be computed.
     */
    enum
    {
      CACHE_SRC = 1, CACHE_DST = 2, CACHE_ALL = CACHE_SRC + CACHE_DST
    };

    OdometryFrame();
    OdometryFrame(const Mat& image, const Mat& depth, const Mat& mask=Mat(), const Mat& normals=Mat(), int ID=-1);

    CV_WRAP static Ptr<OdometryFrame> create(const Mat& image=Mat(), const Mat& depth=Mat(), const Mat& mask=Mat(), const Mat& normals=Mat(), int ID=-1);

    CV_WRAP virtual void
    release() CV_OVERRIDE;

    CV_WRAP void
    releasePyramids();

    CV_PROP std::vector<Mat> pyramidImage;
    CV_PROP std::vector<Mat> pyramidDepth;
    CV_PROP std::vector<Mat> pyramidMask;

    CV_PROP std::vector<Mat> pyramidCloud;

    CV_PROP std::vector<Mat> pyramid_dI_dx;
    CV_PROP std::vector<Mat> pyramid_dI_dy;
    CV_PROP std::vector<Mat> pyramidTexturedMask;

    CV_PROP std::vector<Mat> pyramidNormals;
    CV_PROP std::vector<Mat> pyramidNormalsMask;
  };

  /** Base class for computation of odometry.
   */
  class CV_EXPORTS_W Odometry: public Algorithm
  {
  public:

    /** A class of transformation*/
    enum
    {
      ROTATION = 1, TRANSLATION = 2, RIGID_BODY_MOTION = 4
    };

    CV_WRAP static inline float
    DEFAULT_MIN_DEPTH()
    {
      return 0.f; // in meters
    }
    CV_WRAP static inline float
    DEFAULT_MAX_DEPTH()
    {
      return 4.f; // in meters
    }
    CV_WRAP static inline float
    DEFAULT_MAX_DEPTH_DIFF()
    {
      return 0.07f; // in meters
    }
    CV_WRAP static inline float
    DEFAULT_MAX_POINTS_PART()
    {
      return 0.07f; // in [0, 1]
    }
    CV_WRAP static inline float
    DEFAULT_MAX_TRANSLATION()
    {
      return 0.15f; // in meters
    }
    CV_WRAP static inline float
    DEFAULT_MAX_ROTATION()
    {
      return 15; // in degrees
    }

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
    CV_WRAP bool
    compute(const Mat& srcImage, const Mat& srcDepth, const Mat& srcMask, const Mat& dstImage, const Mat& dstDepth,
            const Mat& dstMask, OutputArray Rt, const Mat& initRt = Mat()) const;

    /** One more method to compute a transformation from the source frame to the destination one.
     * It is designed to save on computing the frame data (image pyramids, normals, etc.).
     */
    CV_WRAP_AS(compute2) bool
    compute(Ptr<OdometryFrame>& srcFrame, Ptr<OdometryFrame>& dstFrame, OutputArray Rt, const Mat& initRt = Mat()) const;

    /** Prepare a cache for the frame. The function checks the precomputed/passed data (throws the error if this data
     * does not satisfy) and computes all remaining cache data needed for the frame. Returned size is a resolution
     * of the prepared frame.
     * @param frame The odometry which will process the frame.
     * @param cacheType The cache type: CACHE_SRC, CACHE_DST or CACHE_ALL.
     */
    CV_WRAP virtual Size prepareFrameCache(Ptr<OdometryFrame>& frame, int cacheType) const;

    CV_WRAP static Ptr<Odometry> create(const String & odometryType);

    /** @see setCameraMatrix */
    CV_WRAP virtual cv::Mat getCameraMatrix() const = 0;
    /** @copybrief getCameraMatrix @see getCameraMatrix */
    CV_WRAP virtual void setCameraMatrix(const cv::Mat &val) = 0;
    /** @see setTransformType */
    CV_WRAP virtual int getTransformType() const = 0;
    /** @copybrief getTransformType @see getTransformType */
    CV_WRAP virtual void setTransformType(int val) = 0;

  protected:
    virtual void
    checkParams() const = 0;

    virtual bool
    computeImpl(const Ptr<OdometryFrame>& srcFrame, const Ptr<OdometryFrame>& dstFrame, OutputArray Rt,
                const Mat& initRt) const = 0;
  };

  /** Odometry based on the paper "Real-Time Visual Odometry from Dense RGB-D Images",
   * F. Steinbucker, J. Strum, D. Cremers, ICCV, 2011.
   */
  class CV_EXPORTS_W RgbdOdometry: public Odometry
  {
  public:
    RgbdOdometry();
    /** Constructor.
     * @param cameraMatrix Camera matrix
     * @param minDepth Pixels with depth less than minDepth will not be used (in meters)
     * @param maxDepth Pixels with depth larger than maxDepth will not be used (in meters)
     * @param maxDepthDiff Correspondences between pixels of two given frames will be filtered out
     *                     if their depth difference is larger than maxDepthDiff (in meters)
     * @param iterCounts Count of iterations on each pyramid level.
     * @param minGradientMagnitudes For each pyramid level the pixels will be filtered out
     *                              if they have gradient magnitude less than minGradientMagnitudes[level].
     * @param maxPointsPart The method uses a random pixels subset of size frameWidth x frameHeight x pointsPart
     * @param transformType Class of transformation
     */
    RgbdOdometry(const Mat& cameraMatrix, float minDepth = Odometry::DEFAULT_MIN_DEPTH(), float maxDepth = Odometry::DEFAULT_MAX_DEPTH(),
                 float maxDepthDiff = Odometry::DEFAULT_MAX_DEPTH_DIFF(), const std::vector<int>& iterCounts = std::vector<int>(),
                 const std::vector<float>& minGradientMagnitudes = std::vector<float>(), float maxPointsPart = Odometry::DEFAULT_MAX_POINTS_PART(),
                 int transformType = Odometry::RIGID_BODY_MOTION);

    CV_WRAP static Ptr<RgbdOdometry> create(const Mat& cameraMatrix = Mat(), float minDepth = Odometry::DEFAULT_MIN_DEPTH(), float maxDepth = Odometry::DEFAULT_MAX_DEPTH(),
                 float maxDepthDiff = Odometry::DEFAULT_MAX_DEPTH_DIFF(), const std::vector<int>& iterCounts = std::vector<int>(),
                 const std::vector<float>& minGradientMagnitudes = std::vector<float>(), float maxPointsPart = Odometry::DEFAULT_MAX_POINTS_PART(),
                 int transformType = Odometry::RIGID_BODY_MOTION);

    CV_WRAP virtual Size prepareFrameCache(Ptr<OdometryFrame>& frame, int cacheType) const CV_OVERRIDE;

    CV_WRAP cv::Mat getCameraMatrix() const CV_OVERRIDE
    {
        return cameraMatrix;
    }
    CV_WRAP void setCameraMatrix(const cv::Mat &val) CV_OVERRIDE
    {
        cameraMatrix = val;
    }
    CV_WRAP double getMinDepth() const
    {
        return minDepth;
    }
    CV_WRAP void setMinDepth(double val)
    {
        minDepth = val;
    }
    CV_WRAP double getMaxDepth() const
    {
        return maxDepth;
    }
    CV_WRAP void setMaxDepth(double val)
    {
        maxDepth = val;
    }
    CV_WRAP double getMaxDepthDiff() const
    {
        return maxDepthDiff;
    }
    CV_WRAP void setMaxDepthDiff(double val)
    {
        maxDepthDiff = val;
    }
    CV_WRAP cv::Mat getIterationCounts() const
    {
        return iterCounts;
    }
    CV_WRAP void setIterationCounts(const cv::Mat &val)
    {
        iterCounts = val;
    }
    CV_WRAP cv::Mat getMinGradientMagnitudes() const
    {
        return minGradientMagnitudes;
    }
    CV_WRAP void setMinGradientMagnitudes(const cv::Mat &val)
    {
        minGradientMagnitudes = val;
    }
    CV_WRAP double getMaxPointsPart() const
    {
        return maxPointsPart;
    }
    CV_WRAP void setMaxPointsPart(double val)
    {
        maxPointsPart = val;
    }
    CV_WRAP int getTransformType() const CV_OVERRIDE
    {
        return transformType;
    }
    CV_WRAP void setTransformType(int val) CV_OVERRIDE
    {
        transformType = val;
    }
    CV_WRAP double getMaxTranslation() const
    {
        return maxTranslation;
    }
    CV_WRAP void setMaxTranslation(double val)
    {
        maxTranslation = val;
    }
    CV_WRAP double getMaxRotation() const
    {
        return maxRotation;
    }
    CV_WRAP void setMaxRotation(double val)
    {
        maxRotation = val;
    }

  protected:
    virtual void
    checkParams() const CV_OVERRIDE;

    virtual bool
    computeImpl(const Ptr<OdometryFrame>& srcFrame, const Ptr<OdometryFrame>& dstFrame, OutputArray Rt,
                const Mat& initRt) const CV_OVERRIDE;

    // Some params have commented desired type. It's due to AlgorithmInfo::addParams does not support it now.
    /*float*/
    double minDepth, maxDepth, maxDepthDiff;
    /*vector<int>*/
    Mat iterCounts;
    /*vector<float>*/
    Mat minGradientMagnitudes;
    double maxPointsPart;

    Mat cameraMatrix;
    int transformType;

    double maxTranslation, maxRotation;
  };

  /** Odometry based on the paper "KinectFusion: Real-Time Dense Surface Mapping and Tracking",
   * Richard A. Newcombe, Andrew Fitzgibbon, at al, SIGGRAPH, 2011.
   */
  class CV_EXPORTS_W ICPOdometry: public Odometry
  {
  public:
    ICPOdometry();
    /** Constructor.
     * @param cameraMatrix Camera matrix
     * @param minDepth Pixels with depth less than minDepth will not be used
     * @param maxDepth Pixels with depth larger than maxDepth will not be used
     * @param maxDepthDiff Correspondences between pixels of two given frames will be filtered out
     *                     if their depth difference is larger than maxDepthDiff
     * @param maxPointsPart The method uses a random pixels subset of size frameWidth x frameHeight x pointsPart
     * @param iterCounts Count of iterations on each pyramid level.
     * @param transformType Class of trasformation
     */
    ICPOdometry(const Mat& cameraMatrix, float minDepth = Odometry::DEFAULT_MIN_DEPTH(), float maxDepth = Odometry::DEFAULT_MAX_DEPTH(),
                float maxDepthDiff = Odometry::DEFAULT_MAX_DEPTH_DIFF(), float maxPointsPart = Odometry::DEFAULT_MAX_POINTS_PART(),
                const std::vector<int>& iterCounts = std::vector<int>(), int transformType = Odometry::RIGID_BODY_MOTION);

    CV_WRAP static Ptr<ICPOdometry> create(const Mat& cameraMatrix = Mat(), float minDepth = Odometry::DEFAULT_MIN_DEPTH(), float maxDepth = Odometry::DEFAULT_MAX_DEPTH(),
                float maxDepthDiff = Odometry::DEFAULT_MAX_DEPTH_DIFF(), float maxPointsPart = Odometry::DEFAULT_MAX_POINTS_PART(),
                const std::vector<int>& iterCounts = std::vector<int>(), int transformType = Odometry::RIGID_BODY_MOTION);

    CV_WRAP virtual Size prepareFrameCache(Ptr<OdometryFrame>& frame, int cacheType) const CV_OVERRIDE;

    CV_WRAP cv::Mat getCameraMatrix() const CV_OVERRIDE
    {
        return cameraMatrix;
    }
    CV_WRAP void setCameraMatrix(const cv::Mat &val) CV_OVERRIDE
    {
        cameraMatrix = val;
    }
    CV_WRAP double getMinDepth() const
    {
        return minDepth;
    }
    CV_WRAP void setMinDepth(double val)
    {
        minDepth = val;
    }
    CV_WRAP double getMaxDepth() const
    {
        return maxDepth;
    }
    CV_WRAP void setMaxDepth(double val)
    {
        maxDepth = val;
    }
    CV_WRAP double getMaxDepthDiff() const
    {
        return maxDepthDiff;
    }
    CV_WRAP void setMaxDepthDiff(double val)
    {
        maxDepthDiff = val;
    }
    CV_WRAP cv::Mat getIterationCounts() const
    {
        return iterCounts;
    }
    CV_WRAP void setIterationCounts(const cv::Mat &val)
    {
        iterCounts = val;
    }
    CV_WRAP double getMaxPointsPart() const
    {
        return maxPointsPart;
    }
    CV_WRAP void setMaxPointsPart(double val)
    {
        maxPointsPart = val;
    }
    CV_WRAP int getTransformType() const CV_OVERRIDE
    {
        return transformType;
    }
    CV_WRAP void setTransformType(int val) CV_OVERRIDE
    {
        transformType = val;
    }
    CV_WRAP double getMaxTranslation() const
    {
        return maxTranslation;
    }
    CV_WRAP void setMaxTranslation(double val)
    {
        maxTranslation = val;
    }
    CV_WRAP double getMaxRotation() const
    {
        return maxRotation;
    }
    CV_WRAP void setMaxRotation(double val)
    {
        maxRotation = val;
    }
    CV_WRAP Ptr<RgbdNormals> getNormalsComputer() const
    {
        return normalsComputer;
    }

  protected:
    virtual void
    checkParams() const CV_OVERRIDE;

    virtual bool
    computeImpl(const Ptr<OdometryFrame>& srcFrame, const Ptr<OdometryFrame>& dstFrame, OutputArray Rt,
                const Mat& initRt) const CV_OVERRIDE;

    // Some params have commented desired type. It's due to AlgorithmInfo::addParams does not support it now.
    /*float*/
    double minDepth, maxDepth, maxDepthDiff;
    /*float*/
    double maxPointsPart;
    /*vector<int>*/
    Mat iterCounts;

    Mat cameraMatrix;
    int transformType;

    double maxTranslation, maxRotation;

    mutable Ptr<RgbdNormals> normalsComputer;
  };

  /** Odometry that merges RgbdOdometry and ICPOdometry by minimize sum of their energy functions.
   */

  class CV_EXPORTS_W RgbdICPOdometry: public Odometry
  {
  public:
    RgbdICPOdometry();
    /** Constructor.
     * @param cameraMatrix Camera matrix
     * @param minDepth Pixels with depth less than minDepth will not be used
     * @param maxDepth Pixels with depth larger than maxDepth will not be used
     * @param maxDepthDiff Correspondences between pixels of two given frames will be filtered out
     *                     if their depth difference is larger than maxDepthDiff
     * @param maxPointsPart The method uses a random pixels subset of size frameWidth x frameHeight x pointsPart
     * @param iterCounts Count of iterations on each pyramid level.
     * @param minGradientMagnitudes For each pyramid level the pixels will be filtered out
     *                              if they have gradient magnitude less than minGradientMagnitudes[level].
     * @param transformType Class of trasformation
     */
    RgbdICPOdometry(const Mat& cameraMatrix, float minDepth = Odometry::DEFAULT_MIN_DEPTH(), float maxDepth = Odometry::DEFAULT_MAX_DEPTH(),
                    float maxDepthDiff = Odometry::DEFAULT_MAX_DEPTH_DIFF(), float maxPointsPart = Odometry::DEFAULT_MAX_POINTS_PART(),
                    const std::vector<int>& iterCounts = std::vector<int>(),
                    const std::vector<float>& minGradientMagnitudes = std::vector<float>(),
                    int transformType = Odometry::RIGID_BODY_MOTION);

    CV_WRAP static Ptr<RgbdICPOdometry> create(const Mat& cameraMatrix = Mat(), float minDepth = Odometry::DEFAULT_MIN_DEPTH(), float maxDepth = Odometry::DEFAULT_MAX_DEPTH(),
                    float maxDepthDiff = Odometry::DEFAULT_MAX_DEPTH_DIFF(), float maxPointsPart = Odometry::DEFAULT_MAX_POINTS_PART(),
                    const std::vector<int>& iterCounts = std::vector<int>(),
                    const std::vector<float>& minGradientMagnitudes = std::vector<float>(),
                    int transformType = Odometry::RIGID_BODY_MOTION);

    CV_WRAP virtual Size prepareFrameCache(Ptr<OdometryFrame>& frame, int cacheType) const CV_OVERRIDE;

    CV_WRAP cv::Mat getCameraMatrix() const CV_OVERRIDE
    {
        return cameraMatrix;
    }
    CV_WRAP void setCameraMatrix(const cv::Mat &val) CV_OVERRIDE
    {
        cameraMatrix = val;
    }
    CV_WRAP double getMinDepth() const
    {
        return minDepth;
    }
    CV_WRAP void setMinDepth(double val)
    {
        minDepth = val;
    }
    CV_WRAP double getMaxDepth() const
    {
        return maxDepth;
    }
    CV_WRAP void setMaxDepth(double val)
    {
        maxDepth = val;
    }
    CV_WRAP double getMaxDepthDiff() const
    {
        return maxDepthDiff;
    }
    CV_WRAP void setMaxDepthDiff(double val)
    {
        maxDepthDiff = val;
    }
    CV_WRAP double getMaxPointsPart() const
    {
        return maxPointsPart;
    }
    CV_WRAP void setMaxPointsPart(double val)
    {
        maxPointsPart = val;
    }
    CV_WRAP cv::Mat getIterationCounts() const
    {
        return iterCounts;
    }
    CV_WRAP void setIterationCounts(const cv::Mat &val)
    {
        iterCounts = val;
    }
    CV_WRAP cv::Mat getMinGradientMagnitudes() const
    {
        return minGradientMagnitudes;
    }
    CV_WRAP void setMinGradientMagnitudes(const cv::Mat &val)
    {
        minGradientMagnitudes = val;
    }
    CV_WRAP int getTransformType() const CV_OVERRIDE
    {
        return transformType;
    }
    CV_WRAP void setTransformType(int val) CV_OVERRIDE
    {
        transformType = val;
    }
    CV_WRAP double getMaxTranslation() const
    {
        return maxTranslation;
    }
    CV_WRAP void setMaxTranslation(double val)
    {
        maxTranslation = val;
    }
    CV_WRAP double getMaxRotation() const
    {
        return maxRotation;
    }
    CV_WRAP void setMaxRotation(double val)
    {
        maxRotation = val;
    }
    CV_WRAP Ptr<RgbdNormals> getNormalsComputer() const
    {
        return normalsComputer;
    }

  protected:
    virtual void
    checkParams() const CV_OVERRIDE;

    virtual bool
    computeImpl(const Ptr<OdometryFrame>& srcFrame, const Ptr<OdometryFrame>& dstFrame, OutputArray Rt,
                const Mat& initRt) const CV_OVERRIDE;

    // Some params have commented desired type. It's due to AlgorithmInfo::addParams does not support it now.
    /*float*/
    double minDepth, maxDepth, maxDepthDiff;
    /*float*/
    double maxPointsPart;
    /*vector<int>*/
    Mat iterCounts;
    /*vector<float>*/
    Mat minGradientMagnitudes;

    Mat cameraMatrix;
    int transformType;

    double maxTranslation, maxRotation;

    mutable Ptr<RgbdNormals> normalsComputer;
  };

  /** A faster version of ICPOdometry which is used in KinectFusion implementation
   * Partial list of differences:
   * - Works in parallel
   * - Written in universal intrinsics
   * - Filters points by angle
   * - Interpolates points and normals
   * - Doesn't use masks or min/max depth filtering
   * - Doesn't use random subsets of points
   * - Supports only Rt transform type
   * - Supports only 4-float vectors as input type
   */
  class CV_EXPORTS_W FastICPOdometry: public Odometry
  {
  public:
    FastICPOdometry();
    /** Constructor.
     * @param cameraMatrix Camera matrix
     * @param maxDistDiff Correspondences between pixels of two given frames will be filtered out
     *                     if their depth difference is larger than maxDepthDiff
     * @param angleThreshold Correspondence will be filtered out
     *                     if an angle between their normals is bigger than threshold
     * @param sigmaDepth Depth sigma in meters for bilateral smooth
     * @param sigmaSpatial Spatial sigma in pixels for bilateral smooth
     * @param kernelSize Kernel size in pixels for bilateral smooth
     * @param iterCounts Count of iterations on each pyramid level
     */
    FastICPOdometry(const Mat& cameraMatrix,
                    float maxDistDiff = Odometry::DEFAULT_MAX_DEPTH_DIFF(),
                    float angleThreshold = (float)(30. * CV_PI / 180.),
                    float sigmaDepth = 0.04f,
                    float sigmaSpatial = 4.5f,
                    int kernelSize = 7,
                    const std::vector<int>& iterCounts = std::vector<int>());

    CV_WRAP static Ptr<FastICPOdometry> create(const Mat& cameraMatrix,
                                               float maxDistDiff = Odometry::DEFAULT_MAX_DEPTH_DIFF(),
                                               float angleThreshold = (float)(30. * CV_PI / 180.),
                                               float sigmaDepth = 0.04f,
                                               float sigmaSpatial = 4.5f,
                                               int kernelSize = 7,
                                               const std::vector<int>& iterCounts = std::vector<int>());

    CV_WRAP virtual Size prepareFrameCache(Ptr<OdometryFrame>& frame, int cacheType) const CV_OVERRIDE;

    CV_WRAP cv::Mat getCameraMatrix() const CV_OVERRIDE
    {
        return cameraMatrix;
    }
    CV_WRAP void setCameraMatrix(const cv::Mat &val) CV_OVERRIDE
    {
        cameraMatrix = val;
    }
    CV_WRAP double getMaxDistDiff() const
    {
        return maxDistDiff;
    }
    CV_WRAP void setMaxDistDiff(float val)
    {
        maxDistDiff = val;
    }
    CV_WRAP float getAngleThreshold() const
    {
        return angleThreshold;
    }
    CV_WRAP void setAngleThreshold(float f)
    {
        angleThreshold = f;
    }
    CV_WRAP float getSigmaDepth() const
    {
        return sigmaDepth;
    }
    CV_WRAP void setSigmaDepth(float f)
    {
        sigmaDepth = f;
    }
    CV_WRAP float getSigmaSpatial() const
    {
        return sigmaSpatial;
    }
    CV_WRAP void setSigmaSpatial(float f)
    {
        sigmaSpatial = f;
    }
    CV_WRAP int getKernelSize() const
    {
        return kernelSize;
    }
    CV_WRAP void setKernelSize(int f)
    {
        kernelSize = f;
    }
    CV_WRAP cv::Mat getIterationCounts() const
    {
        return iterCounts;
    }
    CV_WRAP void setIterationCounts(const cv::Mat &val)
    {
        iterCounts = val;
    }
    CV_WRAP int getTransformType() const CV_OVERRIDE
    {
        return Odometry::RIGID_BODY_MOTION;
    }
    CV_WRAP void setTransformType(int val) CV_OVERRIDE
    {
        if(val != Odometry::RIGID_BODY_MOTION)
            throw std::runtime_error("Rigid Body Motion is the only accepted transformation type"
                                     " for this odometry method");
    }

  protected:
    virtual void
    checkParams() const CV_OVERRIDE;

    virtual bool
    computeImpl(const Ptr<OdometryFrame>& srcFrame, const Ptr<OdometryFrame>& dstFrame, OutputArray Rt,
                const Mat& initRt) const CV_OVERRIDE;

    // Some params have commented desired type. It's due to AlgorithmInfo::addParams does not support it now.
    float maxDistDiff;

    float angleThreshold;

    float sigmaDepth;

    float sigmaSpatial;

    int kernelSize;

    /*vector<int>*/
    Mat iterCounts;

    Mat cameraMatrix;
  };

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
  CV_EXPORTS_W
  void
  warpFrame(const Mat& image, const Mat& depth, const Mat& mask, const Mat& Rt, const Mat& cameraMatrix,
            const Mat& distCoeff, OutputArray warpedImage, OutputArray warpedDepth = noArray(), OutputArray warpedMask = noArray());

// TODO Depth interpolation
// Curvature
// Get rescaleDepth return dubles if asked for

//! @}

} /* namespace rgbd */
} /* namespace cv */

#endif

/* End of file. */
