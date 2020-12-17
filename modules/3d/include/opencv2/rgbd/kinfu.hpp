// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

#ifndef __OPENCV_RGBD_KINFU_HPP__
#define __OPENCV_RGBD_KINFU_HPP__

#include "opencv2/core.hpp"
#include "opencv2/core/affine.hpp"
#include <opencv2/rgbd/volume.hpp>

namespace cv {
namespace kinfu {
//! @addtogroup kinect_fusion
//! @{

struct CV_EXPORTS_W Params
{

    CV_WRAP Params(){}

    /**
     * @brief Constructor for Params
     * Sets the initial pose of the TSDF volume.
     * @param volumeInitialPoseRot rotation matrix
     * @param volumeInitialPoseTransl translation vector
     */
    CV_WRAP Params(Matx33f volumeInitialPoseRot, Vec3f volumeInitialPoseTransl)
    {
      setInitialVolumePose(volumeInitialPoseRot,volumeInitialPoseTransl);
    }

    /**
     * @brief Constructor for Params
     * Sets the initial pose of the TSDF volume.
     * @param volumeInitialPose 4 by 4 Homogeneous Transform matrix to set the intial pose of TSDF volume
     */
    CV_WRAP Params(Matx44f volumeInitialPose)
    {
      setInitialVolumePose(volumeInitialPose);
    }

    /**
     * @brief Set Initial Volume Pose
     * Sets the initial pose of the TSDF volume.
     * @param R rotation matrix
     * @param t translation vector
     */
    CV_WRAP void setInitialVolumePose(Matx33f R, Vec3f t);

    /**
     * @brief Set Initial Volume Pose
     * Sets the initial pose of the TSDF volume.
     * @param homogen_tf 4 by 4 Homogeneous Transform matrix to set the intial pose of TSDF volume
     */
    CV_WRAP void setInitialVolumePose(Matx44f homogen_tf);

    /**
     * @brief Default parameters
     * A set of parameters which provides better model quality, can be very slow.
     */
    CV_WRAP static Ptr<Params> defaultParams();

    /** @brief Coarse parameters
    A set of parameters which provides better speed, can fail to match frames
    in case of rapid sensor motion.
    */
    CV_WRAP static Ptr<Params> coarseParams();

    /** @brief HashTSDF parameters
      A set of parameters suitable for use with HashTSDFVolume
    */
    CV_WRAP static Ptr<Params> hashTSDFParams(bool isCoarse);

    /** @brief frame size in pixels */
    CV_PROP_RW Size frameSize;

    CV_PROP_RW kinfu::VolumeType volumeType;

    /** @brief camera intrinsics */
    CV_PROP_RW Matx33f intr;

    /** @brief pre-scale per 1 meter for input values

    Typical values are:
         * 5000 per 1 meter for the 16-bit PNG files of TUM database
         * 1000 per 1 meter for Kinect 2 device
         * 1 per 1 meter for the 32-bit float images in the ROS bag files
    */
    CV_PROP_RW float depthFactor;

    /** @brief Depth sigma in meters for bilateral smooth */
    CV_PROP_RW float bilateral_sigma_depth;
    /** @brief Spatial sigma in pixels for bilateral smooth */
    CV_PROP_RW float bilateral_sigma_spatial;
    /** @brief Kernel size in pixels for bilateral smooth */
    CV_PROP_RW int   bilateral_kernel_size;

    /** @brief Number of pyramid levels for ICP */
    CV_PROP_RW int pyramidLevels;

    /** @brief Resolution of voxel space

    Number of voxels in each dimension.
    */
    CV_PROP_RW Vec3i volumeDims;
    /** @brief Size of voxel in meters */
    CV_PROP_RW float voxelSize;

    /** @brief Minimal camera movement in meters

    Integrate new depth frame only if camera movement exceeds this value.
    */
    CV_PROP_RW float tsdf_min_camera_movement;

    /** @brief initial volume pose in meters */
    Affine3f volumePose;

    /** @brief distance to truncate in meters

    Distances to surface that exceed this value will be truncated to 1.0.
    */
    CV_PROP_RW float tsdf_trunc_dist;

    /** @brief max number of frames per voxel

    Each voxel keeps running average of distances no longer than this value.
    */
    CV_PROP_RW int tsdf_max_weight;

    /** @brief A length of one raycast step

    How much voxel sizes we skip each raycast step
    */
    CV_PROP_RW float raycast_step_factor;

    // gradient delta in voxel sizes
    // fixed at 1.0f
    // float gradient_delta_factor;

    /** @brief light pose for rendering in meters */
    CV_PROP_RW Vec3f lightPose;

    /** @brief distance theshold for ICP in meters */
    CV_PROP_RW float icpDistThresh;
    /** angle threshold for ICP in radians */
    CV_PROP_RW float icpAngleThresh;
    /** number of ICP iterations for each pyramid level */
    CV_PROP_RW std::vector<int> icpIterations;

    /** @brief Threshold for depth truncation in meters

    All depth values beyond this threshold will be set to zero
    */
    CV_PROP_RW float truncateThreshold;
};

/** @brief KinectFusion implementation

  This class implements a 3d reconstruction algorithm described in
  @cite kinectfusion paper.

  It takes a sequence of depth images taken from depth sensor
  (or any depth images source such as stereo camera matching algorithm or even raymarching renderer).
  The output can be obtained as a vector of points and their normals
  or can be Phong-rendered from given camera pose.

  An internal representation of a model is a voxel cuboid that keeps TSDF values
  which are a sort of distances to the surface (for details read the @cite kinectfusion article about TSDF).
  There is no interface to that representation yet.

  KinFu uses OpenCL acceleration automatically if available.
  To enable or disable it explicitly use cv::setUseOptimized() or cv::ocl::setUseOpenCL().

  This implementation is based on [kinfu-remake](https://github.com/Nerei/kinfu_remake).

  Note that the KinectFusion algorithm was patented and its use may be restricted by
  the list of patents mentioned in README.md file in this module directory.

  That's why you need to set the OPENCV_ENABLE_NONFREE option in CMake to use KinectFusion.
*/
class CV_EXPORTS_W KinFu
{
public:
    CV_WRAP static Ptr<KinFu> create(const Ptr<Params>& _params);
    virtual ~KinFu();

    /** @brief Get current parameters */
    virtual const Params& getParams() const = 0;

    /** @brief Renders a volume into an image

      Renders a 0-surface of TSDF using Phong shading into a CV_8UC4 Mat.
      Light pose is fixed in KinFu params.

        @param image resulting image
        @param cameraPose pose of camera to render from. If empty then render from current pose
        which is a last frame camera pose.
    */

    CV_WRAP virtual void render(OutputArray image, const Matx44f& cameraPose = Matx44f::eye()) const = 0;

    /** @brief Gets points and normals of current 3d mesh

      The order of normals corresponds to order of points.
      The order of points is undefined.

        @param points vector of points which are 4-float vectors
        @param normals vector of normals which are 4-float vectors
     */
    CV_WRAP virtual void getCloud(OutputArray points, OutputArray normals) const = 0;

    /** @brief Gets points of current 3d mesh

     The order of points is undefined.

        @param points vector of points which are 4-float vectors
     */
    CV_WRAP virtual void getPoints(OutputArray points) const = 0;

    /** @brief Calculates normals for given points
        @param points input vector of points which are 4-float vectors
        @param normals output vector of corresponding normals which are 4-float vectors
     */
    CV_WRAP virtual  void getNormals(InputArray points, OutputArray normals) const = 0;

    /** @brief Resets the algorithm

    Clears current model and resets a pose.
    */
    CV_WRAP virtual void reset() = 0;

    /** @brief Get current pose in voxel space */
    virtual const Affine3f getPose() const = 0;

    /** @brief Process next depth frame

      Integrates depth into voxel space with respect to its ICP-calculated pose.
      Input image is converted to CV_32F internally if has another type.

    @param depth one-channel image which size and depth scale is described in algorithm's parameters
    @return true if succeeded to align new frame with current scene, false if opposite
    */
    CV_WRAP virtual bool update(InputArray depth) = 0;
};

//! @}
}
}
#endif
