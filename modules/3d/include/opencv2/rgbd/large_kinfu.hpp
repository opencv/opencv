// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this
// module's directory

#ifndef __OPENCV_RGBD_LARGEKINFU_HPP__
#define __OPENCV_RGBD_LARGEKINFU_HPP__

#include <opencv2/rgbd/volume.hpp>

#include "opencv2/core.hpp"
#include "opencv2/core/affine.hpp"

namespace cv
{
namespace large_kinfu
{
struct CV_EXPORTS_W Params
{
    /** @brief Default parameters
        A set of parameters which provides better model quality, can be very slow.
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
    CV_PROP_RW int bilateral_kernel_size;

    /** @brief Number of pyramid levels for ICP */
    CV_PROP_RW int pyramidLevels;

    /** @brief Minimal camera movement in meters
        Integrate new depth frame only if camera movement exceeds this value.
    */
    CV_PROP_RW float tsdf_min_camera_movement;

    /** @brief light pose for rendering in meters */
    CV_PROP_RW Vec3f lightPose;

    /** @brief distance theshold for ICP in meters */
    CV_PROP_RW float icpDistThresh;
    /** @brief angle threshold for ICP in radians */
    CV_PROP_RW float icpAngleThresh;
    /** @brief number of ICP iterations for each pyramid level */
    CV_PROP_RW std::vector<int> icpIterations;

    /** @brief Threshold for depth truncation in meters
        All depth values beyond this threshold will be set to zero
    */
    CV_PROP_RW float truncateThreshold;

    /** @brief Volume parameters
    */
    kinfu::VolumeParams volumeParams;
};

/** @brief Large Scale Dense Depth Fusion implementation

  This class implements a 3d reconstruction algorithm for larger environments using
  Spatially hashed TSDF volume "Submaps".
  It also runs a periodic posegraph optimization to minimize drift in tracking over long sequences.
  Currently the algorithm does not implement a relocalization or loop closure module.
  Potentially a Bag of words implementation or RGBD relocalization as described in
  Glocker et al. ISMAR 2013 will be implemented

  It takes a sequence of depth images taken from depth sensor
  (or any depth images source such as stereo camera matching algorithm or even raymarching
  renderer). The output can be obtained as a vector of points and their normals or can be
  Phong-rendered from given camera pose.

  An internal representation of a model is a spatially hashed voxel cube that stores TSDF values
  which represent the distance to the closest surface (for details read the @cite kinectfusion article
  about TSDF). There is no interface to that representation yet.

  For posegraph optimization, a Submap abstraction over the Volume class is created.
  New submaps are added to the model when there is low visibility overlap between current viewing frustrum
  and the existing volume/model. Multiple submaps are simultaneously tracked and a posegraph is created and
  optimized periodically.

  LargeKinfu does not use any OpenCL acceleration yet.
  To enable or disable it explicitly use cv::setUseOptimized() or cv::ocl::setUseOpenCL().

  This implementation is inspired from Kintinuous, InfiniTAM and other SOTA algorithms

  You need to set the OPENCV_ENABLE_NONFREE option in CMake to use KinectFusion.
*/
class CV_EXPORTS_W LargeKinfu
{
   public:
    CV_WRAP static Ptr<LargeKinfu> create(const Ptr<Params>& _params);
    virtual ~LargeKinfu() = default;

    virtual const Params& getParams() const = 0;

    CV_WRAP virtual void render(OutputArray image,
                                const Matx44f& cameraPose = Matx44f::eye()) const = 0;

    CV_WRAP virtual void getCloud(OutputArray points, OutputArray normals) const = 0;

    CV_WRAP virtual void getPoints(OutputArray points) const = 0;

    CV_WRAP virtual void getNormals(InputArray points, OutputArray normals) const = 0;

    CV_WRAP virtual void reset() = 0;

    virtual const Affine3f getPose() const = 0;

    CV_WRAP virtual bool update(InputArray depth) = 0;
};

}  // namespace large_kinfu
}  // namespace cv
#endif
