// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this
// module's directory

#ifndef __OPENCV_RGBD_VOLUME_H__
#define __OPENCV_RGBD_VOLUME_H__

#include "intrinsics.hpp"
#include "opencv2/core/affine.hpp"

namespace cv
{
namespace kinfu
{
class CV_EXPORTS_W Volume
{
   public:
    Volume(float _voxelSize, Matx44f _pose, float _raycastStepFactor)
        : voxelSize(_voxelSize),
          voxelSizeInv(1.0f / voxelSize),
          pose(_pose),
          raycastStepFactor(_raycastStepFactor)
    {
    }

    virtual ~Volume(){};

    virtual void integrate(InputArray _depth, float depthFactor, const Matx44f& cameraPose,
                           const kinfu::Intr& intrinsics, const int frameId = 0)               = 0;
    virtual void integrate(InputArray _depth, InputArray _rgb, float depthFactor,
                           const Matx44f& cameraPose, const kinfu::Intr& intrinsics,
                           const Intr& rgb_intrinsics, const int frameId = 0)                  = 0;
    virtual void raycast(const Matx44f& cameraPose, const kinfu::Intr& intrinsics,
                         const Size& frameSize, OutputArray points, OutputArray normals) const = 0;
    virtual void raycast(const Matx44f& cameraPose, const kinfu::Intr& intrinsics, const Size& frameSize,
                         OutputArray points, OutputArray normals, OutputArray colors) const    = 0;
    virtual void fetchNormals(InputArray points, OutputArray _normals) const                   = 0;
    virtual void fetchPointsNormals(OutputArray points, OutputArray normals) const             = 0;
    virtual void reset()                                                                       = 0;

   public:
    const float voxelSize;
    const float voxelSizeInv;
    const Affine3f pose;
    const float raycastStepFactor;
};

enum class VolumeType
{
    TSDF        = 0,
    HASHTSDF    = 1,
    COLOREDTSDF = 2
};

struct CV_EXPORTS_W VolumeParams
{
    /** @brief Type of Volume
        Values can be TSDF (single volume) or HASHTSDF (hashtable of volume units)
    */
    CV_PROP_RW VolumeType type;

    /** @brief Resolution of voxel space
        Number of voxels in each dimension.
        Applicable only for TSDF Volume.
        HashTSDF volume only supports equal resolution in all three dimensions
    */
    CV_PROP_RW Vec3i resolution;

    /** @brief Resolution of volumeUnit in voxel space
        Number of voxels in each dimension for volumeUnit
        Applicable only for hashTSDF.
    */
    CV_PROP_RW int unitResolution = {0};

    /** @brief Initial pose of the volume in meters */
    Affine3f pose;

    /** @brief Length of voxels in meters */
    CV_PROP_RW float voxelSize;

    /** @brief TSDF truncation distance
        Distances greater than value from surface will be truncated to 1.0
    */
    CV_PROP_RW float tsdfTruncDist;

    /** @brief Max number of frames to integrate per voxel
        Represents the max number of frames over which a running average
        of the TSDF is calculated for a voxel
    */
    CV_PROP_RW int maxWeight;

    /** @brief Threshold for depth truncation in meters
        Truncates the depth greater than threshold to 0
    */
    CV_PROP_RW float depthTruncThreshold;

    /** @brief Length of single raycast step
        Describes the percentage of voxel length that is skipped per march
    */
    CV_PROP_RW float raycastStepFactor;

    /** @brief Default set of parameters that provide higher quality reconstruction
        at the cost of slow performance.
    */
    CV_WRAP static Ptr<VolumeParams> defaultParams(VolumeType _volumeType);

    /** @brief Coarse set of parameters that provides relatively higher performance
        at the cost of reconstrution quality.
    */
    CV_WRAP static Ptr<VolumeParams> coarseParams(VolumeType _volumeType);
};


Ptr<Volume> makeVolume(const VolumeParams& _volumeParams);
CV_EXPORTS_W Ptr<Volume> makeVolume(VolumeType _volumeType, float _voxelSize, Matx44f _pose,
                                    float _raycastStepFactor, float _truncDist, int _maxWeight,
                                    float _truncateThreshold, Vec3i _resolution);

}  // namespace kinfu
}  // namespace cv
#endif
