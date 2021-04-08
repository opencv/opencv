// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef __OPENCV_HASH_TSDF_H__
#define __OPENCV_HASH_TSDF_H__

#include <opencv2/rgbd/volume.hpp>
#include <unordered_map>
#include <unordered_set>

#include "tsdf_functions.hpp"

namespace cv
{
namespace kinfu
{
class HashTSDFVolume : public Volume
{
   public:
    // dimension in voxels, size in meters
    //! Use fixed volume cuboid
    HashTSDFVolume(float _voxelSize, cv::Matx44f _pose, float _raycastStepFactor, float _truncDist,
                   int _maxWeight, float _truncateThreshold, int _volumeUnitRes,
                   bool zFirstMemOrder = true);

    virtual ~HashTSDFVolume() = default;

    virtual int getVisibleBlocks(int currFrameId, int frameThreshold) const = 0;
    virtual size_t getTotalVolumeUnits() const = 0;

   public:
    int maxWeight;
    float truncDist;
    float truncateThreshold;
    int volumeUnitResolution;
    int volumeUnitDegree;
    float volumeUnitSize;
    bool zFirstMemOrder;
    Vec4i volStrides;
};

//template<typename T>
Ptr<HashTSDFVolume> makeHashTSDFVolume(const VolumeParams& _volumeParams);
//template<typename T>
Ptr<HashTSDFVolume> makeHashTSDFVolume(float _voxelSize, Matx44f _pose, float _raycastStepFactor, float _truncDist,
    int _maxWeight, float truncateThreshold, int volumeUnitResolution = 16);

}  // namespace kinfu
}  // namespace cv
#endif
