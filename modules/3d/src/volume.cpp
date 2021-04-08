// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include <opencv2/rgbd/volume.hpp>

#include "opencv2/core/base.hpp"
#include "precomp.hpp"
#include "tsdf.hpp"
#include "hash_tsdf.hpp"
#include "colored_tsdf.hpp"

namespace cv
{
namespace kinfu
{
Ptr<VolumeParams> VolumeParams::defaultParams(VolumeType _volumeType)
{
    VolumeParams params;
    params.type              = _volumeType;
    params.maxWeight         = 64;
    params.raycastStepFactor = 0.25f;
    params.unitResolution    = 0;  // unitResolution not used for TSDF
    float volumeSize         = 3.0f;
    params.pose = Affine3f().translate(Vec3f(-volumeSize / 2.f, -volumeSize / 2.f, 0.5f));
    if(params.type == VolumeType::TSDF)
    {
        params.resolution          = Vec3i::all(512);
        params.voxelSize           = volumeSize / 512.f;
        params.depthTruncThreshold = 0.f;  // depthTruncThreshold not required for TSDF
        params.tsdfTruncDist = 7 * params.voxelSize;  //! About 0.04f in meters
        return makePtr<VolumeParams>(params);
    }
    else if(params.type == VolumeType::HASHTSDF)
    {
        params.unitResolution      = 16;
        params.voxelSize           = volumeSize / 512.f;
        params.depthTruncThreshold = rgbd::Odometry::DEFAULT_MAX_DEPTH();
        params.tsdfTruncDist = 7 * params.voxelSize;  //! About 0.04f in meters
        return makePtr<VolumeParams>(params);
    }
    else if (params.type == VolumeType::COLOREDTSDF)
    {
        params.resolution = Vec3i::all(512);
        params.voxelSize = volumeSize / 512.f;
        params.depthTruncThreshold = 0.f;  // depthTruncThreshold not required for TSDF
        params.tsdfTruncDist = 7 * params.voxelSize;  //! About 0.04f in meters
        return makePtr<VolumeParams>(params);
    }
    CV_Error(Error::StsBadArg, "Invalid VolumeType does not have parameters");
}

Ptr<VolumeParams> VolumeParams::coarseParams(VolumeType _volumeType)
{
    Ptr<VolumeParams> params = defaultParams(_volumeType);

    params->raycastStepFactor = 0.75f;
    float volumeSize          = 3.0f;
    if(params->type == VolumeType::TSDF)
    {
        params->resolution = Vec3i::all(128);
        params->voxelSize  = volumeSize / 128.f;
        params->tsdfTruncDist = 2 * params->voxelSize;  //! About 0.04f in meters
        return params;
    }
    else if(params->type == VolumeType::HASHTSDF)
    {
        params->voxelSize = volumeSize / 128.f;
        params->tsdfTruncDist = 2 * params->voxelSize;  //! About 0.04f in meters
        return params;
    }
    else if (params->type == VolumeType::COLOREDTSDF)
    {
        params->resolution = Vec3i::all(128);
        params->voxelSize = volumeSize / 128.f;
        params->tsdfTruncDist = 2 * params->voxelSize;  //! About 0.04f in meters
        return params;
    }
    CV_Error(Error::StsBadArg, "Invalid VolumeType does not have parameters");
}

Ptr<Volume> makeVolume(const VolumeParams& _volumeParams)
{
    if(_volumeParams.type == VolumeType::TSDF)
        return kinfu::makeTSDFVolume(_volumeParams);
    else if(_volumeParams.type == VolumeType::HASHTSDF)
        return kinfu::makeHashTSDFVolume(_volumeParams);
    else if(_volumeParams.type == VolumeType::COLOREDTSDF)
        return kinfu::makeColoredTSDFVolume(_volumeParams);
    CV_Error(Error::StsBadArg, "Invalid VolumeType does not have parameters");
}

Ptr<Volume> makeVolume(VolumeType _volumeType, float _voxelSize, Matx44f _pose,
                       float _raycastStepFactor, float _truncDist, int _maxWeight,
                       float _truncateThreshold, Vec3i _resolution)
{
    Point3i _presolution = _resolution;
    if (_volumeType == VolumeType::TSDF)
    {
        return makeTSDFVolume(_voxelSize, _pose, _raycastStepFactor, _truncDist, _maxWeight, _presolution);
    }
    else if (_volumeType == VolumeType::HASHTSDF)
    {
        return makeHashTSDFVolume(_voxelSize, _pose, _raycastStepFactor, _truncDist, _maxWeight, _truncateThreshold);
    }
    else if (_volumeType == VolumeType::COLOREDTSDF)
    {
        return makeColoredTSDFVolume(_voxelSize, _pose, _raycastStepFactor, _truncDist, _maxWeight, _presolution);
    }
    CV_Error(Error::StsBadArg, "Invalid VolumeType does not have parameters");
}

}  // namespace kinfu
}  // namespace cv
