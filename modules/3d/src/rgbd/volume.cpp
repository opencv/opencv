// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "../precomp.hpp"
#include "tsdf.hpp"
#include "hash_tsdf.hpp"
#include "colored_tsdf.hpp"

namespace cv
{

Ptr<VolumeParams> VolumeParams::defaultParams(int _volumeKind)
{
    VolumeParams params;
    params.kind              = _volumeKind;
    params.maxWeight         = 64;
    params.raycastStepFactor = 0.25f;
    params.unitResolution    = 0;  // unitResolution not used for TSDF
    float volumeSize         = 3.0f;
    Matx44f pose = Affine3f().translate(Vec3f(-volumeSize / 2.f, -volumeSize / 2.f, 0.5f)).matrix;
    params.pose = Mat(pose);

    if(params.kind == VolumeKind::TSDF)
    {
        params.resolutionX = 512;
        params.resolutionY = 512;
        params.resolutionZ = 512;
        params.voxelSize           = volumeSize / 512.f;
        params.depthTruncThreshold = 0.f;  // depthTruncThreshold not required for TSDF
        params.tsdfTruncDist = 7 * params.voxelSize;  //! About 0.04f in meters
        return makePtr<VolumeParams>(params);
    }
    else if(params.kind == VolumeKind::HASHTSDF)
    {
        params.unitResolution      = 16;
        params.voxelSize           = volumeSize / 512.f;
        params.depthTruncThreshold = Odometry::DEFAULT_MAX_DEPTH();
        params.tsdfTruncDist = 7 * params.voxelSize;  //! About 0.04f in meters
        return makePtr<VolumeParams>(params);
    }
    else if (params.kind == VolumeKind::COLOREDTSDF)
    {
        params.resolutionX = 512;
        params.resolutionY = 512;
        params.resolutionZ = 512;
        params.voxelSize = volumeSize / 512.f;
        params.depthTruncThreshold = 0.f;  // depthTruncThreshold not required for TSDF
        params.tsdfTruncDist = 7 * params.voxelSize;  //! About 0.04f in meters
        return makePtr<VolumeParams>(params);
    }
    CV_Error(Error::StsBadArg, "Invalid VolumeType does not have parameters");
}

Ptr<VolumeParams> VolumeParams::coarseParams(int _volumeKind)
{
    Ptr<VolumeParams> params = defaultParams(_volumeKind);

    params->raycastStepFactor = 0.75f;
    float volumeSize          = 3.0f;
    if(params->kind == VolumeKind::TSDF)
    {
        params->resolutionX = 128;
        params->resolutionY = 128;
        params->resolutionZ = 128;
        params->voxelSize  = volumeSize / 128.f;
        params->tsdfTruncDist = 2 * params->voxelSize;  //! About 0.04f in meters
        return params;
    }
    else if(params->kind == VolumeKind::HASHTSDF)
    {
        params->voxelSize = volumeSize / 128.f;
        params->tsdfTruncDist = 2 * params->voxelSize;  //! About 0.04f in meters
        return params;
    }
    else if (params->kind == VolumeKind::COLOREDTSDF)
    {
        params->resolutionX = 128;
        params->resolutionY = 128;
        params->resolutionZ = 128;
        params->voxelSize = volumeSize / 128.f;
        params->tsdfTruncDist = 2 * params->voxelSize;  //! About 0.04f in meters
        return params;
    }
    CV_Error(Error::StsBadArg, "Invalid VolumeType does not have parameters");
}

Ptr<Volume> makeVolume(const Ptr<VolumeParams>& _volumeParams)
{
    int kind = _volumeParams->kind;
    if(kind == VolumeParams::VolumeKind::TSDF)
        return makeTSDFVolume(*_volumeParams);
    else if(kind == VolumeParams::VolumeKind::HASHTSDF)
        return makeHashTSDFVolume(*_volumeParams);
    else if(kind == VolumeParams::VolumeKind::COLOREDTSDF)
        return makeColoredTSDFVolume(*_volumeParams);
    CV_Error(Error::StsBadArg, "Invalid VolumeType does not have parameters");
}

Ptr<Volume> makeVolume(int _volumeKind, float _voxelSize, Matx44f _pose,
                       float _raycastStepFactor, float _truncDist, int _maxWeight, float _truncateThreshold,
                       int _resolutionX, int _resolutionY, int _resolutionZ)
{
    Point3i _presolution(_resolutionX, _resolutionY, _resolutionZ);
    if (_volumeKind == VolumeParams::VolumeKind::TSDF)
    {
        return makeTSDFVolume(_voxelSize, _pose, _raycastStepFactor, _truncDist, _maxWeight, _presolution);
    }
    else if (_volumeKind == VolumeParams::VolumeKind::HASHTSDF)
    {
        return makeHashTSDFVolume(_voxelSize, _pose, _raycastStepFactor, _truncDist, _maxWeight, _truncateThreshold);
    }
    else if (_volumeKind == VolumeParams::VolumeKind::COLOREDTSDF)
    {
        return makeColoredTSDFVolume(_voxelSize, _pose, _raycastStepFactor, _truncDist, _maxWeight, _presolution);
    }
    CV_Error(Error::StsBadArg, "Invalid VolumeType does not have parameters");
}

}  // namespace cv
