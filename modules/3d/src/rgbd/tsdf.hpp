// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// Partially rewritten from https://github.com/Nerei/kinfu_remake
// Copyright(c) 2012, Anatoly Baksheev. All rights reserved.

#ifndef __OPENCV_KINFU_TSDF_H__
#define __OPENCV_KINFU_TSDF_H__

#include "../precomp.hpp"
#include "tsdf_functions.hpp"

namespace cv
{

class TSDFVolume : public Volume
{
   public:
    // dimension in voxels, size in meters
    TSDFVolume(float _voxelSize, Matx44f _pose, float _raycastStepFactor, float _truncDist,
               int _maxWeight, Point3i _resolution, bool zFirstMemOrder = true);
    virtual ~TSDFVolume() = default;

   public:

    Point3i volResolution;
    WeightType maxWeight;

    Point3f volSize;
    float truncDist;
    Vec4i volDims;
    Vec8i neighbourCoords;
};

Ptr<TSDFVolume> makeTSDFVolume(float _voxelSize, Matx44f _pose, float _raycastStepFactor,
                               float _truncDist, int _maxWeight, Point3i _resolution);
Ptr<TSDFVolume> makeTSDFVolume(const VolumeParams& _params);
}  // namespace cv
#endif
