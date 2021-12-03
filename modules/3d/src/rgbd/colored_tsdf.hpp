// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef OPENCV_3D_COLORED_TSDF_HPP
#define OPENCV_3D_COLORED_TSDF_HPP

#include "../precomp.hpp"
#include "tsdf_functions.hpp"

namespace cv
{

class ColoredTSDFVolume : public _Volume
{
   public:
    // dimension in voxels, size in meters
    ColoredTSDFVolume(float _voxelSize, Matx44f _pose, float _raycastStepFactor, float _truncDist,
                      int _maxWeight, Point3i _resolution, bool zFirstMemOrder = true);
    virtual ~ColoredTSDFVolume() = default;

   public:

    Point3i volResolution;
    WeightType maxWeight;

    Point3f volSize;
    float truncDist;
    Vec4i volDims;
    Vec8i neighbourCoords;
};

Ptr<ColoredTSDFVolume> makeColoredTSDFVolume(float _voxelSize, Matx44f _pose, float _raycastStepFactor,
                                             float _truncDist, int _maxWeight, Point3i _resolution);
Ptr<ColoredTSDFVolume> makeColoredTSDFVolume(const VolumeParams& _params);

}  // namespace cv

#endif // include guard
