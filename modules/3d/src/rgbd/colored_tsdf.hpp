// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this
// module's directory

#ifndef __OPENCV_KINFU_COLORED_TSDF_H__
#define __OPENCV_KINFU_COLORED_TSDF_H__

#include <opencv2/rgbd/volume.hpp>

#include "kinfu_frame.hpp"
#include "utils.hpp"

namespace cv
{
namespace kinfu
{

typedef int8_t TsdfType;
typedef uchar WeightType;
typedef short int ColorType;
struct RGBTsdfVoxel
{
    RGBTsdfVoxel(TsdfType _tsdf, WeightType _weight, ColorType _r, ColorType _g, ColorType _b) :
        tsdf(_tsdf), weight(_weight), r(_r), g(_g), b(_b)
    { }
    TsdfType tsdf;
    WeightType weight;
    ColorType r, g, b;
};

typedef Vec<uchar, sizeof(RGBTsdfVoxel)> VecRGBTsdfVoxel;

class ColoredTSDFVolume : public Volume
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

}  // namespace kinfu
}  // namespace cv
#endif
