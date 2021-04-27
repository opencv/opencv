// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this
// module's directory

#ifndef __OPENCV_KINFU_TSDF_H__
#define __OPENCV_KINFU_TSDF_H__

#include <opencv2/rgbd/volume.hpp>

#include "kinfu_frame.hpp"
#include "utils.hpp"

namespace cv
{
namespace kinfu
{

typedef int8_t TsdfType;
typedef uchar WeightType;

struct TsdfVoxel
{
    TsdfVoxel(TsdfType _tsdf, WeightType _weight) :
        tsdf(_tsdf), weight(_weight)
    { }
    TsdfType tsdf;
    WeightType weight;
};

typedef Vec<uchar, sizeof(TsdfVoxel)> VecTsdfVoxel;

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

class TSDFVolumeCPU : public TSDFVolume
{
   public:
    // dimension in voxels, size in meters
    TSDFVolumeCPU(float _voxelSize, cv::Matx44f _pose, float _raycastStepFactor, float _truncDist,
                  int _maxWeight, Vec3i _resolution, bool zFirstMemOrder = true);

    virtual void integrate(InputArray _depth, float depthFactor, const Matx44f& cameraPose,
                           const kinfu::Intr& intrinsics, const int frameId = 0) override;
    virtual void raycast(const Matx44f& cameraPose, const kinfu::Intr& intrinsics, const Size& frameSize,
                         OutputArray points, OutputArray normals) const override;
    virtual void integrate(InputArray, InputArray, float, const Matx44f&, const kinfu::Intr&, const Intr&, const int) override
        { CV_Error(Error::StsNotImplemented, "Not implemented"); };
    virtual void raycast(const Matx44f&, const kinfu::Intr&, const Size&, OutputArray, OutputArray, OutputArray) const override
        { CV_Error(Error::StsNotImplemented, "Not implemented"); };

    virtual void fetchNormals(InputArray points, OutputArray _normals) const override;
    virtual void fetchPointsNormals(OutputArray points, OutputArray normals) const override;

    virtual void reset() override;
    virtual TsdfVoxel at(const Vec3i& volumeIdx) const;

    float interpolateVoxel(const cv::Point3f& p) const;
    Point3f getNormalVoxel(const cv::Point3f& p) const;

#if USE_INTRINSICS
    float interpolateVoxel(const v_float32x4& p) const;
    v_float32x4 getNormalVoxel(const v_float32x4& p) const;
#endif
    Vec4i volStrides;
    Vec6f frameParams;
    Mat pixNorms;
    // See zFirstMemOrder arg of parent class constructor
    // for the array layout info
    // Consist of Voxel elements
    Mat volume;
};

#ifdef HAVE_OPENCL
class TSDFVolumeGPU : public TSDFVolume
{
   public:
    // dimension in voxels, size in meters
    TSDFVolumeGPU(float _voxelSize, Matx44f _pose, float _raycastStepFactor, float _truncDist,
                  int _maxWeight, Point3i _resolution);

    virtual void integrate(InputArray _depth, float depthFactor, const Matx44f& cameraPose,
                           const kinfu::Intr& intrinsics, const int frameId = 0) override;
    virtual void raycast(const Matx44f& cameraPose, const kinfu::Intr& intrinsics, const Size& frameSize,
                         OutputArray _points, OutputArray _normals) const override;
    virtual void integrate(InputArray, InputArray, float, const Matx44f&, const kinfu::Intr&, const Intr&, const int) override
        { CV_Error(Error::StsNotImplemented, "Not implemented"); };
    virtual void raycast(const Matx44f&, const kinfu::Intr&, const Size&, OutputArray, OutputArray, OutputArray) const override
        { CV_Error(Error::StsNotImplemented, "Not implemented"); };

    virtual void fetchPointsNormals(OutputArray points, OutputArray normals) const override;
    virtual void fetchNormals(InputArray points, OutputArray normals) const override;

    virtual void reset() override;

    Vec6f frameParams;
    UMat pixNorms;
    // See zFirstMemOrder arg of parent class constructor
    // for the array layout info
    // Array elem is CV_8UC2, read as (int8, uint8)
    UMat volume;
};
#endif
Ptr<TSDFVolume> makeTSDFVolume(float _voxelSize, Matx44f _pose, float _raycastStepFactor,
                               float _truncDist, int _maxWeight, Point3i _resolution);
Ptr<TSDFVolume> makeTSDFVolume(const VolumeParams& _params);
}  // namespace kinfu
}  // namespace cv
#endif
