// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// Partially rewritten from https://github.com/Nerei/kinfu_remake
// Copyright(c) 2012, Anatoly Baksheev. All rights reserved.

#ifndef OPENCV_3D_TSDF_FUNCTIONS_HPP
#define OPENCV_3D_TSDF_FUNCTIONS_HPP

#include "../precomp.hpp"
#include "utils.hpp"

namespace cv
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

#if CV_SIMD128
inline v_float32x4 tsdfToFloat_INTR(const v_int32x4& num)
{
    v_float32x4 num128 = v_setall_f32(-1.f / 128.f);
    return v_mul(v_cvt_f32(num), num128);
}
#endif

inline TsdfType floatToTsdf(float num)
{
    //CV_Assert(-1 < num <= 1);
    int8_t res = int8_t(num * (-128.f));
    res = res ? res : (num < 0 ? 1 : -1);
    return res;
}

inline float tsdfToFloat(TsdfType num)
{
    return float(num) * (-1.f / 128.f);
}

inline void colorFix(ColorType& r, ColorType& g, ColorType&b)
{
    if (r > 255) r = 255;
    if (g > 255) g = 255;
    if (b > 255) b = 255;
}

inline void colorFix(Point3f& c)
{
    if (c.x > 255) c.x = 255;
    if (c.y > 255) c.y = 255;
    if (c.z > 255) c.z = 255;
}

void preCalculationPixNorm(Size size, const Intr& intrinsics, Mat& pixNorm);
#ifdef HAVE_OPENCL
void ocl_preCalculationPixNorm(Size size, const Intr& intrinsics, UMat& pixNorm);
#endif

inline depthType bilinearDepth(const Depth& m, cv::Point2f pt)
{
    const bool fixMissingData = false;
    const depthType defaultValue = qnan;
    if (pt.x < 0 || pt.x >= m.cols - 1 ||
        pt.y < 0 || pt.y >= m.rows - 1)
        return defaultValue;

    int xi = cvFloor(pt.x), yi = cvFloor(pt.y);

    const depthType* row0 = m[yi + 0];
    const depthType* row1 = m[yi + 1];

    depthType v00 = row0[xi + 0];
    depthType v01 = row0[xi + 1];
    depthType v10 = row1[xi + 0];
    depthType v11 = row1[xi + 1];

    // assume correct depth is positive
    bool b00 = v00 > 0;
    bool b01 = v01 > 0;
    bool b10 = v10 > 0;
    bool b11 = v11 > 0;

    if (!fixMissingData)
    {
        if (!(b00 && b01 && b10 && b11))
            return defaultValue;
        else
        {
            float tx = pt.x - xi, ty = pt.y - yi;
            depthType v0 = v00 + tx * (v01 - v00);
            depthType v1 = v10 + tx * (v11 - v10);
            return v0 + ty * (v1 - v0);
        }
    }
    else
    {
        int nz = b00 + b01 + b10 + b11;
        if (nz == 0)
        {
            return defaultValue;
        }
        if (nz == 1)
        {
            if (b00) return v00;
            if (b01) return v01;
            if (b10) return v10;
            if (b11) return v11;
        }
        if (nz == 2)
        {
            if (b00 && b10) v01 = v00, v11 = v10;
            if (b01 && b11) v00 = v01, v10 = v11;
            if (b00 && b01) v10 = v00, v11 = v01;
            if (b10 && b11) v00 = v10, v01 = v11;
            if (b00 && b11) v01 = v10 = (v00 + v11) * 0.5f;
            if (b01 && b10) v00 = v11 = (v01 + v10) * 0.5f;
        }
        if (nz == 3)
        {
            if (!b00) v00 = v10 + v01 - v11;
            if (!b01) v01 = v00 + v11 - v10;
            if (!b10) v10 = v00 + v11 - v01;
            if (!b11) v11 = v01 + v10 - v00;
        }

        float tx = pt.x - xi, ty = pt.y - yi;
        depthType v0 = v00 + tx * (v01 - v00);
        depthType v1 = v10 + tx * (v11 - v10);
        return v0 + ty * (v1 - v0);
    }
}

void _integrateVolumeUnit(
    float truncDist, float voxelSize, int maxWeight,
    cv::Matx44f _pose, Point3i volResolution, Vec4i volStrides,
    InputArray _depth, float depthFactor, const cv::Matx44f& cameraPose,
    const cv::Intr& intrinsics, InputArray _pixNorms, InputArray _volume);

void _integrateRGBVolumeUnit(
    float truncDist, float voxelSize, int maxWeight,
    cv::Matx44f _pose, Point3i volResolution, Vec4i volStrides,
    InputArray _depth, InputArray _rgb, float depthFactor, const cv::Matx44f& cameraPose,
    const cv::Intr& depth_intrinsics, const cv::Intr& rgb_intrinsics, InputArray _pixNorms, InputArray _volume);


void integrateTsdfVolumeUnit(
    const VolumeSettings& settings, const Matx44f& cameraPose,
    InputArray _depth, InputArray _pixNorms, InputArray _volume);

void integrateTsdfVolumeUnit(
    const VolumeSettings& settings, const Matx44f& volumePose, const Matx44f& cameraPose,
    InputArray _depth, InputArray _pixNorms, InputArray _volume);


void raycastTsdfVolumeUnit(const VolumeSettings& settings, const Matx44f& cameraPose, int height, int width, InputArray intr,
                       InputArray _volume, OutputArray _points, OutputArray _normals);

void fetchNormalsFromTsdfVolumeUnit(const VolumeSettings& settings, InputArray _volume,
                                    InputArray _points, OutputArray _normals);

void fetchPointsNormalsFromTsdfVolumeUnit(const VolumeSettings& settings, InputArray _volume,
                                          OutputArray points, OutputArray normals);


#ifdef HAVE_OPENCL
void ocl_integrateTsdfVolumeUnit(
    const VolumeSettings& settings, const Matx44f& cameraPose,
    InputArray _depth, InputArray _pixNorms, InputArray _volume);

void ocl_raycastTsdfVolumeUnit(
    const VolumeSettings& settings, const Matx44f& cameraPose, int height, int width, InputArray intr,
    InputArray _volume, OutputArray _points, OutputArray _normals);

void ocl_fetchNormalsFromTsdfVolumeUnit(
    const VolumeSettings& settings, InputArray _volume,
    InputArray _points, OutputArray _normals);

void ocl_fetchPointsNormalsFromTsdfVolumeUnit(
    const VolumeSettings& settings, InputArray _volume,
    OutputArray _points, OutputArray _normals);
#endif


}  // namespace cv

#endif // include guard
