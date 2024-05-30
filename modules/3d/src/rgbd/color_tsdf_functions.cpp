// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// Partially rewritten from https://github.com/Nerei/kinfu_remake
// Copyright(c) 2012, Anatoly Baksheev. All rights reserved.

#include "../precomp.hpp"
#include "color_tsdf_functions.hpp"
#include "opencl_kernels_3d.hpp"

namespace cv {


void integrateColorTsdfVolumeUnit(const VolumeSettings& settings, const Matx44f& cameraPose,
                                  InputArray _depth, InputArray _rgb, InputArray _pixNorms, InputArray _volume)
{
    Matx44f volumePose;
    settings.getVolumePose(volumePose);
    integrateColorTsdfVolumeUnit(settings, volumePose, cameraPose, _depth, _rgb, _pixNorms, _volume);
}


void integrateColorTsdfVolumeUnit(const VolumeSettings& settings, const Matx44f& volumePose, const Matx44f& cameraPose,
                                  InputArray _depth, InputArray _rgb, InputArray _pixNorms, InputArray _volume)
{
    CV_TRACE_FUNCTION();

    Depth depth = _depth.getMat();
    Colors color = _rgb.getMat();
    Mat volume = _volume.getMat();
    Mat pixNorms = _pixNorms.getMat();

    RGBTsdfVoxel* volDataStart = volume.ptr<RGBTsdfVoxel>();

    Vec4i volStrides;
    settings.getVolumeStrides(volStrides);

    Vec3i resolution;
    settings.getVolumeResolution(resolution);
    const Point3i volResolution = Point3i(resolution);
    float voxelSize = settings.getVoxelSize();

    const Affine3f pose = Affine3f(volumePose);
    const Affine3f vol2cam(Affine3f(cameraPose.inv()) * pose);

    Matx33f intr;
    settings.getCameraIntegrateIntrinsics(intr);
    const Intr::Projector projDepth = Intr(intr).makeProjector();

    const float dfac(1.f / settings.getDepthFactor());
    const float truncDist = settings.getTsdfTruncateDistance();
    const float truncDistInv = 1.f / truncDist;
    const int maxWeight = settings.getMaxWeight();

    Range integrateRange(0, volResolution.x);

#if USE_INTRINSICS
    auto IntegrateInvoker = [&](const Range& range)
    {
        // zStep == vol2cam*(Point3f(x, y, 1)*voxelSize) - basePt;
        Point3f zStepPt = Point3f(vol2cam.matrix(0, 2),
            vol2cam.matrix(1, 2),
            vol2cam.matrix(2, 2)) * voxelSize;

        v_float32x4 zStep(zStepPt.x, zStepPt.y, zStepPt.z, 0);
        v_float32x4 vfxy(projDepth.fx, projDepth.fy, 0.f, 0.f), vcxy(projDepth.cx, projDepth.cy, 0.f, 0.f);
        const v_float32x4 upLimits = v_cvt_f32(v_int32x4(depth.cols - 1, depth.rows - 1, 0, 0));

        for (int x = range.start; x < range.end; x++)
        {
            RGBTsdfVoxel* volDataX = volDataStart + x * volStrides[0];
            for (int y = 0; y < volResolution.y; y++)
            {
                RGBTsdfVoxel* volDataY = volDataX + y * volStrides[1];
                // optimization of camSpace transformation (vector addition instead of matmul at each z)
                Point3f basePt = vol2cam * (Point3f((float)x, (float)y, 0) * voxelSize);
                v_float32x4 camSpacePt(basePt.x, basePt.y, basePt.z, 0);

                int startZ, endZ;
                if (abs(zStepPt.z) > 1e-5)
                {
                    int baseZ = (int)(-basePt.z / zStepPt.z);
                    if (zStepPt.z > 0)
                    {
                        startZ = baseZ;
                        endZ = volResolution.z;
                    }
                    else
                    {
                        startZ = 0;
                        endZ = baseZ;
                    }
                }
                else
                {
                    if (basePt.z > 0)
                    {
                        startZ = 0;
                        endZ = volResolution.z;
                    }
                    else
                    {
                        // z loop shouldn't be performed
                        startZ = endZ = 0;
                    }
                }
                startZ = max(0, startZ);
                endZ = min(int(volResolution.z), endZ);
                for (int z = startZ; z < endZ; z++)
                {
                    // optimization of the following:
                    //Point3f volPt = Point3f(x, y, z)*voxelSize;
                    //Point3f camSpacePt = vol2cam * volPt;
                    camSpacePt = v_add(camSpacePt, zStep);

                    float zCamSpace = v_get0(v_reinterpret_as_f32(v_rotate_right<2>(v_reinterpret_as_u32(camSpacePt))));
                    if (zCamSpace <= 0.f)
                        continue;

                    v_float32x4 camPixVec = v_div(camSpacePt, v_setall_f32(zCamSpace));
                    v_float32x4 projected = v_muladd(camPixVec, vfxy, vcxy);
                    // leave only first 2 lanes
                    projected = v_reinterpret_as_f32(v_and(v_reinterpret_as_u32(projected),
                                                           v_uint32x4(0xFFFFFFFF, 0xFFFFFFFF, 0, 0)));

                    depthType v;
                    // bilinearly interpolate depth at projected
                    {
                        const v_float32x4& pt = projected;
                        // check coords >= 0 and < imgSize
                        v_uint32x4 limits = v_or(v_reinterpret_as_u32(v_lt(pt, v_setzero_f32())),
                                                 v_reinterpret_as_u32(v_ge(pt, upLimits)));
                        limits = v_or(limits, v_rotate_right<1>(limits));
                        if (v_get0(limits))
                            continue;

                        // xi, yi = floor(pt)
                        v_int32x4 ip = v_floor(pt);
                        v_int32x4 ipshift = ip;
                        int xi = v_get0(ipshift);
                        ipshift = v_rotate_right<1>(ipshift);
                        int yi = v_get0(ipshift);

                        const depthType* row0 = depth[yi + 0];
                        const depthType* row1 = depth[yi + 1];

                        // v001 = [v(xi + 0, yi + 0), v(xi + 1, yi + 0)]
                        v_float32x4 v001 = v_load_low(row0 + xi);
                        // v101 = [v(xi + 0, yi + 1), v(xi + 1, yi + 1)]
                        v_float32x4 v101 = v_load_low(row1 + xi);

                        v_float32x4 vall = v_combine_low(v001, v101);

                        // assume correct depth is positive
                        // don't fix missing data
                        if (v_check_all( v_gt(vall, v_setzero_f32())))
                        {
                            v_float32x4 t = v_sub(pt, v_cvt_f32(ip));
                            float tx = v_get0(t);
                            t = v_reinterpret_as_f32(v_rotate_right<1>(v_reinterpret_as_u32(t)));
                            v_float32x4 ty = v_setall_f32(v_get0(t));
                            // vx is y-interpolated between rows 0 and 1
                            v_float32x4 vx = v_add(v001, v_mul(ty, v_sub(v101, v001)));
                            float v0 = v_get0(vx);
                            vx = v_reinterpret_as_f32(v_rotate_right<1>(v_reinterpret_as_u32(vx)));
                            float v1 = v_get0(vx);
                            v = v0 + tx * (v1 - v0);
                        }
                        else
                            continue;
                    }

                    // norm(camPixVec) produces double which is too slow
                    int _u = (int)v_get0(projected);
                    int _v = (int)v_get0(v_rotate_right<1>(projected));

                    if (!(_u >= 0 && _u < depth.cols && _v >= 0 && _v < depth.rows))
                        continue;
                    float pixNorm = pixNorms.at<float>(_v, _u);
                    // TODO: Add support of 3point and 4 point representation
                    Vec3f colorRGB = color.at<Vec3f>(_v, _u);
                    //float pixNorm = sqrt(v_reduce_sum(camPixVec*camPixVec));
                    // difference between distances of point and of surface to camera
                    float sdf = pixNorm * (v * dfac - zCamSpace);
                    // possible alternative is:
                    // kftype sdf = norm(camSpacePt)*(v*dfac/camSpacePt.z - 1);
                    if (sdf >= -truncDist)
                    {
                        TsdfType tsdf = floatToTsdf(fmin(1.f, sdf * truncDistInv));

                        RGBTsdfVoxel& voxel = volDataY[z * volStrides[2]];
                        WeightType& weight = voxel.weight;
                        TsdfType& value = voxel.tsdf;
                        ColorType& r = voxel.r;
                        ColorType& g = voxel.g;
                        ColorType& b = voxel.b;

                        // update RGB
                        r = (ColorType)((float)(r * weight) + (colorRGB[0])) / (weight + 1);
                        g = (ColorType)((float)(g * weight) + (colorRGB[1])) / (weight + 1);
                        b = (ColorType)((float)(b * weight) + (colorRGB[2])) / (weight + 1);
                        colorFix(r, g, b);
                        // update TSDF
                        value = floatToTsdf((tsdfToFloat(value) * weight + tsdfToFloat(tsdf)) / (weight + 1));
                        weight = WeightType(min(int(weight + 1), int(maxWeight)));
                    }
                }
            }
        }
    };
#else
    auto IntegrateInvoker = [&](const Range& range)
    {
        for (int x = range.start; x < range.end; x++)
        {
            RGBTsdfVoxel* volDataX = volDataStart + x * volStrides[0];
            for (int y = 0; y < volResolution.y; y++)
            {
                RGBTsdfVoxel* volDataY = volDataX + y * volStrides[1];
                // optimization of camSpace transformation (vector addition instead of matmul at each z)
                Point3f basePt = vol2cam * (Point3f(float(x), float(y), 0.0f) * voxelSize);
                Point3f camSpacePt = basePt;
                // zStep == vol2cam*(Point3f(x, y, 1)*voxelSize) - basePt;
                // zStep == vol2cam*[Point3f(x, y, 1) - Point3f(x, y, 0)]*voxelSize
                Point3f zStep = Point3f(vol2cam.matrix(0, 2),
                    vol2cam.matrix(1, 2),
                    vol2cam.matrix(2, 2)) * voxelSize;
                int startZ, endZ;
                if (abs(zStep.z) > 1e-5)
                {
                    int baseZ = int(-basePt.z / zStep.z);
                    if (zStep.z > 0)
                    {
                        startZ = baseZ;
                        endZ = volResolution.z;
                    }
                    else
                    {
                        startZ = 0;
                        endZ = baseZ;
                    }
                }
                else
                {
                    if (basePt.z > 0)
                    {
                        startZ = 0;
                        endZ = volResolution.z;
                    }
                    else
                    {
                        // z loop shouldn't be performed
                        startZ = endZ = 0;
                    }
                }
                startZ = max(0, startZ);
                endZ = min(int(volResolution.z), endZ);

                for (int z = startZ; z < endZ; z++)
                {
                    // optimization of the following:
                    //Point3f volPt = Point3f(x, y, z)*volume.voxelSize;
                    //Point3f camSpacePt = vol2cam * volPt;

                    camSpacePt += zStep;
                    if (camSpacePt.z <= 0)
                        continue;

                    Point3f camPixVec;
                    Point2f projected = projDepth(camSpacePt, camPixVec);

                    depthType v = bilinearDepth(depth, projected);
                    if (v == 0) {
                        continue;
                    }
                    int _u = (int)projected.x;
                    int _v = (int)projected.y;

                    if (!(_u >= 0 && _u < depth.cols && _v >= 0 && _v < depth.rows))
                        continue;

                    float pixNorm = pixNorms.at<float>(_v, _u);
                    // TODO: Add support of 3point and 4 point representation
                    Vec3f colorRGB = color.at<Vec3f>(_v, _u);

                    // difference between distances of point and of surface to camera
                    float sdf = pixNorm * (v * dfac - camSpacePt.z);
                    // possible alternative is:
                    // kftype sdf = norm(camSpacePt)*(v*dfac/camSpacePt.z - 1);
                    if (sdf >= -truncDist)
                    {
                        TsdfType tsdf = floatToTsdf(fmin(1.f, sdf * truncDistInv));

                        RGBTsdfVoxel& voxel = volDataY[z * volStrides[2]];
                        WeightType& weight = voxel.weight;
                        TsdfType& value = voxel.tsdf;

                        ColorType& r = voxel.r;
                        ColorType& g = voxel.g;
                        ColorType& b = voxel.b;
                        // update RGB
                        if (weight < 1)
                        {
                            r = (ColorType)((float)(r * weight) + (colorRGB[0])) / (weight + 1);
                            g = (ColorType)((float)(g * weight) + (colorRGB[1])) / (weight + 1);
                            b = (ColorType)((float)(b * weight) + (colorRGB[2])) / (weight + 1);
                        }

                        // update TSDF
                        value = floatToTsdf((tsdfToFloat(value) * weight + tsdfToFloat(tsdf)) / (weight + 1));
                        weight = (WeightType)min(weight + 1, maxWeight);
                    }
                }
            }
        }
    };
#endif
    parallel_for_(integrateRange, IntegrateInvoker);
}



#if USE_INTRINSICS
// all coordinate checks should be done in inclosing cycle

inline float interpolateColorVoxel(const Mat& volume,
                                   const Vec4i& volDims, const Vec8i& neighbourCoords,
                                   const v_float32x4& p)
{
    // tx, ty, tz = floor(p)
    v_int32x4 ip = v_floor(p);
    v_float32x4 t = v_sub(p, v_cvt_f32(ip));
    float tx = v_get0(t);
    t = v_reinterpret_as_f32(v_rotate_right<1>(v_reinterpret_as_u32(t)));
    float ty = v_get0(t);
    t = v_reinterpret_as_f32(v_rotate_right<1>(v_reinterpret_as_u32(t)));
    float tz = v_get0(t);

    int xdim = volDims[0], ydim = volDims[1], zdim = volDims[2];
    const RGBTsdfVoxel* volData = volume.ptr<RGBTsdfVoxel>();

    int ix = v_get0(ip);
    ip = v_rotate_right<1>(ip);
    int iy = v_get0(ip);
    ip = v_rotate_right<1>(ip);
    int iz = v_get0(ip);

    int coordBase = ix * xdim + iy * ydim + iz * zdim;

    TsdfType vx[8];
    for (int i = 0; i < 8; i++)
        vx[i] = volData[neighbourCoords[i] + coordBase].tsdf;

    v_float32x4 v0246 = tsdfToFloat_INTR(v_int32x4(vx[0], vx[2], vx[4], vx[6]));
    v_float32x4 v1357 = tsdfToFloat_INTR(v_int32x4(vx[1], vx[3], vx[5], vx[7]));
    v_float32x4 vxx = v_add(v0246, v_mul(v_setall_f32(tz), v_sub(v1357, v0246)));

    v_float32x4 v00_10 = vxx;
    v_float32x4 v01_11 = v_reinterpret_as_f32(v_rotate_right<1>(v_reinterpret_as_u32(vxx)));

    v_float32x4 v0_1 = v_add(v00_10, v_mul(v_setall_f32(ty), v_sub(v01_11, v00_10)));
    float v0 = v_get0(v0_1);
    v0_1 = v_reinterpret_as_f32(v_rotate_right<2>(v_reinterpret_as_u32(v0_1)));
    float v1 = v_get0(v0_1);

    return v0 + tx * (v1 - v0);
}

inline float interpolateColorVoxel(const Mat& volume,
                                   const Vec4i& volDims, const Vec8i& neighbourCoords,
                                   const Point3f& _p)
{
    v_float32x4 p(_p.x, _p.y, _p.z, 0);
    return interpolateColorVoxel(volume, volDims, neighbourCoords, p);
}


#else
inline float interpolateColorVoxel(const Mat& volume,
                                   const Vec4i& volDims, const Vec8i& neighbourCoords,
                                   const Point3f& p)
{
    int xdim = volDims[0], ydim = volDims[1], zdim = volDims[2];

    int ix = cvFloor(p.x);
    int iy = cvFloor(p.y);
    int iz = cvFloor(p.z);

    float tx = p.x - ix;
    float ty = p.y - iy;
    float tz = p.z - iz;

    int coordBase = ix * xdim + iy * ydim + iz * zdim;
    const RGBTsdfVoxel* volData = volume.ptr<RGBTsdfVoxel>();

    float vx[8];
    for (int i = 0; i < 8; i++)
        vx[i] = tsdfToFloat(volData[neighbourCoords[i] + coordBase].tsdf);

    float v00 = vx[0] + tz * (vx[1] - vx[0]);
    float v01 = vx[2] + tz * (vx[3] - vx[2]);
    float v10 = vx[4] + tz * (vx[5] - vx[4]);
    float v11 = vx[6] + tz * (vx[7] - vx[6]);

    float v0 = v00 + ty * (v01 - v00);
    float v1 = v10 + ty * (v11 - v10);

    return v0 + tx * (v1 - v0);

}
#endif


#if USE_INTRINSICS
//gradientDeltaFactor is fixed at 1.0 of voxel size

inline v_float32x4 getNormalColorVoxel(const Mat& volume,
                                       const Vec4i& volDims, const Vec8i& neighbourCoords, const Point3i volResolution,
                                       const v_float32x4& p)
{
    if (v_check_any(v_lt(p, v_float32x4(1.f, 1.f, 1.f, 0.f))) ||
        v_check_any(v_ge(p, v_float32x4((float)(volResolution.x - 2),
            (float)(volResolution.y - 2),
            (float)(volResolution.z - 2), 1.f)))
        )
        return nanv;

    v_int32x4 ip = v_floor(p);
    v_float32x4 t = v_sub(p, v_cvt_f32(ip));
    float tx = v_get0(t);
    t = v_reinterpret_as_f32(v_rotate_right<1>(v_reinterpret_as_u32(t)));
    float ty = v_get0(t);
    t = v_reinterpret_as_f32(v_rotate_right<1>(v_reinterpret_as_u32(t)));
    float tz = v_get0(t);

    const int xdim = volDims[0], ydim = volDims[1], zdim = volDims[2];
    const RGBTsdfVoxel* volData = volume.ptr<RGBTsdfVoxel>();

    int ix = v_get0(ip); ip = v_rotate_right<1>(ip);
    int iy = v_get0(ip); ip = v_rotate_right<1>(ip);
    int iz = v_get0(ip);

    int coordBase = ix * xdim + iy * ydim + iz * zdim;

    float CV_DECL_ALIGNED(16) an[4];
    an[0] = an[1] = an[2] = an[3] = 0.f;
    for (int c = 0; c < 3; c++)
    {
        const int dim = volDims[c];
        float& nv = an[c];

        float vx[8];
        for (int i = 0; i < 8; i++)
            vx[i] = tsdfToFloat(volData[neighbourCoords[i] + coordBase + 1 * dim].tsdf) -
            tsdfToFloat(volData[neighbourCoords[i] + coordBase - 1 * dim].tsdf);

        v_float32x4 v0246(vx[0], vx[2], vx[4], vx[6]);
        v_float32x4 v1357(vx[1], vx[3], vx[5], vx[7]);
        v_float32x4 vxx = v_add(v0246, v_mul(v_setall_f32(tz), v_sub(v1357, v0246)));

        v_float32x4 v00_10 = vxx;
        v_float32x4 v01_11 = v_reinterpret_as_f32(v_rotate_right<1>(v_reinterpret_as_u32(vxx)));

        v_float32x4 v0_1 = v_add(v00_10, v_mul(v_setall_f32(ty), v_sub(v01_11, v00_10)));
        float v0 = v_get0(v0_1);
        v0_1 = v_reinterpret_as_f32(v_rotate_right<2>(v_reinterpret_as_u32(v0_1)));
        float v1 = v_get0(v0_1);

        nv = v0 + tx * (v1 - v0);
    }

    v_float32x4 n = v_load_aligned(an);
    v_float32x4 Norm = v_sqrt(v_setall_f32(v_reduce_sum(v_mul(n, n))));

    return v_get0(Norm) < 0.0001f ? nanv : v_div(n, Norm);
}

inline Point3f getNormalColorVoxel(const Mat& volume,
                                   const Vec4i& volDims, const Vec8i& neighbourCoords, const Point3i volResolution,
                                   const Point3f& _p)
{
    v_float32x4 p(_p.x, _p.y, _p.z, 0.f);
    v_float32x4 result = getNormalColorVoxel(volume, volDims, neighbourCoords, volResolution, p);
    float CV_DECL_ALIGNED(16) ares[4];
    v_store_aligned(ares, result);
    return Point3f(ares[0], ares[1], ares[2]);
}
#else
inline Point3f getNormalColorVoxel(const Mat& volume,
                                   const Vec4i& volDims, const Vec8i& neighbourCoords, const Point3i volResolution,
                                   const Point3f& p)
{
    int xdim = volDims[0], ydim = volDims[1], zdim = volDims[2];
    const RGBTsdfVoxel* volData = volume.ptr<RGBTsdfVoxel>();

    if (p.x < 1 || p.x >= volResolution.x - 2 ||
        p.y < 1 || p.y >= volResolution.y - 2 ||
        p.z < 1 || p.z >= volResolution.z - 2)
        return nan3;

    int ix = cvFloor(p.x);
    int iy = cvFloor(p.y);
    int iz = cvFloor(p.z);

    float tx = p.x - ix;
    float ty = p.y - iy;
    float tz = p.z - iz;

    int coordBase = ix * xdim + iy * ydim + iz * zdim;

    Vec3f an;
    for (int c = 0; c < 3; c++)
    {
        const int dim = volDims[c];
        float& nv = an[c];

        float vx[8];
        for (int i = 0; i < 8; i++)
            vx[i] = tsdfToFloat(volData[neighbourCoords[i] + coordBase + 1 * dim].tsdf) -
            tsdfToFloat(volData[neighbourCoords[i] + coordBase - 1 * dim].tsdf);

        float v00 = vx[0] + tz * (vx[1] - vx[0]);
        float v01 = vx[2] + tz * (vx[3] - vx[2]);
        float v10 = vx[4] + tz * (vx[5] - vx[4]);
        float v11 = vx[6] + tz * (vx[7] - vx[6]);

        float v0 = v00 + ty * (v01 - v00);
        float v1 = v10 + ty * (v11 - v10);

        nv = v0 + tx * (v1 - v0);
    }

    float nv = sqrt(an[0] * an[0] +
        an[1] * an[1] +
        an[2] * an[2]);
    return nv < 0.0001f ? nan3 : an / nv;
}
#endif

#if USE_INTRINSICS
inline float interpolateColor(float tx, float ty, float tz, float vx[8])
{
    v_float32x4 v0246, v1357;
    v_load_deinterleave(vx, v0246, v1357);

    v_float32x4 vxx = v_add(v0246, v_mul(v_setall_f32(tz), v_sub(v1357, v0246)));

    v_float32x4 v00_10 = vxx;
    v_float32x4 v01_11 = v_reinterpret_as_f32(v_rotate_right<1>(v_reinterpret_as_u32(vxx)));

    v_float32x4 v0_1 = v_add(v00_10, v_mul(v_setall_f32(ty), v_sub(v01_11, v00_10)));
    float v0 = v_get0(v0_1);
    v0_1 = v_reinterpret_as_f32(v_rotate_right<2>(v_reinterpret_as_u32(v0_1)));
    float v1 = v_get0(v0_1);

    return v0 + tx * (v1 - v0);
}
#else
inline float interpolateColor(float tx, float ty, float tz, float vx[8])
{
    float v00 = vx[0] + tz * (vx[1] - vx[0]);
    float v01 = vx[2] + tz * (vx[3] - vx[2]);
    float v10 = vx[4] + tz * (vx[5] - vx[4]);
    float v11 = vx[6] + tz * (vx[7] - vx[6]);

    float v0 = v00 + ty * (v01 - v00);
    float v1 = v10 + ty * (v11 - v10);

    return v0 + tx * (v1 - v0);
}
#endif


#if USE_INTRINSICS
//gradientDeltaFactor is fixed at 1.0 of voxel size

inline v_float32x4 getColorVoxel(const Mat& volume,
    const Vec4i& volDims, const Vec8i& neighbourCoords, const Point3i volResolution,
    const float voxelSizeInv, const v_float32x4& p)
{
    if (v_check_any(v_lt(p, v_float32x4(1.f, 1.f, 1.f, 0.f))) ||
        v_check_any(v_ge(p, v_float32x4((float)(volResolution.x - 2),
            (float)(volResolution.y - 2),
            (float)(volResolution.z - 2), 1.f)))
        )
        return nanv;

    v_int32x4 ip = v_floor(p);

    const int xdim = volDims[0], ydim = volDims[1], zdim = volDims[2];
    const RGBTsdfVoxel* volData = volume.ptr<RGBTsdfVoxel>();

    int ix = v_get0(ip); ip = v_rotate_right<1>(ip);
    int iy = v_get0(ip); ip = v_rotate_right<1>(ip);
    int iz = v_get0(ip);

    int coordBase = ix * xdim + iy * ydim + iz * zdim;
    float CV_DECL_ALIGNED(16) rgb[4];

#if USE_INTERPOLATION_IN_GETNORMAL
    float r[8], g[8], b[8];
    for (int i = 0; i < 8; i++)
    {
        r[i] = (float)volData[neighbourCoords[i] + coordBase].r;
        g[i] = (float)volData[neighbourCoords[i] + coordBase].g;
        b[i] = (float)volData[neighbourCoords[i] + coordBase].b;
    }

    v_float32x4 vsi(voxelSizeInv, voxelSizeInv, voxelSizeInv, voxelSizeInv);
    v_float32x4 ptVox = v_mul(p, vsi);
    v_int32x4 iptVox = v_floor(ptVox);
    v_float32x4 t = v_sub(ptVox, v_cvt_f32(iptVox));
    float tx = v_get0(t); t = v_rotate_right<1>(t);
    float ty = v_get0(t); t = v_rotate_right<1>(t);
    float tz = v_get0(t);
    rgb[0] = interpolateColor(tx, ty, tz, r);
    rgb[1] = interpolateColor(tx, ty, tz, g);
    rgb[2] = interpolateColor(tx, ty, tz, b);
    rgb[3] = 0.f;
#else
    rgb[0] = volData[coordBase].r;
    rgb[1] = volData[coordBase].g;
    rgb[2] = volData[coordBase].b;
    rgb[3] = 0.f;
#endif
    v_float32x4 res = v_load_aligned(rgb);
    return res;
}

inline Point3f getColorVoxel(const Mat& volume,
    const Vec4i& volDims, const Vec8i& neighbourCoords, const Point3i volResolution,
    const float voxelSizeInv, const Point3f& _p)
{
    v_float32x4 p(_p.x, _p.y, _p.z, 0.f);
    v_float32x4 result = getColorVoxel(volume, volDims, neighbourCoords, volResolution, voxelSizeInv, p);
    float CV_DECL_ALIGNED(16) ares[4];
    v_store_aligned(ares, result);
    return Point3f(ares[0], ares[1], ares[2]);
}


#else
inline Point3f getColorVoxel(const Mat& volume,
    const Vec4i& volDims, const Vec8i& neighbourCoords, const Point3i volResolution,
    const float voxelSizeInv, const Point3f& p)
{
    const int xdim = volDims[0], ydim = volDims[1], zdim = volDims[2];
    const RGBTsdfVoxel* volData = volume.ptr<RGBTsdfVoxel>();

    if (p.x < 1 || p.x >= volResolution.x - 2 ||
        p.y < 1 || p.y >= volResolution.y - 2 ||
        p.z < 1 || p.z >= volResolution.z - 2)
        return nan3;

    int ix = cvFloor(p.x);
    int iy = cvFloor(p.y);
    int iz = cvFloor(p.z);

    int coordBase = ix * xdim + iy * ydim + iz * zdim;
    Point3f res;

#if USE_INTERPOLATION_IN_GETNORMAL
    // TODO: create better interpolation or remove this simple version
    float r[8], g[8], b[8];
    for (int i = 0; i < 8; i++)
    {
        r[i] = (float)volData[neighbourCoords[i] + coordBase].r;
        g[i] = (float)volData[neighbourCoords[i] + coordBase].g;
        b[i] = (float)volData[neighbourCoords[i] + coordBase].b;
    }

    Point3f ptVox = p * voxelSizeInv;
    Vec3i iptVox(cvFloor(ptVox.x), cvFloor(ptVox.y), cvFloor(ptVox.z));
    float tx = ptVox.x - iptVox[0];
    float ty = ptVox.y - iptVox[1];
    float tz = ptVox.z - iptVox[2];

    res = Point3f(interpolateColor(tx, ty, tz, r),
        interpolateColor(tx, ty, tz, g),
        interpolateColor(tx, ty, tz, b));
#else
    res = Point3f(volData[coordBase].r, volData[coordBase].g, volData[coordBase].b);
#endif
    colorFix(res);
    return res;
}
#endif


void raycastColorTsdfVolumeUnit(const VolumeSettings &settings, const Matx44f &cameraPose,
                                int height, int width, InputArray intr,
                                InputArray _volume, OutputArray _points, OutputArray _normals, OutputArray _colors)
{
    CV_TRACE_FUNCTION();

    Size frameSize(width, height);
    CV_Assert(frameSize.area() > 0);

    Matx33f mintr(intr.getMat());

    _points.create(frameSize, POINT_TYPE);
    _normals.create(frameSize, POINT_TYPE);
    _colors.create(frameSize, COLOR_TYPE);

    Points points = _points.getMat();
    Normals normals = _normals.getMat();
    Colors colors = _colors.getMat();

    const Vec4i volDims;
    settings.getVolumeStrides(volDims);
    const Vec8i neighbourCoords = Vec8i(
        volDims.dot(Vec4i(0, 0, 0)),
        volDims.dot(Vec4i(0, 0, 1)),
        volDims.dot(Vec4i(0, 1, 0)),
        volDims.dot(Vec4i(0, 1, 1)),
        volDims.dot(Vec4i(1, 0, 0)),
        volDims.dot(Vec4i(1, 0, 1)),
        volDims.dot(Vec4i(1, 1, 0)),
        volDims.dot(Vec4i(1, 1, 1))
    );

    Vec3i resolution;
    settings.getVolumeResolution(resolution);
    const Point3i volResolution = Point3i(resolution);
    const Point3f volSize = Point3f(volResolution) * settings.getVoxelSize();

    const Intr::Reprojector reprojDepth = Intr(mintr).makeReprojector();

    Matx44f _pose;
    settings.getVolumePose(_pose);
    const Affine3f pose = Affine3f(_pose);

    const Point3f boxMax(volSize - Point3f(settings.getVoxelSize(), settings.getVoxelSize(), settings.getVoxelSize()));
    const Point3f boxMin = Point3f(0, 0, 0);
    const Affine3f cam2vol(pose.inv() * Affine3f(cameraPose));
    const Affine3f vol2cam(Affine3f(cameraPose.inv()) * pose);

    const Mat volume = _volume.getMat();
    float voxelSize = settings.getVoxelSize();
    float voxelSizeInv = 1.0f / voxelSize;
    float tstep = settings.getTsdfTruncateDistance() * settings.getRaycastStepFactor();

    Range raycastRange = Range(0, points.rows);

#if USE_INTRINSICS
    auto RaycastInvoker = [&](const Range& range)
    {
        const v_float32x4 vfxy(reprojDepth.fxinv, reprojDepth.fyinv, 0, 0);
        const v_float32x4 vcxy(reprojDepth.cx, reprojDepth.cy, 0, 0);

        const float(&cm)[16] = cam2vol.matrix.val;
        const v_float32x4 camRot0(cm[0], cm[4], cm[8], 0);
        const v_float32x4 camRot1(cm[1], cm[5], cm[9], 0);
        const v_float32x4 camRot2(cm[2], cm[6], cm[10], 0);
        const v_float32x4 camTrans(cm[3], cm[7], cm[11], 0);

        const v_float32x4 boxDown(boxMin.x, boxMin.y, boxMin.z, 0.f);
        const v_float32x4 boxUp(boxMax.x, boxMax.y, boxMax.z, 0.f);

        const v_float32x4 invVoxelSize = v_float32x4(voxelSizeInv,
            voxelSizeInv,
            voxelSizeInv, 1.f);

        const float(&vm)[16] = vol2cam.matrix.val;
        const v_float32x4 volRot0(vm[0], vm[4], vm[8], 0);
        const v_float32x4 volRot1(vm[1], vm[5], vm[9], 0);
        const v_float32x4 volRot2(vm[2], vm[6], vm[10], 0);
        const v_float32x4 volTrans(vm[3], vm[7], vm[11], 0);

        for (int y = range.start; y < range.end; y++)
        {
            ptype* ptsRow = points[y];
            ptype* nrmRow = normals[y];
            ptype* clrRow = colors[y];

            for (int x = 0; x < points.cols; x++)
            {
                v_float32x4 point = nanv, normal = nanv, color = nanv;

                v_float32x4 orig = camTrans;

                // get direction through pixel in volume space:

                // 1. reproject (x, y) on projecting plane where z = 1.f
                v_float32x4 planed = v_mul(v_sub(v_float32x4((float)x, (float)y, 0.f, 0.f), vcxy), vfxy);
                planed = v_combine_low(planed, v_float32x4(1.f, 0.f, 0.f, 0.f));

                // 2. rotate to volume space
                planed = v_matmuladd(planed, camRot0, camRot1, camRot2, v_setzero_f32());

                // 3. normalize
                v_float32x4 invNorm = v_invsqrt(v_setall_f32(v_reduce_sum(v_mul(planed, planed))));
                v_float32x4 dir = v_mul(planed, invNorm);

                // compute intersection of ray with all six bbox planes
                v_float32x4 rayinv = v_div(v_setall_f32(1.f), dir);
                // div by zero should be eliminated by these products
                v_float32x4 tbottom = v_mul(rayinv, v_sub(boxDown, orig));
                v_float32x4 ttop = v_mul(rayinv, v_sub(boxUp, orig));

                // re-order intersections to find smallest and largest on each axis
                v_float32x4 minAx = v_min(ttop, tbottom);
                v_float32x4 maxAx = v_max(ttop, tbottom);

                // near clipping plane
                const float clip = 0.f;
                float _minAx[4], _maxAx[4];
                v_store(_minAx, minAx);
                v_store(_maxAx, maxAx);
                float tmin = max({ _minAx[0], _minAx[1], _minAx[2], clip });
                float tmax = min({ _maxAx[0], _maxAx[1], _maxAx[2] });

                // precautions against getting coordinates out of bounds
                tmin = tmin + tstep;
                tmax = tmax - tstep;

                if (tmin < tmax)
                {
                    // interpolation optimized a little
                    orig = v_mul(orig, invVoxelSize);
                    dir = v_mul(dir, invVoxelSize);

                    int xdim = volDims[0];
                    int ydim = volDims[1];
                    int zdim = volDims[2];
                    v_float32x4 rayStep = v_mul(dir, v_setall_f32(tstep));
                    v_float32x4 next = v_add(orig, v_mul(dir, v_setall_f32(tmin)));
                    float f = interpolateColorVoxel(volume, volDims, neighbourCoords, next);
                    float fnext = f;

                    //raymarch
                    int steps = 0;
                    int nSteps = cvFloor((tmax - tmin) / tstep);
                    for (; steps < nSteps; steps++)
                    {
                        next = v_add(next, rayStep);
                        v_int32x4 ip = v_round(next);
                        int ix = v_get0(ip); ip = v_rotate_right<1>(ip);
                        int iy = v_get0(ip); ip = v_rotate_right<1>(ip);
                        int iz = v_get0(ip);
                        int coord = ix * xdim + iy * ydim + iz * zdim;

                        fnext = tsdfToFloat(volume.at<RGBTsdfVoxel>(coord).tsdf);
                        if (fnext != f)
                        {
                            fnext = interpolateColorVoxel(volume, volDims, neighbourCoords, next);

                            // when ray crosses a surface
                            if (std::signbit(f) != std::signbit(fnext))
                                break;

                            f = fnext;
                        }
                    }

                    // if ray penetrates a surface from outside
                    // linearly interpolate t between two f values
                    if (f > 0.f && fnext < 0.f)
                    {
                        v_float32x4 tp = v_sub(next, rayStep);
                        float ft = interpolateColorVoxel(volume, volDims, neighbourCoords, tp);
                        float ftdt = interpolateColorVoxel(volume, volDims, neighbourCoords, next);
                        float ts = tmin + tstep * (steps - ft / (ftdt - ft));

                        // avoid division by zero
                        if (!cvIsNaN(ts) && !cvIsInf(ts))
                        {
                            v_float32x4 pv = v_add(orig, v_mul(dir, v_setall_f32(ts)));
                            v_float32x4 nv = getNormalColorVoxel(volume, volDims, neighbourCoords, volResolution, pv);
                            v_float32x4 cv = getColorVoxel(volume, volDims, neighbourCoords, volResolution, voxelSizeInv, pv);

                            if (!isNaN(nv))
                            {
                                color = cv;
                                //convert pv and nv to camera space
                                normal = v_matmuladd(nv, volRot0, volRot1, volRot2, v_setzero_f32());
                                // interpolation optimized a little
                                point = v_matmuladd(v_mul(pv, v_float32x4(voxelSize, voxelSize, voxelSize, 1.f)),
                                    volRot0, volRot1, volRot2, volTrans);
                            }
                        }
                    }
                }

                v_store((float*)(&ptsRow[x]), point);
                v_store((float*)(&nrmRow[x]), normal);
                v_store((float*)(&clrRow[x]), color);
            }
        }
    };
#else
    auto RaycastInvoker = [&](const Range& range)
    {
        const Point3f camTrans = cam2vol.translation();
        const Matx33f  camRot = cam2vol.rotation();
        const Matx33f  volRot = vol2cam.rotation();

        for (int y = range.start; y < range.end; y++)
        {
            ptype* ptsRow = points[y];
            ptype* nrmRow = normals[y];
            ptype* clrRow = colors[y];

            for (int x = 0; x < points.cols; x++)
            {
                Point3f point = nan3, normal = nan3, color = nan3;

                Point3f orig = camTrans;
                // direction through pixel in volume space
                Point3f dir = normalize(Vec3f(camRot * reprojDepth(Point3f(float(x), float(y), 1.f))));

                // compute intersection of ray with all six bbox planes
                Vec3f rayinv(1.f / dir.x, 1.f / dir.y, 1.f / dir.z);
                Point3f tbottom = rayinv.mul(boxMin - orig);
                Point3f ttop = rayinv.mul(boxMax - orig);

                // re-order intersections to find smallest and largest on each axis
                Point3f minAx(min(ttop.x, tbottom.x), min(ttop.y, tbottom.y), min(ttop.z, tbottom.z));
                Point3f maxAx(max(ttop.x, tbottom.x), max(ttop.y, tbottom.y), max(ttop.z, tbottom.z));

                // near clipping plane
                const float clip = 0.f;
                //float tmin = max(max(max(minAx.x, minAx.y), max(minAx.x, minAx.z)), clip);
                //float tmax =     min(min(maxAx.x, maxAx.y), min(maxAx.x, maxAx.z));
                float tmin = max({ minAx.x, minAx.y, minAx.z, clip });
                float tmax = min({ maxAx.x, maxAx.y, maxAx.z });

                // precautions against getting coordinates out of bounds
                tmin = tmin + tstep;
                tmax = tmax - tstep;

                if (tmin < tmax)
                {
                    // interpolation optimized a little
                    orig = orig * voxelSizeInv;
                    dir = dir * voxelSizeInv;

                    Point3f rayStep = dir * tstep;
                    Point3f next = (orig + dir * tmin);
                    float f = interpolateColorVoxel(volume, volDims, neighbourCoords, next);
                    float fnext = f;

                    //raymarch
                    int steps = 0;
                    int nSteps = int(floor((tmax - tmin) / tstep));
                    for (; steps < nSteps; steps++)
                    {
                        next += rayStep;
                        int xdim = volDims[0];
                        int ydim = volDims[1];
                        int zdim = volDims[2];
                        int ix = cvRound(next.x);
                        int iy = cvRound(next.y);
                        int iz = cvRound(next.z);
                        fnext = tsdfToFloat(volume.at<RGBTsdfVoxel>(ix * xdim + iy * ydim + iz * zdim).tsdf);
                        if (fnext != f)
                        {
                            fnext = interpolateColorVoxel(volume, volDims, neighbourCoords, next);
                            // when ray crosses a surface
                            if (std::signbit(f) != std::signbit(fnext))
                                break;

                            f = fnext;
                        }
                    }
                    // if ray penetrates a surface from outside
                    // linearly interpolate t between two f values
                    if (f > 0.f && fnext < 0.f)
                    {
                        Point3f tp = next - rayStep;
                        float ft = interpolateColorVoxel(volume, volDims, neighbourCoords, tp);
                        float ftdt = interpolateColorVoxel(volume, volDims, neighbourCoords, next);
                        // float t = tmin + steps*tstep;
                        // float ts = t - tstep*ft/(ftdt - ft);
                        float ts = tmin + tstep * (steps - ft / (ftdt - ft));

                        // avoid division by zero
                        if (!cvIsNaN(ts) && !cvIsInf(ts))
                        {
                            Point3f pv = (orig + dir * ts);
                            Point3f nv = getNormalColorVoxel(volume, volDims, neighbourCoords, volResolution, pv);
                            Point3f cv = getColorVoxel(volume, volDims, neighbourCoords, volResolution, voxelSizeInv, pv);
                            if (!isNaN(nv))
                            {
                                //convert pv and nv to camera space
                                normal = volRot * nv;
                                color = cv;
                                // interpolation optimized a little
                                point = vol2cam * (pv * voxelSize);
                            }
                        }
                    }
                }
                ptsRow[x] = toPtype(point);
                nrmRow[x] = toPtype(normal);
                clrRow[x] = toPtype(color);
            }
        }

    };
#endif

    parallel_for_(raycastRange, RaycastInvoker);
}


void fetchNormalsFromColorTsdfVolumeUnit(const VolumeSettings& settings, InputArray _volume,
                                         InputArray _points, OutputArray _normals)
{
    CV_TRACE_FUNCTION();
    CV_Assert(!_points.empty());
    if (!_normals.needed())
        return;

    Points points = _points.getMat();
    CV_Assert(points.type() == POINT_TYPE);

    _normals.createSameSize(_points, _points.type());
    Normals normals = _normals.getMat();

    const Mat volume = _volume.getMat();

    Matx44f _pose;
    settings.getVolumePose(_pose);
    const Affine3f pose = Affine3f(_pose);
    Affine3f invPose(pose.inv());
    float voxelSizeInv = 1.f / settings.getVoxelSize();

    const Vec4i volDims;
    settings.getVolumeStrides(volDims);
    const Vec8i neighbourCoords = Vec8i(
        volDims.dot(Vec4i(0, 0, 0)),
        volDims.dot(Vec4i(0, 0, 1)),
        volDims.dot(Vec4i(0, 1, 0)),
        volDims.dot(Vec4i(0, 1, 1)),
        volDims.dot(Vec4i(1, 0, 0)),
        volDims.dot(Vec4i(1, 0, 1)),
        volDims.dot(Vec4i(1, 1, 0)),
        volDims.dot(Vec4i(1, 1, 1))
    );

    Vec3i resolution;
    settings.getVolumeResolution(resolution);
    const Point3i volResolution = Point3i(resolution);

    auto PushNormals = [&](const ptype& pp, const int* position)
    {
        Point3f p = fromPtype(pp);
        Point3f n = nan3;
        if (!isNaN(p))
        {
            Point3f voxPt = (invPose * p);
            voxPt = voxPt * voxelSizeInv;
            n = pose.rotation() * getNormalColorVoxel(volume, volDims, neighbourCoords, volResolution, voxPt);
        }
        normals(position[0], position[1]) = toPtype(n);
    };
    points.forEach(PushNormals);

}

inline void coord(
    const Mat& volume, const RGBTsdfVoxel* volDataStart, std::vector<ptype>& points, std::vector<ptype>& normals, std::vector<ptype>& colors,
    const Point3i volResolution, const Vec4i volDims, const Vec8i neighbourCoords, const Affine3f pose,
    const float voxelSize, const float voxelSizeInv, bool needNormals, bool needColors, int x, int y, int z, Point3f V, float v0, int axis)
{
    // 0 for x, 1 for y, 2 for z
    bool limits = false;
    Point3i shift;
    float Vc = 0.f;
    if (axis == 0)
    {
        shift = Point3i(1, 0, 0);
        limits = (x + 1 < volResolution.x);
        Vc = V.x;
    }
    if (axis == 1)
    {
        shift = Point3i(0, 1, 0);
        limits = (y + 1 < volResolution.y);
        Vc = V.y;
    }
    if (axis == 2)
    {
        shift = Point3i(0, 0, 1);
        limits = (z + 1 < volResolution.z);
        Vc = V.z;
    }

    if (limits)
    {
        const RGBTsdfVoxel& voxeld = volDataStart[(x + shift.x) * volDims[0] +
            (y + shift.y) * volDims[1] +
            (z + shift.z) * volDims[2]];
        float vd = tsdfToFloat(voxeld.tsdf);

        if (voxeld.weight != 0 && vd != 1.f)
        {
            if ((v0 > 0 && vd < 0) || (v0 < 0 && vd > 0))
            {
                //linearly interpolate coordinate
                float Vn = Vc + voxelSize;
                float dinv = 1.f / (abs(v0) + abs(vd));
                float inter = (Vc * abs(vd) + Vn * abs(v0)) * dinv;

                Point3f p(shift.x ? inter : V.x,
                    shift.y ? inter : V.y,
                    shift.z ? inter : V.z);
                {
                    points.push_back(toPtype(pose * p));
                    if (needNormals)
                        normals.push_back(toPtype(pose.rotation() *
                            getNormalColorVoxel(volume, volDims, neighbourCoords, volResolution, p * voxelSizeInv)));
                    if (needColors)
                        colors.push_back(toPtype(pose.rotation() *
                            getColorVoxel(volume, volDims, neighbourCoords, volResolution, voxelSizeInv, p * voxelSizeInv)));
                }
            }
        }
    }
}

void fetchPointsNormalsFromColorTsdfVolumeUnit(const VolumeSettings& settings, InputArray _volume,
    OutputArray _points, OutputArray _normals)
{
    fetchPointsNormalsColorsFromColorTsdfVolumeUnit(settings, _volume, _points, _normals, noArray());
}

void fetchPointsNormalsColorsFromColorTsdfVolumeUnit(const VolumeSettings& settings, InputArray _volume,
    OutputArray _points, OutputArray _normals, OutputArray _colors)
{
    if (!_points.needed())
        return;
    const Mat volume = _volume.getMat();

    Matx44f _pose;
    settings.getVolumePose(_pose);
    const Affine3f pose = Affine3f(_pose);
    float voxelSize = settings.getVoxelSize();
    float voxelSizeInv = 1.f / settings.getVoxelSize();

    const Vec4i volDims;
    settings.getVolumeStrides(volDims);
    const Vec8i neighbourCoords = Vec8i(
        volDims.dot(Vec4i(0, 0, 0)),
        volDims.dot(Vec4i(0, 0, 1)),
        volDims.dot(Vec4i(0, 1, 0)),
        volDims.dot(Vec4i(0, 1, 1)),
        volDims.dot(Vec4i(1, 0, 0)),
        volDims.dot(Vec4i(1, 0, 1)),
        volDims.dot(Vec4i(1, 1, 0)),
        volDims.dot(Vec4i(1, 1, 1))
    );

    Vec3i resolution;
    settings.getVolumeResolution(resolution);
    const Point3i volResolution = Point3i(resolution);

    bool needNormals = _normals.needed();
    bool needColors  = _colors.needed();

    std::vector<std::vector<ptype>> pVecs, nVecs, cVecs;
    Range fetchRange(0, volResolution.x);
    const int nstripes = -1;
    const RGBTsdfVoxel* volDataStart = volume.ptr<RGBTsdfVoxel>();
    Mutex mutex;
    auto FetchPointsNormalsInvoker = [&](const Range& range) {

        std::vector<ptype> points, normals, colors;
        for (int x = range.start; x < range.end; x++)
        {
            const RGBTsdfVoxel* volDataX = volDataStart + x * volDims[0];
            for (int y = 0; y < volResolution.y; y++)
            {
                const RGBTsdfVoxel* volDataY = volDataX + y * volDims[1];
                for (int z = 0; z < volResolution.z; z++)
                {
                    const RGBTsdfVoxel& voxel0 = volDataY[z * volDims[2]];
                    float v0 = tsdfToFloat(voxel0.tsdf);
                    if (voxel0.weight != 0 && v0 != 1.f)
                    {
                        Point3f V(Point3f((float)x + 0.5f, (float)y + 0.5f, (float)z + 0.5f) * voxelSize);

                        coord(volume, volDataStart, points, normals, colors, volResolution, volDims, neighbourCoords, pose, voxelSize, voxelSizeInv, needNormals, needColors, x, y, z, V, v0, 0);
                        coord(volume, volDataStart, points, normals, colors, volResolution, volDims, neighbourCoords, pose, voxelSize, voxelSizeInv, needNormals, needColors, x, y, z, V, v0, 1);
                        coord(volume, volDataStart, points, normals, colors, volResolution, volDims, neighbourCoords, pose, voxelSize, voxelSizeInv, needNormals, needColors, x, y, z, V, v0, 2);

                    } // if voxel is not empty
                }
            }
        }
        AutoLock al(mutex);
        pVecs.push_back(points);
        nVecs.push_back(normals);
        cVecs.push_back(colors);
    };

    parallel_for_(fetchRange, FetchPointsNormalsInvoker, nstripes);

    std::vector<ptype> points, normals, colors;
    for (size_t i = 0; i < pVecs.size(); i++)
    {
        points.insert(points.end(), pVecs[i].begin(), pVecs[i].end());
        normals.insert(normals.end(), nVecs[i].begin(), nVecs[i].end());
        colors.insert(colors.end(), cVecs[i].begin(), cVecs[i].end());
    }

    _points.create((int)points.size(), 1, POINT_TYPE);
    if (!points.empty())
        Mat((int)points.size(), 1, POINT_TYPE, &points[0]).copyTo(_points.getMat());

    if (_normals.needed())
    {
        _normals.create((int)normals.size(), 1, POINT_TYPE);
        if (!normals.empty())
            Mat((int)normals.size(), 1, POINT_TYPE, &normals[0]).copyTo(_normals.getMat());
    }

    if (_colors.needed())
    {
        _colors.create((int)colors.size(), 1, COLOR_TYPE);
        if (!colors.empty())
            Mat((int)colors.size(), 1, COLOR_TYPE, &colors[0]).copyTo(_colors.getMat());
    }
}


} // namespace cv
