// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// Partially rewritten from https://github.com/Nerei/kinfu_remake
// Copyright(c) 2012, Anatoly Baksheev. All rights reserved.

#include "../precomp.hpp"
#include "tsdf_functions.hpp"
#include "opencl_kernels_3d.hpp"

namespace cv {

void preCalculationPixNorm(Size size, const Intr& intrinsics, Mat& pixNorm)
{
    CV_TRACE_FUNCTION();

    Point2f fl(intrinsics.fx, intrinsics.fy);
    Point2f pp(intrinsics.cx, intrinsics.cy);
    pixNorm = Mat(size.height, size.width, CV_32F);
    std::vector<float> x(size.width);
    std::vector<float> y(size.height);
    for (int i = 0; i < size.width; i++)
        x[i] = (i - pp.x) / fl.x;
    for (int i = 0; i < size.height; i++)
        y[i] = (i - pp.y) / fl.y;

    for (int i = 0; i < size.height; i++)
    {
        for (int j = 0; j < size.width; j++)
        {
            pixNorm.at<float>(i, j) = sqrtf(x[j] * x[j] + y[i] * y[i] + 1.0f);
        }
    }

}

#ifdef HAVE_OPENCL
void ocl_preCalculationPixNorm(Size size, const Intr& intrinsics, UMat& pixNorm)
{
    // calculating this on CPU then uploading to GPU is faster than calculating this on GPU
    Mat cpuPixNorm;
    preCalculationPixNorm(size, intrinsics, cpuPixNorm);
    cpuPixNorm.copyTo(pixNorm);

}
#endif

// Integrate

void integrateTsdfVolumeUnit(const VolumeSettings& settings, const Matx44f& cameraPose,
                             InputArray _depth, InputArray _pixNorms, InputArray _volume)
{
    Matx44f volumePose;
    settings.getVolumePose(volumePose);
    integrateTsdfVolumeUnit(settings, volumePose, cameraPose, _depth, _pixNorms, _volume);
}


void integrateTsdfVolumeUnit(const VolumeSettings& settings, const Matx44f& volumePose, const Matx44f& cameraPose,
                             InputArray _depth, InputArray _pixNorms, InputArray _volume)
{
    Depth depth = _depth.getMat();
    Mat volume = _volume.getMat();
    Mat pixNorms = _pixNorms.getMat();

    TsdfVoxel* volDataStart = volume.ptr<TsdfVoxel>();

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
    const Intr::Projector proj = Intr(intr).makeProjector();
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
        v_float32x4 vfxy(proj.fx, proj.fy, 0.f, 0.f), vcxy(proj.cx, proj.cy, 0.f, 0.f);
        const v_float32x4 upLimits = v_cvt_f32(v_int32x4(depth.cols - 1, depth.rows - 1, 0, 0));

        for (int x = range.start; x < range.end; x++)
        {
            TsdfVoxel* volDataX = volDataStart + x * volStrides[0];
            for (int y = 0; y < volResolution.y; y++)
            {
                TsdfVoxel* volDataY = volDataX + y * volStrides[1];
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
                        if (v_check_all(v_gt(vall, v_setzero_f32())))
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
                    // float pixNorm = sqrt(v_reduce_sum(camPixVec*camPixVec));
                    // difference between distances of point and of surface to camera
                    float sdf = pixNorm * (v * dfac - zCamSpace);
                    // possible alternative is:
                    // kftype sdf = norm(camSpacePt)*(v*dfac/camSpacePt.z - 1);

                    if (sdf >= -truncDist)
                    {
                        TsdfType tsdf = floatToTsdf(fmin(1.f, sdf * truncDistInv));

                        TsdfVoxel& voxel = volDataY[z * volStrides[2]];
                        WeightType& weight = voxel.weight;
                        TsdfType& value = voxel.tsdf;

                        // update TSDF
                        value = floatToTsdf((tsdfToFloat(value) * weight + tsdfToFloat(tsdf)) / (weight + 1));
                        weight = (weight + 1) < maxWeight ? (weight + 1) : (WeightType)maxWeight;
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
            TsdfVoxel* volDataX = volDataStart + x * volStrides[0];
            for (int y = 0; y < volResolution.y; y++)
            {
                TsdfVoxel* volDataY = volDataX + y * volStrides[1];
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
                    Point2f projected = proj(camSpacePt, camPixVec);

                    depthType v = bilinearDepth(depth, projected);
                    if (v == 0) {
                        continue;
                    }
                    int _u = (int)projected.x;
                    int _v = (int)projected.y;
                    if (!(_u >= 0 && _u < depth.cols && _v >= 0 && _v < depth.rows))
                        continue;

                    float pixNorm = pixNorms.at<float>(_v, _u);

                    // difference between distances of point and of surface to camera
                    float sdf = pixNorm * (v * dfac - camSpacePt.z);
                    // possible alternative is:
                    // kftype sdf = norm(camSpacePt)*(v*dfac/camSpacePt.z - 1);
                    if (sdf >= -truncDist)
                    {
                        TsdfType tsdf = floatToTsdf(fmin(1.f, sdf * truncDistInv));

                        TsdfVoxel& voxel = volDataY[z * volStrides[2]];
                        WeightType& weight = voxel.weight;
                        TsdfType& value = voxel.tsdf;

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
    //IntegrateInvoker(integrateRange);
}

#ifdef HAVE_OPENCL
void ocl_integrateTsdfVolumeUnit(const VolumeSettings& settings, const Matx44f& cameraPose,
                                 InputArray _depth, InputArray _pixNorms, InputArray _volume)
{
    CV_TRACE_FUNCTION();
    CV_Assert(!_depth.empty());

    UMat depth = _depth.getUMat();
    UMat volume = _volume.getUMat();
    UMat pixNorms = _pixNorms.getUMat();

    String errorStr;
    String name = "integrate";
    ocl::ProgramSource source = ocl::_3d::tsdf_oclsrc;
    String options = "-cl-mad-enable";
    ocl::Kernel k;
    k.create(name.c_str(), source, options, &errorStr);

    if (k.empty())
        throw std::runtime_error("Failed to create kernel: " + errorStr);

    Matx44f _pose;
    settings.getVolumePose(_pose);
    const Affine3f pose = Affine3f(_pose);
    UMat vol2camGpu;
    Affine3f vol2cam(Affine3f(cameraPose.inv()) * pose);
    Mat(vol2cam.matrix).copyTo(vol2camGpu);

    float dfac = 1.f / settings.getDepthFactor();
    Vec3i resolution;
    settings.getVolumeResolution(resolution);
    const Point3i volResolution = Point3i(resolution);
    Vec4i volResGpu(volResolution.x, volResolution.y, volResolution.z);
    Matx33f intr;
    settings.getCameraIntegrateIntrinsics(intr);
    Intr intrinsics(intr);
    Vec2f fxy(intrinsics.fx, intrinsics.fy), cxy(intrinsics.cx, intrinsics.cy);
    const Vec4i volDims;
    settings.getVolumeStrides(volDims);

    const float voxelSize = settings.getVoxelSize();
    const float truncatedDistance = settings.getTsdfTruncateDistance();
    const int maxWeight = settings.getMaxWeight();

    // TODO: optimization possible
    // Use sampler for depth (mask needed)
    k.args(ocl::KernelArg::ReadOnly(depth),
        ocl::KernelArg::PtrReadWrite(volume),
        ocl::KernelArg::PtrReadOnly(vol2camGpu),
        voxelSize,
        volResGpu.val,
        volDims.val,
        fxy.val,
        cxy.val,
        dfac,
        truncatedDistance,
        maxWeight,
        ocl::KernelArg::PtrReadOnly(pixNorms));

    size_t globalSize[2];
    globalSize[0] = (size_t)volResolution.x;
    globalSize[1] = (size_t)volResolution.y;

    if (!k.run(2, globalSize, NULL, true))
        throw std::runtime_error("Failed to run kernel");

}
#endif


// Raycast

#if USE_INTRINSICS
// all coordinate checks should be done in inclosing cycle
inline float interpolateTsdfVoxel(const Mat& volume,
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
    const TsdfVoxel* volData = volume.ptr<TsdfVoxel>();

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

inline float interpolateTsdfVoxel( const Mat& volume,
    const Vec4i& volDims, const Vec8i& neighbourCoords,
    const Point3f& _p)
{
    v_float32x4 p(_p.x, _p.y, _p.z, 0);
    return interpolateTsdfVoxel(volume, volDims, neighbourCoords, p);
}

#else
inline float interpolateTsdfVoxel( const Mat& volume,
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
    const TsdfVoxel* volData = volume.ptr<TsdfVoxel>();

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
inline v_float32x4 getNormalVoxel( const Mat& volume,
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
    const TsdfVoxel* volData = volume.ptr<TsdfVoxel>();

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

inline Point3f getNormalVoxel( const Mat& volume,
    const Vec4i& volDims, const Vec8i& neighbourCoords, const Point3i volResolution,
    const Point3f& _p)
{
    v_float32x4 p(_p.x, _p.y, _p.z, 0.f);
    v_float32x4 result = getNormalVoxel(volume, volDims, neighbourCoords, volResolution, p);
    float CV_DECL_ALIGNED(16) ares[4];
    v_store_aligned(ares, result);
    return Point3f(ares[0], ares[1], ares[2]);
}
#else
inline Point3f getNormalVoxel( const Mat& volume,
    const Vec4i& volDims, const Vec8i& neighbourCoords, const Point3i volResolution,
    const Point3f& p)
{
    int xdim = volDims[0], ydim = volDims[1], zdim = volDims[2];
    const TsdfVoxel* volData = volume.ptr<TsdfVoxel>();

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

void raycastTsdfVolumeUnit(const VolumeSettings& settings, const Matx44f& cameraPose,
                           int height, int width, InputArray intr,
                           InputArray _volume, OutputArray _points, OutputArray _normals)
{
    CV_TRACE_FUNCTION();

    const Size frameSize(width, height);
    CV_Assert(frameSize.area() > 0);

    Matx33f mintr(intr.getMat());

    _points.create(frameSize, POINT_TYPE);
    _normals.create(frameSize, POINT_TYPE);

    Points points = _points.getMat();
    Normals normals = _normals.getMat();

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
    const Intr::Reprojector reproj = Intr(mintr).makeReprojector();
    float tstep = settings.getTsdfTruncateDistance() * settings.getRaycastStepFactor();

    Range raycastRange = Range(0, points.rows);
    //TODO::  swap realization, they are missplaced :)
#if USE_INTRINSICS
    auto RaycastInvoker = [&](const Range& range)
    {
        const v_float32x4 vfxy(reproj.fxinv, reproj.fyinv, 0, 0);
        const v_float32x4 vcxy(reproj.cx, reproj.cy, 0, 0);

        const float(&cm)[16] = cam2vol.matrix.val;
        const v_float32x4 camRot0(cm[0], cm[4], cm[8], 0);
        const v_float32x4 camRot1(cm[1], cm[5], cm[9], 0);
        const v_float32x4 camRot2(cm[2], cm[6], cm[10], 0);
        const v_float32x4 camTrans(cm[3], cm[7], cm[11], 0);

        const v_float32x4 boxDown(boxMin.x, boxMin.y, boxMin.z, 0.f);
        const v_float32x4 boxUp(boxMax.x, boxMax.y, boxMax.z, 0.f);

        const v_float32x4 invVoxelSize = v_float32x4(voxelSizeInv, voxelSizeInv, voxelSizeInv, 1.f);

        const float(&vm)[16] = vol2cam.matrix.val;
        const v_float32x4 volRot0(vm[0], vm[4], vm[8], 0);
        const v_float32x4 volRot1(vm[1], vm[5], vm[9], 0);
        const v_float32x4 volRot2(vm[2], vm[6], vm[10], 0);
        const v_float32x4 volTrans(vm[3], vm[7], vm[11], 0);

        for (int y = range.start; y < range.end; y++)
        {
            ptype* ptsRow = points[y];
            ptype* nrmRow = normals[y];

            for (int x = 0; x < points.cols; x++)
            {
                v_float32x4 point = nanv, normal = nanv;

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
                    float f = interpolateTsdfVoxel(volume, volDims, neighbourCoords, next);
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

                        fnext = tsdfToFloat(volume.at<TsdfVoxel>(coord).tsdf);
                        if (fnext != f)
                        {
                            fnext = interpolateTsdfVoxel(volume, volDims, neighbourCoords, next);

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
                        float ft = interpolateTsdfVoxel(volume, volDims, neighbourCoords, tp);
                        float ftdt = interpolateTsdfVoxel(volume, volDims, neighbourCoords, next);
                        float ts = tmin + tstep * (steps - ft / (ftdt - ft));

                        // avoid division by zero
                        if (!cvIsNaN(ts) && !cvIsInf(ts))
                        {
                            v_float32x4 pv = v_add(orig, v_mul(dir, v_setall_f32(ts)));
                            v_float32x4 nv = getNormalVoxel(volume, volDims, neighbourCoords, volResolution, pv);

                            if (!isNaN(nv))
                            {
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

            for (int x = 0; x < points.cols; x++)
            {
                Point3f point = nan3, normal = nan3;

                Point3f orig = camTrans;
                // direction through pixel in volume space
                Point3f dir = normalize(Vec3f(camRot * reproj(Point3f(float(x), float(y), 1.f))));

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
                    float f = interpolateTsdfVoxel(volume, volDims, neighbourCoords, next);
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
                        fnext = tsdfToFloat(volume.at<TsdfVoxel>(ix * xdim + iy * ydim + iz * zdim).tsdf);
                        if (fnext != f)
                        {
                            fnext = interpolateTsdfVoxel(volume, volDims, neighbourCoords, next);
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
                        float ft = interpolateTsdfVoxel(volume, volDims, neighbourCoords, tp);
                        float ftdt = interpolateTsdfVoxel(volume, volDims, neighbourCoords, next);
                        // float t = tmin + steps*tstep;
                        // float ts = t - tstep*ft/(ftdt - ft);
                        float ts = tmin + tstep * (steps - ft / (ftdt - ft));

                        // avoid division by zero
                        if (!cvIsNaN(ts) && !cvIsInf(ts))
                        {
                            Point3f pv = (orig + dir * ts);
                            Point3f nv = getNormalVoxel(volume, volDims, neighbourCoords, volResolution, pv);

                            if (!isNaN(nv))
                            {
                                //convert pv and nv to camera space
                                normal = volRot * nv;
                                // interpolation optimized a little
                                point = vol2cam * (pv * voxelSize);
                            }
                        }
                    }
                }
                ptsRow[x] = toPtype(point);
                nrmRow[x] = toPtype(normal);
            }
        }
    };

#endif
    parallel_for_(raycastRange, RaycastInvoker);
}


#ifdef HAVE_OPENCL
void ocl_raycastTsdfVolumeUnit(const VolumeSettings& settings, const Matx44f& cameraPose,
                               int height, int width, InputArray intr,
                               InputArray _volume, OutputArray _points, OutputArray _normals)
{
    CV_TRACE_FUNCTION();

    const Size frameSize(width, height);
    CV_Assert(frameSize.area() > 0);

    Matx33f mintr(intr.getMat());

    String errorStr;
    String name = "raycast";
    ocl::ProgramSource source = ocl::_3d::tsdf_oclsrc;
    String options = "-cl-mad-enable";
    ocl::Kernel k;
    k.create(name.c_str(), source, options, &errorStr);

    if (k.empty())
        throw std::runtime_error("Failed to create kernel: " + errorStr);

    _points.create(frameSize, CV_32FC4);
    _normals.create(frameSize, CV_32FC4);

    UMat points = _points.getUMat();
    UMat normals = _normals.getUMat();

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

    Matx44f _pose;
    settings.getVolumePose(_pose);
    const Affine3f pose = Affine3f(_pose);

    UMat vol2camGpu, cam2volGpu;
    Affine3f vol2cam = Affine3f(cameraPose.inv()) * pose;
    Affine3f cam2vol = pose.inv() * Affine3f(cameraPose);
    Mat(cam2vol.matrix).copyTo(cam2volGpu);
    Mat(vol2cam.matrix).copyTo(vol2camGpu);

    Intr intrinsics(mintr);
    Intr::Reprojector r = intrinsics.makeReprojector();

    const UMat volume = _volume.getUMat();
    float voxelSize = settings.getVoxelSize();
    float raycastStepFactor = settings.getRaycastStepFactor();
    float truncatedDistance = settings.getTsdfTruncateDistance();

    // We do subtract voxel size to minimize checks after
    // Note: origin of volume coordinate is placed
    // in the center of voxel (0,0,0), not in the corner of the voxel!
    Vec4f boxMin, boxMax(volSize.x - voxelSize,
        volSize.y - voxelSize,
        volSize.z - voxelSize);
    Vec2f finv(r.fxinv, r.fyinv), cxy(r.cx, r.cy);
    float tstep = truncatedDistance * raycastStepFactor;

    Vec4i volResGpu(volResolution.x, volResolution.y, volResolution.z);

    k.args(ocl::KernelArg::WriteOnlyNoSize(points),
        ocl::KernelArg::WriteOnlyNoSize(normals),
        frameSize,
        ocl::KernelArg::PtrReadOnly(volume),
        ocl::KernelArg::PtrReadOnly(vol2camGpu),
        ocl::KernelArg::PtrReadOnly(cam2volGpu),
        finv.val, cxy.val,
        boxMin.val, boxMax.val,
        tstep,
        voxelSize,
        volResGpu.val,
        volDims.val,
        neighbourCoords.val);

    size_t globalSize[2];
    globalSize[0] = (size_t)frameSize.width;
    globalSize[1] = (size_t)frameSize.height;

    if (!k.run(2, globalSize, NULL, true))
        throw std::runtime_error("Failed to run kernel");
}
#endif

// Fetch

void fetchNormalsFromTsdfVolumeUnit(const VolumeSettings& settings, InputArray _volume, InputArray _points, OutputArray _normals)
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
    Matx33f r = pose.rotation();
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
            n = r * getNormalVoxel(volume, volDims, neighbourCoords, volResolution, voxPt);
        }
        normals(position[0], position[1]) = toPtype(n);
    };
    points.forEach(PushNormals);
}

#ifdef HAVE_OPENCL
void ocl_fetchNormalsFromTsdfVolumeUnit(const VolumeSettings& settings, InputArray _volume, InputArray _points, OutputArray _normals)
{
    CV_TRACE_FUNCTION();
    CV_Assert(!_points.empty());
    if (!_normals.needed())
        return;

    UMat points = _points.getUMat();
    CV_Assert(points.type() == POINT_TYPE);

    _normals.createSameSize(_points, POINT_TYPE);
    UMat normals = _normals.getUMat();

    const UMat volume = _volume.getUMat();

    Matx44f _pose;
    settings.getVolumePose(_pose);
    const Affine3f pose = Affine3f(_pose);
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

    String errorStr;
    String name = "getNormals";
    ocl::ProgramSource source = ocl::_3d::tsdf_oclsrc;
    String options = "-cl-mad-enable";
    ocl::Kernel k;
    k.create(name.c_str(), source, options, &errorStr);

    if (k.empty())
        throw std::runtime_error("Failed to create kernel: " + errorStr);

    UMat volPoseGpu, invPoseGpu;
    Mat(pose.matrix).copyTo(volPoseGpu);
    Mat(pose.inv().matrix).copyTo(invPoseGpu);
    Vec4i volResGpu(volResolution.x, volResolution.y, volResolution.z);
    Size frameSize = points.size();

    k.args(ocl::KernelArg::ReadOnlyNoSize(points),
           ocl::KernelArg::WriteOnlyNoSize(normals),
           frameSize,
           ocl::KernelArg::PtrReadOnly(volume),
           ocl::KernelArg::PtrReadOnly(volPoseGpu),
           ocl::KernelArg::PtrReadOnly(invPoseGpu),
           voxelSizeInv,
           volResGpu.val,
           volDims.val,
           neighbourCoords.val);

    size_t globalSize[2];
    globalSize[0] = (size_t)points.cols;
    globalSize[1] = (size_t)points.rows;

    if (!k.run(2, globalSize, NULL, true))
        throw std::runtime_error("Failed to run kernel");

}
#endif

inline void coord(const Mat& volume, const TsdfVoxel* volDataStart, std::vector<ptype>& points, std::vector<ptype>& normals,
                  const Point3i volResolution, const Vec4i volDims, const Vec8i neighbourCoords, const Affine3f pose,
                  const float voxelSize, const float voxelSizeInv, bool needNormals, int x, int y, int z, Point3f V, float v0, int axis)
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
        const TsdfVoxel &voxeld = volDataStart[(x + shift.x) * volDims[0] +
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
                                          getNormalVoxel(volume, volDims, neighbourCoords, volResolution, p * voxelSizeInv)));
                }
            }
        }
    }
}


void fetchPointsNormalsFromTsdfVolumeUnit(const VolumeSettings& settings, InputArray _volume, OutputArray _points, OutputArray _normals)
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

    std::vector<std::vector<ptype>> pVecs, nVecs;
    Range fetchRange(0, volResolution.x);
    const int nstripes = -1;
    const TsdfVoxel* volDataStart = volume.ptr<TsdfVoxel>();
    Mutex mutex;

    auto FetchPointsNormalsInvoker = [&](const Range& range) {
        std::vector<ptype> points, normals;
        for (int x = range.start; x < range.end; x++)
        {
            const TsdfVoxel* volDataX = volDataStart + x * volDims[0];
            for (int y = 0; y < volResolution.y; y++)
            {
                const TsdfVoxel* volDataY = volDataX + y * volDims[1];
                for (int z = 0; z < volResolution.z; z++)
                {
                    const TsdfVoxel& voxel0 = volDataY[z * volDims[2]];
                    float v0 = tsdfToFloat(voxel0.tsdf);
                    if (voxel0.weight != 0 && v0 != 1.f)
                    {
                        Point3f V(Point3f((float)x + 0.5f, (float)y + 0.5f, (float)z + 0.5f) * voxelSize);

                        coord(volume, volDataStart, points, normals, volResolution, volDims, neighbourCoords, pose, voxelSize, voxelSizeInv, needNormals, x, y, z, V, v0, 0);
                        coord(volume, volDataStart, points, normals, volResolution, volDims, neighbourCoords, pose, voxelSize, voxelSizeInv, needNormals, x, y, z, V, v0, 1);
                        coord(volume, volDataStart, points, normals, volResolution, volDims, neighbourCoords, pose, voxelSize, voxelSizeInv, needNormals, x, y, z, V, v0, 2);

                    } // if voxel is not empty
                }
            }
        }

        AutoLock al(mutex);
        pVecs.push_back(points);
        nVecs.push_back(normals);
    };

    parallel_for_(fetchRange, FetchPointsNormalsInvoker, nstripes);

    std::vector<ptype> points, normals;
    for (size_t i = 0; i < pVecs.size(); i++)
    {
        points.insert(points.end(), pVecs[i].begin(), pVecs[i].end());
        normals.insert(normals.end(), nVecs[i].begin(), nVecs[i].end());
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

}


#ifdef HAVE_OPENCL
void ocl_fetchPointsNormalsFromTsdfVolumeUnit(const VolumeSettings& settings, InputArray _volume, OutputArray points, OutputArray normals)
{
    CV_TRACE_FUNCTION();

    if (!points.needed())
        return;


    const UMat volume = _volume.getUMat();

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


    bool needNormals = normals.needed();

    // 1. scan to count points in each group and allocate output arrays

    ocl::Kernel kscan;

    String errorStr;
    ocl::ProgramSource source = ocl::_3d::tsdf_oclsrc;
    String options = "-cl-mad-enable";

    kscan.create("scanSize", source, options, &errorStr);

    if (kscan.empty())
        throw std::runtime_error("Failed to create kernel: " + errorStr);

    size_t globalSize[3];
    globalSize[0] = (size_t)volResolution.x;
    globalSize[1] = (size_t)volResolution.y;
    globalSize[2] = (size_t)volResolution.z;

    const ocl::Device& device = ocl::Device::getDefault();
    size_t wgsLimit = device.maxWorkGroupSize();
    size_t memSize = device.localMemSize();
    // local mem should keep a point (and a normal) for each thread in a group
    // use 4 float per each point and normal
    size_t elemSize = (sizeof(float) * 4) * (needNormals ? 2 : 1);
    const size_t lcols = 8;
    const size_t lrows = 8;
    size_t lplanes = min(memSize / elemSize, wgsLimit) / lcols / lrows;
    lplanes = roundDownPow2(lplanes);
    size_t localSize[3] = { lcols, lrows, lplanes };
    Vec3i ngroups((int)divUp(globalSize[0], (unsigned int)localSize[0]),
        (int)divUp(globalSize[1], (unsigned int)localSize[1]),
        (int)divUp(globalSize[2], (unsigned int)localSize[2]));

    const size_t counterSize = sizeof(int);
    size_t lszscan = localSize[0] * localSize[1] * localSize[2] * counterSize;

    const int gsz[3] = { ngroups[2], ngroups[1], ngroups[0] };
    UMat groupedSum(3, gsz, CV_32S, Scalar(0));

    UMat volPoseGpu;
    Mat(pose.matrix).copyTo(volPoseGpu);
    Vec4i volResGpu(volResolution.x, volResolution.y, volResolution.z);

    kscan.args(ocl::KernelArg::PtrReadOnly(volume),
        volResGpu.val,
        volDims.val,
        neighbourCoords.val,
        ocl::KernelArg::PtrReadOnly(volPoseGpu),
        voxelSize,
        voxelSizeInv,
        ocl::KernelArg::Local(lszscan),
        ocl::KernelArg::WriteOnlyNoSize(groupedSum));

    if (!kscan.run(3, globalSize, localSize, true))
        throw std::runtime_error("Failed to run kernel");

    Mat groupedSumCpu = groupedSum.getMat(ACCESS_READ);
    int gpuSum = (int)sum(groupedSumCpu)[0];
    // should be no CPU copies when new kernel is executing
    groupedSumCpu.release();

    // 2. fill output arrays according to per-group points count

    points.create(gpuSum, 1, POINT_TYPE);
    UMat pts = points.getUMat();
    UMat nrm;
    if (needNormals)
    {
        normals.create(gpuSum, 1, POINT_TYPE);
        nrm = normals.getUMat();
    }
    else
    {
        // it won't be accessed but empty args are forbidden
        nrm = UMat(1, 1, POINT_TYPE);
    }

    if (gpuSum)
    {
        ocl::Kernel kfill;
        kfill.create("fillPtsNrm", source, options, &errorStr);

        if (kfill.empty())
            throw std::runtime_error("Failed to create kernel: " + errorStr);

        UMat atomicCtr(1, 1, CV_32S, Scalar(0));

        // mem size to keep pts (and normals optionally) for all work-items in a group
        size_t lszfill = localSize[0] * localSize[1] * localSize[2] * elemSize;

        kfill.args(ocl::KernelArg::PtrReadOnly(volume),
            volResGpu.val,
            volDims.val,
            neighbourCoords.val,
            ocl::KernelArg::PtrReadOnly(volPoseGpu),
            voxelSize,
            voxelSizeInv,
            ((int)needNormals),
            ocl::KernelArg::Local(lszfill),
            ocl::KernelArg::PtrReadWrite(atomicCtr),
            ocl::KernelArg::ReadOnlyNoSize(groupedSum),
            ocl::KernelArg::WriteOnlyNoSize(pts),
            ocl::KernelArg::WriteOnlyNoSize(nrm)
        );

        if (!kfill.run(3, globalSize, localSize, true))
            throw std::runtime_error("Failed to run kernel");
    }
}
#endif



} // namespace cv
