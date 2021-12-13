// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// Partially rewritten from https://github.com/Nerei/kinfu_remake
// Copyright(c) 2012, Anatoly Baksheev. All rights reserved.

#include "../precomp.hpp"
#include "tsdf_functions.hpp"
#include "opencl_kernels_3d.hpp"

namespace cv {

cv::Mat preCalculationPixNorm(Size size, const Intr& intrinsics)
{
    Point2f fl(intrinsics.fx, intrinsics.fy);
    Point2f pp(intrinsics.cx, intrinsics.cy);
    Mat pixNorm(size.height, size.width, CV_32F);
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
    return pixNorm;
}

#ifdef HAVE_OPENCL
cv::UMat preCalculationPixNormGPU(Size size, const Intr& intrinsics)
{
    // calculating this on CPU then uploading to GPU is faster than calculating this on GPU
    Mat cpuPixNorm = preCalculationPixNorm(size, intrinsics);

    UMat pixNorm(size, CV_32F);
    cpuPixNorm.copyTo(pixNorm);

    return pixNorm;
}
#endif


void integrateVolumeUnit(
    float truncDist, float voxelSize, int maxWeight,
    cv::Matx44f _pose, Point3i volResolution, Vec4i volStrides,
    InputArray _depth, float depthFactor, const cv::Matx44f& cameraPose,
    const cv::Intr& intrinsics, InputArray _pixNorms, InputArray _volume)
{
    CV_TRACE_FUNCTION();

    CV_Assert(_depth.type() == DEPTH_TYPE);
    CV_Assert(!_depth.empty());
    cv::Affine3f vpose(_pose);
    Depth depth = _depth.getMat();

    Range integrateRange(0, volResolution.x);

    Mat volume = _volume.getMat();
    Mat pixNorms = _pixNorms.getMat();
    const Intr::Projector proj(intrinsics.makeProjector());
    const cv::Affine3f vol2cam(Affine3f(cameraPose.inv()) * vpose);
    const float truncDistInv(1.f / truncDist);
    const float dfac(1.f / depthFactor);
    TsdfVoxel* volDataStart = volume.ptr<TsdfVoxel>();;

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
                    camSpacePt += zStep;

                    float zCamSpace = v_reinterpret_as_f32(v_rotate_right<2>(v_reinterpret_as_u32(camSpacePt))).get0();
                    if (zCamSpace <= 0.f)
                        continue;

                    v_float32x4 camPixVec = camSpacePt / v_setall_f32(zCamSpace);
                    v_float32x4 projected = v_muladd(camPixVec, vfxy, vcxy);
                    // leave only first 2 lanes
                    projected = v_reinterpret_as_f32(v_reinterpret_as_u32(projected) &
                        v_uint32x4(0xFFFFFFFF, 0xFFFFFFFF, 0, 0));

                    depthType v;
                    // bilinearly interpolate depth at projected
                    {
                        const v_float32x4& pt = projected;
                        // check coords >= 0 and < imgSize
                        v_uint32x4 limits = v_reinterpret_as_u32(pt < v_setzero_f32()) |
                            v_reinterpret_as_u32(pt >= upLimits);
                        limits = limits | v_rotate_right<1>(limits);
                        if (limits.get0())
                            continue;

                        // xi, yi = floor(pt)
                        v_int32x4 ip = v_floor(pt);
                        v_int32x4 ipshift = ip;
                        int xi = ipshift.get0();
                        ipshift = v_rotate_right<1>(ipshift);
                        int yi = ipshift.get0();

                        const depthType* row0 = depth[yi + 0];
                        const depthType* row1 = depth[yi + 1];

                        // v001 = [v(xi + 0, yi + 0), v(xi + 1, yi + 0)]
                        v_float32x4 v001 = v_load_low(row0 + xi);
                        // v101 = [v(xi + 0, yi + 1), v(xi + 1, yi + 1)]
                        v_float32x4 v101 = v_load_low(row1 + xi);

                        v_float32x4 vall = v_combine_low(v001, v101);

                        // assume correct depth is positive
                        // don't fix missing data
                        if (v_check_all(vall > v_setzero_f32()))
                        {
                            v_float32x4 t = pt - v_cvt_f32(ip);
                            float tx = t.get0();
                            t = v_reinterpret_as_f32(v_rotate_right<1>(v_reinterpret_as_u32(t)));
                            v_float32x4 ty = v_setall_f32(t.get0());
                            // vx is y-interpolated between rows 0 and 1
                            v_float32x4 vx = v001 + ty * (v101 - v001);
                            float v0 = vx.get0();
                            vx = v_reinterpret_as_f32(v_rotate_right<1>(v_reinterpret_as_u32(vx)));
                            float v1 = vx.get0();
                            v = v0 + tx * (v1 - v0);
                        }
                        else
                            continue;
                    }

                    // norm(camPixVec) produces double which is too slow
                    int _u = (int)projected.get0();
                    int _v = (int)v_rotate_right<1>(projected).get0();
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
                        weight = (weight + 1) < maxWeight ? (weight + 1) : (WeightType) maxWeight;
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

                    int _u = projected.x;
                    int _v = projected.y;
                    if (!(_u >= 0 && _u < depth.cols && _v >= 0 && _v < depth.rows))
                        continue;
                    float pixNorm = pixNorms.at<float>(_v, _u);

                    // difference between distances of point and of surface to camera
                    float sdf = pixNorm * (v * dfac - camSpacePt.z);
                    // possible alternative is:
                    // kftype sdf = norm(camSpacePt)*(v*dfac/camSpacePt.z - 1);
                    if (sdf >= -truncDist)
                    {
                        //std::cout << sdf << " " << -truncDist  << " " << truncDistInv << std::endl;

                        TsdfType tsdf = floatToTsdf(fmin(1.f, sdf * truncDistInv));

                        TsdfVoxel& voxel = volDataY[z * volStrides[2]];
                        WeightType& weight = voxel.weight;
                        TsdfType& value = voxel.tsdf;

                        // update TSDF
                        value = floatToTsdf((tsdfToFloat(value) * weight + tsdfToFloat(tsdf)) / (weight + 1));
                        weight = min(int(weight + 1), int(maxWeight));
                    }
                }
            }
        }
    };
#endif

    parallel_for_(integrateRange, IntegrateInvoker);
}


void integrateRGBVolumeUnit(
    float truncDist, float voxelSize, int maxWeight,
    cv::Matx44f _pose, Point3i volResolution, Vec4i volStrides,
    InputArray _depth, InputArray _rgb, float depthFactor, const cv::Matx44f& cameraPose,
    const cv::Intr& depth_intrinsics, const cv::Intr& rgb_intrinsics, InputArray _pixNorms, InputArray _volume)
{
    CV_TRACE_FUNCTION();

    CV_Assert(_depth.type() == DEPTH_TYPE);
    CV_Assert(!_depth.empty());
    cv::Affine3f vpose(_pose);
    Depth depth = _depth.getMat();
    Colors color = _rgb.getMat();
    Range integrateRange(0, volResolution.x);

    Mat volume = _volume.getMat();
    Mat pixNorms = _pixNorms.getMat();
    const Intr::Projector projDepth(depth_intrinsics.makeProjector());
    const Intr::Projector projRGB(rgb_intrinsics);
    const cv::Affine3f vol2cam(Affine3f(cameraPose.inv()) * vpose);
    const float truncDistInv(1.f / truncDist);
    const float dfac(1.f / depthFactor);
    RGBTsdfVoxel* volDataStart = volume.ptr<RGBTsdfVoxel>();

#if USE_INTRINSICS
    auto IntegrateInvoker = [&](const Range& range)
    {
        // zStep == vol2cam*(Point3f(x, y, 1)*voxelSize) - basePt;
        Point3f zStepPt = Point3f(vol2cam.matrix(0, 2),
            vol2cam.matrix(1, 2),
            vol2cam.matrix(2, 2)) * voxelSize;

        v_float32x4 zStep(zStepPt.x, zStepPt.y, zStepPt.z, 0);
        v_float32x4 vfxy(projDepth.fx, projDepth.fy, 0.f, 0.f), vcxy(projDepth.cx, projDepth.cy, 0.f, 0.f);
        v_float32x4 rgb_vfxy(projRGB.fx, projRGB.fy, 0.f, 0.f), rgb_vcxy(projRGB.cx, projRGB.cy, 0.f, 0.f);
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
                    camSpacePt += zStep;

                    float zCamSpace = v_reinterpret_as_f32(v_rotate_right<2>(v_reinterpret_as_u32(camSpacePt))).get0();
                    if (zCamSpace <= 0.f)
                        continue;

                    v_float32x4 camPixVec = camSpacePt / v_setall_f32(zCamSpace);
                    v_float32x4 projected = v_muladd(camPixVec, vfxy, vcxy);
                    // leave only first 2 lanes
                    projected = v_reinterpret_as_f32(v_reinterpret_as_u32(projected) &
                        v_uint32x4(0xFFFFFFFF, 0xFFFFFFFF, 0, 0));

                    depthType v;
                    // bilinearly interpolate depth at projected
                    {
                        const v_float32x4& pt = projected;
                        // check coords >= 0 and < imgSize
                        v_uint32x4 limits = v_reinterpret_as_u32(pt < v_setzero_f32()) |
                            v_reinterpret_as_u32(pt >= upLimits);
                        limits = limits | v_rotate_right<1>(limits);
                        if (limits.get0())
                            continue;

                        // xi, yi = floor(pt)
                        v_int32x4 ip = v_floor(pt);
                        v_int32x4 ipshift = ip;
                        int xi = ipshift.get0();
                        ipshift = v_rotate_right<1>(ipshift);
                        int yi = ipshift.get0();

                        const depthType* row0 = depth[yi + 0];
                        const depthType* row1 = depth[yi + 1];

                        // v001 = [v(xi + 0, yi + 0), v(xi + 1, yi + 0)]
                        v_float32x4 v001 = v_load_low(row0 + xi);
                        // v101 = [v(xi + 0, yi + 1), v(xi + 1, yi + 1)]
                        v_float32x4 v101 = v_load_low(row1 + xi);

                        v_float32x4 vall = v_combine_low(v001, v101);

                        // assume correct depth is positive
                        // don't fix missing data
                        if (v_check_all(vall > v_setzero_f32()))
                        {
                            v_float32x4 t = pt - v_cvt_f32(ip);
                            float tx = t.get0();
                            t = v_reinterpret_as_f32(v_rotate_right<1>(v_reinterpret_as_u32(t)));
                            v_float32x4 ty = v_setall_f32(t.get0());
                            // vx is y-interpolated between rows 0 and 1
                            v_float32x4 vx = v001 + ty * (v101 - v001);
                            float v0 = vx.get0();
                            vx = v_reinterpret_as_f32(v_rotate_right<1>(v_reinterpret_as_u32(vx)));
                            float v1 = vx.get0();
                            v = v0 + tx * (v1 - v0);
                        }
                        else
                            continue;
                    }

                    v_float32x4 projectedRGB = v_muladd(camPixVec, rgb_vfxy, rgb_vcxy);
                    // leave only first 2 lanes
                    projectedRGB = v_reinterpret_as_f32(v_reinterpret_as_u32(projected) &
                        v_uint32x4(0xFFFFFFFF, 0xFFFFFFFF, 0, 0));

                    // norm(camPixVec) produces double which is too slow
                    int _u = (int)projected.get0();
                    int _v = (int)v_rotate_right<1>(projected).get0();
                    int rgb_u = (int)projectedRGB.get0();
                    int rgb_v = (int)v_rotate_right<1>(projectedRGB).get0();

                    if (!(_u >= 0 && _u < depth.cols && _v >= 0 && _v < depth.rows &&
                        rgb_v >= 0 && rgb_v < color.rows && rgb_u >= 0 && rgb_u < color.cols))
                        continue;
                    float pixNorm = pixNorms.at<float>(_v, _u);
                    Vec4f colorRGB = color.at<Vec4f>(rgb_v, rgb_u);
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
                    Point2f projectedRGB = projRGB(camSpacePt, camPixVec);


                    depthType v = bilinearDepth(depth, projected);
                    if (v == 0) {
                        continue;
                    }

                    int _u = (int) projected.x;
                    int _v = (int) projected.y;

                    int rgb_u = (int) projectedRGB.x;
                    int rgb_v = (int) projectedRGB.y;

                    if (!(_v >= 0 && _v < depth.rows && _u >= 0 && _u < depth.cols  &&
                        rgb_v >= 0 && rgb_v < color.rows && rgb_u >= 0 && rgb_u < color.cols))
                        continue;

                    float pixNorm = pixNorms.at<float>(_v, _u);
                    Vec4f colorRGB = color.at<Vec4f>(rgb_v, rgb_u);
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
                        weight = WeightType( min(int(weight + 1), int(maxWeight)) );
                    }
                }
            }
        }
    };
#endif
    parallel_for_(integrateRange, IntegrateInvoker);
}


// Raycast

inline float interpolateVoxel(
    const VolumeSettings& settings, const Mat& volume,
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

inline Point3f getNormalVoxel(
    const VolumeSettings& settings, const Mat& volume,
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


void raycastVolumeUnit(const VolumeSettings& settings,
                   const Matx44f& cameraPose,
                   InputArray _volume,
                   OutputArray _points, OutputArray _normals)
{
    const Size frameSize(settings.getWidth(), settings.getHeight());
    CV_Assert(frameSize.area() > 0);

    _points.create(frameSize, POINT_TYPE);
    _normals.create(frameSize, POINT_TYPE);

    Points points = _points.getMat();
    Normals normals = _normals.getMat();


    const Vec4i volDims;
    settings.getVolumeDimentions(volDims);
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

    Matx33f intr;
    settings.getCameraIntrinsics(intr);

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
    float raycastStepFactor = settings.getRaycastStepFactor();
    const Intr::Reprojector reproj = Intr(intr).makeReprojector();
    float tstep = settings.getTruncatedDistance() * settings.getRaycastStepFactor();

    Range raycastRange = Range(0, points.rows);
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
                    float f = interpolateVoxel(settings, volume, volDims, neighbourCoords, next);
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
                            fnext = interpolateVoxel(settings, volume, volDims, neighbourCoords, next);
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
                        float ft = interpolateVoxel(settings, volume, volDims, neighbourCoords, tp);
                        float ftdt = interpolateVoxel(settings, volume, volDims, neighbourCoords, next);
                        // float t = tmin + steps*tstep;
                        // float ts = t - tstep*ft/(ftdt - ft);
                        float ts = tmin + tstep * (steps - ft / (ftdt - ft));

                        // avoid division by zero
                        if (!cvIsNaN(ts) && !cvIsInf(ts))
                        {
                            Point3f pv = (orig + dir * ts);
                            Point3f nv = getNormalVoxel(settings, volume, volDims, neighbourCoords, volResolution, pv);

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

    parallel_for_(raycastRange, RaycastInvoker);
}


} // namespace cv
