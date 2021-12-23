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


void integrateColorTsdfVolumeUnit(
    const VolumeSettings& settings, const Matx44f& volumePose, const Matx44f& cameraPose,
    InputArray _depth, InputArray _rgb, InputArray _pixNorms, InputArray _volume)
{
    std::cout << "integrateColorTsdfVolumeUnit()" << std::endl;

    Depth depth = _depth.getMat();
    Colors color = _rgb.getMat();
    Mat volume = _volume.getMat();
    Mat pixNorms = _pixNorms.getMat();

    RGBTsdfVoxel* volDataStart = volume.ptr<RGBTsdfVoxel>();

    Vec4i volStrides;
    settings.getVolumeDimentions(volStrides);

    Vec3i resolution;
    settings.getVolumeResolution(resolution);
    const Point3i volResolution = Point3i(resolution);
    float voxelSize = settings.getVoxelSize();

    const Affine3f pose = Affine3f(volumePose);
    const Affine3f vol2cam(Affine3f(cameraPose.inv()) * pose);

    Matx33f intr;
    settings.getCameraIntrinsics(intr);
    const Intr::Projector projDepth = Intr(intr).makeProjector();

    Matx33f rgb_intr;
    settings.getRGBCameraIntrinsics(rgb_intr);
    const Intr::Projector projColor = Intr(rgb_intr);

    const float dfac(1.f / settings.getDepthFactor());
    const float truncDist = settings.getTruncatedDistance();
    const float truncDistInv = 1.f / truncDist;
    const int maxWeight = settings.getMaxWeight();

    Range integrateRange(0, volResolution.x);

#if USE_INTRINSICS_
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
                    Point2f projectedRGB = projColor(camSpacePt, camPixVec);

                    depthType v = bilinearDepth(depth, projected);
                    if (v == 0) {
                        continue;
                    }
                    int _u = projected.x;
                    int _v = projected.y;

                    int rgb_u = (int)projectedRGB.x;
                    int rgb_v = (int)projectedRGB.y;

                    if (!(_u >= 0 && _u < depth.cols && _v >= 0 && _v < depth.rows
                        && rgb_v >= 0 && rgb_v < color.rows && rgb_u >= 0 && rgb_u < color.cols
                        ))
                        continue;

                    float pixNorm = pixNorms.at<float>(_v, _u);
                    // TODO: Add support of 3point and 4 point representation
                    Vec3f colorRGB = color.at<Vec3f>(rgb_v, rgb_u);
                    
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
                        weight = min(int(weight + 1), int(maxWeight));
                    }
                }
            }
        }
    };
#endif
    //parallel_for_(integrateRange, IntegrateInvoker);
    IntegrateInvoker(integrateRange);

    std::cout << "integrateColorTsdfVolumeUnit() end" << std::endl;

}




inline float interpolateVoxel(const Mat& volume,
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

inline Point3f getNormalVoxel(const Mat& volume,
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

inline Point3f getColorVoxel(const Mat& volume,
    const Vec4i& volDims, const Vec8i& neighbourCoords, const Point3i volResolution,
    const Point3f& p)
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
    res = Point3f(volData[coordBase].r, volData[coordBase].g, volData[coordBase].b);
    //std::cout << res << " ";
    colorFix(res);
    return res;
}


void raycastColorTsdfVolumeUnit(const VolumeSettings& settings, const Matx44f& cameraPose, int height, int width,
    InputArray _volume, OutputArray _points, OutputArray _normals, OutputArray _colors)
{
    std::cout << "raycastColorTsdfVolumeUnit()" << std::endl;

    Size frameSize(width, height);
    CV_Assert(frameSize.area() > 0);

    _points.create(frameSize, POINT_TYPE);
    _normals.create(frameSize, POINT_TYPE);
    _colors.create(frameSize, COLOR_TYPE);

    Points points = _points.getMat();
    Normals normals = _normals.getMat();
    Colors colors = _colors.getMat();

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
    const Intr::Reprojector reprojDepth = Intr(intr).makeReprojector();

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
                    float f = interpolateVoxel(volume, volDims, neighbourCoords, next);
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
                            fnext = interpolateVoxel(volume, volDims, neighbourCoords, next);
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
                        float ft = interpolateVoxel(volume, volDims, neighbourCoords, tp);
                        float ftdt = interpolateVoxel(volume, volDims, neighbourCoords, next);
                        // float t = tmin + steps*tstep;
                        // float ts = t - tstep*ft/(ftdt - ft);
                        float ts = tmin + tstep * (steps - ft / (ftdt - ft));

                        // avoid division by zero
                        if (!cvIsNaN(ts) && !cvIsInf(ts))
                        {
                            Point3f pv = (orig + dir * ts);
                            Point3f nv = getNormalVoxel(volume, volDims, neighbourCoords, volResolution, pv);
                            Point3f cv = getColorVoxel(volume, volDims, neighbourCoords, volResolution, pv);
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

    //parallel_for_(raycastRange, RaycastInvoker);
    RaycastInvoker(raycastRange);








    std::cout << "raycastColorTsdfVolumeUnit() end" << std::endl;
}






} // namespace cv
