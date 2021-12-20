// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// Partially rewritten from https://github.com/Nerei/kinfu_remake
// Copyright(c) 2012, Anatoly Baksheev. All rights reserved.

#include "../precomp.hpp"
#include "hash_tsdf_functions.hpp"
#include "opencl_kernels_3d.hpp"

namespace cv {


void integrateHashTsdfVolumeUnit(
    const VolumeSettings& settings, const Matx44f& cameraPose, int& lastVolIndex, const int frameId,
    InputArray _depth, InputArray _pixNorms, InputArray _volUnitsData, VolumeUnitIndexes& volumeUnits)
{
    std::cout << "integrateHashTsdfVolumeUnit()" << std::endl;

    CV_TRACE_FUNCTION();

    CV_Assert(_depth.type() == DEPTH_TYPE);
    Depth depth = _depth.getMat();
    Mat volUnitsData = _volUnitsData.getMat();
    Mat pixNorms = _pixNorms.getMat();

    //! Compute volumes to be allocated
    const int depthStride = settings.getVolumeUnitDegree();
    const float invDepthFactor = 1.f / settings.getDepthFactor();
    const float truncDist = settings.getTruncatedDistance();
    const float truncateThreshold = settings.getTruncateThreshold();
    const float voxelSize = settings.getVoxelSize();

    Vec3i resolution;
    settings.getVolumeResolution(resolution);
    const float volumeUnitSize = voxelSize * resolution[0];

    Matx33f intr;
    settings.getCameraIntrinsics(intr);
    const Intr intrinsics(intr);
    const Intr::Reprojector reproj(intrinsics.makeReprojector());

    Matx44f _pose;
    settings.getVolumePose(_pose);
    const Affine3f pose = Affine3f(_pose);
    const Affine3f cam2vol(pose.inv() * Affine3f(cameraPose));

    const Point3f truncPt(truncDist, truncDist, truncDist);
    VolumeUnitIndexSet newIndices;
    Mutex mutex;
    Range allocateRange(0, depth.rows);

    auto AllocateVolumeUnitsInvoker = [&](const Range& range) {
        VolumeUnitIndexSet localAccessVolUnits;
        for (int y = range.start; y < range.end; y += depthStride)
        {
            const depthType* depthRow = depth[y];
            for (int x = 0; x < depth.cols; x += depthStride)
            {
                depthType z = depthRow[x] * invDepthFactor;

                if (z <= 0 || z > truncateThreshold)
                    continue;

                Point3f camPoint = reproj(Point3f((float)x, (float)y, z));
                Point3f volPoint = cam2vol * camPoint;
                //! Find accessed TSDF volume unit for valid 3D vertex
                Vec3i lower_bound = volumeToVolumeUnitIdx(volPoint - truncPt, volumeUnitSize);
                Vec3i upper_bound = volumeToVolumeUnitIdx(volPoint + truncPt, volumeUnitSize);

                for (int i = lower_bound[0]; i <= upper_bound[0]; i++)
                    for (int j = lower_bound[1]; j <= upper_bound[1]; j++)
                        for (int k = lower_bound[2]; k <= upper_bound[2]; k++)
                        {
                            const Vec3i tsdf_idx = Vec3i(i, j, k);
                            if (localAccessVolUnits.count(tsdf_idx) <= 0 && volumeUnits.count(tsdf_idx) <= 0)
                            {
                                //! This volume unit will definitely be required for current integration
                                localAccessVolUnits.emplace(tsdf_idx);
                            }
                        }
            }
        }

        mutex.lock();
        for (const auto& tsdf_idx : localAccessVolUnits)
        {
            //! If the insert into the global set passes
            if (!newIndices.count(tsdf_idx))
            {
                // Volume allocation can be performed outside of the lock
                newIndices.emplace(tsdf_idx);
            }
        }
        mutex.unlock();
    };
    parallel_for_(allocateRange, AllocateVolumeUnitsInvoker);
    //AllocateVolumeUnitsInvoker(allocateRange);

    //! Perform the allocation
    for (auto idx : newIndices)
    {
        VolumeUnit& vu = volumeUnits.emplace(idx, VolumeUnit()).first->second;

        Matx44f subvolumePose = pose.translate(volumeUnitIdxToVolume(idx, volumeUnitSize)).matrix;

        vu.pose = subvolumePose;
        vu.index = lastVolIndex; lastVolIndex++;
        if (lastVolIndex > int(volUnitsData.size().height))
        {
            volUnitsData.resize((lastVolIndex - 1) * 2);
        }
        volUnitsData.row(vu.index).forEach<VecTsdfVoxel>([](VecTsdfVoxel& vv, const int* /* position */)
            {
                TsdfVoxel& v = reinterpret_cast<TsdfVoxel&>(vv);
                v.tsdf = floatToTsdf(0.0f); v.weight = 0;
            });
        //! This volume unit will definitely be required for current integration
        vu.lastVisibleIndex = frameId;
        vu.isActive = true;
    }

    //! Get keys for all the allocated volume Units
    std::vector<Vec3i> totalVolUnits;
    for (const auto& keyvalue : volumeUnits)
    {
        totalVolUnits.push_back(keyvalue.first);
    }


    //! Mark volumes in the camera frustum as active
    Range inFrustumRange(0, (int)volumeUnits.size());
    parallel_for_(inFrustumRange, [&](const Range& range) {
        const Affine3f vol2cam(Affine3f(cameraPose.inv()) * pose);
        const Intr::Projector proj(intrinsics.makeProjector());

        for (int i = range.start; i < range.end; ++i)
        {
            Vec3i tsdf_idx = totalVolUnits[i];
            VolumeUnitIndexes::iterator it = volumeUnits.find(tsdf_idx);
            if (it == volumeUnits.end())
                continue;

            Point3f volumeUnitPos = volumeUnitIdxToVolume(it->first, volumeUnitSize);
            Point3f volUnitInCamSpace = vol2cam * volumeUnitPos;
            if (volUnitInCamSpace.z < 0 || volUnitInCamSpace.z > truncateThreshold)
            {
                it->second.isActive = false;
                continue;
            }
            Point2f cameraPoint = proj(volUnitInCamSpace);
            if (cameraPoint.x >= 0 && cameraPoint.y >= 0 && cameraPoint.x < depth.cols && cameraPoint.y < depth.rows)
            {
                assert(it != volumeUnits.end());
                it->second.lastVisibleIndex = frameId;
                it->second.isActive = true;
            }
        }
        });


    //! Integrate the correct volumeUnits
    parallel_for_(Range(0, (int)totalVolUnits.size()), [&](const Range& range) {
        for (int i = range.start; i < range.end; i++)
        {
            Vec3i tsdf_idx = totalVolUnits[i];
            VolumeUnitIndexes::iterator it = volumeUnits.find(tsdf_idx);
            if (it == volumeUnits.end())
                return;

            VolumeUnit& volumeUnit = it->second;
            if (volumeUnit.isActive)
            {
                //! The volume unit should already be added into the Volume from the allocator
                integrateTsdfVolumeUnit(settings, volumeUnit.pose, cameraPose, depth, pixNorms, volUnitsData.row(volumeUnit.index));

                //! Ensure all active volumeUnits are set to inactive for next integration
                volumeUnit.isActive = false;
            }
        }
        });

    std::cout << "integrateHashTsdfVolumeUnit() end" << std::endl;
}


inline TsdfVoxel _at(Mat& volUnitsData, const cv::Vec3i& volumeIdx, int indx,
    const int volumeUnitResolution, const Vec4i volStrides)
{
    //! Out of bounds
    if ((volumeIdx[0] >= volumeUnitResolution || volumeIdx[0] < 0) ||
        (volumeIdx[1] >= volumeUnitResolution || volumeIdx[1] < 0) ||
        (volumeIdx[2] >= volumeUnitResolution || volumeIdx[2] < 0))
    {
        return TsdfVoxel(floatToTsdf(1.f), 0);
    }

    const TsdfVoxel* volData = volUnitsData.ptr<TsdfVoxel>(indx);
    int coordBase =
        volumeIdx[0] * volStrides[0] + volumeIdx[1] * volStrides[1] + volumeIdx[2] * volStrides[2];
    return volData[coordBase];
}

#if USE_INTRINSICS
inline float interpolate(float tx, float ty, float tz, float vx[8])
{
    v_float32x4 v0246, v1357;
    v_load_deinterleave(vx, v0246, v1357);

    v_float32x4 vxx = v0246 + v_setall_f32(tz) * (v1357 - v0246);

    v_float32x4 v00_10 = vxx;
    v_float32x4 v01_11 = v_reinterpret_as_f32(v_rotate_right<1>(v_reinterpret_as_u32(vxx)));

    v_float32x4 v0_1 = v00_10 + v_setall_f32(ty) * (v01_11 - v00_10);
    float v0 = v0_1.get0();
    v0_1 = v_reinterpret_as_f32(v_rotate_right<2>(v_reinterpret_as_u32(v0_1)));
    float v1 = v0_1.get0();

    return v0 + tx * (v1 - v0);
}

#else
inline float interpolate(float tx, float ty, float tz, float vx[8])
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

TsdfVoxel atVolumeUnit(
    const Mat& volUnitsData, const VolumeUnitIndexes& volumeUnits,
    const Vec3i& point, const Vec3i& volumeUnitIdx, VolumeUnitIndexes::const_iterator it,
    const int volumeUnitDegree, const Vec4i volStrides)
{
    if (it == volumeUnits.end())
    {
        return TsdfVoxel(floatToTsdf(1.f), 0);
    }
    Vec3i volUnitLocalIdx = point - Vec3i(volumeUnitIdx[0] << volumeUnitDegree,
        volumeUnitIdx[1] << volumeUnitDegree,
        volumeUnitIdx[2] << volumeUnitDegree);

    // expanding at(), removing bounds check
    const TsdfVoxel* volData = volUnitsData.ptr<TsdfVoxel>(it->second.index);
    int coordBase = volUnitLocalIdx[0] * volStrides[0] + volUnitLocalIdx[1] * volStrides[1] + volUnitLocalIdx[2] * volStrides[2];
    return volData[coordBase];
}

Point3f getNormalVoxel(
    const Point3f& point, const float voxelSizeInv,
    const int volumeUnitDegree,  const Vec4i volStrides,
    const Mat& volUnitsData, const VolumeUnitIndexes& volumeUnits)
{
    Vec3f normal = Vec3f(0, 0, 0);

    Point3f ptVox = point * voxelSizeInv;
    Vec3i iptVox(cvFloor(ptVox.x), cvFloor(ptVox.y), cvFloor(ptVox.z));

    // A small hash table to reduce a number of find() calls
    bool queried[8];
    VolumeUnitIndexes::const_iterator iterMap[8];
    for (int i = 0; i < 8; i++)
    {
        iterMap[i] = volumeUnits.end();
        queried[i] = false;
    }

#if !USE_INTERPOLATION_IN_GETNORMAL
    const Vec3i offsets[] = { { 1,  0,  0}, {-1,  0,  0}, { 0,  1,  0}, // 0-3
                              { 0, -1,  0}, { 0,  0,  1}, { 0,  0, -1}  // 4-7
    };
    const int nVals = 6;

#else
    const Vec3i offsets[] = { { 0,  0,  0}, { 0,  0,  1}, { 0,  1,  0}, { 0,  1,  1}, //  0-3
                              { 1,  0,  0}, { 1,  0,  1}, { 1,  1,  0}, { 1,  1,  1}, //  4-7
                              {-1,  0,  0}, {-1,  0,  1}, {-1,  1,  0}, {-1,  1,  1}, //  8-11
                              { 2,  0,  0}, { 2,  0,  1}, { 2,  1,  0}, { 2,  1,  1}, // 12-15
                              { 0, -1,  0}, { 0, -1,  1}, { 1, -1,  0}, { 1, -1,  1}, // 16-19
                              { 0,  2,  0}, { 0,  2,  1}, { 1,  2,  0}, { 1,  2,  1}, // 20-23
                              { 0,  0, -1}, { 0,  1, -1}, { 1,  0, -1}, { 1,  1, -1}, // 24-27
                              { 0,  0,  2}, { 0,  1,  2}, { 1,  0,  2}, { 1,  1,  2}, // 28-31
    };
    const int nVals = 32;
#endif

    float vals[nVals];
    for (int i = 0; i < nVals; i++)
    {
        Vec3i pt = iptVox + offsets[i];

        Vec3i volumeUnitIdx = Vec3i(pt[0] >> volumeUnitDegree, pt[1] >> volumeUnitDegree, pt[2] >> volumeUnitDegree);

        int dictIdx = (volumeUnitIdx[0] & 1) + (volumeUnitIdx[1] & 1) * 2 + (volumeUnitIdx[2] & 1) * 4;
        auto it = iterMap[dictIdx];
        if (!queried[dictIdx])
        {
            it = volumeUnits.find(volumeUnitIdx);
            iterMap[dictIdx] = it;
            queried[dictIdx] = true;
        }

        vals[i] = tsdfToFloat(atVolumeUnit(volUnitsData, volumeUnits, pt, volumeUnitIdx, it, volumeUnitDegree, volStrides).tsdf);
    }

#if !USE_INTERPOLATION_IN_GETNORMAL
    for (int c = 0; c < 3; c++)
    {
        normal[c] = vals[c * 2] - vals[c * 2 + 1];
    }
#else

    float cxv[8], cyv[8], czv[8];

    // How these numbers were obtained:
    // 1. Take the basic interpolation sequence:
    // 000, 001, 010, 011, 100, 101, 110, 111
    // where each digit corresponds to shift by x, y, z axis respectively.
    // 2. Add +1 for next or -1 for prev to each coordinate to corresponding axis
    // 3. Search corresponding values in offsets
    const int idxxp[8] = { 8,  9, 10, 11,  0,  1,  2,  3 };
    const int idxxn[8] = { 4,  5,  6,  7, 12, 13, 14, 15 };
    const int idxyp[8] = { 16, 17,  0,  1, 18, 19,  4,  5 };
    const int idxyn[8] = { 2,  3, 20, 21,  6,  7, 22, 23 };
    const int idxzp[8] = { 24,  0, 25,  2, 26,  4, 27,  6 };
    const int idxzn[8] = { 1, 28,  3, 29,  5, 30,  7, 31 };

#if !USE_INTRINSICS
    for (int i = 0; i < 8; i++)
    {
        cxv[i] = vals[idxxn[i]] - vals[idxxp[i]];
        cyv[i] = vals[idxyn[i]] - vals[idxyp[i]];
        czv[i] = vals[idxzn[i]] - vals[idxzp[i]];
    }
#else

# if CV_SIMD >= 32
    v_float32x8 cxp = v_lut(vals, idxxp);
    v_float32x8 cxn = v_lut(vals, idxxn);

    v_float32x8 cyp = v_lut(vals, idxyp);
    v_float32x8 cyn = v_lut(vals, idxyn);

    v_float32x8 czp = v_lut(vals, idxzp);
    v_float32x8 czn = v_lut(vals, idxzn);

    v_float32x8 vcxv = cxn - cxp;
    v_float32x8 vcyv = cyn - cyp;
    v_float32x8 vczv = czn - czp;

    v_store(cxv, vcxv);
    v_store(cyv, vcyv);
    v_store(czv, vczv);
# else
    v_float32x4 cxp0 = v_lut(vals, idxxp + 0); v_float32x4 cxp1 = v_lut(vals, idxxp + 4);
    v_float32x4 cxn0 = v_lut(vals, idxxn + 0); v_float32x4 cxn1 = v_lut(vals, idxxn + 4);

    v_float32x4 cyp0 = v_lut(vals, idxyp + 0); v_float32x4 cyp1 = v_lut(vals, idxyp + 4);
    v_float32x4 cyn0 = v_lut(vals, idxyn + 0); v_float32x4 cyn1 = v_lut(vals, idxyn + 4);

    v_float32x4 czp0 = v_lut(vals, idxzp + 0); v_float32x4 czp1 = v_lut(vals, idxzp + 4);
    v_float32x4 czn0 = v_lut(vals, idxzn + 0); v_float32x4 czn1 = v_lut(vals, idxzn + 4);

    v_float32x4 cxv0 = cxn0 - cxp0; v_float32x4 cxv1 = cxn1 - cxp1;
    v_float32x4 cyv0 = cyn0 - cyp0; v_float32x4 cyv1 = cyn1 - cyp1;
    v_float32x4 czv0 = czn0 - czp0; v_float32x4 czv1 = czn1 - czp1;

    v_store(cxv + 0, cxv0); v_store(cxv + 4, cxv1);
    v_store(cyv + 0, cyv0); v_store(cyv + 4, cyv1);
    v_store(czv + 0, czv0); v_store(czv + 4, czv1);
#endif

#endif

    float tx = ptVox.x - iptVox[0];
    float ty = ptVox.y - iptVox[1];
    float tz = ptVox.z - iptVox[2];

    normal[0] = interpolate(tx, ty, tz, cxv);
    normal[1] = interpolate(tx, ty, tz, cyv);
    normal[2] = interpolate(tx, ty, tz, czv);
#endif

    float nv = sqrt(normal[0] * normal[0] +
        normal[1] * normal[1] +
        normal[2] * normal[2]);
    return nv < 0.0001f ? nan3 : normal / nv;
}


void raycastHashTsdfVolumeUnit(
    const VolumeSettings& settings, const Matx44f& cameraPose, int height, int width,
    InputArray _volUnitsData, const VolumeUnitIndexes& volumeUnits, OutputArray _points, OutputArray _normals)
{
    std::cout << "raycastHashTsdfVolumeUnit()" << std::endl;

    CV_TRACE_FUNCTION();
    Size frameSize(width, height);
    CV_Assert(frameSize.area() > 0);

    Mat volUnitsData = _volUnitsData.getMat();

    _points.create(frameSize, POINT_TYPE);
    _normals.create(frameSize, POINT_TYPE);

    Points points1 = _points.getMat();
    Normals normals1 = _normals.getMat();

    Points& points(points1);
    Normals& normals(normals1);

    const float truncDist = settings.getTruncatedDistance();
    const float raycastStepFactor = settings.getRaycastStepFactor();
    const float tstep = truncDist * raycastStepFactor;
    const float truncateThreshold = settings.getTruncateThreshold();
    const float voxelSize = settings.getVoxelSize();
    const float voxelSizeInv = 1.f / voxelSize;
    const int volumeUnitDegree = settings.getVolumeUnitDegree();


    const Vec4i volDims;
    settings.getVolumeDimentions(volDims);
    Vec3i resolution;
    settings.getVolumeResolution(resolution);
    const Point3i volResolution = Point3i(resolution);
    const float volumeUnitSize = voxelSize * resolution[0];

    Matx44f _pose;
    settings.getVolumePose(_pose);
    const Affine3f pose = Affine3f(_pose);
    const Affine3f cam2vol(pose.inv() * Affine3f(cameraPose));
    const Affine3f vol2cam(Affine3f(cameraPose.inv()) * pose);

    Matx33f intr;
    settings.getCameraIntrinsics(intr);
    const Intr intrinsics(intr);
    const Intr::Reprojector reproj(intrinsics.makeReprojector());

    const int nstripes = -1;

    auto _HashRaycastInvoker = [&](const Range& range)
    {
        const Point3f cam2volTrans = cam2vol.translation();
        const Matx33f cam2volRot = cam2vol.rotation();
        const Matx33f vol2camRot = vol2cam.rotation();

        const float blockSize = volumeUnitSize;

        for (int y = range.start; y < range.end; y++)
        {
            ptype* ptsRow = points[y];
            ptype* nrmRow = normals[y];

            for (int x = 0; x < points.cols; x++)
            {
                //! Initialize default value
                Point3f point = nan3, normal = nan3;

                //! Ray origin and direction in the volume coordinate frame
                Point3f orig = cam2volTrans;
                Point3f rayDirV = normalize(Vec3f(cam2volRot * reproj(Point3f(float(x), float(y), 1.f))));

                float tmin = 0;
                float tmax = truncateThreshold;
                float tcurr = tmin;

                cv::Vec3i prevVolumeUnitIdx =
                    cv::Vec3i(std::numeric_limits<int>::min(), std::numeric_limits<int>::min(),
                        std::numeric_limits<int>::min());

                float tprev = tcurr;
                float prevTsdf = truncDist;
                while (tcurr < tmax)
                {

                    Point3f currRayPos = orig + tcurr * rayDirV;
                    cv::Vec3i currVolumeUnitIdx = volumeToVolumeUnitIdx(currRayPos, volumeUnitSize);

                    VolumeUnitIndexes::const_iterator it = volumeUnits.find(currVolumeUnitIdx);

                    float currTsdf = prevTsdf;
                    int currWeight = 0;
                    float stepSize = 0.5f * blockSize;
                    cv::Vec3i volUnitLocalIdx;


                    //! The subvolume exists in hashtable
                    if (it != volumeUnits.end())
                    {
                        cv::Point3f currVolUnitPos = volumeUnitIdxToVolume(currVolumeUnitIdx, volumeUnitSize);
                        volUnitLocalIdx = volumeToVoxelCoord(currRayPos - currVolUnitPos, voxelSizeInv);

                        //! TODO: Figure out voxel interpolation
                        TsdfVoxel currVoxel = _at(volUnitsData, volUnitLocalIdx, it->second.index, volResolution.x, volDims);
                        currTsdf = tsdfToFloat(currVoxel.tsdf);
                        currWeight = currVoxel.weight;
                        stepSize = tstep;
                    }

                    //std::cout << prevTsdf << " " << currTsdf << " " << currWeight << std::endl;
                    //! Surface crossing
                    if (prevTsdf > 0.f && currTsdf <= 0.f && currWeight > 0)
                    {
                        float tInterp = (tcurr * prevTsdf - tprev * currTsdf) / (prevTsdf - currTsdf);
                        if (!cvIsNaN(tInterp) && !cvIsInf(tInterp))
                        {
                            Point3f pv = orig + tInterp * rayDirV;
                            Point3f nv = getNormalVoxel(pv, voxelSizeInv, volumeUnitDegree, volDims, volUnitsData, volumeUnits);

                            if (!isNaN(nv))
                            {
                                normal = vol2camRot * nv;
                                point = vol2cam * pv;
                            }
                        }
                        break;
                    }
                    prevVolumeUnitIdx = currVolumeUnitIdx;
                    prevTsdf = currTsdf;
                    tprev = tcurr;
                    tcurr += stepSize;
                }
                ptsRow[x] = toPtype(point);
                nrmRow[x] = toPtype(normal);
            }
        }
    };



    parallel_for_(Range(0, points.rows), _HashRaycastInvoker, nstripes);





    std::cout << "raycastHashTsdfVolumeUnit() end" << std::endl;
}

} // namespace cv
