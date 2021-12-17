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

    Vec3i resolution;
    settings.getVolumeResolution(resolution);
    const int volumeUnitSize = resolution[0];

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
                integrateTsdfVolumeUnit(settings, cameraPose, depth, pixNorms, volUnitsData.row(volumeUnit.index));

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
    const float tstep(truncDist * raycastStepFactor);
    const float truncateThreshold = settings.getTruncateThreshold();
    const float voxelSize = settings.getVoxelSize();
    const float voxelSizeInv = 1.f / voxelSize;

    const Vec4i volDims;
    settings.getVolumeDimentions(volDims);
    Vec3i resolution;
    settings.getVolumeResolution(resolution);
    const Point3i volResolution = Point3i(resolution);
    const int volumeUnitSize = resolution[0];

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
                    //! Surface crossing
                    if (prevTsdf > 0.f && currTsdf <= 0.f && currWeight > 0)
                    {
                        float tInterp = (tcurr * prevTsdf - tprev * currTsdf) / (prevTsdf - currTsdf);
                        if (!cvIsNaN(tInterp) && !cvIsInf(tInterp))
                        {
                            Point3f pv = orig + tInterp * rayDirV;
                            //Point3f nv = volume.getNormalVoxel(pv);
                            Point3f nv = nan3;

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








    std::cout << "raycastHashTsdfVolumeUnit() end" << std::endl;
}

} // namespace cv
