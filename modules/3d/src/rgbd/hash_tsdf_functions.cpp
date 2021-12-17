// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// Partially rewritten from https://github.com/Nerei/kinfu_remake
// Copyright(c) 2012, Anatoly Baksheev. All rights reserved.

#include "../precomp.hpp"
#include "hash_tsdf_functions.hpp"
#include "opencl_kernels_3d.hpp"

namespace cv {


void integrateHashTsdfVolumeUnit(const VolumeSettings& settings, const Matx44f& cameraPose,
    InputArray _depth, InputArray _pixNorms, InputArray _volUnitsData, VolumeUnitIndexes& volumeUnits)
{
    std::cout << "integrateHashTsdfVolumeUnit()" << std::endl;

    CV_TRACE_FUNCTION();

    CV_Assert(_depth.type() == DEPTH_TYPE);
    Depth depth = _depth.getMat();

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


    std::cout << "integrateHashTsdfVolumeUnit() end" << std::endl;

}


} // namespace cv
