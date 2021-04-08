// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#include "precomp.hpp"
#include "hash_tsdf.hpp"

#include <atomic>
#include <functional>
#include <iostream>
#include <limits>
#include <vector>

#include "kinfu_frame.hpp"
#include "opencv2/core/cvstd.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/core/utils/trace.hpp"
#include "utils.hpp"
#include "opencl_kernels_rgbd.hpp"

#define USE_INTERPOLATION_IN_GETNORMAL 1
#define VOLUMES_SIZE 8192

namespace cv
{
namespace kinfu
{

HashTSDFVolume::HashTSDFVolume(float _voxelSize, cv::Matx44f _pose, float _raycastStepFactor,
    float _truncDist, int _maxWeight, float _truncateThreshold,
    int _volumeUnitRes, bool _zFirstMemOrder)
    : Volume(_voxelSize, _pose, _raycastStepFactor),
    maxWeight(_maxWeight),
    truncateThreshold(_truncateThreshold),
    volumeUnitResolution(_volumeUnitRes),
    volumeUnitSize(voxelSize* volumeUnitResolution),
    zFirstMemOrder(_zFirstMemOrder)
{
    truncDist = std::max(_truncDist, 4.0f * voxelSize);

    if (!(volumeUnitResolution & (volumeUnitResolution - 1)))
    {
        // vuRes is a power of 2, let's get this power
        volumeUnitDegree = trailingZeros32(volumeUnitResolution);
    }
    else
    {
        CV_Error(Error::StsBadArg, "Volume unit resolution should be a power of 2");
    }

    int xdim, ydim, zdim;
    if (zFirstMemOrder)
    {
        xdim = volumeUnitResolution * volumeUnitResolution;
        ydim = volumeUnitResolution;
        zdim = 1;
    }
    else
    {
        xdim = 1;
        ydim = volumeUnitResolution;
        zdim = volumeUnitResolution * volumeUnitResolution;
    }
    volStrides = Vec4i(xdim, ydim, zdim);
}

//! Spatial hashing
struct tsdf_hash
{
    size_t operator()(const Vec3i& x) const noexcept
    {
        size_t seed = 0;
        constexpr uint32_t GOLDEN_RATIO = 0x9e3779b9;
        for (uint16_t i = 0; i < 3; i++)
        {
            seed ^= std::hash<int>()(x[i]) + GOLDEN_RATIO + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

struct VolumeUnit
{
    cv::Vec3i coord;
    int index;
    cv::Matx44f pose;
    int lastVisibleIndex = 0;
    bool isActive;
};

typedef std::unordered_set<cv::Vec3i, tsdf_hash> VolumeUnitIndexSet;
typedef std::unordered_map<cv::Vec3i, VolumeUnit, tsdf_hash> VolumeUnitIndexes;

class HashTSDFVolumeCPU : public HashTSDFVolume
{
public:
    // dimension in voxels, size in meters
    HashTSDFVolumeCPU(float _voxelSize, const Matx44f& _pose, float _raycastStepFactor, float _truncDist, int _maxWeight,
        float _truncateThreshold, int _volumeUnitRes, bool zFirstMemOrder = true);

    HashTSDFVolumeCPU(const VolumeParams& _volumeParams, bool zFirstMemOrder = true);

    virtual void integrate(InputArray, InputArray, float, const Matx44f&, const kinfu::Intr&, const Intr&, const int) override
        { CV_Error(Error::StsNotImplemented, "Not implemented"); };
    void integrate(InputArray _depth, float depthFactor, const Matx44f& cameraPose, const kinfu::Intr& intrinsics,
        const int frameId = 0) override;
    void raycast(const Matx44f& cameraPose, const kinfu::Intr& intrinsics, const Size& frameSize, OutputArray points,
        OutputArray normals) const override;
    void raycast(const Matx44f&, const kinfu::Intr&, const Size&, OutputArray, OutputArray, OutputArray) const override
        { CV_Error(Error::StsNotImplemented, "Not implemented"); };
    void fetchNormals(InputArray points, OutputArray _normals) const override;
    void fetchPointsNormals(OutputArray points, OutputArray normals) const override;

    void reset() override;
    size_t getTotalVolumeUnits() const override { return volumeUnits.size(); }
    int getVisibleBlocks(int currFrameId, int frameThreshold) const override;

    //! Return the voxel given the voxel index in the universal volume (1 unit = 1 voxel_length)
    TsdfVoxel at(const Vec3i& volumeIdx) const;

    //! Return the voxel given the point in volume coordinate system i.e., (metric scale 1 unit =
    //! 1m)
    virtual TsdfVoxel at(const cv::Point3f& point) const;
    virtual TsdfVoxel _at(const cv::Vec3i& volumeIdx, int indx) const;

    TsdfVoxel atVolumeUnit(const Vec3i& point, const Vec3i& volumeUnitIdx, VolumeUnitIndexes::const_iterator it) const;


    float interpolateVoxelPoint(const Point3f& point) const;
    float interpolateVoxel(const cv::Point3f& point) const;
    Point3f getNormalVoxel(const cv::Point3f& p) const;

    //! Utility functions for coordinate transformations
    Vec3i volumeToVolumeUnitIdx(const Point3f& point) const;
    Point3f volumeUnitIdxToVolume(const Vec3i& volumeUnitIdx) const;

    Point3f voxelCoordToVolume(const Vec3i& voxelIdx) const;
    Vec3i volumeToVoxelCoord(const Point3f& point) const;

public:
    Vec6f frameParams;
    Mat pixNorms;
    VolumeUnitIndexes volumeUnits;
    cv::Mat volUnitsData;
    int lastVolIndex;
};


HashTSDFVolumeCPU::HashTSDFVolumeCPU(float _voxelSize, const Matx44f& _pose, float _raycastStepFactor, float _truncDist,
                                     int _maxWeight, float _truncateThreshold, int _volumeUnitRes, bool _zFirstMemOrder)
    :HashTSDFVolume(_voxelSize, _pose, _raycastStepFactor, _truncDist, _maxWeight, _truncateThreshold, _volumeUnitRes,
           _zFirstMemOrder)
{
    reset();
}

HashTSDFVolumeCPU::HashTSDFVolumeCPU(const VolumeParams& _params, bool _zFirstMemOrder)
    : HashTSDFVolumeCPU(_params.voxelSize, _params.pose.matrix, _params.raycastStepFactor, _params.tsdfTruncDist, _params.maxWeight,
           _params.depthTruncThreshold, _params.unitResolution, _zFirstMemOrder)
{
}

// zero volume, leave rest params the same
void HashTSDFVolumeCPU::reset()
{
    CV_TRACE_FUNCTION();
    lastVolIndex = 0;
    volUnitsData = cv::Mat(VOLUMES_SIZE, volumeUnitResolution * volumeUnitResolution * volumeUnitResolution, rawType<TsdfVoxel>());
    frameParams = Vec6f();
    pixNorms = Mat();
    volumeUnits = VolumeUnitIndexes();
}

void HashTSDFVolumeCPU::integrate(InputArray _depth, float depthFactor, const Matx44f& cameraPose, const Intr& intrinsics, const int frameId)
{
    CV_TRACE_FUNCTION();

    CV_Assert(_depth.type() == DEPTH_TYPE);
    Depth depth = _depth.getMat();

    //! Compute volumes to be allocated
    const int depthStride = volumeUnitDegree;
    const float invDepthFactor = 1.f / depthFactor;
    const Intr::Reprojector reproj(intrinsics.makeReprojector());
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
                if (z <= 0 || z > this->truncateThreshold)
                    continue;
                Point3f camPoint = reproj(Point3f((float)x, (float)y, z));
                Point3f volPoint = cam2vol * camPoint;
                //! Find accessed TSDF volume unit for valid 3D vertex
                Vec3i lower_bound = this->volumeToVolumeUnitIdx(volPoint - truncPt);
                Vec3i upper_bound = this->volumeToVolumeUnitIdx(volPoint + truncPt);

                for (int i = lower_bound[0]; i <= upper_bound[0]; i++)
                    for (int j = lower_bound[1]; j <= upper_bound[1]; j++)
                        for (int k = lower_bound[2]; k <= upper_bound[2]; k++)
                        {
                            const Vec3i tsdf_idx = Vec3i(i, j, k);
                            if (localAccessVolUnits.count(tsdf_idx) <= 0 && this->volumeUnits.count(tsdf_idx) <= 0)
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
        VolumeUnit& vu = this->volumeUnits.emplace(idx, VolumeUnit()).first->second;

        Matx44f subvolumePose = pose.translate(volumeUnitIdxToVolume(idx)).matrix;

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

            Point3f volumeUnitPos = volumeUnitIdxToVolume(it->first);
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
                it->second.isActive         = true;
            }
        }
        });

    Vec6f newParams((float)depth.rows, (float)depth.cols,
        intrinsics.fx, intrinsics.fy,
        intrinsics.cx, intrinsics.cy);
    if ( !(frameParams==newParams) )
    {
        frameParams = newParams;
        pixNorms = preCalculationPixNorm(depth, intrinsics);
    }

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
                integrateVolumeUnit(truncDist, voxelSize, maxWeight, volumeUnit.pose,
                    Point3i(volumeUnitResolution, volumeUnitResolution, volumeUnitResolution), volStrides, depth,
                    depthFactor, cameraPose, intrinsics, pixNorms, volUnitsData.row(volumeUnit.index));

                //! Ensure all active volumeUnits are set to inactive for next integration
                volumeUnit.isActive = false;
            }
        }
        });
}

cv::Vec3i HashTSDFVolumeCPU::volumeToVolumeUnitIdx(const cv::Point3f& p) const
{
    return cv::Vec3i(cvFloor(p.x / volumeUnitSize), cvFloor(p.y / volumeUnitSize),
                     cvFloor(p.z / volumeUnitSize));
}

cv::Point3f HashTSDFVolumeCPU::volumeUnitIdxToVolume(const cv::Vec3i& volumeUnitIdx) const
{
    return cv::Point3f(volumeUnitIdx[0] * volumeUnitSize, volumeUnitIdx[1] * volumeUnitSize,
                       volumeUnitIdx[2] * volumeUnitSize);
}

cv::Point3f HashTSDFVolumeCPU::voxelCoordToVolume(const cv::Vec3i& voxelIdx) const
{
    return cv::Point3f(voxelIdx[0] * voxelSize, voxelIdx[1] * voxelSize, voxelIdx[2] * voxelSize);
}

cv::Vec3i HashTSDFVolumeCPU::volumeToVoxelCoord(const cv::Point3f& point) const
{
    return cv::Vec3i(cvFloor(point.x * voxelSizeInv), cvFloor(point.y * voxelSizeInv),
                     cvFloor(point.z * voxelSizeInv));
}

inline TsdfVoxel HashTSDFVolumeCPU::_at(const cv::Vec3i& volumeIdx, int indx) const
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

inline TsdfVoxel HashTSDFVolumeCPU::at(const cv::Vec3i& volumeIdx) const
{
    Vec3i volumeUnitIdx = Vec3i(volumeIdx[0] >> volumeUnitDegree,
                                volumeIdx[1] >> volumeUnitDegree,
                                volumeIdx[2] >> volumeUnitDegree);

    VolumeUnitIndexes::const_iterator it = volumeUnits.find(volumeUnitIdx);

    if (it == volumeUnits.end())
    {
        return TsdfVoxel(floatToTsdf(1.f), 0);
    }

    cv::Vec3i volUnitLocalIdx = volumeIdx - cv::Vec3i(volumeUnitIdx[0] << volumeUnitDegree,
                                                      volumeUnitIdx[1] << volumeUnitDegree,
                                                      volumeUnitIdx[2] << volumeUnitDegree);

    volUnitLocalIdx =
        cv::Vec3i(abs(volUnitLocalIdx[0]), abs(volUnitLocalIdx[1]), abs(volUnitLocalIdx[2]));
    return _at(volUnitLocalIdx, it->second.index);

}

TsdfVoxel HashTSDFVolumeCPU::at(const Point3f& point) const
{
    cv::Vec3i volumeUnitIdx = volumeToVolumeUnitIdx(point);
    VolumeUnitIndexes::const_iterator it = volumeUnits.find(volumeUnitIdx);

    if (it == volumeUnits.end())
    {
        return TsdfVoxel(floatToTsdf(1.f), 0);
    }

    cv::Point3f volumeUnitPos = volumeUnitIdxToVolume(volumeUnitIdx);
    cv::Vec3i volUnitLocalIdx = volumeToVoxelCoord(point - volumeUnitPos);
    volUnitLocalIdx =
        cv::Vec3i(abs(volUnitLocalIdx[0]), abs(volUnitLocalIdx[1]), abs(volUnitLocalIdx[2]));
    return _at(volUnitLocalIdx, it->second.index);
}

TsdfVoxel HashTSDFVolumeCPU::atVolumeUnit(const Vec3i& point, const Vec3i& volumeUnitIdx, VolumeUnitIndexes::const_iterator it) const
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

float HashTSDFVolumeCPU::interpolateVoxelPoint(const Point3f& point) const
{
    const Vec3i neighbourCoords[] = { {0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1},
                                      {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1} };

    // A small hash table to reduce a number of find() calls
    bool queried[8];
    VolumeUnitIndexes::const_iterator iterMap[8];
    for (int i = 0; i < 8; i++)
    {
        iterMap[i] = volumeUnits.end();
        queried[i] = false;
    }

    int ix = cvFloor(point.x);
    int iy = cvFloor(point.y);
    int iz = cvFloor(point.z);

    float tx = point.x - ix;
    float ty = point.y - iy;
    float tz = point.z - iz;

    Vec3i iv(ix, iy, iz);
    float vx[8];
    for (int i = 0; i < 8; i++)
    {
        Vec3i pt = iv + neighbourCoords[i];

        Vec3i volumeUnitIdx = Vec3i(pt[0] >> volumeUnitDegree, pt[1] >> volumeUnitDegree, pt[2] >> volumeUnitDegree);
        int dictIdx = (volumeUnitIdx[0] & 1) + (volumeUnitIdx[1] & 1) * 2 + (volumeUnitIdx[2] & 1) * 4;
        auto it = iterMap[dictIdx];
        if (!queried[dictIdx])
        {
            it = volumeUnits.find(volumeUnitIdx);
            iterMap[dictIdx] = it;
            queried[dictIdx] = true;
        }

        vx[i] = atVolumeUnit(pt, volumeUnitIdx, it).tsdf;
    }

    return interpolate(tx, ty, tz, vx);
}

inline float HashTSDFVolumeCPU::interpolateVoxel(const cv::Point3f& point) const
{
    return interpolateVoxelPoint(point * voxelSizeInv);
}


Point3f HashTSDFVolumeCPU::getNormalVoxel(const Point3f &point) const
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

        vals[i] = tsdfToFloat(atVolumeUnit(pt, volumeUnitIdx, it).tsdf);
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
    const int idxxp[8] = {  8,  9, 10, 11,  0,  1,  2,  3 };
    const int idxxn[8] = {  4,  5,  6,  7, 12, 13, 14, 15 };
    const int idxyp[8] = { 16, 17,  0,  1, 18, 19,  4,  5 };
    const int idxyn[8] = {  2,  3, 20, 21,  6,  7, 22, 23 };
    const int idxzp[8] = { 24,  0, 25,  2, 26,  4, 27,  6 };
    const int idxzn[8] = {  1, 28,  3, 29,  5, 30,  7, 31 };

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

void HashTSDFVolumeCPU::raycast(const Matx44f& cameraPose, const kinfu::Intr& intrinsics, const Size& frameSize,
                                OutputArray _points, OutputArray _normals) const
{
    CV_TRACE_FUNCTION();
    CV_Assert(frameSize.area() > 0);

    _points.create(frameSize, POINT_TYPE);
    _normals.create(frameSize, POINT_TYPE);

    Points points1   = _points.getMat();
    Normals normals1 = _normals.getMat();

    Points& points(points1);
    Normals& normals(normals1);
    const HashTSDFVolumeCPU& volume(*this);
    const float tstep(volume.truncDist * volume.raycastStepFactor);
    const Affine3f cam2vol(volume.pose.inv() * Affine3f(cameraPose));
    const Affine3f vol2cam(Affine3f(cameraPose.inv()) * volume.pose);
    const Intr::Reprojector reproj(intrinsics.makeReprojector());

    const int nstripes = -1;

    auto _HashRaycastInvoker = [&](const Range& range)
    {
        const Point3f cam2volTrans = cam2vol.translation();
        const Matx33f cam2volRot = cam2vol.rotation();
        const Matx33f vol2camRot = vol2cam.rotation();

        const float blockSize = volume.volumeUnitSize;

        for (int y = range.start; y < range.end; y++)
        {
            ptype* ptsRow = points[y];
            ptype* nrmRow = normals[y];

            for (int x = 0; x < points.cols; x++)
            {
                //! Initialize default value
                Point3f point = nan3, normal = nan3;

                //! Ray origin and direction in the volume coordinate frame
                Point3f orig    = cam2volTrans;
                Point3f rayDirV = normalize(Vec3f(cam2volRot * reproj(Point3f(float(x), float(y), 1.f))));

                float tmin = 0;
                float tmax = volume.truncateThreshold;
                float tcurr = tmin;

                cv::Vec3i prevVolumeUnitIdx =
                    cv::Vec3i(std::numeric_limits<int>::min(), std::numeric_limits<int>::min(),
                        std::numeric_limits<int>::min());

                float tprev = tcurr;
                float prevTsdf = volume.truncDist;
                Ptr<TSDFVolumeCPU> currVolumeUnit;
                while (tcurr < tmax)
                {
                    Point3f currRayPos = orig + tcurr * rayDirV;
                    cv::Vec3i currVolumeUnitIdx = volume.volumeToVolumeUnitIdx(currRayPos);

                    VolumeUnitIndexes::const_iterator it = volume.volumeUnits.find(currVolumeUnitIdx);

                    float currTsdf = prevTsdf;
                    int currWeight = 0;
                    float stepSize = 0.5f * blockSize;
                    cv::Vec3i volUnitLocalIdx;


                    //! The subvolume exists in hashtable
                    if (it != volume.volumeUnits.end())
                    {
                        cv::Point3f currVolUnitPos =
                            volume.volumeUnitIdxToVolume(currVolumeUnitIdx);
                        volUnitLocalIdx = volume.volumeToVoxelCoord(currRayPos - currVolUnitPos);

                        //! TODO: Figure out voxel interpolation
                        TsdfVoxel currVoxel = _at(volUnitLocalIdx, it->second.index);
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
                            Point3f nv = volume.getNormalVoxel(pv);

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
}

void HashTSDFVolumeCPU::fetchPointsNormals(OutputArray _points, OutputArray _normals) const
{
    CV_TRACE_FUNCTION();

    if (_points.needed())
    {
        std::vector<std::vector<ptype>> pVecs, nVecs;

        std::vector<Vec3i> totalVolUnits;
        for (const auto& keyvalue : volumeUnits)
        {
            totalVolUnits.push_back(keyvalue.first);
        }
        Range fetchRange(0, (int)totalVolUnits.size());
        const int nstripes = -1;

        const HashTSDFVolumeCPU& volume(*this);
        bool needNormals(_normals.needed());
        Mutex mutex;

        auto HashFetchPointsNormalsInvoker = [&](const Range& range)
        {
            std::vector<ptype> points, normals;
            for (int i = range.start; i < range.end; i++)
            {
                cv::Vec3i tsdf_idx = totalVolUnits[i];

                VolumeUnitIndexes::const_iterator it = volume.volumeUnits.find(tsdf_idx);
                Point3f base_point = volume.volumeUnitIdxToVolume(tsdf_idx);
                if (it != volume.volumeUnits.end())
                {
                    std::vector<ptype> localPoints;
                    std::vector<ptype> localNormals;
                    for (int x = 0; x < volume.volumeUnitResolution; x++)
                        for (int y = 0; y < volume.volumeUnitResolution; y++)
                            for (int z = 0; z < volume.volumeUnitResolution; z++)
                            {
                                cv::Vec3i voxelIdx(x, y, z);
                                TsdfVoxel voxel = _at(voxelIdx, it->second.index);

                                if (voxel.tsdf != -128 && voxel.weight != 0)
                                {
                                    Point3f point = base_point + volume.voxelCoordToVolume(voxelIdx);
                                    localPoints.push_back(toPtype(point));
                                    if (needNormals)
                                    {
                                        Point3f normal = volume.getNormalVoxel(point);
                                        localNormals.push_back(toPtype(normal));
                                    }
                                }
                            }

                    AutoLock al(mutex);
                    pVecs.push_back(localPoints);
                    nVecs.push_back(localNormals);
                }
            }
        };

        parallel_for_(fetchRange, HashFetchPointsNormalsInvoker, nstripes);

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
}

void HashTSDFVolumeCPU::fetchNormals(InputArray _points, OutputArray _normals) const
{
    CV_TRACE_FUNCTION();

    if (_normals.needed())
    {
        Points points = _points.getMat();
        CV_Assert(points.type() == POINT_TYPE);

        _normals.createSameSize(_points, _points.type());
        Normals normals = _normals.getMat();

        const HashTSDFVolumeCPU& _volume = *this;
        auto HashPushNormals             = [&](const ptype& point, const int* position) {
            const HashTSDFVolumeCPU& volume(_volume);
            Affine3f invPose(volume.pose.inv());
            Point3f p = fromPtype(point);
            Point3f n = nan3;
            if (!isNaN(p))
            {
                Point3f voxelPoint = invPose * p;
                n                  = volume.pose.rotation() * volume.getNormalVoxel(voxelPoint);
            }
            normals(position[0], position[1]) = toPtype(n);
        };
        points.forEach(HashPushNormals);
    }
}

int HashTSDFVolumeCPU::getVisibleBlocks(int currFrameId, int frameThreshold) const
{
    int numVisibleBlocks = 0;
    //! TODO: Iterate over map parallely?
    for (const auto& keyvalue : volumeUnits)
    {
        const VolumeUnit& volumeUnit = keyvalue.second;
        if (volumeUnit.lastVisibleIndex > (currFrameId - frameThreshold))
            numVisibleBlocks++;
    }
    return numVisibleBlocks;
}


///////// GPU implementation /////////

#ifdef HAVE_OPENCL

class HashTSDFVolumeGPU : public HashTSDFVolume
{
public:
    HashTSDFVolumeGPU(float _voxelSize, const Matx44f& _pose, float _raycastStepFactor, float _truncDist, int _maxWeight,
        float _truncateThreshold, int _volumeUnitRes, bool zFirstMemOrder = false);

    HashTSDFVolumeGPU(const VolumeParams& _volumeParams, bool zFirstMemOrder = false);

    void reset() override;

    void integrateAllVolumeUnitsGPU(const UMat& depth, float depthFactor, const Matx44f& cameraPose, const Intr& intrinsics);

    void allocateVolumeUnits(const UMat& depth, float depthFactor, const Matx44f& cameraPose, const Intr& intrinsics);

    void markActive(const Matx44f& cameraPose, const Intr& intrinsics, const Size frameSz, const int frameId);

    virtual void integrate(InputArray, InputArray, float, const Matx44f&, const kinfu::Intr&, const Intr&, const int) override
        { CV_Error(Error::StsNotImplemented, "Not implemented"); };
    void integrate(InputArray _depth, float depthFactor, const Matx44f& cameraPose, const kinfu::Intr& intrinsics,
        const int frameId = 0) override;
    void raycast(const Matx44f& cameraPose, const kinfu::Intr& intrinsics, const Size& frameSize, OutputArray points,
        OutputArray normals) const override;
    void raycast(const Matx44f&, const kinfu::Intr&, const Size&, OutputArray, OutputArray, OutputArray) const override
        { CV_Error(Error::StsNotImplemented, "Not implemented"); };

    void fetchNormals(InputArray points, OutputArray _normals) const override;
    void fetchPointsNormals(OutputArray points, OutputArray normals) const override;

    size_t getTotalVolumeUnits() const override { return size_t(hashTable.last); }
    int getVisibleBlocks(int currFrameId, int frameThreshold) const override;



    //! Return the voxel given the point in volume coordinate system i.e., (metric scale 1 unit =
    //! 1m)
    virtual TsdfVoxel new_at(const cv::Vec3i& volumeIdx, int indx) const;
    TsdfVoxel new_atVolumeUnit(const Vec3i& point, const Vec3i& volumeUnitIdx, int indx) const;


    float interpolateVoxelPoint(const Point3f& point) const;
    float interpolateVoxel(const cv::Point3f& point) const;
    Point3f getNormalVoxel(const cv::Point3f& p) const;

    //! Utility functions for coordinate transformations
    Vec3i volumeToVolumeUnitIdx(const Point3f& point) const;
    Point3f volumeUnitIdxToVolume(const Vec3i& volumeUnitIdx) const;

    Point3f voxelCoordToVolume(const Vec3i& voxelIdx) const;
    Vec3i volumeToVoxelCoord(const Point3f& point) const;

public:
    Vec6f frameParams;
    int bufferSizeDegree;

    // per-volume-unit data
    cv::UMat lastVisibleIndices;

    cv::UMat isActiveFlags;

    cv::UMat volUnitsData;
    //TODO: remove it when there's no CPU parts
    cv::Mat volUnitsDataCopy;

    cv::UMat pixNorms;

    //TODO: move indexes.volumes to GPU
    CustomHashSet hashTable;
};

HashTSDFVolumeGPU::HashTSDFVolumeGPU(float _voxelSize, const Matx44f& _pose, float _raycastStepFactor, float _truncDist, int _maxWeight,
    float _truncateThreshold, int _volumeUnitRes, bool _zFirstMemOrder)
    :HashTSDFVolume(_voxelSize, _pose, _raycastStepFactor, _truncDist, _maxWeight, _truncateThreshold, _volumeUnitRes, _zFirstMemOrder)
{
    if (truncDist >= volumeUnitSize)
    {
        std::cout << "truncDist exceeds volume unit size, allocation may work incorrectly" << std::endl;
    }

    reset();
}

HashTSDFVolumeGPU::HashTSDFVolumeGPU(const VolumeParams & _params, bool _zFirstMemOrder)
    : HashTSDFVolumeGPU(_params.voxelSize, _params.pose.matrix, _params.raycastStepFactor, _params.tsdfTruncDist, _params.maxWeight,
                     _params.depthTruncThreshold, _params.unitResolution, _zFirstMemOrder)
{
}

// zero volume, leave rest params the same
void HashTSDFVolumeGPU::reset()
{
    CV_TRACE_FUNCTION();

    bufferSizeDegree = 15;
    int buff_lvl = (int) (1 << bufferSizeDegree);

    int volCubed = volumeUnitResolution * volumeUnitResolution * volumeUnitResolution;
    volUnitsDataCopy = cv::Mat(buff_lvl, volCubed, rawType<TsdfVoxel>());

    volUnitsData = cv::UMat(buff_lvl, volCubed, CV_8UC2);

    lastVisibleIndices = cv::UMat(buff_lvl, 1, CV_32S);

    isActiveFlags = cv::UMat(buff_lvl, 1, CV_8U);

    hashTable = CustomHashSet();

    frameParams = Vec6f();
    pixNorms = UMat();
}


void HashTSDFVolumeGPU::integrateAllVolumeUnitsGPU(const UMat& depth, float depthFactor, const Matx44f& cameraPose, const Intr& intrinsics)
{
    CV_TRACE_FUNCTION();
    CV_Assert(!depth.empty());

    String errorStr;
    String name = "integrateAllVolumeUnits";
    ocl::ProgramSource source = ocl::rgbd::hash_tsdf_oclsrc;
    String options = "-cl-mad-enable";
    ocl::Kernel k;
    k.create(name.c_str(), source, options, &errorStr);

    if (k.empty())
        throw std::runtime_error("Failed to create kernel: " + errorStr);

    float dfac = 1.f / depthFactor;
    Vec2f fxy(intrinsics.fx, intrinsics.fy), cxy(intrinsics.cx, intrinsics.cy);
    Matx44f vol2camMatrix = (Affine3f(cameraPose).inv() * pose).matrix;
    Matx44f camInvMatrix = Affine3f(cameraPose).inv().matrix;

    UMat hashesGpu = Mat(hashTable.hashes, false).getUMat(ACCESS_READ);
    UMat hashDataGpu = Mat(hashTable.data, false).getUMat(ACCESS_READ);

    k.args(ocl::KernelArg::ReadOnly(depth),
           ocl::KernelArg::PtrReadOnly(hashesGpu),
           ocl::KernelArg::PtrReadOnly(hashDataGpu),
           ocl::KernelArg::ReadWrite(volUnitsData),
           ocl::KernelArg::ReadOnly(pixNorms),
           ocl::KernelArg::ReadOnly(isActiveFlags),
           vol2camMatrix,
           camInvMatrix,
           voxelSize,
           volumeUnitResolution,
           volStrides.val,
           fxy.val,
           cxy.val,
           dfac,
           truncDist,
           int(maxWeight)
    );

    int resol = volumeUnitResolution;
    size_t globalSize[3];
    globalSize[0] = (size_t)resol;
    globalSize[1] = (size_t)resol;
    globalSize[2] = (size_t)hashTable.last; // num of volume units

    if (!k.run(3, globalSize, NULL, true))
        throw std::runtime_error("Failed to run kernel");
}


void HashTSDFVolumeGPU::allocateVolumeUnits(const UMat& _depth, float depthFactor, const Matx44f& cameraPose, const Intr& intrinsics)
{
    constexpr int pixCapacity = 16;
    typedef std::array<Vec3i, pixCapacity> LocalVolUnits;

    Depth depth = _depth.getMat(ACCESS_READ);

    //! Compute volumes to be allocated
    const int depthStride = volumeUnitDegree;
    const float invDepthFactor = 1.f / depthFactor;
    const Intr::Reprojector reproj(intrinsics.makeReprojector());
    const Affine3f cam2vol(pose.inv() * Affine3f(cameraPose));
    const Point3f truncPt(truncDist, truncDist, truncDist);
    Mutex mutex;

    // for new indices
    CustomHashSet thm;

    auto fillLocalAcessVolUnits = [&](const Range& xrange, const Range& yrange, CustomHashSet& ghm)
    {
        for (int y = yrange.start; y < yrange.end; y += depthStride)
        {
            const depthType* depthRow = depth[y];
            for (int x = xrange.start; x < xrange.end; x += depthStride)
            {
                depthType z = depthRow[x] * invDepthFactor;
                if (z <= 0 || z > this->truncateThreshold)
                    continue;
                Point3f camPoint = reproj(Point3f((float)x, (float)y, z));
                Point3f volPoint = cam2vol * camPoint;
                //! Find accessed TSDF volume unit for valid 3D vertex
                Vec3i lower_bound = this->volumeToVolumeUnitIdx(volPoint - truncPt);
                Vec3i upper_bound = this->volumeToVolumeUnitIdx(volPoint + truncPt);

                int pixLocalCounter = 0;
                LocalVolUnits pixLocalVolUnits;
                for (int i = lower_bound[0]; i <= upper_bound[0]; i++)
                    for (int j = lower_bound[1]; j <= upper_bound[1]; j++)
                        for (int k = lower_bound[2]; k <= upper_bound[2]; k++)
                        {
                            const Vec3i tsdf_idx = Vec3i(i, j, k);

                            if (hashTable.find(tsdf_idx) < 0)
                            {
                                bool found = false;
                                for (int c = 0; c < pixLocalCounter; c++)
                                {
                                    if (pixLocalVolUnits[c] == tsdf_idx)
                                    {
                                        found = true; break;
                                    }
                                }
                                if (!found)
                                {
                                    pixLocalVolUnits[pixLocalCounter++] = tsdf_idx;
                                    if (pixLocalCounter >= pixCapacity)
                                    {
                                        return;
                                    }
                                }
                            }
                        }

                // lock localAccessVolUnits somehow
                for (int i = 0; i < pixLocalCounter; i++)
                {
                    Vec3i idx = pixLocalVolUnits[i];
                    if (!ghm.insert(idx))
                    {
                        //return;
                    }
                }
                // unlock
            }
        }
    };

    Rect dim(0, 0, depth.cols, depth.rows);
    Size gsz(32, 32);
    Size gg(divUp(dim.width, gsz.width), divUp(dim.height, gsz.height));

    bool needReallocation = false;
    auto allocateLambda = [&](const Range& r)
    {

    for (int yg = r.start; yg < r.end; yg++)
    {
        for (int xg = 0; xg < gg.width; xg++)
        {
            Rect gr(xg * gsz.width, yg * gsz.height, (xg + 1) * gsz.width, (yg + 1) * gsz.height);
            gr = gr & dim;
            Range xr(gr.tl().x, gr.br().x), yr(gr.tl().y, gr.br().y);

            CustomHashSet ghm;

            fillLocalAcessVolUnits(xr, yr, ghm);

            if (ghm.last)
            {
                std::lock_guard<std::recursive_mutex> al(mutex);

                for (int i = 0; i < ghm.last; i++)
                {
                    Vec4i node = ghm.data[i];
                    Vec3i idx(node[0], node[1], node[2]);

                    //TODO: 1. add to separate hash map instead, then merge on GPU side

                    int result = thm.insert(idx);
                    if (!result)
                    {
                        needReallocation = true;
                        return;
                    }
                }
            }
        }
    }

    };

    do
    {
        if (needReallocation)
        {
            thm.capacity *= 2;
            thm.data.resize(thm.capacity);

            needReallocation = false;
        }

        parallel_for_(Range(0, gg.height), allocateLambda);
    } while (needReallocation);


    auto pushToGlobal = [](const CustomHashSet _thm, CustomHashSet& _globalHashMap,
                           bool& _needReallocation, Mutex& _mutex)
    {
        for (int i = 0; i < _thm.last; i++)
        {
                Vec4i node = _thm.data[i];
                Vec3i idx(node[0], node[1], node[2]);

                std::lock_guard<std::recursive_mutex> al(_mutex);

                int result = _globalHashMap.insert(idx);
                if (result == 0)
                {
                    _needReallocation = true;
                    return;
                }
        }
    };

    needReallocation = false;
    do
    {
        if (needReallocation)
        {
            hashTable.capacity *= 2;
            hashTable.data.resize(hashTable.capacity);

            needReallocation = false;
        }

        pushToGlobal(thm, hashTable, needReallocation, mutex);
    } while (needReallocation);
}


void HashTSDFVolumeGPU::markActive(const Matx44f& cameraPose, const Intr& intrinsics, const Size frameSz, const int frameId)
{
    //! Mark volumes in the camera frustum as active
    String errorStr;
    String name = "markActive";
    ocl::ProgramSource source = ocl::rgbd::hash_tsdf_oclsrc;
    String options = "-cl-mad-enable";
    ocl::Kernel k;
    k.create(name.c_str(), source, options, &errorStr);

    if (k.empty())
        throw std::runtime_error("Failed to create kernel: " + errorStr);

    const Affine3f vol2cam(Affine3f(cameraPose.inv()) * pose);
    const Intr::Projector proj(intrinsics.makeProjector());
    Vec2f fxy(proj.fx, proj.fy), cxy(proj.cx, proj.cy);

    UMat hashDataGpu = Mat(hashTable.data, false).getUMat(ACCESS_READ);

    k.args(
        ocl::KernelArg::PtrReadOnly(hashDataGpu),
        ocl::KernelArg::WriteOnly(isActiveFlags),
        ocl::KernelArg::WriteOnly(lastVisibleIndices),
        vol2cam.matrix,
        fxy,
        cxy,
        frameSz,
        volumeUnitSize,
        hashTable.last,
        truncateThreshold,
        frameId
    );

    size_t globalSize[1] = { (size_t)hashTable.last };
    if (!k.run(1, globalSize, nullptr, true))
        throw std::runtime_error("Failed to run kernel");
}


void HashTSDFVolumeGPU::integrate(InputArray _depth, float depthFactor, const Matx44f& cameraPose, const Intr& intrinsics, const int frameId)
{
    CV_TRACE_FUNCTION();

    CV_Assert(_depth.type() == DEPTH_TYPE);
    UMat depth = _depth.getUMat();

    // Save length to fill new data in ranges
    int sizeBefore = hashTable.last;
    allocateVolumeUnits(depth, depthFactor, cameraPose, intrinsics);
    int sizeAfter = hashTable.last;
    //! Perform the allocation

    // Grow buffers
    int buff_lvl = (int)(1 << bufferSizeDegree);
    if (sizeAfter >= buff_lvl)
    {
        bufferSizeDegree = (int)(log2(sizeAfter) + 1); // clz() would be better
        int oldBuffSize = buff_lvl;
        buff_lvl = (int)pow(2, bufferSizeDegree);

        volUnitsDataCopy.resize(buff_lvl);

        Range oldr(0, oldBuffSize);
        int volCubed = volumeUnitResolution * volumeUnitResolution * volumeUnitResolution;
        UMat newData(buff_lvl, volCubed, CV_8UC2);
        volUnitsData.copyTo(newData.rowRange(oldr));
        volUnitsData = newData;

        UMat newLastVisibleIndices(buff_lvl, 1, CV_32S);
        lastVisibleIndices.copyTo(newLastVisibleIndices.rowRange(oldr));
        lastVisibleIndices = newLastVisibleIndices;

        UMat newIsActiveFlags(buff_lvl, 1, CV_8U);
        isActiveFlags.copyTo(newIsActiveFlags.rowRange(oldr));
        isActiveFlags = newIsActiveFlags;
    }

    // Fill data for new volume units
    Range r(sizeBefore, sizeAfter);
    if (r.start < r.end)
    {
        lastVisibleIndices.rowRange(r) = frameId;
        isActiveFlags.rowRange(r) = 1;

        TsdfVoxel emptyVoxel(floatToTsdf(0.0f), 0);
        volUnitsData.rowRange(r) = Vec2b((uchar)(emptyVoxel.tsdf), (uchar)(emptyVoxel.weight));
    }

    //! Mark volumes in the camera frustum as active
    markActive(cameraPose, intrinsics, depth.size(), frameId);

    Vec6f newParams((float)depth.rows, (float)depth.cols,
                     intrinsics.fx, intrinsics.fy,
                     intrinsics.cx, intrinsics.cy);
    if (!(frameParams == newParams))
    {
        frameParams = newParams;
        pixNorms = preCalculationPixNormGPU(depth, intrinsics);
    }

    //! Integrate the correct volumeUnits
    integrateAllVolumeUnitsGPU(depth, depthFactor, cameraPose, intrinsics);
}


//TODO: remove these functions when everything is done on GPU
cv::Vec3i HashTSDFVolumeGPU::volumeToVolumeUnitIdx(const cv::Point3f& p) const
{
    return cv::Vec3i(cvFloor(p.x / volumeUnitSize), cvFloor(p.y / volumeUnitSize),
        cvFloor(p.z / volumeUnitSize));
}

cv::Point3f HashTSDFVolumeGPU::volumeUnitIdxToVolume(const cv::Vec3i& volumeUnitIdx) const
{
    return cv::Point3f(volumeUnitIdx[0] * volumeUnitSize, volumeUnitIdx[1] * volumeUnitSize,
        volumeUnitIdx[2] * volumeUnitSize);
}

cv::Point3f HashTSDFVolumeGPU::voxelCoordToVolume(const cv::Vec3i& voxelIdx) const
{
    return cv::Point3f(voxelIdx[0] * voxelSize, voxelIdx[1] * voxelSize, voxelIdx[2] * voxelSize);
}

cv::Vec3i HashTSDFVolumeGPU::volumeToVoxelCoord(const cv::Point3f& point) const
{
    return cv::Vec3i(cvFloor(point.x * voxelSizeInv), cvFloor(point.y * voxelSizeInv),
        cvFloor(point.z * voxelSizeInv));
}

inline TsdfVoxel HashTSDFVolumeGPU::new_at(const cv::Vec3i& volumeIdx, int indx) const
{
    //! Out of bounds
    if ((volumeIdx[0] >= volumeUnitResolution || volumeIdx[0] < 0) ||
        (volumeIdx[1] >= volumeUnitResolution || volumeIdx[1] < 0) ||
        (volumeIdx[2] >= volumeUnitResolution || volumeIdx[2] < 0))
    {
        return TsdfVoxel(floatToTsdf(1.0f), 0);
    }

    const TsdfVoxel* volData = volUnitsDataCopy.ptr<TsdfVoxel>(indx);
    int coordBase =
        volumeIdx[0] * volStrides[0] +
        volumeIdx[1] * volStrides[1] +
        volumeIdx[2] * volStrides[2];
    return volData[coordBase];
}

TsdfVoxel HashTSDFVolumeGPU::new_atVolumeUnit(const Vec3i& point, const Vec3i& volumeUnitIdx, int indx) const
{
    if (indx < 0)
    {
        return TsdfVoxel(floatToTsdf(1.f), 0);
    }
    Vec3i volUnitLocalIdx = point - Vec3i(volumeUnitIdx[0] << volumeUnitDegree,
                                          volumeUnitIdx[1] << volumeUnitDegree,
                                          volumeUnitIdx[2] << volumeUnitDegree);

    // expanding at(), removing bounds check
    const TsdfVoxel* volData = volUnitsDataCopy.ptr<TsdfVoxel>(indx);
    int coordBase = volUnitLocalIdx[0] * volStrides[0] +
        volUnitLocalIdx[1] * volStrides[1] +
        volUnitLocalIdx[2] * volStrides[2];
    return volData[coordBase];
}

float HashTSDFVolumeGPU::interpolateVoxelPoint(const Point3f& point) const
{
    const Vec3i local_neighbourCoords[] = { {0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1},
                                            {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1} };

    // A small hash table to reduce a number of find() calls
    // -2 and lower means not queried yet
    // -1 means not found
    // 0+ means found
    int iterMap[8];
    for (int i = 0; i < 8; i++)
    {
        iterMap[i] = -2;
    }

    int ix = cvFloor(point.x);
    int iy = cvFloor(point.y);
    int iz = cvFloor(point.z);

    float tx = point.x - ix;
    float ty = point.y - iy;
    float tz = point.z - iz;

    Vec3i iv(ix, iy, iz);
    float vx[8];
    for (int i = 0; i < 8; i++)
    {
        Vec3i pt = iv + local_neighbourCoords[i];

        Vec3i volumeUnitIdx = Vec3i(pt[0] >> volumeUnitDegree, pt[1] >> volumeUnitDegree, pt[2] >> volumeUnitDegree);
        int dictIdx = (volumeUnitIdx[0] & 1) + (volumeUnitIdx[1] & 1) * 2 + (volumeUnitIdx[2] & 1) * 4;
        auto it = iterMap[dictIdx];
        if (it < -1)
        {
            it = hashTable.find(volumeUnitIdx);
            iterMap[dictIdx] = it;
        }

        vx[i] = new_atVolumeUnit(pt, volumeUnitIdx, it).tsdf;
    }

    return interpolate(tx, ty, tz, vx);
}

inline float HashTSDFVolumeGPU::interpolateVoxel(const cv::Point3f& point) const
{
    return interpolateVoxelPoint(point * voxelSizeInv);
}

Point3f HashTSDFVolumeGPU::getNormalVoxel(const Point3f& point) const
{
    Vec3f normal = Vec3f(0, 0, 0);

    Point3f ptVox = point * voxelSizeInv;
    Vec3i iptVox(cvFloor(ptVox.x), cvFloor(ptVox.y), cvFloor(ptVox.z));

    // A small hash table to reduce a number of find() calls
    // -2 and lower means not queried yet
    // -1 means not found
    // 0+ means found
    int iterMap[8];
    for (int i = 0; i < 8; i++)
    {
        iterMap[i] = -2;
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
        if (it < -1)
        {
            it = hashTable.find(volumeUnitIdx);
            iterMap[dictIdx] = it;
        }

        vals[i] = tsdfToFloat(new_atVolumeUnit(pt, volumeUnitIdx, it).tsdf);
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


void HashTSDFVolumeGPU::raycast(const Matx44f& cameraPose, const kinfu::Intr& intrinsics, const Size& frameSize,
                                OutputArray _points, OutputArray _normals) const
{
    CV_TRACE_FUNCTION();
    CV_Assert(frameSize.area() > 0);

    String errorStr;
    String name = "raycast";
    ocl::ProgramSource source = ocl::rgbd::hash_tsdf_oclsrc;
    String options = "-cl-mad-enable";
    ocl::Kernel k;
    k.create(name.c_str(), source, options, &errorStr);

    if (k.empty())
        throw std::runtime_error("Failed to create kernel: " + errorStr);

    _points.create(frameSize, CV_32FC4);
    _normals.create(frameSize, CV_32FC4);

    UMat points = _points.getUMat();
    UMat normals = _normals.getUMat();

    Intr::Reprojector r = intrinsics.makeReprojector();
    Vec2f finv(r.fxinv, r.fyinv), cxy(r.cx, r.cy);

    Vec4f boxMin, boxMax(volumeUnitSize - voxelSize,
                         volumeUnitSize - voxelSize,
                         volumeUnitSize - voxelSize);

    float tstep = truncDist * raycastStepFactor;

    const HashTSDFVolumeGPU& volume(*this);
    const Affine3f cam2vol(volume.pose.inv() * Affine3f(cameraPose));
    const Affine3f vol2cam(Affine3f(cameraPose.inv()) * volume.pose);

    Matx44f cam2volRotGPU = cam2vol.matrix;
    Matx44f vol2camRotGPU = vol2cam.matrix;

    UMat volPoseGpu = Mat(pose.matrix).getUMat(ACCESS_READ);
    UMat invPoseGpu = Mat(pose.inv().matrix).getUMat(ACCESS_READ);

    UMat hashesGpu = Mat(hashTable.hashes, false).getUMat(ACCESS_READ);
    UMat hashDataGpu = Mat(hashTable.data, false).getUMat(ACCESS_READ);

    k.args(
        ocl::KernelArg::PtrReadOnly(hashesGpu),
        ocl::KernelArg::PtrReadOnly(hashDataGpu),
        ocl::KernelArg::WriteOnlyNoSize(points),
        ocl::KernelArg::WriteOnlyNoSize(normals),
        frameSize,
        ocl::KernelArg::ReadOnly(volUnitsData),
        cam2volRotGPU,
        vol2camRotGPU,
        float(volume.truncateThreshold),
        finv.val, cxy.val,
        boxMin.val, boxMax.val,
        tstep,
        voxelSize,
        voxelSizeInv,
        volumeUnitSize,
        volume.truncDist,
        volumeUnitDegree,
        volStrides
    );

    size_t globalSize[2];
    globalSize[0] = (size_t)frameSize.width;
    globalSize[1] = (size_t)frameSize.height;

    if (!k.run(2, globalSize, NULL, true))
        throw std::runtime_error("Failed to run kernel");
}

void HashTSDFVolumeGPU::fetchPointsNormals(OutputArray _points, OutputArray _normals) const
{
    CV_TRACE_FUNCTION();

    if (_points.needed())
    {
        //TODO: remove it when it works w/o CPU code
        volUnitsData.copyTo(volUnitsDataCopy);
        //TODO: remove it when it works w/o CPU code
        //TODO: enable it when it's on GPU
        //UMat hashDataGpu(hashMap.capacity, 1, CV_32SC4);
        //Mat(hashMap.data, false).copyTo(hashDataGpu);

        std::vector<std::vector<ptype>> pVecs, nVecs;

        Range _fetchRange(0, hashTable.last);

        const int nstripes = -1;

        const HashTSDFVolumeGPU& volume(*this);
        bool needNormals(_normals.needed());
        Mutex mutex;

        auto _HashFetchPointsNormalsInvoker = [&](const Range& range)
        {
            std::vector<ptype> points, normals;
            for (int row = range.start; row < range.end; row++)
            {
                cv::Vec4i idx4 = hashTable.data[row];
                cv::Vec3i idx(idx4[0], idx4[1], idx4[2]);

                Point3f base_point = volume.volumeUnitIdxToVolume(idx);

                std::vector<ptype> localPoints;
                std::vector<ptype> localNormals;
                for (int x = 0; x < volume.volumeUnitResolution; x++)
                    for (int y = 0; y < volume.volumeUnitResolution; y++)
                        for (int z = 0; z < volume.volumeUnitResolution; z++)
                        {
                            cv::Vec3i voxelIdx(x, y, z);
                            TsdfVoxel voxel = new_at(voxelIdx, row);

                            if (voxel.tsdf != -128 && voxel.weight != 0)
                            {
                                Point3f point = base_point + volume.voxelCoordToVolume(voxelIdx);

                                localPoints.push_back(toPtype(point));
                                if (needNormals)
                                {
                                    Point3f normal = volume.getNormalVoxel(point);
                                    localNormals.push_back(toPtype(normal));
                                }
                            }
                        }

                AutoLock al(mutex);
                pVecs.push_back(localPoints);
                nVecs.push_back(localNormals);
            }
        };

        parallel_for_(_fetchRange, _HashFetchPointsNormalsInvoker, nstripes);

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
}

void HashTSDFVolumeGPU::fetchNormals(InputArray _points, OutputArray _normals) const
{
    CV_TRACE_FUNCTION();

    if (_normals.needed())
    {
        //TODO: remove it when it works w/o CPU code
        volUnitsData.copyTo(volUnitsDataCopy);

        Points points = _points.getMat();
        CV_Assert(points.type() == POINT_TYPE);
        _normals.createSameSize(_points, _points.type());
        Normals normals = _normals.getMat();
        const HashTSDFVolumeGPU& _volume = *this;
        auto HashPushNormals             = [&](const ptype& point, const int* position) {
            const HashTSDFVolumeGPU& volume(_volume);
            Affine3f invPose(volume.pose.inv());
            Point3f p = fromPtype(point);
            Point3f n = nan3;
            if (!isNaN(p))
            {
                Point3f voxelPoint = invPose * p;
                n                  = volume.pose.rotation() * volume.getNormalVoxel(voxelPoint);
            }
            normals(position[0], position[1]) = toPtype(n);
        };
        points.forEach(HashPushNormals);
    }

}

int HashTSDFVolumeGPU::getVisibleBlocks(int currFrameId, int frameThreshold) const
{
    Mat cpuIndices = lastVisibleIndices.getMat(ACCESS_READ);

    int numVisibleBlocks = 0;
    //! TODO: Iterate over map parallely?
    for (int i = 0; i < hashTable.last; i++)
    {
        if (cpuIndices.at<int>(i) > (currFrameId - frameThreshold))
            numVisibleBlocks++;
    }
    return numVisibleBlocks;
}

#endif

//template<typename T>
Ptr<HashTSDFVolume> makeHashTSDFVolume(const VolumeParams& _params)
{
#ifdef HAVE_OPENCL
    if (ocl::useOpenCL())
        return makePtr<HashTSDFVolumeGPU>(_params.voxelSize, _params.pose.matrix, _params.raycastStepFactor, _params.tsdfTruncDist, _params.maxWeight,
            _params.depthTruncThreshold, _params.unitResolution);
#endif
    return makePtr<HashTSDFVolumeCPU>(_params.voxelSize, _params.pose.matrix, _params.raycastStepFactor, _params.tsdfTruncDist, _params.maxWeight,
        _params.depthTruncThreshold, _params.unitResolution);
}

//template<typename T>
Ptr<HashTSDFVolume> makeHashTSDFVolume(float _voxelSize, Matx44f _pose, float _raycastStepFactor, float _truncDist,
    int _maxWeight, float truncateThreshold, int volumeUnitResolution)
{
#ifdef HAVE_OPENCL
    if (ocl::useOpenCL())
        return makePtr<HashTSDFVolumeGPU>(_voxelSize, _pose, _raycastStepFactor, _truncDist, _maxWeight, truncateThreshold,
            volumeUnitResolution);
#endif
    return makePtr<HashTSDFVolumeCPU>(_voxelSize, _pose, _raycastStepFactor, _truncDist, _maxWeight, truncateThreshold,
        volumeUnitResolution);
}

}  // namespace kinfu
}  // namespace cv
