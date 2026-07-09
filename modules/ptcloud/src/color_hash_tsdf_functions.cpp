// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// ColorHashTSDF is built on top of HashTSDF: it shares the spatially-hashed
// volume-unit storage and the raycast/fetch structure, but stores RGBTsdfVoxel
// voxels (TSDF + running-average RGB) and integrates color alongside depth.
// It is a CPU-only implementation.

#include "precomp.hpp"
#include "color_hash_tsdf_functions.hpp"

namespace cv {

namespace {

inline Vec3i volumeToVolumeUnitIdx(const Point3f& point, const float volumeUnitSize)
{
    return cv::Vec3i(
        cvFloor(point.x / volumeUnitSize),
        cvFloor(point.y / volumeUnitSize),
        cvFloor(point.z / volumeUnitSize));
}

inline cv::Point3f volumeUnitIdxToVolume(const cv::Vec3i& volumeUnitIdx, const float volumeUnitSize)
{
    return cv::Point3f(
        volumeUnitIdx[0] * volumeUnitSize,
        volumeUnitIdx[1] * volumeUnitSize,
        volumeUnitIdx[2] * volumeUnitSize);
}

inline cv::Point3f voxelCoordToVolume(const cv::Vec3i& voxelIdx, const float voxelSize)
{
    return cv::Point3f(
        voxelIdx[0] * voxelSize,
        voxelIdx[1] * voxelSize,
        voxelIdx[2] * voxelSize);
}

inline cv::Vec3i volumeToVoxelCoord(const cv::Point3f& point, const float voxelSizeInv)
{
    return cv::Vec3i(
        cvFloor(point.x * voxelSizeInv),
        cvFloor(point.y * voxelSizeInv),
        cvFloor(point.z * voxelSizeInv));
}

inline float interpolate(float tx, float ty, float tz, float vx[8])
{
    float v00 = vx[0] + tz * (vx[1] - vx[0]);
    float v01 = vx[2] + tz * (vx[3] - vx[2]);
    float v10 = vx[4] + tz * (vx[5] - vx[4]);
    float v11 = vx[6] + tz * (vx[7] - vx[6]);
    float v0  = v00 + ty * (v01 - v00);
    float v1  = v10 + ty * (v11 - v10);
    return v0 + tx * (v1 - v0);
}

// Out-of-bounds sentinel reused from HashTSDF: tsdf == floatToTsdf(1.f) (-128), weight 0.
RGBTsdfVoxel atColorHashVolumeUnit(
    const Mat& volUnitsData, const VolumeUnitIndexes& volumeUnits,
    const Vec3i& point, const Vec3i& volumeUnitIdx, VolumeUnitIndexes::const_iterator it,
    const int volumeUnitDegree, const Vec4i volStrides)
{
    if (it == volumeUnits.end())
        return RGBTsdfVoxel(floatToTsdf(1.f), 0, 0, 0, 0);

    Vec3i volUnitLocalIdx = point - Vec3i(volumeUnitIdx[0] << volumeUnitDegree,
                                          volumeUnitIdx[1] << volumeUnitDegree,
                                          volumeUnitIdx[2] << volumeUnitDegree);

    const RGBTsdfVoxel* volData = volUnitsData.ptr<RGBTsdfVoxel>(it->second.index);
    int coordBase = volUnitLocalIdx[0] * volStrides[0] +
                    volUnitLocalIdx[1] * volStrides[1] +
                    volUnitLocalIdx[2] * volStrides[2];
    return volData[coordBase];
}

inline RGBTsdfVoxel _atColorHash(Mat& volUnitsData, const cv::Vec3i& volumeIdx, int indx,
                                 const int volumeUnitResolution, const Vec4i volStrides)
{
    if ((volumeIdx[0] >= volumeUnitResolution || volumeIdx[0] < 0) ||
        (volumeIdx[1] >= volumeUnitResolution || volumeIdx[1] < 0) ||
        (volumeIdx[2] >= volumeUnitResolution || volumeIdx[2] < 0))
    {
        return RGBTsdfVoxel(floatToTsdf(1.f), 0, 0, 0, 0);
    }
    const RGBTsdfVoxel* volData = volUnitsData.ptr<RGBTsdfVoxel>(indx);
    int coordBase =
        volumeIdx[0] * volStrides[0] + volumeIdx[1] * volStrides[1] + volumeIdx[2] * volStrides[2];
    return volData[coordBase];
}

// Normal estimation reuses the HashTSDF gradient scheme; only the .tsdf field is read.
Point3f getNormalColorHashVoxel(
    const Point3f& point, const float voxelSizeInv,
    const int volumeUnitDegree, const Vec4i volStrides,
    const Mat& volUnitsData, const VolumeUnitIndexes& volumeUnits)
{
    Vec3f normal = Vec3f(0, 0, 0);

    Point3f ptVox = point * voxelSizeInv;
    Vec3i iptVox(cvFloor(ptVox.x), cvFloor(ptVox.y), cvFloor(ptVox.z));

    bool queried[8];
    VolumeUnitIndexes::const_iterator iterMap[8];
    for (int i = 0; i < 8; i++)
    {
        iterMap[i] = volumeUnits.end();
        queried[i] = false;
    }

#if !USE_INTERPOLATION_IN_GETNORMAL
    const Vec3i offsets[] = { { 1,  0,  0}, {-1,  0,  0}, { 0,  1,  0},
                              { 0, -1,  0}, { 0,  0,  1}, { 0,  0, -1} };
    const int nVals = 6;
#else
    const Vec3i offsets[] = { { 0,  0,  0}, { 0,  0,  1}, { 0,  1,  0}, { 0,  1,  1},
                              { 1,  0,  0}, { 1,  0,  1}, { 1,  1,  0}, { 1,  1,  1},
                              {-1,  0,  0}, {-1,  0,  1}, {-1,  1,  0}, {-1,  1,  1},
                              { 2,  0,  0}, { 2,  0,  1}, { 2,  1,  0}, { 2,  1,  1},
                              { 0, -1,  0}, { 0, -1,  1}, { 1, -1,  0}, { 1, -1,  1},
                              { 0,  2,  0}, { 0,  2,  1}, { 1,  2,  0}, { 1,  2,  1},
                              { 0,  0, -1}, { 0,  1, -1}, { 1,  0, -1}, { 1,  1, -1},
                              { 0,  0,  2}, { 0,  1,  2}, { 1,  0,  2}, { 1,  1,  2} };
    const int nVals = 32;
#endif

    float vals[nVals];
    for (int i = 0; i < nVals; i++)
    {
        Vec3i pt = iptVox + offsets[i];
        Vec3i volumeUnitIdx = Vec3i(pt[0] >> volumeUnitDegree,
                                    pt[1] >> volumeUnitDegree,
                                    pt[2] >> volumeUnitDegree);

        int dictIdx = (volumeUnitIdx[0] & 1) + (volumeUnitIdx[1] & 1) * 2 + (volumeUnitIdx[2] & 1) * 4;
        auto it = iterMap[dictIdx];
        if (!queried[dictIdx])
        {
            it = volumeUnits.find(volumeUnitIdx);
            iterMap[dictIdx] = it;
            queried[dictIdx] = true;
        }
        vals[i] = tsdfToFloat(atColorHashVolumeUnit(volUnitsData, volumeUnits, pt, volumeUnitIdx, it,
                                                    volumeUnitDegree, volStrides).tsdf);
    }

#if !USE_INTERPOLATION_IN_GETNORMAL
    for (int c = 0; c < 3; c++)
        normal[c] = vals[c * 2] - vals[c * 2 + 1];
#else
    const int idxxp[8] = { 8,  9, 10, 11,  0,  1,  2,  3 };
    const int idxxn[8] = { 4,  5,  6,  7, 12, 13, 14, 15 };
    const int idxyp[8] = { 16, 17,  0,  1, 18, 19,  4,  5 };
    const int idxyn[8] = { 2,  3, 20, 21,  6,  7, 22, 23 };
    const int idxzp[8] = { 24,  0, 25,  2, 26,  4, 27,  6 };
    const int idxzn[8] = { 1, 28,  3, 29,  5, 30,  7, 31 };

    float cxv[8], cyv[8], czv[8];
    for (int i = 0; i < 8; i++)
    {
        cxv[i] = vals[idxxn[i]] - vals[idxxp[i]];
        cyv[i] = vals[idxyn[i]] - vals[idxyp[i]];
        czv[i] = vals[idxzn[i]] - vals[idxzp[i]];
    }

    float tx = ptVox.x - iptVox[0];
    float ty = ptVox.y - iptVox[1];
    float tz = ptVox.z - iptVox[2];

    normal[0] = interpolate(tx, ty, tz, cxv);
    normal[1] = interpolate(tx, ty, tz, cyv);
    normal[2] = interpolate(tx, ty, tz, czv);
#endif

    float nv = sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
    return nv < 0.0001f ? nan3 : normal / nv;
}

// Trilinear color interpolation at a point given in volume coordinates (meters).
// Neighbors are looked up across volume-unit boundaries through the hash map.
Point3f getColorHashVoxel(
    const Point3f& point, const float voxelSizeInv,
    const int volumeUnitDegree, const Vec4i volStrides,
    const Mat& volUnitsData, const VolumeUnitIndexes& volumeUnits)
{
    Point3f ptVox = point * voxelSizeInv;
    Vec3i iptVox(cvFloor(ptVox.x), cvFloor(ptVox.y), cvFloor(ptVox.z));
    float tx = ptVox.x - iptVox[0];
    float ty = ptVox.y - iptVox[1];
    float tz = ptVox.z - iptVox[2];

    const Vec3i offsets[8] = {
        {0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1},
        {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}
    };

    float r[8], g[8], b[8];
    bool allValid = true;
    for (int i = 0; i < 8; i++)
    {
        Vec3i pt = iptVox + offsets[i];
        Vec3i volumeUnitIdx(pt[0] >> volumeUnitDegree, pt[1] >> volumeUnitDegree, pt[2] >> volumeUnitDegree);
        auto it = volumeUnits.find(volumeUnitIdx);
        RGBTsdfVoxel v = atColorHashVolumeUnit(volUnitsData, volumeUnits, pt, volumeUnitIdx, it,
                                               volumeUnitDegree, volStrides);
        if (v.weight == 0)
        {
            allValid = false;
            break;
        }
        r[i] = float(v.r);
        g[i] = float(v.g);
        b[i] = float(v.b);
    }

    if (!allValid)
    {
        // Fall back to the nearest voxel if any corner of the cube is empty.
        Vec3i volumeUnitIdx(iptVox[0] >> volumeUnitDegree,
                            iptVox[1] >> volumeUnitDegree,
                            iptVox[2] >> volumeUnitDegree);
        auto it = volumeUnits.find(volumeUnitIdx);
        RGBTsdfVoxel v = atColorHashVolumeUnit(volUnitsData, volumeUnits, iptVox, volumeUnitIdx, it,
                                               volumeUnitDegree, volStrides);
        return Point3f(float(v.r), float(v.g), float(v.b));
    }

    Point3f res(interpolate(tx, ty, tz, r),
                interpolate(tx, ty, tz, g),
                interpolate(tx, ty, tz, b));
    colorFix(res);
    return res;
}

} // namespace

void integrateColorHashTsdfVolumeUnit(
    const VolumeSettings& settings, const Matx44f& cameraPose,
    int& lastVolIndex, const int frameId, const int volumeUnitDegree, bool enableGrowth,
    InputArray _depth, InputArray _rgb, InputArray _pixNorms,
    InputOutputArray _volUnitsData, VolumeUnitIndexes& volumeUnits)
{
    CV_TRACE_FUNCTION();

    CV_Assert(_depth.type() == DEPTH_TYPE);
    Depth depth = _depth.getMat();
    Mat rgb = _rgb.getMat();
    Mat& volUnitsData = _volUnitsData.getMatRef();
    Mat pixNorms = _pixNorms.getMat();

    CV_Assert(!rgb.empty());
    CV_Assert(depth.size() == rgb.size());

    Matx44f _pose;
    settings.getVolumePose(_pose);
    const Affine3f pose = Affine3f(_pose);
    const Affine3f cam2vol(pose.inv() * Affine3f(cameraPose));

    Matx33f intr;
    settings.getCameraIntegrateIntrinsics(intr);
    const Intr intrinsics(intr);
    const Intr::Reprojector reproj(intrinsics.makeReprojector());

    const float maxDepth = settings.getMaxDepth();
    const float voxelSize = settings.getVoxelSize();

    Vec3i resolution;
    settings.getVolumeResolution(resolution);
    const float volumeUnitSize = voxelSize * resolution[0];

    if (enableGrowth)
    {
        const int depthStride = volumeUnitDegree;
        const float invDepthFactor = 1.f / settings.getDepthFactor();
        const float truncDist = settings.getTsdfTruncateDistance();

        const Point3f truncPt(truncDist, truncDist, truncDist);
        std::unordered_set<cv::Vec3i, tsdf_hash> newIndices;
        Mutex mutex;
        Range allocateRange(0, depth.rows);

        auto AllocateVolumeUnitsInvoker = [&](const Range& range)
        {
            std::unordered_set<cv::Vec3i, tsdf_hash> localAccessVolUnits;
            for (int y = range.start; y < range.end; y += depthStride)
            {
                const depthType* depthRow = depth[y];
                for (int x = 0; x < depth.cols; x += depthStride)
                {
                    depthType z = depthRow[x] * invDepthFactor;
                    if (z <= 0 || z > maxDepth)
                        continue;

                    Point3f camPoint = reproj(Point3f((float)x, (float)y, z));
                    Point3f volPoint = cam2vol * camPoint;
                    Vec3i lower_bound = volumeToVolumeUnitIdx(volPoint - truncPt, volumeUnitSize);
                    Vec3i upper_bound = volumeToVolumeUnitIdx(volPoint + truncPt, volumeUnitSize);

                    for (int i = lower_bound[0]; i <= upper_bound[0]; i++)
                        for (int j = lower_bound[1]; j <= upper_bound[1]; j++)
                            for (int k = lower_bound[2]; k <= upper_bound[2]; k++)
                            {
                                const Vec3i tsdf_idx = Vec3i(i, j, k);
                                if (localAccessVolUnits.count(tsdf_idx) <= 0 && volumeUnits.count(tsdf_idx) <= 0)
                                    localAccessVolUnits.emplace(tsdf_idx);
                            }
                }
            }

            mutex.lock();
            for (const auto& tsdf_idx : localAccessVolUnits)
            {
                if (!newIndices.count(tsdf_idx))
                    newIndices.emplace(tsdf_idx);
            }
            mutex.unlock();
        };
        parallel_for_(allocateRange, AllocateVolumeUnitsInvoker);

        for (auto idx : newIndices)
        {
            VolumeUnit& vu = volumeUnits.emplace(idx, VolumeUnit()).first->second;

            Matx44f subvolumePose = pose.translate(pose.rotation() * volumeUnitIdxToVolume(idx, volumeUnitSize)).matrix;
            vu.pose = subvolumePose;
            vu.index = lastVolIndex;
            if (lastVolIndex >= int(volUnitsData.size().height))
            {
                volUnitsData.resize(lastVolIndex * 2);
                CV_LOG_DEBUG(NULL, "ColorHashTSDF storage extended from " << lastVolIndex << " to " << lastVolIndex * 2 << " volume units");
            }
            lastVolIndex++;
            volUnitsData.row(vu.index).forEach<VecRGBTsdfVoxel>([](VecRGBTsdfVoxel& vv, const int* /*position*/)
            {
                RGBTsdfVoxel& v = reinterpret_cast<RGBTsdfVoxel&>(vv);
                v.tsdf = floatToTsdf(0.0f); v.weight = 0;
                v.r = v.g = v.b = 0;
            });
            vu.lastVisibleIndex = frameId;
            vu.isActive = true;
        }
    }

    std::vector<Vec3i> totalVolUnits;
    for (const auto& keyvalue : volumeUnits)
        totalVolUnits.push_back(keyvalue.first);

    Range inFrustumRange(0, (int)volumeUnits.size());
    parallel_for_(inFrustumRange, [&](const Range& range)
    {
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
            if (volUnitInCamSpace.z < 0 || volUnitInCamSpace.z > maxDepth)
            {
                it->second.isActive = false;
                continue;
            }
            Point2f cameraPoint = proj(volUnitInCamSpace);
            if (cameraPoint.x >= 0 && cameraPoint.y >= 0 && cameraPoint.x < depth.cols && cameraPoint.y < depth.rows)
            {
                it->second.lastVisibleIndex = frameId;
                it->second.isActive = true;
            }
        }
    });

    parallel_for_(Range(0, (int)totalVolUnits.size()), [&](const Range& range)
    {
        for (int i = range.start; i < range.end; i++)
        {
            Vec3i tsdf_idx = totalVolUnits[i];
            VolumeUnitIndexes::iterator it = volumeUnits.find(tsdf_idx);
            if (it == volumeUnits.end())
                return;

            VolumeUnit& volumeUnit = it->second;
            if (volumeUnit.isActive)
            {
                integrateColorTsdfVolumeUnit(settings, volumeUnit.pose, cameraPose,
                                             depth, rgb, pixNorms, volUnitsData.row(volumeUnit.index));
                volumeUnit.isActive = false;
            }
        }
    });
}

void raycastColorHashTsdfVolumeUnit(
    const VolumeSettings& settings, const Matx44f& cameraPose,
    int height, int width, InputArray intr, const int volumeUnitDegree,
    InputArray _volUnitsData, const VolumeUnitIndexes& volumeUnits,
    OutputArray _points, OutputArray _normals, OutputArray _colors)
{
    CV_TRACE_FUNCTION();
    Size frameSize(width, height);
    CV_Assert(frameSize.area() > 0);

    Matx33f mintr(intr.getMat());
    Mat volUnitsData = _volUnitsData.getMat();

    _points.create(frameSize, POINT_TYPE);
    _normals.create(frameSize, POINT_TYPE);
    if (_colors.needed())
        _colors.create(frameSize, COLOR_TYPE);

    Points points1 = _points.getMat();
    Normals normals1 = _normals.getMat();
    Points& points(points1);
    Normals& normals(normals1);

    Colors colors1;
    Colors* colors = nullptr;
    if (_colors.needed())
    {
        colors1 = _colors.getMat();
        colors = &colors1;
    }

    const float truncDist = settings.getTsdfTruncateDistance();
    const float raycastStepFactor = settings.getRaycastStepFactor();
    const float tstep = truncDist * raycastStepFactor;
    const float maxDepth = settings.getMaxDepth();
    const float voxelSize = settings.getVoxelSize();
    const float voxelSizeInv = 1.f / voxelSize;

    const Vec4i volDims;
    settings.getVolumeStrides(volDims);
    Vec3i resolution;
    settings.getVolumeResolution(resolution);
    const Point3i volResolution = Point3i(resolution);
    const float volumeUnitSize = voxelSize * resolution[0];

    Matx44f _pose;
    settings.getVolumePose(_pose);
    const Affine3f pose = Affine3f(_pose);
    const Affine3f cam2vol(pose.inv() * Affine3f(cameraPose));
    const Affine3f vol2cam(Affine3f(cameraPose.inv()) * pose);

    const Intr intrinsics(mintr);
    const Intr::Reprojector reproj(intrinsics.makeReprojector());

    const int nstripes = -1;

    auto HashRaycastInvoker = [&](const Range& range)
    {
        const Point3f cam2volTrans = cam2vol.translation();
        const Matx33f cam2volRot = cam2vol.rotation();
        const Matx33f vol2camRot = vol2cam.rotation();

        const float blockSize = volumeUnitSize;

        for (int y = range.start; y < range.end; y++)
        {
            ptype* ptsRow = points[y];
            ptype* nrmRow = normals[y];
            ptype* clrRow = colors ? (*colors)[y] : nullptr;

            for (int x = 0; x < points.cols; x++)
            {
                Point3f point = nan3, normal = nan3, color = nan3;

                Point3f orig = cam2volTrans;
                Point3f rayDirV = normalize(Vec3f(cam2volRot * reproj(Point3f(float(x), float(y), 1.f))));

                float tmin = 0;
                float tmax = maxDepth;
                float tcurr = tmin;
                cv::Vec3i prevVolumeUnitIdx(std::numeric_limits<int>::min(),
                                            std::numeric_limits<int>::min(),
                                            std::numeric_limits<int>::min());
                float tprev = tcurr;
                float prevTsdf = truncDist;
                (void)prevVolumeUnitIdx;

                while (tcurr < tmax)
                {
                    Point3f currRayPos = orig + tcurr * rayDirV;
                    cv::Vec3i currVolumeUnitIdx = volumeToVolumeUnitIdx(currRayPos, volumeUnitSize);

                    VolumeUnitIndexes::const_iterator it = volumeUnits.find(currVolumeUnitIdx);

                    float currTsdf = prevTsdf;
                    int currWeight = 0;
                    float stepSize = 0.5f * blockSize;
                    cv::Vec3i volUnitLocalIdx;

                    if (it != volumeUnits.end())
                    {
                        cv::Point3f currVolUnitPos = volumeUnitIdxToVolume(currVolumeUnitIdx, volumeUnitSize);
                        volUnitLocalIdx = volumeToVoxelCoord(currRayPos - currVolUnitPos, voxelSizeInv);
                        RGBTsdfVoxel currVoxel = _atColorHash(volUnitsData, volUnitLocalIdx, it->second.index,
                                                              volResolution.x, volDims);
                        currTsdf = tsdfToFloat(currVoxel.tsdf);
                        currWeight = currVoxel.weight;
                        stepSize = tstep;
                    }

                    if (prevTsdf > 0.f && currTsdf <= 0.f && currWeight > 0)
                    {
                        float tInterp = (tcurr * prevTsdf - tprev * currTsdf) / (prevTsdf - currTsdf);
                        if (!cvIsNaN(tInterp) && !cvIsInf(tInterp))
                        {
                            Point3f pv = orig + tInterp * rayDirV;
                            Point3f nv = getNormalColorHashVoxel(pv, voxelSizeInv, volumeUnitDegree, volDims,
                                                                 volUnitsData, volumeUnits);
                            if (!isNaN(nv))
                            {
                                normal = vol2camRot * nv;
                                point = vol2cam * pv;
                                if (colors)
                                    color = getColorHashVoxel(pv, voxelSizeInv, volumeUnitDegree, volDims,
                                                              volUnitsData, volumeUnits);
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
                if (clrRow)
                    clrRow[x] = toPtype(color);
            }
        }
    };

    parallel_for_(Range(0, points.rows), HashRaycastInvoker, nstripes);
}

void fetchNormalsFromColorHashTsdfVolumeUnit(
    const VolumeSettings& settings, InputArray _volUnitsData, const VolumeUnitIndexes& volumeUnits,
    const int volumeUnitDegree, InputArray _points, OutputArray _normals)
{
    CV_TRACE_FUNCTION();

    if (!_normals.needed())
        return;

    Points points = _points.getMat();
    CV_Assert(points.type() == POINT_TYPE);

    _normals.createSameSize(_points, _points.type());
    Normals normals = _normals.getMat();
    Mat volUnitsData = _volUnitsData.getMat();

    const float voxelSize = settings.getVoxelSize();
    const float voxelSizeInv = 1.f / voxelSize;

    const Vec4i volDims;
    settings.getVolumeStrides(volDims);

    Matx44f _pose;
    settings.getVolumePose(_pose);
    const Affine3f pose(_pose);

    auto HashPushNormals = [&](const ptype& point, const int* position) {
        Affine3f invPose(pose.inv());
        Point3f p = fromPtype(point);
        Point3f n = nan3;
        if (!isNaN(p))
        {
            Point3f voxelPoint = invPose * p;
            n = pose.rotation() * getNormalColorHashVoxel(voxelPoint, voxelSizeInv, volumeUnitDegree,
                                                          volDims, volUnitsData, volumeUnits);
        }
        normals(position[0], position[1]) = toPtype(n);
    };
    points.forEach(HashPushNormals);
}

void fetchPointsNormalsColorsFromColorHashTsdfVolumeUnit(
    const VolumeSettings& settings, InputArray _volUnitsData, const VolumeUnitIndexes& volumeUnits,
    const int volumeUnitDegree, OutputArray _points, OutputArray _normals, OutputArray _colors)
{
    CV_TRACE_FUNCTION();

    if (!_points.needed())
        return;

    std::vector<std::vector<ptype>> pVecs, nVecs, cVecs;
    Mat volUnitsData = _volUnitsData.getMat();

    const float voxelSize = settings.getVoxelSize();
    const float voxelSizeInv = 1.f / voxelSize;

    Vec3i resolution;
    settings.getVolumeResolution(resolution);
    const Point3i volResolution = Point3i(resolution);
    const int volumeUnitResolution = volResolution.x;
    const float volumeUnitSize = voxelSize * resolution[0];

    const Vec4i volDims;
    settings.getVolumeStrides(volDims);

    Matx44f mpose;
    settings.getVolumePose(mpose);
    const Affine3f pose(mpose);

    std::vector<Vec3i> totalVolUnits;
    for (const auto& keyvalue : volumeUnits)
        totalVolUnits.push_back(keyvalue.first);
    Range fetchRange(0, (int)totalVolUnits.size());
    const int nstripes = -1;

    bool needNormals(_normals.needed());
    bool needColors(_colors.needed());
    Mutex mutex;

    //TODO: same as HashTSDF - a 0-surface should be captured instead of all non-zero voxels
    auto HashFetchPointsNormalsColorsInvoker = [&](const Range& range)
    {
        std::vector<ptype> points, normals, colors;
        for (int i = range.start; i < range.end; i++)
        {
            cv::Vec3i tsdf_idx = totalVolUnits[i];
            VolumeUnitIndexes::const_iterator it = volumeUnits.find(tsdf_idx);
            Point3f base_point = volumeUnitIdxToVolume(tsdf_idx, volumeUnitSize);
            if (it != volumeUnits.end())
            {
                std::vector<ptype> localPoints, localNormals, localColors;
                for (int x = 0; x < volumeUnitResolution; x++)
                    for (int y = 0; y < volumeUnitResolution; y++)
                        for (int z = 0; z < volumeUnitResolution; z++)
                        {
                            cv::Vec3i voxelIdx(x, y, z);
                            RGBTsdfVoxel voxel = _atColorHash(volUnitsData, voxelIdx, it->second.index,
                                                              volResolution.x, volDims);

                            // floatToTsdf(1.0) == -128
                            if (voxel.tsdf != -128 && voxel.weight != 0)
                            {
                                Point3f point = base_point + voxelCoordToVolume(voxelIdx, voxelSize);
                                localPoints.push_back(toPtype(pose * point));
                                if (needNormals)
                                {
                                    Point3f normal = getNormalColorHashVoxel(point, voxelSizeInv, volumeUnitDegree,
                                                                            volDims, volUnitsData, volumeUnits);
                                    localNormals.push_back(toPtype(pose.rotation() * normal));
                                }
                                if (needColors)
                                {
                                    Point3f c(float(voxel.r), float(voxel.g), float(voxel.b));
                                    localColors.push_back(toPtype(c));
                                }
                            }
                        }

                AutoLock al(mutex);
                pVecs.push_back(localPoints);
                nVecs.push_back(localNormals);
                cVecs.push_back(localColors);
            }
        }
    };

    parallel_for_(fetchRange, HashFetchPointsNormalsColorsInvoker, nstripes);

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
