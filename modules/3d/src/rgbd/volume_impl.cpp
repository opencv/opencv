// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html


#include <iostream>
#include "volume_impl.hpp"
#include "tsdf_functions.hpp"

namespace cv
{

Volume::Impl::Impl(VolumeSettings settings) :
    voxelSize(settings.getVoxelSize()),
    voxelSizeInv(1.0f / voxelSize),
    raycastStepFactor(settings.getRaycastStepFactor())
{
    std::cout << "Volume::Impl::Impl()" << std::endl;

    Matx44f _pose;
    settings.getPose(_pose);
    this->pose = Affine3f(_pose);
}

// TSDF

TsdfVolume::TsdfVolume(VolumeSettings settings) :
    Volume::Impl(settings)
{
    std::cout << "TsdfVolume::TsdfVolume()" << std::endl;

    this->settings = settings;

    CV_Assert(settings.getMaxWeight() < 255);
    // Unlike original code, this should work with any volume size
    // Not only when (x,y,z % 32) == 0
    Vec3i resolution;
    settings.getResolution(resolution);
    volResolution = Point3i(resolution);
    volSize = Point3f(volResolution) * voxelSize;
    truncDist = std::max(settings.getTruncDist(), 2.1f * voxelSize);

    // (xRes*yRes*zRes) array
    // Depending on zFirstMemOrder arg:
    // &elem(x, y, z) = data + x*zRes*yRes + y*zRes + z;
    // &elem(x, y, z) = data + x + y*xRes + z*xRes*yRes;
    int xdim, ydim, zdim;
    if (settings.getMaxWeight())
    {
        xdim = volResolution.z * volResolution.y;
        ydim = volResolution.z;
        zdim = 1;
    }
    else
    {
        xdim = 1;
        ydim = volResolution.x;
        zdim = volResolution.x * volResolution.y;
    }
    volDims = Vec4i(xdim, ydim, zdim);
    this->neighbourCoords = Vec8i(
        volDims.dot(Vec4i(0, 0, 0)),
        volDims.dot(Vec4i(0, 0, 1)),
        volDims.dot(Vec4i(0, 1, 0)),
        volDims.dot(Vec4i(0, 1, 1)),
        volDims.dot(Vec4i(1, 0, 0)),
        volDims.dot(Vec4i(1, 0, 1)),
        volDims.dot(Vec4i(1, 1, 0)),
        volDims.dot(Vec4i(1, 1, 1))
    );

    volume = Mat(1, volResolution.x * volResolution.y * volResolution.z, rawType<TsdfVoxel>());

    reset();
}
TsdfVolume::~TsdfVolume() {}

void TsdfVolume::integrate(OdometryFrame frame, InputArray pose)
{
    std::cout << "TsdfVolume::integrate()" << std::endl;

    CV_TRACE_FUNCTION();
    Depth depth;
    frame.getDepth(depth);
    CV_Assert(depth.type() == DEPTH_TYPE);
    CV_Assert(!depth.empty());

    Matx33f intr;
    settings.getIntrinsics(intr);
    Intr intrinsics(intr);
    Vec6f newParams((float)depth.rows, (float)depth.cols,
        intrinsics.fx, intrinsics.fy,
        intrinsics.cx, intrinsics.cy);
    if (!(frameParams == newParams))
    {
        frameParams = newParams;
        pixNorms = preCalculationPixNorm(depth.size(), intrinsics);
    }
    Matx44f cameraPose = pose.getMat();
    integrateVolumeUnit(truncDist, voxelSize, settings.getMaxWeight(), (this->pose).matrix, volResolution, volStrides, depth,
        settings.getDepthFactor(), cameraPose, intrinsics, pixNorms, volume);

}



inline float TsdfVolume::interpolateVoxel(const Point3f& p) const
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

inline Point3f TsdfVolume::getNormalVoxel(const Point3f& p) const
{
    const int xdim = volDims[0], ydim = volDims[1], zdim = volDims[2];
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

struct RaycastInvoker : ParallelLoopBody
{
    RaycastInvoker(Points& _points, Normals& _normals, const Matx44f& cameraPose,
        const Intr& intrinsics, const TsdfVolume& _volume) :
        ParallelLoopBody(),
        points(_points),
        normals(_normals),
        volume(_volume),
        tstep(volume.truncDist* volume.raycastStepFactor),
        // We do subtract voxel size to minimize checks after
        // Note: origin of volume coordinate is placed
        // in the center of voxel (0,0,0), not in the corner of the voxel!
        boxMax(volume.volSize - Point3f(volume.voxelSize,
            volume.voxelSize,
            volume.voxelSize)),
        boxMin(),
        cam2vol(volume.pose.inv()* Affine3f(cameraPose)),
        vol2cam(Affine3f(cameraPose.inv())* volume.pose),
        reproj(intrinsics.makeReprojector())
    {  }

    virtual void operator() (const Range& range) const override
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
                    orig = orig * volume.voxelSizeInv;
                    dir = dir * volume.voxelSizeInv;

                    Point3f rayStep = dir * tstep;
                    Point3f next = (orig + dir * tmin);
                    float f = volume.interpolateVoxel(next), fnext = f;

                    //raymarch
                    int steps = 0;
                    int nSteps = int(floor((tmax - tmin) / tstep));
                    for (; steps < nSteps; steps++)
                    {
                        next += rayStep;
                        int xdim = volume.volDims[0];
                        int ydim = volume.volDims[1];
                        int zdim = volume.volDims[2];
                        int ix = cvRound(next.x);
                        int iy = cvRound(next.y);
                        int iz = cvRound(next.z);
                        fnext = tsdfToFloat(volume.volume.at<TsdfVoxel>(ix * xdim + iy * ydim + iz * zdim).tsdf);
                        if (fnext != f)
                        {
                            fnext = volume.interpolateVoxel(next);
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
                        float ft = volume.interpolateVoxel(tp);
                        float ftdt = volume.interpolateVoxel(next);
                        // float t = tmin + steps*tstep;
                        // float ts = t - tstep*ft/(ftdt - ft);
                        float ts = tmin + tstep * (steps - ft / (ftdt - ft));

                        // avoid division by zero
                        if (!cvIsNaN(ts) && !cvIsInf(ts))
                        {
                            Point3f pv = (orig + dir * ts);
                            Point3f nv = volume.getNormalVoxel(pv);

                            if (!isNaN(nv))
                            {
                                //convert pv and nv to camera space
                                normal = volRot * nv;
                                // interpolation optimized a little
                                point = vol2cam * (pv * volume.voxelSize);
                            }
                        }
                    }
                }
                ptsRow[x] = toPtype(point);
                nrmRow[x] = toPtype(normal);
            }
        }
    }

    Points& points;
    Normals& normals;
    const TsdfVolume& volume;

    const float tstep;

    const Point3f boxMax;
    const Point3f boxMin;

    const Affine3f cam2vol;
    const Affine3f vol2cam;
    const Intr::Reprojector reproj;
};


void TsdfVolume::raycast(const Matx44f& cameraPose, int height, int width, OutputArray _points, OutputArray _normals) const
{
    std::cout << "TsdfVolume::raycast()" << std::endl;

    CV_TRACE_FUNCTION();
    Size frameSize(height, width);
    CV_Assert(frameSize.area() > 0);

    _points.create(frameSize, POINT_TYPE);
    _normals.create(frameSize, POINT_TYPE);

    Points points = _points.getMat();
    Normals normals = _normals.getMat();

    Matx33f intr;
    this->settings.getIntrinsics(intr);

    RaycastInvoker ri(points, normals, cameraPose, Intr(intr), *this);

    const int nstripes = -1;
    parallel_for_(Range(0, points.rows), ri, nstripes);
}

void TsdfVolume::fetchNormals() const {}
void TsdfVolume::fetchPointsNormals() const {}

void TsdfVolume::reset()
{
    CV_TRACE_FUNCTION();

    volume.forEach<VecTsdfVoxel>([](VecTsdfVoxel& vv, const int* /* position */)
        {
            TsdfVoxel& v = reinterpret_cast<TsdfVoxel&>(vv);
            v.tsdf = floatToTsdf(0.0f); v.weight = 0;
        });
}
int TsdfVolume::getVisibleBlocks() const { return 1; }
size_t TsdfVolume::getTotalVolumeUnits() const { return 1; }




// HASH_TSDF

HashTsdfVolume::HashTsdfVolume(VolumeSettings settings) :
    Volume::Impl(settings)
{ this->settings = settings; }
HashTsdfVolume::~HashTsdfVolume() {}

void HashTsdfVolume::integrate(OdometryFrame frame, InputArray pose) { std::cout << "HashTsdfVolume::integrate()" << std::endl; }
void HashTsdfVolume::raycast(const Matx44f& cameraPose, int height, int width, OutputArray _points, OutputArray _normals) const { std::cout << "HashTsdfVolume::raycast()" << std::endl; }

void HashTsdfVolume::fetchNormals() const {}
void HashTsdfVolume::fetchPointsNormals() const {}

void HashTsdfVolume::reset() {}
int HashTsdfVolume::getVisibleBlocks() const { return 1; }
size_t HashTsdfVolume::getTotalVolumeUnits() const { return 1; }

// COLOR_TSDF

ColorTsdfVolume::ColorTsdfVolume(VolumeSettings settings) :
    Volume::Impl(settings)
{ this->settings = settings; }
ColorTsdfVolume::~ColorTsdfVolume() {}

void ColorTsdfVolume::integrate(OdometryFrame frame, InputArray pose) { std::cout << "ColorTsdfVolume::integrate()" << std::endl; }
void ColorTsdfVolume::raycast(const Matx44f& cameraPose, int height, int width, OutputArray _points, OutputArray _normals) const { std::cout << "ColorTsdfVolume::raycast()" << std::endl; }

void ColorTsdfVolume::fetchNormals() const {}
void ColorTsdfVolume::fetchPointsNormals() const {}

void ColorTsdfVolume::reset() {}
int ColorTsdfVolume::getVisibleBlocks() const { return 1; }
size_t ColorTsdfVolume::getTotalVolumeUnits() const { return 1; }

}
