// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

#include "precomp.hpp"
//#include "tsdf.hpp"
#include "tsdf_functions.hpp"
#include "opencl_kernels_rgbd.hpp"

namespace cv {

namespace kinfu {

TSDFVolume::TSDFVolume(float _voxelSize, Matx44f _pose, float _raycastStepFactor, float _truncDist,
                       int _maxWeight, Point3i _resolution, bool zFirstMemOrder)
    : Volume(_voxelSize, _pose, _raycastStepFactor),
      volResolution(_resolution),
      maxWeight( WeightType(_maxWeight) )
{
    CV_Assert(_maxWeight < 255);
    // Unlike original code, this should work with any volume size
    // Not only when (x,y,z % 32) == 0
    volSize   = Point3f(volResolution) * voxelSize;
    truncDist = std::max(_truncDist, 2.1f * voxelSize);

    // (xRes*yRes*zRes) array
    // Depending on zFirstMemOrder arg:
    // &elem(x, y, z) = data + x*zRes*yRes + y*zRes + z;
    // &elem(x, y, z) = data + x + y*xRes + z*xRes*yRes;
    int xdim, ydim, zdim;
    if(zFirstMemOrder)
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
    neighbourCoords = Vec8i(
        volDims.dot(Vec4i(0, 0, 0)),
        volDims.dot(Vec4i(0, 0, 1)),
        volDims.dot(Vec4i(0, 1, 0)),
        volDims.dot(Vec4i(0, 1, 1)),
        volDims.dot(Vec4i(1, 0, 0)),
        volDims.dot(Vec4i(1, 0, 1)),
        volDims.dot(Vec4i(1, 1, 0)),
        volDims.dot(Vec4i(1, 1, 1))
    );
}

// dimension in voxels, size in meters
TSDFVolumeCPU::TSDFVolumeCPU(float _voxelSize, cv::Matx44f _pose, float _raycastStepFactor,
                             float _truncDist, int _maxWeight, Vec3i _resolution,
                             bool zFirstMemOrder)
    : TSDFVolume(_voxelSize, _pose, _raycastStepFactor, _truncDist, _maxWeight, _resolution,
                 zFirstMemOrder)
{
    int xdim, ydim, zdim;
    if (zFirstMemOrder)
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
    volStrides = Vec4i(xdim, ydim, zdim);

    volume = Mat(1, volResolution.x * volResolution.y * volResolution.z, rawType<TsdfVoxel>());

    reset();
}

// zero volume, leave rest params the same
void TSDFVolumeCPU::reset()
{
    CV_TRACE_FUNCTION();

    volume.forEach<VecTsdfVoxel>([](VecTsdfVoxel& vv, const int* /* position */)
    {
        TsdfVoxel& v = reinterpret_cast<TsdfVoxel&>(vv);
        v.tsdf = floatToTsdf(0.0f); v.weight = 0;
    });
}

TsdfVoxel TSDFVolumeCPU::at(const Vec3i& volumeIdx) const
{
    //! Out of bounds
    if ((volumeIdx[0] >= volResolution.x || volumeIdx[0] < 0) ||
        (volumeIdx[1] >= volResolution.y || volumeIdx[1] < 0) ||
        (volumeIdx[2] >= volResolution.z || volumeIdx[2] < 0))
    {
        return TsdfVoxel(floatToTsdf(1.f), 0);
    }

    const TsdfVoxel* volData = volume.ptr<TsdfVoxel>();
    int coordBase =
        volumeIdx[0] * volDims[0] + volumeIdx[1] * volDims[1] + volumeIdx[2] * volDims[2];
    return volData[coordBase];
}

// use depth instead of distance (optimization)
void TSDFVolumeCPU::integrate(InputArray _depth, float depthFactor, const Matx44f& cameraPose,
                              const Intr& intrinsics, const int frameId)
{
    CV_TRACE_FUNCTION();
    CV_UNUSED(frameId);
    CV_Assert(_depth.type() == DEPTH_TYPE);
    CV_Assert(!_depth.empty());
    Depth depth = _depth.getMat();

    Vec6f newParams((float)depth.rows, (float)depth.cols,
        intrinsics.fx, intrinsics.fy,
        intrinsics.cx, intrinsics.cy);
    if (!(frameParams == newParams))
    {
        frameParams = newParams;
        pixNorms = preCalculationPixNorm(depth, intrinsics);
    }

    integrateVolumeUnit(truncDist, voxelSize, maxWeight, (this->pose).matrix, volResolution, volStrides, depth,
        depthFactor, cameraPose, intrinsics, pixNorms, volume);
}

#if USE_INTRINSICS
// all coordinate checks should be done in inclosing cycle
inline float TSDFVolumeCPU::interpolateVoxel(const Point3f& _p) const
{
    v_float32x4 p(_p.x, _p.y, _p.z, 0);
    return interpolateVoxel(p);
}

inline float TSDFVolumeCPU::interpolateVoxel(const v_float32x4& p) const
{
    // tx, ty, tz = floor(p)
    v_int32x4 ip  = v_floor(p);
    v_float32x4 t = p - v_cvt_f32(ip);
    float tx      = t.get0();
    t             = v_reinterpret_as_f32(v_rotate_right<1>(v_reinterpret_as_u32(t)));
    float ty      = t.get0();
    t             = v_reinterpret_as_f32(v_rotate_right<1>(v_reinterpret_as_u32(t)));
    float tz      = t.get0();

    int xdim = volDims[0], ydim = volDims[1], zdim = volDims[2];
    const TsdfVoxel* volData = volume.ptr<TsdfVoxel>();

    int ix = ip.get0();
    ip     = v_rotate_right<1>(ip);
    int iy = ip.get0();
    ip     = v_rotate_right<1>(ip);
    int iz = ip.get0();

    int coordBase = ix*xdim + iy*ydim + iz*zdim;

    TsdfType vx[8];
    for(int i = 0; i < 8; i++)
        vx[i] = volData[neighbourCoords[i] + coordBase].tsdf;

    v_float32x4 v0246 = tsdfToFloat_INTR(v_int32x4(vx[0], vx[2], vx[4], vx[6]));
    v_float32x4 v1357 = tsdfToFloat_INTR(v_int32x4(vx[1], vx[3], vx[5], vx[7]));
    v_float32x4 vxx = v0246 + v_setall_f32(tz)*(v1357 - v0246);

    v_float32x4 v00_10 = vxx;
    v_float32x4 v01_11 = v_reinterpret_as_f32(v_rotate_right<1>(v_reinterpret_as_u32(vxx)));

    v_float32x4 v0_1 = v00_10 + v_setall_f32(ty)*(v01_11 - v00_10);
    float v0         = v0_1.get0();
    v0_1             = v_reinterpret_as_f32(v_rotate_right<2>(v_reinterpret_as_u32(v0_1)));
    float v1         = v0_1.get0();

    return v0 + tx*(v1 - v0);
}
#else
inline float TSDFVolumeCPU::interpolateVoxel(const Point3f& p) const
{
    int xdim = volDims[0], ydim = volDims[1], zdim = volDims[2];

    int ix = cvFloor(p.x);
    int iy = cvFloor(p.y);
    int iz = cvFloor(p.z);

    float tx = p.x - ix;
    float ty = p.y - iy;
    float tz = p.z - iz;

    int coordBase = ix*xdim + iy*ydim + iz*zdim;
    const TsdfVoxel* volData = volume.ptr<TsdfVoxel>();

    float vx[8];
    for (int i = 0; i < 8; i++)
        vx[i] = tsdfToFloat(volData[neighbourCoords[i] + coordBase].tsdf);

    float v00 = vx[0] + tz*(vx[1] - vx[0]);
    float v01 = vx[2] + tz*(vx[3] - vx[2]);
    float v10 = vx[4] + tz*(vx[5] - vx[4]);
    float v11 = vx[6] + tz*(vx[7] - vx[6]);

    float v0 = v00 + ty*(v01 - v00);
    float v1 = v10 + ty*(v11 - v10);

    return v0 + tx*(v1 - v0);

}
#endif

#if USE_INTRINSICS
//gradientDeltaFactor is fixed at 1.0 of voxel size
inline Point3f TSDFVolumeCPU::getNormalVoxel(const Point3f& _p) const
{
    v_float32x4 p(_p.x, _p.y, _p.z, 0.f);
    v_float32x4 result = getNormalVoxel(p);
    float CV_DECL_ALIGNED(16) ares[4];
    v_store_aligned(ares, result);
    return Point3f(ares[0], ares[1], ares[2]);
}

inline v_float32x4 TSDFVolumeCPU::getNormalVoxel(const v_float32x4& p) const
{
    if(v_check_any (p < v_float32x4(1.f, 1.f, 1.f, 0.f)) ||
       v_check_any (p >= v_float32x4((float)(volResolution.x-2),
                                     (float)(volResolution.y-2),
                                     (float)(volResolution.z-2), 1.f))
                   )
        return nanv;

    v_int32x4 ip  = v_floor(p);
    v_float32x4 t = p - v_cvt_f32(ip);
    float tx      = t.get0();
    t             = v_reinterpret_as_f32(v_rotate_right<1>(v_reinterpret_as_u32(t)));
    float ty      = t.get0();
    t             = v_reinterpret_as_f32(v_rotate_right<1>(v_reinterpret_as_u32(t)));
    float tz      = t.get0();

    const int xdim = volDims[0], ydim = volDims[1], zdim = volDims[2];
    const TsdfVoxel* volData = volume.ptr<TsdfVoxel>();

    int ix = ip.get0(); ip = v_rotate_right<1>(ip);
    int iy = ip.get0(); ip = v_rotate_right<1>(ip);
    int iz = ip.get0();

    int coordBase = ix*xdim + iy*ydim + iz*zdim;

    float CV_DECL_ALIGNED(16) an[4];
    an[0] = an[1] = an[2] = an[3] = 0.f;
    for(int c = 0; c < 3; c++)
    {
        const int dim = volDims[c];
        float& nv     = an[c];

        float vx[8];
        for(int i = 0; i < 8; i++)
            vx[i] = tsdfToFloat(volData[neighbourCoords[i] + coordBase + 1*dim].tsdf) -
                    tsdfToFloat(volData[neighbourCoords[i] + coordBase - 1*dim].tsdf);

        v_float32x4 v0246 (vx[0], vx[2], vx[4], vx[6]);
        v_float32x4 v1357 (vx[1], vx[3], vx[5], vx[7]);
        v_float32x4 vxx = v0246 + v_setall_f32(tz)*(v1357 - v0246);

        v_float32x4 v00_10 = vxx;
        v_float32x4 v01_11 = v_reinterpret_as_f32(v_rotate_right<1>(v_reinterpret_as_u32(vxx)));

        v_float32x4 v0_1 = v00_10 + v_setall_f32(ty)*(v01_11 - v00_10);
        float v0         = v0_1.get0();
        v0_1             = v_reinterpret_as_f32(v_rotate_right<2>(v_reinterpret_as_u32(v0_1)));
        float v1         = v0_1.get0();

        nv = v0 + tx*(v1 - v0);
    }

    v_float32x4 n       = v_load_aligned(an);
    v_float32x4 Norm = v_sqrt(v_setall_f32(v_reduce_sum(n*n)));

    return Norm.get0() < 0.0001f ? nanv : n/Norm;
}
#else
inline Point3f TSDFVolumeCPU::getNormalVoxel(const Point3f& p) const
{
    const int xdim = volDims[0], ydim = volDims[1], zdim = volDims[2];
    const TsdfVoxel* volData = volume.ptr<TsdfVoxel>();

    if(p.x < 1 || p.x >= volResolution.x - 2 ||
       p.y < 1 || p.y >= volResolution.y - 2 ||
       p.z < 1 || p.z >= volResolution.z - 2)
        return nan3;

    int ix = cvFloor(p.x);
    int iy = cvFloor(p.y);
    int iz = cvFloor(p.z);

    float tx = p.x - ix;
    float ty = p.y - iy;
    float tz = p.z - iz;

    int coordBase = ix*xdim + iy*ydim + iz*zdim;

    Vec3f an;
    for(int c = 0; c < 3; c++)
    {
        const int dim = volDims[c];
        float& nv = an[c];

        float vx[8];
        for (int i = 0; i < 8; i++)
            vx[i] = tsdfToFloat(volData[neighbourCoords[i] + coordBase + 1 * dim].tsdf) -
                    tsdfToFloat(volData[neighbourCoords[i] + coordBase - 1 * dim].tsdf);

        float v00 = vx[0] + tz*(vx[1] - vx[0]);
        float v01 = vx[2] + tz*(vx[3] - vx[2]);
        float v10 = vx[4] + tz*(vx[5] - vx[4]);
        float v11 = vx[6] + tz*(vx[7] - vx[6]);

        float v0 = v00 + ty*(v01 - v00);
        float v1 = v10 + ty*(v11 - v10);

        nv = v0 + tx*(v1 - v0);
    }

    float nv = sqrt(an[0] * an[0] +
                    an[1] * an[1] +
                    an[2] * an[2]);
    return nv < 0.0001f ? nan3 : an / nv;
}
#endif

struct RaycastInvoker : ParallelLoopBody
{
    RaycastInvoker(Points& _points, Normals& _normals, const Matx44f& cameraPose,
                  const Intr& intrinsics, const TSDFVolumeCPU& _volume) :
        ParallelLoopBody(),
        points(_points),
        normals(_normals),
        volume(_volume),
        tstep(volume.truncDist * volume.raycastStepFactor),
        // We do subtract voxel size to minimize checks after
        // Note: origin of volume coordinate is placed
        // in the center of voxel (0,0,0), not in the corner of the voxel!
        boxMax(volume.volSize - Point3f(volume.voxelSize,
                                        volume.voxelSize,
                                        volume.voxelSize)),
        boxMin(),
        cam2vol(volume.pose.inv() * Affine3f(cameraPose)),
        vol2cam(Affine3f(cameraPose.inv()) * volume.pose),
        reproj(intrinsics.makeReprojector())
    {  }

#if USE_INTRINSICS
    virtual void operator() (const Range& range) const override
    {
        const v_float32x4 vfxy(reproj.fxinv, reproj.fyinv, 0, 0);
        const v_float32x4 vcxy(reproj.cx, reproj.cy, 0, 0);

        const float (&cm)[16] = cam2vol.matrix.val;
        const v_float32x4 camRot0(cm[0], cm[4], cm[8], 0);
        const v_float32x4 camRot1(cm[1], cm[5], cm[9], 0);
        const v_float32x4 camRot2(cm[2], cm[6], cm[10], 0);
        const v_float32x4 camTrans(cm[3], cm[7], cm[11], 0);

        const v_float32x4 boxDown(boxMin.x, boxMin.y, boxMin.z, 0.f);
        const v_float32x4 boxUp(boxMax.x, boxMax.y, boxMax.z, 0.f);

        const v_float32x4 invVoxelSize = v_float32x4(volume.voxelSizeInv,
                                                     volume.voxelSizeInv,
                                                     volume.voxelSizeInv, 1.f);

        const float (&vm)[16] = vol2cam.matrix.val;
        const v_float32x4 volRot0(vm[0], vm[4], vm[8], 0);
        const v_float32x4 volRot1(vm[1], vm[5], vm[9], 0);
        const v_float32x4 volRot2(vm[2], vm[6], vm[10], 0);
        const v_float32x4 volTrans(vm[3], vm[7], vm[11], 0);

        for(int y = range.start; y < range.end; y++)
        {
            ptype* ptsRow = points[y];
            ptype* nrmRow = normals[y];

            for(int x = 0; x < points.cols; x++)
            {
                v_float32x4 point = nanv, normal = nanv;

                v_float32x4 orig = camTrans;

                // get direction through pixel in volume space:

                // 1. reproject (x, y) on projecting plane where z = 1.f
                v_float32x4 planed = (v_float32x4((float)x, (float)y, 0.f, 0.f) - vcxy)*vfxy;
                planed             = v_combine_low(planed, v_float32x4(1.f, 0.f, 0.f, 0.f));

                // 2. rotate to volume space
                planed = v_matmuladd(planed, camRot0, camRot1, camRot2, v_setzero_f32());

                // 3. normalize
                v_float32x4 invNorm = v_invsqrt(v_setall_f32(v_reduce_sum(planed*planed)));
                v_float32x4 dir     = planed*invNorm;

                // compute intersection of ray with all six bbox planes
                v_float32x4 rayinv = v_setall_f32(1.f)/dir;
                // div by zero should be eliminated by these products
                v_float32x4 tbottom = rayinv*(boxDown - orig);
                v_float32x4 ttop    = rayinv*(boxUp   - orig);

                // re-order intersections to find smallest and largest on each axis
                v_float32x4 minAx = v_min(ttop, tbottom);
                v_float32x4 maxAx = v_max(ttop, tbottom);

                // near clipping plane
                const float clip = 0.f;
                float _minAx[4], _maxAx[4];
                v_store(_minAx, minAx);
                v_store(_maxAx, maxAx);
                float tmin       = max( {_minAx[0], _minAx[1], _minAx[2], clip} );
                float tmax       = min( {_maxAx[0], _maxAx[1], _maxAx[2]} );

                // precautions against getting coordinates out of bounds
                tmin = tmin + tstep;
                tmax = tmax - tstep;

                if(tmin < tmax)
                {
                    // interpolation optimized a little
                    orig *= invVoxelSize;
                    dir  *= invVoxelSize;

                    int xdim            = volume.volDims[0];
                    int ydim            = volume.volDims[1];
                    int zdim            = volume.volDims[2];
                    v_float32x4 rayStep = dir * v_setall_f32(tstep);
                    v_float32x4 next    = (orig + dir * v_setall_f32(tmin));
                    float f = volume.interpolateVoxel(next), fnext = f;

                    //raymarch
                    int steps = 0;
                    int nSteps = cvFloor((tmax - tmin)/tstep);
                    for(; steps < nSteps; steps++)
                    {
                        next += rayStep;
                        v_int32x4 ip = v_round(next);
                        int ix = ip.get0(); ip = v_rotate_right<1>(ip);
                        int iy = ip.get0(); ip = v_rotate_right<1>(ip);
                        int iz = ip.get0();
                        int coord = ix*xdim + iy*ydim + iz*zdim;

                        fnext = tsdfToFloat(volume.volume.at<TsdfVoxel>(coord).tsdf);
                        if(fnext != f)
                        {
                            fnext = volume.interpolateVoxel(next);

                            // when ray crosses a surface
                            if(std::signbit(f) != std::signbit(fnext))
                                break;

                            f = fnext;
                        }
                    }

                    // if ray penetrates a surface from outside
                    // linearly interpolate t between two f values
                    if(f > 0.f && fnext < 0.f)
                    {
                        v_float32x4 tp = next - rayStep;
                        float ft    = volume.interpolateVoxel(tp);
                        float ftdt  = volume.interpolateVoxel(next);
                        float ts = tmin + tstep*(steps - ft/(ftdt - ft));

                        // avoid division by zero
                        if(!cvIsNaN(ts) && !cvIsInf(ts))
                        {
                            v_float32x4 pv = (orig + dir*v_setall_f32(ts));
                            v_float32x4 nv = volume.getNormalVoxel(pv);

                            if(!isNaN(nv))
                            {
                                //convert pv and nv to camera space
                                normal = v_matmuladd(nv, volRot0, volRot1, volRot2, v_setzero_f32());
                                // interpolation optimized a little
                                point = v_matmuladd(pv*v_float32x4(volume.voxelSize,
                                                                   volume.voxelSize,
                                                                   volume.voxelSize, 1.f),
                                                    volRot0, volRot1, volRot2, volTrans);
                            }
                        }
                    }
                }

                v_store((float*)(&ptsRow[x]), point);
                v_store((float*)(&nrmRow[x]), normal);
            }
        }
    }
#else
    virtual void operator() (const Range& range) const override
    {
        const Point3f camTrans = cam2vol.translation();
        const Matx33f  camRot  = cam2vol.rotation();
        const Matx33f  volRot  = vol2cam.rotation();

        for(int y = range.start; y < range.end; y++)
        {
            ptype* ptsRow = points[y];
            ptype* nrmRow = normals[y];

            for(int x = 0; x < points.cols; x++)
            {
                Point3f point = nan3, normal = nan3;

                Point3f orig = camTrans;
                // direction through pixel in volume space
                Point3f dir = normalize(Vec3f(camRot * reproj(Point3f(float(x), float(y), 1.f))));

                // compute intersection of ray with all six bbox planes
                Vec3f rayinv(1.f/dir.x, 1.f/dir.y, 1.f/dir.z);
                Point3f tbottom = rayinv.mul(boxMin - orig);
                Point3f ttop    = rayinv.mul(boxMax - orig);

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

                if(tmin < tmax)
                {
                    // interpolation optimized a little
                    orig = orig*volume.voxelSizeInv;
                    dir  =  dir*volume.voxelSizeInv;

                    Point3f rayStep = dir * tstep;
                    Point3f next = (orig + dir * tmin);
                    float f = volume.interpolateVoxel(next), fnext = f;

                    //raymarch
                    int steps = 0;
                    int nSteps = int(floor((tmax - tmin)/tstep));
                    for(; steps < nSteps; steps++)
                    {
                        next += rayStep;
                        int xdim = volume.volDims[0];
                        int ydim = volume.volDims[1];
                        int zdim = volume.volDims[2];
                        int ix = cvRound(next.x);
                        int iy = cvRound(next.y);
                        int iz = cvRound(next.z);
                        fnext = tsdfToFloat(volume.volume.at<TsdfVoxel>(ix*xdim + iy*ydim + iz*zdim).tsdf);
                        if(fnext != f)
                        {
                            fnext = volume.interpolateVoxel(next);
                            // when ray crosses a surface
                            if(std::signbit(f) != std::signbit(fnext))
                                break;

                            f = fnext;
                        }
                    }
                    // if ray penetrates a surface from outside
                    // linearly interpolate t between two f values
                    if(f > 0.f && fnext < 0.f)
                    {
                        Point3f tp    = next - rayStep;
                        float ft   = volume.interpolateVoxel(tp);
                        float ftdt = volume.interpolateVoxel(next);
                        // float t = tmin + steps*tstep;
                        // float ts = t - tstep*ft/(ftdt - ft);
                        float ts = tmin + tstep*(steps - ft/(ftdt - ft));

                        // avoid division by zero
                        if(!cvIsNaN(ts) && !cvIsInf(ts))
                        {
                            Point3f pv = (orig + dir*ts);
                            Point3f nv = volume.getNormalVoxel(pv);

                            if(!isNaN(nv))
                            {
                                //convert pv and nv to camera space
                                normal = volRot * nv;
                                // interpolation optimized a little
                                point = vol2cam * (pv*volume.voxelSize);
                            }
                        }
                    }
                }
                ptsRow[x] = toPtype(point);
                nrmRow[x] = toPtype(normal);
            }
        }
    }
#endif

    Points& points;
    Normals& normals;
    const TSDFVolumeCPU& volume;

    const float tstep;

    const Point3f boxMax;
    const Point3f boxMin;

    const Affine3f cam2vol;
    const Affine3f vol2cam;
    const Intr::Reprojector reproj;
};


void TSDFVolumeCPU::raycast(const Matx44f& cameraPose, const Intr& intrinsics, const Size& frameSize,
                            OutputArray _points, OutputArray _normals) const
{
    CV_TRACE_FUNCTION();

    CV_Assert(frameSize.area() > 0);

    _points.create (frameSize, POINT_TYPE);
    _normals.create(frameSize, POINT_TYPE);

    Points points   =  _points.getMat();
    Normals normals = _normals.getMat();

    RaycastInvoker ri(points, normals, cameraPose, intrinsics, *this);

    const int nstripes = -1;
    parallel_for_(Range(0, points.rows), ri, nstripes);
}


struct FetchPointsNormalsInvoker : ParallelLoopBody
{
    FetchPointsNormalsInvoker(const TSDFVolumeCPU& _volume,
                              std::vector<std::vector<ptype>>& _pVecs,
                              std::vector<std::vector<ptype>>& _nVecs,
                              bool _needNormals) :
        ParallelLoopBody(),
        vol(_volume),
        pVecs(_pVecs),
        nVecs(_nVecs),
        needNormals(_needNormals)
    {
        volDataStart = vol.volume.ptr<TsdfVoxel>();
    }

    inline void coord(std::vector<ptype>& points, std::vector<ptype>& normals,
                      int x, int y, int z, Point3f V, float v0, int axis) const
    {
        // 0 for x, 1 for y, 2 for z
        bool limits = false;
        Point3i shift;
        float Vc = 0.f;
        if(axis == 0)
        {
            shift  = Point3i(1, 0, 0);
            limits = (x + 1 < vol.volResolution.x);
            Vc     = V.x;
        }
        if(axis == 1)
        {
            shift  = Point3i(0, 1, 0);
            limits = (y + 1 < vol.volResolution.y);
            Vc     = V.y;
        }
        if(axis == 2)
        {
            shift  = Point3i(0, 0, 1);
            limits = (z + 1 < vol.volResolution.z);
            Vc     = V.z;
        }

        if(limits)
        {
            const TsdfVoxel& voxeld = volDataStart[(x+shift.x)*vol.volDims[0] +
                                                   (y+shift.y)*vol.volDims[1] +
                                                   (z+shift.z)*vol.volDims[2]];
            float vd = tsdfToFloat(voxeld.tsdf);
            if(voxeld.weight != 0 && vd != 1.f)
            {
                if((v0 > 0 && vd < 0) || (v0 < 0 && vd > 0))
                {
                    //linearly interpolate coordinate
                    float Vn    = Vc + vol.voxelSize;
                    float dinv  = 1.f/(abs(v0)+abs(vd));
                    float inter = (Vc*abs(vd) + Vn*abs(v0))*dinv;

                    Point3f p(shift.x ? inter : V.x,
                              shift.y ? inter : V.y,
                              shift.z ? inter : V.z);
                    {
                        points.push_back(toPtype(vol.pose * p));
                        if(needNormals)
                            normals.push_back(toPtype(vol.pose.rotation() *
                                                      vol.getNormalVoxel(p*vol.voxelSizeInv)));
                    }
                }
            }
        }
    }

    virtual void operator() (const Range& range) const override
    {
        std::vector<ptype> points, normals;
        for(int x = range.start; x < range.end; x++)
        {
            const TsdfVoxel* volDataX = volDataStart + x*vol.volDims[0];
            for(int y = 0; y < vol.volResolution.y; y++)
            {
                const TsdfVoxel* volDataY = volDataX + y*vol.volDims[1];
                for(int z = 0; z < vol.volResolution.z; z++)
                {
                    const TsdfVoxel& voxel0 = volDataY[z*vol.volDims[2]];
                    float v0             = tsdfToFloat(voxel0.tsdf);
                    if(voxel0.weight != 0 && v0 != 1.f)
                    {
                        Point3f V(Point3f((float)x + 0.5f, (float)y + 0.5f, (float)z + 0.5f)*vol.voxelSize);

                        coord(points, normals, x, y, z, V, v0, 0);
                        coord(points, normals, x, y, z, V, v0, 1);
                        coord(points, normals, x, y, z, V, v0, 2);

                    } // if voxel is not empty
                }
            }
        }

        AutoLock al(mutex);
        pVecs.push_back(points);
        nVecs.push_back(normals);
    }

    const TSDFVolumeCPU& vol;
    std::vector<std::vector<ptype>>& pVecs;
    std::vector<std::vector<ptype>>& nVecs;
    const TsdfVoxel* volDataStart;
    bool needNormals;
    mutable Mutex mutex;
};

void TSDFVolumeCPU::fetchPointsNormals(OutputArray _points, OutputArray _normals) const
{
    CV_TRACE_FUNCTION();

    if(_points.needed())
    {
        std::vector<std::vector<ptype>> pVecs, nVecs;
        FetchPointsNormalsInvoker fi(*this, pVecs, nVecs, _normals.needed());
        Range range(0, volResolution.x);
        const int nstripes = -1;
        parallel_for_(range, fi, nstripes);

        std::vector<ptype> points, normals;
        for(size_t i = 0; i < pVecs.size(); i++)
        {
            points.insert(points.end(), pVecs[i].begin(), pVecs[i].end());
            normals.insert(normals.end(), nVecs[i].begin(), nVecs[i].end());
        }

        _points.create((int)points.size(), 1, POINT_TYPE);
        if(!points.empty())
            Mat((int)points.size(), 1, POINT_TYPE, &points[0]).copyTo(_points.getMat());

        if(_normals.needed())
        {
            _normals.create((int)normals.size(), 1, POINT_TYPE);
            if(!normals.empty())
                Mat((int)normals.size(), 1, POINT_TYPE, &normals[0]).copyTo(_normals.getMat());
        }
    }
}

void TSDFVolumeCPU::fetchNormals(InputArray _points, OutputArray _normals) const
{
    CV_TRACE_FUNCTION();
    CV_Assert(!_points.empty());
    if(_normals.needed())
    {
        Points points = _points.getMat();
        CV_Assert(points.type() == POINT_TYPE);

        _normals.createSameSize(_points, _points.type());
        Normals normals = _normals.getMat();

        const TSDFVolumeCPU& _vol = *this;
        auto PushNormals = [&](const ptype& pp, const int* position)
        {
            const TSDFVolumeCPU& vol(_vol);
            Affine3f invPose(vol.pose.inv());
            Point3f p = fromPtype(pp);
            Point3f n = nan3;
            if (!isNaN(p))
            {
                Point3f voxPt = (invPose * p);
                voxPt = voxPt * vol.voxelSizeInv;
                n = vol.pose.rotation() * vol.getNormalVoxel(voxPt);
            }
            normals(position[0], position[1]) = toPtype(n);
        };
        points.forEach(PushNormals);
    }
}

///////// GPU implementation /////////

#ifdef HAVE_OPENCL
TSDFVolumeGPU::TSDFVolumeGPU(float _voxelSize, Matx44f _pose, float _raycastStepFactor, float _truncDist, int _maxWeight,
                             Point3i _resolution) :
    TSDFVolume(_voxelSize, _pose, _raycastStepFactor, _truncDist, _maxWeight, _resolution, false)
{
    volume = UMat(1, volResolution.x * volResolution.y * volResolution.z, CV_8UC2);

    reset();
}


// zero volume, leave rest params the same
void TSDFVolumeGPU::reset()
{
    CV_TRACE_FUNCTION();

    volume.setTo(Scalar(0, 0));
}

// use depth instead of distance (optimization)
void TSDFVolumeGPU::integrate(InputArray _depth, float depthFactor,
                              const Matx44f& cameraPose, const Intr& intrinsics, const int frameId)
{
    CV_TRACE_FUNCTION();
    CV_UNUSED(frameId);
    CV_Assert(!_depth.empty());

    UMat depth = _depth.getUMat();

    String errorStr;
    String name           = "integrate";
    ocl::ProgramSource source = ocl::rgbd::tsdf_oclsrc;
    String options        = "-cl-mad-enable";
    ocl::Kernel k;
    k.create(name.c_str(), source, options, &errorStr);

    if(k.empty())
        throw std::runtime_error("Failed to create kernel: " + errorStr);

    Affine3f vol2cam(Affine3f(cameraPose.inv()) * pose);
    float dfac = 1.f/depthFactor;
    Vec4i volResGpu(volResolution.x, volResolution.y, volResolution.z);
    Vec2f fxy(intrinsics.fx, intrinsics.fy), cxy(intrinsics.cx, intrinsics.cy);
    Vec6f newParams((float)depth.rows, (float)depth.cols,
        intrinsics.fx, intrinsics.fy,
        intrinsics.cx, intrinsics.cy);
    if (!(frameParams == newParams))
    {
        frameParams = newParams;
        pixNorms = preCalculationPixNormGPU(depth, intrinsics);
    }

    // TODO: optimization possible
    // Use sampler for depth (mask needed)
    k.args(ocl::KernelArg::ReadOnly(depth),
           ocl::KernelArg::PtrReadWrite(volume),
           ocl::KernelArg::Constant(vol2cam.matrix.val,
                                    sizeof(vol2cam.matrix.val)),
           voxelSize,
           volResGpu.val,
           volDims.val,
           fxy.val,
           cxy.val,
           dfac,
           truncDist,
           int(maxWeight),
           ocl::KernelArg::PtrReadOnly(pixNorms));

    size_t globalSize[2];
    globalSize[0] = (size_t)volResolution.x;
    globalSize[1] = (size_t)volResolution.y;

    if(!k.run(2, globalSize, NULL, true))
        throw std::runtime_error("Failed to run kernel");
}


void TSDFVolumeGPU::raycast(const Matx44f& cameraPose, const Intr& intrinsics, const Size& frameSize,
                            OutputArray _points, OutputArray _normals) const
{
    CV_TRACE_FUNCTION();

    CV_Assert(frameSize.area() > 0);

    String errorStr;
    String name           = "raycast";
    ocl::ProgramSource source = ocl::rgbd::tsdf_oclsrc;
    String options        = "-cl-mad-enable";
    ocl::Kernel k;
    k.create(name.c_str(), source, options, &errorStr);

    if(k.empty())
        throw std::runtime_error("Failed to create kernel: " + errorStr);

    _points.create (frameSize, CV_32FC4);
    _normals.create(frameSize, CV_32FC4);

    UMat points  =  _points.getUMat();
    UMat normals = _normals.getUMat();

    UMat vol2camGpu, cam2volGpu;
    Affine3f vol2cam = Affine3f(cameraPose.inv()) * pose;
    Affine3f cam2vol = pose.inv() * Affine3f(cameraPose);
    Mat(cam2vol.matrix).copyTo(cam2volGpu);
    Mat(vol2cam.matrix).copyTo(vol2camGpu);
    Intr::Reprojector r = intrinsics.makeReprojector();
    // We do subtract voxel size to minimize checks after
    // Note: origin of volume coordinate is placed
    // in the center of voxel (0,0,0), not in the corner of the voxel!
    Vec4f boxMin, boxMax(volSize.x - voxelSize,
                         volSize.y - voxelSize,
                         volSize.z - voxelSize);
    Vec2f finv(r.fxinv, r.fyinv), cxy(r.cx, r.cy);
    float tstep = truncDist * raycastStepFactor;

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

    if(!k.run(2, globalSize, NULL, true))
        throw std::runtime_error("Failed to run kernel");
}


void TSDFVolumeGPU::fetchNormals(InputArray _points, OutputArray _normals) const
{
    CV_TRACE_FUNCTION();
    CV_Assert(!_points.empty());

    if(_normals.needed())
    {
        UMat points = _points.getUMat();
        CV_Assert(points.type() == POINT_TYPE);

        _normals.createSameSize(_points, POINT_TYPE);
        UMat normals = _normals.getUMat();

        String errorStr;
        String name           = "getNormals";
        ocl::ProgramSource source = ocl::rgbd::tsdf_oclsrc;
        String options        = "-cl-mad-enable";
        ocl::Kernel k;
        k.create(name.c_str(), source, options, &errorStr);

        if(k.empty())
            throw std::runtime_error("Failed to create kernel: " + errorStr);

        UMat volPoseGpu, invPoseGpu;
        Mat(pose      .matrix).copyTo(volPoseGpu);
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

        if(!k.run(2, globalSize, NULL, true))
            throw std::runtime_error("Failed to run kernel");
    }
}

void TSDFVolumeGPU::fetchPointsNormals(OutputArray points, OutputArray normals) const
{
    CV_TRACE_FUNCTION();

    if(points.needed())
    {
        bool needNormals = normals.needed();

        // 1. scan to count points in each group and allocate output arrays

        ocl::Kernel kscan;

        String errorStr;
        ocl::ProgramSource source = ocl::rgbd::tsdf_oclsrc;
        String options        = "-cl-mad-enable";

        kscan.create("scanSize", source, options, &errorStr);

        if(kscan.empty())
            throw std::runtime_error("Failed to create kernel: " + errorStr);

        size_t globalSize[3];
        globalSize[0] = (size_t)volResolution.x;
        globalSize[1] = (size_t)volResolution.y;
        globalSize[2] = (size_t)volResolution.z;

        const ocl::Device& device = ocl::Device::getDefault();
        size_t wgsLimit           = device.maxWorkGroupSize();
        size_t memSize            = device.localMemSize();
        // local mem should keep a point (and a normal) for each thread in a group
        // use 4 float per each point and normal
        size_t elemSize     = (sizeof(float)*4)*(needNormals ? 2 : 1);
        const size_t lcols  = 8;
        const size_t lrows  = 8;
        size_t lplanes      = min(memSize/elemSize, wgsLimit)/lcols/lrows;
        lplanes             = roundDownPow2(lplanes);
        size_t localSize[3] = {lcols, lrows, lplanes};
        Vec3i ngroups((int)divUp(globalSize[0], (unsigned int)localSize[0]),
                      (int)divUp(globalSize[1], (unsigned int)localSize[1]),
                      (int)divUp(globalSize[2], (unsigned int)localSize[2]));

        const size_t counterSize = sizeof(int);
        size_t lszscan = localSize[0]*localSize[1]*localSize[2]*counterSize;

        const int gsz[3] = {ngroups[2], ngroups[1], ngroups[0]};
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

        if(!kscan.run(3, globalSize, localSize, true))
            throw std::runtime_error("Failed to run kernel");

        Mat groupedSumCpu = groupedSum.getMat(ACCESS_READ);
        int gpuSum        = (int)sum(groupedSumCpu)[0];
        // should be no CPU copies when new kernel is executing
        groupedSumCpu.release();

        // 2. fill output arrays according to per-group points count

        points.create(gpuSum, 1, POINT_TYPE);
        UMat pts = points.getUMat();
        UMat nrm;
        if(needNormals)
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

            if(kfill.empty())
                throw std::runtime_error("Failed to create kernel: " + errorStr);

            UMat atomicCtr(1, 1, CV_32S, Scalar(0));

            // mem size to keep pts (and normals optionally) for all work-items in a group
            size_t lszfill = localSize[0]*localSize[1]*localSize[2]*elemSize;

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

            if(!kfill.run(3, globalSize, localSize, true))
                throw std::runtime_error("Failed to run kernel");
        }
    }
}

#endif

Ptr<TSDFVolume> makeTSDFVolume(float _voxelSize, Matx44f _pose, float _raycastStepFactor,
                                   float _truncDist, int _maxWeight, Point3i _resolution)
{
#ifdef HAVE_OPENCL
    if (ocl::useOpenCL())
        return makePtr<TSDFVolumeGPU>(_voxelSize, _pose, _raycastStepFactor, _truncDist, _maxWeight,
                                      _resolution);
#endif
    return makePtr<TSDFVolumeCPU>(_voxelSize, _pose, _raycastStepFactor, _truncDist, _maxWeight,
                                  _resolution);
}

Ptr<TSDFVolume> makeTSDFVolume(const VolumeParams& _params)
{
#ifdef HAVE_OPENCL
    if (ocl::useOpenCL())
        return makePtr<TSDFVolumeGPU>(_params.voxelSize, _params.pose.matrix, _params.raycastStepFactor,
                                      _params.tsdfTruncDist, _params.maxWeight, _params.resolution);
#endif
    return makePtr<TSDFVolumeCPU>(_params.voxelSize, _params.pose.matrix, _params.raycastStepFactor,
                                  _params.tsdfTruncDist, _params.maxWeight, _params.resolution);

}

} // namespace kinfu
} // namespace cv
