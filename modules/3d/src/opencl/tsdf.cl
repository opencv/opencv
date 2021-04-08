// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

typedef char int8_t;
typedef int8_t TsdfType;
typedef uchar WeightType;

struct TsdfVoxel
{
    TsdfType tsdf;
    WeightType weight;
};

static inline TsdfType floatToTsdf(float num)
{
    int8_t res = (int8_t) ( (num * (-128)) );
    res = res ? res : (num < 0 ? 1 : -1);
    return res;
}

static inline float tsdfToFloat(TsdfType num)
{
    return ( (float) num ) / (-128);
}

__kernel void integrate(__global const char * depthptr,
                        int depth_step, int depth_offset,
                        int depth_rows, int depth_cols,
                        __global struct TsdfVoxel * volumeptr,
                        const float16 vol2camMatrix,
                        const float voxelSize,
                        const int4 volResolution4,
                        const int4 volDims4,
                        const float2 fxy,
                        const float2 cxy,
                        const float dfac,
                        const float truncDist,
                        const int maxWeight,
                        const __global float * pixNorms)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    const int3 volResolution = volResolution4.xyz;

    if(x >= volResolution.x || y >= volResolution.y)
        return;

    // coord-independent constants
    const int3 volDims = volDims4.xyz;
    const float2 limits = (float2)(depth_cols-1, depth_rows-1);

    const float4 vol2cam0 = vol2camMatrix.s0123;
    const float4 vol2cam1 = vol2camMatrix.s4567;
    const float4 vol2cam2 = vol2camMatrix.s89ab;

    const float truncDistInv = 1.f/truncDist;

    // optimization of camSpace transformation (vector addition instead of matmul at each z)
    float4 inPt = (float4)(x*voxelSize, y*voxelSize, 0, 1);
    float3 basePt = (float3)(dot(vol2cam0, inPt),
                             dot(vol2cam1, inPt),
                             dot(vol2cam2, inPt));

    float3 camSpacePt = basePt;

    // zStep == vol2cam*(float3(x, y, 1)*voxelSize) - basePt;
    float3 zStep = ((float3)(vol2cam0.z, vol2cam1.z, vol2cam2.z))*voxelSize;

    int volYidx = x*volDims.x + y*volDims.y;

    int startZ, endZ;
    if(fabs(zStep.z) > 1e-5f)
    {
        int baseZ = convert_int(-basePt.z / zStep.z);
        if(zStep.z > 0)
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
        if(basePt.z > 0)
        {
            startZ = 0; endZ = volResolution.z;
        }
        else
        {
            // z loop shouldn't be performed
            //startZ = endZ = 0;
            return;
        }
    }

    startZ = max(0, startZ);
    endZ = min(volResolution.z, endZ);

    for(int z = startZ; z < endZ; z++)
    {
        // optimization of the following:
        //float3 camSpacePt = vol2cam * ((float3)(x, y, z)*voxelSize);
        camSpacePt += zStep;

        if(camSpacePt.z <= 0)
            continue;

        float3 camPixVec = camSpacePt / camSpacePt.z;
        float2 projected = mad(camPixVec.xy, fxy, cxy);

        float v;
        // bilinearly interpolate depth at projected
        if(all(projected >= 0) && all(projected < limits))
        {
            float2 ip = floor(projected);
            int xi = ip.x, yi = ip.y;

            __global const float* row0 = (__global const float*)(depthptr + depth_offset +
                                                                 (yi+0)*depth_step);
            __global const float* row1 = (__global const float*)(depthptr + depth_offset +
                                                                 (yi+1)*depth_step);

            float v00 = row0[xi+0];
            float v01 = row0[xi+1];
            float v10 = row1[xi+0];
            float v11 = row1[xi+1];
            float4 vv = (float4)(v00, v01, v10, v11);

            // assume correct depth is positive
            if(all(vv > 0))
            {
                float2 t = projected - ip;
                float2 vf = mix(vv.xz, vv.yw, t.x);
                v = mix(vf.s0, vf.s1, t.y);
            }
            else
                continue;
        }
        else
            continue;

        if(v == 0)
            continue;

        int idx = projected.y * depth_cols + projected.x;
        float pixNorm = pixNorms[idx];
        //float pixNorm = length(camPixVec);

        // difference between distances of point and of surface to camera
        float sdf = pixNorm*(v*dfac - camSpacePt.z);
        // possible alternative is:
        // float sdf = length(camSpacePt)*(v*dfac/camSpacePt.z - 1.0);

        if(sdf >= -truncDist)
        {
            float tsdf = fmin(1.0f, sdf * truncDistInv);
            int volIdx = volYidx + z*volDims.z;

            struct TsdfVoxel voxel = volumeptr[volIdx];
            float value  = tsdfToFloat(voxel.tsdf);
            int weight = voxel.weight;

            // update TSDF
            value = (value*weight + tsdf) / (weight + 1);
            weight = min(weight + 1, maxWeight);

            voxel.tsdf = floatToTsdf(value);
            voxel.weight = weight;
            volumeptr[volIdx] = voxel;
        }
    }
}


inline float interpolateVoxel(float3 p, __global const struct TsdfVoxel* volumePtr,
                              int3 volDims, int8 neighbourCoords)
{
    float3 fip = floor(p);
    int3 ip = convert_int3(fip);
    float3 t = p - fip;

    int3 cmul = volDims*ip;
    int coordBase = cmul.x + cmul.y + cmul.z;
    int nco[8];
    vstore8(neighbourCoords + coordBase, 0, nco);

    float vaz[8];
    for(int i = 0; i < 8; i++)
        vaz[i] = tsdfToFloat(volumePtr[nco[i]].tsdf);

    float8 vz = vload8(0, vaz);

    float4 vy = mix(vz.s0246, vz.s1357, t.z);
    float2 vx = mix(vy.s02, vy.s13, t.y);
    return mix(vx.s0, vx.s1, t.x);
}

inline float3 getNormalVoxel(float3 p, __global const struct TsdfVoxel* volumePtr,
                             int3 volResolution, int3 volDims, int8 neighbourCoords)
{
    if(any(p < 1) || any(p >= convert_float3(volResolution - 2)))
        return nan((uint)0);

    float3 fip = floor(p);
    int3 ip = convert_int3(fip);
    float3 t = p - fip;

    int3 cmul = volDims*ip;
    int coordBase = cmul.x + cmul.y + cmul.z;
    int nco[8];
    vstore8(neighbourCoords + coordBase, 0, nco);

    int arDims[3];
    vstore3(volDims, 0, arDims);
    float an[3];
    for(int c = 0; c < 3; c++)
    {
        int dim = arDims[c];

        float vaz[8];
        for(int i = 0; i < 8; i++)
            vaz[i] = tsdfToFloat(volumePtr[nco[i] + dim].tsdf) -
                     tsdfToFloat(volumePtr[nco[i] - dim].tsdf);

        float8 vz = vload8(0, vaz);

        float4 vy = mix(vz.s0246, vz.s1357, t.z);
        float2 vx = mix(vy.s02, vy.s13, t.y);

        an[c] = mix(vx.s0, vx.s1, t.x);
    }

    //gradientDeltaFactor is fixed at 1.0 of voxel size
    float3 n = vload3(0, an);
    float Norm = sqrt(n.x*n.x + n.y*n.y + n.z*n.z);
    return Norm < 0.0001f ? nan((uint)0) : n / Norm;
    //return fast_normalize(vload3(0, an));
}

typedef float4 ptype;

__kernel void raycast(__global char * pointsptr,
                      int points_step, int points_offset,
                      __global char * normalsptr,
                      int normals_step, int normals_offset,
                      const int2 frameSize,
                      __global const struct TsdfVoxel * volumeptr,
                      __global const float * vol2camptr,
                      __global const float * cam2volptr,
                      const float2 fixy,
                      const float2 cxy,
                      const float4 boxDown4,
                      const float4 boxUp4,
                      const float tstep,
                      const float voxelSize,
                      const int4 volResolution4,
                      const int4 volDims4,
                      const int8 neighbourCoords
                      )
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x >= frameSize.x || y >= frameSize.y)
        return;

    // coordinate-independent constants

    __global const float* cm = cam2volptr;
    const float3 camRot0  = vload4(0, cm).xyz;
    const float3 camRot1  = vload4(1, cm).xyz;
    const float3 camRot2  = vload4(2, cm).xyz;
    const float3 camTrans = (float3)(cm[3], cm[7], cm[11]);

    __global const float* vm = vol2camptr;
    const float3 volRot0  = vload4(0, vm).xyz;
    const float3 volRot1  = vload4(1, vm).xyz;
    const float3 volRot2  = vload4(2, vm).xyz;
    const float3 volTrans = (float3)(vm[3], vm[7], vm[11]);

    const float3 boxDown = boxDown4.xyz;
    const float3 boxUp   = boxUp4.xyz;
    const int3   volDims = volDims4.xyz;

    const int3 volResolution = volResolution4.xyz;

    const float invVoxelSize = native_recip(voxelSize);

    // kernel itself

    float3 point  = nan((uint)0);
    float3 normal = nan((uint)0);

    float3 orig = camTrans;

    // get direction through pixel in volume space:
    // 1. reproject (x, y) on projecting plane where z = 1.f
    float3 planed = (float3)(((float2)(x, y) - cxy)*fixy, 1.f);

    // 2. rotate to volume space
    planed = (float3)(dot(planed, camRot0),
                      dot(planed, camRot1),
                      dot(planed, camRot2));

    // 3. normalize
    float3 dir = fast_normalize(planed);

    // compute intersection of ray with all six bbox planes
    float3 rayinv = native_recip(dir);
    float3 tbottom = rayinv*(boxDown - orig);
    float3 ttop    = rayinv*(boxUp   - orig);

    // re-order intersections to find smallest and largest on each axis
    float3 minAx = min(ttop, tbottom);
    float3 maxAx = max(ttop, tbottom);

    // near clipping plane
    const float clip = 0.f;
    float tmin = max(max(max(minAx.x, minAx.y), max(minAx.x, minAx.z)), clip);
    float tmax =     min(min(maxAx.x, maxAx.y), min(maxAx.x, maxAx.z));

    // precautions against getting coordinates out of bounds
    tmin = tmin + tstep;
    tmax = tmax - tstep;

    if(tmin < tmax)
    {
        // interpolation optimized a little
        orig *= invVoxelSize;
        dir  *= invVoxelSize;

        float3 rayStep = dir*tstep;
        float3 next = (orig + dir*tmin);
        float f = interpolateVoxel(next, volumeptr, volDims, neighbourCoords);
        float fnext = f;

        // raymarch
        int steps = 0;
        int nSteps = floor(native_divide(tmax - tmin, tstep));
        bool stop = false;
        for(int i = 0; i < nSteps; i++)
        {
            // fix for wrong steps counting
            if(!stop)
            {
                next += rayStep;

                // fetch voxel
                int3 ip = convert_int3(round(next));
                int3 cmul = ip*volDims;
                int idx = cmul.x + cmul.y + cmul.z;
                fnext = tsdfToFloat(volumeptr[idx].tsdf);

                if(fnext != f)
                {
                    fnext = interpolateVoxel(next, volumeptr, volDims, neighbourCoords);

                    // when ray crosses a surface
                    if(signbit(f) != signbit(fnext))
                    {
                        stop = true; continue;
                    }

                    f = fnext;
                }
                steps++;
            }
        }

        // if ray penetrates a surface from outside
        // linearly interpolate t between two f values
        if(f > 0 && fnext < 0)
        {
            float3 tp = next - rayStep;
            float ft   = interpolateVoxel(tp,   volumeptr, volDims, neighbourCoords);
            float ftdt = interpolateVoxel(next, volumeptr, volDims, neighbourCoords);
            // float t = tmin + steps*tstep;
            // float ts = t - tstep*ft/(ftdt - ft);
            float ts = tmin + tstep*(steps - native_divide(ft, ftdt - ft));

            // avoid division by zero
            if(!isnan(ts) && !isinf(ts))
            {
                float3 pv = orig + dir*ts;
                float3 nv = getNormalVoxel(pv, volumeptr, volResolution, volDims, neighbourCoords);

                if(!any(isnan(nv)))
                {
                    //convert pv and nv to camera space
                    normal = (float3)(dot(nv, volRot0),
                                      dot(nv, volRot1),
                                      dot(nv, volRot2));
                    // interpolation optimized a little
                    pv *= voxelSize;
                    point = (float3)(dot(pv, volRot0),
                                     dot(pv, volRot1),
                                     dot(pv, volRot2)) + volTrans;
                }
            }
        }
    }

    __global float* pts = (__global float*)(pointsptr  +  points_offset + y*points_step  + x*sizeof(ptype));
    __global float* nrm = (__global float*)(normalsptr + normals_offset + y*normals_step + x*sizeof(ptype));
    vstore4((float4)(point,  0), 0, pts);
    vstore4((float4)(normal, 0), 0, nrm);
}


__kernel void getNormals(__global const char * pointsptr,
                         int points_step, int points_offset,
                         __global char * normalsptr,
                         int normals_step, int normals_offset,
                         const int2 frameSize,
                         __global const struct TsdfVoxel* volumeptr,
                         __global const float * volPoseptr,
                         __global const float * invPoseptr,
                         const float voxelSizeInv,
                         const int4 volResolution4,
                         const int4 volDims4,
                         const int8 neighbourCoords
                         )
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x >= frameSize.x || y >= frameSize.y)
        return;

    // coordinate-independent constants

    __global const float* vp = volPoseptr;
    const float3 volRot0  = vload4(0, vp).xyz;
    const float3 volRot1  = vload4(1, vp).xyz;
    const float3 volRot2  = vload4(2, vp).xyz;
    const float3 volTrans = (float3)(vp[3], vp[7], vp[11]);

    __global const float* iv = invPoseptr;
    const float3 invRot0 = vload4(0, iv).xyz;
    const float3 invRot1 = vload4(1, iv).xyz;
    const float3 invRot2 = vload4(2, iv).xyz;
    const float3 invTrans = (float3)(iv[3], iv[7], iv[11]);

    const int3 volResolution = volResolution4.xyz;
    const int3 volDims = volDims4.xyz;

    // kernel itself

    __global const ptype* ptsRow = (__global const ptype*)(pointsptr +
                                                           points_offset +
                                                           y*points_step);
    float3 p = ptsRow[x].xyz;
    float3 n = nan((uint)0);
    if(!any(isnan(p)))
    {
        float3 voxPt = (float3)(dot(p, invRot0),
                                dot(p, invRot1),
                                dot(p, invRot2)) + invTrans;
        voxPt = voxPt * voxelSizeInv;
        n = getNormalVoxel(voxPt, volumeptr, volResolution, volDims, neighbourCoords);
        n = (float3)(dot(n, volRot0),
                     dot(n, volRot1),
                     dot(n, volRot2));
    }

    __global float* nrm = (__global float*)(normalsptr +
                                            normals_offset +
                                            y*normals_step +
                                            x*sizeof(ptype));

    vstore4((float4)(n, 0), 0, nrm);
}

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics:enable

struct CoordReturn
{
    bool result;
    float3 point;
    float3 normal;
};

inline struct CoordReturn coord(int x, int y, int z, float3 V, float v0, int axis,
                                __global const struct TsdfVoxel* volumeptr,
                                int3 volResolution, int3 volDims,
                                int8 neighbourCoords,
                                float voxelSize, float voxelSizeInv,
                                const float3 volRot0,
                                const float3 volRot1,
                                const float3 volRot2,
                                const float3 volTrans,
                                bool needNormals,
                                bool scan
                                )
{
    struct CoordReturn cr;

    // 0 for x, 1 for y, 2 for z
    bool limits = false;
    int3 shift;
    float Vc = 0.f;
    if(axis == 0)
    {
        shift = (int3)(1, 0, 0);
        limits = (x + 1 < volResolution.x);
        Vc = V.x;
    }
    if(axis == 1)
    {
        shift = (int3)(0, 1, 0);
        limits = (y + 1 < volResolution.y);
        Vc = V.y;
    }
    if(axis == 2)
    {
        shift = (int3)(0, 0, 1);
        limits = (z + 1 < volResolution.z);
        Vc = V.z;
    }

    if(limits)
    {
        int3 ip = ((int3)(x, y, z)) + shift;
        int3 cmul = ip*volDims;
        int idx = cmul.x + cmul.y + cmul.z;

        struct TsdfVoxel voxel = volumeptr[idx];
        float vd  = tsdfToFloat(voxel.tsdf);
        int weight = voxel.weight;

        if(weight != 0 && vd != 1.f)
        {
            if((v0 > 0 && vd < 0) || (v0 < 0 && vd > 0))
            {
                // calc actual values or estimate amount of space
                if(!scan)
                {
                    // linearly interpolate coordinate
                    float Vn = Vc + voxelSize;
                    float dinv = 1.f/(fabs(v0)+fabs(vd));
                    float inter = (Vc*fabs(vd) + Vn*fabs(v0))*dinv;

                    float3 p = (float3)(shift.x ? inter : V.x,
                                        shift.y ? inter : V.y,
                                        shift.z ? inter : V.z);

                    cr.point = (float3)(dot(p, volRot0),
                                        dot(p, volRot1),
                                        dot(p, volRot2)) + volTrans;

                    if(needNormals)
                    {
                        float3 nv = getNormalVoxel(p * voxelSizeInv,
                                                   volumeptr, volResolution, volDims, neighbourCoords);

                        cr.normal = (float3)(dot(nv, volRot0),
                                             dot(nv, volRot1),
                                             dot(nv, volRot2));
                    }
                }

                cr.result = true;
                return cr;
            }
        }
    }

    cr.result = false;
    return cr;
}


__kernel void scanSize(__global const struct TsdfVoxel* volumeptr,
                       const int4 volResolution4,
                       const int4 volDims4,
                       const int8 neighbourCoords,
                       __global const float * volPoseptr,
                       const float voxelSize,
                       const float voxelSizeInv,
                       __local int* reducebuf,
                       __global char* groupedSumptr,
                       int groupedSum_slicestep,
                       int groupedSum_step, int groupedSum_offset
                       )
{
    const int3 volDims = volDims4.xyz;
    const int3 volResolution = volResolution4.xyz;

    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);

    bool validVoxel = true;
    if(x >= volResolution.x || y >= volResolution.y || z >= volResolution.z)
        validVoxel = false;

    const int gx = get_group_id(0);
    const int gy = get_group_id(1);
    const int gz = get_group_id(2);

    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int lz = get_local_id(2);
    const int lw = get_local_size(0);
    const int lh = get_local_size(1);
    const int ld = get_local_size(2);
    const int lsz = lw*lh*ld;
    const int lid = lx + ly*lw + lz*lw*lh;

    // coordinate-independent constants

    __global const float* vp = volPoseptr;
    const float3 volRot0  = vload4(0, vp).xyz;
    const float3 volRot1  = vload4(1, vp).xyz;
    const float3 volRot2  = vload4(2, vp).xyz;
    const float3 volTrans = (float3)(vp[3], vp[7], vp[11]);

    // kernel itself
    int npts = 0;
    if(validVoxel)
    {
        int3 ip = (int3)(x, y, z);
        int3 cmul = ip*volDims;
        int idx = cmul.x + cmul.y + cmul.z;
        struct TsdfVoxel voxel = volumeptr[idx];
        float value  = tsdfToFloat(voxel.tsdf);
        int weight = voxel.weight;

        // if voxel is not empty
        if(weight != 0 && value != 1.f)
        {
            float3 V = (((float3)(x, y, z)) + 0.5f)*voxelSize;

            #pragma unroll
            for(int i = 0; i < 3; i++)
            {
                struct CoordReturn cr;
                cr = coord(x, y, z, V, value, i,
                           volumeptr, volResolution, volDims,
                           neighbourCoords,
                           voxelSize, voxelSizeInv,
                           volRot0, volRot1, volRot2, volTrans,
                           false, true);
                if(cr.result)
                {
                    npts++;
                }
            }
        }
    }

    // reducebuf keeps counters for each thread
    reducebuf[lid] = npts;

    // reduce counter to local mem

    // maxStep = ctz(lsz), ctz isn't supported on CUDA devices
    const int c = clz(lsz & -lsz);
    const int maxStep = c ? 31 - c : c;
    for(int nstep = 1; nstep <= maxStep; nstep++)
    {
        if(lid % (1 << nstep) == 0)
        {
            int rto   = lid;
            int rfrom = lid + (1 << (nstep-1));
            reducebuf[rto] += reducebuf[rfrom];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(lid == 0)
    {
        __global int* groupedRow = (__global int*)(groupedSumptr +
                                                   groupedSum_offset +
                                                   gy*groupedSum_step +
                                                   gz*groupedSum_slicestep);

        groupedRow[gx] = reducebuf[0];
    }
}


__kernel void fillPtsNrm(__global const struct TsdfVoxel* volumeptr,
                         const int4 volResolution4,
                         const int4 volDims4,
                         const int8 neighbourCoords,
                         __global const float * volPoseptr,
                         const float voxelSize,
                         const float voxelSizeInv,
                         const int needNormals,
                         __local float* localbuf,
                         volatile __global int* atomicCtr,
                         __global const char* groupedSumptr,
                         int groupedSum_slicestep,
                         int groupedSum_step, int groupedSum_offset,
                         __global char * pointsptr,
                         int points_step, int points_offset,
                         __global char * normalsptr,
                         int normals_step, int normals_offset
                         )
{
    const int3 volDims = volDims4.xyz;
    const int3 volResolution = volResolution4.xyz;

    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);

    bool validVoxel = true;
    if(x >= volResolution.x || y >= volResolution.y || z >= volResolution.z)
        validVoxel = false;

    const int gx = get_group_id(0);
    const int gy = get_group_id(1);
    const int gz = get_group_id(2);

    __global int* groupedRow = (__global int*)(groupedSumptr +
                                               groupedSum_offset +
                                               gy*groupedSum_step +
                                               gz*groupedSum_slicestep);

    // this group contains 0 pts, skip it
    int nptsGroup = groupedRow[gx];
    if(nptsGroup == 0)
        return;

    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int lz = get_local_id(2);
    const int lw = get_local_size(0);
    const int lh = get_local_size(1);
    const int ld = get_local_size(2);
    const int lsz = lw*lh*ld;
    const int lid = lx + ly*lw + lz*lw*lh;

    // coordinate-independent constants

    __global const float* vp = volPoseptr;
    const float3 volRot0  = vload4(0, vp).xyz;
    const float3 volRot1  = vload4(1, vp).xyz;
    const float3 volRot2  = vload4(2, vp).xyz;
    const float3 volTrans = (float3)(vp[3], vp[7], vp[11]);

    // kernel itself
    int npts = 0;
    float3 parr[3], narr[3];
    if(validVoxel)
    {
        int3 ip = (int3)(x, y, z);
        int3 cmul = ip*volDims;
        int idx = cmul.x + cmul.y + cmul.z;
        struct TsdfVoxel voxel = volumeptr[idx];
        float value  = tsdfToFloat(voxel.tsdf);
        int weight = voxel.weight;

        // if voxel is not empty
        if(weight != 0 && value != 1.f)
        {
            float3 V = (((float3)(x, y, z)) + 0.5f)*voxelSize;

            #pragma unroll
            for(int i = 0; i < 3; i++)
            {
                struct CoordReturn cr;
                cr = coord(x, y, z, V, value, i,
                           volumeptr, volResolution, volDims,
                           neighbourCoords,
                           voxelSize, voxelSizeInv,
                           volRot0, volRot1, volRot2, volTrans,
                           needNormals, false);

                if(cr.result)
                {
                    parr[npts] = cr.point;
                    narr[npts] = cr.normal;
                    npts++;
                }
            }
        }
    }

    // 4 floats per point or normal
    const int elemStep = 4;

    __local float* normAddr;
    __local int localCtr;
    if(lid == 0)
        localCtr = 0;

    // push all pts (and nrm) from private array to local mem
    int privateCtr = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    privateCtr = atomic_add(&localCtr, npts);
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int i = 0; i < npts; i++)
    {
        __local float* addr = localbuf + (privateCtr+i)*elemStep;
        vstore4((float4)(parr[i], 0), 0, addr);
    }

    if(needNormals)
    {
        normAddr = localbuf + localCtr*elemStep;

        for(int i = 0; i < npts; i++)
        {
            __local float* addr = normAddr + (privateCtr+i)*elemStep;
            vstore4((float4)(narr[i], 0), 0, addr);
        }
    }

    // debugging purposes
    if(lid == 0)
    {
        if(localCtr != nptsGroup)
        {
            printf("!!! fetchPointsNormals result may be incorrect, npts != localCtr at %3d %3d %3d: %3d vs %3d\n",
                   gx, gy, gz, localCtr, nptsGroup);
        }
    }

    // copy local buffer to global mem
    __local int whereToWrite;
    if(lid == 0)
        whereToWrite = atomic_add(atomicCtr, localCtr);
    barrier(CLK_GLOBAL_MEM_FENCE);

    event_t ev[2];
    int evn = 0;
    // points and normals are 1-column matrices
    __global float* pts = (__global float*)(pointsptr +
                                            points_offset +
                                            whereToWrite*points_step);
    ev[evn++] = async_work_group_copy(pts, localbuf, localCtr*elemStep, 0);

    if(needNormals)
    {
        __global float* nrm = (__global float*)(normalsptr +
                                                normals_offset +
                                                whereToWrite*normals_step);
        ev[evn++] = async_work_group_copy(nrm, normAddr, localCtr*elemStep, 0);
    }

    wait_group_events(evn, ev);
}
