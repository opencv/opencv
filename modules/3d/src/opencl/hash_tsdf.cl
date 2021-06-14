// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

#define USE_INTERPOLATION_IN_GETNORMAL 1
#define HASH_DIVISOR 32768

typedef char int8_t;
typedef uint int32_t;

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

static uint calc_hash(int3 x)
{
    unsigned int seed = 0;
    unsigned int GOLDEN_RATIO = 0x9e3779b9;
    seed ^= x.s0 + GOLDEN_RATIO + (seed << 6) + (seed >> 2);
    seed ^= x.s1 + GOLDEN_RATIO + (seed << 6) + (seed >> 2);
    seed ^= x.s2 + GOLDEN_RATIO + (seed << 6) + (seed >> 2);
    return seed;
}


//TODO: make hashDivisor a power of 2
//TODO: put it to this .cl file as a constant
static int custom_find(int3 idx, const int hashDivisor, __global const int* hashes,
                    __global const int4* data)
{
    int hash = calc_hash(idx) % hashDivisor;
    int place = hashes[hash];
    // search a place
    while (place >= 0)
    {
        if (all(data[place].s012 == idx))
            break;
        else
            place = data[place].s3;
    }

    return place;
}



static void integrateVolumeUnit(
                        int x, int y,
                        __global const char * depthptr,
                        int depth_step, int depth_offset,
                        int depth_rows, int depth_cols,
                        __global struct TsdfVoxel * volumeptr,
                        const __global char * pixNormsPtr,
                        int pixNormsStep, int pixNormsOffset,
                        int pixNormsRows, int pixNormsCols,
                        const float16 vol2camMatrix,
                        const float voxelSize,
                        const int4 volResolution4,
                        const int4 volStrides4,
                        const float2 fxy,
                        const float2 cxy,
                        const float dfac,
                        const float truncDist,
                        const int maxWeight
                        )
{
    const int3 volResolution = volResolution4.xyz;

    if(x >= volResolution.x || y >= volResolution.y)
        return;

    // coord-independent constants
    const int3 volStrides = volStrides4.xyz;
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

    int volYidx = x*volStrides.x + y*volStrides.y;

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
            startZ = endZ = 0;
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
        float2 projected = mad(camPixVec.xy, fxy, cxy); // mad(a,b,c) = a * b + c

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

        int2 projInt = convert_int2(projected);
        float pixNorm = *(__global const float*)(pixNormsPtr + pixNormsOffset + projInt.y*pixNormsStep + projInt.x*sizeof(float));
        //float pixNorm = length(camPixVec);

        // difference between distances of point and of surface to camera
        float sdf = pixNorm*(v*dfac - camSpacePt.z);
        // possible alternative is:
        // float sdf = length(camSpacePt)*(v*dfac/camSpacePt.z - 1.0);
        if(sdf >= -truncDist)
        {
            float tsdf = fmin(1.0f, sdf * truncDistInv);
            int volIdx = volYidx + z*volStrides.z;

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


__kernel void integrateAllVolumeUnits(
                        // depth
                        __global const char * depthptr,
                        int depth_step, int depth_offset,
                        int depth_rows, int depth_cols,
                        // hashMap
                        __global const int* hashes,
                        __global const int4* data,
                        // volUnitsData
                        __global struct TsdfVoxel * allVolumePtr,
                        int table_step, int table_offset,
                        int table_rows, int table_cols,
                        // pixNorms
                        const __global char * pixNormsPtr,
                        int pixNormsStep, int pixNormsOffset,
                        int pixNormsRows, int pixNormsCols,
                        // isActiveFlags
                        __global const uchar* isActiveFlagsPtr,
                        int isActiveFlagsStep, int isActiveFlagsOffset,
                        int isActiveFlagsRows, int isActiveFlagsCols,
                        // cam matrices:
                        const float16 vol2cam,
                        const float16 camInv,
                        // scalars:
                        const float voxelSize,
                        const int volUnitResolution,
                        const int4 volStrides4,
                        const float2 fxy,
                        const float2 cxy,
                        const float dfac,
                        const float truncDist,
                        const int maxWeight
                        )
{
    const int hash_divisor = HASH_DIVISOR;
    int i = get_global_id(0);
    int j = get_global_id(1);
    int row = get_global_id(2);
    int3 idx = data[row].xyz;

    const int4 volResolution4 = (int4)(volUnitResolution,
                                       volUnitResolution,
                                       volUnitResolution,
                                       volUnitResolution);

    int isActive = *(__global const uchar*)(isActiveFlagsPtr + isActiveFlagsOffset + row);

    if (isActive)
    {
        int volCubed = volUnitResolution * volUnitResolution * volUnitResolution;
        __global struct TsdfVoxel * volumeptr = (__global struct TsdfVoxel*)
                                                (allVolumePtr + table_offset + row * volCubed);

        // volUnit2cam = world2cam * volUnit2world =
        // camPoseInv * volUnitPose = camPoseInv * (volPose + idx * volUnitSize) =
        // camPoseInv * (volPose + idx * volUnitResolution * voxelSize) =
        // camPoseInv * (volPose + mulIdx) = camPoseInv * volPose + camPoseInv * mulIdx =
        // vol2cam + camPoseInv * mulIdx
        float3 mulIdx = convert_float3(idx * volUnitResolution) * voxelSize;
        float16 volUnit2cam = vol2cam;
        volUnit2cam.s37b += (float3)(dot(mulIdx, camInv.s012),
                                     dot(mulIdx, camInv.s456),
                                     dot(mulIdx, camInv.s89a));

        integrateVolumeUnit(
            i, j,
            depthptr,
            depth_step, depth_offset,
            depth_rows, depth_cols,
            volumeptr,
            pixNormsPtr,
            pixNormsStep, pixNormsOffset,
            pixNormsRows, pixNormsCols,
            volUnit2cam,
            voxelSize,
            volResolution4,
            volStrides4,
            fxy,
            cxy,
            dfac,
            truncDist,
            maxWeight
            );
    }
}


static struct TsdfVoxel at(int3 volumeIdx, int row, int volumeUnitDegree,
                           int3 volStrides, __global const struct TsdfVoxel * allVolumePtr, int table_offset)

{
    //! Out of bounds
    if (any(volumeIdx >= (1 << volumeUnitDegree)) ||
        any(volumeIdx < 0))
    {
        struct TsdfVoxel dummy;
        dummy.tsdf = floatToTsdf(1.0f);
        dummy.weight = 0;
        return dummy;
    }

    int volCubed = 1 << (volumeUnitDegree*3);
    __global struct TsdfVoxel * volData = (__global struct TsdfVoxel*)
                                          (allVolumePtr + table_offset + row * volCubed);
    int3 ismul = volumeIdx * volStrides;
    int coordBase = ismul.x + ismul.y + ismul.z;
    return volData[coordBase];
}


static struct TsdfVoxel atVolumeUnit(int3 volumeIdx, int3 volumeUnitIdx, int row,
                                     int volumeUnitDegree, int3 volStrides,
                                     __global const struct TsdfVoxel * allVolumePtr, int table_offset)

{
    //! Out of bounds
    if (row < 0)
    {
        struct TsdfVoxel dummy;
        dummy.tsdf = floatToTsdf(1.0f);
        dummy.weight = 0;
        return dummy;
    }

    int3 volUnitLocalIdx = volumeIdx - (volumeUnitIdx << volumeUnitDegree);
    int volCubed = 1 << (volumeUnitDegree*3);
    __global struct TsdfVoxel * volData = (__global struct TsdfVoxel*)
                                          (allVolumePtr + table_offset + row * volCubed);
    int3 ismul = volUnitLocalIdx * volStrides;
    int coordBase = ismul.x + ismul.y + ismul.z;
    return volData[coordBase];
}

inline float interpolate(float3 t, float8 vz)
{
    float4 vy = mix(vz.s0246, vz.s1357, t.z);
    float2 vx = mix(vy.s02, vy.s13, t.y);
    return mix(vx.s0, vx.s1, t.x);
}

inline float3 getNormalVoxel(float3 ptVox, __global const struct TsdfVoxel* allVolumePtr,
                             int volumeUnitDegree,
                             const int hash_divisor,
                             __global const int* hashes,
                             __global const int4* data,

                             int3 volStrides, int table_offset)
{
    float3 normal = (float3) (0.0f, 0.0f, 0.0f);
    float3 fip = floor(ptVox);
    int3 iptVox = convert_int3(fip);

    // A small hash table to reduce a number of findRow() calls
    // -2 and lower means not queried yet
    // -1 means not found
    // 0+ means found
    int iterMap[8];
    for (int i = 0; i < 8; i++)
    {
        iterMap[i] = -2;
    }

#if !USE_INTERPOLATION_IN_GETNORMAL
    int4 offsets[] = { (int4)( 1,  0,  0, 0), (int4)(-1,  0,  0, 0), (int4)( 0,  1,  0, 0), // 0-3
                       (int4)( 0, -1,  0, 0), (int4)( 0,  0,  1, 0), (int4)( 0,  0, -1, 0)  // 4-7
    };

    const int nVals = 6;
    float vals[6];
#else
    int4 offsets[]={(int4)( 0,  0,  0, 0), (int4)( 0,  0,  1, 0), (int4)( 0,  1,  0, 0), (int4)( 0,  1,  1, 0), //  0-3
                    (int4)( 1,  0,  0, 0), (int4)( 1,  0,  1, 0), (int4)( 1,  1,  0, 0), (int4)( 1,  1,  1, 0), //  4-7
                    (int4)(-1,  0,  0, 0), (int4)(-1,  0,  1, 0), (int4)(-1,  1,  0, 0), (int4)(-1,  1,  1, 0), //  8-11
                    (int4)( 2,  0,  0, 0), (int4)( 2,  0,  1, 0), (int4)( 2,  1,  0, 0), (int4)( 2,  1,  1, 0), // 12-15
                    (int4)( 0, -1,  0, 0), (int4)( 0, -1,  1, 0), (int4)( 1, -1,  0, 0), (int4)( 1, -1,  1, 0), // 16-19
                    (int4)( 0,  2,  0, 0), (int4)( 0,  2,  1, 0), (int4)( 1,  2,  0, 0), (int4)( 1,  2,  1, 0), // 20-23
                    (int4)( 0,  0, -1, 0), (int4)( 0,  1, -1, 0), (int4)( 1,  0, -1, 0), (int4)( 1,  1, -1, 0), // 24-27
                    (int4)( 0,  0,  2, 0), (int4)( 0,  1,  2, 0), (int4)( 1,  0,  2, 0), (int4)( 1,  1,  2, 0), // 28-31
    };
    const int nVals = 32;
    float vals[32];
#endif

    for (int i = 0; i < nVals; i++)
    {
        int3 pt = iptVox + offsets[i].s012;

        // VoxelToVolumeUnitIdx()
        int3 volumeUnitIdx = pt >> volumeUnitDegree;

        int3 vand = (volumeUnitIdx & 1);
        int dictIdx = vand.s0 + vand.s1 * 2 + vand.s2 * 4;

        int it = iterMap[dictIdx];
        if (it < -1)
        {
            it = custom_find(volumeUnitIdx, hash_divisor, hashes, data);
            iterMap[dictIdx] = it;
        }

        struct TsdfVoxel tmp = atVolumeUnit(pt, volumeUnitIdx, it, volumeUnitDegree, volStrides, allVolumePtr, table_offset);
        vals[i] = tsdfToFloat( tmp.tsdf );
    }

#if !USE_INTERPOLATION_IN_GETNORMAL
    float3 pv, nv;

    pv = (float3)(vals[0*2  ], vals[1*2  ], vals[2*2  ]);
    nv = (float3)(vals[0*2+1], vals[1*2+1], vals[2*2+1]);
    normal = pv - nv;
#else

    float cxv[8], cyv[8], czv[8];

    // How these numbers were obtained:
    // 1. Take the basic interpolation sequence:
    // 000, 001, 010, 011, 100, 101, 110, 111
    // where each digit corresponds to shift by x, y, z axis respectively.
    // 2. Add +1 for next or -1 for prev to each coordinate to corresponding axis
    // 3. Search corresponding values in offsets
    const int idxxn[8] = {  8,  9, 10, 11,  0,  1,  2,  3 };
    const int idxxp[8] = {  4,  5,  6,  7, 12, 13, 14, 15 };
    const int idxyn[8] = { 16, 17,  0,  1, 18, 19,  4,  5 };
    const int idxyp[8] = {  2,  3, 20, 21,  6,  7, 22, 23 };
    const int idxzn[8] = { 24,  0, 25,  2, 26,  4, 27,  6 };
    const int idxzp[8] = {  1, 28,  3, 29,  5, 30,  7, 31 };

    float vcxp[8], vcxn[8];
    float vcyp[8], vcyn[8];
    float vczp[8], vczn[8];

    for (int i = 0; i < 8; i++)
    {
        vcxp[i] = vals[idxxp[i]]; vcxn[i] = vals[idxxn[i]];
        vcyp[i] = vals[idxyp[i]]; vcyn[i] = vals[idxyn[i]];
        vczp[i] = vals[idxzp[i]]; vczn[i] = vals[idxzn[i]];
    }

    float8 cxp = vload8(0, vcxp), cxn = vload8(0, vcxn);
    float8 cyp = vload8(0, vcyp), cyn = vload8(0, vcyn);
    float8 czp = vload8(0, vczp), czn = vload8(0, vczn);
    float8 cx = cxp - cxn;
    float8 cy = cyp - cyn;
    float8 cz = czp - czn;

    float3 tv = ptVox - fip;
    normal.x = interpolate(tv, cx);
    normal.y = interpolate(tv, cy);
    normal.z = interpolate(tv, cz);
#endif

    float norm = sqrt(dot(normal, normal));
    return norm < 0.0001f ? nan((uint)0) : normal / norm;
}

typedef float4 ptype;

__kernel void raycast(
                    __global const int* hashes,
                    __global const int4* data,
                    __global char * pointsptr,
                      int points_step, int points_offset,
                    __global char * normalsptr,
                      int normals_step, int normals_offset,
                    const int2 frameSize,
                    __global const struct TsdfVoxel * allVolumePtr,
                        int table_step, int table_offset,
                        int table_rows, int table_cols,
                    float16 cam2volRotGPU,
                    float16 vol2camRotGPU,
                    float truncateThreshold,
                    const float2 fixy, const float2 cxy,
                    const float4 boxDown4, const float4 boxUp4,
                    const float tstep,
                    const float voxelSize,
                    const float voxelSizeInv,
                    float volumeUnitSize,
                    float truncDist,
                    int volumeUnitDegree,
                    int4 volStrides4
                    )
{
    const int hash_divisor = HASH_DIVISOR;
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x >= frameSize.x || y >= frameSize.y)
        return;

    float3 point  = nan((uint)0);
    float3 normal = nan((uint)0);

    const float3 camRot0  = cam2volRotGPU.s012;
    const float3 camRot1  = cam2volRotGPU.s456;
    const float3 camRot2  = cam2volRotGPU.s89a;
    const float3 camTrans = cam2volRotGPU.s37b;

    const float3 volRot0  = vol2camRotGPU.s012;
    const float3 volRot1  = vol2camRotGPU.s456;
    const float3 volRot2  = vol2camRotGPU.s89a;
    const float3 volTrans = vol2camRotGPU.s37b;

    float3 planed = (float3)(((float2)(x, y) - cxy)*fixy, 1.f);
    planed = (float3)(dot(planed, camRot0),
                      dot(planed, camRot1),
                      dot(planed, camRot2));

    float3 orig = (float3) (camTrans.s0, camTrans.s1, camTrans.s2);
    float3 dir = fast_normalize(planed);
    float3 origScaled = orig * voxelSizeInv;
    float3 dirScaled = dir * voxelSizeInv;

    float tmin = 0;
    float tmax = truncateThreshold;
    float tcurr = tmin;
    float tprev = tcurr;
    float prevTsdf = truncDist;

    int3 volStrides = volStrides4.xyz;

    while (tcurr < tmax)
    {
        float3 currRayPosVox = origScaled + tcurr * dirScaled;

        // VolumeToVolumeUnitIdx()
        int3 currVoxel = convert_int3(floor(currRayPosVox));
        int3 currVolumeUnitIdx = currVoxel >> volumeUnitDegree;

        int row = custom_find(currVolumeUnitIdx, hash_divisor, hashes, data);

        float currTsdf = prevTsdf;
        int currWeight = 0;
        float stepSize = 0.5 * volumeUnitSize;
        int3 volUnitLocalIdx;

        if (row >= 0)
        {
            volUnitLocalIdx = currVoxel - (currVolumeUnitIdx << volumeUnitDegree);
            struct TsdfVoxel currVoxel = at(volUnitLocalIdx, row, volumeUnitDegree, volStrides, allVolumePtr, table_offset);

            currTsdf = tsdfToFloat(currVoxel.tsdf);
            currWeight = currVoxel.weight;
            stepSize = tstep;
        }

        if (prevTsdf > 0.f && currTsdf <= 0.f && currWeight > 0)
        {
            float tInterp = (tcurr * prevTsdf - tprev * currTsdf) / (prevTsdf - currTsdf);
            if ( !isnan(tInterp) && !isinf(tInterp) )
            {
                float3 pvox = origScaled + tInterp * dirScaled;
                float3 nv = getNormalVoxel( pvox, allVolumePtr, volumeUnitDegree,
                                            hash_divisor, hashes, data,
                                            volStrides, table_offset);

                if(!any(isnan(nv)))
                {
                    //convert pv and nv to camera space
                    normal = (float3)(dot(nv, volRot0),
                                      dot(nv, volRot1),
                                      dot(nv, volRot2));
                    // interpolation optimized a little
                    float3 pv = pvox * voxelSize;
                    point = (float3)(dot(pv, volRot0),
                                     dot(pv, volRot1),
                                     dot(pv, volRot2)) + volTrans;
                }
            }
            break;
        }
        prevTsdf = currTsdf;
        tprev = tcurr;
        tcurr += stepSize;
    }

    __global float* pts = (__global float*)(pointsptr  +  points_offset + y*points_step   + x*sizeof(ptype));
    __global float* nrm = (__global float*)(normalsptr + normals_offset + y*normals_step  + x*sizeof(ptype));
    vstore4((float4)(point,  0), 0, pts);
    vstore4((float4)(normal, 0), 0, nrm);
}


__kernel void markActive (
        __global const int4* hashSetData,

        __global char* isActiveFlagsPtr,
        int isActiveFlagsStep, int isActiveFlagsOffset,
        int isActiveFlagsRows, int isActiveFlagsCols,

        __global char* lastVisibleIndicesPtr,
        int lastVisibleIndicesStep, int lastVisibleIndicesOffset,
        int lastVisibleIndicesRows, int lastVisibleIndicesCols,

        const float16 vol2cam,
        const float2 fxy,
        const float2 cxy,
        const int2 frameSz,
        const float volumeUnitSize,
        const int lastVolIndex,
        const float truncateThreshold,
        const int frameId
        )
{
    const int hash_divisor = HASH_DIVISOR;
    int row = get_global_id(0);

    if (row < lastVolIndex)
    {
        int3 idx = hashSetData[row].xyz;

        float3 volumeUnitPos = convert_float3(idx) * volumeUnitSize;

        float3 volUnitInCamSpace = (float3) (dot(volumeUnitPos, vol2cam.s012),
                                             dot(volumeUnitPos, vol2cam.s456),
                                             dot(volumeUnitPos, vol2cam.s89a)) + vol2cam.s37b;

        if (volUnitInCamSpace.z < 0 || volUnitInCamSpace.z > truncateThreshold)
        {
            *(isActiveFlagsPtr + isActiveFlagsOffset + row * isActiveFlagsStep) = 0;
            return;
        }

        float2 cameraPoint;
        float invz = 1.f / volUnitInCamSpace.z;
        cameraPoint = fxy * volUnitInCamSpace.xy * invz + cxy;

        if (all(cameraPoint >= 0) && all(cameraPoint < convert_float2(frameSz)))
        {
            *(__global int*)(lastVisibleIndicesPtr + lastVisibleIndicesOffset + row * lastVisibleIndicesStep) = frameId;
            *(isActiveFlagsPtr + isActiveFlagsOffset + row * isActiveFlagsStep) = 1;
        }
    }
}
