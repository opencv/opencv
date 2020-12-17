// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

#define UTSIZE 27

typedef float4 ptype;

/*
    Calculate an upper triangle of Ab matrix then reduce it across workgroup
*/

inline void calcAb7(__global const char * oldPointsptr,
                    int oldPoints_step, int oldPoints_offset,
                    __global const char * oldNormalsptr,
                    int oldNormals_step, int oldNormals_offset,
                    const int2 oldSize,
                    __global const char * newPointsptr,
                    int newPoints_step, int newPoints_offset,
                    __global const char * newNormalsptr,
                    int newNormals_step, int newNormals_offset,
                    const int2 newSize,
                    const float16 poseMatrix,
                    const float2 fxy,
                    const float2 cxy,
                    const float sqDistanceThresh,
                    const float minCos,
                    float* ab7
                    )
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if(x >= newSize.x || y >= newSize.y)
        return;

    // coord-independent constants

    const float3 poseRot0 = poseMatrix.s012;
    const float3 poseRot1 = poseMatrix.s456;
    const float3 poseRot2 = poseMatrix.s89a;
    const float3 poseTrans = poseMatrix.s37b;

    const float2 oldEdge = (float2)(oldSize.x - 1, oldSize.y - 1);

    __global const ptype* newPtsRow = (__global const ptype*)(newPointsptr +
                                                              newPoints_offset +
                                                              y*newPoints_step);

    __global const ptype* newNrmRow = (__global const ptype*)(newNormalsptr +
                                                              newNormals_offset +
                                                              y*newNormals_step);

    float3 newP = newPtsRow[x].xyz;
    float3 newN = newNrmRow[x].xyz;

    if( any(isnan(newP)) || any(isnan(newN)) ||
        any(isinf(newP)) || any(isinf(newN)) )
        return;

    //transform to old coord system
    newP = (float3)(dot(newP, poseRot0),
                    dot(newP, poseRot1),
                    dot(newP, poseRot2)) + poseTrans;
    newN = (float3)(dot(newN, poseRot0),
                    dot(newN, poseRot1),
                    dot(newN, poseRot2));

    //find correspondence by projecting the point
    float2 oldCoords = (newP.xy/newP.z)*fxy+cxy;

    if(!(all(oldCoords >= 0.f) && all(oldCoords < oldEdge)))
        return;

    // bilinearly interpolate oldPts and oldNrm under oldCoords point
    float3 oldP, oldN;
    float2 ip = floor(oldCoords);
    float2 t = oldCoords - ip;
    int xi = ip.x, yi = ip.y;

    __global const ptype* prow0 = (__global const ptype*)(oldPointsptr +
                                                          oldPoints_offset +
                                                          (yi+0)*oldPoints_step);
    __global const ptype* prow1 = (__global const ptype*)(oldPointsptr +
                                                          oldPoints_offset +
                                                          (yi+1)*oldPoints_step);
    float3 p00 = prow0[xi+0].xyz;
    float3 p01 = prow0[xi+1].xyz;
    float3 p10 = prow1[xi+0].xyz;
    float3 p11 = prow1[xi+1].xyz;

    // NaN check is done later

    __global const ptype* nrow0 = (__global const ptype*)(oldNormalsptr +
                                                          oldNormals_offset +
                                                          (yi+0)*oldNormals_step);
    __global const ptype* nrow1 = (__global const ptype*)(oldNormalsptr +
                                                          oldNormals_offset +
                                                          (yi+1)*oldNormals_step);

    float3 n00 = nrow0[xi+0].xyz;
    float3 n01 = nrow0[xi+1].xyz;
    float3 n10 = nrow1[xi+0].xyz;
    float3 n11 = nrow1[xi+1].xyz;

    // NaN check is done later

    float3 p0 = mix(p00, p01, t.x);
    float3 p1 = mix(p10, p11, t.x);
    oldP = mix(p0, p1, t.y);

    float3 n0 = mix(n00, n01, t.x);
    float3 n1 = mix(n10, n11, t.x);
    oldN = mix(n0, n1, t.y);

    if( any(isnan(oldP)) || any(isnan(oldN)) ||
        any(isinf(oldP)) || any(isinf(oldN)) )
        return;

    //filter by distance
    float3 diff = newP - oldP;
    if(dot(diff, diff) > sqDistanceThresh)
        return;

    //filter by angle
    if(fabs(dot(newN, oldN)) < minCos)
        return;

    // build point-wise vector ab = [ A | b ]

    float3 VxN = cross(newP, oldN);
    float ab[7] = {VxN.x, VxN.y, VxN.z, oldN.x, oldN.y, oldN.z, -dot(oldN, diff)};

    for(int i = 0; i < 7; i++)
        ab7[i] = ab[i];
}


__kernel void getAb(__global const char * oldPointsptr,
                    int oldPoints_step, int oldPoints_offset,
                    __global const char * oldNormalsptr,
                    int oldNormals_step, int oldNormals_offset,
                    const int2 oldSize,
                    __global const char * newPointsptr,
                    int newPoints_step, int newPoints_offset,
                    __global const char * newNormalsptr,
                    int newNormals_step, int newNormals_offset,
                    const int2 newSize,
                    const float16 poseMatrix,
                    const float2 fxy,
                    const float2 cxy,
                    const float sqDistanceThresh,
                    const float minCos,
                    __local float * reducebuf,
                    __global char* groupedSumptr,
                    int groupedSum_step, int groupedSum_offset
)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if(x >= newSize.x || y >= newSize.y)
        return;

    const int gx = get_group_id(0);
    const int gy = get_group_id(1);
    const int gw = get_num_groups(0);
    const int gh = get_num_groups(1);

    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int lw = get_local_size(0);
    const int lh = get_local_size(1);
    const int lsz = lw*lh;
    const int lid = lx + ly*lw;

    float ab[7];
    for(int i = 0; i < 7; i++)
        ab[i] = 0;

    calcAb7(oldPointsptr,
            oldPoints_step, oldPoints_offset,
            oldNormalsptr,
            oldNormals_step, oldNormals_offset,
            oldSize,
            newPointsptr,
            newPoints_step, newPoints_offset,
            newNormalsptr,
            newNormals_step, newNormals_offset,
            newSize,
            poseMatrix,
            fxy, cxy,
            sqDistanceThresh,
            minCos,
            ab);

    // build point-wise upper-triangle matrix [ab^T * ab] w/o last row
    // which is [A^T*A | A^T*b]
    // and gather sum

    __local float* upperTriangle = reducebuf + lid*UTSIZE;

    int pos = 0;
    for(int i = 0; i < 6; i++)
    {
        for(int j = i; j < 7; j++)
        {
            upperTriangle[pos++] = ab[i]*ab[j];
        }
    }

    // reduce upperTriangle to local mem

    // maxStep = ctz(lsz), ctz isn't supported on CUDA devices
    const int c = clz(lsz & -lsz);
    const int maxStep = c ? 31 - c : c;
    for(int nstep = 1; nstep <= maxStep; nstep++)
    {
        if(lid % (1 << nstep) == 0)
        {
            __local float* rto   = reducebuf + UTSIZE*lid;
            __local float* rfrom = reducebuf + UTSIZE*(lid+(1 << (nstep-1)));
            for(int i = 0; i < UTSIZE; i++)
                rto[i] += rfrom[i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // here group sum should be in reducebuf[0...UTSIZE]
    if(lid == 0)
    {
        __global float* groupedRow = (__global float*)(groupedSumptr +
                                                       groupedSum_offset +
                                                       gy*groupedSum_step);

        for(int i = 0; i < UTSIZE; i++)
            groupedRow[gx*UTSIZE + i] = reducebuf[i];
    }
}
