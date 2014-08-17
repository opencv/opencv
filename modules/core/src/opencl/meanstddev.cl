// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2014, Itseez, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

#ifdef DOUBLE_SUPPORT
#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#elif defined (cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif
#endif

#define noconvert

#if cn != 3
#define loadpix(addr) *(__global const srcT *)(addr)
#define storepix(val, addr)  *(__global dstT *)(addr) = val
#define storesqpix(val, addr)  *(__global sqdstT *)(addr) = val
#define srcTSIZE (int)sizeof(srcT)
#define dstTSIZE (int)sizeof(dstT)
#define sqdstTSIZE (int)sizeof(sqdstT)
#else
#define loadpix(addr) vload3(0, (__global const srcT1 *)(addr))
#define storepix(val, addr) vstore3(val, 0, (__global dstT1 *)(addr))
#define storesqpix(val, addr) vstore3(val, 0, (__global sqdstT1 *)(addr))
#define srcTSIZE ((int)sizeof(srcT1)*3)
#define dstTSIZE ((int)sizeof(dstT1)*3)
#define sqdstTSIZE ((int)sizeof(sqdstT1)*3)
#endif

__kernel void meanStdDev(__global const uchar * srcptr, int src_step, int src_offset, int cols,
                         int total, int groups, __global uchar * dstptr
 #ifdef HAVE_MASK
                         , __global const uchar * mask, int mask_step, int mask_offset
 #endif
                        )
{
    int lid = get_local_id(0);
    int gid = get_group_id(0);
    int id = get_global_id(0);

    __local dstT localMemSum[WGS2_ALIGNED];
    __local sqdstT localMemSqSum[WGS2_ALIGNED];
#ifdef HAVE_MASK
    __local int localMemNonZero[WGS2_ALIGNED];
#endif

    dstT accSum = (dstT)(0);
    sqdstT accSqSum = (sqdstT)(0);
#ifdef HAVE_MASK
    int accNonZero = 0;
    mask += mask_offset;
#endif
    srcptr += src_offset;

    for (int grain = groups * WGS; id < total; id += grain)
    {
#ifdef HAVE_MASK
#ifdef HAVE_MASK_CONT
        int mask_index = id;
#else
        int mask_index = mad24(id / cols, mask_step, id % cols);
#endif
        if (mask[mask_index])
#endif
        {
#ifdef HAVE_SRC_CONT
            int src_index = mul24(id, srcTSIZE);
#else
            int src_index = mad24(id / cols, src_step, mul24(id % cols, srcTSIZE));
#endif

            srcT value = loadpix(srcptr + src_index);
            accSum += convertToDT(value);
            sqdstT dvalue = convertToSDT(value);
            accSqSum = fma(dvalue, dvalue, accSqSum);

#ifdef HAVE_MASK
            ++accNonZero;
#endif
        }
    }

    if (lid < WGS2_ALIGNED)
    {
        localMemSum[lid] = accSum;
        localMemSqSum[lid] = accSqSum;
#ifdef HAVE_MASK
        localMemNonZero[lid] = accNonZero;
#endif
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid >= WGS2_ALIGNED && total >= WGS2_ALIGNED)
    {
        localMemSum[lid - WGS2_ALIGNED] += accSum;
        localMemSqSum[lid - WGS2_ALIGNED] += accSqSum;
#ifdef HAVE_MASK
        localMemNonZero[lid - WGS2_ALIGNED] += accNonZero;
#endif
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int lsize = WGS2_ALIGNED >> 1; lsize > 0; lsize >>= 1)
    {
        if (lid < lsize)
        {
            int lid2 = lsize + lid;
            localMemSum[lid] += localMemSum[lid2];
            localMemSqSum[lid] += localMemSqSum[lid2];
#ifdef HAVE_MASK
            localMemNonZero[lid] += localMemNonZero[lid2];
#endif
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0)
    {
        storepix(localMemSum[0], dstptr + dstTSIZE * gid);
        storesqpix(localMemSqSum[0], dstptr + mad24(dstTSIZE, groups, sqdstTSIZE * gid));
#ifdef HAVE_MASK
        *(__global int *)(dstptr + mad24(dstTSIZE + sqdstTSIZE, groups, (int)sizeof(int) * gid)) = localMemNonZero[0];
#endif
    }
}
