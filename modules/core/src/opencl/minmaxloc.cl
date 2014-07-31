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

#ifdef DEPTH_0
#define MIN_VAL 0
#define MAX_VAL UCHAR_MAX
#elif defined DEPTH_1
#define MIN_VAL SCHAR_MIN
#define MAX_VAL SCHAR_MAX
#elif defined DEPTH_2
#define MIN_VAL 0
#define MAX_VAL USHRT_MAX
#elif defined DEPTH_3
#define MIN_VAL SHRT_MIN
#define MAX_VAL SHRT_MAX
#elif defined DEPTH_4
#define MIN_VAL INT_MIN
#define MAX_VAL INT_MAX
#elif defined DEPTH_5
#define MIN_VAL (-FLT_MAX)
#define MAX_VAL FLT_MAX
#elif defined DEPTH_6
#define MIN_VAL (-DBL_MAX)
#define MAX_VAL DBL_MAX
#endif

#define noconvert
#define INDEX_MAX UINT_MAX

#if wdepth <= 4
#define MIN_ABS(a) convertFromU(abs(a))
#define MIN_ABS2(a, b) convertFromU(abs_diff(a, b))
#define MIN(a, b) min(a, b)
#define MAX(a, b) max(a, b)
#else
#define MIN_ABS(a) fabs(a)
#define MIN_ABS2(a, b) fabs(a - b)
#define MIN(a, b) fmin(a, b)
#define MAX(a, b) fmax(a, b)
#endif

#if kercn != 3
#define loadpix(addr) *(__global const srcT *)(addr)
#define srcTSIZE (int)sizeof(srcT)
#else
#define loadpix(addr) vload3(0, (__global const srcT1 *)(addr))
#define srcTSIZE ((int)sizeof(srcT1) * 3)
#endif

#ifndef HAVE_MASK
#undef srcTSIZE
#define srcTSIZE (int)sizeof(srcT1)
#endif

#ifdef NEED_MINVAL
#ifdef NEED_MINLOC
#define CALC_MIN(p, inc) \
    if (minval > temp.p) \
    { \
        minval = temp.p; \
        minloc = id + inc; \
    }
#else
#define CALC_MIN(p, inc) \
    minval = MIN(minval, temp.p);
#endif
#else
#define CALC_MIN(p, inc)
#endif

#ifdef NEED_MAXVAL
#ifdef NEED_MAXLOC
#define CALC_MAX(p, inc) \
    if (maxval < temp.p) \
    { \
        maxval = temp.p; \
        maxloc = id + inc; \
    }
#else
#define CALC_MAX(p, inc) \
    maxval = MAX(maxval, temp.p);
#endif
#else
#define CALC_MAX(p, inc)
#endif

#ifdef OP_CALC2
#define CALC_MAX2(p) \
    maxval2 = MAX(maxval2, temp.p);
#else
#define CALC_MAX2(p)
#endif

#define CALC_P(p, inc) \
    CALC_MIN(p, inc) \
    CALC_MAX(p, inc) \
    CALC_MAX2(p)

__kernel void minmaxloc(__global const uchar * srcptr, int src_step, int src_offset, int cols,
                        int total, int groupnum, __global uchar * dstptr
#ifdef HAVE_MASK
                        , __global const uchar * mask, int mask_step, int mask_offset
#endif
#ifdef HAVE_SRC2
                        , __global const uchar * src2ptr, int src2_step, int src2_offset
#endif
                        )
{
    int lid = get_local_id(0);
    int gid = get_group_id(0);
    int  id = get_global_id(0)
#ifndef HAVE_MASK
    * kercn;
#else
    ;
#endif

    srcptr += src_offset;
#ifdef HAVE_MASK
    mask += mask_offset;
#endif
#ifdef HAVE_SRC2
    src2ptr += src2_offset;
#endif

#ifdef NEED_MINVAL
    __local dstT1 localmem_min[WGS2_ALIGNED];
    dstT1 minval = MAX_VAL;
#ifdef NEED_MINLOC
    __local uint localmem_minloc[WGS2_ALIGNED];
    uint minloc = INDEX_MAX;
#endif
#endif
#ifdef NEED_MAXVAL
    dstT1 maxval = MIN_VAL;
    __local dstT1 localmem_max[WGS2_ALIGNED];
#ifdef NEED_MAXLOC
    __local uint localmem_maxloc[WGS2_ALIGNED];
    uint maxloc = INDEX_MAX;
#endif
#endif
#ifdef OP_CALC2
    __local dstT1 localmem_max2[WGS2_ALIGNED];
    dstT1 maxval2 = MIN_VAL;
#endif

    int src_index;
#ifdef HAVE_MASK
    int mask_index;
#endif
#ifdef HAVE_SRC2
    int src2_index;
#endif

    dstT temp;
#ifdef HAVE_SRC2
    dstT temp2;
#endif

    for (int grain = groupnum * WGS
#ifndef HAVE_MASK
        * kercn
#endif
        ; id < total; id += grain)
    {
#ifdef HAVE_MASK
#ifdef HAVE_MASK_CONT
        mask_index = id;
#else
        mask_index = mad24(id / cols, mask_step, id % cols);
#endif
        if (mask[mask_index])
#endif
        {
#ifdef HAVE_SRC_CONT
            src_index = mul24(id, srcTSIZE);
#else
            src_index = mad24(id / cols, src_step, mul24(id % cols, srcTSIZE));
#endif
            temp = convertToDT(loadpix(srcptr + src_index));
#ifdef OP_ABS
            temp = MIN_ABS(temp);
#endif

#ifdef HAVE_SRC2
#ifdef HAVE_SRC2_CONT
            src2_index = mul24(id, srcTSIZE);
#else
            src2_index = mad24(id / cols, src2_step, mul24(id % cols, srcTSIZE));
#endif
            temp2 = convertToDT(loadpix(src2ptr + src2_index));
            temp = MIN_ABS2(temp, temp2);
#ifdef OP_CALC2
            temp2 = MIN_ABS(temp2);
#endif
#endif

#if kercn == 1
#ifdef NEED_MINVAL
#if NEED_MINLOC
            if (minval > temp)
            {
                minval = temp;
                minloc = id;
            }
#else
            minval = MIN(minval, temp);
#endif
#endif
#ifdef NEED_MAXVAL
#ifdef NEED_MAXLOC
            if (maxval < temp)
            {
                maxval = temp;
                maxloc = id;
            }
#else
            maxval = MAX(maxval, temp);
#endif
#ifdef OP_CALC2
            maxval2 = MAX(maxval2, temp2);
#endif
#endif
#elif kercn >= 2
            CALC_P(s0, 0)
            CALC_P(s1, 1)
#if kercn >= 3
            CALC_P(s2, 2)
#if kercn >= 4
            CALC_P(s3, 3)
#if kercn >= 8
            CALC_P(s4, 4)
            CALC_P(s5, 5)
            CALC_P(s6, 6)
            CALC_P(s7, 7)
#if kercn == 16
            CALC_P(s8, 8)
            CALC_P(s9, 9)
            CALC_P(sA, 10)
            CALC_P(sB, 11)
            CALC_P(sC, 12)
            CALC_P(sD, 13)
            CALC_P(sE, 14)
            CALC_P(sF, 15)
#endif
#endif
#endif
#endif
#endif
        }
    }

    if (lid < WGS2_ALIGNED)
    {
#ifdef NEED_MINVAL
        localmem_min[lid] = minval;
#endif
#ifdef NEED_MAXVAL
        localmem_max[lid] = maxval;
#endif
#ifdef NEED_MINLOC
        localmem_minloc[lid] = minloc;
#endif
#ifdef NEED_MAXLOC
        localmem_maxloc[lid] = maxloc;
#endif
#ifdef OP_CALC2
        localmem_max2[lid] = maxval2;
#endif
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid >= WGS2_ALIGNED && total >= WGS2_ALIGNED)
    {
        int lid3 = lid - WGS2_ALIGNED;
#ifdef NEED_MINVAL
#ifdef NEED_MINLOC
        if (localmem_min[lid3] >= minval)
        {
            if (localmem_min[lid3] == minval)
                localmem_minloc[lid3] = min(localmem_minloc[lid3], minloc);
            else
                localmem_minloc[lid3] = minloc,
            localmem_min[lid3] = minval;
        }
#else
        localmem_min[lid3] = MIN(localmem_min[lid3], minval);
#endif
#endif
#ifdef NEED_MAXVAL
#ifdef NEED_MAXLOC
        if (localmem_max[lid3] <= maxval)
        {
            if (localmem_max[lid3] == maxval)
                localmem_maxloc[lid3] = min(localmem_maxloc[lid3], maxloc);
            else
                localmem_maxloc[lid3] = maxloc,
            localmem_max[lid3] = maxval;
        }
#else
        localmem_max[lid3] = MAX(localmem_max[lid3], maxval);
#endif
#endif
#ifdef OP_CALC2
        localmem_max2[lid3] = MAX(localmem_max2[lid3], maxval2);
#endif
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int lsize = WGS2_ALIGNED >> 1; lsize > 0; lsize >>= 1)
    {
        if (lid < lsize)
        {
            int lid2 = lsize + lid;

#ifdef NEED_MINVAL
#ifdef NEED_MAXLOC
            if (localmem_min[lid] >= localmem_min[lid2])
            {
                if (localmem_min[lid] == localmem_min[lid2])
                    localmem_minloc[lid] = min(localmem_minloc[lid2], localmem_minloc[lid]);
                else
                    localmem_minloc[lid] = localmem_minloc[lid2],
                localmem_min[lid] = localmem_min[lid2];
            }
#else
            localmem_min[lid] = MIN(localmem_min[lid], localmem_min[lid2]);
#endif
#endif
#ifdef NEED_MAXVAL
#ifdef NEED_MAXLOC
            if (localmem_max[lid] <= localmem_max[lid2])
            {
                if (localmem_max[lid] == localmem_max[lid2])
                    localmem_maxloc[lid] = min(localmem_maxloc[lid2], localmem_maxloc[lid]);
                else
                    localmem_maxloc[lid] = localmem_maxloc[lid2],
                localmem_max[lid] = localmem_max[lid2];
            }
#else
            localmem_max[lid] = MAX(localmem_max[lid], localmem_max[lid2]);
#endif
#endif
#ifdef OP_CALC2
            localmem_max2[lid] = MAX(localmem_max2[lid], localmem_max2[lid2]);
#endif
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0)
    {
        int pos = 0;
#ifdef NEED_MINVAL
        *(__global dstT1 *)(dstptr + mad24(gid, (int)sizeof(dstT1), pos)) = localmem_min[0];
        pos = mad24(groupnum, (int)sizeof(dstT1), pos);
#endif
#ifdef NEED_MAXVAL
        *(__global dstT1 *)(dstptr + mad24(gid, (int)sizeof(dstT1), pos)) = localmem_max[0];
        pos = mad24(groupnum, (int)sizeof(dstT1), pos);
#endif
#ifdef NEED_MINLOC
        *(__global uint *)(dstptr + mad24(gid, (int)sizeof(uint), pos)) = localmem_minloc[0];
        pos = mad24(groupnum, (int)sizeof(uint), pos);
#endif
#ifdef NEED_MAXLOC
        *(__global uint *)(dstptr + mad24(gid, (int)sizeof(uint), pos)) = localmem_maxloc[0];
#ifdef OP_CALC2
        pos = mad24(groupnum, (int)sizeof(uint), pos);
#endif
#endif
#ifdef OP_CALC2
        *(__global dstT1 *)(dstptr + mad24(gid, (int)sizeof(dstT1), pos)) = localmem_max2[0];
#endif
    }
}
