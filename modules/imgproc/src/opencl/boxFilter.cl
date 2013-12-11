/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors as is and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

///////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////Macro for border type////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef BORDER_REPLICATE
//BORDER_REPLICATE:     aaaaaa|abcdefgh|hhhhhhh
#define ADDR_L(i, l_edge, r_edge)  ((i) <  (l_edge) ? (l_edge)   : (i))
#define ADDR_R(i, r_edge, addr)    ((i) >= (r_edge) ? (r_edge)-1 : (addr))
#define ADDR_H(i, t_edge, b_edge)  ((i) <  (t_edge) ? (t_edge)   :(i))
#define ADDR_B(i, b_edge, addr)    ((i) >= (b_edge) ? (b_edge)-1 :(addr))
#endif

#ifdef BORDER_REFLECT
//BORDER_REFLECT:       fedcba|abcdefgh|hgfedcb
#define ADDR_L(i, l_edge, r_edge)  ((i) <  (l_edge) ? -(i)-1               : (i))
#define ADDR_R(i, r_edge, addr)    ((i) >= (r_edge) ? -(i)-1+((r_edge)<<1) : (addr))
#define ADDR_H(i, t_edge, b_edge)  ((i) <  (t_edge) ? -(i)-1 : (i))
#define ADDR_B(i, b_edge, addr)    ((i) >= (b_edge) ? -(i)-1+((b_edge)<<1) : (addr))
#endif

#ifdef BORDER_REFLECT_101
//BORDER_REFLECT_101:   gfedcb|abcdefgh|gfedcba
#define ADDR_L(i, l_edge, r_edge)  ((i) <  (l_edge) ? -(i)                 : (i))
#define ADDR_R(i, r_edge, addr)    ((i) >= (r_edge) ? -(i)-2+((r_edge)<<1) : (addr))
#define ADDR_H(i, t_edge, b_edge)  ((i) <  (t_edge) ? -(i)                 : (i))
#define ADDR_B(i, b_edge, addr)    ((i) >= (b_edge) ? -(i)-2+((b_edge)<<1) : (addr))
#endif

//blur function does not support BORDER_WRAP
#ifdef BORDER_WRAP
//BORDER_WRAP:          cdefgh|abcdefgh|abcdefg
#define ADDR_L(i, l_edge, r_edge)  ((i) <  (l_edge) ? (i)+(r_edge) : (i))
#define ADDR_R(i, r_edge, addr)    ((i) >= (r_edge) ? (i)-(r_edge) : (addr))
#define ADDR_H(i, t_edge, b_edge)  ((i) <  (t_edge) ? (i)+(b_edge) : (i))
#define ADDR_B(i, b_edge, addr)    ((i) >= (b_edge) ? (i)-(b_edge) : (addr))
#endif

#ifdef EXTRA_EXTRAPOLATION // border > src image size
#ifdef BORDER_CONSTANT
// None
#elif defined BORDER_REPLICATE
#define EXTRAPOLATE(x, y, minX, minY, maxX, maxY) \
    { \
        x = max(min(x, maxX - 1), minX); \
        y = max(min(y, maxY - 1), minY); \
    }
#elif defined BORDER_WRAP
#define EXTRAPOLATE(x, y, minX, minY, maxX, maxY) \
    { \
        if (x < minX) \
            x -= ((x - maxX + 1) / maxX) * maxX; \
        if (x >= maxX) \
            x %= maxX; \
        if (y < minY) \
            y -= ((y - maxY + 1) / maxY) * maxY; \
        if (y >= maxY) \
            y %= maxY; \
    }
#elif defined(BORDER_REFLECT) || defined(BORDER_REFLECT_101)
#define EXTRAPOLATE_(x, y, minX, minY, maxX, maxY, delta) \
    { \
        if (maxX - minX == 1) \
            x = minX; \
        else \
            do \
            { \
                if (x < minX) \
                    x = minX - (x - minX) - 1 + delta; \
                else \
                    x = maxX - 1 - (x - maxX) - delta; \
            } \
            while (x >= maxX || x < minX); \
        \
        if (maxY - minY == 1) \
            y = minY; \
        else \
            do \
            { \
                if (y < minY) \
                    y = minY - (y - minY) - 1 + delta; \
                else \
                    y = maxY - 1 - (y - maxY) - delta; \
            } \
            while (y >= maxY || y < minY); \
    }
#ifdef BORDER_REFLECT
#define EXTRAPOLATE(x, y, minX, minY, maxX, maxY) EXTRAPOLATE_(x, y, minX, minY, maxX, maxY, 0)
#elif defined(BORDER_REFLECT_101)
#define EXTRAPOLATE(x, y, minX, minY, maxX, maxY) EXTRAPOLATE_(x, y, minX, minY, maxX, maxY, 1)
#endif
#else
#error No extrapolation method
#endif
#else
#define EXTRAPOLATE(x, y, minX, minY, maxX, maxY) \
    { \
        int _row = y - minY, _col = x - minX; \
        _row = ADDR_H(_row, 0, maxY - minY); \
        _row = ADDR_B(_row, maxY - minY, _row); \
        y = _row + minY; \
        \
        _col = ADDR_L(_col, 0, maxX - minX); \
        _col = ADDR_R(_col, maxX - minX, _col); \
        x = _col + minX; \
    }
#endif

#if USE_DOUBLE
#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#elif defined (cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif
#define FPTYPE double
#define CONVERT_TO_FPTYPE CAT(convert_double, VEC_SIZE)
#else
#define FPTYPE float
#define CONVERT_TO_FPTYPE CAT(convert_float, VEC_SIZE)
#endif

#if DATA_DEPTH == 0
#define BASE_TYPE uchar
#elif DATA_DEPTH == 1
#define BASE_TYPE char
#elif DATA_DEPTH == 2
#define BASE_TYPE ushort
#elif DATA_DEPTH == 3
#define BASE_TYPE short
#elif DATA_DEPTH == 4
#define BASE_TYPE int
#elif DATA_DEPTH == 5
#define BASE_TYPE float
#elif DATA_DEPTH == 6
#define BASE_TYPE double
#else
#error data_depth
#endif

#define __CAT(x, y) x##y
#define CAT(x, y) __CAT(x, y)

#define uchar1 uchar
#define char1 char
#define ushort1 ushort
#define short1 short
#define int1 int
#define float1 float
#define double1 double

#define convert_uchar1_sat_rte convert_uchar_sat_rte
#define convert_char1_sat_rte convert_char_sat_rte
#define convert_ushort1_sat_rte convert_ushort_sat_rte
#define convert_short1_sat_rte convert_short_sat_rte
#define convert_int1_sat_rte convert_int_sat_rte
#define convert_float1
#define convert_double1

#if DATA_DEPTH == 5 || DATA_DEPTH == 6
#define CONVERT_TO_TYPE CAT(CAT(convert_, BASE_TYPE), VEC_SIZE)
#else
#define CONVERT_TO_TYPE CAT(CAT(CAT(convert_, BASE_TYPE), VEC_SIZE), _sat_rte)
#endif

#define VEC_SIZE DATA_CHAN

#define VEC_TYPE CAT(BASE_TYPE, VEC_SIZE)
#define TYPE VEC_TYPE

#define SCALAR_TYPE CAT(FPTYPE, VEC_SIZE)

#define INTERMEDIATE_TYPE CAT(FPTYPE, VEC_SIZE)

#define TYPE_SIZE   (VEC_SIZE*sizeof(BASE_TYPE))

struct RectCoords
{
    int x1, y1, x2, y2;
};

//#define DEBUG
#ifdef DEBUG
#define DEBUG_ONLY(x) x
#define ASSERT(condition) do { if (!(condition)) { printf("BUG in boxFilter kernel (global=%d,%d): " #condition "\n", get_global_id(0), get_global_id(1)); } } while (0)
#else
#define DEBUG_ONLY(x)
#define ASSERT(condition)
#endif


inline INTERMEDIATE_TYPE readSrcPixel(int2 pos, __global const uchar* srcptr, int srcstep, const struct RectCoords srcCoords
#ifdef BORDER_CONSTANT
               , SCALAR_TYPE borderValue
#endif
    )
{
#ifdef BORDER_ISOLATED
    if(pos.x >= srcCoords.x1 && pos.y >= srcCoords.y1 && pos.x < srcCoords.x2 && pos.y < srcCoords.y2)
#else
    if(pos.x >= 0 && pos.y >= 0 && pos.x < srcCoords.x2 && pos.y < srcCoords.y2)
#endif
    {
        __global TYPE* ptr = (__global TYPE*)(srcptr + pos.y * srcstep + pos.x * sizeof(TYPE));
        return CONVERT_TO_FPTYPE(*ptr);
    }
    else
    {
#ifdef BORDER_CONSTANT
        return borderValue;
#else
        int selected_col = pos.x;
        int selected_row = pos.y;

        EXTRAPOLATE(selected_col, selected_row,
#ifdef BORDER_ISOLATED
                srcCoords.x1, srcCoords.y1,
#else
                0, 0,
#endif
                srcCoords.x2, srcCoords.y2
         );

        // debug border mapping
        //printf("pos=%d,%d --> %d, %d\n", pos.x, pos.y, selected_col, selected_row);

        pos = (int2)(selected_col, selected_row);
        if(pos.x >= 0 && pos.y >= 0 && pos.x < srcCoords.x2 && pos.y < srcCoords.y2)
        {
            __global TYPE* ptr = (__global TYPE*)(srcptr + pos.y * srcstep + pos.x * sizeof(TYPE));
            return CONVERT_TO_FPTYPE(*ptr);
        }
        else
        {
            // for debug only
            DEBUG_ONLY(printf("BUG in boxFilter kernel\n"));
            return (FPTYPE)(0.0f);
        }
#endif
    }
}

// INPUT PARAMETER: BLOCK_SIZE_Y (via defines)

__kernel
__attribute__((reqd_work_group_size(LOCAL_SIZE, 1, 1)))
void boxFilter(__global const uchar* srcptr, int srcstep, int srcOffsetX, int srcOffsetY, int srcEndX, int srcEndY,
                __global uchar* dstptr, int dststep, int dstoffset,
               int rows, int cols,
#ifdef BORDER_CONSTANT
               SCALAR_TYPE borderValue,
#endif
               FPTYPE alpha
               )
{
    const struct RectCoords srcCoords = {srcOffsetX, srcOffsetY, srcEndX, srcEndY}; // for non-isolated border: offsetX, offsetY, wholeX, wholeY

    const int x = get_local_id(0) + (LOCAL_SIZE - (KERNEL_SIZE_X - 1)) * get_group_id(0) - ANCHOR_X;
    const int y = get_global_id(1) * BLOCK_SIZE_Y;

    const int local_id = get_local_id(0);

    INTERMEDIATE_TYPE data[KERNEL_SIZE_Y];
    __local INTERMEDIATE_TYPE sumOfCols[LOCAL_SIZE];

    int2 srcPos = (int2)(srcCoords.x1 + x, srcCoords.y1 + y - ANCHOR_Y);
    for(int sy = 0; sy < KERNEL_SIZE_Y; sy++, srcPos.y++)
    {
        data[sy] = readSrcPixel(srcPos, srcptr, srcstep, srcCoords
#ifdef BORDER_CONSTANT
                , borderValue
#endif
                );
    }

    INTERMEDIATE_TYPE tmp_sum = 0;
    for(int sy = 0; sy < KERNEL_SIZE_Y; sy++)
    {
        tmp_sum += (data[sy]);
    }

    sumOfCols[local_id] = tmp_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    int2 pos = (int2)(x, y);
    __global TYPE* dstPtr = (__global TYPE*)(dstptr + pos.y * dststep + dstoffset + pos.x * TYPE_SIZE/*sizeof(TYPE)*/); // Pointer can be out of bounds!

    int sy_index = 0; // current index in data[] array
    int stepsY = min(rows - pos.y, BLOCK_SIZE_Y);
    ASSERT(stepsY > 0);
    for (; ;)
    {
        ASSERT(pos.y < rows);

        if(local_id >= ANCHOR_X && local_id < LOCAL_SIZE - (KERNEL_SIZE_X - 1 - ANCHOR_X) &&
            pos.x >= 0 && pos.x < cols)
        {
            ASSERT(pos.y >= 0 && pos.y < rows);

            INTERMEDIATE_TYPE total_sum = 0;
#pragma unroll
            for (int sx = 0; sx < KERNEL_SIZE_X; sx++)
            {
                total_sum += sumOfCols[local_id + sx - ANCHOR_X];
            }
            *dstPtr = CONVERT_TO_TYPE(((INTERMEDIATE_TYPE)alpha) * total_sum);
        }

#if BLOCK_SIZE_Y == 1
        break;
#else
        if (--stepsY == 0)
            break;

        barrier(CLK_LOCAL_MEM_FENCE);

        tmp_sum = sumOfCols[local_id]; // TODO FIX IT: workaround for BUG in OpenCL compiler
        // only works with scalars: ASSERT(fabs(tmp_sum - sumOfCols[local_id]) < (INTERMEDIATE_TYPE)1e-6);
        tmp_sum -= data[sy_index];

        data[sy_index] = readSrcPixel(srcPos, srcptr, srcstep, srcCoords
#ifdef BORDER_CONSTANT
                , borderValue
#endif
                );
        srcPos.y++;

        tmp_sum += data[sy_index];
        sumOfCols[local_id] = tmp_sum;

        sy_index = (sy_index + 1 < KERNEL_SIZE_Y) ? sy_index + 1 : 0;

        barrier(CLK_LOCAL_MEM_FENCE);

        // next line
        DEBUG_ONLY(pos.y++);
        dstPtr = (__global TYPE*)((__global char*)dstPtr + dststep); // Pointer can be out of bounds!
#endif // BLOCK_SIZE_Y == 1
    }
}
