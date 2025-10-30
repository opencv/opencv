// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2014, Itseez, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.


#define noconvert

#ifdef ONLY_SUM_CONVERT

__kernel void sumConvert(__global const uchar * src1ptr, int src1_step, int src1_offset,
                         __global const uchar * src2ptr, int src2_step, int src2_offset,
                         __global uchar * dstptr, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                         COEFF_T scale, COEFF_T delta)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (y < dst_rows && x < dst_cols)
    {
        int src1_index = mad24(y, src1_step, mad24(x, (int)sizeof(SRC_T), src1_offset));
        int src2_index = mad24(y, src2_step, mad24(x, (int)sizeof(SRC_T), src2_offset));
        int dst_index = mad24(y, dst_step, mad24(x, (int)sizeof(DST_T), dst_offset));

        __global const SRC_T * src1 = (__global const SRC_T *)(src1ptr + src1_index);
        __global const SRC_T * src2 = (__global const SRC_T *)(src2ptr + src2_index);
        __global DST_T * dst = (__global DST_T *)(dstptr + dst_index);

#if WDEPTH <= 4
        dst[0] = CONVERT_TO_DT( mad24((WT)(scale), CONVERT_TO_WT(src1[0]) + CONVERT_TO_WT(src2[0]), (WT)(delta)) );
#else
        dst[0] = CONVERT_TO_DT( mad((WT)(scale), CONVERT_TO_WT(src1[0]) + CONVERT_TO_WT(src2[0]), (WT)(delta)) );
#endif
    }
}

#else

///////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////Macro for border type////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef BORDER_CONSTANT
// CCCCCC|abcdefgh|CCCCCCC
#define EXTRAPOLATE(x, maxV)
#elif defined BORDER_REPLICATE
// aaaaaa|abcdefgh|hhhhhhh
#define EXTRAPOLATE(x, maxV) \
    { \
        (x) = clamp((x), 0, (maxV)-1); \
    }
#elif defined BORDER_WRAP
// cdefgh|abcdefgh|abcdefg
#define EXTRAPOLATE(x, maxV) \
    { \
        (x) = ( (x) + (maxV) ) % (maxV); \
    }
#elif defined BORDER_REFLECT
// fedcba|abcdefgh|hgfedcb
#define EXTRAPOLATE(x, maxV) \
    { \
        (x) = min(((maxV)-1)*2-(x)+1, max((x),-(x)-1) ); \
    }
#elif defined BORDER_REFLECT_101
// gfedcb|abcdefgh|gfedcba
#define EXTRAPOLATE(x, maxV) \
    { \
        (x) = min(((maxV)-1)*2-(x), max((x),-(x)) ); \
    }
#else
#error No extrapolation method
#endif

#if CN != 3
#define loadpix(addr) *(__global const SRC_T *)(addr)
#define storepix(val, addr)  *(__global DST_T *)(addr) = val
#define SRCSIZE (int)sizeof(SRC_T)
#define DSTSIZE (int)sizeof(DST_T)
#else
#define loadpix(addr)  vload3(0, (__global const SRC_T1 *)(addr))
#define storepix(val, addr) vstore3(val, 0, (__global DST_T1 *)(addr))
#define SRCSIZE (int)sizeof(SRC_T1)*3
#define DSTSIZE (int)sizeof(DST_T1)*3
#endif

#define SRC(_x,_y) CONVERT_TO_WT(loadpix(Src + mad24(_y, src_step, SRCSIZE * _x)))

#ifdef BORDER_CONSTANT
// CCCCCC|abcdefgh|CCCCCCC
#define ELEM(_x,_y,r_edge,t_edge,const_v) (_x)<0 | (_x) >= (r_edge) | (_y)<0 | (_y) >= (t_edge) ? (const_v) : SRC((_x),(_y))
#else
#define ELEM(_x,_y,r_edge,t_edge,const_v) SRC((_x),(_y))
#endif

// horizontal and vertical filter kernels
// should be defined on host during compile time to avoid overhead
#define DIG(a) a,
__constant WT1 mat_kernelX[] = { KERNEL_MATRIX_X };
__constant WT1 mat_kernelY[] = { KERNEL_MATRIX_Y };

__kernel void laplacian(__global uchar* Src, int src_step, int srcOffsetX, int srcOffsetY, int height, int width,
                         __global uchar* Dst, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                         WT1 scale, WT1 delta)
{
    __local WT lsmem[BLK_Y + 2 * RADIUS][BLK_X + 2 * RADIUS];
    __local WT lsmemDy1[BLK_Y][BLK_X + 2 * RADIUS];
    __local WT lsmemDy2[BLK_Y][BLK_X + 2 * RADIUS];

    int lix = get_local_id(0);
    int liy = get_local_id(1);

    int x = get_global_id(0);

    int srcX = x + srcOffsetX - RADIUS;

    int clocY = liy;
    do
    {
        int yb = clocY + srcOffsetY - RADIUS;
        EXTRAPOLATE(yb, (height));

        int clocX = lix;
        int cSrcX = srcX;
        do
        {
            int xb = cSrcX;
            EXTRAPOLATE(xb,(width));
            lsmem[clocY][clocX] = ELEM(xb, yb, (width), (height), 0 );

            clocX += BLK_X;
            cSrcX += BLK_X;
        }
        while(clocX < BLK_X+(RADIUS*2));

        clocY += BLK_Y;
    }
    while (clocY < BLK_Y+(RADIUS*2));
    barrier(CLK_LOCAL_MEM_FENCE);

    WT scale_v = (WT)scale;
    WT delta_v = (WT)delta;
    for (int y = 0; y < dst_rows; y+=BLK_Y)
    {
        int i, clocX = lix;
        WT sum1 = (WT) 0;
        WT sum2 = (WT) 0;
        do
        {
            sum1 = (WT) 0;
            sum2 = (WT) 0;
            for (i=0; i<=2*RADIUS; i++)
            {
                sum1 = mad(lsmem[liy + i][clocX], mat_kernelY[i], sum1);
                sum2 = mad(lsmem[liy + i][clocX], mat_kernelX[i], sum2);
            }
            lsmemDy1[liy][clocX] = sum1;
            lsmemDy2[liy][clocX] = sum2;
            clocX += BLK_X;
        }
        while(clocX < BLK_X+(RADIUS*2));
        barrier(CLK_LOCAL_MEM_FENCE);

        if ((x < dst_cols) && (y + liy < dst_rows))
        {
            sum1 = (WT) 0;
            sum2 = (WT) 0;
            for (i=0; i<=2*RADIUS; i++)
            {
                sum1 = mad(lsmemDy1[liy][lix+i], mat_kernelX[i], sum1);
                sum2 = mad(lsmemDy2[liy][lix+i], mat_kernelY[i], sum2);
            }

            WT sum = mad(scale_v, (sum1 + sum2), delta_v);
            storepix(CONVERT_TO_DT(sum), Dst + mad24(y + liy, dst_step, mad24(x, DSTSIZE, dst_offset)));
        }

        for (int i = liy * BLK_X + lix; i < (RADIUS*2) * (BLK_X+(RADIUS*2)); i += BLK_X * BLK_Y)
        {
            int clocX = i % (BLK_X+(RADIUS*2));
            int clocY = i / (BLK_X+(RADIUS*2));
            lsmem[clocY][clocX] = lsmem[clocY + BLK_Y][clocX];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        int yb = y + liy + BLK_Y + srcOffsetY + RADIUS;
        EXTRAPOLATE(yb, (height));

        clocX = lix;
        int cSrcX = x + srcOffsetX - RADIUS;
        do
        {
            int xb = cSrcX;
            EXTRAPOLATE(xb,(width));
            lsmem[liy + 2*RADIUS][clocX] = ELEM(xb, yb, (width), (height), 0 );

            clocX += BLK_X;
            cSrcX += BLK_X;
        }
        while(clocX < BLK_X+(RADIUS*2));
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

#endif
