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
// Copyright (C) 2014, Intel Corporation, all rights reserved.
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
// This software is provided by the copyright holders and contributors "as is" and
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
#elif defined BORDER_REFLECT_101 || defined BORDER_REFLECT101
// gfedcb|abcdefgh|gfedcba
#define EXTRAPOLATE(x, maxV) \
    { \
        (x) = min(((maxV)-1)*2-(x), max((x),-(x)) ); \
    }
#else
#error No extrapolation method
#endif

#if CN != 3
#define loadpix(addr) *(__global const srcT *)(addr)
#define storepix(val, addr)  *(__global dstT *)(addr) = val
#define SRCSIZE (int)sizeof(srcT)
#define DSTSIZE (int)sizeof(dstT)
#else
#define loadpix(addr)  vload3(0, (__global const srcT1 *)(addr))
#define storepix(val, addr) vstore3(val, 0, (__global dstT1 *)(addr))
#define SRCSIZE (int)sizeof(srcT1)*3
#define DSTSIZE (int)sizeof(dstT1)*3
#endif

#define SRC(_x,_y) convertToWT(loadpix(Src + mad24(_y, src_step, SRCSIZE * _x)))

#ifdef BORDER_CONSTANT
// CCCCCC|abcdefgh|CCCCCCC
#define ELEM(_x,_y,r_edge,t_edge,const_v) (_x)<0 | (_x) >= (r_edge) | (_y)<0 | (_y) >= (t_edge) ? (const_v) : SRC((_x),(_y))
#else
#define ELEM(_x,_y,r_edge,t_edge,const_v) SRC((_x),(_y))
#endif

#define noconvert

// horizontal and vertical filter kernels
// should be defined on host during compile time to avoid overhead
#define DIG(a) a,
__constant WT1 mat_kernelX[] = { KERNEL_MATRIX_X };
__constant WT1 mat_kernelY[] = { KERNEL_MATRIX_Y };

__kernel void sep_filter(__global uchar* Src, int src_step, int srcOffsetX, int srcOffsetY, int height, int width,
                         __global uchar* Dst, int dst_step, int dst_offset, int dst_rows, int dst_cols, float delta)
{
    // RADIUSX, RADIUSY are filter dimensions
    // BLK_X, BLK_Y are local wrogroup sizes
    // all these should be defined on host during compile time
    // first lsmem array for source pixels used in first pass,
    // second lsmemDy for storing first pass results
    __local WT lsmem[BLK_Y + 2 * RADIUSY][BLK_X + 2 * RADIUSX];
    __local WT lsmemDy[BLK_Y][BLK_X + 2 * RADIUSX];

    // get local and global ids - used as image and local memory array indexes
    int lix = get_local_id(0);
    int liy = get_local_id(1);

    int x = get_global_id(0);

    // calculate pixel position in source image taking image offset into account
    int srcX = x + srcOffsetX - RADIUSX;

    // extrapolate coordinates, if needed
    // and read my own source pixel into local memory
    // with account for extra border pixels, which will be read by starting workitems
    int clocY = liy;
    do
    {
        int yb = clocY + srcOffsetY - RADIUSY;
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
        while(clocX < BLK_X+(RADIUSX*2));

        clocY += BLK_Y;
    }
    while (clocY < BLK_Y+(RADIUSY*2));
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int y = 0; y < dst_rows; y+=BLK_Y)
    {
        // do vertical filter pass
        // and store intermediate results to second local memory array
        int i, clocX = lix;
        WT sum = (WT) 0;
        do
        {
            sum = (WT) 0;
            for (i=0; i<=2*RADIUSY; i++)
#if (defined(INTEGER_ARITHMETIC) && !INTEL_DEVICE)
                sum = mad24(lsmem[liy + i][clocX], mat_kernelY[i], sum);
#else
                sum = mad(lsmem[liy + i][clocX], mat_kernelY[i], sum);
#endif
            lsmemDy[liy][clocX] = sum;
            clocX += BLK_X;
        }
        while(clocX < BLK_X+(RADIUSX*2));
        barrier(CLK_LOCAL_MEM_FENCE);

        // if this pixel happened to be out of image borders because of global size rounding,
        // then just return
        if ((x < dst_cols) && (y + liy < dst_rows))
        {
            // do second horizontal filter pass
            // and calculate final result
            sum = 0.0f;
            for (i=0; i<=2*RADIUSX; i++)
#if (defined(INTEGER_ARITHMETIC) && !INTEL_DEVICE)
                sum = mad24(lsmemDy[liy][lix+i], mat_kernelX[i], sum);
#else
                sum = mad(lsmemDy[liy][lix+i], mat_kernelX[i], sum);
#endif

#ifdef INTEGER_ARITHMETIC
#ifdef INTEL_DEVICE
            sum = (sum + (1 << (SHIFT_BITS-1))) / (1 << SHIFT_BITS);
#else
            sum = (sum + (1 << (SHIFT_BITS-1))) >> SHIFT_BITS;
#endif
#endif
            // store result into destination image
            storepix(convertToDstT(sum + (WT)(delta)), Dst + mad24(y + liy, dst_step, mad24(x, DSTSIZE, dst_offset)));
        }

        for (int i = liy * BLK_X + lix; i < (RADIUSY*2) * (BLK_X+(RADIUSX*2)); i += BLK_X * BLK_Y)
        {
            int clocX = i % (BLK_X+(RADIUSX*2));
            int clocY = i / (BLK_X+(RADIUSX*2));
            lsmem[clocY][clocX] = lsmem[clocY + BLK_Y][clocX];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        int yb = y + liy + BLK_Y + srcOffsetY + RADIUSY;
        EXTRAPOLATE(yb, (height));

        clocX = lix;
        int cSrcX = x + srcOffsetX - RADIUSX;
        do
        {
            int xb = cSrcX;
            EXTRAPOLATE(xb,(width));
            lsmem[liy + 2*RADIUSY][clocX] = ELEM(xb, yb, (width), (height), 0 );

            clocX += BLK_X;
            cSrcX += BLK_X;
        }
        while(clocX < BLK_X+(RADIUSX*2));
        barrier(CLK_LOCAL_MEM_FENCE);
    }

}
