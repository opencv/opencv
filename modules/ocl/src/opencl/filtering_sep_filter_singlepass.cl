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
// Copyright (C) 2013, Intel Corporation, all rights reserved.
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
//CCCCCC|abcdefgh|CCCCCCC
#define EXTRAPOLATE(x, maxV)
#elif defined BORDER_REPLICATE
//aaaaaa|abcdefgh|hhhhhhh
#define EXTRAPOLATE(x, maxV) \
    { \
        (x) = max(min((x), (maxV) - 1), 0); \
    }
#elif defined BORDER_WRAP
//cdefgh|abcdefgh|abcdefg
#define EXTRAPOLATE(x, maxV) \
    { \
        (x) = ( (x) + (maxV) ) % (maxV); \
    }
#elif defined BORDER_REFLECT
//fedcba|abcdefgh|hgfedcb
#define EXTRAPOLATE(x, maxV) \
    { \
        (x) = min(((maxV)-1)*2-(x)+1, max((x),-(x)-1) ); \
    }
#elif defined BORDER_REFLECT_101
//gfedcb|abcdefgh|gfedcba
#define EXTRAPOLATE(x, maxV) \
    { \
        (x) = min(((maxV)-1)*2-(x), max((x),-(x)) ); \
    }
#else
#error No extrapolation method
#endif

#define SRC(_x,_y) CONVERT_SRCTYPE(((global SRCTYPE*)(Src+(_y)*SrcPitch))[_x])

#ifdef BORDER_CONSTANT
//CCCCCC|abcdefgh|CCCCCCC
#define ELEM(_x,_y,r_edge,t_edge,const_v) (_x)<0 | (_x) >= (r_edge) | (_y)<0 | (_y) >= (t_edge) ? (const_v) : SRC((_x),(_y))
#else
#define ELEM(_x,_y,r_edge,t_edge,const_v) SRC((_x),(_y))
#endif

#define DST(_x,_y) (((global DSTTYPE*)(Dst+DstOffset+(_y)*DstPitch))[_x])

//horizontal and vertical filter kernels
//should be defined on host during compile time to avoid overhead
__constant uint mat_kernelX[] = {KERNEL_MATRIX_X};
__constant uint mat_kernelY[] = {KERNEL_MATRIX_Y};

__kernel __attribute__((reqd_work_group_size(BLK_X,BLK_Y,1))) void sep_filter_singlepass
        (
        __global uchar* Src,
        const uint      SrcPitch,
        const int       srcOffsetX,
        const int       srcOffsetY,
        __global uchar* Dst,
        const int       DstOffset,
        const uint      DstPitch,
        int             width,
        int             height,
        int             dstWidth,
        int             dstHeight
        )
{
    //RADIUSX, RADIUSY are filter dimensions
    //BLK_X, BLK_Y are local wrogroup sizes
    //all these should be defined on host during compile time
    //first lsmem array for source pixels used in first pass,
    //second lsmemDy for storing first pass results
    __local WORKTYPE lsmem[BLK_Y+2*RADIUSY][BLK_X+2*RADIUSX];
    __local WORKTYPE lsmemDy[BLK_Y][BLK_X+2*RADIUSX];

    //get local and global ids - used as image and local memory array indexes
    int lix = get_local_id(0);
    int liy = get_local_id(1);

    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);

    //calculate pixel position in source image taking image offset into account
    int srcX = x + srcOffsetX - RADIUSX;
    int srcY = y + srcOffsetY - RADIUSY;
    int xb = srcX;
    int yb = srcY;

    //extrapolate coordinates, if needed
    //and read my own source pixel into local memory
    //with account for extra border pixels, which will be read by starting workitems
    int clocY = liy;
    int cSrcY = srcY;
    do
    {
        int yb = cSrcY;
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
        cSrcY += BLK_Y;
    }
    while(clocY < BLK_Y+(RADIUSY*2));
    barrier(CLK_LOCAL_MEM_FENCE);

    //do vertical filter pass
    //and store intermediate results to second local memory array
    int i;
    WORKTYPE sum = 0.0f;
    int clocX = lix;
    do
    {
        sum = 0.0f;
        for(i=0; i<=2*RADIUSY; i++)
            sum = mad(lsmem[liy+i][clocX], as_float(mat_kernelY[i]), sum);
        lsmemDy[liy][clocX] = sum;
        clocX += BLK_X;
    }
    while(clocX < BLK_X+(RADIUSX*2));
    barrier(CLK_LOCAL_MEM_FENCE);

    //if this pixel happened to be out of image borders because of global size rounding,
    //then just return
    if( x >= dstWidth || y >=dstHeight )  return;

    //do second horizontal filter pass
    //and calculate final result
    sum = 0.0f;
    for(i=0; i<=2*RADIUSX; i++)
        sum = mad(lsmemDy[liy][lix+i], as_float(mat_kernelX[i]), sum);

    //store result into destination image
    DST(x,y) = CONVERT_DSTTYPE(sum);
}
