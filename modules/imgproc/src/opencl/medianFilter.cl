//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
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

#define DATA_TYPE type

#define scnbytes ((int)sizeof(type))

#define op(a,b) {    mid=a; a=min(a,b); b=max(mid,b);}

__kernel void medianFilter3(__global const uchar* srcptr, int srcStep, int srcOffset,
                            __global uchar* dstptr, int dstStep, int dstOffset,
                            int rows, int cols)
{
    __local DATA_TYPE data[18][18];

    int x = get_local_id(0);
    int y = get_local_id(1);

    int gx= get_global_id(0);
    int gy= get_global_id(1);

    int dx = gx - x - 1;
    int dy = gy - y - 1;

    const int id = min((int)(x*16+y), 9*18-1);

    int dr = id / 18;
    int dc = id % 18;

    int c = clamp(dx+dc, 0, cols-1);

    int r = clamp(dy+dr, 0, rows-1);
    int index1 = mad24(r, srcStep, srcOffset + c*scnbytes);

    r = clamp(dy+dr+9, 0, rows-1);
    int index9 = mad24(r, srcStep, srcOffset + c*scnbytes);

    __global DATA_TYPE * src = (__global DATA_TYPE *)(srcptr + index1);
    data[dr][dc] = src[0];

    src = (__global DATA_TYPE *)(srcptr + index9);
    data[dr+9][dc] = src[0];

    barrier(CLK_LOCAL_MEM_FENCE);

    DATA_TYPE p0=data[y][x], p1=data[y][(x+1)], p2=data[y][(x+2)];
    DATA_TYPE p3=data[y+1][x], p4=data[y+1][(x+1)], p5=data[y+1][(x+2)];
    DATA_TYPE p6=data[y+2][x], p7=data[y+2][(x+1)], p8=data[y+2][(x+2)];
    DATA_TYPE mid;

    op(p1, p2); op(p4, p5); op(p7, p8); op(p0, p1);
    op(p3, p4); op(p6, p7); op(p1, p2); op(p4, p5);
    op(p7, p8); op(p0, p3); op(p5, p8); op(p4, p7);
    op(p3, p6); op(p1, p4); op(p2, p5); op(p4, p7);
    op(p4, p2); op(p6, p4); op(p4, p2);

    int dst_index = mad24( gy, dstStep, dstOffset + gx * scnbytes);

    if( gy < rows && gx < cols)
    {
        __global DATA_TYPE* dst = (__global DATA_TYPE *)(dstptr + dst_index);
        dst[0] = p4;
    }
}

__kernel void medianFilter5(__global const uchar* srcptr, int srcStep, int srcOffset,
                            __global uchar* dstptr, int dstStep, int dstOffset,
                            int rows, int cols)
{
    __local DATA_TYPE data[20][20];

    int x =get_local_id(0);
    int y =get_local_id(1);

    int gx=get_global_id(0);
    int gy=get_global_id(1);

    int dx = gx - x - 2;
    int dy = gy - y - 2;

    const int id = min((int)(x*16+y), 10*20-1);

    int dr=id/20;
    int dc=id%20;

    int c=clamp(dx+dc, 0, cols-1);

    int r = clamp(dy+dr, 0, rows-1);
    int index1 = mad24(r, srcStep, srcOffset + c*scnbytes);

    r = clamp(dy+dr+10, 0, rows-1);
    int index10 = mad24(r, srcStep, srcOffset + c*scnbytes);

    __global DATA_TYPE * src = (__global DATA_TYPE *)(srcptr + index1);
    data[dr][dc] = src[0];
    src = (__global DATA_TYPE *)(srcptr + index10);
    data[dr+10][dc] = src[0];

    barrier(CLK_LOCAL_MEM_FENCE);

    DATA_TYPE p0=data[y][x], p1=data[y][x+1], p2=data[y][x+2], p3=data[y][x+3], p4=data[y][x+4];
    DATA_TYPE p5=data[y+1][x], p6=data[y+1][x+1], p7=data[y+1][x+2], p8=data[y+1][x+3], p9=data[y+1][x+4];
    DATA_TYPE p10=data[y+2][x], p11=data[y+2][x+1], p12=data[y+2][x+2], p13=data[y+2][x+3], p14=data[y+2][x+4];
    DATA_TYPE p15=data[y+3][x], p16=data[y+3][x+1], p17=data[y+3][x+2], p18=data[y+3][x+3], p19=data[y+3][x+4];
    DATA_TYPE p20=data[y+4][x], p21=data[y+4][x+1], p22=data[y+4][x+2], p23=data[y+4][x+3], p24=data[y+4][x+4];
    DATA_TYPE mid;

    op(p1, p2); op(p0, p1); op(p1, p2); op(p4, p5); op(p3, p4);
    op(p4, p5); op(p0, p3); op(p2, p5); op(p2, p3); op(p1, p4);
    op(p1, p2); op(p3, p4); op(p7, p8); op(p6, p7); op(p7, p8);
    op(p10, p11); op(p9, p10); op(p10, p11); op(p6, p9); op(p8, p11);
    op(p8, p9); op(p7, p10); op(p7, p8); op(p9, p10); op(p0, p6);
    op(p4, p10); op(p4, p6); op(p2, p8); op(p2, p4); op(p6, p8);
    op(p1, p7); op(p5, p11); op(p5, p7); op(p3, p9); op(p3, p5);
    op(p7, p9); op(p1, p2); op(p3, p4); op(p5, p6); op(p7, p8);
    op(p9, p10); op(p13, p14); op(p12, p13); op(p13, p14); op(p16, p17);
    op(p15, p16); op(p16, p17); op(p12, p15); op(p14, p17); op(p14, p15);
    op(p13, p16); op(p13, p14); op(p15, p16); op(p19, p20); op(p18, p19);
    op(p19, p20); op(p21, p22); op(p23, p24); op(p21, p23); op(p22, p24);
    op(p22, p23); op(p18, p21); op(p20, p23); op(p20, p21); op(p19, p22);
    op(p22, p24); op(p19, p20); op(p21, p22); op(p23, p24); op(p12, p18);
    op(p16, p22); op(p16, p18); op(p14, p20); op(p20, p24); op(p14, p16);
    op(p18, p20); op(p22, p24); op(p13, p19); op(p17, p23); op(p17, p19);
    op(p15, p21); op(p15, p17); op(p19, p21); op(p13, p14); op(p15, p16);
    op(p17, p18); op(p19, p20); op(p21, p22); op(p23, p24); op(p0, p12);
    op(p8, p20); op(p8, p12); op(p4, p16); op(p16, p24); op(p12, p16);
    op(p2, p14); op(p10, p22); op(p10, p14); op(p6, p18); op(p6, p10);
    op(p10, p12); op(p1, p13); op(p9, p21); op(p9, p13); op(p5, p17);
    op(p13, p17); op(p3, p15); op(p11, p23); op(p11, p15); op(p7, p19);
    op(p7, p11); op(p11, p13); op(p11, p12);

    int dst_index = mad24( gy, dstStep, dstOffset + gx * scnbytes);

    if( gy < rows && gx < cols)
    {
        __global DATA_TYPE* dst = (__global DATA_TYPE *)(dstptr + dst_index);
        dst[0] = p12;
    }
}