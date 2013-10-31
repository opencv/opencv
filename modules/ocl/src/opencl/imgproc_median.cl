//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Niko Li, newlife20080214@gmail.com
//    Zero Lin, zero.lin@amd.com
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
//


/*
__kernel void medianFilter_C1(__global uchar * src, __global uchar * dst,  int srcOffset, int dstOffset, int cols,
                                int rows, int srcStep, int dstStep, int m)
{
    int dx = get_global_id(0)-(m>>1);
    int dy = get_global_id(1)-(m>>1);

    short histom[256];
    for(int i=0;i<256;++i)
        histom[i]=0;


    for(int i=0;i<m;++i)
    {
        __global uchar * data = src + srcOffset + mul24(srcStep,clamp(dy + (i), 0, rows-1));
        for(int j=dx;j<dx+m;++j)
        {
            histom[data[clamp(j, 0, cols-1)]]++;
        }
    }

    int now=0;
    int goal=(m*m+1)>>1;
    int v;
    for(int i=0;i<256;++i)
    {
        v=(now<goal?i:v);
        now+=histom[i];
    }

    if(dy<rows && dx<cols)
        dst[dstOffset + get_global_id(1)*dstStep + get_global_id(0)]=v;
}
*/
#define op(a,b) {mid=a; a=min(a,b); b=max(mid,b);}
__kernel void medianFilter3_C4_D0(__global uchar4 * src, __global uchar4 * dst,  int srcOffset, int dstOffset, int cols,
                                int rows, int srcStep, int dstStep)
{

    __local uchar4 data[18][18];
    __global uchar4* source=src + srcOffset;

    int dx = get_global_id(0) - get_local_id(0) -1;
    int dy = get_global_id(1) - get_local_id(1) -1;

    const int id = min((int)(get_local_id(0)*16+get_local_id(1)), 9*18-1);

    int dr=id/18;
    int dc=id%18;
    int r=clamp(dy+dr, 0, rows-1);
    int c=clamp(dx+dc, 0, cols-1);

    data[dr][dc] = source[r*srcStep + c];
    r=clamp(dy+dr+9, 0, rows-1);
    data[dr+9][dc] = source[r*srcStep + c];

    barrier(CLK_LOCAL_MEM_FENCE);

    int x =get_local_id(0);
    int y =get_local_id(1);
    uchar4 p0=data[y][x], p1=data[y][x+1], p2=data[y][x+2];
    uchar4 p3=data[y+1][x], p4=data[y+1][x+1], p5=data[y+1][x+2];
    uchar4 p6=data[y+2][x], p7=data[y+2][x+1], p8=data[y+2][x+2];
    uchar4 mid;

    op(p1, p2); op(p4, p5); op(p7, p8); op(p0, p1);
    op(p3, p4); op(p6, p7); op(p1, p2); op(p4, p5);
    op(p7, p8); op(p0, p3); op(p5, p8); op(p4, p7);
    op(p3, p6); op(p1, p4); op(p2, p5); op(p4, p7);
    op(p4, p2); op(p6, p4); op(p4, p2);

    if((int)get_global_id(1)<rows && (int)get_global_id(0)<cols)
        dst[dstOffset + get_global_id(1)*dstStep + get_global_id(0)]=p4;
}
#undef op

#define op(a,b) {mid=a; a=min(a,b); b=max(mid,b);}
__kernel void medianFilter3_C1_D0(__global uchar * src, __global uchar * dst,  int srcOffset, int dstOffset, int cols,
                                int rows, int srcStep, int dstStep)
{

    __local uchar data[18][18];
    __global uchar* source=src + srcOffset;

    int dx = get_global_id(0) - get_local_id(0) -1;
    int dy = get_global_id(1) - get_local_id(1) -1;

    const int id = min((int)(get_local_id(0)*16+get_local_id(1)), 9*18-1);

    int dr=id/18;
    int dc=id%18;
    int r=clamp(dy+dr, 0, rows-1);
    int c=clamp(dx+dc, 0, cols-1);

    data[dr][dc] = source[r*srcStep + c];
    r=clamp(dy+dr+9, 0, rows-1);
    data[dr+9][dc] = source[r*srcStep + c];

    barrier(CLK_LOCAL_MEM_FENCE);

    int x =get_local_id(0);
    int y =get_local_id(1);
    uchar p0=data[y][x], p1=data[y][x+1], p2=data[y][x+2];
    uchar p3=data[y+1][x], p4=data[y+1][x+1], p5=data[y+1][x+2];
    uchar p6=data[y+2][x], p7=data[y+2][x+1], p8=data[y+2][x+2];
    uchar mid;

    op(p1, p2); op(p4, p5); op(p7, p8); op(p0, p1);
    op(p3, p4); op(p6, p7); op(p1, p2); op(p4, p5);
    op(p7, p8); op(p0, p3); op(p5, p8); op(p4, p7);
    op(p3, p6); op(p1, p4); op(p2, p5); op(p4, p7);
    op(p4, p2); op(p6, p4); op(p4, p2);

    if((int)get_global_id(1)<rows && (int)get_global_id(0)<cols)
        dst[dstOffset + get_global_id(1)*dstStep + get_global_id(0)]=p4;
}
#undef op

#define op(a,b) {mid=a; a=min(a,b); b=max(mid,b);}
__kernel void medianFilter3_C1_D5(__global float * src, __global float * dst,  int srcOffset, int dstOffset, int cols,
                                int rows, int srcStep, int dstStep)
{

    __local float data[18][18];
    __global float* source=src + srcOffset;

    int dx = get_global_id(0) - get_local_id(0) -1;
    int dy = get_global_id(1) - get_local_id(1) -1;

    const int id = min((int)(get_local_id(0)*16+get_local_id(1)), 9*18-1);

    int dr=id/18;
    int dc=id%18;
    int r=clamp(dy+dr, 0, rows-1);
    int c=clamp(dx+dc, 0, cols-1);

    data[dr][dc] = source[r*srcStep + c];
    r=clamp(dy+dr+9, 0, rows-1);
    data[dr+9][dc] = source[r*srcStep + c];

    barrier(CLK_LOCAL_MEM_FENCE);

    int x =get_local_id(0);
    int y =get_local_id(1);
    float p0=data[y][x], p1=data[y][x+1], p2=data[y][x+2];
    float p3=data[y+1][x], p4=data[y+1][x+1], p5=data[y+1][x+2];
    float p6=data[y+2][x], p7=data[y+2][x+1], p8=data[y+2][x+2];
    float mid;

    op(p1, p2); op(p4, p5); op(p7, p8); op(p0, p1);
    op(p3, p4); op(p6, p7); op(p1, p2); op(p4, p5);
    op(p7, p8); op(p0, p3); op(p5, p8); op(p4, p7);
    op(p3, p6); op(p1, p4); op(p2, p5); op(p4, p7);
    op(p4, p2); op(p6, p4); op(p4, p2);

    if((int)get_global_id(1)<rows && (int)get_global_id(0)<cols)
        dst[dstOffset + get_global_id(1)*dstStep + get_global_id(0)]=p4;
}
#undef op

#define op(a,b) {mid=a; a=min(a,b); b=max(mid,b);}
__kernel void medianFilter3_C4_D5(__global float4 * src, __global float4 * dst,  int srcOffset, int dstOffset, int cols,
                                int rows, int srcStep, int dstStep)
{

    __local float4 data[18][18];
    __global float4* source=src + srcOffset;

    int dx = get_global_id(0) - get_local_id(0) -1;
    int dy = get_global_id(1) - get_local_id(1) -1;

    const int id = min((int)(get_local_id(0)*16+get_local_id(1)), 9*18-1);

    int dr=id/18;
    int dc=id%18;
    int r=clamp(dy+dr, 0, rows-1);
    int c=clamp(dx+dc, 0, cols-1);

    data[dr][dc] = source[r*srcStep + c];
    r=clamp(dy+dr+9, 0, rows-1);
    data[dr+9][dc] = source[r*srcStep + c];

    barrier(CLK_LOCAL_MEM_FENCE);

    int x =get_local_id(0);
    int y =get_local_id(1);
    float4 p0=data[y][x], p1=data[y][x+1], p2=data[y][x+2];
    float4 p3=data[y+1][x], p4=data[y+1][x+1], p5=data[y+1][x+2];
    float4 p6=data[y+2][x], p7=data[y+2][x+1], p8=data[y+2][x+2];
    float4 mid;

    op(p1, p2); op(p4, p5); op(p7, p8); op(p0, p1);
    op(p3, p4); op(p6, p7); op(p1, p2); op(p4, p5);
    op(p7, p8); op(p0, p3); op(p5, p8); op(p4, p7);
    op(p3, p6); op(p1, p4); op(p2, p5); op(p4, p7);
    op(p4, p2); op(p6, p4); op(p4, p2);

    if((int)get_global_id(1)<rows && (int)get_global_id(0)<cols)
        dst[dstOffset + get_global_id(1)*dstStep + get_global_id(0)]=p4;
}
#undef op

#define op(a,b) {mid=a; a=min(a,b); b=max(mid,b);}
__kernel void medianFilter5_C4_D0(__global uchar4 * src, __global uchar4 * dst,  int srcOffset, int dstOffset, int cols,
                                int rows, int srcStep, int dstStep)
{

    __local uchar4 data[20][20];
    __global uchar4* source=src + srcOffset;

    int dx = get_global_id(0) - get_local_id(0) -2;
    int dy = get_global_id(1) - get_local_id(1) -2;

    const int id = min((int)(get_local_id(0)*16+get_local_id(1)), 10*20-1);

    int dr=id/20;
    int dc=id%20;
    int r=clamp(dy+dr, 0, rows-1);
    int c=clamp(dx+dc, 0, cols-1);

    data[dr][dc] = source[r*srcStep + c];
    r=clamp(dy+dr+10, 0, rows-1);
    data[dr+10][dc] = source[r*srcStep + c];

    barrier(CLK_LOCAL_MEM_FENCE);

    int x =get_local_id(0);
    int y =get_local_id(1);
    uchar4 p0=data[y][x], p1=data[y][x+1], p2=data[y][x+2], p3=data[y][x+3], p4=data[y][x+4];
    uchar4 p5=data[y+1][x], p6=data[y+1][x+1], p7=data[y+1][x+2], p8=data[y+1][x+3], p9=data[y+1][x+4];
    uchar4 p10=data[y+2][x], p11=data[y+2][x+1], p12=data[y+2][x+2], p13=data[y+2][x+3], p14=data[y+2][x+4];
    uchar4 p15=data[y+3][x], p16=data[y+3][x+1], p17=data[y+3][x+2], p18=data[y+3][x+3], p19=data[y+3][x+4];
    uchar4 p20=data[y+4][x], p21=data[y+4][x+1], p22=data[y+4][x+2], p23=data[y+4][x+3], p24=data[y+4][x+4];
    uchar4 mid;

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

    if((int)get_global_id(1)<rows && (int)get_global_id(0)<cols)
        dst[dstOffset + get_global_id(1)*dstStep + get_global_id(0)]=p12;
}
#undef op

#define op(a,b) {mid=a; a=min(a,b); b=max(mid,b);}
__kernel void medianFilter5_C1_D0(__global uchar * src, __global uchar * dst,  int srcOffset, int dstOffset, int cols,
                                int rows, int srcStep, int dstStep)
{

    __local uchar data[20][20];
    __global uchar* source=src + srcOffset;

    int dx = get_global_id(0) - get_local_id(0) -2;
    int dy = get_global_id(1) - get_local_id(1) -2;

    const int id = min((int)(get_local_id(0)*16+get_local_id(1)), 10*20-1);

    int dr=id/20;
    int dc=id%20;
    int r=clamp(dy+dr, 0, rows-1);
    int c=clamp(dx+dc, 0, cols-1);

    data[dr][dc] = source[r*srcStep + c];
    r=clamp(dy+dr+10, 0, rows-1);
    data[dr+10][dc] = source[r*srcStep + c];

    barrier(CLK_LOCAL_MEM_FENCE);

    int x =get_local_id(0);
    int y =get_local_id(1);
    uchar p0=data[y][x], p1=data[y][x+1], p2=data[y][x+2], p3=data[y][x+3], p4=data[y][x+4];
    uchar p5=data[y+1][x], p6=data[y+1][x+1], p7=data[y+1][x+2], p8=data[y+1][x+3], p9=data[y+1][x+4];
    uchar p10=data[y+2][x], p11=data[y+2][x+1], p12=data[y+2][x+2], p13=data[y+2][x+3], p14=data[y+2][x+4];
    uchar p15=data[y+3][x], p16=data[y+3][x+1], p17=data[y+3][x+2], p18=data[y+3][x+3], p19=data[y+3][x+4];
    uchar p20=data[y+4][x], p21=data[y+4][x+1], p22=data[y+4][x+2], p23=data[y+4][x+3], p24=data[y+4][x+4];
    uchar mid;

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

    if((int)get_global_id(1)<rows && (int)get_global_id(0)<cols)
        dst[dstOffset + get_global_id(1)*dstStep + get_global_id(0)]=p12;
}
#undef op

#define op(a,b) {mid=a; a=min(a,b); b=max(mid,b);}
__kernel void medianFilter5_C4_D5(__global float4 * src, __global float4 * dst,  int srcOffset, int dstOffset, int cols,
                                int rows, int srcStep, int dstStep)
{

    __local float4 data[20][20];
    __global float4* source=src + srcOffset;

    int dx = get_global_id(0) - get_local_id(0) -2;
    int dy = get_global_id(1) - get_local_id(1) -2;

    const int id = min((int)(get_local_id(0)*16+get_local_id(1)), 10*20-1);

    int dr=id/20;
    int dc=id%20;
    int r=clamp(dy+dr, 0, rows-1);
    int c=clamp(dx+dc, 0, cols-1);

    data[dr][dc] = source[r*srcStep + c];
    r=clamp(dy+dr+10, 0, rows-1);
    data[dr+10][dc] = source[r*srcStep + c];

    barrier(CLK_LOCAL_MEM_FENCE);

    int x =get_local_id(0);
    int y =get_local_id(1);
    float4 p0=data[y][x], p1=data[y][x+1], p2=data[y][x+2], p3=data[y][x+3], p4=data[y][x+4];
    float4 p5=data[y+1][x], p6=data[y+1][x+1], p7=data[y+1][x+2], p8=data[y+1][x+3], p9=data[y+1][x+4];
    float4 p10=data[y+2][x], p11=data[y+2][x+1], p12=data[y+2][x+2], p13=data[y+2][x+3], p14=data[y+2][x+4];
    float4 p15=data[y+3][x], p16=data[y+3][x+1], p17=data[y+3][x+2], p18=data[y+3][x+3], p19=data[y+3][x+4];
    float4 p20=data[y+4][x], p21=data[y+4][x+1], p22=data[y+4][x+2], p23=data[y+4][x+3], p24=data[y+4][x+4];
    float4 mid;

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

    if((int)get_global_id(1)<rows && (int)get_global_id(0)<cols)
        dst[dstOffset + get_global_id(1)*dstStep + get_global_id(0)]=p12;
}
#undef op

#define op(a,b) {mid=a; a=min(a,b); b=max(mid,b);}
__kernel void medianFilter5_C1_D5(__global float * src, __global float * dst,  int srcOffset, int dstOffset, int cols,
                                int rows, int srcStep, int dstStep)
{

    __local float data[20][20];
    __global float* source=src + srcOffset;

    int dx = get_global_id(0) - get_local_id(0) -2;
    int dy = get_global_id(1) - get_local_id(1) -2;

    const int id = min((int)(get_local_id(0)*16+get_local_id(1)), 10*20-1);

    int dr=id/20;
    int dc=id%20;
    int r=clamp(dy+dr, 0, rows-1);
    int c=clamp(dx+dc, 0, cols-1);

    data[dr][dc] = source[r*srcStep + c];
    r=clamp(dy+dr+10, 0, rows-1);
    data[dr+10][dc] = source[r*srcStep + c];

    barrier(CLK_LOCAL_MEM_FENCE);

    int x =get_local_id(0);
    int y =get_local_id(1);
    float p0=data[y][x], p1=data[y][x+1], p2=data[y][x+2], p3=data[y][x+3], p4=data[y][x+4];
    float p5=data[y+1][x], p6=data[y+1][x+1], p7=data[y+1][x+2], p8=data[y+1][x+3], p9=data[y+1][x+4];
    float p10=data[y+2][x], p11=data[y+2][x+1], p12=data[y+2][x+2], p13=data[y+2][x+3], p14=data[y+2][x+4];
    float p15=data[y+3][x], p16=data[y+3][x+1], p17=data[y+3][x+2], p18=data[y+3][x+3], p19=data[y+3][x+4];
    float p20=data[y+4][x], p21=data[y+4][x+1], p22=data[y+4][x+2], p23=data[y+4][x+3], p24=data[y+4][x+4];
    float mid;

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

    if((int)get_global_id(1)<rows && (int)get_global_id(0)<cols)
        dst[dstOffset + get_global_id(1)*dstStep + get_global_id(0)]=p12;
}
#undef op
