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
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Zhang Chunpeng	chunpeng@multicorewareinc.com
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other oclMaterials provided with the distribution.
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

//#pragma OPENCL EXTENSION cl_amd_printf : enable

uchar get_valid_uchar(uchar data)
{
    return (uchar)(data <= 255 ? data : data > 0 ? 255 : 0);
}
///////////////////////////////////////////////////////////////////////
//////////////////////////  CV_8UC1  //////////////////////////////////
///////////////////////////////////////////////////////////////////////
__kernel void pyrUp_C1_D0(__global uchar* src,__global uchar* dst,
                int srcRows,int dstRows,int srcCols,int dstCols,
                int srcOffset,int dstOffset,int srcStep,int dstStep)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    __local float s_srcPatch[10][10];
    __local float s_dstPatch[20][16];


    if( get_local_id(0) < 10 && get_local_id(1) < 10 )
    {
        int srcx = (int)(get_group_id(0) * get_local_size(0) / 2 + get_local_id(0)) - 1;
        int srcy = (int)(get_group_id(1) * get_local_size(1) / 2 + get_local_id(1)) - 1;

        srcx = abs(srcx);
        srcx = min(srcCols - 1,srcx);

        srcy = abs(srcy);
        srcy = min(srcRows -1 ,srcy);

        s_srcPatch[get_local_id(1)][get_local_id(0)] = (float)(src[srcx + srcy * srcStep]);

    }

    barrier(CLK_LOCAL_MEM_FENCE);

    float sum = 0;
    const int evenFlag = (int)((get_local_id(0) & 1) == 0);
    const int oddFlag = (int)((get_local_id(0) & 1) != 0);
    const bool  eveny = ((get_local_id(1) & 1) == 0);
    const int tidx = get_local_id(0);

    if(eveny)
    {
        sum = sum + (evenFlag * 0.0625f) * s_srcPatch[1 + (get_local_id(1) >> 1)][1 + ((tidx - 2) >> 1)];
        sum = sum + ( oddFlag * 0.25f  ) * s_srcPatch[1 + (get_local_id(1) >> 1)][1 + ((tidx - 1) >> 1)];
        sum = sum + (evenFlag * 0.375f ) * s_srcPatch[1 + (get_local_id(1) >> 1)][1 + ((tidx    ) >> 1)];
        sum = sum + ( oddFlag * 0.25f  ) * s_srcPatch[1 + (get_local_id(1) >> 1)][1 + ((tidx + 1) >> 1)];
        sum = sum + (evenFlag * 0.0625f) * s_srcPatch[1 + (get_local_id(1) >> 1)][1 + ((tidx + 2) >> 1)];
    }

    s_dstPatch[2 + get_local_id(1)][get_local_id(0)] = sum;

    if (get_local_id(1) < 2)
    {
        sum = 0;

        if (eveny)
        {
            sum = sum + (evenFlag * 0.0625f) * s_srcPatch[0][1 + ((tidx - 2) >> 1)];
            sum = sum + ( oddFlag * 0.25f  ) * s_srcPatch[0][1 + ((tidx - 1) >> 1)];
            sum = sum + (evenFlag * 0.375f ) * s_srcPatch[0][1 + ((tidx    ) >> 1)];
            sum = sum + ( oddFlag * 0.25f  ) * s_srcPatch[0][1 + ((tidx + 1) >> 1)];
            sum = sum + (evenFlag * 0.0625f) * s_srcPatch[0][1 + ((tidx + 2) >> 1)];
         }

         s_dstPatch[get_local_id(1)][get_local_id(0)] = sum;
     }

     if (get_local_id(1) > 13)
     {
        sum = 0;

        if (eveny)
        {
            sum = sum + (evenFlag * 0.0625f) * s_srcPatch[9][1 + ((tidx - 2) >> 1)];
            sum = sum + ( oddFlag * 0.25f  ) * s_srcPatch[9][1 + ((tidx - 1) >> 1)];
            sum = sum + (evenFlag * 0.375f ) * s_srcPatch[9][1 + ((tidx    ) >> 1)];
            sum = sum + ( oddFlag * 0.25f  ) * s_srcPatch[9][1 + ((tidx + 1) >> 1)];
            sum = sum + (evenFlag * 0.0625f) * s_srcPatch[9][1 + ((tidx + 2) >> 1)];
        }
        s_dstPatch[4 + get_local_id(1)][get_local_id(0)] = sum;
     }

    barrier(CLK_LOCAL_MEM_FENCE);

    sum = 0;

    const int tidy = get_local_id(1);

    sum = sum + 0.0625f * s_dstPatch[2 + tidy - 2][get_local_id(0)];
    sum = sum + 0.25f   * s_dstPatch[2 + tidy - 1][get_local_id(0)];
    sum = sum + 0.375f  * s_dstPatch[2 + tidy    ][get_local_id(0)];
    sum = sum + 0.25f   * s_dstPatch[2 + tidy + 1][get_local_id(0)];
    sum = sum + 0.0625f * s_dstPatch[2 + tidy + 2][get_local_id(0)];

    if ((x < dstCols) && (y < dstRows))
        dst[x + y * dstStep] = (float)(4.0f * sum);

}

///////////////////////////////////////////////////////////////////////
//////////////////////////  CV_16UC1  /////////////////////////////////
///////////////////////////////////////////////////////////////////////
__kernel void pyrUp_C1_D2(__global ushort* src,__global ushort* dst,
                int srcRows,int dstRows,int srcCols,int dstCols,
                int srcOffset,int dstOffset,int srcStep,int dstStep)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    __local float s_srcPatch[10][10];
    __local float s_dstPatch[20][16];

    srcStep = srcStep >> 1;
    dstStep = dstStep >> 1;
    srcOffset = srcOffset >> 1;
    dstOffset = dstOffset >> 1;


    if( get_local_id(0) < 10 && get_local_id(1) < 10 )
    {
        int srcx = (int)(get_group_id(0) * get_local_size(0) / 2 + get_local_id(0)) - 1;
        int srcy = (int)(get_group_id(1) * get_local_size(1) / 2 + get_local_id(1)) - 1;

        srcx = abs(srcx);
        srcx = min(srcCols - 1,srcx);

        srcy = abs(srcy);
        srcy = min(srcRows -1 ,srcy);

        s_srcPatch[get_local_id(1)][get_local_id(0)] = (float)(src[srcx + srcy * srcStep]);

    }

    barrier(CLK_LOCAL_MEM_FENCE);

    float sum = 0;

    const int evenFlag = (int)((get_local_id(0) & 1) == 0);
    const int oddFlag = (int)((get_local_id(0) & 1) != 0);
    const bool  eveny = ((get_local_id(1) & 1) == 0);
    const int tidx = get_local_id(0);

    if(eveny)
    {
        sum = sum + (evenFlag * 0.0625f) * s_srcPatch[1 + (get_local_id(1) >> 1)][1 + ((tidx - 2) >> 1)];
        sum = sum + ( oddFlag * 0.25f  ) * s_srcPatch[1 + (get_local_id(1) >> 1)][1 + ((tidx - 1) >> 1)];
        sum = sum + (evenFlag * 0.375f ) * s_srcPatch[1 + (get_local_id(1) >> 1)][1 + ((tidx    ) >> 1)];
        sum = sum + ( oddFlag * 0.25f  ) * s_srcPatch[1 + (get_local_id(1) >> 1)][1 + ((tidx + 1) >> 1)];
        sum = sum + (evenFlag * 0.0625f) * s_srcPatch[1 + (get_local_id(1) >> 1)][1 + ((tidx + 2) >> 1)];
    }

    s_dstPatch[2 + get_local_id(1)][get_local_id(0)] = sum;

    if (get_local_id(1) < 2)
    {
        sum = 0;

        if (eveny)
        {
            sum = sum + (evenFlag * 0.0625f) * s_srcPatch[0][1 + ((tidx - 2) >> 1)];
            sum = sum + ( oddFlag * 0.25f  ) * s_srcPatch[0][1 + ((tidx - 1) >> 1)];
            sum = sum + (evenFlag * 0.375f ) * s_srcPatch[0][1 + ((tidx    ) >> 1)];
            sum = sum + ( oddFlag * 0.25f  ) * s_srcPatch[0][1 + ((tidx + 1) >> 1)];
            sum = sum + (evenFlag * 0.0625f) * s_srcPatch[0][1 + ((tidx + 2) >> 1)];
         }

         s_dstPatch[get_local_id(1)][get_local_id(0)] = sum;
     }

     if (get_local_id(1) > 13)
     {
        sum = 0;

        if (eveny)
        {
            sum = sum + (evenFlag * 0.0625f) * s_srcPatch[9][1 + ((tidx - 2) >> 1)];
            sum = sum + ( oddFlag * 0.25f  ) * s_srcPatch[9][1 + ((tidx - 1) >> 1)];
            sum = sum + (evenFlag * 0.375f ) * s_srcPatch[9][1 + ((tidx    ) >> 1)];
            sum = sum + ( oddFlag * 0.25f  ) * s_srcPatch[9][1 + ((tidx + 1) >> 1)];
            sum = sum + (evenFlag * 0.0625f) * s_srcPatch[9][1 + ((tidx + 2) >> 1)];
        }
        s_dstPatch[4 + get_local_id(1)][get_local_id(0)] = sum;
     }

    barrier(CLK_LOCAL_MEM_FENCE);

    sum = 0;

    const int tidy = get_local_id(1);

    sum = sum + 0.0625f * s_dstPatch[2 + tidy - 2][get_local_id(0)];
    sum = sum + 0.25f   * s_dstPatch[2 + tidy - 1][get_local_id(0)];
    sum = sum + 0.375f  * s_dstPatch[2 + tidy    ][get_local_id(0)];
    sum = sum + 0.25f   * s_dstPatch[2 + tidy + 1][get_local_id(0)];
    sum = sum + 0.0625f * s_dstPatch[2 + tidy + 2][get_local_id(0)];

    if ((x < dstCols) && (y < dstRows))
        dst[x + y * dstStep] = (float)(4.0f * sum);

}

///////////////////////////////////////////////////////////////////////
//////////////////////////  CV_32FC1  /////////////////////////////////
///////////////////////////////////////////////////////////////////////
__kernel void pyrUp_C1_D5(__global float* src,__global float* dst,
                int srcRows,int dstRows,int srcCols,int dstCols,
                int srcOffset,int dstOffset,int srcStep,int dstStep)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    __local float s_srcPatch[10][10];
    __local float s_dstPatch[20][16];

    srcOffset = srcOffset >> 2;
    dstOffset = dstOffset >> 2;
    srcStep = srcStep >> 2;
    dstStep = dstStep >> 2;


    if( get_local_id(0) < 10 && get_local_id(1) < 10 )
    {
        int srcx = (int)(get_group_id(0) * get_local_size(0) / 2 + get_local_id(0)) - 1;
        int srcy = (int)(get_group_id(1) * get_local_size(1) / 2 + get_local_id(1)) - 1;

        srcx = abs(srcx);
        srcx = min(srcCols - 1,srcx);

        srcy = abs(srcy);
        srcy = min(srcRows -1 ,srcy);

        s_srcPatch[get_local_id(1)][get_local_id(0)] = (float)(src[srcx + srcy * srcStep]);

    }

    barrier(CLK_LOCAL_MEM_FENCE);

    float sum = 0;
    const int evenFlag = (int)((get_local_id(0) & 1) == 0);
    const int oddFlag = (int)((get_local_id(0) & 1) != 0);
    const bool  eveny = ((get_local_id(1) & 1) == 0);
    const int tidx = get_local_id(0);

    if(eveny)
    {
        sum = sum + (evenFlag * 0.0625f) * s_srcPatch[1 + (get_local_id(1) >> 1)][1 + ((tidx - 2) >> 1)];
        sum = sum + ( oddFlag * 0.25f  ) * s_srcPatch[1 + (get_local_id(1) >> 1)][1 + ((tidx - 1) >> 1)];
        sum = sum + (evenFlag * 0.375f ) * s_srcPatch[1 + (get_local_id(1) >> 1)][1 + ((tidx    ) >> 1)];
        sum = sum + ( oddFlag * 0.25f  ) * s_srcPatch[1 + (get_local_id(1) >> 1)][1 + ((tidx + 1) >> 1)];
        sum = sum + (evenFlag * 0.0625f) * s_srcPatch[1 + (get_local_id(1) >> 1)][1 + ((tidx + 2) >> 1)];
    }

    s_dstPatch[2 + get_local_id(1)][get_local_id(0)] = sum;

    if (get_local_id(1) < 2)
    {
        sum = 0;

        if (eveny)
        {
            sum = sum + (evenFlag * 0.0625f) * s_srcPatch[0][1 + ((tidx - 2) >> 1)];
            sum = sum + ( oddFlag * 0.25f  ) * s_srcPatch[0][1 + ((tidx - 1) >> 1)];
            sum = sum + (evenFlag * 0.375f ) * s_srcPatch[0][1 + ((tidx    ) >> 1)];
            sum = sum + ( oddFlag * 0.25f  ) * s_srcPatch[0][1 + ((tidx + 1) >> 1)];
            sum = sum + (evenFlag * 0.0625f) * s_srcPatch[0][1 + ((tidx + 2) >> 1)];
         }

         s_dstPatch[get_local_id(1)][get_local_id(0)] = sum;
     }

     if (get_local_id(1) > 13)
     {
        sum = 0;

        if (eveny)
        {
            sum = sum + (evenFlag * 0.0625f) * s_srcPatch[9][1 + ((tidx - 2) >> 1)];
            sum = sum + ( oddFlag * 0.25f  ) * s_srcPatch[9][1 + ((tidx - 1) >> 1)];
            sum = sum + (evenFlag * 0.375f ) * s_srcPatch[9][1 + ((tidx    ) >> 1)];
            sum = sum + ( oddFlag * 0.25f  ) * s_srcPatch[9][1 + ((tidx + 1) >> 1)];
            sum = sum + (evenFlag * 0.0625f) * s_srcPatch[9][1 + ((tidx + 2) >> 1)];
        }
        s_dstPatch[4 + get_local_id(1)][get_local_id(0)] = sum;
     }

    barrier(CLK_LOCAL_MEM_FENCE);

    sum = 0;

    const int tidy = get_local_id(1);

    sum = sum + 0.0625f * s_dstPatch[2 + tidy - 2][get_local_id(0)];
    sum = sum + 0.25f   * s_dstPatch[2 + tidy - 1][get_local_id(0)];
    sum = sum + 0.375f  * s_dstPatch[2 + tidy    ][get_local_id(0)];
    sum = sum + 0.25f   * s_dstPatch[2 + tidy + 1][get_local_id(0)];
    sum = sum + 0.0625f * s_dstPatch[2 + tidy + 2][get_local_id(0)];

    if ((x < dstCols) && (y < dstRows))
        dst[x + y * dstStep] = (float)(4.0f * sum);

}

///////////////////////////////////////////////////////////////////////
//////////////////////////  CV_8UC4  //////////////////////////////////
///////////////////////////////////////////////////////////////////////
float4 covert_uchar4_to_float4(uchar4 data)
{
    float4 f4Data = {0,0,0,0};

    f4Data.x = (float)data.x;
    f4Data.y = (float)data.y;
    f4Data.z = (float)data.z;
    f4Data.w = (float)data.w;

    return f4Data;
}


uchar4 convert_float4_to_uchar4(float4 data)
{
    uchar4 u4Data;

    u4Data.x = get_valid_uchar(data.x);
    u4Data.y = get_valid_uchar(data.y);
    u4Data.z = get_valid_uchar(data.z);
    u4Data.w = get_valid_uchar(data.w);

    return u4Data;
}

float4 int_x_float4(int leftOpr,float4 rightOpr)
{
    float4 result = {0,0,0,0};

    result.x = rightOpr.x * leftOpr;
    result.y = rightOpr.y * leftOpr;
    result.z = rightOpr.z * leftOpr;
    result.w = rightOpr.w * leftOpr;

    return result;
}

float4 float4_x_float4(float4 leftOpr,float4 rightOpr)
{
    float4 result;

    result.x = leftOpr.x * rightOpr.x;
    result.y = leftOpr.y * rightOpr.y;
    result.z = leftOpr.z * rightOpr.z;
    result.w = leftOpr.w * rightOpr.w;

    return result;
}

__kernel void pyrUp_C4_D0(__global uchar4* src,__global uchar4* dst,
                int srcRows,int dstRows,int srcCols,int dstCols,
                int srcOffset,int dstOffset,int srcStep,int dstStep)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    __local float4 s_srcPatch[10][10];
    __local float4 s_dstPatch[20][16];

    srcOffset >>= 2;
    dstOffset >>= 2;
    srcStep >>= 2;
    dstStep >>= 2;


    if( get_local_id(0) < 10 && get_local_id(1) < 10 )
    {
        int srcx = (int)(get_group_id(0) * get_local_size(0) / 2 + get_local_id(0)) - 1;
        int srcy = (int)(get_group_id(1) * get_local_size(1) / 2 + get_local_id(1)) - 1;

        srcx = abs(srcx);
        srcx = min(srcCols - 1,srcx);

        srcy = abs(srcy);
        srcy = min(srcRows -1 ,srcy);

        s_srcPatch[get_local_id(1)][get_local_id(0)] = covert_uchar4_to_float4(src[srcx + srcy * srcStep]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    float4 sum = (float4)(0,0,0,0);

    const int evenFlag = (int)((get_local_id(0) & 1) == 0);
    const int oddFlag = (int)((get_local_id(0) & 1) != 0);
    const bool  eveny = ((get_local_id(1) & 1) == 0);
    const int tidx = get_local_id(0);

    float4 co1 = (float4)(0.375f, 0.375f, 0.375f, 0.375f);
    float4 co2 = (float4)(0.25f, 0.25f, 0.25f, 0.25f);
    float4 co3 = (float4)(0.0625f, 0.0625f, 0.0625f, 0.0625f);


    if(eveny)
    {
        sum = sum + float4_x_float4(int_x_float4( evenFlag, co3 ) , s_srcPatch[1 + (get_local_id(1) >> 1)][1 + ((tidx - 2) >> 1)]);
        sum = sum + float4_x_float4(int_x_float4( oddFlag , co2 ) , s_srcPatch[1 + (get_local_id(1) >> 1)][1 + ((tidx - 1) >> 1)]);
        sum = sum + float4_x_float4(int_x_float4( evenFlag, co1 ) , s_srcPatch[1 + (get_local_id(1) >> 1)][1 + ((tidx    ) >> 1)]);
        sum = sum + float4_x_float4(int_x_float4( oddFlag , co2 ) , s_srcPatch[1 + (get_local_id(1) >> 1)][1 + ((tidx + 1) >> 1)]);
        sum = sum + float4_x_float4(int_x_float4( evenFlag, co3 ) , s_srcPatch[1 + (get_local_id(1) >> 1)][1 + ((tidx + 2) >> 1)]);

    }

    s_dstPatch[2 + get_local_id(1)][get_local_id(0)] = sum;

    if (get_local_id(1) < 2)
    {
        sum = 0;

        if (eveny)
        {
            sum = sum + float4_x_float4(int_x_float4(evenFlag , co3) , s_srcPatch[0][1 + ((tidx - 2) >> 1)]);
            sum = sum + float4_x_float4(int_x_float4( oddFlag , co2  ) , s_srcPatch[0][1 + ((tidx - 1) >> 1)]);
            sum = sum + float4_x_float4(int_x_float4(evenFlag , co1 ) , s_srcPatch[0][1 + ((tidx    ) >> 1)]);
            sum = sum + float4_x_float4(int_x_float4( oddFlag , co2  ) , s_srcPatch[0][1 + ((tidx + 1) >> 1)]);
            sum = sum + float4_x_float4(int_x_float4(evenFlag , co3) , s_srcPatch[0][1 + ((tidx + 2) >> 1)]);
         }

         s_dstPatch[get_local_id(1)][get_local_id(0)] = sum;
     }

     if (get_local_id(1) > 13)
     {
        sum = 0;

        if (eveny)
        {
            sum = sum + float4_x_float4(int_x_float4(evenFlag , co3) , s_srcPatch[9][1 + ((tidx - 2) >> 1)]);
            sum = sum + float4_x_float4(int_x_float4( oddFlag , co2) , s_srcPatch[9][1 + ((tidx - 1) >> 1)]);
            sum = sum + float4_x_float4(int_x_float4(evenFlag , co1) , s_srcPatch[9][1 + ((tidx    ) >> 1)]);
            sum = sum + float4_x_float4(int_x_float4( oddFlag , co2) , s_srcPatch[9][1 + ((tidx + 1) >> 1)]);
            sum = sum + float4_x_float4(int_x_float4(evenFlag , co3) , s_srcPatch[9][1 + ((tidx + 2) >> 1)]);

        }
        s_dstPatch[4 + get_local_id(1)][get_local_id(0)] = sum;
     }

    barrier(CLK_LOCAL_MEM_FENCE);

    sum = 0;

    const int tidy = get_local_id(1);

    sum = sum + float4_x_float4(co3 , s_dstPatch[2 + tidy - 2][get_local_id(0)]);
    sum = sum + float4_x_float4(co2 , s_dstPatch[2 + tidy - 1][get_local_id(0)]);
    sum = sum + float4_x_float4(co1 , s_dstPatch[2 + tidy    ][get_local_id(0)]);
    sum = sum + float4_x_float4(co2 , s_dstPatch[2 + tidy + 1][get_local_id(0)]);
    sum = sum + float4_x_float4(co3 , s_dstPatch[2 + tidy + 2][get_local_id(0)]);

    if ((x < dstCols) && (y < dstRows))
    {
        dst[x + y * dstStep] = convert_float4_to_uchar4(int_x_float4(4.0f,sum));
    }
}
///////////////////////////////////////////////////////////////////////
//////////////////////////  CV_16UC4 //////////////////////////////////
///////////////////////////////////////////////////////////////////////
float4 covert_ushort4_to_float4(ushort4 data)
{
    float4 f4Data = {0,0,0,0};

    f4Data.x = (float)data.x;
    f4Data.y = (float)data.y;
    f4Data.z = (float)data.z;
    f4Data.w = (float)data.w;

    return f4Data;
}


ushort4 convert_float4_to_ushort4(float4 data)
{
    ushort4 u4Data;

    u4Data.x = (float)data.x;
    u4Data.y = (float)data.y;
    u4Data.z = (float)data.z;
    u4Data.w = (float)data.w;

    return u4Data;
}


__kernel void pyrUp_C4_D2(__global ushort4* src,__global ushort4* dst,
                int srcRows,int dstRows,int srcCols,int dstCols,
                int srcOffset,int dstOffset,int srcStep,int dstStep)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    __local float4 s_srcPatch[10][10];
    __local float4 s_dstPatch[20][16];

    srcOffset >>= 3;
    dstOffset >>= 3;
    srcStep >>= 3;
    dstStep >>= 3;


    if( get_local_id(0) < 10 && get_local_id(1) < 10 )
    {
        int srcx = (int)(get_group_id(0) * get_local_size(0) / 2 + get_local_id(0)) - 1;
        int srcy = (int)(get_group_id(1) * get_local_size(1) / 2 + get_local_id(1)) - 1;

        srcx = abs(srcx);
        srcx = min(srcCols - 1,srcx);

        srcy = abs(srcy);
        srcy = min(srcRows -1 ,srcy);

        s_srcPatch[get_local_id(1)][get_local_id(0)] = covert_ushort4_to_float4(src[srcx + srcy * srcStep]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    float4 sum = (float4)(0,0,0,0);

    const int evenFlag = (int)((get_local_id(0) & 1) == 0);
    const int oddFlag = (int)((get_local_id(0) & 1) != 0);
    const bool  eveny = ((get_local_id(1) & 1) == 0);
    const int tidx = get_local_id(0);

    float4 co1 = (float4)(0.375f, 0.375f, 0.375f, 0.375f);
    float4 co2 = (float4)(0.25f, 0.25f, 0.25f, 0.25f);
    float4 co3 = (float4)(0.0625f, 0.0625f, 0.0625f, 0.0625f);


    if(eveny)
    {
        sum = sum + float4_x_float4(int_x_float4( evenFlag, co3 ) , s_srcPatch[1 + (get_local_id(1) >> 1)][1 + ((tidx - 2) >> 1)]);
        sum = sum + float4_x_float4(int_x_float4( oddFlag , co2 ) , s_srcPatch[1 + (get_local_id(1) >> 1)][1 + ((tidx - 1) >> 1)]);
        sum = sum + float4_x_float4(int_x_float4( evenFlag, co1 ) , s_srcPatch[1 + (get_local_id(1) >> 1)][1 + ((tidx    ) >> 1)]);
        sum = sum + float4_x_float4(int_x_float4( oddFlag , co2 ) , s_srcPatch[1 + (get_local_id(1) >> 1)][1 + ((tidx + 1) >> 1)]);
        sum = sum + float4_x_float4(int_x_float4( evenFlag, co3 ) , s_srcPatch[1 + (get_local_id(1) >> 1)][1 + ((tidx + 2) >> 1)]);

    }

    s_dstPatch[2 + get_local_id(1)][get_local_id(0)] = sum;

    if (get_local_id(1) < 2)
    {
        sum = 0;

        if (eveny)
        {
            sum = sum + float4_x_float4(int_x_float4(evenFlag , co3) , s_srcPatch[0][1 + ((tidx - 2) >> 1)]);
            sum = sum + float4_x_float4(int_x_float4( oddFlag , co2  ) , s_srcPatch[0][1 + ((tidx - 1) >> 1)]);
            sum = sum + float4_x_float4(int_x_float4(evenFlag , co1 ) , s_srcPatch[0][1 + ((tidx    ) >> 1)]);
            sum = sum + float4_x_float4(int_x_float4( oddFlag , co2  ) , s_srcPatch[0][1 + ((tidx + 1) >> 1)]);
            sum = sum + float4_x_float4(int_x_float4(evenFlag , co3) , s_srcPatch[0][1 + ((tidx + 2) >> 1)]);
         }

         s_dstPatch[get_local_id(1)][get_local_id(0)] = sum;
     }

     if (get_local_id(1) > 13)
     {
        sum = 0;

        if (eveny)
        {
            sum = sum + float4_x_float4(int_x_float4(evenFlag , co3) , s_srcPatch[9][1 + ((tidx - 2) >> 1)]);
            sum = sum + float4_x_float4(int_x_float4( oddFlag , co2) , s_srcPatch[9][1 + ((tidx - 1) >> 1)]);
            sum = sum + float4_x_float4(int_x_float4(evenFlag , co1) , s_srcPatch[9][1 + ((tidx    ) >> 1)]);
            sum = sum + float4_x_float4(int_x_float4( oddFlag , co2) , s_srcPatch[9][1 + ((tidx + 1) >> 1)]);
            sum = sum + float4_x_float4(int_x_float4(evenFlag , co3) , s_srcPatch[9][1 + ((tidx + 2) >> 1)]);

        }
        s_dstPatch[4 + get_local_id(1)][get_local_id(0)] = sum;
     }

    barrier(CLK_LOCAL_MEM_FENCE);

    sum = 0;

    const int tidy = get_local_id(1);

    sum = sum + float4_x_float4(co3 , s_dstPatch[2 + tidy - 2][get_local_id(0)]);
    sum = sum + float4_x_float4(co2 , s_dstPatch[2 + tidy - 1][get_local_id(0)]);
    sum = sum + float4_x_float4(co1 , s_dstPatch[2 + tidy    ][get_local_id(0)]);
    sum = sum + float4_x_float4(co2 , s_dstPatch[2 + tidy + 1][get_local_id(0)]);
    sum = sum + float4_x_float4(co3 , s_dstPatch[2 + tidy + 2][get_local_id(0)]);

    if ((x < dstCols) && (y < dstRows))
    {
        dst[x + y * dstStep] = convert_float4_to_ushort4(int_x_float4(4.0f,sum));
    }
}

///////////////////////////////////////////////////////////////////////
//////////////////////////  CV_32FC4 //////////////////////////////////
///////////////////////////////////////////////////////////////////////
__kernel void pyrUp_C4_D5(__global float4* src,__global float4* dst,
                int srcRows,int dstRows,int srcCols,int dstCols,
                int srcOffset,int dstOffset,int srcStep,int dstStep)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    __local float4 s_srcPatch[10][10];
    __local float4 s_dstPatch[20][16];

    srcOffset >>= 4;
    dstOffset >>= 4;
    srcStep >>= 4;
    dstStep >>= 4;


    if( get_local_id(0) < 10 && get_local_id(1) < 10 )
    {
        int srcx = (int)(get_group_id(0) * get_local_size(0) / 2 + get_local_id(0)) - 1;
        int srcy = (int)(get_group_id(1) * get_local_size(1) / 2 + get_local_id(1)) - 1;

        srcx = abs(srcx);
        srcx = min(srcCols - 1,srcx);

        srcy = abs(srcy);
        srcy = min(srcRows -1 ,srcy);

        s_srcPatch[get_local_id(1)][get_local_id(0)] = (float4)(src[srcx + srcy * srcStep]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    float4 sum = (float4)(0,0,0,0);

    const int evenFlag = (int)((get_local_id(0) & 1) == 0);
    const int oddFlag = (int)((get_local_id(0) & 1) != 0);
    const bool  eveny = ((get_local_id(1) & 1) == 0);
    const int tidx = get_local_id(0);

    float4 co1 = (float4)(0.375f, 0.375f, 0.375f, 0.375f);
    float4 co2 = (float4)(0.25f, 0.25f, 0.25f, 0.25f);
    float4 co3 = (float4)(0.0625f, 0.0625f, 0.0625f, 0.0625f);


    if(eveny)
    {
        sum = sum + float4_x_float4(int_x_float4( evenFlag, co3 ) , s_srcPatch[1 + (get_local_id(1) >> 1)][1 + ((tidx - 2) >> 1)]);
        sum = sum + float4_x_float4(int_x_float4( oddFlag , co2 ) , s_srcPatch[1 + (get_local_id(1) >> 1)][1 + ((tidx - 1) >> 1)]);
        sum = sum + float4_x_float4(int_x_float4( evenFlag, co1 ) , s_srcPatch[1 + (get_local_id(1) >> 1)][1 + ((tidx    ) >> 1)]);
        sum = sum + float4_x_float4(int_x_float4( oddFlag , co2 ) , s_srcPatch[1 + (get_local_id(1) >> 1)][1 + ((tidx + 1) >> 1)]);
        sum = sum + float4_x_float4(int_x_float4( evenFlag, co3 ) , s_srcPatch[1 + (get_local_id(1) >> 1)][1 + ((tidx + 2) >> 1)]);

    }

    s_dstPatch[2 + get_local_id(1)][get_local_id(0)] = sum;

    if (get_local_id(1) < 2)
    {
        sum = 0;

        if (eveny)
        {
            sum = sum + float4_x_float4(int_x_float4(evenFlag , co3) , s_srcPatch[0][1 + ((tidx - 2) >> 1)]);
            sum = sum + float4_x_float4(int_x_float4( oddFlag , co2  ) , s_srcPatch[0][1 + ((tidx - 1) >> 1)]);
            sum = sum + float4_x_float4(int_x_float4(evenFlag , co1 ) , s_srcPatch[0][1 + ((tidx    ) >> 1)]);
            sum = sum + float4_x_float4(int_x_float4( oddFlag , co2  ) , s_srcPatch[0][1 + ((tidx + 1) >> 1)]);
            sum = sum + float4_x_float4(int_x_float4(evenFlag , co3) , s_srcPatch[0][1 + ((tidx + 2) >> 1)]);
         }

         s_dstPatch[get_local_id(1)][get_local_id(0)] = sum;
     }

     if (get_local_id(1) > 13)
     {
        sum = 0;

        if (eveny)
        {
            sum = sum + float4_x_float4(int_x_float4(evenFlag , co3) , s_srcPatch[9][1 + ((tidx - 2) >> 1)]);
            sum = sum + float4_x_float4(int_x_float4( oddFlag , co2) , s_srcPatch[9][1 + ((tidx - 1) >> 1)]);
            sum = sum + float4_x_float4(int_x_float4(evenFlag , co1) , s_srcPatch[9][1 + ((tidx    ) >> 1)]);
            sum = sum + float4_x_float4(int_x_float4( oddFlag , co2) , s_srcPatch[9][1 + ((tidx + 1) >> 1)]);
            sum = sum + float4_x_float4(int_x_float4(evenFlag , co3) , s_srcPatch[9][1 + ((tidx + 2) >> 1)]);

        }
        s_dstPatch[4 + get_local_id(1)][get_local_id(0)] = sum;
     }

    barrier(CLK_LOCAL_MEM_FENCE);

    sum = 0;

    const int tidy = get_local_id(1);

    sum = sum + float4_x_float4(co3 , s_dstPatch[2 + tidy - 2][get_local_id(0)]);
    sum = sum + float4_x_float4(co2 , s_dstPatch[2 + tidy - 1][get_local_id(0)]);
    sum = sum + float4_x_float4(co1 , s_dstPatch[2 + tidy    ][get_local_id(0)]);
    sum = sum + float4_x_float4(co2 , s_dstPatch[2 + tidy + 1][get_local_id(0)]);
    sum = sum + float4_x_float4(co3 , s_dstPatch[2 + tidy + 2][get_local_id(0)]);

    if ((x < dstCols) && (y < dstRows))
    {
        dst[x + y * dstStep] = 4.0f * sum;
    }
}