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
//    Jin Ma,  jin@multicorewareinc.com
//    Sen Liu, swjtuls1987@126.com
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

#if defined (DOUBLE_SUPPORT)
#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#elif defined (cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#endif
typedef double T;
#else
typedef long T;
#endif

#define DST_ROW_00     0
#define DST_ROW_10     1
#define DST_ROW_01     2
#define DST_ROW_20     3
#define DST_ROW_11     4
#define DST_ROW_02     5
#define DST_ROW_30     6
#define DST_ROW_21     7
#define DST_ROW_12     8
#define DST_ROW_03     9

__kernel void icvContourMoments(int contour_total,
                                __global float* reader_oclmat_data,
                                __global T* dst_a,
                                int dst_step)
{
    T xi_1, yi_1, xi_12, yi_12, xi, yi, xi2, yi2, dxy, xii_1, yii_1;
    int idx = get_global_id(0);

    if (idx < 0 || idx >= contour_total)
        return;

    xi_1 = (T)(*(reader_oclmat_data + (get_global_id(0) << 1)));
    yi_1 = (T)(*(reader_oclmat_data + (get_global_id(0) << 1) + 1));
    xi_12 = xi_1 * xi_1;
    yi_12 = yi_1 * yi_1;

    if(idx == contour_total - 1)
    {
        xi = (T)(*(reader_oclmat_data));
        yi = (T)(*(reader_oclmat_data + 1));
    }
    else
    {
        xi = (T)(*(reader_oclmat_data + (idx + 1) * 2));
        yi = (T)(*(reader_oclmat_data + (idx + 1) * 2 + 1));
    }
    xi2 = xi * xi;
    yi2 = yi * yi;
    dxy = xi_1 * yi - xi * yi_1;
    xii_1 = xi_1 + xi;
    yii_1 = yi_1 + yi;

    dst_step /= sizeof(T);
    *( dst_a + DST_ROW_00 * dst_step + idx) = dxy;
    *( dst_a + DST_ROW_10 * dst_step + idx) = dxy * xii_1;
    *( dst_a + DST_ROW_01 * dst_step + idx) = dxy * yii_1;
    *( dst_a + DST_ROW_20 * dst_step + idx) = dxy * (xi_1 * xii_1 + xi2);
    *( dst_a + DST_ROW_11 * dst_step + idx) = dxy * (xi_1 * (yii_1 + yi_1) + xi * (yii_1 + yi));
    *( dst_a + DST_ROW_02 * dst_step + idx) = dxy * (yi_1 * yii_1 + yi2);
    *( dst_a + DST_ROW_30 * dst_step + idx) = dxy * xii_1 * (xi_12 + xi2);
    *( dst_a + DST_ROW_03 * dst_step + idx) = dxy * yii_1 * (yi_12 + yi2);
    *( dst_a + DST_ROW_21 * dst_step + idx) =
        dxy * (xi_12 * (3 * yi_1 + yi) + 2 * xi * xi_1 * yii_1 +
        xi2 * (yi_1 + 3 * yi));
    *( dst_a + DST_ROW_12 * dst_step + idx) =
        dxy * (yi_12 * (3 * xi_1 + xi) + 2 * yi * yi_1 * xii_1 +
        yi2 * (xi_1 + 3 * xi));
}

#if defined (DOUBLE_SUPPORT)
#define WT double
#define WT4 double4
#define convert_T4 convert_double4
#define convert_T convert_double
#else
#define WT float
#define WT4 float4
#define convert_T4 convert_float4
#define convert_T convert_float
#endif

#ifdef CV_8UC1
#define TT uchar
#elif defined CV_16UC1
#define TT ushort
#elif defined CV_16SC1
#define TT short
#elif defined CV_32FC1
#define TT float
#elif defined CV_64FC1
#ifdef DOUBLE_SUPPORT
#define TT double
#else
#define TT float
#endif
#endif
__kernel void CvMoments(__global TT* src_data, int src_rows, int src_cols, int src_step,
                        __global WT* dst_m,
                        int dst_cols, int dst_step, int binary)
{
    int dy = get_global_id(1);
    int ly = get_local_id(1);
    int gidx = get_group_id(0);
    int gidy = get_group_id(1);
    int x_rest = src_cols % 256;
    int y_rest = src_rows % 256;
    __local int codxy[256];
    codxy[ly] = ly;
    barrier(CLK_LOCAL_MEM_FENCE);

    WT4 x0 = (WT4)(0.f);
    WT4 x1 = (WT4)(0.f);
    WT4 x2 = (WT4)(0.f);
    WT4 x3 = (WT4)(0.f);

    __global TT* row = src_data + gidy * src_step + ly * src_step + gidx * 256;

    WT4 p;
    WT4 x;
    WT4 xp;
    WT4 xxp;

    WT py = 0.f, sy = 0.f;

    if(dy < src_rows)
    {
        if((x_rest > 0) && (gidx == ((int)get_num_groups(0) - 1)))
        {
            int i;
            for(i = 0; i < x_rest - 4; i += 4)
            {
                p = convert_T4(vload4(0, row + i));
                x = convert_T4(vload4(0, codxy + i));
                xp = x * p;
                xxp = xp * x;

                x0 += p;
                x1 += xp;
                x2 += xxp;
                x3 += convert_T4(xxp * x);
            }

            x0.s0 = x0.s0 + x0.s1 + x0.s2 + x0.s3;
            x1.s0 = x1.s0 + x1.s1 + x1.s2 + x1.s3;
            x2.s0 = x2.s0 + x2.s1 + x2.s2 + x2.s3;
            x3.s0 = x3.s0 + x3.s1 + x3.s2 + x3.s3;

            WT x0_ = 0;
            WT x1_ = 0;
            WT x2_ = 0;
            WT x3_ = 0;

            for(; i < x_rest; i++)
            {
                WT p_ = 0;
                p_ = row[i];
                WT x_ = convert_T(codxy[i]);


                WT xp_ = x_ * p_;
                WT xxp_ = xp_ * x_;

                x0_ += p_;
                x1_ += xp_;
                x2_ += xxp_;
                x3_ += xxp_ * x_;
            }

            x0.s0 += x0_;
            x1.s0 += x1_;
            x2.s0 += x2_;
            x3.s0 += x3_;
        }else
        {
            for(int i = 0; i < 256; i += 4)
            {
                p = convert_T4(vload4(0, row + i));
                x = convert_T4(vload4(0, codxy + i));
                xp = x * p;
                xxp = xp * x;

                x0 += p;
                x1 += xp;
                x2 += xxp;
                x3 += convert_T4(xxp * x);
            }

            x0.s0 = x0.s0 + x0.s1 + x0.s2 + x0.s3;
            x1.s0 = x1.s0 + x1.s1 + x1.s2 + x1.s3;
            x2.s0 = x2.s0 + x2.s1 + x2.s2 + x2.s3;
            x3.s0 = x3.s0 + x3.s1 + x3.s2 + x3.s3;
        }

        py = ly * x0.s0;
        sy = ly * ly;
    }
    __local WT mom[10][256];

    if((y_rest > 0) && (gidy == ((int)get_num_groups(1) - 1)))
    {
        if(ly < y_rest)
        {
            mom[9][ly] = py * sy;
            mom[8][ly] = x1.s0 * sy;
            mom[7][ly] = x2.s0 * ly;
            mom[6][ly] = x3.s0;
            mom[5][ly] = x0.s0 * sy;
            mom[4][ly] = x1.s0 * ly;
            mom[3][ly] = x2.s0;
            mom[2][ly] = py;
            mom[1][ly] = x1.s0;
            mom[0][ly] = x0.s0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if(ly < 10)
            for(int i = 1; i < y_rest; i++)
                mom[ly][0] = mom[ly][i] + mom[ly][0];
    }
    else
    {
        mom[9][ly] = py * sy;
        mom[8][ly] = x1.s0 * sy;
        mom[7][ly] = x2.s0 * ly;
        mom[6][ly] = x3.s0;
        mom[5][ly] = x0.s0 * sy;
        mom[4][ly] = x1.s0 * ly;
        mom[3][ly] = x2.s0;
        mom[2][ly] = py;
        mom[1][ly] = x1.s0;
        mom[0][ly] = x0.s0;

        barrier(CLK_LOCAL_MEM_FENCE);

        if(ly < 128)
        {
            mom[0][ly] = mom[0][ly] + mom[0][ly + 128];
            mom[1][ly] = mom[1][ly] + mom[1][ly + 128];
            mom[2][ly] = mom[2][ly] + mom[2][ly + 128];
            mom[3][ly] = mom[3][ly] + mom[3][ly + 128];
            mom[4][ly] = mom[4][ly] + mom[4][ly + 128];
            mom[5][ly] = mom[5][ly] + mom[5][ly + 128];
            mom[6][ly] = mom[6][ly] + mom[6][ly + 128];
            mom[7][ly] = mom[7][ly] + mom[7][ly + 128];
            mom[8][ly] = mom[8][ly] + mom[8][ly + 128];
            mom[9][ly] = mom[9][ly] + mom[9][ly + 128];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if(ly < 64)
        {
            mom[0][ly] = mom[0][ly] + mom[0][ly + 64];
            mom[1][ly] = mom[1][ly] + mom[1][ly + 64];
            mom[2][ly] = mom[2][ly] + mom[2][ly + 64];
            mom[3][ly] = mom[3][ly] + mom[3][ly + 64];
            mom[4][ly] = mom[4][ly] + mom[4][ly + 64];
            mom[5][ly] = mom[5][ly] + mom[5][ly + 64];
            mom[6][ly] = mom[6][ly] + mom[6][ly + 64];
            mom[7][ly] = mom[7][ly] + mom[7][ly + 64];
            mom[8][ly] = mom[8][ly] + mom[8][ly + 64];
            mom[9][ly] = mom[9][ly] + mom[9][ly + 64];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if(ly < 32)
        {
            mom[0][ly] = mom[0][ly] + mom[0][ly + 32];
            mom[1][ly] = mom[1][ly] + mom[1][ly + 32];
            mom[2][ly] = mom[2][ly] + mom[2][ly + 32];
            mom[3][ly] = mom[3][ly] + mom[3][ly + 32];
            mom[4][ly] = mom[4][ly] + mom[4][ly + 32];
            mom[5][ly] = mom[5][ly] + mom[5][ly + 32];
            mom[6][ly] = mom[6][ly] + mom[6][ly + 32];
            mom[7][ly] = mom[7][ly] + mom[7][ly + 32];
            mom[8][ly] = mom[8][ly] + mom[8][ly + 32];
            mom[9][ly] = mom[9][ly] + mom[9][ly + 32];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if(ly < 16)
        {
            mom[0][ly] = mom[0][ly] + mom[0][ly + 16];
            mom[1][ly] = mom[1][ly] + mom[1][ly + 16];
            mom[2][ly] = mom[2][ly] + mom[2][ly + 16];
            mom[3][ly] = mom[3][ly] + mom[3][ly + 16];
            mom[4][ly] = mom[4][ly] + mom[4][ly + 16];
            mom[5][ly] = mom[5][ly] + mom[5][ly + 16];
            mom[6][ly] = mom[6][ly] + mom[6][ly + 16];
            mom[7][ly] = mom[7][ly] + mom[7][ly + 16];
            mom[8][ly] = mom[8][ly] + mom[8][ly + 16];
            mom[9][ly] = mom[9][ly] + mom[9][ly + 16];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if(ly < 8)
        {
            mom[0][ly] = mom[0][ly] + mom[0][ly + 8];
            mom[1][ly] = mom[1][ly] + mom[1][ly + 8];
            mom[2][ly] = mom[2][ly] + mom[2][ly + 8];
            mom[3][ly] = mom[3][ly] + mom[3][ly + 8];
            mom[4][ly] = mom[4][ly] + mom[4][ly + 8];
            mom[5][ly] = mom[5][ly] + mom[5][ly + 8];
            mom[6][ly] = mom[6][ly] + mom[6][ly + 8];
            mom[7][ly] = mom[7][ly] + mom[7][ly + 8];
            mom[8][ly] = mom[8][ly] + mom[8][ly + 8];
            mom[9][ly] = mom[9][ly] + mom[9][ly + 8];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if(ly < 4)
        {
            mom[0][ly] = mom[0][ly] + mom[0][ly + 4];
            mom[1][ly] = mom[1][ly] + mom[1][ly + 4];
            mom[2][ly] = mom[2][ly] + mom[2][ly + 4];
            mom[3][ly] = mom[3][ly] + mom[3][ly + 4];
            mom[4][ly] = mom[4][ly] + mom[4][ly + 4];
            mom[5][ly] = mom[5][ly] + mom[5][ly + 4];
            mom[6][ly] = mom[6][ly] + mom[6][ly + 4];
            mom[7][ly] = mom[7][ly] + mom[7][ly + 4];
            mom[8][ly] = mom[8][ly] + mom[8][ly + 4];
            mom[9][ly] = mom[9][ly] + mom[9][ly + 4];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if(ly < 2)
        {
            mom[0][ly] = mom[0][ly] + mom[0][ly + 2];
            mom[1][ly] = mom[1][ly] + mom[1][ly + 2];
            mom[2][ly] = mom[2][ly] + mom[2][ly + 2];
            mom[3][ly] = mom[3][ly] + mom[3][ly + 2];
            mom[4][ly] = mom[4][ly] + mom[4][ly + 2];
            mom[5][ly] = mom[5][ly] + mom[5][ly + 2];
            mom[6][ly] = mom[6][ly] + mom[6][ly + 2];
            mom[7][ly] = mom[7][ly] + mom[7][ly + 2];
            mom[8][ly] = mom[8][ly] + mom[8][ly + 2];
            mom[9][ly] = mom[9][ly] + mom[9][ly + 2];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if(ly < 1)
        {
            mom[0][ly] = mom[0][ly] + mom[0][ly + 1];
            mom[1][ly] = mom[1][ly] + mom[1][ly + 1];
            mom[2][ly] = mom[2][ly] + mom[2][ly + 1];
            mom[3][ly] = mom[3][ly] + mom[3][ly + 1];
            mom[4][ly] = mom[4][ly] + mom[4][ly + 1];
            mom[5][ly] = mom[5][ly] + mom[5][ly + 1];
            mom[6][ly] = mom[6][ly] + mom[6][ly + 1];
            mom[7][ly] = mom[7][ly] + mom[7][ly + 1];
            mom[8][ly] = mom[8][ly] + mom[8][ly + 1];
            mom[9][ly] = mom[9][ly] + mom[9][ly + 1];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if(binary)
    {
        WT s = 1.0f/255;
        if(ly < 10)
            mom[ly][0] *= s;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    WT xm = (gidx * 256) * mom[0][0];
    WT ym = (gidy * 256) * mom[0][0];

    if(ly == 0)
    {
        mom[0][1] = mom[0][0];
        mom[1][1] = mom[1][0] + xm;
        mom[2][1] = mom[2][0] + ym;
        mom[3][1] = mom[3][0] + gidx * 256 * (mom[1][0] * 2 + xm);
        mom[4][1] = mom[4][0] + gidx * 256 * (mom[2][0] + ym) + gidy * 256 * mom[1][0];
        mom[5][1] = mom[5][0] + gidy * 256 * (mom[2][0] * 2 + ym);
        mom[6][1] = mom[6][0] + gidx * 256 * (3 * mom[3][0] + 256 * gidx * (3 * mom[1][0] + xm));
        mom[7][1] = mom[7][0] + gidx * 256 * (2 * (mom[4][0] + 256 * gidy * mom[1][0]) + 256 * gidx * (mom[2][0] + ym)) + 256 * gidy * mom[3][0];
        mom[8][1] = mom[8][0] + gidy * 256 * (2 * (mom[4][0] + 256 * gidx * mom[2][0]) + 256 * gidy * (mom[1][0] + xm)) + 256 * gidx * mom[5][0];
        mom[9][1] = mom[9][0] + gidy * 256 * (3 * mom[5][0] + 256 * gidy * (3 * mom[2][0] + ym));
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if(ly < 10)
        dst_m[10 * gidy * dst_step + ly * dst_step + gidx] = mom[ly][1];
}
