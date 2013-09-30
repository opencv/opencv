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

#if defined (DOUBLE_SUPPORT)

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#elif defined (cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#endif
typedef double T;
typedef double F;
typedef double16 F16;
#define convert_F16 convert_double16
#define convert_F convert_double

#else
typedef float F;
typedef float16 F16;
typedef long T;
#define convert_F16 convert_float16
#define convert_F convert_float
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

__kernel void CvMoments(__global F* src_data, int src_rows, int src_cols, int src_step,
    __global F* dst_m,
    int dst_cols, int dst_step)
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

    F16 x0 = (F16)(0.f);
    F16 x1 = (F16)(0.f);
    F16 x2 = (F16)(0.f);
    F16 x3 = (F16)(0.f);

    __global F* row = 0;

    if(y_rest > 0 && gidy == (get_num_groups(1) - 1))
    {
        if(ly < y_rest)
        {
            row = src_data + gidy * src_step + ly * src_step + gidx * 256;
        }
    }else
    {
        row = src_data + gidy * src_step + ly * src_step + gidx * 256;
    }

    F16 p;
    F16 x;
    F16 xp;
    F16 xxp;

    F py = 0.f, sy = 0.f;

    if(dy < src_rows)
    {
        if(x_rest > 0 && gidx == (get_num_groups(0) - 1))
        {
            int i;
            for(i = 0; i < x_rest - 16; i += 16)
            {
                p = vload16(0, row + i);
                x = convert_F16(vload16(0, codxy + i));
                xp = x * p;
                xxp = xp * x;

                x0 += p;
                x1 += xp;
                x2 += xxp;
                x3 += xxp * x;
            }

            x0.s0 = x0.s0 + x0.s1 + x0.s2 + x0.s3 + x0.s4 + x0.s5 + x0.s6 + x0.s7 
                + x0.s8 + x0.s9 + x0.sa + x0.sb + x0.sc + x0.sd + x0.se + x0.sf;

            x1.s0 = x1.s0 + x1.s1 + x1.s2 + x1.s3 + x1.s4 + x1.s5 + x1.s6 + x1.s7 
                + x1.s8 + x1.s9 + x1.sa + x1.sb + x1.sc + x1.sd + x1.se + x1.sf;

            x2.s0 = x2.s0 + x2.s1 + x2.s2 + x2.s3 + x2.s4 + x2.s5 + x2.s6 + x2.s7 
                + x2.s8 + x2.s9 + x2.sa + x2.sb + x2.sc + x2.sd + x2.se + x2.sf;

            x3.s0 = x3.s0 + x3.s1 + x3.s2 + x3.s3 + x3.s4 + x3.s5 + x3.s6 + x3.s7 
                + x3.s8 + x3.s9 + x3.sa + x3.sb + x3.sc + x3.sd + x3.se + x3.sf;

            F x0_ = 0;
            F x1_ = 0;
            F x2_ = 0;
            F x3_ = 0;

            for(; i < x_rest; i++)
            {
                F p_ = 0;
                p_ = row[i];
                F x_ = convert_F(codxy[i]);


                F xp_ = x_ * p_;
                F xxp_ = xp_ * x_;

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
            for(int i = 0; i < 256; i += 16)
            {
                p = vload16(0, row + i);
                x = convert_F16(vload16(0, codxy + i));
                xp = x * p;
                xxp = xp * x;

                x0 += p;
                x1 += xp;
                x2 += xxp;
                x3 += xxp * x;
            }

            x0.s0 = x0.s0 + x0.s1 + x0.s2 + x0.s3 + x0.s4 + x0.s5 + x0.s6 + x0.s7 
                + x0.s8 + x0.s9 + x0.sa + x0.sb + x0.sc + x0.sd + x0.se + x0.sf;

            x1.s0 = x1.s0 + x1.s1 + x1.s2 + x1.s3 + x1.s4 + x1.s5 + x1.s6 + x1.s7 
                + x1.s8 + x1.s9 + x1.sa + x1.sb + x1.sc + x1.sd + x1.se + x1.sf;

            x2.s0 = x2.s0 + x2.s1 + x2.s2 + x2.s3 + x2.s4 + x2.s5 + x2.s6 + x2.s7 
                + x2.s8 + x2.s9 + x2.sa + x2.sb + x2.sc + x2.sd + x2.se + x2.sf;

            x3.s0 = x3.s0 + x3.s1 + x3.s2 + x3.s3 + x3.s4 + x3.s5 + x3.s6 + x3.s7 
                + x3.s8 + x3.s9 + x3.sa + x3.sb + x3.sc + x3.sd + x3.se + x3.sf;
        }

        if(y_rest > 0 && gidy == (get_num_groups(1) - 1))
        {
            if(ly < y_rest)
            {
                py = ly * x0.s0;
                sy = ly * ly;
            }
        }else
        {
            py = ly * x0.s0;
            sy = ly * ly;
        }
    }
    __local F mom[10][256];


    if(y_rest > 0 && gidy == (get_num_groups(1) - 1))
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
        {
            for(int i = 1; i < y_rest; i++)
            {
                mom[ly][0] = mom[ly][i] + mom[ly][0];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }else
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
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    F xm = (gidx * 256) * mom[0][0];
    F ym = (gidy * 256) * mom[0][0];

    __local F mom2[10][1];
    if(ly == 0)
    {
        mom2[0][0] = mom[0][0];
        mom2[1][0] = mom[1][0] + xm;
        mom2[2][0] = mom[2][0] + ym;
        mom2[3][0] = mom[3][0] + gidx * 256 * (mom[1][0] * 2 + xm);
        mom2[4][0] = mom[4][0] + gidx * 256 * (mom[2][0] + ym) + gidy * 256 * mom[1][0];
        mom2[5][0] = mom[5][0] + gidy * 256 * (mom[2][0] * 2 + ym);
        mom2[6][0] = mom[6][0] + gidx * 256 * (3 * mom[3][0] + 256 * gidx * (3 * mom[1][0] + xm));
        mom2[7][0] = mom[7][0] + gidx * 256 * (2 * (mom[4][0] + 256 * gidy * mom[1][0]) + 256 * gidx * (mom[2][0] + ym)) + 256 * gidy * mom[3][0];
        mom2[8][0] = mom[8][0] + gidy * 256 * (2 * (mom[4][0] + 256 * gidx * mom[2][0]) + 256 * gidy * (mom[1][0] + xm)) + 256 * gidx * mom[5][0];
        mom2[9][0] = mom[9][0] + gidy * 256 * (3 * mom[5][0] + 256 * gidy * (3 * mom[2][0] + ym));
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if(ly < 10)
    {
        dst_m[10 * gidy * dst_step + ly * dst_step + gidx] = mom2[ly][0];
    }
}