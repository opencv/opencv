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
//    Peng Xiao, pengxiao@multicorewareinc.com
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

__kernel void buildWarpPlaneMaps(__global uchar * xmapptr, int xmap_step, int xmap_offset,
                                 __global uchar * ymapptr, int ymap_step, int ymap_offset, int rows, int cols,
                                 __constant float * ck_rinv, __constant float * ct,
                                 int tl_u, int tl_v, float scale, int rowsPerWI)
{
    int du = get_global_id(0);
    int dv0 = get_global_id(1) * rowsPerWI;

    if (du < cols)
    {
        int xmap_index = mad24(dv0, xmap_step, mad24(du, (int)sizeof(float), xmap_offset));
        int ymap_index = mad24(dv0, ymap_step, mad24(du, (int)sizeof(float), ymap_offset));

        float u = tl_u + du;
        float x_ = fma(u, scale, -ct[0]);
        float ct1 = 1 - ct[2];

        for (int dv = dv0, dv1 = min(rows, dv0 + rowsPerWI); dv < dv1; ++dv, xmap_index += xmap_step,
            ymap_index += ymap_step)
        {
            __global float * xmap = (__global float *)(xmapptr + xmap_index);
            __global float * ymap = (__global float *)(ymapptr + ymap_index);

            float v = tl_v + dv;
            float y_ = fma(v, scale, -ct[1]);

            float x = fma(ck_rinv[0], x_, fma(ck_rinv[1], y_, ck_rinv[2] * ct1));
            float y = fma(ck_rinv[3], x_, fma(ck_rinv[4], y_, ck_rinv[5] * ct1));
            float z = fma(ck_rinv[6], x_, fma(ck_rinv[7], y_, ck_rinv[8] * ct1));

            if (z != 0)
                x /= z, y /= z;
            else
                x = y = -1;

            xmap[0] = x;
            ymap[0] = y;
        }
    }
}

__kernel void buildWarpCylindricalMaps(__global uchar * xmapptr, int xmap_step, int xmap_offset,
                                       __global uchar * ymapptr, int ymap_step, int ymap_offset, int rows, int cols,
                                       __constant float * ck_rinv, int tl_u, int tl_v, float scale, int rowsPerWI)
{
    int du = get_global_id(0);
    int dv0 = get_global_id(1) * rowsPerWI;

    if (du < cols)
    {
        int xmap_index = mad24(dv0, xmap_step, mad24(du, (int)sizeof(float), xmap_offset));
        int ymap_index = mad24(dv0, ymap_step, mad24(du, (int)sizeof(float), ymap_offset));

        float u = (tl_u + du) * scale;
        float x_, z_;
        x_ = sincos(u, &z_);

        for (int dv = dv0, dv1 = min(rows, dv0 + rowsPerWI); dv < dv1; ++dv, xmap_index += xmap_step,
            ymap_index += ymap_step)
        {
            __global float * xmap = (__global float *)(xmapptr + xmap_index);
            __global float * ymap = (__global float *)(ymapptr + ymap_index);

            float y_ = (tl_v + dv) * scale;

            float x, y, z;
            x = fma(ck_rinv[0], x_, fma(ck_rinv[1], y_, ck_rinv[2] * z_));
            y = fma(ck_rinv[3], x_, fma(ck_rinv[4], y_, ck_rinv[5] * z_));
            z = fma(ck_rinv[6], x_, fma(ck_rinv[7], y_, ck_rinv[8] * z_));

            if (z > 0)
                x /= z, y /= z;
            else
                x = y = -1;

            xmap[0] = x;
            ymap[0] = y;
        }
    }
}

__kernel void buildWarpSphericalMaps(__global uchar * xmapptr, int xmap_step, int xmap_offset,
                                     __global uchar * ymapptr, int ymap_step, int ymap_offset, int rows, int cols,
                                     __constant float * ck_rinv, int tl_u, int tl_v, float scale, int rowsPerWI)
{
    int du = get_global_id(0);
    int dv0 = get_global_id(1) * rowsPerWI;

    if (du < cols)
    {
        int xmap_index = mad24(dv0, xmap_step, mad24(du, (int)sizeof(float), xmap_offset));
        int ymap_index = mad24(dv0, ymap_step, mad24(du, (int)sizeof(float), ymap_offset));

        float u = (tl_u + du) * scale;
        float cosu, sinu = sincos(u, &cosu);

        for (int dv = dv0, dv1 = min(rows, dv0 + rowsPerWI); dv < dv1; ++dv, xmap_index += xmap_step,
            ymap_index += ymap_step)
        {
            __global float * xmap = (__global float *)(xmapptr + xmap_index);
            __global float * ymap = (__global float *)(ymapptr + ymap_index);

            float v = (tl_v + dv) * scale;

            float cosv, sinv = sincos(v, &cosv);
            float x_ = sinv * sinu;
            float y_ = -cosv;
            float z_ = sinv * cosu;

            float x, y, z;
            x = fma(ck_rinv[0], x_, fma(ck_rinv[1], y_, ck_rinv[2] * z_));
            y = fma(ck_rinv[3], x_, fma(ck_rinv[4], y_, ck_rinv[5] * z_));
            z = fma(ck_rinv[6], x_, fma(ck_rinv[7], y_, ck_rinv[8] * z_));

            if (z > 0)
                x /= z, y /= z;
            else
                x = y = -1;

            xmap[0] = x;
            ymap[0] = y;
        }
    }
}
